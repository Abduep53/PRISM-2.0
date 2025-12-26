import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from prism.utils.repro import set_determinism, get_device
from prism.utils.io import ensure_dir, write_text, write_json, capture_env, capture_git
from prism.utils.hashing import sha256_file
from prism.data.ucf101_processed import UCF101ProcessedDataset
from prism.train import train_model
from prism.eval import evaluate
from prism.infer_bench import benchmark_inference, model_size_mb

def load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def write_evidence(run_dir: Path):
    evidence = {"files": {}}
    for p in run_dir.iterdir():
        if p.is_file():
            evidence["files"][p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    (run_dir / "evidence.json").write_text(yaml.safe_dump(evidence, sort_keys=True), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split", type=int, default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.split is not None:
        cfg["data"]["split_id"] = int(args.split)

    run_name = cfg["run"]["name"]
    split_id = int(cfg["data"]["split_id"])
    seed = int(cfg["run"]["seed"])

    set_determinism(seed)
    device = get_device(cfg["run"]["device"])

    out_root = Path(cfg["run"]["output_root"]) / f"ucf101_split{split_id}" / run_name
    ensure_dir(out_root)

    write_text(out_root / "cmd.txt", f"python prism/cli_run_benchmark.py --config {args.config} --split {split_id}\n")
    write_text(out_root / "env.txt", capture_env())
    write_text(out_root / "git.txt", capture_git())
    write_text(out_root / "config_resolved.yaml", yaml.safe_dump(cfg, sort_keys=False))

    dcfg = cfg["data"]
    train_ds = UCF101ProcessedDataset(
        processed_root=dcfg["processed_root"],
        splits_dir=dcfg["splits_dir"],
        split_id=split_id,
        train=True,
        clip_len=int(dcfg["clip_len"]),
        num_joints=int(dcfg["num_joints"]),
        channels=int(dcfg["channels"]),
    )
    test_ds = UCF101ProcessedDataset(
        processed_root=dcfg["processed_root"],
        splits_dir=dcfg["splits_dir"],
        split_id=split_id,
        train=False,
        clip_len=int(dcfg["clip_len"]),
        num_joints=int(dcfg["num_joints"]),
        channels=int(dcfg["channels"]),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        drop_last=False,
    )

    result = train_model(cfg, train_ds, device=device)
    model = result.model

    metrics, y_true, y_pred, ids = evaluate(model, test_loader, device=device)

    write_json(out_root / "metrics.json", metrics)
    (out_root / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    np.savez_compressed(out_root / "preds.npz", y_true=y_true.astype(np.int64), y_pred=y_pred.astype(np.int64))

    write_json(out_root / "dp.json", result.dp_info)

    infer_cfg = cfg.get("infer_bench", {"enabled": False})
    if bool(infer_cfg.get("enabled", False)):
        inf = benchmark_inference(
            model=model,
            device=device,
            batch_size=int(infer_cfg["batch_size"]),
            channels=int(dcfg["channels"]),
            clip_len=int(dcfg["clip_len"]),
            num_joints=int(dcfg["num_joints"]),
            warmup=int(infer_cfg["warmup"]),
            iters=int(infer_cfg["iters"]),
        )
        inf["model_size_mb"] = float(model_size_mb(model))
        write_json(out_root / "infer.json", inf)

    torch.save({"model_state": model.state_dict(), "config": cfg}, out_root / "checkpoint.pt")

    # Evidence hash pack (anti-tamper)
    # Use JSON for evidence.json expected by verifier; we store both YAML evidence + JSON evidence.
    evidence_json = {"files": {}}
    for p in out_root.iterdir():
        if p.is_file():
            evidence_json["files"][p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    write_json(out_root / "evidence.json", evidence_json)

    print(f"Saved artifacts to: {out_root}")

if __name__ == "__main__":
    main()
