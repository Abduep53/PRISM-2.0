import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score

from prism.utils.hashing import sha256_file

EXPECTED = {
    "lstm_baseline":    {"acc": 0.823, "f1": 0.815, "prec": 0.821, "rec": 0.819, "kappa": 0.801},
    "stgcn_nodp":       {"acc": 0.891, "f1": 0.887, "prec": 0.889, "rec": 0.885, "kappa": 0.872},
    "stgcn_dp_eps1.0":  {"acc": 0.847, "f1": 0.841, "prec": 0.844, "rec": 0.838, "kappa": 0.823},
    "stgcn_dp_eps0.1":  {"acc": 0.812, "f1": 0.806, "prec": 0.810, "rec": 0.808, "kappa": 0.794},
    "stgcn_dp_eps10.0": {"acc": 0.883, "f1": 0.879, "prec": 0.881, "rec": 0.877, "kappa": 0.864},
}

REQUIRED_FILES = ["preds.npz", "ids.txt", "metrics.json", "dp.json", "evidence.json"]

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }

def approx_equal(a, b, tol=1e-3):
    return abs(a - b) <= tol

def verify_hashes(run_dir: Path):
    ev_path = run_dir / "evidence.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    files = ev.get("files", {})
    for fname, meta in files.items():
        p = run_dir / fname
        if not p.exists():
            raise AssertionError(f"Evidence lists missing file: {p}")
        h = sha256_file(p)
        if h != meta["sha256"]:
            raise AssertionError(f"SHA256 mismatch for {p}: got={h}, expected={meta['sha256']}")

def verify_run(run_dir: Path, run_name: str):
    for f in REQUIRED_FILES:
        if not (run_dir / f).exists():
            raise FileNotFoundError(f"Missing {run_dir / f}")

    verify_hashes(run_dir)

    npz = np.load(run_dir / "preds.npz")
    y_true = npz["y_true"].astype(int)
    y_pred = npz["y_pred"].astype(int)

    recomputed = compute_metrics(y_true, y_pred)
    stored = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    # Recomputed == stored
    for k in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "cohen_kappa"]:
        if not approx_equal(float(recomputed[k]), float(stored[k]), tol=1e-6):
            raise AssertionError(f"[{run_name}] stored mismatch {k}: stored={stored[k]} recomputed={recomputed[k]}")

    # Stored == expected table values
    exp = EXPECTED[run_name]
    if not approx_equal(float(stored["accuracy"]), exp["acc"], tol=1e-3): raise AssertionError(f"[{run_name}] accuracy mismatch")
    if not approx_equal(float(stored["f1_macro"]), exp["f1"], tol=1e-3): raise AssertionError(f"[{run_name}] f1 mismatch")
    if not approx_equal(float(stored["precision_macro"]), exp["prec"], tol=1e-3): raise AssertionError(f"[{run_name}] precision mismatch")
    if not approx_equal(float(stored["recall_macro"]), exp["rec"], tol=1e-3): raise AssertionError(f"[{run_name}] recall mismatch")
    if not approx_equal(float(stored["cohen_kappa"]), exp["kappa"], tol=1e-3): raise AssertionError(f"[{run_name}] kappa mismatch")

    print(f"✅ VERIFIED: {run_name}")

def main():
    base = Path("results/benchmarks/ucf101_split1")
    if not base.exists():
        raise FileNotFoundError(f"Missing {base}")

    for run_name in EXPECTED.keys():
        rd = base / run_name
        if not rd.exists():
            raise FileNotFoundError(f"Missing run folder: {rd}")
        verify_run(rd, run_name)

    print("\nALL VERIFIED ✅")

if __name__ == "__main__":
    main()
