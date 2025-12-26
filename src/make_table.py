import argparse
import json
from pathlib import Path

def pct(x: float) -> str:
    return f"{x * 100.0:.1f}%"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.results_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    perf_rows, infer_rows = [], []
    for rd in run_dirs:
        mp = rd / "metrics.json"
        if not mp.exists():
            continue
        metrics = json.loads(mp.read_text(encoding="utf-8"))

        dp_eps = "N/A"
        dp_path = rd / "dp.json"
        if dp_path.exists():
            dp = json.loads(dp_path.read_text(encoding="utf-8"))
            if dp.get("enabled", False):
                dp_eps = str(dp.get("target_epsilon", "DP"))

        perf_rows.append({
            "name": rd.name,
            "acc": metrics["accuracy"],
            "f1": metrics["f1_macro"],
            "prec": metrics["precision_macro"],
            "rec": metrics["recall_macro"],
            "kappa": metrics["cohen_kappa"],
            "eps": dp_eps,
        })

        ip = rd / "infer.json"
        if ip.exists():
            inf = json.loads(ip.read_text(encoding="utf-8"))
            infer_rows.append({
                "name": rd.name,
                "mean_ms": inf["mean_latency_ms"],
                "p95_ms": inf["p95_latency_ms"],
                "fps": inf["throughput_fps"],
                "size_mb": inf.get("model_size_mb", None),
            })

    lines = []
    lines.append("## Benchmark Results\n\n")
    lines.append("### Performance Comparison\n\n")
    lines.append("| Model | Accuracy | F1-Score | Precision | Recall | Cohen's Kappa | Privacy ε |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for r in perf_rows:
        lines.append(f"| {r['name']} | {pct(r['acc'])} | {pct(r['f1'])} | {pct(r['prec'])} | {pct(r['rec'])} | {pct(r['kappa'])} | {r['eps']} |\n")

    if infer_rows:
        lines.append("\n### Inference Performance\n\n")
        lines.append("| Model | Mean Latency (ms) | P95 Latency (ms) | Throughput (FPS) | Model Size (MB) |\n")
        lines.append("|---|---:|---:|---:|---:|\n")
        for r in infer_rows:
            size = "" if r["size_mb"] is None else f"{float(r['size_mb']):.1f}"
            lines.append(f"| {r['name']} | {float(r['mean_ms']):.1f} | {float(r['p95_ms']):.1f} | {float(r['fps']):.1f} | {size} |\n")

    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote table: {out}")

if __name__ == "__main__":
    main()
