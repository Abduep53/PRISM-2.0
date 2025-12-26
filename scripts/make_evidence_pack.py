import json
from pathlib import Path
from prism.utils.hashing import sha256_file

REQUIRED = [
    "preds.npz",
    "ids.txt",
    "metrics.json",
    "dp.json",
    "infer.json",
    "env.txt",
    "git.txt",
    "cmd.txt",
    "config_resolved.yaml",
]

def main():
    base = Path("results/benchmarks/ucf101_split1")
    if not base.exists():
        raise SystemExit(f"Missing {base}")

    run_dirs = [p for p in base.iterdir() if p.is_dir()]
    if not run_dirs:
        raise SystemExit("No run directories found")

    for rd in run_dirs:
        evidence = {
            "run_dir": rd.as_posix(),
            "files": {},
        }

        for name in REQUIRED:
            p = rd / name
            if not p.exists():
                continue
            evidence["files"][name] = {
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size,
            }

        out = rd / "evidence.json"
        out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {out}")

if __name__ == "__main__":
    main()
