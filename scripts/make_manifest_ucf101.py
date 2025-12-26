import hashlib
from pathlib import Path

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    processed_root = Path("data/ucf101/processed")
    out_path = Path("data/ucf101/manifest_sha256.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in processed_root.rglob("*.npz") if p.is_file()])
    if not files:
        raise SystemExit(f"No .npz files found under {processed_root}. Provide processed features first.")

    lines = []
    for p in files:
        digest = sha256_file(p)
        rel = p.relative_to(processed_root).as_posix()
        lines.append(f"{digest}  {rel}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {out_path} ({len(files)} files)")

if __name__ == "__main__":
    main()
