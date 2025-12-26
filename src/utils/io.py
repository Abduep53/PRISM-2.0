import json
import subprocess
import sys
from pathlib import Path

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")

def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def capture_env() -> str:
    lines = []
    lines.append(f"python: {sys.version}")
    try:
        import torch
        lines.append(f"torch: {torch.__version__}")
        lines.append(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"cuda_device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        lines.append(f"torch_info_error: {e}")

    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        lines.append("\n[pip_freeze]\n" + out)
    except Exception as e:
        lines.append(f"pip_freeze_error: {e}")

    return "\n".join(lines)

def capture_git() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return f"commit: {commit}\nstatus_porcelain:\n{status}\n"
    except Exception as e:
        return f"git_capture_error: {e}\n"
