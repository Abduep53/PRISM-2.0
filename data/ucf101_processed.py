from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def read_class_ind(splits_dir: Path) -> dict:
    p = splits_dir / "classInd.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    mapping = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        idx_str, name = line.split()
        mapping[name] = int(idx_str) - 1
    return mapping

def read_train_list(splits_dir: Path, split_id: int):
    p = splits_dir / f"trainlist{split_id:02d}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    items = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rel, label_str = line.split()
        items.append((rel, int(label_str) - 1))
    return items

def read_test_list(splits_dir: Path, split_id: int, class_to_idx: dict):
    p = splits_dir / f"testlist{split_id:02d}.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    items = []
    for line in p.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel:
            continue
        cls = rel.split("/")[0]
        items.append((rel, class_to_idx[cls]))
    return items

def to_npz_path(processed_root: Path, rel_avi: str) -> Path:
    return processed_root / rel_avi.replace(".avi", ".npz")

def crop_or_pad(x: np.ndarray, clip_len: int) -> np.ndarray:
    T = x.shape[0]
    if T == clip_len:
        return x
    if T > clip_len:
        s = (T - clip_len) // 2
        return x[s:s + clip_len]
    pad = np.repeat(x[-1:], repeats=(clip_len - T), axis=0)
    return np.concatenate([x, pad], axis=0)

class UCF101ProcessedDataset(Dataset):
    """
    Expects .npz files under data/ucf101/processed/<Class>/<Video>.npz
    Required key: x float32 [T,V,C]
    """
    def __init__(self, processed_root: str, splits_dir: str, split_id: int, train: bool,
                 clip_len: int, num_joints: int, channels: int):
        self.processed_root = Path(processed_root)
        self.splits_dir = Path(splits_dir)
        self.split_id = int(split_id)
        self.train = bool(train)
        self.clip_len = int(clip_len)
        self.num_joints = int(num_joints)
        self.channels = int(channels)

        class_to_idx = read_class_ind(self.splits_dir)
        self.items = read_train_list(self.splits_dir, self.split_id) if self.train else read_test_list(self.splits_dir, self.split_id, class_to_idx)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        rel_avi, y = self.items[i]
        p = to_npz_path(self.processed_root, rel_avi)
        if not p.exists():
            raise FileNotFoundError(f"Missing processed file: {p}")

        d = np.load(p)
        if "x" not in d:
            raise KeyError(f"Processed file must contain key 'x': {p}")

        x = d["x"].astype(np.float32)  # [T,V,C]
        x = crop_or_pad(x, self.clip_len)
        x = x[:, :self.num_joints, :self.channels]
        x = np.transpose(x, (2, 0, 1))  # [C,T,V]

        return torch.from_numpy(x), torch.tensor(int(y), dtype=torch.long), rel_avi
