from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from opacus import PrivacyEngine

from prism.models.lstm_baseline import LSTMBaseline
from prism.models.stgcn import STGCN

@dataclass
class TrainResult:
    model: torch.nn.Module
    dp_info: dict

def build_model(cfg: dict) -> torch.nn.Module:
    m = cfg["model"]
    d = cfg["data"]
    if m["type"] == "lstm":
        return LSTMBaseline(
            num_classes=int(d["num_classes"]),
            channels=int(d["channels"]),
            num_joints=int(d["num_joints"]),
            hidden_dim=int(m["hidden_dim"]),
            num_layers=int(m["num_layers"]),
            dropout=float(m["dropout"]),
        )
    if m["type"] == "stgcn":
        return STGCN(
            num_classes=int(d["num_classes"]),
            channels=int(d["channels"]),
            num_joints=int(d["num_joints"]),
            hidden_dim=int(m["hidden_dim"]),
            num_layers=int(m["num_layers"]),
            dropout=float(m["dropout"]),
        )
    raise ValueError(f"Unknown model type: {m['type']}")

def train_model(cfg: dict, train_ds, device: torch.device) -> TrainResult:
    train_cfg = cfg["train"]
    dp_cfg = cfg["dp"]

    model = build_model(cfg).to(device)
    model.train()

    loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    dp_info = {"enabled": False}
    privacy_engine = None

    if bool(dp_cfg.get("enabled", False)):
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=float(dp_cfg["noise_multiplier"]),
            max_grad_norm=float(dp_cfg["max_grad_norm"]),
        )
        dp_info = {
            "enabled": True,
            "target_epsilon": float(dp_cfg.get("target_epsilon", -1.0)),
            "delta": float(dp_cfg["delta"]),
            "noise_multiplier": float(dp_cfg["noise_multiplier"]),
            "max_grad_norm": float(dp_cfg["max_grad_norm"]),
            "accountant": "opacus_default",
        }

    epochs = int(train_cfg["epochs"])
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total = 0
        correct = 0

        for X, y, _ in tqdm(loader, desc=f"train epoch {epoch}/{epochs}", leave=False):
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * y.size(0)
            total += y.size(0)
            correct += int((logits.argmax(dim=1) == y).sum().item())

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        print(f"epoch={epoch} loss={avg_loss:.6f} train_acc={acc:.4f}")

    if privacy_engine is not None:
        eps = privacy_engine.accountant.get_epsilon(delta=dp_info["delta"])
        dp_info["achieved_epsilon"] = float(eps)

    return TrainResult(model=model, dp_info=dp_info)
