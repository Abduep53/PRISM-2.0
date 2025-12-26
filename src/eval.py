import numpy as np
import torch
from prism.utils.metrics import compute_metrics

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device):
    model.eval()
    y_true, y_pred, ids = [], [], []

    for X, y, sid in loader:
        X = X.to(device)
        logits = model(X)
        pred = logits.argmax(dim=1).cpu().numpy().tolist()
        y_true += y.numpy().tolist()
        y_pred += pred
        ids += list(sid)

    m = compute_metrics(np.array(y_true, dtype=int), np.array(y_pred, dtype=int))
    return m, np.array(y_true, dtype=int), np.array(y_pred, dtype=int), ids
