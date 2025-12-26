import torch
import torch.nn as nn
import torch.nn.functional as F

def build_adjacency(num_joints: int) -> torch.Tensor:
    """
    Builds a simple normalized adjacency matrix.
    Keep this fixed for reproducibility.
    """
    A = torch.eye(num_joints, dtype=torch.float32)
    for i in range(num_joints - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    deg = A.sum(dim=1).clamp(min=1.0)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt

class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, dropout: float, stride: int = 1):
        super().__init__()
        self.register_buffer("A", A)
        self.spatial_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.temporal = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout)

        if in_channels != out_channels or stride != 1:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.res = nn.Identity()

    def forward(self, x):
        res = self.res(x)
        x = torch.einsum("bctv,vw->bctw", x, self.A)
        x = self.spatial_proj(x)
        x = self.temporal(x)
        x = self.bn(x)
        x = F.relu(x + res)
        return self.drop(x)

class STGCN(nn.Module):
    def __init__(self, num_classes: int, channels: int, num_joints: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        A = build_adjacency(num_joints)

        layers = []
        in_c = channels
        out_c = hidden_dim
        for i in range(num_layers):
            stride = 2 if i in (3, 6) else 1
            layers.append(STGCNBlock(in_c, out_c, A, dropout=dropout, stride=stride))
            in_c = out_c

        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_c, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)
