import torch.nn as nn

class LSTMBaseline(nn.Module):
    """
    Input: [B,C,T,V] -> flatten joints per frame -> LSTM -> classifier
    """
    def __init__(self, num_classes: int, channels: int, num_joints: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.in_dim = channels * num_joints
        self.lstm = nn.LSTM(
            input_size=self.in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        B, C, T, V = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * V)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
