import torch
import torch.nn as nn


class Classification(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.pool(x).flatten(1)  # GAP
        x = self.dropout(x)
        x = self.fc(x)
        return x