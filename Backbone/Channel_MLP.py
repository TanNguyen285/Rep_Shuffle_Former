from __future__ import annotations
import torch
import torch.nn as nn


class ConvFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        hidden = max(int(dim * mlp_ratio), dim)

        # Pointwise expand
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)

        # Depthwise 3x3 (spatial mixing trên hidden dim)
        self.dw = nn.Conv2d(
            hidden, hidden, kernel_size=3, padding=1,
            groups=hidden, bias=False
        )
        self.bn_dw = nn.BatchNorm2d(hidden)

        self.act = nn.ReLU6(inplace=True)

        # Pointwise project
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.pw1(x)))
        x = self.act(self.bn_dw(self.dw(x)))
        x = self.bn2(self.pw2(x))
        return x