import math
import torch
import torch.nn as nn


class ECABlock(nn.Module):


    def __init__(self, dim: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs((math.log2(dim) + b) / gamma))
        k = k if k % 2 else k + 1  # ép kernel size thành số lẻ
        k = max(k, 3)              # tối thiểu 3

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def get_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Chỉ tính sigmoid gate (B, C, 1, 1), KHÔNG nhân vào x.

        Dùng khi cần lấy `sigmoid_ca` ra để trao đổi chéo với nhánh SA
        (xem Rep_Shuffle._run_branches), thay vì gọi forward() bình thường
        (forward() vẫn giữ nguyên hành vi cũ: tự nhân sigmoid vào x)."""
        y = self.avg_pool(x)                      # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)        # (B, 1, C)
        y = self.conv(y)                           # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)       # (B, C, 1, 1)
        return self.sigmoid(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.get_gate(x)
        return x * y.expand_as(x)