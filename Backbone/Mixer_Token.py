from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class TokenMixer(nn.Module):
    """
    Token Mixer v3 (Fix: downsample bay giờ thực sự đổi số kênh, residual thực sự cộng dồn):
    - Nhánh CA ("what"): Local DWConv + Global-conditioned gating (GAP)
    - Nhánh SA ("where"): Local DWConv + Cross-branch guide từ CA
    - `dim`  : số kênh input
    - `dim_out`: số kênh output mong muốn. None => bằng `dim` (block stride=1 thông thường).
                 Khi stride=2 và dim_out != dim (thường = dim*2), một lớp pointwise
                 conv+BN "expand" được thêm vào SAU khi ghép + shuffle 2 nhánh để
                 đổi đúng số kênh output — đây là phần còn thiếu ở bản gốc khiến
                 downsample không hề tăng kênh (depthwise conv giữ nguyên số kênh).
    """

    def __init__(self, dim: int, stride: int = 1, dim_out: Optional[int] = None):
        super().__init__()
        assert stride in (1, 2), "Stride chỉ có thể là 1 hoặc 2"
        assert dim % 2 == 0, "Dim phải chia hết cho 2 để tách 2 nhánh"

        self.dim = dim
        self.dim_out = dim_out if dim_out is not None else dim
        self.stride = stride
        # Cố định mỗi nhánh chiếm dim // 2 kênh để khi cat lại luôn bằng dim
        self.branch_dim = dim // 2
        bd = self.branch_dim

        # ===== Nhánh CA ("what") =====
        self.ca_dwconv = nn.Conv2d(
            bd, bd, kernel_size=3, stride=stride, padding=1,
            groups=bd, bias=False
        )
        self.ca_bn = nn.BatchNorm2d(bd)
        self.global_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(bd, bd, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # ===== Nhánh SA ("where") =====
        self.sa_dwconv = nn.Conv2d(
            bd, bd, kernel_size=3, stride=stride, padding=1,
            groups=bd, bias=False
        )
        self.sa_bn = nn.BatchNorm2d(bd)

        # Cross-branch projection (CA -> SA)
        self.cross_proj = nn.Conv2d(bd, bd, kernel_size=1, bias=False)

        self.act = nn.ReLU6(inplace=True)

        # ===== Expand kênh (chỉ tồn tại khi cần đổi số kênh, vd. downsample) =====
        if self.dim_out != self.dim:
            self.expand = nn.Sequential(
                nn.Conv2d(self.dim, self.dim_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.dim_out),
            )
        else:
            self.expand = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tách x thành 2 nhánh cân bằng (mỗi nhánh bd = dim // 2 kênh)
        x_ca, x_sa = torch.chunk(x, 2, dim=1)

        # ===== Nhánh CA =====
        ca_feat = self.ca_dwconv(x_ca)
        if self.stride == 1:
            # Residual thực sự (trước đây là x + (y - x) = y, tức là no-op)
            ca_feat = x_ca + ca_feat
        ca = self.ca_bn(ca_feat)
        g = self.global_fc(ca)
        ca = self.act(ca * g)

        # ===== Nhánh SA =====
        sa_feat = self.sa_dwconv(x_sa)
        if self.stride == 1:
            sa_feat = x_sa + sa_feat
        sa = self.sa_bn(sa_feat)

        # Cross-branch guiding
        guide = torch.sigmoid(self.cross_proj(ca))
        sa = self.act(sa * guide)

        # Ghép 2 nhánh lại (bd + bd = dim) và xáo trộn channel
        out = torch.cat([ca, sa], dim=1)
        out = channel_shuffle(out, groups=2)

        # Đổi số kênh nếu đây là downsample block (dim_out != dim)
        if self.expand is not None:
            out = self.expand(out)
        return out