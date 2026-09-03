from __future__ import annotations
from typing import List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


def _dw_smooth(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
        nn.BatchNorm2d(channels),
        nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(channels),
    )


class FPNLateralNeck(nn.Module):
    def __init__(self, in_channels_list: Sequence[int], out_channels: int):
        super().__init__()
        assert len(in_channels_list) == 3, \
            f"Neck hiện cố định cho backbone 3-stage (f1,f2,f3), nhận {len(in_channels_list)} level."
        c1, c2, c3 = in_channels_list  # fine -> coarse

        # Lateral 1x1: đồng bộ channel về out_channels (giữ BN cho ổn định — nhất quán
        # với style BN xuyên suốt project, khác với FPN gốc dùng lateral trần không BN).
        self.lateral1 = nn.Sequential(nn.Conv2d(c1, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.lateral2 = nn.Sequential(nn.Conv2d(c2, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.lateral3 = nn.Sequential(nn.Conv2d(c3, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))

        self.smooth2 = _dw_smooth(out_channels)  # áp sau khi merge p3 -> p2
        self.smooth1 = _dw_smooth(out_channels)  # áp sau khi merge p2 -> p1

        self.out_channels = out_channels

    def forward(self, feats_fine_to_coarse: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(feats_fine_to_coarse) == 3, \
            f"Neck cần đúng 3 level feature thật (f1,f2,f3), nhận {len(feats_fine_to_coarse)}."
        f1, f2, f3 = feats_fine_to_coarse

        p3 = self.lateral3(f3)

        p2 = self.lateral2(f2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        p2 = self.smooth2(p2)

        p1 = self.lateral1(f1) + F.interpolate(p2, scale_factor=2, mode="nearest")
        p1 = self.smooth1(p1)

        return [p2, p1]  # coarse -> fine: [context ~28x28, detail ~56x56]