from __future__ import annotations
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv + BN + SiLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, g: int = 1):
        super().__init__()
        p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DetectionHead(nn.Module):
    """
    Anchor-free decoupled detection head (kiểu YOLOX/FCOS).

    Input: list feature map đa scale [f1, f2, f3] (từ backbone, stride tăng dần)
    Output (mỗi scale i, tại vị trí (h, w)):
        - cls_logits[i]: (B, num_classes, H_i, W_i)
        - reg[i]:        (B, 4, H_i, W_i)   -> (l, t, r, b) offset tới box, đã relu() để >= 0
        - obj[i]:        (B, 1, H_i, W_i)   -> objectness logit

    Head này KHÔNG decode ra box tuyệt đối — bạn tự viết hàm decode dùng
    stride thực tế của từng level (vd: box = decode(cls, reg, obj, stride)).
    """

    def __init__(
        self,
        in_channels_list: Sequence[int],
        num_classes: int,
        feat_channels: int = 96,
        num_convs: int = 2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = len(in_channels_list)

        self.stems = nn.ModuleList(
            [ConvBNAct(c, feat_channels, k=1) for c in in_channels_list]
        )

        self.cls_branch = nn.ModuleList(
            [self._make_branch(feat_channels, num_convs) for _ in in_channels_list]
        )
        self.reg_branch = nn.ModuleList(
            [self._make_branch(feat_channels, num_convs) for _ in in_channels_list]
        )

        self.cls_pred = nn.ModuleList(
            [nn.Conv2d(feat_channels, num_classes, 1) for _ in in_channels_list]
        )
        self.reg_pred = nn.ModuleList(
            [nn.Conv2d(feat_channels, 4, 1) for _ in in_channels_list]
        )
        self.obj_pred = nn.ModuleList(
            [nn.Conv2d(feat_channels, 1, 1) for _ in in_channels_list]
        )

        self._init_bias()

    @staticmethod
    def _make_branch(ch: int, num_convs: int) -> nn.Sequential:
        return nn.Sequential(*[ConvBNAct(ch, ch, k=3) for _ in range(num_convs)])

    def _init_bias(self, prior_prob: float = 0.01) -> None:
        # bias âm cho cls/obj để training ổn định lúc đầu (giống RetinaNet/YOLOX)
        b = -math.log((1 - prior_prob) / prior_prob)
        for m in self.cls_pred:
            nn.init.constant_(m.bias, b)
        for m in self.obj_pred:
            nn.init.constant_(m.bias, b)

    def forward(self, feats: Sequence[torch.Tensor]):
        assert len(feats) == self.num_levels, (
            f"Head expects {self.num_levels} scales, got {len(feats)}"
        )
        cls_outs, reg_outs, obj_outs = [], [], []
        for i, f in enumerate(feats):
            x = self.stems[i](f)
            cls_feat = self.cls_branch[i](x)
            reg_feat = self.reg_branch[i](x)

            cls_outs.append(self.cls_pred[i](cls_feat))
            obj_outs.append(self.obj_pred[i](reg_feat))
            reg_outs.append(F.relu(self.reg_pred[i](reg_feat)))  # l,t,r,b >= 0

        return cls_outs, reg_outs, obj_outs


if __name__ == "__main__":
    ch_list = [64, 128, 256]
    feats = [
        torch.randn(2, 64, 56, 56),
        torch.randn(2, 128, 28, 28),
        torch.randn(2, 256, 14, 14),
    ]

    det_head = DetectionHead(in_channels_list=ch_list, num_classes=20)
    cls_o, reg_o, obj_o = det_head(feats)
    print("[Detection]")
    for i, (c, r, o) in enumerate(zip(cls_o, reg_o, obj_o)):
        print(f"  level {i}: cls={c.shape} reg={r.shape} obj={o.shape}")
