from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    SegFormer Linear Embedding.
    """

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer All-MLP Decoder.

    Input:
        Scale S:      [c1, c2, c3]       (3 stage, c1 = H/4,W/4 ... c3 = H/16,W/16)
        Scale M/L:    [c1, c2, c3, c4]   (4 stage, thêm c4 = H/32,W/32)

    Output:
        B, num_classes, H/4, W/4
        (mọi level được resize về kích thước của c1 — level có độ phân giải LỚN NHẤT)
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        embed_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_levels = len(in_channels)
        self.embed_dim = embed_dim

        self.linear_layers = nn.ModuleList([
            MLP(c, embed_dim)
            for c in in_channels
        ])

        self.linear_fuse = nn.Conv2d(
            embed_dim * self.num_levels,
            embed_dim,
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(embed_dim)
        self.act = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(dropout)

        self.linear_pred = nn.Conv2d(
            embed_dim,
            num_classes,
            kernel_size=1,
        )

    def forward(self, inputs):

        assert len(inputs) == self.num_levels, (
            f"SegFormerHead được khởi tạo cho {self.num_levels} level "
            f"(in_channels={self.in_channels}) nhưng nhận {len(inputs)} feature map. "
            f"Kiểm tra lại backbone (scale S=3 stage, M/L=4 stage) có khớp với head không."
        )

        # unpack tường minh theo đúng số level, để tên biến c1..c4 giữ ý nghĩa dễ đọc
        if self.num_levels == 3:
            c1, c2, c3 = inputs
        elif self.num_levels == 4:
            c1, c2, c3, c4 = inputs
        else:
            c1 = inputs[0]  # fallback tổng quát nếu sau này có 5+ stage

        n = c1.shape[0]
        target_size = c1.shape[-2:]  # luôn resize về c1 (độ phân giải lớn nhất)

        outs = []

        for i, x in enumerate(inputs):

            h, w = x.shape[-2:]

            x = self.linear_layers[i](x)

            x = x.permute(0, 2, 1).reshape(
                n,
                self.embed_dim,
                h,
                w,
            )

            if x.shape[-2:] != target_size:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            outs.append(x)

        x = torch.cat(outs, dim=1)

        x = self.linear_fuse(x)
        x = self.bn(x)
        x = self.act(x)

        x = self.dropout(x)

        x = self.linear_pred(x)

        return x