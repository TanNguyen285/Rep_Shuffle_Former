from __future__ import annotations

import torch
import torch.nn as nn

from Backbone.RepShuffle import RepShuffleBackbone
from head_seg import SegFormerHead


class SegmentationModel(nn.Module):

    def __init__(
        self,
        scale="S",
        in_channels=3,
        num_classes=2,
        stem_stride=2,
        embed_dim=256,
        dropout=0.1,
        **backbone_overrides,
    ):
        super().__init__()

        self.backbone = RepShuffleBackbone.from_scale(
            scale=scale,
            in_channels=in_channels,
            stem_stride=stem_stride,
            **backbone_overrides,
        )

        self.head = SegFormerHead(
            in_channels=self.backbone.out_channels_list,
            num_classes=num_classes,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        self.num_classes = num_classes

    def forward(self, x, return_features=False):

        input_size = x.shape[-2:]

        feats = self.backbone(x)

        logits = self.head(feats)

        logits = nn.functional.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        if return_features:
            return logits, feats

        return logits

    def switch_to_deploy(self):
        self.backbone.switch_to_deploy()

    @classmethod
    def from_scale(cls, scale, **kwargs):
        return cls(scale=scale, **kwargs)


if __name__ == "__main__":

    for scale in ["S", "M", "L"]:

        model = SegmentationModel(
            scale=scale,
            num_classes=2,
            embed_dim=256,
        )

        x = torch.randn(2, 3, 224, 224)

        logits, feats = model(
            x,
            return_features=True,
        )

        print(f"=== Scale {scale} (num_stages={model.backbone.num_stages}) ===")

        for i, f in enumerate(feats, 1):
            print(f"f{i}: {f.shape}")

        print("logits:", logits.shape)
        print()