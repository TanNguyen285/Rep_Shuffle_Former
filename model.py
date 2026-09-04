from __future__ import annotations
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from Backbone.RepShuffle import RepShuffleBackbone
from Head import Classification  # head.py: định nghĩa riêng (GAP + Dropout + Linear)


class Model(nn.Module):
  

    def __init__(
        self,
        scale: str = "S",
        in_channels: int = 3,
        num_classes: Optional[int] = None,
        stem_stride: int = 2,
        dropout: float = 0.2,
        **backbone_overrides,
    ):
        super().__init__()

        self.backbone = RepShuffleBackbone.from_scale(
            scale=scale,
            in_channels=in_channels,
            stem_stride=stem_stride,
            **backbone_overrides,
        )

        self.num_classes = num_classes
        self.head = (
            Classification(
                in_channels=self.backbone.out_channels,
                num_classes=num_classes,
                dropout=dropout,
            )
            if num_classes is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        feats = self.backbone(x)  # [f1, f2, f3]
        if self.head is None:
            return feats
        return self.head(feats[-1])

    def switch_to_deploy(self) -> None:
        """Re-parameterize toàn bộ backbone (Conv+BN fuse) để deploy."""
        self.backbone.switch_to_deploy()

    @classmethod
    def from_scale(cls, scale: str, **kwargs) -> "Model":
        return cls(scale=scale, **kwargs)


if __name__ == "__main__":
    # Quick sanity check
    model = Model(scale="S", num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("logits:", out.shape)

    feat_model = Model(scale="S", num_classes=None)
    feats = feat_model(x)
    print("multi-scale feats:", [f.shape for f in feats])