from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from Backbone.RepShuffle import RepShuffleBackbone
from Head.head_cls import Classification  # GAP + Dropout + Linear


class Classification_Head(nn.Module):

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

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        feats = self.backbone(x)

        # Nếu không dùng Head (Feature Extractor thuần)
        if self.head is None:
            return feats

        # Đi qua Classification Head (lấy feature stage cuối cùng feats[-1])
        logits = self.head(feats[-1])

        if return_features:
            return logits, feats

        return logits

    def switch_to_deploy(self) -> None:
        """Re-parameterize toàn bộ backbone (Conv+BN fuse) để deploy."""
        self.backbone.switch_to_deploy()

    @classmethod
    def from_scale(cls, scale: str, **kwargs) -> Classification:
        return cls(scale=scale, **kwargs)


if __name__ == "__main__":

    for scale in ["S", "M", "L"]:
        print(f"=== Testing Classification Scale {scale} ===")

        # 1. Standard Forward Pass (Task Classification)
        model = Classification(
            scale=scale,
            num_classes=10,
            dropout=0.2,
        )
        x = torch.randn(2, 3, 224, 224)

        logits, feats = model(x, return_features=True)

        for i, f in enumerate(feats, 1):
            print(f"  f{i}: {f.shape}")

        print("  logits:", logits.shape)

        # 2. Re-parameterization (Structural Re-param) check
        model.eval()
        model.switch_to_deploy()
        out_deploy = model(x)
        print("  deploy forward success! logits:", out_deploy.shape)
        print()

    # 3. Test Feature Extractor Mode (Khi num_classes=None)
    print("=== Testing Feature Extractor Mode (num_classes=None) ===")
    feat_model = Classification(scale="S", num_classes=None)
    feats_only = feat_model(x)
    print("  multi-scale feats:", [f.shape for f in feats_only])