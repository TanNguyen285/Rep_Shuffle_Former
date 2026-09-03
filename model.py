from __future__ import annotations
import torch
import torch.nn as nn

from backbone_rep_shuffle import RepShuffleBackbone
from neck import FPNLateralNeck
from linear_transformer import LinearTransformer


class RepShuffleFormer(nn.Module):
    """
    VERSION DEBUG:
    - BỎ tokenizer (quadtree)
    - Dùng feature p2 (28x28) -> flatten -> transformer
    - Mục tiêu: kiểm tra backbone + neck + transformer có học được không
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 22,
        img_size: int = 224,

        # Backbone
        stem_channels: int = 16,
        stem_stride: int = 2,
        stage_repeats=(2, 2, 2),
        K: int = 3,

        # Transformer
        token_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # =====================
        # 1. Backbone
        # =====================
        self.backbone = RepShuffleBackbone(
            in_channels=in_channels,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem=True,
            stage_repeats_s1=stage_repeats,
            K=K,
        )
        c1, c2, c3 = self.backbone.out_channels_list

        # =====================
        # 2. Neck (FPN)
        # =====================
        self.neck = FPNLateralNeck(
            in_channels_list=[c1, c2, c3],
            out_channels=token_dim
        )

        # =====================
        # 3. Transformer
        # =====================
        self.transformer = LinearTransformer(
            dim=token_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # =====================
        # 4. Head
        # =====================
        self.head = nn.Linear(token_dim, num_classes)

    def forward(self, x: torch.Tensor):
        # ===== Backbone =====
        feats = self.backbone(x)

        # ===== Neck =====
        feat_pyramid = self.neck(feats)
        p2, p1 = feat_pyramid   # p2=28x28, p1=56x56

        # ===== CHỈ DÙNG p2 (ổn định) =====
        x = p2   # [B, C, 28, 28]

        B, C, H, W = x.shape

        # flatten -> tokens
        tokens = x.flatten(2).transpose(1, 2)   # [B, N=784, C]

        # mask full
        mask = torch.ones(B, tokens.shape[1], device=x.device)

        # ===== Transformer =====
        tokens = self.transformer(tokens, mask)

        # ===== Global pooling =====
        feat = tokens.mean(dim=1)

        logits = self.head(feat)

        return logits, {}   # aux empty


# =========================
# Builder
# =========================
def build_model(cfg):
    return RepShuffleFormer(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        img_size=cfg.img_size,

        stem_channels=cfg.backbone_stem_channels,
        stem_stride=getattr(cfg, "backbone_stem_stride", 1),
        stage_repeats=cfg.backbone_stage_repeats_s1,
        K=cfg.K,

        token_dim=cfg.token_dim,
        depth=cfg.mixer_depth,
        num_heads=cfg.transformer_heads,
        mlp_ratio=cfg.transformer_mlp_ratio,
        dropout=cfg.mixer_dropout,
    )