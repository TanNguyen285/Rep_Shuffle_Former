import math
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

MixerType = Literal["linear_attn"]
Scale = Literal["s", "m", "l"]


@dataclass
class ModelConfig:
    # ---------------- Input / Task ----------------
    in_channels: int = 3
    img_size: int = 224                    # 224 + stem_stride=2 -> f1=56,f2=28,f3=14 (khớp neck 2-level)
    num_classes: int = 22
    batch_size: int = 32
    label_smoothing: float = 0.00
    budget_weight: float = 0.00

    # ---------------- Backbone ----------------
    backbone_stem_channels: int = 24
    backbone_stem_stride: int = 2          # 2 -> tổng downsample tới neck-coarse = 2*2*2=8x (224/8=28)
    backbone_stage_repeats_s1: Tuple[int, int, int] = (2, 2, 2)
    K: int = 3
    feature_grid_size: Optional[int] = None

    # ---------------- Tokenizer ----------------
    # FPNLateralNeck (top-down FPN) chỉ trả về 2 level thật cho tokenizer: [context, detail]
    # với context = grid của f2, detail = grid của f1. coarsest_grid PHẢI chia hết grid
    # context đó theo đúng luỹ thừa 2 (xem validate()). Với default img=224/stem_stride=2
    # -> context=28 -> coarsest_grid=7 hợp lệ (28/7=4=2^2); coarsest_grid=4 SẼ VỠ (28/4=7, không phải luỹ thừa 2).
    coarsest_grid: int = 7
    token_dim: int = 128
    token_order: Literal["raster", "morton"] = "morton"
    scale: Literal["S", "M", "L"] = "S"

    # ---------------- Transformer ----------------
    mixer_type: MixerType = "linear_attn"
    mixer_depth: int = 4
    transformer_heads: int = 4
    transformer_mlp_ratio: float = 4.0
    mixer_dropout: float = 0.1

    # ---------------- Scale presets ----------------
    @classmethod
    def from_scale(cls, scale: Scale, **overrides):
        presets = {
            "s": dict(
                backbone_stem_channels=16,
                backbone_stage_repeats_s1=(2, 2, 2),
                token_dim=128,
                mixer_depth=3,
                transformer_heads=4,
                scale="S",
            ),
            "m": dict(
                backbone_stem_channels=24,
                backbone_stage_repeats_s1=(2, 2, 2),
                token_dim=192,
                mixer_depth=6,
                transformer_heads=6,
                scale="M",
            ),
            "l": dict(
                backbone_stem_channels=32,
                backbone_stage_repeats_s1=(3, 3, 3),
                token_dim=320,
                mixer_depth=10,
                transformer_heads=8,
                scale="L",
            ),
        }

        if scale not in presets:
            raise ValueError("scale phải là s/m/l")

        cfg = cls(**presets[scale], **overrides)
        cfg.validate()
        return cfg

    def _neck_context_grid(self) -> int:
        """Grid của level 'context' (f2) mà FPNLateralNeck trả về cho tokenizer —
        đúng bằng grid coarse nhất mà AdaptiveQuadtreeTokenizer nhận (stage_grid_sizes[0]).
        Tổng downsample tới f2 = stem_stride * 2 (stage1) * 2 (stage2)."""
        total_stride_to_f2 = self.backbone_stem_stride * 4
        assert self.img_size % total_stride_to_f2 == 0, (
            f"img_size={self.img_size} không chia hết cho tổng stride tới f2 "
            f"({total_stride_to_f2} = stem_stride*4)."
        )
        return self.img_size // total_stride_to_f2

    def validate(self):
        assert self.backbone_stem_channels % 2 == 0
        assert len(self.backbone_stage_repeats_s1) == 3
        assert self.backbone_stem_stride in (1, 2)
        assert self.token_dim % self.transformer_heads == 0
        assert self.scale in ("S", "M", "L")

        # coarsest_grid phải chia hết grid context (f2) theo đúng luỹ thừa 2 — bắt lỗi
        # sớm ở config thay vì để AdaptiveQuadtreeTokenizer.__init__ mới nổ assert.
        context_grid = self._neck_context_grid()
        assert context_grid % self.coarsest_grid == 0, (
            f"coarsest_grid={self.coarsest_grid} phải chia hết grid context của neck "
            f"({context_grid}, suy từ img_size={self.img_size} & backbone_stem_stride={self.backbone_stem_stride})."
        )
        ratio = context_grid / self.coarsest_grid
        k = round(math.log2(ratio))
        assert 2 ** k == ratio, (
            f"coarsest_grid={self.coarsest_grid} phải nhỏ hơn grid context ({context_grid}) "
            f"theo đúng luỹ thừa 2 (vd context=28 -> coarsest_grid hợp lệ: 28, 14, 7)."
        )