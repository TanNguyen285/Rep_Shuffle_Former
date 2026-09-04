from __future__ import annotations
from typing import Sequence, Optional
import torch
import torch.nn as nn

from Backbone.Channel_MLP import ConvFFN
from Backbone.Mixer_Token import TokenMixer


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
   
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)
    w_fused = conv.weight * t
    b_fused = bn.bias + (b_conv - bn.running_mean) * (bn.weight / std)
    return w_fused, b_fused


def _fuse_into(module: nn.Module, conv_name: str, bn_name: str, **conv_kwargs) -> None:
   
    conv: nn.Conv2d = getattr(module, conv_name)
    bn: nn.BatchNorm2d = getattr(module, bn_name)
    w, b = _fuse_conv_bn(conv, bn)
    new_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, bias=True,
        **conv_kwargs,
    )
    new_conv.weight.data, new_conv.bias.data = w, b
    setattr(module, conv_name, new_conv)
    setattr(module, bn_name, nn.Identity())


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        mask.div_(keep_prob)
    return x * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-2):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.view(1, -1, 1, 1)


class RepShuffleBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        stride: int = 1,
        dim_out: Optional[int] = None,
        mlp_ratio: float = 2.0,
        layer_scale_init: float = 1e-2,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.stride = stride
        self.dim = dim
        self.dim_out = dim_out if dim_out is not None else dim
        self.token_mixer = TokenMixer(dim, stride=stride, dim_out=self.dim_out if stride == 2 else None)

        if stride == 1:
            assert self.dim_out == dim, "stride=1 block không được đổi số kênh"
            self.ffn = ConvFFN(dim, mlp_ratio=mlp_ratio)
            self.ls1 = LayerScale(dim, layer_scale_init)
            self.ls2 = LayerScale(dim, layer_scale_init)
            self.drop_path1 = DropPath(drop_path)
            self.drop_path2 = DropPath(drop_path)
        else:
            self.ffn = None

        self._deployed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x = x + self.drop_path1(self.ls1(self.token_mixer(x)))
            x = x + self.drop_path2(self.ls2(self.ffn(x)))
            return x
        # stride == 2: downsample thuần, không skip connection
        return self.token_mixer(x)

    def switch_to_deploy(self) -> None:
   
        if self._deployed:
            return

        tm = self.token_mixer
        bd = tm.branch_dim

        _fuse_into(
            tm, "ca_dwconv", "ca_bn",
            kernel_size=3, stride=tm.stride, padding=1, groups=bd,
        )
        _fuse_into(
            tm, "sa_dwconv", "sa_bn",
            kernel_size=3, stride=tm.stride, padding=1, groups=bd,
        )

        # 1b. Fuse lớp expand (chỉ tồn tại khi downsample đổi số kênh)
        if tm.expand is not None:
            expand_conv, expand_bn = tm.expand[0], tm.expand[1]
            w, b = _fuse_conv_bn(expand_conv, expand_bn)
            new_expand = nn.Conv2d(
                expand_conv.in_channels, expand_conv.out_channels,
                kernel_size=1, bias=True,
            )
            new_expand.weight.data, new_expand.bias.data = w, b
            tm.expand = new_expand  # giờ chỉ còn 1 conv, forward vẫn gọi self.expand(out) bình thường

        # 2. Fuse Conv + BN trong ConvFFN (nếu có) — depthwise separable: pw1 -> dw -> pw2
        if self.ffn is not None:
            _fuse_into(self.ffn, "pw1", "bn1", kernel_size=1)
            _fuse_into(self.ffn, "dw", "bn_dw", kernel_size=3, padding=1, groups=self.ffn.dw.groups)
            _fuse_into(self.ffn, "pw2", "bn2", kernel_size=1)

        self._deployed = True


class Stem(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 16, stride: int = 1):
        super().__init__()
        assert stride in (1, 2), "Stem stride chỉ nhận 1 hoặc 2."
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def switch_to_deploy(self) -> None:
        w, b = _fuse_conv_bn(self.conv, self.bn)
        new_conv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, kernel_size=3,
                              stride=self.stride, padding=1, bias=True)
        new_conv.weight.data, new_conv.bias.data = w, b
        self.conv = new_conv
        self.bn = nn.Identity()


class RepShuffleStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_blocks_s1: int,
        mlp_ratio: float = 2.0,
        layer_scale_init: float = 1e-2,
        drop_path_rates: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        assert num_blocks_s1 >= 0

        out_channels = in_channels * 2
        self.downsample_block = RepShuffleBlock(dim=in_channels, stride=2, dim_out=out_channels)

        dpr = list(drop_path_rates) if drop_path_rates is not None else [0.0] * num_blocks_s1
        assert len(dpr) == num_blocks_s1
        self.blocks_s1 = nn.ModuleList([
            RepShuffleBlock(
                dim=out_channels, stride=1,
                mlp_ratio=mlp_ratio, layer_scale_init=layer_scale_init, drop_path=dpr[i],
            )
            for i in range(num_blocks_s1)
        ])
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_block(x)
        for block in self.blocks_s1:
            x = block(x)
        return x


class RepShuffleBackbone(nn.Module):
    # S: 3 stage (không có stage4) — nhẹ, dừng ở stride 16
    # M/L: 4 stage — thêm stage4 (stride 32) để lấy global context mạnh hơn
    SCALE_PRESETS = {
        "S": dict(stem_channels=16, stage_repeats_s1=(2, 4, 2), mlp_ratio=2.0),
        "M": dict(stem_channels=24, stage_repeats_s1=(2, 6, 2, 2), mlp_ratio=2.0),
        "L": dict(stem_channels=32, stage_repeats_s1=(3, 6, 3, 3), mlp_ratio=2.0),
    }

    @classmethod
    def from_scale(
        cls,
        scale: str,
        in_channels: int = 3,
        stem_stride: int = 2,
        use_stem: bool = True,
        **overrides,
    ) -> "RepShuffleBackbone":
        scale_key = scale.upper()
        assert scale_key in cls.SCALE_PRESETS, f"scale phải thuộc {list(cls.SCALE_PRESETS)}, nhận '{scale}'."
        cfg = dict(cls.SCALE_PRESETS[scale_key])
        cfg.update(overrides)
        return cls(in_channels=in_channels, stem_stride=stem_stride, use_stem=use_stem, **cfg)

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 16,
        use_stem: bool = True,
        stem_stride: int = 1,
        stage_repeats_s1: Optional[Sequence[int]] = None,
        stage_depths: Optional[Sequence[int]] = None,
        stage_repeats: Optional[Sequence[int]] = None,
        mlp_ratio: float = 2.0,
        layer_scale_init: float = 1e-2,
        drop_path_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        repeats = stage_depths or stage_repeats_s1 or stage_repeats
        if repeats is None:
            repeats = (2, 4, 2)
        repeats = tuple(repeats)

        # 3 giá trị -> không có stage4 (scale S)
        # 4 giá trị -> có stage4 (scale M/L)
        assert len(repeats) in (3, 4), (
            f"Backbone cần 3 giá trị lặp (S, không stage4) hoặc 4 giá trị lặp "
            f"(M/L, có stage4), nhưng nhận được: {repeats}"
        )

        self.stage_repeats_s1 = repeats
        self.num_stages = len(repeats)
        self.use_stage4 = self.num_stages == 4
        self.use_stem = use_stem

        if self.use_stage4:
            n1, n2, n3, n4 = repeats
        else:
            n1, n2, n3 = repeats
            n4 = 0

        total_blocks = n1 + n2 + n3 + n4
        if total_blocks > 0 and drop_path_rate > 0.0:
            dpr_all = [drop_path_rate * i / max(total_blocks - 1, 1) for i in range(total_blocks)]
        else:
            dpr_all = [0.0] * total_blocks

        dpr1 = dpr_all[:n1]
        dpr2 = dpr_all[n1:n1 + n2]
        dpr3 = dpr_all[n1 + n2:n1 + n2 + n3]
        dpr4 = dpr_all[n1 + n2 + n3:n1 + n2 + n3 + n4]

        if use_stem:
            self.stem = Stem(in_channels=in_channels, out_channels=stem_channels, stride=stem_stride)
            stage1_in = stem_channels
        else:
            self.stem = None
            stage1_in = in_channels

        self.stage1 = RepShuffleStage(in_channels=stage1_in, num_blocks_s1=n1,
                                       mlp_ratio=mlp_ratio, layer_scale_init=layer_scale_init,
                                       drop_path_rates=dpr1)
        c1 = self.stage1.out_channels

        self.stage2 = RepShuffleStage(in_channels=c1, num_blocks_s1=n2,
                                       mlp_ratio=mlp_ratio, layer_scale_init=layer_scale_init,
                                       drop_path_rates=dpr2)
        c2 = self.stage2.out_channels

        self.stage3 = RepShuffleStage(in_channels=c2, num_blocks_s1=n3,
                                       mlp_ratio=mlp_ratio, layer_scale_init=layer_scale_init,
                                       drop_path_rates=dpr3)
        c3 = self.stage3.out_channels

        if self.use_stage4:
            self.stage4 = RepShuffleStage(in_channels=c3, num_blocks_s1=n4,
                                           mlp_ratio=mlp_ratio, layer_scale_init=layer_scale_init,
                                           drop_path_rates=dpr4)
            c4 = self.stage4.out_channels
            self.out_channels = c4
            self.out_channels_list = [c1, c2, c3, c4]
        else:
            self.stage4 = None
            self.out_channels = c3
            self.out_channels_list = [c1, c2, c3]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.stem is not None:
            x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        if self.use_stage4:
            f4 = self.stage4(f3)
            return [f1, f2, f3, f4]
        return [f1, f2, f3]

    def switch_to_deploy(self) -> None:
        """Chuyển toàn bộ backbone sang dạng deployed."""
        if self.stem is not None:
            self.stem.switch_to_deploy()
        for m in self.modules():
            if isinstance(m, RepShuffleBlock):
                m.switch_to_deploy()