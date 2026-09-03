from __future__ import annotations
from typing import Sequence, Optional
import torch
import torch.nn as nn

try:
    from .eca_block import ECABlock
except ImportError:
    from eca_block import ECABlock


# ============================================================================
# 1) CORE REP_SHUFFLE & LARGE KERNEL BRANCH
# ============================================================================

def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)

def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=conv.weight.device)
    w_fused = conv.weight * t
    b_fused = bn.bias + (b_conv - bn.running_mean) * (bn.weight / std)
    return w_fused, b_fused

class LargeKernelBranch(nn.Module):

    def __init__(self, dim: int, K: int = 7, stride: int = 1):
        super().__init__()
        assert K in (3, 7), "K chỉ hỗ trợ 3 hoặc 7."
        self.dim = dim
        self.K = K
        self.stride = stride

        self.conv_nor = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True),
        )

        configs = [(7, 1), (3, 2), (3, 3)] if K == 7 else [(3, 1), (1, 1)]
        self.branches_config = configs
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, k, padding=((k - 1) * d) // 2, groups=dim, dilation=d, bias=False),
                nn.BatchNorm2d(dim),
            )
            for k, d in configs
        ])

        self.spatial_fuse = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )

        self.reparam_conv: nn.Conv2d | None = None
        self._deployed = False

    def forward_dilated(self, x: torch.Tensor) -> torch.Tensor:
        sa = self.conv_nor(x)
        if self.reparam_conv is not None:
            return self.reparam_conv(sa)
        return sum(b(sa) for b in self.branches)

    def forward_fuse(self, sa: torch.Tensor) -> torch.Tensor:
        return self.spatial_fuse(sa)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_fuse(self.forward_dilated(x))

    def _to_target_k(self, kernel: torch.Tensor, orig_k: int, d: int) -> torch.Tensor:
        c, m = kernel.shape[:2]
        kd = (orig_k - 1) * d + 1
        out = torch.zeros((c, m, self.K, self.K), device=kernel.device, dtype=kernel.dtype)
        offset = (self.K - kd) // 2
        out[:, :, offset:offset + kd:d, offset:offset + kd:d] = kernel
        return out

    def switch_to_deploy(self) -> None:
        if self._deployed:
            return

        W_equiv, B_equiv = 0, 0
        for branch, (k_size, dil) in zip(self.branches, self.branches_config):
            w, b = _fuse_conv_bn(branch[0], branch[1])
            W_equiv += self._to_target_k(w, orig_k=k_size, d=dil)
            B_equiv += b

        self.reparam_conv = nn.Conv2d(self.dim, self.dim, self.K, padding=self.K // 2, groups=self.dim, bias=True)
        self.reparam_conv.weight.data = W_equiv
        self.reparam_conv.bias.data = B_equiv
        del self.branches

        w_nor, b_nor = _fuse_conv_bn(self.conv_nor[0], self.conv_nor[1])
        new_conv_nor = nn.Conv2d(self.dim, self.dim, 3, stride=self.stride, padding=1, groups=self.dim, bias=True)
        new_conv_nor.weight.data = w_nor
        new_conv_nor.bias.data = b_nor
        self.conv_nor = nn.Sequential(new_conv_nor, nn.ReLU6(inplace=True))

        w_sf, b_sf = _fuse_conv_bn(self.spatial_fuse[0], self.spatial_fuse[1])
        new_sf = nn.Conv2d(self.dim, self.dim, 1, bias=True)
        new_sf.weight.data = w_sf
        new_sf.bias.data = b_sf
        self.spatial_fuse = new_sf

        self._deployed = True


class Rep_Shuffle(nn.Module):
    def __init__(self, dim: int, K: int = 7, stride: int = 1):
        super().__init__()
        assert stride in (1, 2), "Stride chỉ nhận 1 hoặc 2."
        assert K in (3, 7), "K chỉ hỗ trợ 3 hoặc 7."

        self.dim = dim
        self.stride = stride

        if stride == 1:
            assert dim % 2 == 0, "dim phải chẵn khi stride=1."
            self.branch_dim = dim // 2
        else:
            self.branch_dim = dim

        bd = self.branch_dim
        self.ca_conv = nn.Conv2d(bd, bd, kernel_size=3, stride=stride, padding=1, groups=bd, bias=False)
        self.ca_bn = nn.BatchNorm2d(bd)
        self._ca_deployed = False

        self.sa_branch = LargeKernelBranch(bd, K=K, stride=stride)
        self.eca = ECABlock(bd)

    def _ca_dw(self, x: torch.Tensor) -> torch.Tensor:
        if self._ca_deployed:
            return self.ca_conv(x)
        return self.ca_bn(self.ca_conv(x))

    def _run_branches(self, x_ca: torch.Tensor, x_sa: torch.Tensor):
        ca_dw = self._ca_dw(x_ca)

        sigmoid_ca = self.eca.get_gate(ca_dw)
        ca_out = ca_dw * sigmoid_ca
        
        sa_dilated = self.sa_branch.forward_dilated(x_sa)
        sa_out = self.sa_branch.forward_fuse(sa_dilated * sigmoid_ca)

        return ca_out, sa_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            identity = x
            x_ca, x_sa = torch.chunk(x, 2, dim=1)
            ca, sa = self._run_branches(x_ca, x_sa)
            out = torch.cat([ca, sa], dim=1)
            return channel_shuffle(out, groups=2) + identity
        else:
            ca, sa = self._run_branches(x, x)
            out = torch.cat([ca, sa], dim=1)
            return channel_shuffle(out, groups=2)

    def switch_to_deploy(self) -> None:
        self.sa_branch.switch_to_deploy()
        if not self._ca_deployed:
            w, b = _fuse_conv_bn(self.ca_conv, self.ca_bn)
            new_conv = nn.Conv2d(self.branch_dim, self.branch_dim, 3, stride=self.stride, padding=1, groups=self.branch_dim, bias=True)
            new_conv.weight.data = w
            new_conv.bias.data = b
            self.ca_conv = new_conv
            del self.ca_bn
            self._ca_deployed = True


# ============================================================================
# 1.5) STEM
# ============================================================================

class Stem(nn.Module):
   
    def __init__(self, in_channels: int = 3, out_channels: int = 16, stride: int = 1):
        super().__init__()
        assert stride in (1, 2), "Stem stride chỉ nhận 1 hoặc 2."
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def switch_to_deploy(self) -> None:
        conv, bn = self.net[0], self.net[1]
        w, b = _fuse_conv_bn(conv, bn)
        new_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                              stride=conv.stride, padding=conv.padding, bias=True)
        new_conv.weight.data = w
        new_conv.bias.data = b
        self.net = nn.Sequential(new_conv, self.net[2])


# ============================================================================
# 2) REPSHUFFLE STAGE & BACKBONE (WITH OPTIONAL STEM, MULTI-SCALE OUTPUT)
# ============================================================================

class RepShuffleStage(nn.Module):

    def __init__(self, in_channels: int, num_blocks_s1: int, K: int = 7):
        super().__init__()
        assert num_blocks_s1 >= 0
        
        # 1. Block Downsample đầu Stage (stride=2): Nhân đôi channel (vd: 16 -> 32)
        self.downsample_block = Rep_Shuffle(dim=in_channels, K=K, stride=2)
        out_channels = in_channels * 2

        # 2. Thân Stage (stride=1): Giữ nguyên số channel
        self.blocks_s1 = nn.ModuleList([
            Rep_Shuffle(dim=out_channels, K=K, stride=1)
            for _ in range(num_blocks_s1)
        ])
        self.out_channels = out_channels
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_block(x)
        for block in self.blocks_s1:
            x = block(x)
        return x


class RepShuffleBackbone(nn.Module):

    # Preset S/M/L — chỉ đổi độ RỘNG (channel) và ĐỘ SÂU (số block); độ phân giải
    # không gian (56/28/14) là bất biến theo scale, chỉ phụ thuộc img_size + stem_stride.
    SCALE_PRESETS = {
        "S": dict(stem_channels=16, stage_repeats_s1=(2, 2, 2), K=3),
        "M": dict(stem_channels=24, stage_repeats_s1=(2, 2, 2), K=3),
        "L": dict(stem_channels=32, stage_repeats_s1=(3, 3, 3), K=3),
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
        K: int = 7,
        **kwargs
    ):
        super().__init__()

        repeats = stage_depths or stage_repeats_s1 or stage_repeats
        if repeats is None:
            repeats = (2, 4, 2)

        if len(repeats) == 4:
            repeats = repeats[1:]

        assert len(repeats) == 3, f"Backbone cần 3 giá trị lặp cho 3 stage, nhưng nhận được: {repeats}"
        self.stage_repeats_s1 = tuple(repeats)
        n1, n2, n3 = self.stage_repeats_s1

        self.K = K
        self.use_stem = use_stem

        if use_stem:
            self.stem = Stem(in_channels=in_channels, out_channels=stem_channels, stride=stem_stride)
            stage1_in = stem_channels
        else:
            self.stem = None
            stage1_in = in_channels

        # Stage 1: stride=2 -> nhân đôi channel
        self.stage1 = RepShuffleStage(in_channels=stage1_in, num_blocks_s1=n1, K=K)
        c1 = self.stage1.out_channels

        # Stage 2: stride=2 -> nhân đôi channel
        self.stage2 = RepShuffleStage(in_channels=c1, num_blocks_s1=n2, K=K)
        c2 = self.stage2.out_channels

        # Stage 3: stride=2 -> nhân đôi channel
        self.stage3 = RepShuffleStage(in_channels=c2, num_blocks_s1=n3, K=K)
        c3 = self.stage3.out_channels

        self.out_channels = c3          
        self.out_channels_list = [c1, c2, c3]   

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.stem is not None:
            x = self.stem(x)
        f1 = self.stage1(x)  
        f2 = self.stage2(f1) 
        f3 = self.stage3(f2)  
        return [f1, f2, f3]

    def switch_to_deploy(self) -> None:

        if self.stem is not None:
            self.stem.switch_to_deploy()
        for m in self.modules():
            if isinstance(m, Rep_Shuffle):
                m.switch_to_deploy()