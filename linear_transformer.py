from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


class LinearMultiHeadAttention(nn.Module):
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, eps: float = 1e-5):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) phải chia hết cho num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # TỐI ƯU COMPUTE: Norm trên toàn bộ dim D [B, N, D] nhẹ hơn rất nhiều
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type, enabled=False):
            x = x.float()
            q = self.q_norm(self.q_proj(x))
            k = self.k_norm(self.k_proj(x))
            v = self.v_proj(x)

            q = q.view(B, N, H, d).transpose(1, 2)
            k = k.view(B, N, H, d).transpose(1, 2)
            v = v.view(B, N, H, d).transpose(1, 2)

            q, k, v = q.float(), k.float(), v.float()
            q = elu_feature_map(q)
            k = elu_feature_map(k)

            if mask is not None:
                m = mask[:, None, :, None].float()
                k = k * m
                v = v * m

            # Linear Attention Kernels: Tích KV có kích thước [B, H, d, d]
            kv = torch.einsum("bhnd,bhne->bhde", k, v)
            k_sum = k.sum(dim=2)
            numerator = torch.einsum("bhnd,bhde->bhne", q, kv)
            denominator = torch.einsum("bhnd,bhd->bhn", q, k_sum).clamp_min(self.eps)
            out = numerator / denominator.unsqueeze(-1)

            out = torch.clamp(out, min=-60000.0, max=60000.0)

            out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        if mask is not None:
            out = out * mask[..., None]

        return self.dropout(self.out_proj(out))


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearMultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class LinearTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            LinearTransformerBlock(dim, num_heads, mlp_ratio, dropout) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, mask=mask)
        return self.norm(x)