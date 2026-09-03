from typing import List, Literal, Optional, Sequence, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

TokenOrder = Literal["raster", "morton"]
Scale = Literal["S", "M", "L"]


def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Straight-through Gumbel-Sigmoid. CHỈ gọi khi cần rời rạc hoá cứng có gradient
    (Phase 2 / gumbel_hard=True). Không dùng cho nhánh soft — tránh nhiễu áp đảo logits
    nhỏ lúc đầu train."""
    device_type = logits.device.type if logits.device.type in ("cuda", "cpu") else "cpu"
    with torch.amp.autocast(device_type, enabled=False):  # fp32: log-log noise cần range rộng
        logits32 = logits.float()
        u1 = torch.rand_like(logits32).clamp(1e-6, 1 - 1e-6)
        u2 = torch.rand_like(logits32).clamp(1e-6, 1 - 1e-6)
        g1 = -torch.log(-torch.log(u1))
        g2 = -torch.log(-torch.log(u2))
        soft = torch.sigmoid((logits32 + g1 - g2) / tau)
        hard_val = (soft > 0.5).float()
        out = hard_val + soft - soft.detach()
    return out.to(logits.dtype)


def scale_grad(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.detach()
    return x * scale + (x - x * scale).detach()


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _morton_order_index(grid: int, device: torch.device) -> torch.Tensor:
    if not _is_pow2(grid):
        return torch.arange(grid * grid, device=device)

    bits = int(torch.log2(torch.tensor(float(grid))).round().item())
    assert 2 ** bits == grid, f"Morton order yêu cầu grid là luỹ thừa 2, nhận {grid}."

    rows = torch.arange(grid, device=device).view(-1, 1).expand(grid, grid).reshape(-1)
    cols = torch.arange(grid, device=device).view(1, -1).expand(grid, grid).reshape(-1)

    code = torch.zeros(grid * grid, dtype=torch.long, device=device)
    for i in range(bits):
        code = code | (((rows >> i) & 1) << (2 * i + 1))
        code = code | (((cols >> i) & 1) << (2 * i))

    return torch.argsort(code)


class SplitHead(nn.Module):
    """Sinh split-score cho 1 level của quadtree.

    Theo spec: score phải phụ thuộc CHỦ YẾU vào feature chi tiết nhất (F56-equivalent,
    tức `detail_feat` — đã downsample về đúng độ phân giải của level đang xét), và chỉ
    BỔ SUNG thêm ngữ cảnh từ chính feature của level đó (`feat_level`, đóng vai trò
    F28/context). `detail_weight` (0..1) quyết định tỉ trọng của nhánh detail so với
    nhánh context khi fuse — mặc định thiên hẳn về detail (0.7) để không bỏ sót object nhỏ.
    """

    def __init__(self, channels: int, detail_weight: float = 0.7):
        super().__init__()
        assert 0.0 <= detail_weight <= 1.0, "detail_weight phải nằm trong [0,1]."
        self.detail_weight = detail_weight

        # Nhánh context (feature riêng của level đang xét — có thể thô hơn F_detail)
        self.context_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        # Nhánh detail (feature chi tiết nhất, đã pool về đúng grid của level) — nhánh chủ đạo
        self.detail_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
        )
        nn.init.constant_(self.net[-1].bias, 0.5)

    def forward(self, feat_level: torch.Tensor, detail_feat: torch.Tensor) -> torch.Tensor:
        """
        feat_level:  feature của chính level này (context), [B,C,Gl,Gl]
        detail_feat: feature chi tiết nhất (F56-equivalent) đã pool sẵn về [B,C,Gl,Gl]
        """
        fused = (self.detail_weight * self.detail_proj(detail_feat)
                 + (1.0 - self.detail_weight) * self.context_proj(feat_level))
        return self.net(fused)


class PositionSizeEncoder(nn.Module):
    def __init__(self, token_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, token_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim // 2, token_dim),
        )

    def forward(self, geom: torch.Tensor) -> torch.Tensor:
        return self.mlp(geom)


class AdaptiveQuadtreeTokenizer(nn.Module):
    SCALE_PRESETS = {"S": 64, "M": 128, "L": 256}

    def __init__(
        self,
        in_channels: int,
        token_dim: int,
        coarsest_grid: int,
        stage_grid_sizes: Sequence[int],
        gumbel_tau: float = 1.0,
        token_order: TokenOrder = "morton",
        scale: Optional[Scale] = None,
        max_tokens: Optional[int] = None,
        target_ratio: float = 0.8,
        budget_loss_weight: float = 1.0,
        max_depth: Optional[int] = None,
        depth_bias: float = 0.0,
        cost_lambda: float = 0.0,
        gumbel_hard: bool = True,
        detail_scale_weight: float = 1.0,
        token_select_mode: Literal["order", "importance"] = "importance",
        detail_weight: float = 0.7,
    ):
        super().__init__()

        stage_grid_sizes = list(stage_grid_sizes)
        assert len(stage_grid_sizes) >= 2, "Cần ít nhất 2 level feature thật để tạo pyramid."
        for a, b in zip(stage_grid_sizes[:-1], stage_grid_sizes[1:]):
            assert b == a * 2, f"Các level backbone phải đúng gấp đôi nhau liên tiếp, nhận {stage_grid_sizes}"

        real_coarsest = stage_grid_sizes[0]
        assert real_coarsest % coarsest_grid == 0, \
            f"coarsest_grid={coarsest_grid} phải chia hết level coarse nhất của backbone ({real_coarsest})"
        self.num_extra_coarse_levels = int(
            torch.log2(torch.tensor(real_coarsest / coarsest_grid)).round().item()
        )
        assert 2 ** self.num_extra_coarse_levels * coarsest_grid == real_coarsest, \
            "coarsest_grid phải nhỏ hơn level coarse nhất của backbone theo đúng luỹ thừa 2."

        self.num_real_levels = len(stage_grid_sizes) - 1
        self.num_levels = self.num_extra_coarse_levels + self.num_real_levels
        self.G0 = coarsest_grid
        self.GL = stage_grid_sizes[-1]
        self.tau = gumbel_tau
        self.token_order = token_order
        self.depth_bias = depth_bias
        self.cost_lambda = cost_lambda

        # Công tắc runtime, set từ trainer.py mỗi epoch
        self.gumbel_hard = gumbel_hard
        self.detail_scale_weight = detail_scale_weight
        self.token_select_mode = token_select_mode

        if max_tokens is None:
            assert scale in self.SCALE_PRESETS, f"Cần truyền scale ({list(self.SCALE_PRESETS)}) hoặc max_tokens."
            max_tokens = self.SCALE_PRESETS[scale]
        self.max_tokens = max_tokens
        self.target_tokens = max(1.0, max_tokens * target_ratio)
        self.budget_loss_weight = budget_loss_weight
        self.max_depth = self.num_levels if max_depth is None else min(max_depth, self.num_levels)

        self.split_heads = nn.ModuleList(
            [SplitHead(in_channels, detail_weight=detail_weight) for _ in range(self.num_levels)]
        )
        self.token_proj = nn.Linear(in_channels, token_dim)
        self.pos_enc = PositionSizeEncoder(token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.token_dim = token_dim

        geom_buffers = []
        for l in range(self.num_levels + 1):
            Gl = coarsest_grid * (2 ** l)
            level_norm = l / max(1, self.num_levels)
            size = 1.0 / Gl
            ys, xs = torch.meshgrid(torch.arange(Gl), torch.arange(Gl), indexing="ij")
            cx = (xs.float() + 0.5) / Gl
            cy = (ys.float() + 0.5) / Gl
            geom = torch.stack(
                [cx.reshape(-1), cy.reshape(-1),
                 torch.full((Gl * Gl,), size), torch.full((Gl * Gl,), level_norm)],
                dim=-1,
            )
            order = _morton_order_index(Gl, device=torch.device("cpu")) if token_order == "morton" \
                else torch.arange(Gl * Gl)
            geom_buffers.append(geom[order])
            self.register_buffer(f"_order_l{l}", order, persistent=True)

        for l, g in enumerate(geom_buffers):
            self.register_buffer(f"_geom_l{l}", g, persistent=True)

    def _geom(self, l: int) -> torch.Tensor:
        return getattr(self, f"_geom_l{l}")

    def _order(self, l: int) -> torch.Tensor:
        return getattr(self, f"_order_l{l}")

    def _build_pyramid(self, feat_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        assert len(feat_pyramid) == self.num_real_levels + 1, \
            f"Tokenizer cần {self.num_real_levels + 1} level feature thật, nhận {len(feat_pyramid)}."

        extra = []
        cur = feat_pyramid[0]
        for _ in range(self.num_extra_coarse_levels):
            cur = F.avg_pool2d(cur, kernel_size=2, stride=2)
            extra.append(cur)
        extra.reverse()
        return extra + feat_pyramid

    def _compute_leaves(self, pyramid: List[torch.Tensor]):
        B = pyramid[0].shape[0]
        device, dtype = pyramid[0].device, pyramid[0].dtype
        G0 = pyramid[0].shape[-1]

        alive_hard = torch.ones(B, 1, G0, G0, device=device, dtype=dtype)
        alive_soft = torch.ones(B, 1, G0, G0, device=device, dtype=dtype)
        running_soft = torch.zeros(B, device=device, dtype=dtype)

        leaf_masks, split_probs, scaled_pyramid, leaf_scores = [], [], [], []

        # Feature chi tiết nhất (F56-equivalent) — dùng làm nhánh chủ đạo cho MỌI level.
        # Downsample (avg pool) về đúng grid của từng level để giữ đúng vai trò
        # "score phụ thuộc chủ yếu vào F_detail, bổ sung ngữ cảnh từ feature của level đó".
        detail_feat = pyramid[-1]

        for l in range(self.num_levels):
            Gl = pyramid[l].shape[-1]
            if Gl == detail_feat.shape[-1]:
                detail_l = detail_feat
            else:
                detail_l = F.adaptive_avg_pool2d(detail_feat, output_size=Gl)
            logits = self.split_heads[l](pyramid[l], detail_l)
            if self.depth_bias > 0:
                logits = logits - (l * self.depth_bias)
            if l >= self.max_depth:
                logits = logits - 1e4

            # 1 công thức xác suất DUY NHẤT dùng cho mọi nơi (hết lệch pha hard/soft/expected).
            prob = torch.sigmoid(logits / self.tau)

            if not torch.is_grad_enabled():
                split_hard = (prob > 0.5).float()          # eval: luôn rời rạc cứng, không nhiễu
            elif self.gumbel_hard:
                split_hard = gumbel_sigmoid(logits, self.tau)  # train Phase 2: rời rạc + straight-through
            else:
                split_hard = prob                            # train Phase 1: soft thuần, KHÔNG nhiễu Gumbel

            split_probs.append(prob * alive_hard)
            leaf_here_hard = alive_hard * (1.0 - split_hard)
            leaf_masks.append(leaf_here_hard)
            leaf_scores.append(prob)

            leaf_here_soft = alive_soft * (1.0 - prob)
            running_soft = running_soft + leaf_here_soft.flatten(1).sum(dim=1)

            detail_scale = 1.0 + self.detail_scale_weight * prob * alive_hard
            scaled_pyramid.append(scale_grad(pyramid[l], detail_scale))

            alive_hard = F.interpolate(alive_hard * split_hard, scale_factor=2, mode="nearest")
            alive_soft = F.interpolate(alive_soft * prob, scale_factor=2, mode="nearest")

        leaf_masks.append(alive_hard)
        scaled_pyramid.append(pyramid[-1])
        leaf_scores.append(torch.ones_like(alive_hard))  # level cuối luôn leaf, ưu tiên tối đa khi overflow
        expected_tokens = running_soft + alive_soft.flatten(1).sum(dim=1)

        return leaf_masks, split_probs, expected_tokens, scaled_pyramid, leaf_scores

    def _gather_tokens(self, pyramid, leaf_masks, leaf_scores):
        B, C = pyramid[0].shape[0], pyramid[0].shape[1]
        device, dtype = pyramid[0].device, pyramid[0].dtype

        feat_chunks, gate_chunks, geom_chunks, score_chunks = [], [], [], []
        for l, (feat_l, mask_l, score_l) in enumerate(zip(pyramid, leaf_masks, leaf_scores)):
            order = self._order(l).to(device)
            feat_chunks.append(feat_l.flatten(2).index_select(-1, order))
            gate_chunks.append(mask_l[:, 0].flatten(1).index_select(-1, order))
            score_chunks.append(score_l[:, 0].flatten(1).index_select(-1, order))
            geom_chunks.append(self._geom(l).to(device=device, dtype=dtype))

        feat_flat = torch.cat(feat_chunks, dim=-1)
        gate_flat = torch.cat(gate_chunks, dim=-1)
        split_score = torch.cat(score_chunks, dim=-1)
        geom_flat = torch.cat(geom_chunks, dim=0)

        feat_gated = feat_flat * gate_flat.unsqueeze(1)

        mask_bool = gate_flat > 0.5
        counts = mask_bool.sum(dim=1)
        n_max = max(min(int(counts.max().item()), self.max_tokens), 1)
        overflow = (counts - self.max_tokens).clamp(min=0)

        if self.token_select_mode == "importance":
            importance = gate_flat * split_score
            order_idx = torch.argsort(importance, dim=1, descending=True)[:, :n_max]
        else:
            sort_key = (~mask_bool).to(torch.uint8)
            order_idx = torch.argsort(sort_key, dim=1, stable=True)[:, :n_max]

        valid = torch.arange(n_max, device=device)[None, :] < torch.minimum(
            counts, torch.full_like(counts, self.max_tokens))[:, None]

        tok_feat = torch.gather(feat_gated, 2, order_idx.unsqueeze(1).expand(-1, C, -1)).transpose(1, 2)
        geom_batched = geom_flat.unsqueeze(0).expand(B, -1, -1)
        tok_geom = torch.gather(geom_batched, 1, order_idx.unsqueeze(-1).expand(-1, -1, 4))

        valid_f = valid.to(dtype)
        tok_feat = tok_feat * valid_f.unsqueeze(-1)
        tok_geom = tok_geom * valid_f.unsqueeze(-1)

        return tok_feat, tok_geom, valid_f, overflow

    def forward(self, feat_pyramid: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        pyramid = self._build_pyramid(feat_pyramid)
        leaf_masks, split_probs, expected_tokens, scaled_pyramid, leaf_scores = self._compute_leaves(pyramid)
        tok_feat, tok_geom, mask, overflow = self._gather_tokens(scaled_pyramid, leaf_masks, leaf_scores)

        tokens = self.token_proj(tok_feat) + self.pos_enc(tok_geom)

        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        mask = torch.cat([torch.ones(B, 1, device=mask.device, dtype=mask.dtype), mask], dim=1)

        avg_split_ratio = torch.cat([p.flatten(1) for p in split_probs], dim=1).mean()
        avg_num_tokens = mask.sum(dim=1).mean()

        if self.budget_loss_weight > 0:
            budget_loss = self.budget_loss_weight * F.relu(expected_tokens - self.target_tokens).pow(2).mean()
        else:
            budget_loss = torch.zeros((), device=tokens.device, dtype=tokens.dtype)

        if self.cost_lambda > 0:
            budget_loss = budget_loss + self.cost_lambda * expected_tokens.mean()

        aux = {
            "avg_split_ratio": avg_split_ratio,
            "avg_num_tokens": avg_num_tokens,
            "expected_tokens": expected_tokens.mean(),
            "budget_loss": budget_loss,
            "overflow_ratio": (overflow > 0).float().mean(),
        }
        return tokens, mask, aux