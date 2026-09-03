# RepShuffleFormer — Tài liệu Context Kiến trúc & Training (chi tiết, bám sát code)

> Tài liệu này trích xuất lại đầy đủ, chính xác toàn bộ 12 file trong project để làm nền tảng
> tham chiếu khi chỉnh sửa/hoàn thiện model. Không diễn giải thêm những gì code không thể hiện.

---

## 0. Sơ đồ pipeline tổng thể (model.py)

```
Input [B,3,H,W]
   │
   ▼
Stem (Conv3x3 s=stem_stride + BN + ReLU6)                         [backbone_rep_shuffle.py]
   │
   ▼
RepShuffleBackbone
   ├─ stage1 (RepShuffleStage, stride=2, ×(1+n1) block)  -> f1 (fine,  C=stem*2)
   ├─ stage2 (RepShuffleStage, stride=2, ×(1+n2) block)  -> f2 (mid,   C=stem*4)
   └─ stage3 (RepShuffleStage, stride=2, ×(1+n3) block)  -> f3 (coarse,C=stem*8)
   trả về [f1, f2, f3]  (fine -> coarse)
   │
   ▼
FPNLateralNeck: mỗi f_i -> Conv1x1+BN -> token_dim kênh, rồi ĐẢO thứ tự thành [coarse -> fine]
   │
   ▼
AdaptiveQuadtreeTokenizer: pyramid coarse->fine -> quadtree split thích ứng -> tokens [B,N,D] + mask + aux
   │
   ▼
LinearTransformer (kernel-based linear attention, depth blocks)
   │
   ▼
Mean-pool token hợp lệ (bỏ CLS) theo mask -> Linear head -> logits [B, num_classes]
```

`aux["budget_loss"]` được cộng vào loss tổng ở `loss.py`.

---

## 1. `backbone_rep_shuffle.py` — Backbone RepShuffle

### 1.1 Hàm tiện ích
- **`channel_shuffle(x, groups=2)`**: reshape `[B,C,H,W] -> [B,g,C/g,H,W]`, transpose(1,2), flatten lại về `[B,C,H,W]`. Trộn kênh giữa 2 nhánh CA/SA sau khi concat.
- **`_fuse_conv_bn(conv, bn)`**: fuse Conv+BN theo công thức chuẩn `w' = w * (γ/√(var+eps))`, `b' = β + (b_conv - running_mean) * (γ/√(var+eps))`. Dùng cho re-param khi deploy.

### 1.2 `LargeKernelBranch(dim, K∈{3,7}, stride)`
Nhánh xử lý không gian (SA) theo kiểu re-parameterizable multi-branch:

- `conv_nor`: depthwise Conv3x3 (stride=stride) + BN + ReLU6 — nén không gian ban đầu.
- `branches_config`:
  - Nếu `K=7`: `[(k=7,d=1), (k=3,d=2), (k=3,d=3)]`
  - Nếu `K=3`: `[(k=3,d=1), (k=1,d=1)]`
- `branches`: `ModuleList` các `Sequential(depthwise Conv(k,dilation=d) + BN)` tương ứng từng config, **không bias**.
- `spatial_fuse`: Conv1x1 + BN để trộn lại sau khi cộng các nhánh.
- **forward tách 2 bước** (quan trọng vì `Rep_Shuffle` cần chèn gate ECA ở giữa):
  - `forward_dilated(x)`: `sa = conv_nor(x)`; nếu đã deploy → `reparam_conv(sa)`; ngược lại → **tổng (sum)** output của tất cả các branch áp lên `sa`.
  - `forward_fuse(sa)`: `spatial_fuse(sa)`.
  - `forward(x) = forward_fuse(forward_dilated(x))` (dùng khi gọi độc lập, không qua Rep_Shuffle).
- **`_to_target_k(kernel, orig_k, d)`**: pad/dilate kernel gốc (k×k, dilation d) thành kernel `K×K` tương đương bằng cách rải giá trị theo bước `d`, canh giữa bằng `offset = (K - kd)//2` với `kd=(orig_k-1)*d+1`.
- **`switch_to_deploy()`**:
  1. Fuse từng branch (conv+bn) → quy đổi kernel về kích thước `K×K` qua `_to_target_k` → cộng dồn `W_equiv, B_equiv`.
  2. Tạo `reparam_conv` = 1 depthwise Conv `K×K` duy nhất mang `W_equiv/B_equiv`, xoá `self.branches`.
  3. Fuse `conv_nor` (Conv3x3+BN) → conv3x3 có bias, giữ nguyên ReLU6.
  4. Fuse `spatial_fuse` (Conv1x1+BN) → conv1x1 có bias.
  5. Đặt `_deployed=True` (idempotent — gọi lại không làm gì).

### 1.3 `Rep_Shuffle(dim, K, stride)` — khối chính (tương tự ShuffleNetV2 block + ECA cross-gate)
- `stride=1`: yêu cầu `dim` chẵn, `branch_dim = dim//2` (chia đôi kênh input, có residual).
- `stride=2`: `branch_dim = dim` (không chia đôi, không residual — nhân đôi kênh output khi kết hợp 2 nhánh cùng `dim`).
- **Nhánh CA (channel attention path)**: `ca_conv` (depthwise Conv3x3, stride=stride, `bd` kênh, không bias) + `ca_bn` (tách riêng để có thể fuse khi deploy qua cờ `_ca_deployed`).
- **Nhánh SA**: `sa_branch = LargeKernelBranch(bd, K, stride)`.
- **`eca = ECABlock(bd)`** — dùng để sinh gate sigmoid dùng chéo cho cả 2 nhánh (cross-gating), không dùng `forward()` mặc định của ECABlock mà dùng `get_gate()`.
- **`_run_branches(x_ca, x_sa)`** — điểm đặc trưng của kiến trúc:
  1. `ca_dw = _ca_dw(x_ca)` (depthwise conv, +BN nếu chưa deploy).
  2. `sigmoid_ca = eca.get_gate(ca_dw)` — tính gate kênh (B,C,1,1) từ CA sau depthwise, KHÔNG nhân trực tiếp ở đây.
  3. `ca_out = ca_dw * sigmoid_ca`.
  4. `sa_dilated = sa_branch.forward_dilated(x_sa)` (chưa qua spatial_fuse).
  5. `sa_out = sa_branch.forward_fuse(sa_dilated * sigmoid_ca)` — **gate ECA (tính từ nhánh CA) được nhân chéo vào nhánh SA trước khi fuse 1x1**. Đây là cơ chế "trao đổi chéo" (cross-branch gating) giữa 2 nhánh.
- **`forward(x)`**:
  - `stride=1`: `identity=x`; chunk theo kênh làm 2 (`x_ca, x_sa`); chạy `_run_branches`; `cat([ca,sa])`; `channel_shuffle(groups=2)`; **cộng residual** `+ identity`.
  - `stride=2`: cả `x_ca` và `x_sa` đều nhận **toàn bộ `x`** (không chunk); `cat([ca,sa])`; `channel_shuffle`; **không có residual** (vì shape thay đổi do downsample + tăng kênh gấp đôi).
- **`switch_to_deploy()`**: gọi `sa_branch.switch_to_deploy()`; fuse `ca_conv+ca_bn` → conv có bias, xoá `ca_bn`, đặt `_ca_deployed=True`.

### 1.4 `Stem(in_channels=3, out_channels=16, stride∈{1,2})`
- `Conv3x3(stride=stem_stride) + BN + ReLU6`, không bias.
- `switch_to_deploy()`: fuse Conv+BN → Conv có bias, giữ ReLU6.

### 1.5 `RepShuffleStage(in_channels, num_blocks_s1, K)`
- `downsample_block = Rep_Shuffle(dim=in_channels, K, stride=2)` → **nhân đôi channel**: `out_channels = in_channels*2`.
- `blocks_s1`: `num_blocks_s1` khối `Rep_Shuffle(dim=out_channels, K, stride=1)` nối tiếp (giữ nguyên channel, có residual).
- `forward`: downsample trước, rồi qua từng block s1 tuần tự.
- ⚠️ Lưu ý: mỗi stage thực chất có `1 (downsample, stride2) + num_blocks_s1 (stride1)` khối, tổng cộng.

### 1.6 `RepShuffleBackbone(...)`
Tham số nhận vào rất linh hoạt (`stage_repeats_s1` / `stage_depths` / `stage_repeats` — ưu tiên theo thứ tự đó; nếu None mặc định `(2,4,2)`; nếu độ dài 4 thì tự cắt bỏ phần tử đầu để còn 3, tức tương thích style YOLO `[stem, s1, s2, s3]`). Bắt buộc đúng 3 giá trị (assert).
- `use_stem=True` (mặc định) → `Stem(in_channels, stem_channels, stem_stride)`, `stage1_in = stem_channels`; nếu `False` thì bỏ qua stem, `stage1_in = in_channels`.
- 3 stage nối tiếp, mỗi stage **nhân đôi channel + giảm 1/2 spatial** (do `downsample_block` luôn stride=2):
  - `stage1 = RepShuffleStage(stage1_in, n1, K)` → `c1`
  - `stage2 = RepShuffleStage(c1, n2, K)` → `c2`
  - `stage3 = RepShuffleStage(c2, n3, K)` → `c3`
- `out_channels = c3`; `out_channels_list = [c1, c2, c3]` (fine → coarse).
- `forward(x)` trả về **list `[f1, f2, f3]`** (fine → coarse), nếu có stem thì áp dụng trước stage1.
- `switch_to_deploy()`: deploy stem (nếu có) rồi deploy toàn bộ module con kiểu `Rep_Shuffle` (duyệt `self.modules()`).

**Độ giảm kích thước không gian tổng cộng**: `stem_stride × 2(stage1) × 2(stage2) × 2(stage3)`. Với `stem_stride=1` (default trong `config.py`), tổng downsample = **8×**.

---

## 2. `eca_block.py` — Efficient Channel Attention

```python
k = |log2(dim) + b| / gamma   (gamma=2, b=1 mặc định)
k → ép lẻ, tối thiểu 3
```
- `avg_pool`: `AdaptiveAvgPool2d(1)`.
- `conv`: `Conv1d(1,1,kernel_size=k,padding=k//2,bias=False)` — tương tác cục bộ giữa các kênh lân cận (thay vì FC toàn cục như SE).
- **`get_gate(x)`**: `avg_pool → squeeze→transpose thành (B,1,C) → conv1d → transpose lại (B,C,1,1) → sigmoid`. **Không nhân vào x** — dùng để lấy gate tái sử dụng chéo (xem `Rep_Shuffle._run_branches`).
- **`forward(x)`**: hành vi ECA chuẩn = `x * gate.expand_as(x)` (dùng khi gọi module độc lập, không dùng trong `Rep_Shuffle`).

---

## 3. `neck.py` — FPN Lateral Neck

`FPNLateralNeck(in_channels_list, out_channels)`:
- Với mỗi mức channel đầu vào `c` trong `in_channels_list`: `lateral = Conv1x1(c→out_channels, bias=False) + BN`.
- `forward(feats_fine_to_coarse)`: assert số lượng khớp `len(laterals)`; áp lateral tương ứng từng feature; **đảo ngược danh sách** trước khi trả về → output là **coarse → fine**, tất cả cùng `out_channels` kênh (= `token_dim`).

Input backbone: `[f1,f2,f3]` (fine→coarse) → Neck output: `[proj(f3), proj(f2), proj(f1)]` (coarse→fine).

---

## 4. `tokenzin.py` — Adaptive Quadtree Tokenizer

### 4.1 Hàm phụ trợ
- **`gumbel_sigmoid(logits, tau, hard)`**: nếu không cần grad và không trong chế độ autograd → trả thẳng `(sigmoid>0.5).float()` (fast path eval). Ngược lại: ép fp32 (Gumbel noise không ổn định ở fp16), sinh 2 nhiễu Gumbel `g1,g2` từ `u~U(0,1)` clamp `[1e-6, 1-1e-6]`, `soft = sigmoid((logits+g1-g2)/tau)`; nếu `hard=True` → straight-through: `hard_val + soft - soft.detach()`. Trả về đúng dtype gốc.
- **`scale_grad(x, scale)`**: forward = `x*scale`, nhưng gradient chảy như thể chỉ nhân với `scale.detach()` cộng phần còn lại detach — kỹ thuật "scale forward, nhưng không lan truyền gradient hai lần qua scale" (straight-through scaling).
- **`_morton_order_index(grid, device)`**: nếu `grid` không phải luỹ thừa 2 → trả về thứ tự raster mặc định (`arange`). Nếu là luỹ thừa 2 → tính Z-order (Morton code) bit-interleave giữa row/col, trả về `argsort(code)` làm permutation.

### 4.2 `SplitHead(channels)`
`Conv1x1(c→c/2)+ReLU+Conv1x1(c/2→1)`; **bias lớp cuối khởi tạo = -2.0** (thiên về "không split" lúc đầu, giúp ổn định huấn luyện ban đầu — token count thấp).

### 4.3 `PositionSizeEncoder(token_dim)`
MLP: `Linear(4→token_dim/2)+ReLU+Linear(token_dim/2→token_dim)`. Input geometry 4 chiều: `(cx, cy, size, level_norm)`.

### 4.4 `AdaptiveQuadtreeTokenizer.__init__`
Tham số quan trọng:
- `stage_grid_sizes`: **bắt buộc ≥2 phần tử**, mỗi phần tử sau phải đúng **gấp đôi** phần tử trước (assert nghiêm ngặt) — đây là lưới thật từ backbone/neck (coarse→fine).
- `coarsest_grid` (root của quadtree) phải **chia hết** `stage_grid_sizes[0]` theo đúng luỹ thừa 2 → suy ra `num_extra_coarse_levels` (số lần avg-pool thêm để tạo root nhỏ hơn mức coarse nhất mà backbone có thật).
- `num_real_levels = len(stage_grid_sizes) - 1` (số lần split thật giữa các stage có feature thật).
- `num_levels = num_extra_coarse_levels + num_real_levels` — tổng số cấp có `SplitHead`.
- `max_tokens`: nếu không truyền, lấy theo `scale` từ `SCALE_PRESETS = {"S":64, "M":128, "L":256}`.
- `target_tokens = max(1.0, max_tokens * target_ratio)` (mặc định `target_ratio=0.8`).
- `max_depth`: giới hạn số cấp được phép split thật sự (mặc định = `num_levels`).
- Công tắc runtime (được `trainer.py` set theo epoch): `gumbel_hard`, `detail_scale_weight`, `budget_loss_weight` (qua thuộc tính, không qua `__init__` trực tiếp mỗi lần), `depth_bias`, `cost_lambda`, `token_select_mode`.
- Mỗi cấp `l = 0..num_levels`: precompute & **register buffer** `_geom_l{l}` (toạ độ tâm chuẩn hoá `cx,cy`, `size=1/Gl`, `level_norm=l/num_levels`, đã sắp theo `token_order`) và `_order_l{l}` (permutation index Morton/raster).
- `split_heads`: `ModuleList` gồm `num_levels` `SplitHead`, `token_proj: Linear(in_channels→token_dim)`, `pos_enc`, `cls_token` (Parameter `[1,1,token_dim]`, init `trunc_normal_(std=0.02)`).

### 4.5 `_build_pyramid(feat_pyramid)`
Input: `feat_pyramid` (coarse→fine, `num_real_levels+1` phần tử thật từ neck). Lấy `feat_pyramid[0]` (coarse nhất thật) làm gốc, `avg_pool2d(k=2,s=2)` lặp `num_extra_coarse_levels` lần để tạo thêm các mức "ảo" coarser hơn, **reverse** (coarsest trước), rồi nối `extra + feat_pyramid` → pyramid đầy đủ `num_levels+1` phần tử, coarse→fine.

### 4.6 `_compute_leaves(pyramid)` — thuật toán quadtree adaptive split
Khởi tạo `alive_hard = alive_soft = ones(B,1,G0,G0)` (mọi ô ở root đều "sống").
Với mỗi cấp `l = 0..num_levels-1`:
1. `logits = split_heads[l](pyramid[l])`.
2. Nếu `depth_bias>0`: phạt theo độ sâu `logits -= l*depth_bias` (Giai đoạn 2).
3. Nếu `l >= max_depth`: `logits -= 1e4` (khoá cứng không cho split quá độ sâu cho phép).
4. `split_soft_prob = sigmoid(logits)`; `split_hard = gumbel_sigmoid(logits, tau, hard=gumbel_hard)`.
5. Lưu `split_probs[l] = split_soft_prob * alive_hard` (chỉ tính ở ô còn sống) — dùng cho `avg_split_ratio` thống kê.
6. `leaf_here_hard = alive_hard * (1-split_hard)` — ô sống mà KHÔNG split → là leaf ở cấp này → lưu vào `leaf_masks[l]`.
7. `leaf_scores[l] = split_soft_prob` (điểm "độ chi tiết"/importance, dùng khi chọn token nếu overflow).
8. Tính `running_soft` (đếm số leaf kỳ vọng dạng soft, cộng dồn qua các cấp) để ước lượng `expected_tokens` (khả vi, dùng cho budget loss).
9. `detail_scale = 1 + detail_scale_weight * split_soft_prob * alive_hard` → **`scaled_pyramid[l] = scale_grad(pyramid[l], detail_scale)`** — vùng có xu hướng split được khuếch đại đặc trưng (chỉ ảnh hưởng gradient theo kiểu straight-through, không optimize được lặp lại).
10. Truyền "sống" xuống cấp con: `alive_hard = interpolate(alive_hard*split_hard, scale_factor=2, nearest)`, tương tự cho `alive_soft` với `split_soft_prob`.

Sau vòng lặp: cấp cuối cùng (`num_levels`, tức level fine nhất) **không có SplitHead riêng**, luôn là leaf toàn bộ (`leaf_masks.append(alive_hard)`), `scaled_pyramid.append(pyramid[-1])` (không scale), `leaf_scores.append(ones)` (điểm tối đa — luôn ưu tiên giữ khi overflow).
`expected_tokens = running_soft + alive_soft.flatten(1).sum(1)`.

### 4.7 `_gather_tokens(pyramid, leaf_masks, leaf_scores)`
- Với mỗi cấp `l`: flatten feature/mask/score theo permutation `_order(l)` (Morton hoặc raster), rồi **concat tất cả các cấp lại thành 1 chuỗi token dài** (`feat_flat`, `gate_flat`, `split_score`, `geom_flat`).
- `feat_gated = feat_flat * gate_flat` (chỉ giữ giá trị ở vị trí leaf thật).
- `mask_bool = gate_flat > 0.5`; `counts` = số leaf mỗi sample; `n_max = min(max(counts), max_tokens)` (ít nhất 1); `overflow = max(0, counts - max_tokens)`.
- **Chọn token khi vượt `max_tokens`** (2 chế độ, qua `token_select_mode`):
  - `"importance"` (mặc định): `importance = gate_flat * split_score`, lấy `argsort` giảm dần, cắt `n_max` đầu — **giữ token quan trọng nhất** khi overflow (không phải cứ theo thứ tự không gian).
  - `"order"`: sắp theo mask trước (stable sort), cắt `n_max` đầu — giữ theo thứ tự vị trí.
- `valid` mask: `arange(n_max) < min(counts, max_tokens)` — chuẩn hoá về cùng độ dài `n_max` cho cả batch, phần dư = 0 (padding).
- Gather feature (`tok_feat`) và geometry (`tok_geom`) theo `order_idx`, nhân với `valid` để zero-out phần padding.

### 4.8 `forward(feat_pyramid)`
1. `_build_pyramid` → `_compute_leaves` → `_gather_tokens`.
2. `tokens = token_proj(tok_feat) + pos_enc(tok_geom)`.
3. Thêm `cls_token` ở đầu chuỗi (`cat`), `mask` thêm cột `1` tương ứng CLS (luôn valid).
4. Thống kê aux: `avg_split_ratio` (trung bình xác suất split trên mọi cấp/vị trí sống), `avg_num_tokens` (trung bình `mask.sum(1)`), `expected_tokens` (trung bình theo batch).
5. `budget_loss`: nếu `budget_loss_weight>0` → `weight * relu(expected_tokens - target_tokens)^2 .mean()` (chỉ phạt khi VƯỢT target, không phạt khi thấp hơn); ngược lại = 0.
6. Nếu `cost_lambda>0`: cộng thêm `cost_lambda * expected_tokens.mean()` vào `budget_loss` (phạt tuyến tính liên tục theo chi phí — Giai đoạn 2).
7. `overflow_ratio`: tỉ lệ sample trong batch có `overflow>0` (bị cắt bớt token do vượt `max_tokens`).
8. Trả `(tokens, mask, aux)`.

---

## 5. `linear_transformer.py` — Linear Attention Transformer

### 5.1 `elu_feature_map(x) = ELU(x) + 1` — feature map dương hoá cho linear attention (kiểu "Transformers are RNNs").

### 5.2 `LinearMultiHeadAttention(dim, num_heads, dropout, eps=1e-5)`
- `q_proj,k_proj,v_proj,out_proj`: `Linear(dim,dim)`.
- **`q_norm, k_norm`: `LayerNorm(dim)` áp TRƯỚC khi reshape thành head** (trên toàn bộ D, không phải per-head) — ghi chú trong code là để nhẹ compute hơn so với norm per-head.
- `forward(x, mask)`:
  1. `q=q_norm(q_proj(x))`, `k=k_norm(k_proj(x))`, `v=v_proj(x)` — norm áp dụng ở không gian `[B,N,D]` full trước reshape.
  2. Reshape `[B,N,D] → [B,H,N,d]`.
  3. Ép **fp32** cho toàn bộ phần tính linear-attention kernel (tránh lỗi overflow/NaN dưới AMP fp16).
  4. `q=elu_feature_map(q)`, `k=elu_feature_map(k)`.
  5. Nếu có `mask` (padding mask `[B,N]`): nhân `k,v` với mask (loại bỏ đóng góp từ token invalid).
  6. **Kernel trick**: `kv = einsum("bhnd,bhne->bhde", k, v)` (kích thước `[B,H,d,d]`, không phụ thuộc N → độ phức tạp tuyến tính theo N thay vì bậc 2); `k_sum = k.sum(dim=2)`.
  7. `numerator = einsum("bhnd,bhde->bhne", q, kv)`; `denominator = einsum("bhnd,bhd->bhn", q, k_sum).clamp_min(eps)`.
  8. `out = numerator/denominator.unsqueeze(-1)`, **clamp** `[-60000, 60000]` (an toàn gần trần fp16 65504 trước khi cast lại dtype gốc).
  9. Reshape lại `[B,N,D]`, cast về dtype gốc; nếu có mask, nhân lại lần nữa để đảm bảo output token invalid = 0.
  10. `dropout(out_proj(out))`.

### 5.3 `MLP(dim, ratio=4.0, dropout)` — chuẩn `Linear→GELU→Dropout→Linear→Dropout`.

### 5.4 `LinearTransformerBlock(dim, num_heads, mlp_ratio, dropout=0.1)`
Pre-norm residual chuẩn: `x = x + attn(norm1(x), mask)`; `x = x + mlp(norm2(x))`.

### 5.5 `LinearTransformer(dim, depth, num_heads, mlp_ratio, dropout)`
`depth` block nối tiếp + `LayerNorm` cuối (`self.norm`).

---

## 6. `model.py` — `RepShuffleFormer` (lắp ráp toàn bộ)

### 6.1 Khởi tạo (`__init__`)
1. `backbone = RepShuffleBackbone(in_channels, stem_channels, stem_stride, use_stem=True, stage_repeats_s1=stage_repeats, K=K)` → lấy `c1,c2,c3 = backbone.out_channels_list`.
2. `neck = FPNLateralNeck([c1,c2,c3], token_dim)`.
3. **Auto-infer grid size từng stage** bằng cách chạy thử forward với input rỗng `zeros(1,in_channels,img_size,img_size)`:
   - Ép `self.eval()` trước (tránh nhiễu BatchNorm running stats), chạy `no_grad()`, sau đó **khôi phục lại đúng mode training ban đầu** (`self.train(was_training)`).
   - `feats = backbone(dummy)` → `projected = neck(feats)` (coarse→fine) → `stage_grid_sizes = [f.shape[-1] for f in projected]` (coarse→fine).
4. `tokenizer = AdaptiveQuadtreeTokenizer(in_channels=token_dim, token_dim, coarsest_grid, stage_grid_sizes, scale, token_order)`.
5. `transformer = LinearTransformer(token_dim, depth, num_heads, mlp_ratio, dropout)`.
6. `head = Linear(token_dim, num_classes)`.

### 6.2 `forward(x)`
```
feats = backbone(x)                      # [f1,f2,f3] fine->coarse
feat_pyramid = neck(feats)               # coarse->fine, token_dim kênh
tokens, mask, aux = tokenizer(feat_pyramid)
tokens = transformer(tokens, mask)
feat = weighted-mean-pool(tokens[:,1:], mask[:,1:])   # bỏ CLS, mean theo mask, clamp_min 1e-6 mẫu số
logits = head(feat)
return logits, aux
```
⚠️ **Lưu ý quan trọng**: `RepShuffleFormer` **KHÔNG dùng `cls_token` để phân loại** — dù tokenizer có thêm CLS token vào đầu chuỗi và transformer xử lý nó, `forward()` của model lại **bỏ CLS (`tokens[:,1:]`)** và lấy **mean-pooling có trọng số theo mask** trên các token còn lại làm feature cuối cùng cho head. CLS token chỉ tham gia self-attention (ảnh hưởng gián tiếp) chứ không được đọc ra trực tiếp.

### 6.3 `build_model(cfg)` — map từ `ModelConfig` sang tham số constructor `RepShuffleFormer` (bao gồm `getattr(cfg,"backbone_stem_stride",1)` và `getattr(cfg,"token_order","morton")` để tương thích ngược nếu config cũ thiếu field).

### 6.4 Khối `if __name__=="__main__"` — smoke test: dựng model `img_size=256, stem=16, stride=1, stage_repeats=(2,4,2), token_dim=128, depth=4, heads=4`; forward với batch=2; kiểm tra gradient chảy tới `stem`, và `ca_conv` của cả 3 stage (đảm bảo backbone không bị "chết" gradient ở tầng nào).

---

## 7. `config.py` — `ModelConfig`

### 7.1 Field chính (giá trị mặc định của dataclass)
| Nhóm | Field | Default |
|---|---|---|
| Input | `in_channels` | 3 |
| | `img_size` | 256 |
| | `num_classes` | 22 |
| | `batch_size` | 32 |
| | `label_smoothing` | 0.05 |
| | `budget_weight` | 0.05 |
| Backbone | `backbone_stem_channels` | 24 |
| | `backbone_stem_stride` | 1 |
| | `backbone_stage_repeats_s1` | (2,2,2) |
| | `K` | 3 |
| | `feature_grid_size` | None (không dùng trong code hiện tại — trường dự phòng) |
| Tokenizer | `coarsest_grid` | 4 |
| | `token_dim` | 128 |
| | `token_order` | "morton" |
| | `scale` | "M" |
| Transformer | `mixer_type` | "linear_attn" (chỉ 1 lựa chọn) |
| | `mixer_depth` | 4 |
| | `transformer_heads` | 4 |
| | `transformer_mlp_ratio` | 4.0 |
| | `mixer_dropout` | 0.1 |

### 7.2 `from_scale(scale, **overrides)` — 3 preset (override lên dataclass default, không override `K`, `backbone_stem_stride`, `img_size`... trừ khi truyền `**overrides`):

| Preset | stem_channels | stage_repeats_s1 | token_dim | mixer_depth | heads |
|---|---|---|---|---|---|
| `s` | 16 | (2,2,2) | 128 | 3 | 4 |
| `m` | 24 | (2,2,2) | 192 | 6 | 6 |
| `l` | 32 | (3,3,3) | 320 | 10 | 8 |

Gọi `cfg.validate()` ngay sau khi khởi tạo.

### 7.3 `validate()` — assert: `stem_channels` chẵn; `len(stage_repeats_s1)==3`; `stem_stride∈{1,2}`; `token_dim % transformer_heads==0`; `scale∈{S,M,L}`.

⚠️ Lưu ý: `trainer.py` gọi `ModelConfig.from_scale(args.scale, ...)` với `args.scale` **chữ thường** (`s/m/l` từ argparse choices) nhưng `from_scale` yêu cầu khớp key dict cũng chữ thường `presets = {"s":..,"m":..,"l":..}` — khớp đúng. Field `scale` bên trong preset lại lưu **chữ HOA** (`"S"/"M"/"L"`) để khớp `AdaptiveQuadtreeTokenizer.SCALE_PRESETS` và `ModelConfig.validate()`.

---

## 8. `dataset.py` — Data pipeline cho bộ CCMT (cây trồng, đa nhóm)

### 8.1 `DataConfig(yaml_path)`
Đọc `data.yaml`: lấy `path` (tuyệt đối hoặc tương đối so với thư mục chứa yaml) làm `root`; `nc`, `names` optional. Raise `FileNotFoundError` nếu root không tồn tại.

### 8.2 Cấu trúc dataset tự phát hiện (`_detect_structure`)
- **`flat`**: `root/<group>/<class>/*.jpg` — không có split sẵn.
- **`split`**: `root/<group>/(train_set|test_set)/<class>/*.jpg` — có sẵn train/test.
Phát hiện bằng cách xem group đầu tiên có subfolder cấp 2 (`has_deeper`) hay không.

### 8.3 Tên class cuối cùng = `"<group>_<class_không_hậu_tố_số>"` (hàm `_normalize_class_name` bỏ hậu tố số cuối, ví dụ `healthy1` → `healthy`, xử lý trường hợp CCMT đánh số trùng tên).

### 8.4 Luồng `get_dataloaders(cfg, data_yaml, batch_size, ...)`
- `set_seed(seed)` (mặc định 42) — set toàn bộ `PYTHONHASHSEED`, `random`, `numpy`, `torch` (cả CUDA), **`cudnn.deterministic=True`, `cudnn.benchmark=False`** (ở bước chuẩn bị data; `trainer.main()` sau đó set lại `cudnn.benchmark=True` nếu có CUDA — xem mục 10).
- Nếu `struct=="flat"`: scan toàn bộ ảnh (`_scan_flat_ccmt`) → **stratified split 3 phần theo tỉ lệ mỗi class**: mặc định `val_ratio=0.15, test_ratio=0.20` → train ≈70%.
- Nếu `struct=="split"`: tìm folder train (`_find_split_folder(prefer="train")`) và test (`prefer="test"`) qua regex tìm trong tên; scan riêng train/test; **class_names = hợp của cả 2**; val tách `val_ratio` từ train (stratified 2 phần).
- Cảnh báo (không raise) nếu `dcfg.nc` (trong yaml) khác số class thật scan được, trừ khi `strict_num_classes=True` thì raise.
- Transform:
  - **Train**: `RandomResizedCrop(img_size, scale=(0.7,1.0))`, `RandomHorizontalFlip(0.5)`, `RandomVerticalFlip(0.3)`, `ColorJitter(brightness=0.3,contrast=0.3,saturation=0.2,hue=0.05)`, `ToTensor`, `Normalize(ImageNet mean/std)`.
  - **Val/Test**: `Resize((img_size,img_size))`, `ToTensor`, `Normalize` (cùng mean/std ImageNet).
- **`ImageDataset`**: load ảnh qua PIL, dùng `img.draft("RGB", (pre_resize,pre_resize))` để giảm decode cost với JPEG lớn; ép `RGB`; nếu ảnh không vuông → resize theo cạnh nhỏ nhất rồi **center-crop** về `pre_resize` (mặc định `= round(target_size*1.15)`); nếu đã vuông nhưng khác `pre_resize` thì resize thẳng. **Retry tối đa 5 ảnh kế tiếp** (offset 0..4, có wrap) nếu load lỗi; nếu vẫn lỗi hết thì trả `zeros(3,target_size,target_size)` kèm label gốc (không raise, tránh crash cả batch).
- **Xử lý mất cân bằng lớp** (`handle_imbalance=True`): tính tỉ lệ `max_count/min_count`; nếu `>1.5` → dùng `WeightedRandomSampler` (weight = nghịch đảo tần suất class) cho **train_loader** (val/test không dùng sampler, luôn `shuffle=False`).
- `train_loader`: `drop_last=True`; nếu có sampler thì `shuffle=False` (bắt buộc, vì Sampler tự lo random) ngược lại `shuffle=True`.
- `val_loader`/`test_loader`: `batch_size*2` (đánh giá không cần backward, tăng batch cho nhanh), `eval_workers = min(2, num_workers)`, `persistent_workers=False`.
- In summary chi tiết (`_print_summary`) gồm số ảnh mỗi split, danh sách class có index.
- Trả về `(train_loader, val_loader, test_loader, class_names)` — `test_loader` có thể `None` nếu không có test set (trường hợp `flat` nhưng `test_ratio` quá nhỏ dẫn tới rỗng thì vẫn có vì stratified luôn lấy ít nhất theo tỉ lệ; chỉ `None` khi `test_s` rỗng thực sự).

---

## 9. `loss.py` — `VisionEyeLoss` / `compute_loss`

- `ce = CrossEntropyLoss(label_smoothing=cfg.label_smoothing)`.
- `task_loss = ce(logits, targets)`.
- `budget_loss` lấy từ `aux["budget_loss"]` nếu có, else 0; **clamp `[0, 10]`** để tránh nổ loss.
- `weight = budget_weight` (tham số truyền vào override) hoặc `self.budget_weight` (từ config) nếu không truyền.
- `total_loss = task_loss + weight * budget_loss`.
- `stats` trả về (đều `.detach()`, KHÔNG `.item()` — để người gọi tự quyết định khi nào sync GPU→CPU): `total_loss, task_loss, budget_loss, budget_weight`, cộng thêm nếu có trong `aux`: `expected_tokens, avg_num_tokens, overflow_ratio`.
- `compute_loss(logits, targets, aux, cfg, budget_weight)`: hàm tiện ích build `VisionEyeLoss` mỗi lần gọi (đọc `label_smoothing`/`budget_weight` mặc định từ `cfg` nếu có, else 0.05/0.05) rồi forward.

---

## 10. `trainer.py` — Vòng lặp huấn luyện 2 giai đoạn (Phase 1 soft → Phase 2 hard)

### 10.1 CLI args chính (`parse_args`)
`--data` (bắt buộc), `--scale {s,m,l}` (default m), `--batch-size 32`, `--epochs 200`, `--lr 3e-4`, `--warmup-epochs 3`, `--phase1-epochs 0` (0 = tự động = 35% tổng epoch), `--transition-frac 0.5` (độ dài vùng chuyển tiếp tính theo tỉ lệ `phase1_epochs`), `--phase2-detail-scale-weight 0.1`, `--phase2-budget-weight 1.0` (giá trị budget weight MỤC TIÊU cuối cùng, đạt dần qua schedule chứ không áp ngay), `--phase2-depth-bias 0.0`, `--phase2-cost-lambda 0.0`, `--label-smoothing 0.05`, `--budget-weight 0.05`, `--ema-decay 0.999` (0 = tắt EMA), `--val-loss-cap None`, `--project runs/train`, `--name exp`.

### 10.2 Lịch trình theo epoch

**`anneal_tau(epoch, total_epochs, tau_start=2.0, tau_end=0.3)`**: cosine decay (không phải linear) — `frac = min(1, epoch/(total_epochs-1))`, `cos_frac = 0.5*(1-cos(π*frac))`, `tau = tau_start + cos_frac*(tau_end-tau_start)`. Giảm chậm đầu, nhanh giữa, mượt cuối.

**`lr_at_epoch(epoch, base_lr, warmup_epochs)`**: linear warmup `base_lr*(epoch+1)/warmup_epochs` nếu `epoch<warmup_epochs`, else `None` (để scheduler cosine tự quản).

**`budget_weight_schedule(epoch, total_epochs, phase1_epochs, target_weight)`**:
```
epoch < phase1_epochs              -> 0.0
frac < 0.30 (frac=epoch/(total-1)) -> 0.0
0.30 <= frac < 0.60                -> 0.1
0.60 <= frac < 0.80                -> 0.3
frac >= 0.80                       -> target_weight (args.phase2_budget_weight)
```
Lưu ý: điều kiện `frac<0.30` dùng `epoch/(total_epochs-1)` **tính theo TOÀN BỘ epoch**, không phải riêng phần sau `phase1_epochs` — nên nếu `phase1_epochs` đã vượt qua mốc 30% thì budget vẫn có thể là 0.0 cho tới khi đồng thời qua cả `phase1_epochs` VÀ `frac>=0.30`.

**`resolve_gumbel_hard(epoch, phase1_epochs, transition_epochs)`**:
- `epoch < phase1_epochs` → `False` (soft, Phase 1).
- `transition_epochs<=0` hoặc `epoch >= phase1_epochs+transition_epochs` → `True` (hard hẳn, Phase 2).
- Trong vùng transition: `progress=(epoch-phase1_epochs)/transition_epochs`, xác suất chọn hard = `progress`, **random mỗi epoch** `np.random.rand() < hard_prob` (không phải mỗi batch — quyết định 1 lần/epoch, cố định cho cả epoch đó).

**`update_phase(model, epoch, total_epochs, phase1_epochs, transition_epochs, args)`**:
- Tính `tau, gumbel_hard, budget_w` như trên.
- `tok = model.tokenizer`; set `tok.tau = tau`.
- `in_phase2_effects = epoch >= phase1_epochs`.
- Set các thuộc tính tokenizer qua `setattr` (chỉ nếu `hasattr`): `gumbel_hard`, `detail_scale_weight` (= `args.phase2_detail_scale_weight` nếu đã sang phase2 effects, else `0.0`), `budget_loss_weight = budget_w`, `depth_bias` (= `args.phase2_depth_bias` nếu phase2, else 0), `cost_lambda` (tương tự).
- Trả `(tau, gumbel_hard, budget_w)` để log.

### 10.3 `Trainer` class
- `phase1_epochs = args.phase1_epochs if >0 else max(1, int(0.35*epochs))`.
- `transition_epochs = max(0, int(phase1_epochs * transition_frac))`.
- `optimizer`: `AdamW` qua `build_optimizer_with_decay` — **tách 2 nhóm param**: `no_decay` (weight_decay=0) gồm những param `ndim<=1` HOẶC tên chứa `"bias"`/`"norm"`/`"bn"` (không phân biệt hoa thường); còn lại vào `decay` (weight_decay=5e-2 mặc định). `lr=args.lr`.
- `scaler = GradScaler("cuda", enabled=use_amp)` — `use_amp = (device=="cuda")`.
- `scheduler = CosineAnnealingLR(T_max=max(1, epochs-warmup_epochs), eta_min=1e-6)`.
- `EMA` (nếu `ema_decay>0`): giữ bản sao float của mọi tensor floating-point trong `state_dict`, update mỗi bước train bằng `shadow = shadow*decay + param*(1-decay)`; `copy_to(model)` ghi đè state_dict model bằng shadow (dùng khi eval/lưu best).

### 10.4 `train_one_epoch`
Với mỗi batch: `autocast("cuda", dtype=fp16, enabled=use_amp)` → `logits, aux = model(x)` → `loss = compute_loss(...)` → `scaler.scale(loss).backward()` → `scaler.unscale_` → `clip_grad_norm_(max_norm=1.0)` → `scaler.step` → `scaler.update()` → cập nhật EMA (nếu có). Bỏ qua loss NaN/Inf khi cộng dồn `running_loss` (không raise, chỉ skip cộng). Thu thập thống kê tokenizer qua `_extract_tokenizer_stats` — **CHÚ Ý**: hàm này tìm các key `("avg_tokens","num_tokens","token_count","split_ratio","budget_usage")` trong `aux`, nhưng `aux` thực tế từ `AdaptiveQuadtreeTokenizer.forward` chỉ có `("avg_split_ratio","avg_num_tokens","expected_tokens","budget_loss","overflow_ratio")` — **tên key KHÔNG khớp** (`avg_tokens` vs `avg_num_tokens`, `split_ratio` vs `avg_split_ratio`), nên `_extract_tokenizer_stats` hiện tại **luôn trả `{}`** (không log được token stats trong training log — điểm cần lưu ý nếu muốn sửa).

### 10.5 `evaluate` / `evaluate_with_ema`
`evaluate`: `autocast(enabled=False)` (fp32 full), tính loss + accuracy (`argmax==y`). `evaluate_with_ema`: backup state_dict hiện tại, copy EMA weight vào model, evaluate, rồi khôi phục lại state_dict backup (đảm bảo train tiếp tục dùng trọng số thật, không bị lẫn EMA).

### 10.6 `fit()` — vòng lặp epoch
1. Set warmup LR nếu còn trong giai đoạn warmup (ghi đè trực tiếp `pg["lr"]`), else để `scheduler.step()` xử lý sau khi train xong epoch đó.
2. `update_phase(...)` → train 1 epoch → nếu không warmup thì `scheduler.step()`.
3. `evaluate_with_ema(val_loader, budget_weight=budget_w)`.
4. Lưu lịch sử (`train_loss, val_loss, val_acc`, và mọi `tok_*` nếu có).
5. Vẽ lại `results.png` mỗi epoch (`plot_training_metrics`), lưu `last.pt` mỗi epoch (state_dict thật, không phải EMA).
6. `is_better(best_acc, best_loss, val_acc, val_loss, val_loss_cap)`: false nếu `val_loss` NaN hoặc `>= val_loss_cap` (nếu set); true nếu `val_acc` tốt hơn thực sự, hoặc bằng nhau nhưng `val_loss` thấp hơn.
7. Nếu tốt hơn: cập nhật best; nếu có EMA thì lưu **EMA weight** làm `best.pt` (backup/restore state thật sau khi lưu, không làm hỏng quá trình train tiếp); lưu `confusion_matrix.png`.
8. Sau toàn bộ epoch: `_test_and_summarize()` — load lại `best.pt`, evaluate trên `test_loader`, ghi `summary.txt` (kiến trúc, scale, batch size, phase epochs, EMA decay, best val acc/loss, test acc/loss).

### 10.7 `main()`
`set_seed(42)` (cố định, không phụ thuộc arg) → tạo `save_dir` tăng dần số nếu trùng tên (`exp, exp2, exp3,...`) → build `cfg = ModelConfig.from_scale(args.scale, batch_size, label_smoothing, budget_weight)` → nếu CUDA: `cudnn.benchmark=True` (ghi đè lại `False` mà `set_seed` trong `dataset.py` đã set trước đó, vì `set_seed` được gọi lại bên trong `get_dataloaders`) → `get_dataloaders(cfg, args.data, ...)` (`num_workers=4`, `strict_num_classes=False`) → nếu `len(class_names) != cfg.num_classes` thì **tự động cập nhật `cfg.num_classes`** theo số class scan thực tế → `build_model(cfg)` → `Trainer(...).fit()`.

---

## 11. `visual.py` — Trực quan hoá

- **`plot_training_metrics(history, save_path)`**: 2 subplot side-by-side — (trái) Train Loss vs Val Loss theo epoch; (phải) Val Accuracy theo epoch. Lưu PNG dpi=300.
- **`plot_and_save_confusion_matrix(model, loader, device, num_classes, class_names, save_path)`**: chạy inference toàn bộ `loader` (`autocast(enabled=False)`, fp32), tính confusion matrix (`sklearn`), chuẩn hoá theo hàng (`cm_norm`), annotate mỗi ô `"count\n(percent%)"` (chỉ annotate nếu `num_classes<=25`, tránh rối hình khi quá nhiều class), heatmap `seaborn` cmap Blues, `figsize` scale theo `num_classes` (tối thiểu 10). Lưu PNG dpi=300.

---

## 12. Tổng hợp các điểm liên kết giữa các file (data flow tham số)

- `data.yaml.path` → `dataset.DataConfig.root` → `dataset.get_dataloaders` → `class_names` → `trainer.main` gán lại `cfg.num_classes`.
- `config.ModelConfig.from_scale(scale)` → `backbone_stem_channels, backbone_stage_repeats_s1, K, token_dim, mixer_depth, transformer_heads, scale` → `model.build_model(cfg)` → `RepShuffleFormer(...)`.
- `model.RepShuffleFormer.__init__` tự suy `stage_grid_sizes` bằng forward thử (không cần khai báo tay trong config) → truyền vào `AdaptiveQuadtreeTokenizer`.
- `trainer.update_phase` ghi trực tiếp vào thuộc tính runtime của `model.tokenizer` (`tau, gumbel_hard, detail_scale_weight, budget_loss_weight, depth_bias, cost_lambda`) — đây là cách duy nhất kiểm soát hành vi 2-giai-đoạn của tokenizer trong lúc train (không qua lại `__init__`).
- `aux["budget_loss"]` (từ tokenizer) → `loss.compute_loss` cộng vào `total_loss` với trọng số `budget_w` (schedule theo epoch, không phải hằng số cố định từ config sau epoch đầu).

---

## 13. Danh sách điểm cần lưu ý / khả năng rủi ro khi sửa (bám sát đúng những gì code thể hiện, không suy diễn thêm)

1. **`_extract_tokenizer_stats` (trainer.py) tra sai tên key** so với `aux` thật trả về từ `AdaptiveQuadtreeTokenizer.forward` (`tokenzin.py`) → thống kê token trong log training hiện luôn rỗng.
2. `RepShuffleBackbone.__init__` chấp nhận `stage_repeats` độ dài 4 và tự cắt phần tử đầu — cần cẩn thận nếu truyền nhầm định dạng `[n0,n1,n2,n3]` kiểu YOLO, `n0` sẽ bị bỏ hoàn toàn không cảnh báo.
3. `budget_weight_schedule` tính `frac` theo **tổng `epochs`**, không theo phần còn lại sau `phase1_epochs` — nếu `phase1_epochs` lớn (vd 40% total) có thể khiến budget vẫn ở mức 0.1/0.3 khá lâu dù đã ở Phase 2 hard.
4. `resolve_gumbel_hard` quyết định hard/soft **theo epoch** (không theo batch) trong vùng transition — nghĩa là cả epoch đó dùng đồng nhất 1 kiểu (không trộn hard/soft trong cùng epoch).
5. `Rep_Shuffle.switch_to_deploy()` / `LargeKernelBranch.switch_to_deploy()` là **one-way, không thể đảo ngược lại chế độ train được** (đã xoá hẳn `self.branches`/`self.ca_bn`) — chỉ dùng cho inference cuối cùng.
6. `model.py.__init__` dùng 1 lần forward "dò shape" với `torch.zeros` — nếu sau này `Stem`/`Backbone`/`Neck` thay đổi cách tính kích thước output phụ thuộc giá trị input (không chỉ shape) thì bước dò này vẫn an toàn vì chỉ lấy `.shape`, không lấy giá trị.
7. `ImageDataset.__getitem__` khi lỗi hết 5 lần trả `torch.zeros(...)` **kèm label gốc** (không phải ảnh dummy gắn nhãn khác) — nếu file lỗi nhiều, có thể làm nhiễu nhẹ nhãn đó bằng ảnh đen tuyền, cần theo dõi `error_summary()`.

---

*(Tài liệu tổng hợp trực tiếp từ 12 file mã nguồn trong project: backbone_rep_shuffle.py, eca_block.py, neck.py, tokenzin.py, linear_transformer.py, model.py, config.py, dataset.py, loss.py, trainer.py, visual.py, data.yaml — không thêm giả định ngoài những gì code thể hiện.)*
