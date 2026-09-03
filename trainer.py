from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import ModelConfig
from dataset import get_dataloaders, set_seed
from loss import compute_loss
from model import build_model
from visual import plot_and_save_confusion_matrix, plot_training_metrics


# ------------------------------------------------------------------ #
# Utils
# ------------------------------------------------------------------ #

def get_save_dir(base_dir: str = "runs/train", name: str = "exp") -> str:
    os.makedirs(base_dir, exist_ok=True)
    target_dir = os.path.join(base_dir, name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    i = 2
    while True:
        next_dir = os.path.join(base_dir, f"{name}{i}")
        if not os.path.exists(next_dir):
            os.makedirs(next_dir, exist_ok=True)
            return next_dir
        i += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện RepShuffleFormer")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn tới file data.yaml")
    parser.add_argument("--scale", type=str, default="m", choices=["s", "m", "l"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--phase1-epochs", type=int, default=0,
                         help="Số epoch Giai đoạn 1 (soft gumbel). 0 = tự động dùng 35%% tổng epoch.")
    parser.add_argument("--transition-frac", type=float, default=0.5,
                         help="Độ dài vùng chuyển tiếp soft->hard, tính theo tỉ lệ của phase1-epochs. "
                              "0 = tắt transition, chuyển hard ngay khi hết phase1 (không khuyến khích).")
    parser.add_argument("--phase2-detail-scale-weight", type=float, default=0.1)
    parser.add_argument("--phase2-budget-weight", type=float, default=1.0,
                         help="Giá trị budget_loss_weight MỤC TIÊU (đạt được ở cuối training). "
                              "Không áp dụng ngay lập tức — xem budget_weight_schedule().")
    parser.add_argument("--phase2-depth-bias", type=float, default=0.0)
    parser.add_argument("--phase2-cost-lambda", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--budget-weight", type=float, default=0.05)
    parser.add_argument("--ema-decay", type=float, default=0.999,
                         help="Hệ số EMA cho model weights. 0 = tắt EMA.")
    parser.add_argument("--val-loss-cap", type=float, default=None,
                         help="Bỏ qua checkpoint có val_loss >= cap. Mặc định: không giới hạn.")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="exp")
    return parser.parse_args()


def anneal_tau(epoch: int, total_epochs: int, tau_start: float = 2.0, tau_end: float = 0.3) -> float:
    """Cosine decay thay vì linear: giảm chậm lúc đầu, nhanh ở giữa, mượt lúc cuối
    -> tránh 'giảm quá nhanh ở đầu, quá chậm ở cuối' như linear."""
    frac = min(1.0, epoch / max(1, total_epochs - 1))
    cos_frac = 0.5 * (1.0 - math.cos(math.pi * frac))
    return tau_start + cos_frac * (tau_end - tau_start)


def lr_at_epoch(epoch: int, base_lr: float, warmup_epochs: int) -> float | None:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)
    return None


def budget_weight_schedule(epoch: int, total_epochs: int, phase1_epochs: int, target_weight: float) -> float:
    if epoch < phase1_epochs:
        return 0.0
    frac = epoch / max(1, total_epochs - 1)
    if frac < 0.30:
        return 0.0
    elif frac < 0.60:
        return 0.1
    elif frac < 0.80:
        return 0.3
    return target_weight


def resolve_gumbel_hard(epoch: int, phase1_epochs: int, transition_epochs: int) -> bool:
    if epoch < phase1_epochs:
        return False
    if transition_epochs <= 0 or epoch >= phase1_epochs + transition_epochs:
        return True
    progress = (epoch - phase1_epochs) / float(transition_epochs)
    hard_prob = min(1.0, max(0.0, progress))
    return bool(np.random.rand() < hard_prob)


def update_phase(model: nn.Module, epoch: int, total_epochs: int, phase1_epochs: int,
                  transition_epochs: int, args) -> tuple[float, bool, float]:
  
    tau = anneal_tau(epoch, total_epochs)
    gumbel_hard = resolve_gumbel_hard(epoch, phase1_epochs, transition_epochs)
    budget_w = budget_weight_schedule(epoch, total_epochs, phase1_epochs, args.phase2_budget_weight)

    tok = getattr(model, "tokenizer", None)
    if tok is None:
        return tau, gumbel_hard, budget_w

    if hasattr(tok, "tau"):
        tok.tau = tau

    in_phase2_effects = epoch >= phase1_epochs
    switches = dict(
        gumbel_hard=gumbel_hard,
        detail_scale_weight=args.phase2_detail_scale_weight if in_phase2_effects else 0.0,
        budget_loss_weight=budget_w,
        depth_bias=args.phase2_depth_bias if in_phase2_effects else 0.0,
        cost_lambda=args.phase2_cost_lambda if in_phase2_effects else 0.0,
    )

    for attr, val in switches.items():
        if hasattr(tok, attr):
            setattr(tok, attr, val)

    return tau, gumbel_hard, budget_w


def is_better(best_acc: float, best_loss: float, val_acc: float, val_loss: float,
              val_loss_cap: float | None = None) -> bool:
    if np.isnan(val_loss) or (val_loss_cap is not None and val_loss >= val_loss_cap):
        return False
    if val_acc > best_acc:
        return True
    return val_acc == best_acc and val_loss < best_loss


def build_optimizer_with_decay(model: nn.Module, lr: float = 3e-4, weight_decay: float = 5e-2) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        target = no_decay if (p.ndim <= 1 or "bias" in name or "norm" in name.lower() or "bn" in name.lower()) else decay
        target.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=lr
    )


def _extract_tokenizer_stats(aux) -> dict:
    if not isinstance(aux, dict):
        return {}
    keys = ("avg_split_ratio", "avg_num_tokens", "expected_tokens", "overflow_ratio")
    out = {}
    for k in keys:
        if k in aux:
            v = aux[k]
            v = v.item() if torch.is_tensor(v) else float(v)
            out[k] = v
    return out


class EMA:

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.items():
            msd[k].copy_(v.to(msd[k].dtype))


# ------------------------------------------------------------------ #
# Trainer
# ------------------------------------------------------------------ #

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, class_names, cfg, args, device, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.cfg = cfg
        self.args = args
        self.device = device
        self.save_dir = save_dir
        self.weights_dir = os.path.join(save_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)

        self.use_amp = device == "cuda"
        self.optimizer = build_optimizer_with_decay(self.model, lr=args.lr, weight_decay=5e-2)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.warmup_epochs = max(0, args.warmup_epochs)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, args.epochs - self.warmup_epochs), eta_min=1e-6
        )

        # Phase1 dài hơn (30-40% thay vì 15%) để tokenizer học xong trước khi bị ép hard decision.
        self.phase1_epochs = args.phase1_epochs if args.phase1_epochs > 0 else max(1, int(0.35 * args.epochs))
        self.transition_epochs = max(0, int(self.phase1_epochs * args.transition_frac))

        # EMA: tắt nếu ema-decay <= 0.
        self.ema = EMA(self.model, decay=args.ema_decay) if args.ema_decay and args.ema_decay > 0 else None

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.best_ckpt = os.path.join(self.weights_dir, "best.pt")
        self.last_ckpt = os.path.join(self.weights_dir, "last.pt")

    def train_one_epoch(self, epoch: int, tau: float, gumbel_hard: bool, budget_w: float) -> tuple[float, dict]:
        self.model.train()
        running_loss = 0.0
        tok_stats_sum: dict = {}
        tok_stats_count = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1:02d}/{self.args.epochs}")

        for x, y in pbar:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=torch.float16):
                logits, aux = self.model(x)
                loss, _ = compute_loss(logits, y, aux, self.cfg, budget_weight=budget_w)

            if not torch.isfinite(loss):
                print(f"[trainer] Bỏ qua batch {loss.item()} không hữu hạn ở epoch {epoch + 1}.")
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                print(f"[trainer] Bỏ qua gradient không hữu hạn ở epoch {epoch + 1}.")
                continue
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model)

            loss_val = loss.item()
            if not (np.isnan(loss_val) or np.isinf(loss_val)):
                running_loss += loss_val * x.size(0)

            stats = _extract_tokenizer_stats(aux)
            if stats:
                for k, v in stats.items():
                    tok_stats_sum[k] = tok_stats_sum.get(k, 0.0) + v * x.size(0)
                tok_stats_count += x.size(0)

            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "scale": f"{self.scaler.get_scale():.0f}",
                "tau": f"{tau:.2f}",
                "hard": "1" if gumbel_hard else "0",
                "budget_w": f"{budget_w:.2f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

        tok_stats_avg = {k: v / tok_stats_count for k, v in tok_stats_sum.items()} if tok_stats_count else {}
        return running_loss / len(self.train_loader.dataset), tok_stats_avg

    @torch.no_grad()
    def evaluate(self, loader, budget_weight: float | None = None) -> tuple[float, float]:
        self.model.eval()
        total_loss, valid_total, correct, total = 0.0, 0, 0, 0
        weight = self.cfg.budget_weight if budget_weight is None else budget_weight
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=False):
                logits, aux = self.model(x.float())
                loss, _ = compute_loss(logits, y, aux, self.cfg, budget_weight=weight)
            loss_val = loss.item()
            if np.isfinite(loss_val):
                total_loss += loss_val * x.size(0)
                valid_total += x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
        val_loss = total_loss / valid_total if valid_total else float("nan")
        return val_loss, correct / max(1, total)

    @torch.no_grad()
    def evaluate_with_ema(self, loader, budget_weight: float | None = None) -> tuple[float, float]:
       
        if self.ema is None:
            return self.evaluate(loader, budget_weight=budget_weight)

        backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.ema.copy_to(self.model)
        try:
            loss, acc = self.evaluate(loader, budget_weight=budget_weight)
        finally:
            self.model.load_state_dict(backup)
        return loss, acc

    def fit(self) -> None:
        print(f"[trainer] Phase 1 (soft): epoch 0..{self.phase1_epochs - 1} | "
              f"Transition (soft->hard): {self.phase1_epochs}..{self.phase1_epochs + self.transition_epochs - 1} | "
              f"Phase 2 (hard): epoch {self.phase1_epochs + self.transition_epochs}..{self.args.epochs - 1} | "
              f"Warm-up LR: {self.warmup_epochs} epoch(s) | EMA: {'on (decay=' + str(self.args.ema_decay) + ')' if self.ema else 'off'}")

        for epoch in range(self.args.epochs):
            t0 = time.time()

            warmup_lr = lr_at_epoch(epoch, self.args.lr, self.warmup_epochs)
            if warmup_lr is not None:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr

            tau, gumbel_hard, budget_w = update_phase(
                self.model, epoch, self.args.epochs, self.phase1_epochs, self.transition_epochs, self.args
            )

            train_loss, tok_stats = self.train_one_epoch(epoch, tau, gumbel_hard, budget_w)
            if warmup_lr is None:
                self.scheduler.step()

            val_loss, val_acc = self.evaluate_with_ema(self.val_loader, budget_weight=budget_w)
            epoch_time = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            for k, v in tok_stats.items():
                self.history.setdefault(f"tok_{k}", []).append(v)

            tok_stats_str = " | ".join(f"{k}: {v:.3f}" for k, v in tok_stats.items())
            print(
                f"[Epoch {epoch + 1:02d}/{self.args.epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (EMA) | "
                f"Val Acc: {val_acc:.4f} (Best: {self.best_val_acc:.4f}) | "
                f"tau: {tau:.3f} | hard: {int(gumbel_hard)} | budget_w: {budget_w:.2f} | "
                f"time/epoch: {epoch_time:.1f}s" + (f" | {tok_stats_str}" if tok_stats_str else "")
            )

            plot_training_metrics(self.history, save_path=os.path.join(self.save_dir, "results.png"))
            torch.save(self.model.state_dict(), self.last_ckpt)

            if is_better(self.best_val_acc, self.best_val_loss, val_acc, val_loss, self.args.val_loss_cap):
                self.best_val_acc, self.best_val_loss = val_acc, val_loss
                if self.ema is not None:
                    backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    self.ema.copy_to(self.model)
                    try:
                        torch.save(self.model.state_dict(), self.best_ckpt)
                        print(f"--> [Checkpoint] BEST mới: {self.best_ckpt} (Acc: {val_acc:.4f})")
                        plot_and_save_confusion_matrix(
                            self.model, self.val_loader, self.device, len(self.class_names),
                            class_names=self.class_names,
                            save_path=os.path.join(self.save_dir, "confusion_matrix.png"),
                        )
                    finally:
                        self.model.load_state_dict(backup)
                else:
                    torch.save(self.model.state_dict(), self.best_ckpt)
                    print(f"--> [Checkpoint] BEST mới: {self.best_ckpt} (Acc: {val_acc:.4f})")
                    plot_and_save_confusion_matrix(
                        self.model, self.val_loader, self.device, len(self.class_names),
                        class_names=self.class_names,
                        save_path=os.path.join(self.save_dir, "confusion_matrix.png"),
                    )

        self._test_and_summarize()

    def _test_and_summarize(self) -> None:
        print(f"\n--- Hoàn tất Huấn luyện. Đánh giá test bằng Best Model ({self.best_ckpt}) ---")
        if os.path.exists(self.best_ckpt):
            self.model.load_state_dict(torch.load(self.best_ckpt, map_location=self.device, weights_only=True))

        if self.test_loader is not None:
            test_loss, test_acc = self.evaluate(self.test_loader)
        else:
            print("[trainer] Không có test_loader — bỏ qua đánh giá test, dùng kết quả validation cuối cùng.")
            test_loss, test_acc = self.best_val_loss, self.best_val_acc

        with open(os.path.join(self.save_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("Model Architecture: RepShuffleFormer\n")
            f.write(f"Scale Model: {self.args.scale.upper()}\n")
            f.write(f"Batch Size: {self.args.batch_size}\n")
            f.write(f"Phase1 Epochs: {self.phase1_epochs}\n")
            f.write(f"Transition Epochs (soft->hard): {self.transition_epochs}\n")
            f.write(f"Warmup Epochs: {self.warmup_epochs}\n")
            f.write(f"EMA Decay: {self.args.ema_decay if self.ema else 'off'}\n")
            f.write(f"Best Val Accuracy: {self.best_val_acc:.4f}\n")
            f.write(f"Best Val Loss: {self.best_val_loss:.4f}\n")
            if self.test_loader is not None:
                f.write(f"Test Accuracy: {test_acc:.4f}\n")
                f.write(f"Test Loss: {test_loss:.4f}\n")
            else:
                f.write("Test Accuracy: N/A (no test set)\n")
                f.write("Test Loss: N/A (no test set)\n")

        print(f"[FINAL] Scale {self.args.scale.upper()} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        print(f"[DONE] Kết quả lưu tại: {self.save_dir}")


def main():
    args = parse_args()
    set_seed(42)

    save_dir = get_save_dir(base_dir=args.project, name=f"{args.name}_{args.scale.lower()}")
    print(f"[train] Thư mục kết quả: {save_dir}")

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"[train] Không tìm thấy data.yaml: {args.data}")

    cfg = ModelConfig.from_scale(args.scale, batch_size=args.batch_size,
                                label_smoothing=args.label_smoothing,
                                budget_weight=args.budget_weight)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Scale: {args.scale.upper()} | Batch: {args.batch_size} | Device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        cfg, args.data, batch_size=cfg.batch_size,
        num_workers=4, pin_memory=(device == "cuda"),
        persistent_workers=True, strict_num_classes=False,
    )

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(cfg.num_classes)]
    if len(class_names) != cfg.num_classes:
        print(f"[train] Cập nhật cfg.num_classes: {cfg.num_classes} -> {len(class_names)}")
        cfg.num_classes = len(class_names)

    model = build_model(cfg)

    trainer = Trainer(model, train_loader, val_loader, test_loader, class_names, cfg, args, device, save_dir)
    trainer.fit()


if __name__ == "__main__":
    main()