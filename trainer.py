from __future__ import annotations

import argparse
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloaders, set_seed
from loss.cls_loss import build_classification_loss
from model_cls import Classification_Head as Model
from visual import plot_and_save_confusion_matrix, plot_training_metrics


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
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--ema-decay", type=float, default=0.999, help="Hệ số EMA cho model weights.")
    parser.add_argument("--val-loss-cap", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="exp")
    return parser.parse_args()


def cosine_lr(epoch: int, total_epochs: int, warmup_epochs: int, base_lr: float, min_lr: float = 1e-6) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)
    denom = max(1, (total_epochs - warmup_epochs - 1))
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / denom))
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cos_decay


def is_better(best_acc: float, best_loss: float, val_acc: float, val_loss: float, val_loss_cap: float | None = None) -> bool:
    if np.isnan(val_loss) or (val_loss_cap is not None and val_loss >= val_loss_cap):
        return False
    if val_acc > best_acc:
        return True
    return val_acc == best_acc and val_loss < best_loss


def build_optimizer_with_decay(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        target = no_decay if (p.ndim <= 1 or "bias" in name or "norm" in name.lower() or "bn" in name.lower()) else decay
        target.append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 2000):
        self.decay = decay
        # Warmup: những step đầu decay thấp hơn nhiều (ưu tiên trọng số mới học được),
        # rồi mới tiến dần lên `decay` khi đã đủ step — tránh EMA bị "kẹt" gần trạng
        # thái random init trong lúc raw model đã học được kha khá (điều này khiến
        # checkpoint best.pt lưu bằng EMA sớm gần như vô dụng / dự đoán 1 class cố định).
        self.warmup_steps = warmup_steps
        self.n_updates = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.n_updates += 1
        d = self.decay * (1 - math.exp(-self.n_updates / self.warmup_steps))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.items():
            msd[k].copy_(v.to(msd[k].dtype))


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, class_names, args, device, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.args = args
        self.device = device
        self.save_dir = save_dir
        self.weights_dir = os.path.join(save_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)

        self.criterion = build_classification_loss(args.label_smoothing)
        self.optimizer = build_optimizer_with_decay(self.model, lr=args.lr, weight_decay=args.weight_decay)
        self.warmup_epochs = max(0, args.warmup_epochs)

        self.ema = EMA(self.model, decay=args.ema_decay) if args.ema_decay and args.ema_decay > 0 else None

        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.best_val_loss_only = float("inf")
        self.best_ckpt = os.path.join(self.weights_dir, "best.pt")
        self.last_ckpt = os.path.join(self.weights_dir, "last.pt")

        self.train_log_path = os.path.join(self.save_dir, "train_log.txt")
        self.best_loss_log_path = os.path.join(self.save_dir, "best_loss_log.txt")
        self.test_log_path = os.path.join(self.save_dir, "test_log.txt")
        open(self.train_log_path, "w", encoding="utf-8").close()
        open(self.best_loss_log_path, "w", encoding="utf-8").close()

    def _append_log(self, path: str, line: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def train_one_epoch(self, epoch: int, lr: float) -> float:
        self.model.train()
        running_loss = 0.0
        n_seen = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1:02d}/{self.args.epochs}")

        for x, y in pbar:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            # --- FP32 THUẦN: KHÔNG DÙNG AUTOCAST / GRADSCALER ---
            logits = self.model(x)
            loss = self.criterion(logits, y)

            if not torch.isfinite(loss):
                print(f"[trainer] Bỏ qua batch loss không hữu hạn ở epoch {epoch + 1}.")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.ema is not None:
                self.ema.update(self.model)

            loss_val = loss.item()
            running_loss += loss_val * x.size(0)
            n_seen += x.size(0)

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{lr:.2e}"})

        return running_loss / max(1, n_seen)

    @torch.no_grad()
    def evaluate(self, loader) -> tuple[float, float]:
        self.model.eval()
        total_loss, valid_total, correct, total = 0.0, 0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            
            # --- FP32 THUẦN ---
            logits = self.model(x)
            loss = self.criterion(logits, y)
            
            loss_val = loss.item()
            if np.isfinite(loss_val):
                total_loss += loss_val * x.size(0)
                valid_total += x.size(0)
                
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()
            
        val_loss = total_loss / valid_total if valid_total else float("nan")
        return val_loss, correct / max(1, total)

    @torch.no_grad()
    def evaluate_with_ema(self, loader) -> tuple[float, float]:
        if self.ema is None:
            return self.evaluate(loader)
        backup = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        self.ema.copy_to(self.model)
        try:
            loss, acc = self.evaluate(loader)
        finally:
            self.model.load_state_dict(backup)
        return loss, acc

    def _save_best(self, val_acc: float) -> None:
        if self.ema is not None:
            backup = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            self.ema.copy_to(self.model)
            try:
                torch.save(self.model.state_dict(), self.best_ckpt)
                plot_and_save_confusion_matrix(
                    self.model, self.val_loader, self.device, len(self.class_names),
                    class_names=self.class_names,
                    save_path=os.path.join(self.save_dir, "confusion_matrix.png"),
                )
            finally:
                self.model.load_state_dict(backup)
        else:
            torch.save(self.model.state_dict(), self.best_ckpt)
            plot_and_save_confusion_matrix(
                self.model, self.val_loader, self.device, len(self.class_names),
                class_names=self.class_names,
                save_path=os.path.join(self.save_dir, "confusion_matrix.png"),
            )
        print(f"--> [Checkpoint] BEST mới: {self.best_ckpt} (Acc: {val_acc:.4f})")

    def fit(self) -> None:
        print(f"[trainer] Epochs: {self.args.epochs} | Warm-up LR: {self.warmup_epochs} epoch(s) | "
              f"EMA: {'on (decay=' + str(self.args.ema_decay) + ')' if self.ema else 'off'}")

        for epoch in range(self.args.epochs):
            t0 = time.time()

            lr = cosine_lr(epoch, self.args.epochs, self.warmup_epochs, self.args.lr, self.args.min_lr)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            train_loss = self.train_one_epoch(epoch, lr)
            val_loss, val_acc = self.evaluate_with_ema(self.val_loader)
            epoch_time = time.time() - t0

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(lr)

            log_line = (
                f"[Epoch {epoch + 1:02d}/{self.args.epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} (EMA) | "
                f"Val Acc: {val_acc:.4f} (Best: {self.best_val_acc:.4f}) | "
                f"lr: {lr:.2e} | time/epoch: {epoch_time:.1f}s"
            )
            print(log_line)
            self._append_log(self.train_log_path, log_line)

            plot_training_metrics(self.history, save_path=os.path.join(self.save_dir, "results.png"))
            torch.save(self.model.state_dict(), self.last_ckpt)

            if not np.isnan(val_loss) and val_loss < self.best_val_loss_only:
                self.best_val_loss_only = val_loss
                self._append_log(
                    self.best_loss_log_path,
                    f"[Epoch {epoch + 1:02d}] New BEST val_loss: {val_loss:.4f} (val_acc tại epoch này: {val_acc:.4f})"
                )

            if is_better(self.best_val_acc, self.best_val_loss, val_acc, val_loss, self.args.val_loss_cap):
                self.best_val_acc, self.best_val_loss = val_acc, val_loss
                self._save_best(val_acc)

        self._test_and_summarize()

    def _test_and_summarize(self) -> None:
        print(f"\n--- Hoàn tất Huấn luyện. Đánh giá test bằng Best Model ({self.best_ckpt}) ---")
        if os.path.exists(self.best_ckpt):
            self.model.load_state_dict(torch.load(self.best_ckpt, map_location=self.device, weights_only=True))

        if self.test_loader is not None:
            test_loss, test_acc = self.evaluate(self.test_loader)
        else:
            print("[trainer] Không có test_loader — bỏ qua đánh giá test.")
            test_loss, test_acc = self.best_val_loss, self.best_val_acc

        summary_lines = [
            "Model Architecture: RepShuffleFormer",
            f"Scale Model: {self.args.scale.upper()}",
            f"Batch Size: {self.args.batch_size}",
            f"Best Val Accuracy: {self.best_val_acc:.4f}",
            f"Best Val Loss: {self.best_val_loss:.4f}",
        ]
        if self.test_loader is not None:
            summary_lines += [f"Test Accuracy: {test_acc:.4f}", f"Test Loss: {test_loss:.4f}"]
        summary_text = "\n".join(summary_lines)

        with open(os.path.join(self.save_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")
        self._append_log(self.test_log_path, summary_text + "\n" + "-" * 40)

        print(f"[FINAL] Scale {self.args.scale.upper()} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")


def main():
    args = parse_args()
    set_seed(42)

    save_dir = get_save_dir(base_dir=args.project, name=f"{args.name}_{args.scale.lower()}")
    print(f"[train] Thư mục kết quả: {save_dir}")

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"[train] Không tìm thấy data.yaml: {args.data}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Scale: {args.scale.upper()} | Batch: {args.batch_size} | Device: {device}")

    ds_cfg = SimpleNamespace(img_size=args.img_size)
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        ds_cfg, args.data, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        persistent_workers=True, strict_num_classes=False,
    )

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    num_classes = len(class_names)
    print(f"[train] Số lớp phát hiện từ dataset: {num_classes}")

    model = Model(scale=args.scale.upper(), num_classes=num_classes, dropout=args.dropout)

    trainer = Trainer(model, train_loader, val_loader, test_loader, class_names, args, device, save_dir)
    trainer.fit()


if __name__ == "__main__":
    main()