from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from seg_loss import build_segmentation_loss
from model_seg import SegmentationModel
from seg_dataloader import (
    get_ade_dataloaders,
    get_kvasir_dataloaders,
    get_voc_dataloaders,
    set_seed,
)


ADE_NUM_CLASSES = 150
VOC_NUM_CLASSES = 21
ADE_IGNORE_INDEX = 255


def binary_stats(logits: torch.Tensor, target: torch.Tensor) -> tuple[int, int, int, int]:
    target = target.float()
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    prediction = logits.sigmoid() > 0.5
    truth = target > 0.5
    intersection = int((prediction & truth).sum().item())
    predicted = int(prediction.sum().item())
    actual = int(truth.sum().item())
    union = predicted + actual - intersection
    return intersection, union, predicted, actual


def ade_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if target.ndim == 4:
        target = target.squeeze(1)
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    prediction = logits.argmax(dim=1)
    valid = target != ADE_IGNORE_INDEX
    confusion = torch.bincount(
        (target[valid] * num_classes + prediction[valid]).long(),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    return confusion.cpu()


def semantic_metrics(confusion: torch.Tensor) -> tuple[float, float]:
    intersection = confusion.diag().float()
    union = confusion.sum(dim=0).float() + confusion.sum(dim=1).float() - intersection
    valid_classes = union > 0
    miou = (intersection[valid_classes] / union[valid_classes]).mean().item()
    pixel_accuracy = intersection.sum().item() / max(1, confusion.sum().item())
    return float(miou), float(pixel_accuracy)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or "bias" in name or "norm" in name.lower() or "bn" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def cosine_lr(epoch: int, epochs: int, warmup_epochs: int, lr: float, min_lr: float) -> float:
    if warmup_epochs and epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
    progress = min(1.0, max(0.0, progress))
    return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_save_dir(project: str, name: str) -> str:
    os.makedirs(project, exist_ok=True)
    path = os.path.join(project, name)
    index = 2
    while os.path.exists(path):
        path = os.path.join(project, f"{name}{index}")
        index += 1
    os.makedirs(os.path.join(path, "weights"), exist_ok=True)
    return path


def run_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    train: bool,
    multiclass: bool = False,
    num_classes: int = ADE_NUM_CLASSES,
    amp_enabled: bool = False,
    scaler=None,
    max_grad_norm: float = 1.0,
):
    model.train(train)
    total_loss = 0.0
    total_items = 0
    intersection = union = predicted = actual = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        progress = tqdm(loader, leave=False, desc="train" if train else "eval")
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite segmentation loss encountered")

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            if multiclass:
                confusion += ade_stats(
                    logits, masks, num_classes=num_classes
                )
            else:
                batch_intersection, batch_union, batch_predicted, batch_actual = binary_stats(logits, masks)
                intersection += batch_intersection
                union += batch_union
                predicted += batch_predicted
                actual += batch_actual
            total_loss += loss.item() * images.size(0)
            total_items += images.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

    if multiclass:
        mean_iou, pixel_accuracy = semantic_metrics(confusion)
        return total_loss / max(1, total_items), mean_iou, pixel_accuracy

    dice = 2.0 * intersection / max(1, predicted + actual)
    iou = intersection / max(1, union)
    return total_loss / max(1, total_items), dice, iou


def parse_args():
    parser = argparse.ArgumentParser(description="Train RepShuffleFormer for semantic segmentation")
    parser.add_argument("--dataset", choices=["kvasir", "ade", "voc"], default="voc")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root; defaults to the selected dataset path",
    )
    parser.add_argument("--scale", default="s", choices=["s", "m", "l"])
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--n-test", type=int, default=120)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/seg")
    parser.add_argument("--name", default="voc")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = make_save_dir(args.project, f"{args.name}_{args.scale}")

    if args.dataset == "ade":
        data_root = args.data_root or r"D:\ADEChallengeData2016\ADEChallengeData2016"
        train_loader, val_loader, test_loader = get_ade_dataloaders(
            root=data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=True,
            seed=args.seed,
        )
    elif args.dataset == "voc":
        data_root = args.data_root or r"D:\VOCdevkit\VOC2012"
        train_loader, val_loader, test_loader = get_voc_dataloaders(
            root=data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=True,
            seed=args.seed,
        )
    else:
        data_root = args.data_root or r"D:\kvasir-seg\Kvasir-SEG"
        train_loader, val_loader, test_loader = get_kvasir_dataloaders(
            root=data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            n_test=args.n_test,
            val_ratio=args.val_ratio,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=True,
            seed=args.seed,
        )
    if val_loader is None:
        raise ValueError("val_loader is empty. Set --val-ratio to a value greater than 0.")

    is_multiclass = args.dataset in ("ade", "voc")
    num_classes = ADE_NUM_CLASSES if args.dataset == "ade" else VOC_NUM_CLASSES
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model = SegmentationModel(
        scale=args.scale.upper(),
        num_classes=num_classes if is_multiclass else 1,
        dropout=args.dropout,
    ).to(device)
    criterion = build_segmentation_loss(
        multiclass=is_multiclass,
        num_classes=num_classes,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
    )
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    best_metric = -1.0
    best_val_loss = float("inf")
    log_path = os.path.join(save_dir, "train_log.txt")
    metric_name = "mIoU" if is_multiclass else "dice"
    secondary_metric_name = "pixel_acc" if is_multiclass else "iou"

    print(
        f"[seg] dataset={args.dataset} | device={device} | "
        f"scale={args.scale.upper()} | output={save_dir}"
    )
    for epoch in range(args.epochs):
        started = time.time()
        current_lr = cosine_lr(epoch, args.epochs, args.warmup_epochs, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        train_loss, train_dice, train_iou = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train=True,
            multiclass=is_multiclass,
            num_classes=num_classes,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        val_loss, val_dice, val_iou = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            train=False,
            multiclass=is_multiclass,
            num_classes=num_classes,
            amp_enabled=amp_enabled,
        )
        torch.save(model.state_dict(), os.path.join(save_dir, "weights", "last.pt"))

        line = (
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.5f} train_{metric_name}={train_dice:.5f} "
            f"train_{secondary_metric_name}={train_iou:.5f} | "
            f"val_loss={val_loss:.5f} val_{metric_name}={val_dice:.5f} "
            f"val_{secondary_metric_name}={val_iou:.5f} | "
            f"lr={current_lr:.2e} time={time.time() - started:.1f}s"
        )
        print(line)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")

        if val_dice > best_metric or (val_dice == best_metric and val_loss < best_val_loss):
            best_metric = val_dice
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "weights", "best.pt"))
            print(f"[seg] saved best.pt: metric={best_metric:.5f}")

    best_path = os.path.join(save_dir, "weights", "best.pt")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_loss, test_dice, test_iou = run_epoch(
        model,
        test_loader,
        optimizer,
        criterion,
        device,
        train=False,
        multiclass=is_multiclass,
        num_classes=num_classes,
        amp_enabled=amp_enabled,
    )
    summary = (
        f"Best val {metric_name}: {best_metric:.5f}\n"
        f"Best val loss: {best_val_loss:.5f}\n"
        f"Test loss: {test_loss:.5f}\n"
        f"Test {metric_name}: {test_dice:.5f}\n"
        f"Test {secondary_metric_name}: {test_iou:.5f}\n"
    )
    with open(os.path.join(save_dir, "summary.txt"), "w", encoding="utf-8") as summary_file:
        summary_file.write(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
