from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def plot_training_metrics(history: dict, save_path: str = "training_metrics.png") -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2, markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", linewidth=2, markersize=4)
    axes[0].set_title("Loss Trajectory", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(epochs, history["val_acc"], "g-o", label="Val Accuracy", linewidth=2, markersize=4)
    axes[1].set_title("Validation Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


@torch.no_grad()
def plot_and_save_confusion_matrix(model, loader, device, num_classes, class_names=None,
                                    save_path: str = "confusion_matrix.png") -> None:
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=False):
            logits, _ = model(x.float())
            preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.numpy() if isinstance(y, torch.Tensor) else y)

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-6)

    if class_names is None or len(class_names) != num_classes:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    else:
        class_names = [str(c) for c in class_names]

    annot_matrix = np.empty(cm.shape, dtype=object)
    for i in range(num_classes):
        for j in range(num_classes):
            count = cm[i, j]
            percent = cm_norm[i, j] * 100
            annot_matrix[i, j] = f"{count}\n({percent:.1f}%)"

    fig_size = max(10, num_classes * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(
        cm_norm,
        annot=annot_matrix if num_classes <= 25 else False,
        fmt="",
        cmap="Blues",
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Tỷ lệ chính xác (Normalized)"},
        ax=ax,
        annot_kws={"size": max(7, int(12 - num_classes * 0.2))},
    )
    plt.title("Confusion Matrix (Counts & Accuracy)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("True Label", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[vis] Đã cập nhật Confusion Matrix tại: {save_path}")
