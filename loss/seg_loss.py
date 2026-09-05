from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_IGNORE_INDEX = 255


class BCEDiceLoss(nn.Module):
    """BCE plus Dice loss for binary segmentation."""

    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)

        bce = self.bce(logits, target)
        probabilities = logits.sigmoid()
        dims = (1, 2, 3)
        intersection = (probabilities * target).sum(dim=dims)
        dice = 1.0 - ((2.0 * intersection + 1.0) /
                      (probabilities.sum(dim=dims) + target.sum(dim=dims) + 1.0)).mean()
        return self.bce_weight * bce + self.dice_weight * dice


class SemanticCrossEntropyDiceLoss(nn.Module):
    """Cross-entropy plus class-wise Dice loss for semantic segmentation."""

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        ignore_index: int = DEFAULT_IGNORE_INDEX,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 4:
            target = target.squeeze(1)
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        target = target.long()
        cross_entropy = self.cross_entropy(logits, target)

        valid = target != self.ignore_index
        safe_target = target.clamp(0, self.num_classes - 1)
        one_hot = F.one_hot(safe_target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        probabilities = logits.softmax(dim=1) * valid
        one_hot = one_hot * valid
        intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
        denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        present = one_hot.sum(dim=(0, 2, 3)) > 0
        if present.any():
            dice = ((2.0 * intersection[present] + 1.0) /
                    (denominator[present] + 1.0)).mean()
        else:
            dice = probabilities.sum() * 0.0
        return self.ce_weight * cross_entropy + self.dice_weight * (1.0 - dice)


def build_segmentation_loss(
    multiclass: bool,
    num_classes: int,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
) -> nn.Module:
    if multiclass:
        return SemanticCrossEntropyDiceLoss(num_classes=num_classes, dice_weight=dice_weight)
    return BCEDiceLoss(dice_weight=dice_weight, bce_weight=bce_weight)
