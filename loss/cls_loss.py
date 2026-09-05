from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    """Cross-entropy loss used by the image-classification trainer."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, target)


def build_classification_loss(label_smoothing: float = 0.0) -> ClassificationLoss:
    return ClassificationLoss(label_smoothing=label_smoothing)
