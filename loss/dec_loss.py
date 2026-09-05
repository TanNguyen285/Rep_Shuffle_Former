from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn


class DetectionLoss(nn.Module):
    """Loss for DetectionHead outputs: classification, objectness and LTRB boxes.

    Targets are mappings with ``cls``, ``obj`` and ``box`` entries. Each entry is
    a list with one tensor per feature level, matching the prediction shapes.
    ``box`` contains non-negative (left, top, right, bottom) distances.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        obj_weight: float = 1.0,
        box_weight: float = 5.0,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.box_weight = box_weight
        self.binary_loss = nn.BCEWithLogitsLoss()
        self.regression_loss = nn.SmoothL1Loss()

    def forward(
        self,
        predictions: tuple[Sequence[torch.Tensor], Sequence[torch.Tensor], Sequence[torch.Tensor]],
        targets: Mapping[str, Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        cls_predictions, box_predictions, obj_predictions = predictions
        cls_targets = targets["cls"]
        box_targets = targets["box"]
        obj_targets = targets["obj"]
        total = cls_predictions[0].new_zeros(())

        for cls_pred, box_pred, obj_pred, cls_target, box_target, obj_target in zip(
            cls_predictions, box_predictions, obj_predictions,
            cls_targets, box_targets, obj_targets,
        ):
            total = total + self.cls_weight * self.binary_loss(cls_pred, cls_target.float())
            total = total + self.obj_weight * self.binary_loss(obj_pred, obj_target.float())
            positive = obj_target[:, 0] > 0
            if positive.any():
                total = total + self.box_weight * self.regression_loss(
                    box_pred.permute(0, 2, 3, 1)[positive],
                    box_target.permute(0, 2, 3, 1)[positive].float(),
                )

        return total / max(1, len(cls_predictions))


def build_detection_loss(
    cls_weight: float = 1.0,
    obj_weight: float = 1.0,
    box_weight: float = 5.0,
) -> DetectionLoss:
    return DetectionLoss(cls_weight, obj_weight, box_weight)
