import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class VisionEyeLoss(nn.Module):
    def __init__(
        self,
        label_smoothing: float = 0.05,
        budget_weight: float = 0.05,
    ):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.budget_weight = float(budget_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux: Optional[Dict[str, torch.Tensor]] = None,
        budget_weight: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # --- Task loss ---
        task_loss = self.ce(logits, targets)

        # --- Budget loss ---
        if aux is not None and "budget_loss" in aux:
            budget_loss = aux["budget_loss"]
        else:
            budget_loss = torch.zeros(1, device=logits.device, dtype=logits.dtype)

        # Clamp để tránh explode
        budget_loss = torch.clamp(budget_loss.to(logits.dtype), 0.0, 10.0)

        weight = self.budget_weight if budget_weight is None else float(budget_weight)

        # --- Total ---
        total_loss = task_loss + weight * budget_loss

        # --- Stats (KHÔNG .item()) ---
        stats = {
            "total_loss": total_loss.detach(),
            "task_loss": task_loss.detach(),
            "budget_loss": budget_loss.detach(),
            "budget_weight": torch.tensor(float(weight), device=logits.device, dtype=logits.dtype),
        }

        # optional metrics
        if aux is not None:
            for k in ["expected_tokens", "avg_num_tokens", "overflow_ratio"]:
                if k in aux:
                    stats[k] = aux[k].detach() if torch.is_tensor(aux[k]) else aux[k]

        return total_loss, stats


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    aux: Optional[Dict[str, torch.Tensor]] = None,
    cfg: Optional[object] = None,
    budget_weight: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    label_smoothing = getattr(cfg, "label_smoothing", 0.05) if cfg is not None else 0.05
    default_budget_weight = getattr(cfg, "budget_weight", 0.05) if cfg is not None else 0.05
    criterion = VisionEyeLoss(label_smoothing=label_smoothing, budget_weight=default_budget_weight)
    return criterion(logits, targets, aux, budget_weight=budget_weight)
