"""Label smoothing cross-entropy loss for sign language classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    Replaces one-hot targets with smoothed distributions:
        y_smooth = (1 - eps) * y_hard + eps / num_classes

    This prevents overconfident predictions and improves generalisation,
    especially useful when training on imbalanced sign datasets.

    Reference:
        Szegedy et al. (2016) — Rethinking the Inception Architecture.
        Müller, Kornblith & Hinton (NeurIPS 2019) — When Does Label Smoothing Help?

    Args:
        smoothing: Label smoothing factor epsilon in [0, 1). Default 0.1.
        reduction: "mean" or "sum". Default "mean".
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}.")
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute smoothed cross-entropy loss.

        Args:
            logits: Raw model outputs, shape (batch_size, num_classes).
            targets: Ground-truth class indices, shape (batch_size,).

        Returns:
            Scalar loss tensor.
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
