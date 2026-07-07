"""Loss functions for DNA language model training.

This module provides specialized loss functions for sequence
classification and regression tasks.
"""

import torch
import torch.nn as nn
from torch.nn import functional


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in classification.

    Focal Loss down-weights well-classified examples and focuses on
    hard examples, making it effective for imbalanced datasets.

    Reference:
        Lin, T.Y. et al. "Focal Loss for Dense Object Detection." ICCV 2017.

    Args:
        alpha: Weighting factor for the rare class (default: 0.25).
        gamma: Focusing parameter; higher values focus more on hard examples (default: 2.0).
        reduction: Reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss between inputs and targets.

        Args:
            inputs: Raw model outputs (logits).
            targets: Ground truth labels.

        Returns:
            Scalar loss value (or unreduced tensor if reduction='none').
        """
        bce_loss: torch.Tensor = functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        pt: torch.Tensor = torch.exp(-bce_loss)
        focal_loss: torch.Tensor = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
