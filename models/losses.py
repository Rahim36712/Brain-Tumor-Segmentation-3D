"""
Brain Tumor Segmentation — Loss Functions
==========================================

Loss functions tailored for multi-class volumetric medical image
segmentation with severe class imbalance (BraTS tumors occupy a
tiny fraction of the brain volume).

Implemented losses
------------------
- **DiceLoss** — directly optimises the Dice coefficient, the primary
  evaluation metric.  Naturally handles class imbalance because it
  measures overlap rather than per-voxel accuracy.

- **DiceCELoss** — combines Dice Loss with Cross-Entropy.  Dice
  provides good gradient signal for overlapping regions, while CE
  contributes stable per-voxel gradients everywhere (especially
  useful early in training when the model predicts mostly
  background).

- **FocalLoss** — down-weights easy (well-classified) voxels and
  focuses on hard examples.  Useful when many voxels are background.

All losses support multi-class segmentation and BraTS region grouping.

Usage::

    from models.losses import DiceCELoss

    criterion = DiceCELoss(num_classes=4, dice_weight=1.0, ce_weight=1.0)
    loss = criterion(logits, targets)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  Soft Dice Loss
# ------------------------------------------------------------------ #
class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.

    Computes per-class Dice coefficients using soft (probabilistic)
    predictions and averages over classes.

    .. math::

        \\mathcal{L}_{\\text{Dice}} = 1 - \\frac{1}{C} \\sum_{c=1}^{C}
        \\frac{2 \\sum_i p_{ic} g_{ic} + \\epsilon}
             {\\sum_i p_{ic} + \\sum_i g_{ic} + \\epsilon}

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes.
    smooth : float
        Smoothing constant to prevent division by zero.
    include_background : bool
        Whether to include class 0 (background) in the loss.
        Setting ``False`` prevents the dominant background class from
        overwhelming the loss.
    softmax : bool
        If ``True``, apply softmax to input logits.
    """

    def __init__(
        self,
        num_classes: int = 4,
        smooth: float = 1e-5,
        include_background: bool = False,
        softmax: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background
        self.softmax = softmax

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : Tensor
            Logits, shape ``(B, C, D, H, W)``.
        target : Tensor
            Ground-truth labels, shape ``(B, 1, D, H, W)`` with integer
            class indices.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        if self.softmax:
            pred = F.softmax(pred, dim=1)

        # One-hot encode target → (B, C, D, H, W)
        target_oh = self._one_hot(target, self.num_classes).float()

        start_class = 0 if self.include_background else 1
        dice_scores = []

        for c in range(start_class, self.num_classes):
            p = pred[:, c]           # (B, D, H, W)
            g = target_oh[:, c]      # (B, D, H, W)

            intersection = (p * g).sum(dim=(1, 2, 3))
            union = p.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3))

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice.mean())

        mean_dice = torch.stack(dice_scores).mean()
        return 1.0 - mean_dice

    @staticmethod
    def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Convert (B, 1, D, H, W) integer labels to (B, C, D, H, W) one-hot."""
        target = target.long()
        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1)  # → (B, D, H, W)
        # F.one_hot: (B, D, H, W) → (B, D, H, W, C) → permute
        oh = F.one_hot(target, num_classes)  # last dim = C
        oh = oh.permute(0, 4, 1, 2, 3).contiguous()  # → (B, C, D, H, W)
        return oh


# ------------------------------------------------------------------ #
#  Combined Dice + Cross-Entropy Loss
# ------------------------------------------------------------------ #
class DiceCELoss(nn.Module):
    """
    Combined Dice Loss + Cross-Entropy Loss.

    **Why combine them?**
    - Dice Loss directly optimises the evaluation metric and handles
      class imbalance, but its gradients can be noisy early in
      training when predictions are far from the target.
    - Cross-Entropy provides stable per-voxel gradients everywhere,
      helping the model learn basic class assignments quickly.
    - Together they converge faster and achieve higher final Dice.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    dice_weight : float
        Weight for the Dice component.
    ce_weight : float
        Weight for the Cross-Entropy component.
    smooth : float
        Dice smoothing constant.
    include_background : bool
        Include background in Dice calculation.
    ce_class_weights : Tensor, optional
        Per-class weights for Cross-Entropy (for additional imbalance
        handling).
    """

    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-5,
        include_background: bool = False,
        ce_class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            smooth=smooth,
            include_background=include_background,
            softmax=True,
        )
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_class_weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : Tensor
            Logits, shape ``(B, C, D, H, W)``.
        target : Tensor
            Labels, shape ``(B, 1, D, H, W)``.

        Returns
        -------
        Tensor
            Weighted sum of Dice and CE losses.
        """
        dice = self.dice_loss(pred, target)

        # CE expects target shape (B, D, H, W) — squeeze channel dim
        target_ce = target.squeeze(1).long()
        ce = self.ce_loss(pred, target_ce)

        return self.dice_weight * dice + self.ce_weight * ce


# ------------------------------------------------------------------ #
#  Focal Loss
# ------------------------------------------------------------------ #
class FocalLoss(nn.Module):
    """
    Focal Loss for hard-example mining.

    Down-weights easy (well-classified) voxels so the model focuses on
    hard-to-classify boundaries and small structures.

    .. math::

        \\text{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\log(p_t)

    Parameters
    ----------
    num_classes : int
        Number of classes.
    alpha : float
        Balancing factor (often set to inverse class frequency).
    gamma : float
        Focusing parameter.  ``gamma=0`` recovers standard CE.
        ``gamma=2`` is a common choice.
    """

    def __init__(
        self,
        num_classes: int = 4,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred : Tensor
            Logits ``(B, C, D, H, W)``.
        target : Tensor
            Labels ``(B, 1, D, H, W)``.
        """
        target = target.squeeze(1).long()  # (B, D, H, W)
        logpt = F.log_softmax(pred, dim=1)  # (B, C, D, H, W)

        # Gather log-probabilities at the target class
        logpt = logpt.gather(1, target.unsqueeze(1))  # (B, 1, D, H, W)
        logpt = logpt.squeeze(1)  # (B, D, H, W)
        pt = logpt.exp()

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = -focal_weight * logpt
        return loss.mean()


# ------------------------------------------------------------------ #
#  Factory
# ------------------------------------------------------------------ #
def get_loss_function(
    name: str = "dice_ce",
    num_classes: int = 4,
    smooth: float = 1e-5,
) -> nn.Module:
    """
    Build a loss function by name.

    Parameters
    ----------
    name : str
        ``"dice"``, ``"dice_ce"``, or ``"focal"``.
    num_classes : int
        Number of segmentation classes.
    smooth : float
        Dice smoothing constant.

    Returns
    -------
    nn.Module
        The configured loss function.
    """
    name = name.lower()
    if name == "dice":
        return DiceLoss(num_classes=num_classes, smooth=smooth)
    elif name == "dice_ce":
        return DiceCELoss(num_classes=num_classes, smooth=smooth)
    elif name == "focal":
        return FocalLoss(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown loss: {name!r}. Choose 'dice', 'dice_ce', or 'focal'.")
