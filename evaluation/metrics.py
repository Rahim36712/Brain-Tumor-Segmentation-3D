"""
Brain Tumor Segmentation — Evaluation Metrics
==============================================

Medically relevant segmentation metrics for the BraTS challenge.

All metrics are computed **per BraTS evaluation region**:
- **Enhancing Tumor (ET)**: original label 4 → mapped label 3
- **Tumor Core (TC)**: labels 1 + 4 → mapped labels 1 + 3
- **Whole Tumor (WT)**: labels 1 + 2 + 4 → mapped labels 1 + 2 + 3

Why each metric matters
-----------------------
- **Dice Coefficient** — the *primary* BraTS metric.  Measures
  volumetric overlap between prediction and ground truth.  Ranges
  0–1 (1 = perfect match).  Robust to class imbalance because it
  focuses on the positive class.

- **IoU (Jaccard)** — similar to Dice but penalises false positives
  more heavily.  Useful for assessing boundary precision.

- **Sensitivity (Recall)** — fraction of true tumor voxels that are
  correctly detected.  Critical in clinical settings: *missing a
  tumor region is worse than a false positive*.

- **Specificity** — fraction of true negatives correctly identified.
  Guards against the model predicting tumor everywhere.

- **Hausdorff Distance 95** — the 95th percentile of symmetric
  surface distances (mm).  Captures worst-case boundary error.
  Lower is better; clinically relevant for surgical planning.

Usage::

    from evaluation.metrics import compute_brats_metrics

    metrics = compute_brats_metrics(prediction, ground_truth)
    # metrics = {
    #     "ET_Dice": 0.85, "TC_Dice": 0.91, "WT_Dice": 0.93,
    #     "ET_IoU":  0.74, ...
    # }
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


# ------------------------------------------------------------------ #
#  Core metric functions (operate on binary masks)
# ------------------------------------------------------------------ #
def dice_coefficient(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5,
) -> float:
    """
    Dice Similarity Coefficient for two binary masks.

    .. math::

        \\text{DSC} = \\frac{2 |P \\cap G|}{|P| + |G|}
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    return float((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def iou_score(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5,
) -> float:
    """
    Intersection over Union (Jaccard Index).

    .. math::

        \\text{IoU} = \\frac{|P \\cap G|}{|P \\cup G|}
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float((intersection + smooth) / (union + smooth))


def sensitivity(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5,
) -> float:
    """
    Sensitivity (Recall / True Positive Rate).

    .. math::

        \\text{Sens} = \\frac{TP}{TP + FN}

    High sensitivity means the model detects most tumor voxels.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()
    return float((tp + smooth) / (tp + fn + smooth))


def specificity(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-5,
) -> float:
    """
    Specificity (True Negative Rate).

    .. math::

        \\text{Spec} = \\frac{TN}{TN + FP}
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    tn = np.logical_and(~pred, ~target).sum()
    fp = np.logical_and(pred, ~target).sum()
    return float((tn + smooth) / (tn + fp + smooth))


def hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    95th percentile Hausdorff Distance between prediction and
    ground truth surfaces.

    Uses Euclidean distance scaled by voxel spacing.  Returns ``inf``
    if either mask is empty.

    Parameters
    ----------
    pred, target : ndarray
        Binary segmentation masks.
    voxel_spacing : tuple of float
        Physical voxel dimensions (mm).
    """
    from scipy.ndimage import distance_transform_edt

    pred = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() or not target.any():
        return float("inf")

    # Distance transform of the complement → distance from each
    # surface voxel of one mask to the nearest surface of the other
    pred_surface = pred ^ _erode(pred)
    target_surface = target ^ _erode(target)

    if not pred_surface.any() or not target_surface.any():
        return 0.0

    dt_pred = distance_transform_edt(~pred, sampling=voxel_spacing)
    dt_target = distance_transform_edt(~target, sampling=voxel_spacing)

    # Distances from pred surface → target and target surface → pred
    d_pred_to_target = dt_target[pred_surface]
    d_target_to_pred = dt_pred[target_surface]

    all_distances = np.concatenate([d_pred_to_target, d_target_to_pred])
    return float(np.percentile(all_distances, 95))


def _erode(mask: np.ndarray) -> np.ndarray:
    """Simple binary erosion using scipy."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, iterations=1).astype(bool)


# ------------------------------------------------------------------ #
#  BraTS region extraction
# ------------------------------------------------------------------ #
def _extract_regions(
    seg: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Extract BraTS evaluation regions from a label map.

    Assumes labels have been remapped: {0: bg, 1: NCR/NET, 2: ED, 3: ET}.

    Returns binary masks for:
    - ET  = label 3
    - TC  = labels {1, 3}
    - WT  = labels {1, 2, 3}
    """
    return {
        "ET": (seg == 3).astype(np.uint8),
        "TC": np.isin(seg, [1, 3]).astype(np.uint8),
        "WT": np.isin(seg, [1, 2, 3]).astype(np.uint8),
    }


# ------------------------------------------------------------------ #
#  Comprehensive BraTS metrics
# ------------------------------------------------------------------ #
def compute_brats_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    include_hausdorff: bool = True,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for one subject.

    Parameters
    ----------
    pred : ndarray
        Predicted segmentation, shape ``(D, H, W)`` with integer labels.
    target : ndarray
        Ground-truth segmentation, same shape / label format.
    voxel_spacing : tuple
        Physical voxel dimensions in mm.
    include_hausdorff : bool
        Whether to compute Hausdorff Distance (slower).

    Returns
    -------
    dict
        Keys: ``{region}_{metric}`` e.g. ``ET_Dice``, ``WT_IoU``.
    """
    pred_regions = _extract_regions(pred)
    target_regions = _extract_regions(target)

    metrics: Dict[str, float] = {}

    for region in ["ET", "TC", "WT"]:
        p = pred_regions[region]
        g = target_regions[region]

        metrics[f"{region}_Dice"] = dice_coefficient(p, g)
        metrics[f"{region}_IoU"] = iou_score(p, g)
        metrics[f"{region}_Sensitivity"] = sensitivity(p, g)
        metrics[f"{region}_Specificity"] = specificity(p, g)

        if include_hausdorff:
            metrics[f"{region}_HD95"] = hausdorff_distance_95(
                p, g, voxel_spacing=voxel_spacing,
            )

    # Mean Dice across regions (primary comparison metric)
    metrics["Mean_Dice"] = np.mean(
        [metrics["ET_Dice"], metrics["TC_Dice"], metrics["WT_Dice"]]
    ).item()

    return metrics


# ------------------------------------------------------------------ #
#  Batch metric aggregation utility
# ------------------------------------------------------------------ #
def aggregate_metrics(
    metrics_list: list[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate per-subject metrics into mean ± std.

    Parameters
    ----------
    metrics_list : list of dict
        Each dict from ``compute_brats_metrics``.

    Returns
    -------
    dict
        ``{metric_name: {"mean": ..., "std": ...}}``.
    """
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    result = {}
    for key in keys:
        vals = [m[key] for m in metrics_list if np.isfinite(m[key])]
        result[key] = {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "std": float(np.std(vals)) if vals else 0.0,
        }
    return result


# ------------------------------------------------------------------ #
#  Torch convenience wrapper (for use during training)
# ------------------------------------------------------------------ #
def dice_coefficient_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 4,
    smooth: float = 1e-5,
    include_background: bool = False,
) -> torch.Tensor:
    """
    Differentiable per-class Dice for monitoring during training.

    Parameters
    ----------
    pred : Tensor
        Logits ``(B, C, D, H, W)``.
    target : Tensor
        Labels ``(B, 1, D, H, W)``.

    Returns
    -------
    Tensor
        Mean Dice across classes.
    """
    pred_soft = torch.softmax(pred, dim=1)
    target_squeezed = target.squeeze(1).long()
    target_oh = torch.nn.functional.one_hot(
        target_squeezed, num_classes
    ).permute(0, 4, 1, 2, 3).float()

    start = 0 if include_background else 1
    dice_scores = []

    for c in range(start, num_classes):
        p = pred_soft[:, c]
        g = target_oh[:, c]
        intersection = (p * g).sum()
        union = p.sum() + g.sum()
        dice_scores.append((2 * intersection + smooth) / (union + smooth))

    return torch.stack(dice_scores).mean()
