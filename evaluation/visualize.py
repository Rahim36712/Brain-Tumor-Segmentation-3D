"""
Brain Tumor Segmentation — Visualization Utilities
===================================================

Publication-quality visualizations for MRI segmentation results.

Features:
- Multi-panel slice views (FLAIR, T1Gd, ground truth, prediction, overlay)
- Tumor overlay with transparency on MRI slices
- Montage view showing multiple axial slices
- 3D tumor volume rendering (optional)
- Training history plots

Usage::

    from evaluation.visualize import (
        plot_segmentation_overlay,
        plot_multi_panel,
        plot_training_history,
    )

    plot_multi_panel(image, ground_truth, prediction, slice_idx=75)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


# ------------------------------------------------------------------ #
#  Color maps and labels for BraTS regions
# ------------------------------------------------------------------ #
TUMOR_COLORS = {
    1: (0.18, 0.55, 0.84, 0.6),   # NCR/NET — blue
    2: (0.93, 0.79, 0.15, 0.6),   # Edema   — yellow
    3: (0.90, 0.20, 0.20, 0.6),   # ET      — red
}

REGION_NAMES = {
    1: "Necrotic / Non-Enhancing (NCR)",
    2: "Peritumoral Edema (ED)",
    3: "Enhancing Tumor (ET)",
}

MODALITY_NAMES = ["T1", "T1ce (Gd)", "T2", "FLAIR"]


# ------------------------------------------------------------------ #
#  Segmentation overlay on a single slice
# ------------------------------------------------------------------ #
def plot_segmentation_overlay(
    mri_slice: np.ndarray,
    seg_slice: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    alpha: float = 0.5,
    show_legend: bool = True,
) -> plt.Axes:
    """
    Overlay a segmentation mask on an MRI slice.

    Parameters
    ----------
    mri_slice : ndarray
        2D MRI slice (H, W), grayscale.
    seg_slice : ndarray
        2D segmentation mask (H, W) with integer labels.
    ax : Axes, optional
        Matplotlib axes to draw on.
    title : str
        Plot title.
    alpha : float
        Overlay transparency.
    show_legend : bool
        Whether to display the colour legend.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.imshow(mri_slice, cmap="gray", interpolation="none")

    # Create colored overlay
    overlay = np.zeros((*seg_slice.shape, 4), dtype=np.float32)
    for label_val, color in TUMOR_COLORS.items():
        mask = seg_slice == label_val
        if mask.any():
            overlay[mask] = color

    ax.imshow(overlay, interpolation="none", alpha=alpha)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    if show_legend:
        patches = []
        for label_val, name in REGION_NAMES.items():
            if (seg_slice == label_val).any():
                c = TUMOR_COLORS[label_val][:3]
                patches.append(mpatches.Patch(color=c, label=name))
        if patches:
            ax.legend(handles=patches, loc="lower right", fontsize=7,
                      framealpha=0.8, fancybox=True)

    return ax


# ------------------------------------------------------------------ #
#  Multi-panel view
# ------------------------------------------------------------------ #
def plot_multi_panel(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    slice_idx: Optional[int] = None,
    modality_idx: int = 3,  # FLAIR by default
    save_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a 5-panel visualization:
    [MRI FLAIR] [MRI T1ce] [Ground Truth] [Prediction] [Overlay Comparison]

    Parameters
    ----------
    image : ndarray
        4D MRI volume (C, D, H, W).
    ground_truth : ndarray
        3D segmentation mask (D, H, W).
    prediction : ndarray
        3D predicted segmentation (D, H, W).
    slice_idx : int, optional
        Axial slice index.  If ``None``, picks the slice with
        the most tumor voxels.
    modality_idx : int
        Primary modality to show (default 3 = FLAIR).
    save_path : str or Path, optional
        If given, save the figure to this path.
    dpi : int
        Figure resolution.
    """
    if slice_idx is None:
        # Pick slice with most tumor voxels
        tumor_per_slice = (ground_truth > 0).sum(axis=(1, 2))
        slice_idx = int(np.argmax(tumor_per_slice))

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Axial Slice {slice_idx}", fontsize=14, fontweight="bold", y=1.02)

    # Panel 1: FLAIR
    axes[0].imshow(image[modality_idx, slice_idx], cmap="gray")
    axes[0].set_title(MODALITY_NAMES[modality_idx], fontsize=11)
    axes[0].axis("off")

    # Panel 2: T1ce
    t1ce_idx = 1
    axes[1].imshow(image[t1ce_idx, slice_idx], cmap="gray")
    axes[1].set_title(MODALITY_NAMES[t1ce_idx], fontsize=11)
    axes[1].axis("off")

    # Panel 3: Ground truth overlay
    plot_segmentation_overlay(
        image[modality_idx, slice_idx],
        ground_truth[slice_idx],
        ax=axes[2],
        title="Ground Truth",
        show_legend=False,
    )

    # Panel 4: Prediction overlay
    plot_segmentation_overlay(
        image[modality_idx, slice_idx],
        prediction[slice_idx],
        ax=axes[3],
        title="Prediction",
        show_legend=False,
    )

    # Panel 5: Side-by-side comparison (GT=green contour, Pred=red fill)
    axes[4].imshow(image[modality_idx, slice_idx], cmap="gray")
    # Prediction as filled overlay
    pred_overlay = np.zeros((*prediction[slice_idx].shape, 4))
    for lv, c in TUMOR_COLORS.items():
        pred_overlay[prediction[slice_idx] == lv] = c
    axes[4].imshow(pred_overlay, alpha=0.4)
    # GT as contour
    for lv in [1, 2, 3]:
        mask = (ground_truth[slice_idx] == lv).astype(np.float32)
        if mask.any():
            axes[4].contour(mask, levels=[0.5], colors=["lime"], linewidths=1.0)
    axes[4].set_title("Overlay (green=GT contour)", fontsize=11)
    axes[4].axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=TUMOR_COLORS[k][:3], label=REGION_NAMES[k])
        for k in REGION_NAMES
    ]
    legend_patches.append(mpatches.Patch(edgecolor="lime", facecolor="none",
                                          label="GT Contour", linewidth=2))
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ------------------------------------------------------------------ #
#  Slice montage
# ------------------------------------------------------------------ #
def plot_slice_montage(
    image: np.ndarray,
    segmentation: np.ndarray,
    num_slices: int = 8,
    modality_idx: int = 3,
    save_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Show a montage of equally-spaced axial slices with segmentation overlay.

    Parameters
    ----------
    image : ndarray
        (C, D, H, W) MRI volume.
    segmentation : ndarray
        (D, H, W) segmentation mask.
    num_slices : int
        Number of slices to display.
    """
    depth = image.shape[1]
    indices = np.linspace(0, depth - 1, num_slices, dtype=int)

    cols = min(num_slices, 4)
    rows = (num_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(indices):
        r, c = divmod(i, cols)
        plot_segmentation_overlay(
            image[modality_idx, idx],
            segmentation[idx],
            ax=axes[r, c],
            title=f"Slice {idx}",
            show_legend=(i == 0),
        )

    # Hide unused axes
    for i in range(len(indices), rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    plt.suptitle("Segmentation Montage", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


# ------------------------------------------------------------------ #
#  Training history plot
# ------------------------------------------------------------------ #
def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot training/validation loss and Dice curves.

    Parameters
    ----------
    history : dict
        Keys: ``train_loss``, ``val_loss``, ``val_dice``, ``lr``.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss curves
    if "train_loss" in history:
        axes[0].plot(epochs, history["train_loss"], label="Train", color="#2196F3", linewidth=1.5)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Val", color="#F44336", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Dice curve
    if "val_dice" in history:
        axes[1].plot(epochs, history["val_dice"], label="Val Dice", color="#4CAF50", linewidth=1.5)
        best_epoch = int(np.argmax(history["val_dice"]))
        best_dice = history["val_dice"][best_epoch]
        axes[1].axhline(y=best_dice, color="#4CAF50", linestyle="--", alpha=0.5)
        axes[1].scatter([best_epoch + 1], [best_dice], color="#4CAF50", s=100, zorder=5,
                        label=f"Best: {best_dice:.4f} @ epoch {best_epoch + 1}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Coefficient")
    axes[1].set_title("Validation Dice")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Learning rate
    if "lr" in history:
        axes[2].plot(epochs, history["lr"], color="#FF9800", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
