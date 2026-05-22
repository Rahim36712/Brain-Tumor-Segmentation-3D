"""
Brain Tumor Segmentation — Preprocessing Transforms
====================================================

MONAI-based composed transform pipelines for BraTS MRI preprocessing.

The pipeline handles:
1.  **Z-score normalisation** — per-modality, computed on non-zero voxels
    only (standard in BraTS literature; ignores background air).
2.  **Resampling** — resample to isotropic 1 mm³ voxel spacing for
    spatial consistency across subjects.
3.  **Cropping / Padding** — centre-crop or pad volumes to a uniform
    spatial size so they can be batched.
4.  **Conversion** — ensures float32 images and int64 labels.

Usage::

    from preprocessing.transforms import get_train_transforms, get_val_transforms

    train_tf = get_train_transforms(crop_size=(128, 128, 128))
    val_tf   = get_val_transforms(crop_size=(128, 128, 128))
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------ #
#  Z-score normalisation (non-zero masking)
# ------------------------------------------------------------------ #
class ZScoreNormalize:
    """
    Per-channel z-score normalisation using only non-zero voxels.

    For each channel independently:
        x_norm = (x − μ_nz) / (σ_nz + ε)
    where μ_nz, σ_nz are mean/std computed over voxels > 0.

    This avoids the background (air, which is ~0) biasing statistics.

    Parameters
    ----------
    eps : float
        Small constant added to standard deviation to prevent division
        by zero.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image"]  # (C, D, H, W)  float32
        for c in range(image.shape[0]):
            channel = image[c]
            mask = channel > 0
            if mask.any():
                mean = channel[mask].mean()
                std = channel[mask].std()
                image[c] = np.where(mask, (channel - mean) / (std + self.eps), 0.0)
        sample["image"] = image.astype(np.float32)
        return sample


# ------------------------------------------------------------------ #
#  Min-Max normalisation (alternative)
# ------------------------------------------------------------------ #
class MinMaxNormalize:
    """
    Per-channel min-max normalisation to [0, 1] using non-zero voxels.

    Parameters
    ----------
    eps : float
        Small constant to prevent division by zero.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image"]
        for c in range(image.shape[0]):
            channel = image[c]
            mask = channel > 0
            if mask.any():
                vmin = channel[mask].min()
                vmax = channel[mask].max()
                image[c] = np.where(
                    mask,
                    (channel - vmin) / (vmax - vmin + self.eps),
                    0.0,
                )
        sample["image"] = image.astype(np.float32)
        return sample


# ------------------------------------------------------------------ #
#  Spatial padding / cropping to uniform size
# ------------------------------------------------------------------ #
class CropOrPad:
    """
    Centre-crop or zero-pad a volume to a fixed ``target_size``.

    Operates on both ``image`` (C, D, H, W) and ``label`` (1, D, H, W).

    Parameters
    ----------
    target_size : tuple of int
        Desired (D, H, W) spatial dimensions.
    """

    def __init__(self, target_size: Tuple[int, int, int]) -> None:
        self.target_size = target_size

    def _crop_or_pad_volume(self, vol: np.ndarray) -> np.ndarray:
        """
        Crop or pad a single volume array.  Assumes the first axis is
        channels: shape = (C, D, H, W).
        """
        n_channels = vol.shape[0]
        spatial = vol.shape[1:]  # (D, H, W)

        # --- compute padding / cropping for each spatial dim --- #
        slices_src = []
        slices_dst = []
        pad_widths = [(0, 0)]  # no padding on channel axis

        for i in range(3):
            src_size = spatial[i]
            tgt_size = self.target_size[i]

            if src_size >= tgt_size:
                # centre crop
                start = (src_size - tgt_size) // 2
                slices_src.append(slice(start, start + tgt_size))
                slices_dst.append(slice(None))
                pad_widths.append((0, 0))
            else:
                # zero pad
                pad_before = (tgt_size - src_size) // 2
                pad_after = tgt_size - src_size - pad_before
                slices_src.append(slice(None))
                slices_dst.append(slice(pad_before, pad_before + src_size))
                pad_widths.append((pad_before, pad_after))

        # Apply cropping first
        vol_cropped = vol[:, slices_src[0], slices_src[1], slices_src[2]]
        # Then apply padding
        vol_padded = np.pad(vol_cropped, pad_widths, mode="constant", constant_values=0)

        return vol_padded

    def __call__(self, sample: Dict) -> Dict:
        sample["image"] = self._crop_or_pad_volume(sample["image"])
        if "label" in sample:
            sample["label"] = self._crop_or_pad_volume(sample["label"])
        return sample


# ------------------------------------------------------------------ #
#  Ensure dtypes
# ------------------------------------------------------------------ #
class EnsureDtypes:
    """Cast image to float32 and label to int64."""

    def __call__(self, sample: Dict) -> Dict:
        sample["image"] = sample["image"].astype(np.float32)
        if "label" in sample:
            sample["label"] = sample["label"].astype(np.int64)
        return sample


# ------------------------------------------------------------------ #
#  Compose helper (MONAI-style but dependency-free)
# ------------------------------------------------------------------ #
class Compose:
    """
    Chain multiple transforms sequentially.

    Parameters
    ----------
    transforms : list of callable
        Each callable takes and returns a dict.
    """

    def __init__(self, transforms: Sequence) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}(["]
        for t in self.transforms:
            lines.append(f"  {t.__class__.__name__},")
        lines.append("])")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Factory functions — ready-to-use pipelines
# ------------------------------------------------------------------ #
def get_train_transforms(
    crop_size: Tuple[int, int, int] = (128, 128, 128),
    normalize: str = "zscore",
) -> Compose:
    """
    Build the training preprocessing pipeline.

    Order: Normalise → CropOrPad → EnsureDtypes
    (augmentations are applied separately via ``augmentation.py``).

    Parameters
    ----------
    crop_size : tuple
        Target spatial dimensions (D, H, W).
    normalize : str
        ``"zscore"`` (default) or ``"minmax"``.
    """
    norm = ZScoreNormalize() if normalize == "zscore" else MinMaxNormalize()
    return Compose([
        norm,
        CropOrPad(crop_size),
        EnsureDtypes(),
    ])


def get_val_transforms(
    crop_size: Tuple[int, int, int] = (128, 128, 128),
    normalize: str = "zscore",
) -> Compose:
    """
    Build the validation / test preprocessing pipeline.

    Same as training but **no augmentation**.
    """
    norm = ZScoreNormalize() if normalize == "zscore" else MinMaxNormalize()
    return Compose([
        norm,
        CropOrPad(crop_size),
        EnsureDtypes(),
    ])
