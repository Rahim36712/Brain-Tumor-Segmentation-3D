"""
Brain Tumor Segmentation — Streamlit App Utilities
===================================================

Helper functions for the Streamlit deployment interface.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def load_nifti_from_upload(uploaded_file) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a NIfTI file from a Streamlit UploadedFile object.

    Returns
    -------
    data : ndarray
        Volume data.
    affine : ndarray
        4×4 affine matrix.
    """
    import nibabel as nib

    # Write to temp file (nibabel needs a file path)
    suffix = ".nii.gz" if uploaded_file.name.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    img = nib.load(tmp_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine

    # Clean up
    Path(tmp_path).unlink(missing_ok=True)

    return data, affine


def compute_tumor_stats(segmentation: np.ndarray, voxel_volume_mm3: float = 1.0) -> Dict[str, float]:
    """
    Compute tumor volume statistics from a segmentation mask.

    Parameters
    ----------
    segmentation : ndarray
        (D, H, W) with integer labels {0, 1, 2, 3}.
    voxel_volume_mm3 : float
        Volume of a single voxel in mm³.

    Returns
    -------
    dict
        Tumor volumes in mm³ and cm³.
    """
    stats = {}

    # Per-class volumes
    for label, name in [(1, "NCR/NET"), (2, "Edema"), (3, "Enhancing Tumor")]:
        count = int((segmentation == label).sum())
        vol_mm3 = count * voxel_volume_mm3
        stats[f"{name}_voxels"] = count
        stats[f"{name}_mm3"] = vol_mm3
        stats[f"{name}_cm3"] = vol_mm3 / 1000.0

    # Region volumes
    wt_count = int(np.isin(segmentation, [1, 2, 3]).sum())
    tc_count = int(np.isin(segmentation, [1, 3]).sum())
    et_count = int((segmentation == 3).sum())

    stats["Whole_Tumor_cm3"] = wt_count * voxel_volume_mm3 / 1000.0
    stats["Tumor_Core_cm3"] = tc_count * voxel_volume_mm3 / 1000.0
    stats["Enhancing_Tumor_cm3"] = et_count * voxel_volume_mm3 / 1000.0

    return stats


def normalize_slice_for_display(slice_2d: np.ndarray) -> np.ndarray:
    """Normalize a 2D slice to [0, 1] for display."""
    s = slice_2d.astype(np.float32)
    smin, smax = s.min(), s.max()
    if smax - smin > 1e-8:
        return (s - smin) / (smax - smin)
    return np.zeros_like(s)


def create_overlay_rgb(
    mri_slice: np.ndarray,
    seg_slice: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create an RGB overlay of segmentation on MRI.

    Returns (H, W, 3) uint8 array.
    """
    COLORS = {
        1: np.array([46, 141, 214]),    # blue — NCR/NET
        2: np.array([237, 201, 38]),    # yellow — Edema
        3: np.array([230, 51, 51]),     # red — ET
    }

    mri_norm = normalize_slice_for_display(mri_slice)
    mri_rgb = np.stack([mri_norm] * 3, axis=-1)  # (H, W, 3) float

    overlay = mri_rgb.copy()
    for label_val, color in COLORS.items():
        mask = seg_slice == label_val
        if mask.any():
            color_float = color / 255.0
            overlay[mask] = (1 - alpha) * mri_rgb[mask] + alpha * color_float

    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
