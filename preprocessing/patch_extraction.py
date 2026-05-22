"""
Brain Tumor Segmentation — Patch Extraction
============================================

Strategies for extracting 3D patches from full MRI volumes.

**Why patches?**
Full BraTS volumes are 240 × 240 × 155 voxels with 4 channels — far
too large to fit into GPU memory as whole volumes.  Patch-based
training samples smaller sub-volumes (e.g. 128³) per iteration.

**Sampling strategies implemented here:**
1. *Random patches* — uniform random location (fast, simple).
2. *Foreground-biased patches* — centres are biased toward voxels
   containing tumor, ensuring the model sees enough positive examples.
3. *Sliding window patches* — deterministic grid with configurable
   overlap, used at inference time to tile the full volume.

Usage::

    from preprocessing.patch_extraction import (
        RandomPatchSampler,
        ForegroundBiasedSampler,
        SlidingWindowSampler,
    )

    # Training — random patches biased toward tumor
    sampler = ForegroundBiasedSampler(patch_size=(128, 128, 128),
                                      fg_ratio=0.7)
    patches = sampler(image, label)          # list of (image, label) patches

    # Inference — sliding window
    sw = SlidingWindowSampler(patch_size=(128, 128, 128), overlap=0.5)
    for patch, coords in sw(image):
        pred = model(patch)
        # … reassemble using coords
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------ #
#  Random Patch Sampler
# ------------------------------------------------------------------ #
class RandomPatchSampler:
    """
    Extract random 3D patches from a volume.

    Parameters
    ----------
    patch_size : tuple of int
        (D, H, W) size of each patch.
    num_patches : int
        Number of patches to extract per call.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        num_patches: int = 4,
    ) -> None:
        self.patch_size = patch_size
        self.num_patches = num_patches

    def __call__(
        self, sample: Dict[str, np.ndarray]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Extract ``num_patches`` random patches.

        Parameters
        ----------
        sample : dict
            Must contain ``"image"`` (C, D, H, W) and optionally ``"label"``
            (1, D, H, W).

        Returns
        -------
        list of dict
            Each dict has ``"image"`` and ``"label"`` crops.
        """
        image = sample["image"]
        label = sample.get("label")
        spatial = image.shape[1:]  # (D, H, W)

        patches = []
        for _ in range(self.num_patches):
            starts = [
                np.random.randint(0, max(1, s - p))
                for s, p in zip(spatial, self.patch_size)
            ]
            slices = tuple(
                slice(st, st + ps) for st, ps in zip(starts, self.patch_size)
            )
            patch = {"image": image[:, slices[0], slices[1], slices[2]]}
            if label is not None:
                patch["label"] = label[:, slices[0], slices[1], slices[2]]
            # carry over metadata
            if "subject" in sample:
                patch["subject"] = sample["subject"]
            patches.append(patch)
        return patches


# ------------------------------------------------------------------ #
#  Foreground-Biased Patch Sampler
# ------------------------------------------------------------------ #
class ForegroundBiasedSampler:
    """
    Extract patches biased toward tumor-containing regions.

    With probability ``fg_ratio`` the patch centre is placed on a
    randomly chosen foreground (non-zero label) voxel.  Otherwise a
    fully random centre is used.

    Parameters
    ----------
    patch_size : tuple of int
        (D, H, W) patch dimensions.
    num_patches : int
        Patches per sample.
    fg_ratio : float
        Probability of centering on a foreground voxel.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        num_patches: int = 4,
        fg_ratio: float = 0.7,
    ) -> None:
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.fg_ratio = fg_ratio

    def _clamp_start(self, centre: int, patch_len: int, vol_len: int) -> int:
        """Clamp the start index so the patch stays within bounds."""
        start = centre - patch_len // 2
        start = max(0, min(start, vol_len - patch_len))
        return start

    def __call__(
        self, sample: Dict[str, np.ndarray]
    ) -> List[Dict[str, np.ndarray]]:
        image = sample["image"]
        label = sample.get("label")
        spatial = image.shape[1:]

        # find foreground voxel coordinates
        fg_coords: Optional[np.ndarray] = None
        if label is not None:
            fg_mask = label[0] > 0  # (D, H, W)
            if fg_mask.any():
                fg_coords = np.argwhere(fg_mask)  # (N, 3)

        patches = []
        for _ in range(self.num_patches):
            use_fg = (
                fg_coords is not None
                and np.random.rand() < self.fg_ratio
            )
            if use_fg:
                idx = np.random.randint(len(fg_coords))
                centre = fg_coords[idx]
                starts = [
                    self._clamp_start(int(centre[i]), self.patch_size[i], spatial[i])
                    for i in range(3)
                ]
            else:
                starts = [
                    np.random.randint(0, max(1, s - p))
                    for s, p in zip(spatial, self.patch_size)
                ]

            slices = tuple(
                slice(st, st + ps) for st, ps in zip(starts, self.patch_size)
            )
            patch = {"image": image[:, slices[0], slices[1], slices[2]]}
            if label is not None:
                patch["label"] = label[:, slices[0], slices[1], slices[2]]
            if "subject" in sample:
                patch["subject"] = sample["subject"]
            patches.append(patch)
        return patches


# ------------------------------------------------------------------ #
#  Sliding Window Sampler (Inference)
# ------------------------------------------------------------------ #
@dataclass
class PatchCoords:
    """Start and end coordinates for a single patch."""
    starts: Tuple[int, int, int]
    ends: Tuple[int, int, int]


class SlidingWindowSampler:
    """
    Deterministic sliding-window patch extraction with overlap.

    Used at **inference time** to tile the full volume into overlapping
    patches, run the model on each, then reassemble predictions.

    Parameters
    ----------
    patch_size : tuple of int
        (D, H, W) dimensions of each patch.
    overlap : float
        Fraction of overlap between adjacent patches (0.0–0.9).
        Higher overlap → smoother boundaries but slower inference.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        overlap: float = 0.5,
    ) -> None:
        self.patch_size = patch_size
        self.overlap = overlap

    def _compute_steps(self, vol_size: int, patch_size: int) -> List[int]:
        """Compute start positions along one axis."""
        stride = max(1, int(patch_size * (1 - self.overlap)))
        starts = list(range(0, vol_size - patch_size + 1, stride))
        # Ensure the last patch reaches the volume edge
        if not starts or starts[-1] + patch_size < vol_size:
            starts.append(max(0, vol_size - patch_size))
        return starts

    def __call__(
        self, image: np.ndarray
    ) -> Generator[Tuple[np.ndarray, PatchCoords], None, None]:
        """
        Yield (patch, coords) tuples covering the entire volume.

        Parameters
        ----------
        image : ndarray
            Shape (C, D, H, W).

        Yields
        ------
        patch : ndarray
            Shape (C, pD, pH, pW).
        coords : PatchCoords
            Start / end indices for reassembly.
        """
        spatial = image.shape[1:]
        steps = [
            self._compute_steps(spatial[i], self.patch_size[i])
            for i in range(3)
        ]

        for d in steps[0]:
            for h in steps[1]:
                for w in steps[2]:
                    starts = (d, h, w)
                    ends = (
                        d + self.patch_size[0],
                        h + self.patch_size[1],
                        w + self.patch_size[2],
                    )
                    patch = image[
                        :,
                        starts[0]:ends[0],
                        starts[1]:ends[1],
                        starts[2]:ends[2],
                    ]
                    yield patch, PatchCoords(starts=starts, ends=ends)

    def reassemble(
        self,
        patches: List[Tuple[np.ndarray, PatchCoords]],
        volume_shape: Tuple[int, ...],
        num_classes: int,
    ) -> np.ndarray:
        """
        Reassemble predicted patches into a full-volume prediction.

        Uses **averaging in overlapping regions** for smooth boundaries.

        Parameters
        ----------
        patches : list of (prediction, coords)
            Each prediction has shape (num_classes, pD, pH, pW).
        volume_shape : tuple
            Full (D, H, W) spatial shape.
        num_classes : int
            Number of output classes.

        Returns
        -------
        ndarray
            Shape (num_classes, D, H, W) — averaged probability map.
        """
        output = np.zeros((num_classes, *volume_shape), dtype=np.float32)
        count = np.zeros((1, *volume_shape), dtype=np.float32)

        for pred, coords in patches:
            s, e = coords.starts, coords.ends
            output[:, s[0]:e[0], s[1]:e[1], s[2]:e[2]] += pred
            count[:, s[0]:e[0], s[1]:e[1], s[2]:e[2]] += 1.0

        # avoid division by zero
        count = np.maximum(count, 1.0)
        return output / count
