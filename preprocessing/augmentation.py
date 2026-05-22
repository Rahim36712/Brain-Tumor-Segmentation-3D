"""
Brain Tumor Segmentation — Data Augmentation
=============================================

3D-aware augmentation transforms for volumetric MRI data.

All transforms operate on dict samples with keys ``"image"``
(C, D, H, W) and optionally ``"label"`` (1, D, H, W).

**Augmentation strategy rationale:**
- *Random flipping* — brain anatomy is approximately symmetric;
  flipping along all three axes is physically plausible.
- *Random rotation* — small rotations (< ±12°) simulate scan-angle
  variation without distorting anatomy.
- *Intensity shift / scale* — compensates for inter-scanner and
  inter-subject intensity differences.
- *Gaussian noise* — improves robustness to acquisition noise.
- *Elastic deformation* — simulates soft-tissue deformation;
  a well-established augmentation for medical segmentation.

Usage::

    from preprocessing.augmentation import get_augmentation_pipeline

    aug = get_augmentation_pipeline(cfg)
    augmented_sample = aug(sample)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    map_coordinates,
    rotate as nd_rotate,
)


# ------------------------------------------------------------------ #
#  Random 3D Flip
# ------------------------------------------------------------------ #
class RandomFlip3D:
    """
    Randomly flip the volume along each spatial axis independently.

    Parameters
    ----------
    prob : float
        Per-axis probability of flipping.
    """

    def __init__(self, prob: float = 0.5) -> None:
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        image = sample["image"]
        label = sample.get("label")

        for axis in (1, 2, 3):  # spatial axes (skip channel axis 0)
            if np.random.rand() < self.prob:
                image = np.flip(image, axis=axis).copy()
                if label is not None:
                    label = np.flip(label, axis=axis).copy()

        sample["image"] = image
        if label is not None:
            sample["label"] = label
        return sample


# ------------------------------------------------------------------ #
#  Random 3D Rotation (small angles)
# ------------------------------------------------------------------ #
class RandomRotation3D:
    """
    Apply small random rotations around each spatial axis.

    Uses ``scipy.ndimage.rotate`` with bilinear interpolation for the
    image and nearest-neighbour for the label mask.

    Parameters
    ----------
    max_angle : float
        Maximum rotation angle in **degrees** (applied ± uniformly).
    prob : float
        Probability of applying the rotation.
    """

    def __init__(self, max_angle: float = 12.0, prob: float = 0.3) -> None:
        self.max_angle = max_angle
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return sample

        image = sample["image"]
        label = sample.get("label")

        # Pick a random plane to rotate in (axes pair from spatial dims)
        axes_pairs = [(1, 2), (1, 3), (2, 3)]
        axes = axes_pairs[np.random.randint(len(axes_pairs))]
        angle = np.random.uniform(-self.max_angle, self.max_angle)

        # Rotate each channel independently
        rotated_channels = []
        for c in range(image.shape[0]):
            rotated = nd_rotate(
                image[c], angle, axes=(axes[0] - 1, axes[1] - 1),
                reshape=False, order=1, mode="constant", cval=0.0,
            )
            rotated_channels.append(rotated)
        sample["image"] = np.stack(rotated_channels, axis=0).astype(np.float32)

        if label is not None:
            rotated_label = nd_rotate(
                label[0], angle, axes=(axes[0] - 1, axes[1] - 1),
                reshape=False, order=0, mode="constant", cval=0,
            )
            sample["label"] = rotated_label[np.newaxis, ...].astype(np.int64)

        return sample


# ------------------------------------------------------------------ #
#  Random Intensity Shift & Scale
# ------------------------------------------------------------------ #
class RandomIntensityShiftScale:
    """
    Per-channel random intensity affine transform:
        x' = x * (1 + scale) + shift

    Simulates inter-scanner brightness / contrast variation.

    Parameters
    ----------
    shift_range : float
        Maximum absolute shift (sampled uniformly ±).
    scale_range : float
        Maximum relative scale deviation from 1.0.
    prob : float
        Probability of applying.
    """

    def __init__(
        self,
        shift_range: float = 0.1,
        scale_range: float = 0.1,
        prob: float = 0.5,
    ) -> None:
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return sample

        image = sample["image"]
        for c in range(image.shape[0]):
            shift = np.random.uniform(-self.shift_range, self.shift_range)
            scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range)
            image[c] = image[c] * scale + shift

        sample["image"] = image.astype(np.float32)
        return sample


# ------------------------------------------------------------------ #
#  Gaussian Noise
# ------------------------------------------------------------------ #
class RandomGaussianNoise:
    """
    Add Gaussian noise to the image volume.

    Parameters
    ----------
    std : float
        Standard deviation of the noise.
    prob : float
        Probability of applying.
    """

    def __init__(self, std: float = 0.01, prob: float = 0.3) -> None:
        self.std = std
        self.prob = prob

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return sample

        image = sample["image"]
        noise = np.random.normal(0, self.std, size=image.shape).astype(np.float32)
        sample["image"] = (image + noise).astype(np.float32)
        return sample


# ------------------------------------------------------------------ #
#  Elastic Deformation
# ------------------------------------------------------------------ #
class RandomElasticDeformation:
    """
    Apply random elastic deformation to 3D volumes.

    Generates a smooth random displacement field using Gaussian-filtered
    random noise, then applies it with ``scipy.ndimage.map_coordinates``.

    Parameters
    ----------
    sigma : float
        Gaussian smoothing sigma for the displacement field.
        Larger → smoother (more global) deformation.
    magnitude : float
        Maximum displacement magnitude in voxels.
    prob : float
        Probability of applying.
    """

    def __init__(
        self,
        sigma: float = 6.0,
        magnitude: float = 150.0,
        prob: float = 0.2,
    ) -> None:
        self.sigma = sigma
        self.magnitude = magnitude
        self.prob = prob

    def _generate_displacement(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Create a smooth random displacement field."""
        displacement = np.random.randn(*shape).astype(np.float32)
        displacement = gaussian_filter(displacement, sigma=self.sigma)
        displacement = displacement / (displacement.std() + 1e-8) * self.magnitude
        return displacement

    def __call__(self, sample: Dict) -> Dict:
        if np.random.rand() > self.prob:
            return sample

        image = sample["image"]
        label = sample.get("label")
        spatial_shape = image.shape[1:]  # (D, H, W)

        # Generate 3 displacement fields (one per spatial axis)
        coords = np.meshgrid(
            np.arange(spatial_shape[0]),
            np.arange(spatial_shape[1]),
            np.arange(spatial_shape[2]),
            indexing="ij",
        )
        displacements = [
            self._generate_displacement(spatial_shape) for _ in range(3)
        ]
        displaced_coords = [
            coords[i].astype(np.float32) + displacements[i] for i in range(3)
        ]

        # Apply to each image channel (bilinear interpolation)
        deformed_channels = []
        for c in range(image.shape[0]):
            deformed = map_coordinates(
                image[c], displaced_coords, order=1, mode="constant", cval=0.0,
            )
            deformed_channels.append(deformed)
        sample["image"] = np.stack(deformed_channels, axis=0).astype(np.float32)

        # Apply to label (nearest-neighbour)
        if label is not None:
            deformed_label = map_coordinates(
                label[0].astype(np.float32), displaced_coords,
                order=0, mode="constant", cval=0,
            )
            sample["label"] = deformed_label[np.newaxis, ...].astype(np.int64)

        return sample


# ------------------------------------------------------------------ #
#  Compose helper
# ------------------------------------------------------------------ #
class AugmentationCompose:
    """Chain multiple augmentation transforms."""

    def __init__(self, transforms: Sequence) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Dict) -> Dict:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self) -> str:
        names = [t.__class__.__name__ for t in self.transforms]
        return f"AugmentationCompose({names})"


# ------------------------------------------------------------------ #
#  Factory — build from config
# ------------------------------------------------------------------ #
def get_augmentation_pipeline(
    flip_prob: float = 0.5,
    rotate_prob: float = 0.3,
    rotate_max_angle: float = 12.0,
    intensity_shift: float = 0.1,
    intensity_scale: float = 0.1,
    noise_std: float = 0.01,
    elastic_sigma: float = 6.0,
    elastic_magnitude: float = 150.0,
    elastic_prob: float = 0.2,
) -> AugmentationCompose:
    """
    Build the full training augmentation pipeline.

    Parameters match fields in ``config.Config``.  Order matters:
    geometric transforms first, then intensity, then noise.
    """
    return AugmentationCompose([
        RandomFlip3D(prob=flip_prob),
        RandomRotation3D(max_angle=rotate_max_angle, prob=rotate_prob),
        RandomElasticDeformation(
            sigma=elastic_sigma, magnitude=elastic_magnitude, prob=elastic_prob,
        ),
        RandomIntensityShiftScale(
            shift_range=intensity_shift, scale_range=intensity_scale, prob=0.5,
        ),
        RandomGaussianNoise(std=noise_std, prob=0.3),
    ])
