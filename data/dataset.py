"""
Brain Tumor Segmentation — BraTS Dataset Loader
================================================

PyTorch Dataset class for the BraTS (Brain Tumor Segmentation Challenge)
dataset.  Supports BraTS 2020 / 2021 folder layouts out of the box.

**BraTS Directory Structure** (per subject)::

    BraTS2021_00000/
    ├── BraTS2021_00000_t1.nii.gz       # T1-weighted
    ├── BraTS2021_00000_t1ce.nii.gz     # T1 contrast-enhanced (Gadolinium)
    ├── BraTS2021_00000_t2.nii.gz       # T2-weighted
    ├── BraTS2021_00000_flair.nii.gz    # FLAIR
    └── BraTS2021_00000_seg.nii.gz      # Segmentation mask

**MRI Modalities:**
- **T1**: Anatomical contrast — grey/white matter distinction
- **T1Gd (T1ce)**: Gadolinium-enhanced — highlights blood–brain barrier
  breakdown in enhancing tumors
- **T2**: Highlights fluid — useful for edema detection
- **FLAIR**: Suppresses CSF signal — best for perilesional edema

**Label Categories (BraTS convention):**
- 0 → Background
- 1 → Necrotic / Non-Enhancing Tumor core (NCR/NET)
- 2 → Peritumoral Edema (ED)
- 4 → GD-Enhancing Tumor (ET)   ← note: label 3 is unused

**Evaluation Regions** (used for metrics):
- Enhancing Tumor (ET): label 4
- Tumor Core (TC): labels 1 + 4
- Whole Tumor (WT): labels 1 + 2 + 4

Usage::

    from data.dataset import BraTSDataset
    ds = BraTSDataset(root="data/raw/BraTS2021_Training",
                      transform=my_transform)
    sample = ds[0]
    # sample["image"]  → (4, D, H, W) float32 tensor
    # sample["label"]  → (1, D, H, W) int64 tensor
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------------ #
#  BraTS Dataset
# ------------------------------------------------------------------ #
class BraTSDataset(Dataset):
    """
    PyTorch Dataset for the BraTS brain tumor segmentation challenge.

    Loads 4 MRI modalities (T1, T1ce, T2, FLAIR) and the segmentation
    mask for each subject.  Returns them as stacked tensors ready for
    3D convolution.

    Parameters
    ----------
    root : str or Path
        Path to the BraTS training directory containing per-subject
        folders (e.g. ``data/raw/BraTS2021_Training``).
    modalities : list[str]
        Ordered list of modality suffixes to load.
        Default: ``["t1", "t1ce", "t2", "flair"]``.
    transform : callable, optional
        A function / MONAI Compose that takes a dict
        ``{"image": ndarray, "label": ndarray}`` and returns the
        transformed dict.
    include_seg : bool
        If ``False`` the segmentation mask is NOT loaded (useful
        during inference on unlabelled data).
    subject_ids : list[str], optional
        Restrict to a specific subset of subject folder names.
        If ``None``, all folders in *root* are used.
    label_map : dict, optional
        Mapping from raw BraTS labels to contiguous indices.
        Default: ``{0: 0, 1: 1, 2: 2, 4: 3}``.
    """

    MODALITY_SUFFIXES = ("t1", "t1ce", "t2", "flair")
    SEG_SUFFIX = "seg"

    def __init__(
        self,
        root: str | Path,
        modalities: Sequence[str] = MODALITY_SUFFIXES,
        transform: Optional[Callable] = None,
        include_seg: bool = True,
        subject_ids: Optional[List[str]] = None,
        label_map: Optional[Dict[int, int]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.modalities = list(modalities)
        self.transform = transform
        self.include_seg = include_seg
        self.label_map = label_map or {0: 0, 1: 1, 2: 2, 4: 3}

        # ----- discover subject folders ----- #
        if subject_ids is not None:
            self.subjects = sorted(subject_ids)
        else:
            self.subjects = sorted(
                d.name
                for d in self.root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )

        if len(self.subjects) == 0:
            raise FileNotFoundError(
                f"No subject folders found in {self.root}.  "
                "Ensure the BraTS dataset is extracted correctly."
            )

    # ------------------------------------------------------------------ #
    #  File-path resolution
    # ------------------------------------------------------------------ #
    def _resolve_nifti(self, subject_dir: Path, suffix: str) -> Path:
        """
        Find the NIfTI file for a given modality suffix inside a
        subject directory.  Handles both ``_t1.nii.gz`` and
        ``_t1.nii`` naming variants.
        """
        patterns = [
            f"*_{suffix}.nii.gz",
            f"*_{suffix}.nii",
        ]
        for pat in patterns:
            matches = list(subject_dir.glob(pat))
            if matches:
                return matches[0]
        raise FileNotFoundError(
            f"Could not find *_{suffix}.nii[.gz] in {subject_dir}"
        )

    # ------------------------------------------------------------------ #
    #  Loading helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_nifti(path: Path) -> np.ndarray:
        """Load a NIfTI file and return a float32 numpy array."""
        img = nib.load(str(path))
        return np.asarray(img.dataobj, dtype=np.float32)

    def _remap_labels(self, seg: np.ndarray) -> np.ndarray:
        """
        Remap BraTS raw labels → contiguous 0-based indices.

        BraTS uses {0, 1, 2, 4}; we remap to {0, 1, 2, 3}.
        """
        out = np.zeros_like(seg, dtype=np.int64)
        for src, dst in self.label_map.items():
            out[seg == src] = dst
        return out

    # ------------------------------------------------------------------ #
    #  Dataset interface
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        subject_name = self.subjects[idx]
        subject_dir = self.root / subject_name

        # ----- load modalities and stack → (C, D, H, W) ----- #
        volumes = []
        for mod in self.modalities:
            path = self._resolve_nifti(subject_dir, mod)
            vol = self._load_nifti(path)
            volumes.append(vol)
        image = np.stack(volumes, axis=0)  # (4, D, H, W)

        sample: Dict[str, np.ndarray | torch.Tensor] = {
            "image": image,
            "subject": subject_name,
        }

        # ----- load segmentation mask ----- #
        if self.include_seg:
            seg_path = self._resolve_nifti(subject_dir, self.SEG_SUFFIX)
            seg = self._load_nifti(seg_path).astype(np.int64)
            seg = self._remap_labels(seg)
            sample["label"] = seg[np.newaxis, ...]  # (1, D, H, W)

        # ----- apply transforms ----- #
        if self.transform is not None:
            sample = self.transform(sample)

        # ----- convert to tensor if not already ----- #
        if isinstance(sample["image"], np.ndarray):
            sample["image"] = torch.from_numpy(sample["image"]).float()
        if self.include_seg and isinstance(sample.get("label"), np.ndarray):
            sample["label"] = torch.from_numpy(sample["label"]).long()

        return sample

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    def get_subject_path(self, idx: int) -> Path:
        """Return the filesystem path for a subject index."""
        return self.root / self.subjects[idx]

    def split(
        self, val_fraction: float = 0.2, seed: int = 42
    ) -> Tuple["BraTSDataset", "BraTSDataset"]:
        """
        Return (train_dataset, val_dataset) by splitting subjects.

        Uses a fixed random seed for reproducibility.
        """
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.subjects))
        n_val = max(1, int(len(indices) * val_fraction))

        val_ids = [self.subjects[i] for i in indices[:n_val]]
        train_ids = [self.subjects[i] for i in indices[n_val:]]

        train_ds = BraTSDataset(
            root=self.root,
            modalities=self.modalities,
            transform=self.transform,
            include_seg=self.include_seg,
            subject_ids=train_ids,
            label_map=self.label_map,
        )
        val_ds = BraTSDataset(
            root=self.root,
            modalities=self.modalities,
            transform=None,  # validation — no augmentation
            include_seg=self.include_seg,
            subject_ids=val_ids,
            label_map=self.label_map,
        )
        return train_ds, val_ds

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root='{self.root}', "
            f"subjects={len(self.subjects)}, "
            f"modalities={self.modalities}, "
            f"include_seg={self.include_seg})"
        )
