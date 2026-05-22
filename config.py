"""
Brain Tumor Segmentation — Central Configuration
=================================================

Single source of truth for all project hyperparameters, paths, and
training settings.  Import this module anywhere in the project:

    from config import CFG

Override individual fields as needed:

    CFG.learning_rate = 3e-4
    CFG.batch_size = 4

Or use ``CFG.to_dict()`` / ``CFG.from_yaml()`` for serialisation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # YAML serialisation disabled until `pip install pyyaml`


# ------------------------------------------------------------------ #
#  Project-root helper (resolves regardless of the working directory)
# ------------------------------------------------------------------ #
_PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    """Centralised, type-hinted project configuration."""

    # ============================================================== #
    #  Paths
    # ============================================================== #
    project_root: Path = _PROJECT_ROOT
    data_root: Path = _PROJECT_ROOT / "data"
    raw_data_dir: Path = _PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = _PROJECT_ROOT / "data" / "processed"
    checkpoint_dir: Path = _PROJECT_ROOT / "checkpoints"
    log_dir: Path = _PROJECT_ROOT / "logs"
    output_dir: Path = _PROJECT_ROOT / "outputs"

    # ============================================================== #
    #  Dataset
    # ============================================================== #
    brats_year: str = "2021"
    modalities: List[str] = field(
        default_factory=lambda: ["t1", "t1ce", "t2", "flair"]
    )
    num_classes: int = 4          # background + ET + TC + WT
    class_names: List[str] = field(
        default_factory=lambda: [
            "Background",
            "Necrotic / Non-Enhancing Tumor (NCR/NET)",
            "Peritumoral Edema (ED)",
            "GD-Enhancing Tumor (ET)",
        ]
    )
    # BraTS label mapping:  0 → background, 1 → NCR/NET, 2 → ED, 4 → ET
    label_map: dict = field(
        default_factory=lambda: {0: 0, 1: 1, 2: 2, 4: 3}
    )

    # ============================================================== #
    #  Preprocessing
    # ============================================================== #
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    crop_size: Tuple[int, int, int] = (128, 128, 128)
    normalize_method: str = "zscore"   # "zscore" | "minmax"

    # ============================================================== #
    #  Augmentation
    # ============================================================== #
    aug_prob_flip: float = 0.5
    aug_prob_rotate: float = 0.3
    aug_rotate_range: Tuple[float, ...] = (0.2, 0.2, 0.2)  # radians
    aug_intensity_shift: float = 0.1
    aug_intensity_scale: float = 0.1
    aug_elastic_sigma_range: Tuple[float, float] = (5.0, 8.0)
    aug_elastic_magnitude_range: Tuple[float, float] = (100.0, 200.0)
    aug_gaussian_noise_std: float = 0.01

    # ============================================================== #
    #  Model
    # ============================================================== #
    model_name: str = "unet3d"        # "unet3d" | "attention_unet3d"
    in_channels: int = 4              # 4 MRI modalities
    base_filters: int = 32            # first conv layer output channels
    dropout_rate: float = 0.2

    # ============================================================== #
    #  Training
    # ============================================================== #
    seed: int = 42
    epochs: int = 300
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True

    # Optimiser
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler
    scheduler: str = "cosine"         # "cosine" | "plateau"
    cosine_T_max: int = 300
    cosine_eta_min: float = 1e-7
    plateau_patience: int = 20
    plateau_factor: float = 0.5

    # Loss
    loss_fn: str = "dice_ce"          # "dice" | "dice_ce" | "focal"
    dice_smooth: float = 1e-5

    # Mixed Precision & Gradient Accumulation
    use_amp: bool = True
    grad_accum_steps: int = 4         # effective batch = batch_size * grad_accum_steps

    # Checkpointing
    save_every_n_epochs: int = 25
    early_stop_patience: int = 50

    # ============================================================== #
    #  Inference
    # ============================================================== #
    sliding_window_overlap: float = 0.5
    test_time_augmentation: bool = False
    post_process_min_size: int = 100   # remove connected components < N voxels

    # ============================================================== #
    #  Visualisation / App
    # ============================================================== #
    figure_dpi: int = 150
    cmap_tumor: str = "Set1"

    # ============================================================== #
    #  Serialisation helpers
    # ============================================================== #
    def to_dict(self) -> dict:
        """Return a JSON-safe dictionary representation."""
        d = asdict(self)
        # Convert Path objects to strings for YAML / JSON compat
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    def save_yaml(self, path: str | Path) -> None:
        """Persist config to a YAML file."""
        if yaml is None:
            raise ImportError("pyyaml is required: pip install pyyaml")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from a YAML file, merging with defaults."""
        if yaml is None:
            raise ImportError("pyyaml is required: pip install pyyaml")
        with open(path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        cfg = cls()
        for k, v in overrides.items():
            if hasattr(cfg, k):
                # Convert string paths back to Path objects
                field_type = type(getattr(cfg, k))
                if field_type is Path:
                    v = Path(v)
                setattr(cfg, k, v)
        return cfg

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't already exist."""
        for d in [
            self.checkpoint_dir,
            self.log_dir,
            self.output_dir,
            self.processed_data_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for k, v in self.to_dict().items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Global singleton — import everywhere as `from config import CFG`
# ------------------------------------------------------------------ #
CFG = Config()
