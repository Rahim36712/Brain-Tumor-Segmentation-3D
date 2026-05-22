"""
Brain Tumor Segmentation — Main Training Script
================================================

Entry point for training the 3D U-Net or Attention U-Net on BraTS
data.  Fully configurable via CLI arguments or by editing
``config.py``.

**Training Strategy:**

- **Optimizer: AdamW** — Adam with decoupled weight decay.  Proven
  effective for medical imaging; weight decay prevents overfitting
  on the relatively small BraTS dataset (~1200 subjects).

- **Learning Rate Schedule: Cosine Annealing with Warm Restarts** —
  smooth LR decay avoids sharp drops that can destabilise training.
  The warm restarts allow the model to "escape" local minima.
  Alternative: ReduceLROnPlateau (monitors val Dice directly).

- **Batch Size: 2** — limited by GPU memory with 128³ patches and
  4 input channels.  Gradient accumulation (4 steps) gives an
  effective batch size of 8.

- **Loss: Dice + Cross-Entropy** — Dice handles class imbalance,
  CE adds stable gradients (see ``models/losses.py``).

Usage::

    # Train with default config
    python training/train.py

    # Train Attention U-Net with custom LR
    python training/train.py --model attention_unet3d --lr 3e-4

    # Resume from checkpoint
    python training/train.py --resume checkpoints/best_model.pth

    # Override any config field
    python training/train.py --epochs 500 --batch_size 1 --grad_accum 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CFG
from data.dataset import BraTSDataset
from models.unet3d import UNet3D
from models.attention_unet import AttentionUNet3D
from models.losses import get_loss_function
from preprocessing.transforms import get_train_transforms, get_val_transforms
from preprocessing.augmentation import get_augmentation_pipeline
from preprocessing.patch_extraction import ForegroundBiasedSampler
from training.trainer import Trainer


# ------------------------------------------------------------------ #
#  Reproducibility
# ------------------------------------------------------------------ #
def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ------------------------------------------------------------------ #
#  Dataset factory with patch-based collation
# ------------------------------------------------------------------ #
class PatchDataset(torch.utils.data.Dataset):
    """
    Wraps BraTSDataset to return patches instead of full volumes.

    Each ``__getitem__`` call loads a full volume, applies transforms,
    and extracts ``num_patches`` foreground-biased patches.

    This avoids loading all patches into memory — patches are
    generated on-the-fly during training.
    """

    def __init__(
        self,
        base_dataset: BraTSDataset,
        patch_size: tuple = (128, 128, 128),
        num_patches: int = 4,
        fg_ratio: float = 0.7,
        augmentation=None,
    ) -> None:
        self.base = base_dataset
        self.sampler = ForegroundBiasedSampler(
            patch_size=patch_size,
            num_patches=num_patches,
            fg_ratio=fg_ratio,
        )
        self.augmentation = augmentation
        self.num_patches = num_patches

    def __len__(self) -> int:
        return len(self.base) * self.num_patches

    def __getitem__(self, idx: int):
        # Map flat index → (subject, patch_within_subject)
        subject_idx = idx // self.num_patches

        # Load full volume with preprocessing transforms
        sample = self.base[subject_idx]

        # Convert to numpy for patch extraction
        image_np = sample["image"].numpy() if torch.is_tensor(sample["image"]) else sample["image"]
        label_np = sample["label"].numpy() if torch.is_tensor(sample["label"]) else sample["label"]

        # Extract a single random patch
        patches = self.sampler({"image": image_np, "label": label_np})
        patch = patches[0]  # take only one patch per call

        # Apply augmentation
        if self.augmentation is not None:
            patch = self.augmentation(patch)

        # Convert to tensors
        image_tensor = torch.from_numpy(patch["image"].copy()).float()
        label_tensor = torch.from_numpy(patch["label"].copy()).long()

        return {"image": image_tensor, "label": label_tensor}


# ------------------------------------------------------------------ #
#  Model factory
# ------------------------------------------------------------------ #
def build_model(cfg) -> torch.nn.Module:
    """Build model from config."""
    model_name = cfg.model_name.lower()

    kwargs = dict(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_filters=cfg.base_filters,
        dropout=cfg.dropout_rate,
    )

    if model_name in ("unet3d", "unet"):
        model = UNet3D(**kwargs)
    elif model_name in ("attention_unet3d", "attention_unet", "attn_unet"):
        model = AttentionUNet3D(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Model: {model.__class__.__name__}")
    print(f"  Parameters: {model.count_parameters():,}")
    return model


# ------------------------------------------------------------------ #
#  Optimizer & Scheduler factory
# ------------------------------------------------------------------ #
def build_optimizer(model: torch.nn.Module, cfg):
    """Build AdamW optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )


def build_scheduler(optimizer, cfg):
    """Build LR scheduler from config."""
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.cosine_T_max,
            eta_min=cfg.cosine_eta_min,
        )
    elif cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=cfg.plateau_patience,
            factor=cfg.plateau_factor,
        )
    else:
        print(f"  No scheduler ('{cfg.scheduler}' not recognised)")
        return None


# ------------------------------------------------------------------ #
#  CLI argument parser
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3D U-Net for Brain Tumor Segmentation",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to BraTS training data directory",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["unet3d", "attention_unet3d"],
        help="Model architecture",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main() -> None:
    args = parse_args()

    # --- Load config --- #
    cfg = CFG
    if args.config:
        cfg = cfg.from_yaml(args.config)

    # --- Apply CLI overrides --- #
    if args.data_dir:
        cfg.raw_data_dir = Path(args.data_dir)
    if args.model:
        cfg.model_name = args.model
    if args.lr:
        cfg.learning_rate = args.lr
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.grad_accum:
        cfg.grad_accum_steps = args.grad_accum
    if args.seed:
        cfg.seed = args.seed

    cfg.ensure_dirs()
    seed_everything(cfg.seed)

    # --- Device --- #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # --- Dataset --- #
    print(f"\nLoading BraTS data from: {cfg.raw_data_dir}")
    train_transforms = get_train_transforms(
        crop_size=cfg.crop_size,
        normalize=cfg.normalize_method,
    )
    val_transforms = get_val_transforms(
        crop_size=cfg.crop_size,
        normalize=cfg.normalize_method,
    )

    full_dataset = BraTSDataset(
        root=cfg.raw_data_dir,
        modalities=cfg.modalities,
        transform=train_transforms,
    )
    train_base, val_base = full_dataset.split(
        val_fraction=args.val_fraction,
        seed=cfg.seed,
    )
    # Override val transforms (no augmentation)
    val_base.transform = val_transforms

    print(f"  Training subjects:   {len(train_base)}")
    print(f"  Validation subjects: {len(val_base)}")

    # --- Patch-based training dataset --- #
    augmentation = get_augmentation_pipeline(
        flip_prob=cfg.aug_prob_flip,
        rotate_prob=cfg.aug_prob_rotate,
        rotate_max_angle=np.degrees(cfg.aug_rotate_range[0]),
        intensity_shift=cfg.aug_intensity_shift,
        intensity_scale=cfg.aug_intensity_scale,
        noise_std=cfg.aug_gaussian_noise_std,
        elastic_sigma=cfg.aug_elastic_sigma_range[0],
        elastic_magnitude=cfg.aug_elastic_magnitude_range[0],
    )

    train_dataset = PatchDataset(
        base_dataset=train_base,
        patch_size=cfg.crop_size,
        num_patches=4,
        fg_ratio=0.7,
        augmentation=augmentation,
    )
    val_dataset = PatchDataset(
        base_dataset=val_base,
        patch_size=cfg.crop_size,
        num_patches=2,
        fg_ratio=0.5,
        augmentation=None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    # --- Model --- #
    model = build_model(cfg)

    # --- Loss / Optimiser / Scheduler --- #
    criterion = get_loss_function(
        name=cfg.loss_fn,
        num_classes=cfg.num_classes,
        smooth=cfg.dice_smooth,
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # --- Trainer --- #
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=cfg,
        device=device,
    )

    # --- Resume --- #
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    # --- Train --- #
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=start_epoch,
    )

    print("Training history saved. Use TensorBoard to visualise:")
    print(f"  tensorboard --logdir {cfg.log_dir}")


if __name__ == "__main__":
    main()
