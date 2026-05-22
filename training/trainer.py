"""
Brain Tumor Segmentation — Training Engine
===========================================

Reusable training loop with production-grade features:

- **Automatic Mixed Precision (AMP)** — halves GPU memory by using
  float16 for forward/backward while keeping float32 master weights.
- **Gradient Accumulation** — simulates larger batch sizes without
  extra memory.  Effective batch = ``batch_size × grad_accum_steps``.
- **Checkpoint Management** — saves best model (by val Dice), periodic
  snapshots, and full resume state (model + optimizer + scheduler +
  scaler + epoch).
- **TensorBoard Logging** — tracks loss curves, learning rate, and
  validation Dice per region.
- **Early Stopping** — halts training when validation Dice stops
  improving for ``patience`` epochs.
- **Rich Console Logging** — pretty progress bars and epoch summaries.

GPU Memory Optimisation Notes
-----------------------------
3D medical volumes are memory-hungry.  Strategies employed:

1. **Patch-based training** (128³) instead of full volumes (240³).
2. **AMP** — ~50% memory reduction with minimal accuracy loss.
3. **Gradient accumulation** — batch_size=2 with 4 accumulation steps
   gives an effective batch of 8 without holding 8 volumes in memory.
4. **InstanceNorm** — no running stats → less memory than BatchNorm.
5. **Gradient checkpointing** — can be enabled on the model for an
   additional ~30% memory saving (trades compute for memory).

Usage::

    from training.trainer import Trainer

    trainer = Trainer(model, optimizer, scheduler, criterion, cfg)
    trainer.fit(train_loader, val_loader)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ------------------------------------------------------------------ #
#  Trainer
# ------------------------------------------------------------------ #
class Trainer:
    """
    End-to-end training engine for 3D segmentation models.

    Parameters
    ----------
    model : nn.Module
        Segmentation model (UNet3D or AttentionUNet3D).
    optimizer : torch.optim.Optimizer
        Configured optimizer (AdamW).
    scheduler : LR scheduler or None
        Learning-rate scheduler.
    criterion : nn.Module
        Loss function (DiceLoss / DiceCELoss / FocalLoss).
    config : object
        Configuration object with training hyperparameters.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        criterion: nn.Module,
        config: Any,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.cfg = config
        self.device = device

        # AMP scaler
        self.use_amp = getattr(config, "use_amp", True) and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Gradient accumulation
        self.grad_accum_steps = getattr(config, "grad_accum_steps", 4)

        # Checkpointing
        self.checkpoint_dir = Path(getattr(config, "checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_dir = Path(getattr(config, "log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer = None

        # Tracking
        self.best_dice = 0.0
        self.epochs_without_improvement = 0
        self.early_stop_patience = getattr(config, "early_stop_patience", 50)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "lr": [],
        }

    # ------------------------------------------------------------------ #
    #  TensorBoard
    # ------------------------------------------------------------------ #
    @property
    def tb_writer(self):
        """Lazily initialise TensorBoard writer."""
        if self._tb_writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                self._tb_writer = None
        return self._tb_writer

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)

    # ------------------------------------------------------------------ #
    #  Training loop
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 0,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Run the full training loop.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader yielding ``{"image": ..., "label": ...}``.
        val_loader : DataLoader
            Validation data loader.
        start_epoch : int
            Epoch to resume from (for checkpoint resume).
        num_epochs : int, optional
            Override ``config.epochs``.

        Returns
        -------
        dict
            Training history with loss/dice per epoch.
        """
        total_epochs = num_epochs or getattr(self.cfg, "epochs", 300)
        save_every = getattr(self.cfg, "save_every_n_epochs", 25)

        print(f"\n{'='*60}")
        print(f"  Training Configuration")
        print(f"{'='*60}")
        print(f"  Model:           {self.model.__class__.__name__}")
        print(f"  Device:          {self.device}")
        print(f"  Epochs:          {total_epochs}")
        print(f"  Batch size:      {train_loader.batch_size}")
        print(f"  Grad accum:      {self.grad_accum_steps}")
        print(f"  Effective batch: {(train_loader.batch_size or 1) * self.grad_accum_steps}")
        print(f"  AMP:             {self.use_amp}")
        print(f"  Loss:            {self.criterion.__class__.__name__}")
        print(f"{'='*60}\n")

        for epoch in range(start_epoch, total_epochs):
            t0 = time.time()

            # --- Train --- #
            train_loss = self._train_one_epoch(train_loader, epoch)

            # --- Validate --- #
            val_loss, val_dice = self._validate(val_loader, epoch)

            # --- LR Scheduler --- #
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_dice)
                else:
                    self.scheduler.step()

            # --- Logging --- #
            elapsed = time.time() - t0
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_dice)
            self.history["lr"].append(current_lr)

            self._log_scalar("Loss/train", train_loss, epoch)
            self._log_scalar("Loss/val", val_loss, epoch)
            self._log_scalar("Dice/val", val_dice, epoch)
            self._log_scalar("LR", current_lr, epoch)

            print(
                f"Epoch {epoch+1:03d}/{total_epochs} │ "
                f"Train Loss: {train_loss:.4f} │ "
                f"Val Loss: {val_loss:.4f} │ "
                f"Val Dice: {val_dice:.4f} │ "
                f"LR: {current_lr:.2e} │ "
                f"Time: {elapsed:.1f}s"
            )

            # --- Checkpointing --- #
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, is_best=True)
                print(f"  ★ New best Dice: {val_dice:.4f}")
            else:
                self.epochs_without_improvement += 1

            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch, is_best=False)

            # --- Early stopping --- #
            if self.epochs_without_improvement >= self.early_stop_patience:
                print(
                    f"\n⚠ Early stopping triggered after {epoch+1} epochs "
                    f"(no improvement for {self.early_stop_patience} epochs)."
                )
                break

        if self.tb_writer is not None:
            self.tb_writer.close()

        print(f"\n{'='*60}")
        print(f"  Training Complete — Best Val Dice: {self.best_dice:.4f}")
        print(f"{'='*60}\n")

        return self.history

    # ------------------------------------------------------------------ #
    #  Single epoch train
    # ------------------------------------------------------------------ #
    def _train_one_epoch(
        self, loader: DataLoader, epoch: int
    ) -> float:
        """Run one training epoch with gradient accumulation and AMP."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

        for step, batch in enumerate(pbar):
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                output = self.model(image)

                # Handle deep supervision (list of outputs)
                if isinstance(output, list):
                    loss = sum(
                        self.criterion(o, label) for o in output
                    ) / len(output)
                else:
                    loss = self.criterion(output, label)

                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            running_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            pbar.set_postfix({"loss": f"{running_loss / num_batches:.4f}"})

        return running_loss / max(num_batches, 1)

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _validate(
        self, loader: DataLoader, epoch: int
    ) -> Tuple[float, float]:
        """Run validation and compute loss + mean Dice."""
        self.model.eval()
        running_loss = 0.0
        dice_scores = []
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]  ", leave=False)

        for batch in pbar:
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                output = self.model(image)
                if isinstance(output, list):
                    output = output[0]  # use full-res output
                loss = self.criterion(output, label)

            running_loss += loss.item()
            num_batches += 1

            # Compute Dice per sample
            pred_classes = torch.argmax(output, dim=1)  # (B, D, H, W)
            for b in range(pred_classes.shape[0]):
                dice = self._compute_val_dice(
                    pred_classes[b].cpu().numpy(),
                    label[b, 0].cpu().numpy(),
                )
                dice_scores.append(dice)

        avg_loss = running_loss / max(num_batches, 1)
        avg_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        return avg_loss, avg_dice

    @staticmethod
    def _compute_val_dice(pred: np.ndarray, target: np.ndarray) -> float:
        """Mean Dice over BraTS regions (ET, TC, WT)."""
        smooth = 1e-5
        dices = []

        # ET: label 3
        p_et = (pred == 3).astype(np.float32)
        g_et = (target == 3).astype(np.float32)
        dices.append(
            (2 * (p_et * g_et).sum() + smooth) / (p_et.sum() + g_et.sum() + smooth)
        )

        # TC: labels {1, 3}
        p_tc = np.isin(pred, [1, 3]).astype(np.float32)
        g_tc = np.isin(target, [1, 3]).astype(np.float32)
        dices.append(
            (2 * (p_tc * g_tc).sum() + smooth) / (p_tc.sum() + g_tc.sum() + smooth)
        )

        # WT: labels {1, 2, 3}
        p_wt = np.isin(pred, [1, 2, 3]).astype(np.float32)
        g_wt = np.isin(target, [1, 2, 3]).astype(np.float32)
        dices.append(
            (2 * (p_wt * g_wt).sum() + smooth) / (p_wt.sum() + g_wt.sum() + smooth)
        )

        return float(np.mean(dices))

    # ------------------------------------------------------------------ #
    #  Checkpointing
    # ------------------------------------------------------------------ #
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model + training state for resumption."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_dice": self.best_dice,
            "history": self.history,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"

        torch.save(state, path)

    def load_checkpoint(self, path: str | Path) -> int:
        """
        Load a checkpoint and restore full training state.

        Returns the epoch to resume from.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_dice = checkpoint.get("best_dice", 0.0)
        self.history = checkpoint.get("history", self.history)

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        resume_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {resume_epoch} (best Dice: {self.best_dice:.4f})")
        return resume_epoch
