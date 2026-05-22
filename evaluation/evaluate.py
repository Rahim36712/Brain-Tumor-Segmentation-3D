"""
Brain Tumor Segmentation — Evaluation Pipeline
===============================================

Loads a trained model, runs inference on a validation/test set,
computes all BraTS metrics, and saves per-subject and aggregate
results to CSV.

Usage::

    # Evaluate the best model on the validation set
    python evaluation/evaluate.py --checkpoint checkpoints/best_model.pth \\
                                   --data_dir data/raw/BraTS2021_Training

    # Evaluate with Hausdorff distance (slower)
    python evaluation/evaluate.py --checkpoint checkpoints/best_model.pth \\
                                   --data_dir data/raw/BraTS2021_Training \\
                                   --hausdorff
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import CFG
from data.dataset import BraTSDataset
from models.unet3d import UNet3D
from models.attention_unet import AttentionUNet3D
from preprocessing.transforms import get_val_transforms
from evaluation.metrics import compute_brats_metrics, aggregate_metrics
from inference.predict import segment_volume


# ------------------------------------------------------------------ #
#  Evaluation engine
# ------------------------------------------------------------------ #
def evaluate(
    model: torch.nn.Module,
    dataset: BraTSDataset,
    device: str = "cuda",
    patch_size: tuple = (128, 128, 128),
    overlap: float = 0.5,
    include_hausdorff: bool = False,
) -> List[Dict[str, float]]:
    """
    Evaluate a trained model on a dataset using sliding-window
    inference and BraTS metrics.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model.
    dataset : BraTSDataset
        Dataset to evaluate (validation or test).
    device : str
        ``"cuda"`` or ``"cpu"``.
    patch_size : tuple
        Patch dimensions for sliding-window inference.
    overlap : float
        Overlap between adjacent patches.
    include_hausdorff : bool
        Whether to compute HD95 (slower).

    Returns
    -------
    list of dict
        Per-subject metrics.
    """
    model.eval()
    all_metrics: List[Dict[str, float]] = []

    print(f"\nEvaluating {len(dataset)} subjects...")
    for idx in tqdm(range(len(dataset)), desc="Evaluation"):
        sample = dataset[idx]
        image = sample["image"]  # (C, D, H, W) tensor
        label = sample["label"]  # (1, D, H, W) tensor
        subject = sample.get("subject", f"subject_{idx:04d}")

        # Convert to numpy for inference pipeline
        if torch.is_tensor(image):
            image_np = image.numpy()
        else:
            image_np = image

        # Run sliding-window inference
        pred = segment_volume(
            model=model,
            image=image_np,
            patch_size=patch_size,
            overlap=overlap,
            device=device,
            num_classes=CFG.num_classes,
        )

        # Get ground truth
        if torch.is_tensor(label):
            gt = label.squeeze(0).numpy()
        else:
            gt = label.squeeze(0)

        # Compute metrics
        metrics = compute_brats_metrics(
            pred=pred,
            target=gt,
            include_hausdorff=include_hausdorff,
        )
        metrics["subject"] = subject
        all_metrics.append(metrics)

        # Print per-subject summary
        print(
            f"  {subject}: "
            f"ET={metrics['ET_Dice']:.3f}  "
            f"TC={metrics['TC_Dice']:.3f}  "
            f"WT={metrics['WT_Dice']:.3f}  "
            f"Mean={metrics['Mean_Dice']:.3f}"
        )

    return all_metrics


# ------------------------------------------------------------------ #
#  Save results to CSV
# ------------------------------------------------------------------ #
def save_results_csv(
    metrics_list: List[Dict[str, float]],
    output_path: str | Path,
) -> None:
    """Save per-subject metrics to a CSV file."""
    if not metrics_list:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keys = [k for k in metrics_list[0].keys()]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics_list)

    print(f"\nResults saved to: {output_path}")


def print_summary(metrics_list: List[Dict[str, float]]) -> None:
    """Print aggregate metrics table."""
    agg = aggregate_metrics(metrics_list)

    print(f"\n{'='*65}")
    print(f"  Evaluation Summary ({len(metrics_list)} subjects)")
    print(f"{'='*65}")
    print(f"  {'Metric':<25} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*45}")

    priority_keys = [
        "ET_Dice", "TC_Dice", "WT_Dice", "Mean_Dice",
        "ET_IoU", "TC_IoU", "WT_IoU",
        "ET_Sensitivity", "TC_Sensitivity", "WT_Sensitivity",
        "ET_Specificity", "TC_Specificity", "WT_Specificity",
    ]
    for key in priority_keys:
        if key in agg:
            m = agg[key]["mean"]
            s = agg[key]["std"]
            print(f"  {key:<25} {m:>10.4f} {s:>10.4f}")

    # HD95 if present
    hd_keys = [k for k in agg if "HD95" in k]
    if hd_keys:
        print(f"  {'-'*45}")
        for key in hd_keys:
            m = agg[key]["mean"]
            s = agg[key]["std"]
            print(f"  {key:<25} {m:>10.2f} {s:>10.2f}")

    print(f"{'='*65}\n")


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate brain tumor segmentation model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to BraTS data directory")
    parser.add_argument("--output", type=str, default="outputs/evaluation_results.csv", help="CSV output path")
    parser.add_argument("--model", type=str, default="unet3d", choices=["unet3d", "attention_unet3d"])
    parser.add_argument("--overlap", type=float, default=0.5, help="Sliding window overlap")
    parser.add_argument("--hausdorff", action="store_true", help="Include HD95 metric")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    if args.model == "attention_unet3d":
        model = AttentionUNet3D(
            in_channels=CFG.in_channels,
            num_classes=CFG.num_classes,
            base_filters=CFG.base_filters,
        )
    else:
        model = UNet3D(
            in_channels=CFG.in_channels,
            num_classes=CFG.num_classes,
            base_filters=CFG.base_filters,
        )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Best Dice: {checkpoint.get('best_dice', 'N/A')}")

    # Load dataset
    data_dir = args.data_dir or str(CFG.raw_data_dir)
    transforms = get_val_transforms(crop_size=None)  # no cropping for eval
    dataset = BraTSDataset(
        root=data_dir,
        transform=transforms,
        include_seg=True,
    )

    # Run evaluation
    metrics_list = evaluate(
        model=model,
        dataset=dataset,
        device=device,
        patch_size=CFG.crop_size,
        overlap=args.overlap,
        include_hausdorff=args.hausdorff,
    )

    # Save and display results
    save_results_csv(metrics_list, args.output)
    print_summary(metrics_list)


if __name__ == "__main__":
    main()
