"""
Brain Tumor Segmentation — Inference Pipeline
==============================================

Full-volume inference using sliding-window with overlap, optional
test-time augmentation, and post-processing.

Pipeline:
1. Load NIfTI MRI (4 modalities)
2. Preprocess (normalize, optionally resample)
3. Sliding-window patch extraction with configurable overlap
4. Model forward pass on each patch (with AMP)
5. Reassemble patches via overlap-averaging
6. Argmax → integer label map
7. Post-process: connected component analysis, small-region removal
8. Remap labels back to BraTS convention
9. Save as NIfTI

Usage::

    # From Python
    from inference.predict import run_inference
    pred_mask = run_inference(model, "path/to/subject_folder")

    # From CLI
    python inference/predict.py --checkpoint best_model.pth \\
                                 --input path/to/subject \\
                                 --output predictions/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.patch_extraction import SlidingWindowSampler


# ------------------------------------------------------------------ #
#  Core inference function
# ------------------------------------------------------------------ #
@torch.no_grad()
def segment_volume(
    model: nn.Module,
    image: np.ndarray,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    device: str = "cuda",
    num_classes: int = 4,
    use_tta: bool = False,
) -> np.ndarray:
    """
    Run sliding-window inference on a full MRI volume.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model.
    image : ndarray
        Preprocessed MRI volume, shape ``(C, D, H, W)``.
    patch_size : tuple
        Patch dimensions for sliding window.
    overlap : float
        Fractional overlap between patches (0.0–0.9).
    device : str
        ``"cuda"`` or ``"cpu"``.
    num_classes : int
        Number of output classes.
    use_tta : bool
        Enable test-time augmentation (flip averaging).

    Returns
    -------
    ndarray
        Predicted segmentation mask, shape ``(D, H, W)``
        with integer labels.
    """
    model.eval()
    sampler = SlidingWindowSampler(patch_size=patch_size, overlap=overlap)

    # Collect patch predictions
    patch_results = []

    for patch_np, coords in sampler(image):
        # (C, pD, pH, pW) → (1, C, pD, pH, pW) tensor
        patch_tensor = torch.from_numpy(patch_np[np.newaxis]).float().to(device)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            output = model(patch_tensor)
            if isinstance(output, list):
                output = output[0]
            probs = F.softmax(output, dim=1)

            # Test-time augmentation: average with flipped predictions
            if use_tta:
                for flip_axis in [2, 3, 4]:  # D, H, W
                    flipped = torch.flip(patch_tensor, dims=[flip_axis])
                    out_flip = model(flipped)
                    if isinstance(out_flip, list):
                        out_flip = out_flip[0]
                    probs += F.softmax(
                        torch.flip(out_flip, dims=[flip_axis]), dim=1
                    )
                probs /= 4.0  # original + 3 flips

        pred_np = probs[0].cpu().numpy()  # (C, pD, pH, pW)
        patch_results.append((pred_np, coords))

    # Reassemble
    spatial_shape = image.shape[1:]
    full_pred = sampler.reassemble(patch_results, spatial_shape, num_classes)

    # Argmax → class labels
    seg = np.argmax(full_pred, axis=0)  # (D, H, W)

    return seg


# ------------------------------------------------------------------ #
#  Post-processing
# ------------------------------------------------------------------ #
def post_process(
    segmentation: np.ndarray,
    min_component_size: int = 100,
) -> np.ndarray:
    """
    Post-process a predicted segmentation mask.

    1. Remove small connected components (likely false positives)
    2. Fill holes in large components

    Parameters
    ----------
    segmentation : ndarray
        Integer label map (D, H, W).
    min_component_size : int
        Components smaller than this (in voxels) are removed.

    Returns
    -------
    ndarray
        Cleaned segmentation mask.
    """
    from scipy.ndimage import label as nd_label, binary_fill_holes

    cleaned = np.zeros_like(segmentation)

    for class_id in [1, 2, 3]:  # skip background
        binary_mask = (segmentation == class_id).astype(np.uint8)

        if not binary_mask.any():
            continue

        # Connected component analysis
        labelled, num_components = nd_label(binary_mask)

        for comp_id in range(1, num_components + 1):
            component = (labelled == comp_id)
            if component.sum() >= min_component_size:
                # Fill holes in this component
                filled = binary_fill_holes(component)
                cleaned[filled] = class_id

    return cleaned


def remap_to_brats(segmentation: np.ndarray) -> np.ndarray:
    """
    Remap model output labels {0,1,2,3} back to BraTS labels {0,1,2,4}.
    """
    output = np.zeros_like(segmentation)
    output[segmentation == 1] = 1
    output[segmentation == 2] = 2
    output[segmentation == 3] = 4
    return output


# ------------------------------------------------------------------ #
#  Full inference pipeline
# ------------------------------------------------------------------ #
def run_inference(
    model: nn.Module,
    subject_dir: str | Path,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    device: str = "cuda",
    num_classes: int = 4,
    use_tta: bool = False,
    min_component_size: int = 100,
    save_dir: Optional[str | Path] = None,
) -> np.ndarray:
    """
    End-to-end inference on a single subject.

    Loads NIfTI files, preprocesses, runs model, post-processes,
    and optionally saves the result.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    subject_dir : path
        Path to subject folder with NIfTI files.
    save_dir : path, optional
        If given, save the prediction as a NIfTI file.

    Returns
    -------
    ndarray
        Predicted segmentation (D, H, W) with BraTS labels.
    """
    import nibabel as nib
    from data.dataset import BraTSDataset
    from preprocessing.transforms import get_val_transforms

    subject_dir = Path(subject_dir)

    # Load via dataset (handles modality discovery)
    ds = BraTSDataset(
        root=subject_dir.parent,
        subject_ids=[subject_dir.name],
        transform=get_val_transforms(),
        include_seg=False,
    )
    sample = ds[0]
    image = sample["image"].numpy() if torch.is_tensor(sample["image"]) else sample["image"]

    # Segment
    seg = segment_volume(
        model=model,
        image=image,
        patch_size=patch_size,
        overlap=overlap,
        device=device,
        num_classes=num_classes,
        use_tta=use_tta,
    )

    # Post-process
    seg = post_process(seg, min_component_size=min_component_size)

    # Remap to BraTS labels
    seg_brats = remap_to_brats(seg)

    # Save
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Use the reference image's affine for correct spatial orientation
        ref_nifti = list(subject_dir.glob("*_flair.nii.gz"))
        if not ref_nifti:
            ref_nifti = list(subject_dir.glob("*_flair.nii"))
        if ref_nifti:
            ref = nib.load(str(ref_nifti[0]))
            affine = ref.affine
        else:
            affine = np.eye(4)

        out_img = nib.Nifti1Image(seg_brats.astype(np.uint8), affine)
        out_path = save_dir / f"{subject_dir.name}_pred.nii.gz"
        nib.save(out_img, str(out_path))
        print(f"Saved prediction: {out_path}")

    return seg_brats


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run brain tumor segmentation inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Subject folder path")
    parser.add_argument("--output", type=str, default="predictions/", help="Output directory")
    parser.add_argument("--model", type=str, default="unet3d", choices=["unet3d", "attention_unet3d"])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from config import CFG
    from models.unet3d import UNet3D
    from models.attention_unet import AttentionUNet3D

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "attention_unet3d":
        model = AttentionUNet3D(in_channels=CFG.in_channels, num_classes=CFG.num_classes, base_filters=CFG.base_filters)
    else:
        model = UNet3D(in_channels=CFG.in_channels, num_classes=CFG.num_classes, base_filters=CFG.base_filters)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    pred = run_inference(
        model=model,
        subject_dir=args.input,
        overlap=args.overlap,
        device=device,
        use_tta=args.tta,
        save_dir=args.output,
    )
    print(f"Segmentation complete. Unique labels: {np.unique(pred)}")


if __name__ == "__main__":
    main()
