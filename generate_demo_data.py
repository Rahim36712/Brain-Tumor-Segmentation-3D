"""
Generate FAST Synthetic BraTS-format Demo Data
================================================

Creates SMALL synthetic NIfTI MRI files (64x64x64) for instant
demo. These are small enough for a single forward pass — no
sliding window needed = segmentation in seconds, not minutes.

Usage:
    python generate_demo_data.py
"""

import os
import sys
import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel not installed. Run: pip install nibabel")
    sys.exit(1)


def make_sphere(shape, center, radius):
    """Create a binary sphere mask."""
    grid = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = sum((g - c) ** 2 for g, c in zip(grid, center))
    return (dist <= radius ** 2).astype(np.float32)


def make_ellipsoid(shape, center, radii):
    """Create a binary ellipsoid mask."""
    grid = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist = sum(((g - c) / r) ** 2 for g, c, r in zip(grid, center, radii))
    return (dist <= 1.0).astype(np.float32)


def generate_synthetic_brain(shape=(64, 64, 64)):
    """Generate synthetic brain with tumor at given resolution."""
    D, H, W = shape
    center = (D // 2, H // 2, W // 2)

    # Brain shell
    brain = make_ellipsoid(shape, center, radii=(D * 0.45, H * 0.43, W * 0.43))
    wm = make_ellipsoid(shape, center, radii=(D * 0.32, H * 0.30, W * 0.30))
    gm = brain * (1 - wm)
    csf = make_ellipsoid(shape, center, radii=(D * 0.40, H * 0.38, W * 0.38))

    # Tumor (off-center)
    tc = (D // 2 + 2, H // 2 - 8, W // 2 + 6)
    whole_tumor = make_sphere(shape, tc, radius=10)
    tumor_core = make_sphere(shape, tc, radius=7)
    enhancing = make_sphere(shape, tc, radius=6) - make_sphere(shape, tc, radius=4)
    enhancing = np.clip(enhancing, 0, 1)
    necrotic = make_sphere(shape, tc, radius=4)

    # Segmentation mask (BraTS labels: 0=bg, 1=NCR, 2=Edema, 4=ET)
    seg = np.zeros(shape, dtype=np.uint8)
    edema_region = whole_tumor * (1 - tumor_core)
    seg[edema_region > 0.5] = 2
    seg[necrotic > 0.5] = 1
    seg[enhancing > 0.5] = 4
    seg = seg * brain.astype(np.uint8)

    rng = np.random.RandomState(42)
    noise = 0.02

    # T1
    t1 = wm * 0.85 + gm * 0.55 + csf * 0.15
    t1[seg == 1] = 0.35; t1[seg == 2] = 0.50; t1[seg == 4] = 0.65
    t1 = np.clip((t1 + rng.normal(0, noise, shape)) * brain, 0, 1) * 1000

    # T1ce (enhancing tumor is VERY bright with contrast)
    t1ce = t1.copy()
    t1ce[seg == 4] = 900; t1ce[seg == 1] = 250
    t1ce += rng.normal(0, noise * 500, shape).astype(np.float32)
    t1ce = np.clip(t1ce * brain, 0, 1200)

    # T2 (edema bright)
    t2 = wm * 0.45 + gm * 0.60 + csf * 0.90
    t2[seg == 1] = 0.40; t2[seg == 2] = 0.85; t2[seg == 4] = 0.55
    t2 = np.clip((t2 + rng.normal(0, noise, shape)) * brain, 0, 1) * 1000

    # FLAIR (edema bright, CSF suppressed)
    flair = wm * 0.50 + gm * 0.55 + csf * 0.10
    flair[seg == 1] = 0.35; flair[seg == 2] = 0.90; flair[seg == 4] = 0.70
    flair = np.clip((flair + rng.normal(0, noise, shape)) * brain, 0, 1) * 1000

    return t1, t1ce, t2, flair, seg


def save_nifti(data, filename, affine=None):
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(data, affine), filename)
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"  Saved: {filename} ({size_mb:.1f} MB)")


def main():
    subject_name = "BraTS2021_DEMO_00000"
    output_dir = os.path.join("data", "raw", "BraTS2021_Training", subject_name)
    upload_dir = os.path.join("data", "demo_upload")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(upload_dir, exist_ok=True)

    print("=" * 50)
    print("  Generating FAST Demo Data (64x64x64)")
    print("=" * 50)

    t1, t1ce, t2, flair, seg = generate_synthetic_brain(shape=(64, 64, 64))
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    print("\nSaving NIfTI files:")
    for name, data in [("t1", t1), ("t1ce", t1ce), ("t2", t2), ("flair", flair)]:
        save_nifti(data, os.path.join(output_dir, f"{subject_name}_{name}.nii.gz"), affine)
        save_nifti(data, os.path.join(upload_dir, f"demo_{name}.nii.gz"), affine)
    save_nifti(seg.astype(np.uint8), os.path.join(output_dir, f"{subject_name}_seg.nii.gz"), affine)

    print(f"\nTumor Stats:")
    print(f"  NCR (1): {(seg==1).sum():,} voxels")
    print(f"  Edema (2): {(seg==2).sum():,} voxels")
    print(f"  ET (4): {(seg==4).sum():,} voxels")

    print(f"\n  Upload these from: data/demo_upload/")
    print(f"  DONE! These are small enough for instant segmentation.")


if __name__ == "__main__":
    main()
