"""
Brain Tumor Segmentation — Streamlit Inference Interface
========================================================

Interactive web application for brain tumor segmentation.

Features:
- Upload NIfTI MRI scans (4 modalities)
- Run real-time tumor segmentation
- Interactive axial/sagittal/coronal slice viewer with slider
- Segmentation overlay with adjustable transparency
- Tumor volume statistics
- Download segmentation results

Launch::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import streamlit as st

from app.utils import (
    compute_tumor_stats,
    create_overlay_rgb,
    load_nifti_from_upload,
    normalize_slice_for_display,
)


# ------------------------------------------------------------------ #
#  Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------ #
#  Custom CSS
# ------------------------------------------------------------------ #
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin-bottom: 0.8rem;
    }
    .stSlider > div > div > div {
        background-color: #667eea;
    }
    .region-label {
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        display: inline-block;
        margin: 0.2rem;
    }
    .et-label { background-color: rgba(230,51,51,0.2); color: #e63333; }
    .ed-label { background-color: rgba(237,201,38,0.2); color: #edc926; }
    .ncr-label { background-color: rgba(46,141,214,0.2); color: #2e8dd6; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Sidebar
# ------------------------------------------------------------------ #
def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        model_choice = st.selectbox(
            "Model Architecture",
            ["3D U-Net", "Attention U-Net"],
            index=0,
        )

        overlay_alpha = st.slider(
            "Overlay Transparency",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        )

        view_plane = st.radio(
            "View Plane",
            ["Axial", "Sagittal", "Coronal"],
            index=0,
        )

        st.markdown("---")
        st.markdown("### 📖 Color Legend")
        st.markdown(
            '<span class="region-label et-label">🔴 Enhancing Tumor (ET)</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="region-label ed-label">🟡 Peritumoral Edema (ED)</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="region-label ncr-label">🔵 Necrotic Core (NCR)</span>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "This tool performs automatic brain tumor segmentation "
            "from multimodal MRI using a 3D U-Net deep learning model "
            "trained on the BraTS dataset."
        )

    return model_choice, overlay_alpha, view_plane


# ------------------------------------------------------------------ #
#  Model loading (cached)
# ------------------------------------------------------------------ #
@st.cache_resource
def load_model(model_name: str):
    """Load and cache the segmentation model."""
    import torch
    from config import CFG
    from models.unet3d import UNet3D
    from models.attention_unet import AttentionUNet3D

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "attention" in model_name.lower():
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

    # Try loading checkpoint
    ckpt_path = CFG.checkpoint_dir / "best_model.pth"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        st.sidebar.success(f"✅ Loaded: {ckpt_path.name}")
    else:
        st.sidebar.warning("⚠️ No checkpoint found. Using random weights (demo mode).")

    model = model.to(device)
    model.eval()
    return model, device


# ------------------------------------------------------------------ #
#  Slice viewer
# ------------------------------------------------------------------ #
def get_slice(volume, idx, plane="Axial"):
    """Extract a 2D slice from a 3D volume along the given plane."""
    if plane == "Axial":
        return volume[idx, :, :]
    elif plane == "Sagittal":
        return volume[:, :, idx]
    elif plane == "Coronal":
        return volume[:, idx, :]
    return volume[idx, :, :]


def get_max_idx(volume, plane="Axial"):
    """Get the maximum index for a given plane."""
    if plane == "Axial":
        return volume.shape[0] - 1
    elif plane == "Sagittal":
        return volume.shape[2] - 1
    elif plane == "Coronal":
        return volume.shape[1] - 1
    return volume.shape[0] - 1


# ------------------------------------------------------------------ #
#  Main app
# ------------------------------------------------------------------ #
def main():
    st.markdown('<p class="main-header">🧠 Brain Tumor Segmentation</p>', unsafe_allow_html=True)
    st.markdown("Upload multimodal MRI scans to automatically segment brain tumors using deep learning.")

    model_choice, overlay_alpha, view_plane = render_sidebar()

    # --- File upload --- #
    st.markdown("### 📤 Upload MRI Scans")
    st.markdown("Upload all 4 modalities as NIfTI files (.nii or .nii.gz):")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        t1_file = st.file_uploader("T1", type=["nii", "gz"], key="t1")
    with col2:
        t1ce_file = st.file_uploader("T1ce (Gd)", type=["nii", "gz"], key="t1ce")
    with col3:
        t2_file = st.file_uploader("T2", type=["nii", "gz"], key="t2")
    with col4:
        flair_file = st.file_uploader("FLAIR", type=["nii", "gz"], key="flair")

    uploaded_files = [t1_file, t1ce_file, t2_file, flair_file]
    modality_names = ["T1", "T1ce", "T2", "FLAIR"]

    if not all(uploaded_files):
        st.info("👆 Please upload all 4 MRI modalities to begin segmentation.")
        _render_demo_info()
        return

    # --- Load volumes --- #
    with st.spinner("Loading MRI volumes..."):
        volumes = []
        affine = None
        for f in uploaded_files:
            data, aff = load_nifti_from_upload(f)
            volumes.append(data)
            if affine is None:
                affine = aff

        # Stack into (4, D, H, W)
        image = np.stack(volumes, axis=0)
        st.success(f"✅ Loaded {image.shape[0]} modalities — Volume shape: {image.shape[1:]}")

    # --- Run segmentation --- #
    if st.button("🚀 Run Segmentation", type="primary", use_container_width=True):
        with st.spinner("Running segmentation..."):
            import torch
            import torch.nn.functional as F

            model, device = load_model(model_choice)

            from preprocessing.transforms import ZScoreNormalize
            normalizer = ZScoreNormalize()
            normalized = normalizer({"image": image.copy()})["image"]

            # Fast single-pass inference (resize to model-friendly size)
            original_shape = normalized.shape[1:]  # (D, H, W)
            target_size = (64, 64, 64)  # fast inference size

            # Convert to tensor: (1, C, D, H, W)
            img_tensor = torch.from_numpy(normalized[np.newaxis]).float()

            # Resize to target size for fast processing
            if original_shape != target_size:
                img_tensor = F.interpolate(
                    img_tensor, size=target_size,
                    mode="trilinear", align_corners=False,
                )

            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                output = model(img_tensor)
                if isinstance(output, list):
                    output = output[0]
                pred = torch.argmax(output, dim=1)  # (1, D, H, W)

            # Resize prediction back to original size
            if original_shape != target_size:
                pred_float = pred.unsqueeze(1).float()
                pred_float = F.interpolate(
                    pred_float, size=original_shape,
                    mode="nearest",
                )
                pred = pred_float.squeeze(1).long()

            seg = pred[0].cpu().numpy().astype(np.uint8)

        st.session_state["segmentation"] = seg
        st.session_state["image"] = image
        st.success("✅ Segmentation complete!")

    # --- Display results --- #
    if "segmentation" in st.session_state:
        seg = st.session_state["segmentation"]
        image = st.session_state["image"]

        st.markdown("---")
        st.markdown("### 🔬 Results")

        # Metrics row
        stats = compute_tumor_stats(seg)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Whole Tumor", f"{stats['Whole_Tumor_cm3']:.2f} cm³")
        with col2:
            st.metric("Tumor Core", f"{stats['Tumor_Core_cm3']:.2f} cm³")
        with col3:
            st.metric("Enhancing Tumor", f"{stats['Enhancing_Tumor_cm3']:.2f} cm³")
        with col4:
            total_voxels = np.prod(seg.shape)
            tumor_pct = np.isin(seg, [1, 2, 3]).sum() / total_voxels * 100
            st.metric("Tumor %", f"{tumor_pct:.2f}%")

        # Slice viewer
        st.markdown("### 🖼️ Interactive Slice Viewer")
        max_idx = get_max_idx(seg, view_plane)
        default_idx = _find_best_slice(seg, view_plane)

        slice_idx = st.slider(
            f"{view_plane} Slice",
            min_value=0,
            max_value=max_idx,
            value=default_idx,
            key="slice_slider",
        )

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**FLAIR**")
            flair_slice = get_slice(image[3], slice_idx, view_plane)
            flair_norm = normalize_slice_for_display(flair_slice)
            st.image(flair_norm, use_container_width=True, clamp=True)

        with col_right:
            st.markdown("**Segmentation Overlay**")
            mri_slice = get_slice(image[3], slice_idx, view_plane)
            seg_slice = get_slice(seg, slice_idx, view_plane)
            overlay = create_overlay_rgb(mri_slice, seg_slice, alpha=overlay_alpha)
            st.image(overlay, use_container_width=True)

        # All modalities row
        with st.expander("📊 All Modalities", expanded=False):
            cols = st.columns(4)
            for i, name in enumerate(modality_names):
                with cols[i]:
                    st.markdown(f"**{name}**")
                    s = get_slice(image[i], slice_idx, view_plane)
                    st.image(normalize_slice_for_display(s), use_container_width=True, clamp=True)


def _find_best_slice(seg, plane="Axial"):
    """Find the slice with the most tumor voxels."""
    if plane == "Axial":
        tumor_per_slice = np.isin(seg, [1, 2, 3]).sum(axis=(1, 2))
    elif plane == "Coronal":
        tumor_per_slice = np.isin(seg, [1, 2, 3]).sum(axis=(0, 2))
    else:
        tumor_per_slice = np.isin(seg, [1, 2, 3]).sum(axis=(0, 1))
    return int(np.argmax(tumor_per_slice))


def _render_demo_info():
    """Show information when no files are uploaded."""
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Prepare your MRI data** — You need 4 NIfTI files per subject:
       T1, T1ce (contrast-enhanced), T2, and FLAIR.

    2. **Upload all 4 files** using the upload widgets above.

    3. **Click "Run Segmentation"** to process the scan.

    4. **Explore results** using the interactive slice viewer and
       review tumor volume statistics.

    **Supported formats:** `.nii` and `.nii.gz`
    """)

    st.markdown("### 🏥 About BraTS")
    st.markdown("""
    The **Brain Tumor Segmentation (BraTS) Challenge** provides
    multimodal MRI scans with expert-annotated tumor labels:

    | Label | Region | Color |
    |-------|--------|-------|
    | 1 | Necrotic / Non-Enhancing Tumor (NCR/NET) | 🔵 Blue |
    | 2 | Peritumoral Edema (ED) | 🟡 Yellow |
    | 4 | GD-Enhancing Tumor (ET) | 🔴 Red |

    The model segments these three regions plus background from
    the four MRI modalities simultaneously.
    """)


if __name__ == "__main__":
    main()
