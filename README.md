# 🧠 Brain Tumor Segmentation from MRI

**Automatic brain tumor segmentation using 3D U-Net deep learning on the BraTS Challenge dataset.**

A production-ready end-to-end pipeline: data loading → preprocessing → training → evaluation → inference → interactive web interface.

---

## 📋 Table of Contents

- [Problem Background](#-problem-background)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training Strategy](#-training-strategy)
- [Evaluation Metrics](#-evaluation-metrics)
- [Deployment](#-deployment)
- [Limitations & Future Work](#-limitations--future-work)

---

## 🏥 Problem Background

Brain tumors are among the most aggressive diseases. Accurate segmentation of tumor sub-regions from MRI is critical for:
- **Surgical planning** — defining resection boundaries
- **Treatment monitoring** — tracking tumor growth over time
- **Radiotherapy targeting** — focusing radiation on tumor tissue

Manual segmentation by radiologists is time-consuming (30–60 min per patient) and subject to inter-observer variability. Deep learning offers reproducible, near-instant segmentation.

---

## 📊 Dataset

This project uses the **BraTS (Brain Tumor Segmentation Challenge)** dataset.

### MRI Modalities

| Modality | Description | Clinical Use |
|----------|-------------|--------------|
| **T1** | Anatomical contrast | Grey/white matter distinction |
| **T1ce (Gd)** | Gadolinium contrast-enhanced | Blood–brain barrier breakdown → enhancing tumor |
| **T2** | Fluid-sensitive | Edema detection |
| **FLAIR** | CSF-suppressed T2 | Perilesional edema delineation |

### Tumor Labels

| Label | Region | Description |
|-------|--------|-------------|
| 0 | Background | Healthy tissue |
| 1 | NCR/NET | Necrotic / Non-Enhancing Tumor core |
| 2 | ED | Peritumoral Edema |
| 4 | ET | GD-Enhancing Tumor |

**Evaluation regions** (used in the BraTS Challenge):
- **Enhancing Tumor (ET)**: label 4
- **Tumor Core (TC)**: labels 1 + 4
- **Whole Tumor (WT)**: labels 1 + 2 + 4

### Directory Structure (per subject)
```
BraTS2021_00000/
├── BraTS2021_00000_t1.nii.gz
├── BraTS2021_00000_t1ce.nii.gz
├── BraTS2021_00000_t2.nii.gz
├── BraTS2021_00000_flair.nii.gz
└── BraTS2021_00000_seg.nii.gz
```

---

## 🏗️ Architecture

### Primary: 3D U-Net

A volumetric encoder–decoder network with skip connections:

```
Input (4×128³)
    │
    ├─ Encoder 1:  4 → 32   ─── skip ──┐
    ├─ Encoder 2: 32 → 64   ─── skip ──┤
    ├─ Encoder 3: 64 → 128  ─── skip ──┤
    ├─ Encoder 4: 128 → 256 ─── skip ──┤
    │                                    │
    ├─ Bottleneck: 256 → 512            │
    │                                    │
    ├─ Decoder 4: 512 → 256 ◄── concat ─┤
    ├─ Decoder 3: 256 → 128 ◄── concat ─┤
    ├─ Decoder 2: 128 → 64  ◄── concat ─┤
    ├─ Decoder 1: 64  → 32  ◄── concat ─┘
    │
    └─ Output: 32 → 4 (1×1×1 conv)
```

Each block: `Conv3d → InstanceNorm3d → LeakyReLU → Conv3d → InstanceNorm3d → LeakyReLU`

**Why InstanceNorm?** With batch_size=1–2 in 3D, BatchNorm statistics are unreliable. InstanceNorm normalises per-sample.

### Upgrade: Attention U-Net

Adds **Attention Gates** on skip connections that learn to suppress irrelevant regions (healthy tissue) and focus on tumor structures. Particularly improves small Enhancing Tumor detection with < 5% parameter overhead.

---

## 📁 Project Structure

```
brain-tumor-segmentation/
│
├── config.py                     # Central configuration (all hyperparameters)
├── requirements.txt              # Python dependencies
├── README.md
│
├── data/
│   └── dataset.py                # BraTS Dataset class
│
├── preprocessing/
│   ├── transforms.py             # Normalization, crop/pad pipelines
│   ├── patch_extraction.py       # Random, foreground-biased, sliding-window
│   └── augmentation.py           # 3D flip, rotation, elastic deform, noise
│
├── models/
│   ├── unet3d.py                 # 3D U-Net architecture
│   ├── attention_unet.py         # Attention U-Net with attention gates
│   └── losses.py                 # Dice, Dice+CE, Focal losses
│
├── training/
│   ├── trainer.py                # Training engine (AMP, checkpointing)
│   └── train.py                  # Main training script (CLI)
│
├── evaluation/
│   ├── metrics.py                # Dice, IoU, Sensitivity, Specificity, HD95
│   ├── evaluate.py               # Full evaluation pipeline
│   ├── visualize.py              # Publication-quality visualizations
│   └── gradcam.py                # 3D Grad-CAM explainability
│
├── inference/
│   └── predict.py                # Sliding-window inference + post-processing
│
├── app/
│   ├── streamlit_app.py          # Interactive web interface
│   └── utils.py                  # App helper functions
│
├── checkpoints/                  # Saved model weights
├── logs/                         # TensorBoard logs
├── outputs/                      # Evaluation results, figures
└── notebooks/                    # Exploration notebooks
```

---

## ⚡ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (visit https://pytorch.org for your setup)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### BraTS Dataset

1. Register at [Synapse](https://www.synapse.org/#!Synapse:syn27046444) to download BraTS 2021
2. Extract to `data/raw/BraTS2021_Training/`

---

## 🚀 Usage

### Training

```bash
# Train with default config (3D U-Net)
python training/train.py --data_dir data/raw/BraTS2021_Training

# Train Attention U-Net
python training/train.py --model attention_unet3d --data_dir data/raw/BraTS2021_Training

# Custom hyperparameters
python training/train.py --lr 3e-4 --epochs 500 --batch_size 1 --grad_accum 8

# Resume from checkpoint
python training/train.py --resume checkpoints/best_model.pth

# Monitor training
tensorboard --logdir logs/
```

### Evaluation

```bash
python evaluation/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/raw/BraTS2021_Training \
    --hausdorff
```

### Inference

```bash
python inference/predict.py \
    --checkpoint checkpoints/best_model.pth \
    --input data/raw/BraTS2021_Training/BraTS2021_00000 \
    --output predictions/ \
    --tta  # test-time augmentation
```

### Web Interface

```bash
streamlit run app/streamlit_app.py
```

---

## 🎯 Training Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | AdamW | Decoupled weight decay; stable for medical imaging |
| **Learning Rate** | 1e-4 | Conservative start; reduced by cosine schedule |
| **Scheduler** | Cosine Annealing | Smooth decay avoids sharp LR drops |
| **Loss** | Dice + CE | Dice handles imbalance; CE adds gradient stability |
| **Batch Size** | 2 | GPU memory constraint with 128³ patches |
| **Grad Accumulation** | 4 steps | Effective batch = 8 |
| **AMP** | Enabled | ~50% memory reduction |
| **Patch Size** | 128³ | Balance between context and memory |
| **Augmentation** | Flip, rotation, elastic, noise | Simulates scan variability |
| **Early Stopping** | 50 epochs patience | Prevents overfitting |

---

## 📈 Evaluation Metrics

| Metric | What it Measures | Clinical Relevance |
|--------|-----------------|-------------------|
| **Dice** | Volumetric overlap (0–1) | Primary BraTS metric; measures segmentation quality |
| **IoU** | Intersection / Union | Stricter than Dice; penalises FP more |
| **Sensitivity** | TP / (TP + FN) | Missing tumor is worse than a false alarm |
| **Specificity** | TN / (TN + FP) | Guards against over-segmentation |
| **HD95** | 95th percentile surface distance (mm) | Boundary accuracy for surgical planning |

All metrics are computed per BraTS region (ET, TC, WT).

---

## 🌐 Deployment

The Streamlit app provides:
- **File upload** for 4 NIfTI modalities
- **Real-time segmentation** using the trained model
- **Interactive slice viewer** (axial/sagittal/coronal)
- **Adjustable overlay transparency**
- **Tumor volume statistics** (cm³ per region)

---

## ⚠️ Limitations & Future Work

### Current Limitations
- Trained on BraTS data only — may not generalise to other scanners/protocols
- Patch-based training may miss very large tumors extending beyond 128³
- No uncertainty quantification for clinical confidence

### Future Improvements
- **Swin-UNETR** transformer architecture for global context
- **Federated learning** for multi-institutional training
- **Uncertainty estimation** via MC Dropout or ensembles
- **ONNX export** for production deployment
- **3D volume rendering** with VTK/PyVista
- **DICOM support** for direct clinical integration

---

## 📄 License

This project is the sole property of its author. All rights reserved.

## 🙏 Acknowledgments

- [BraTS Challenge](http://braintumorsegmentation.org/) organisers
- [MONAI](https://monai.io/) medical imaging framework
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) for architectural inspiration
