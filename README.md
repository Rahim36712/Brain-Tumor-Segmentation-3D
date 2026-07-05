# 🧠 Multimodal 3D Brain Tumor Segmentation from MRI

Automatic brain tumor segmentation from 3D MRI scans using standard 3D U-Net and Attention 3D U-Net deep learning models on the BraTS (Brain Tumor Segmentation) dataset.

This repository implements a production-ready, modular deep learning pipeline:
`Data Loading ➔ Preprocessing ➔ Augmentation ➔ Patch Extraction ➔ Training ➔ Evaluation ➔ Explainability ➔ Web Deployment`

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Technology Stack](#%EF%B8%8F-technology-stack)
3. [System Architecture](#%EF%B8%8F-system-architecture)
4. [Approach & Design Philosophy](#-approach--design-philosophy)
5. [Project Structure](#-project-structure)
6. [Data Schema & Labels](#-data-schema--labels)
7. [Environment & Configuration](#-environment--configuration)
8. [Installation & Setup](#-installation--setup)
9. [Running the Application](#-running-the-application)
10. [Evaluation & Testing](#-evaluation--testing)
11. [Explainability (Grad-CAM)](#-explainability-grad-cam)
12. [Deployment](#-deployment)
13. [Limitations & Future Work](#-limitations--future-work)
14. [Contributors & License](#-contributors--license)

---

## 🧠 Project Overview
Brain tumors (specifically gliomas) are highly aggressive and show diverse shape, size, and intensity properties on MRI. Manual delineation of tumor boundaries is time-consuming (30–60 minutes per patient), subjective, and prone to inter-observer variability. 

This project provides a fully automated pipeline that inputs **4 MRI modalities (T1, T1-contrast, T2, and FLAIR)** and segment the volume into three clinical sub-regions:
1. **Enhancing Tumor (ET)**: Active tumor parts.
2. **Tumor Core (TC)**: The necrotic core + active tumor.
3. **Whole Tumor (WT)**: The entire abnormal area, including peritumoral edema.

---

## 🛠️ Technology Stack
* **Programming Language:** Python 3.10 / 3.11 / 3.12
* **Deep Learning Framework:** PyTorch (`torch`, `torchvision`)
* **Medical Imaging Libraries:** 
  * `MONAI` (Medical Open Network for AI) for medical deep learning utilities.
  * `NiBabel` for loading and writing NIfTI volumetric data (`.nii`, `.nii.gz`).
  * `SimpleITK` for medical image registration and spacing utilities.
* **Scientific Computing:** `NumPy`, `SciPy`, `Scikit-learn`, `Scikit-image`
* **Visualization & Diagnostics:** `Matplotlib`, `Plotly`, `OpenCV`, `TensorBoard` for tracking runs.
* **Interactive Frontend:** `Streamlit` (interactive UI for medical upload & visualization)
* **Formatting & CLI Tools:** `pyyaml`, `rich`, `tqdm`

---

## 🏗️ System Architecture

The pipeline is decoupled into discrete, functional modules to ensure readability, scalability, and ease of deployment:

```
[ Multimodal MRI Scans ] (T1, T1ce, T2, FLAIR)
          │
          ▼
[ Preprocessing (transforms.py) ] ➔ Spacing & Z-score Intensity Normalisation
          │
          ▼
[ Patch Extraction (patch_extraction.py) ] ➔ Crop 128x128x128 foreground-biased patches
          │
          ▼
[ Data Augmentation (augmentation.py) ] ➔ 3D Rotation, Flip, Elastic Deforms, Noise
          │
          ▼
[ Deep Learning Core (models/) ] ➔ 3D U-Net / Attention 3D U-Net
          │
          ▼
┌───────────────────┴───────────────────┐
▼                                       ▼
[ Training (trainer.py) ]        [ Inference (predict.py) ]
  - Mixed Precision (AMP)          - Sliding Window
  - Dice + Cross-Entropy Loss      - Connected Component Clean-up
  - Cosine Scheduler               - Post-processing
          │                             │
          ▼                             ▼
[ TensorBoard Logs / Checkpoints ]  [ Streamlit UI & Visualizations ]
```

### High-Level Component Interactions:
1. **Data Layer (`data/`, `preprocessing/`):** Reads volumetric NIfTI datasets, resamples voxels to a consistent physical spacing ($1.0 \times 1.0 \times 1.0\text{ mm}$), normalises intensity values, and applies augmentations.
2. **Model Layer (`models/`):** Provides model definitions (3D convolutions, skip connections, and attention mechanisms) and robust losses.
3. **Execution Layer (`training/`, `evaluation/`, `inference/`):** Manages optimization loops, mixed-precision arithmetic (AMP), gradient accumulation, and test-time evaluations.
4. **Presentation Layer (`app/`):** Streamlit-based web dashboard allowing clinicians/users to drag-and-drop scans, execute predictions, inspect interactive slices, and view tumor statistics.

---

## 💡 Approach & Design Philosophy

### 1. 3D Spatial Context vs. 2D Slices
Traditional CNNs operate slice-by-slice (2D). However, brains and tumors are 3D structures. This pipeline uses **3D Convolutions** throughout the network to capture contextual information across the sagittal, coronal, and axial planes simultaneously.

### 2. Encoder-Decoder with Skip Connections (U-Net)
The primary network architecture is a **3D U-Net**. The encoder compresses the spatial dimensions while extracting high-level semantic features (what is a tumor). The decoder reconstructs spatial details (where is the tumor). Skip connections pass fine-grained localization features directly from the encoder to the decoder.

### 3. Attention Gates (Attention U-Net)
To suppress background noise (healthy brain tissues) and focus the network on irregular tumor boundaries, we implement an **Attention 3D U-Net**. At each skip connection, an attention gate uses features from the decoder path to prune redundant encoder activations, resulting in higher precision on small structures (like Enhancing Tumor) without significant parameters overhead.

### 4. Handling Severe Class Imbalance
Tumors occupy a very small percentage of the total brain volume (often $< 5\%$). Standard cross-entropy loss would bias the network to predict the background (healthy tissue). We address this by using **Dice-Cross Entropy Loss (`DiceCELoss`)** and **Focal Loss** to penalise misclassifications on minority target regions heavily.

### 5. Memory Efficiency & Patches
Due to massive memory demands of 3D convolutions, training on full volumes ($240 \times 240 \times 155$ voxels) is impossible on consumer GPUs. The pipeline extracts sub-volumes (**$128^3$ patches**) during training, utilizing **Foreground-Biased Sampling** to ensure $90\%$ of training patches contain tumor parts.

---

## 📁 Project Structure

```
Brain-Tumor-Segmentation-3D/
│
├── config.py                     # Central configuration (hyperparameters, paths, config CLI)
├── requirements.txt              # Project dependencies
├── README.md                     # Technical system documentation
├── .gitignore                    # Git rules for data, weights, logs and cache
├── generate_demo_data.py         # Utility to generate small mock 3D NIfTI files for testing
│
├── data/
│   ├── dataset.py                # Dataset loader for raw BraTS NIfTI files
│   ├── demo_upload/              # Temporary upload directory for web app
│   ├── processed/                # Destination folder for preprocessed volumes
│   └── raw/                      # Root directory for raw dataset
│
├── preprocessing/
│   ├── transforms.py             # Spacing, normalisation, and casting transforms
│   ├── patch_extraction.py       # Patch samplers (Foreground-biased, Sliding window)
│   └── augmentation.py           # 3D spatial & intensity augmentation pipelines
│
├── models/
│   ├── unet3d.py                 # Standard 3D U-Net implementation
│   ├── attention_unet.py         # Attention 3D U-Net implementation
│   └── losses.py                 # Dice, Dice+CE, and Focal losses
│
├── training/
│   ├── trainer.py                # PyTorch training loop engine (AMP, checkpointing)
│   └── train.py                  # CLI training script entry point
│
├── evaluation/
│   ├── metrics.py                # Dice, IoU, Sensitivity, Specificity, HD95 calculations
│   ├── evaluate.py               # Dataset-wide evaluation script
│   ├── visualize.py              # Visualisation generator (slice overlays)
│   └── gradcam.py                # 3D Grad-CAM explainability hooks
│
├── inference/
│   └── predict.py                # Sliding-window inference utility + post-processing
│
└── app/
    ├── streamlit_app.py          # Interactive web UI dashboard
    └── utils.py                  # Web app helper functions (volume stats, overlays)
```

---

## 📊 Data Schema & Labels

### Input Modalities (4 channels):
1. **T1:** Basic anatomical contrast.
2. **T1ce (Gd):** Contrast-enhanced (Gadolinium) scan. Highlights active blood-brain-barrier breakdown.
3. **T2:** Fluid-sensitive scan. Highlights edema and fluid.
4. **FLAIR:** Fluid Attenuated Inversion Recovery. Suppresses healthy cerebrospinal fluid to highlight tumor-induced edema.

### Output Segmentations:
Expert segmentations in the dataset contain 4 label values:
* `0`: Background (Healthy brain tissue/Outside brain).
* `1`: Necrotic / Non-enhancing tumor core (NCR/NET).
* `2`: Peritumoral Edema (ED).
* `4`: GD-enhancing tumor (ET).

### Evaluation Sub-Regions (Target Classes):
During evaluation and prediction, labels are combined into standard evaluation sub-regions:
* **Enhancing Tumor (ET):** Label `4`
* **Tumor Core (TC):** Labels `1` + `4`
* **Whole Tumor (WT):** Labels `1` + `2` + `4`

---

## ⚙️ Environment & Configuration
All configuration variables are stored in the `Config` dataclass in [config.py](file:///d:/AI%20STUFF/PROJECTS/MRI/config.py). You can adjust variables directly in the file, pass arguments to CLI scripts, or serialize configurations using YAML files.

### Key Configuration Variables:
* `crop_size`: Size of training patches, default `(128, 128, 128)`.
* `learning_rate`: Optimization step size, default `1e-4`.
* `batch_size`: Batch size per step, default `2`.
* `grad_accum_steps`: Accumulation steps to simulate larger batch size (default `4`, giving effective batch size = `8`).
* `use_amp`: Enables FP16 precision. Recommended `True`.
* `model_name`: `"unet3d"` or `"attention_unet3d"`.

---

## ⚡ Installation & Setup

### 1. Set Up Environment
```bash
# Clone the repository
git clone https://github.com/Rahim36712/Brain-Tumor-Segmentation-3D.git
cd Brain-Tumor-Segmentation-3D

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install PyTorch with GPU support (Highly Recommended)
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
```

### 2. Dataset Setup
To run training or evaluation, structure the dataset as follows inside the project directory:
```
data/
└── raw/
    └── BraTS2021_Training/
        ├── BraTS2021_00000/
        │   ├── BraTS2021_00000_flair.nii.gz
        │   ├── BraTS2021_00000_seg.nii.gz
        │   ├── BraTS2021_00000_t1.nii.gz
        │   ├── BraTS2021_00000_t1ce.nii.gz
        │   └── BraTS2021_00000_t2.nii.gz
        └── ...
```

---

## 🚀 Running the Application

### 1. Generating Mock Test Data
If you don't have the full BraTS dataset downloaded and want to verify the system works end-to-end, generate synthetic MRI scans instantly:
```bash
python generate_demo_data.py
```
This generates a test subject `BraTS2021_DEMO_00000` under `data/raw/BraTS2021_Training/`.

### 2. Running Training
To start training the segmentation model:
```bash
python training/train.py --data_dir data/raw/BraTS2021_Training --model attention_unet3d --epochs 300 --batch_size 2
```
To monitor progress via TensorBoard:
```bash
tensorboard --logdir logs/
```

### 3. Running the Web Interface
To launch the interactive dashboard:
```bash
streamlit run app/streamlit_app.py
```
* **Step 1:** Select the model architecture on the sidebar.
* **Step 2:** Drag and drop the 4 modalities (`t1`, `t1ce`, `t2`, `flair`) of NIfTI format.
* **Step 3:** Click **"Run Segmentation"** to view interactive overlay slices and tumor volume statistics.

---

## 📈 Evaluation & Testing
To evaluate your model on your validation dataset and calculate Dice scores, Sensitivity, and Hausdorff distances:
```bash
python evaluation/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --data_dir data/raw/BraTS2021_Training \
    --hausdorff
```
Results are saved to `outputs/evaluation_results.csv`.

---

## 🔍 Explainability (Grad-CAM)
Deep learning models are often criticised for being "black boxes". To provide transparency, we implement **3D Grad-CAM** to track the activation maps in the final convolutional layers. This maps out exactly which regions the network looked at to decide whether a tissue was enhancing tumor or edema.
To visualize Grad-CAM overlays, use the script in `evaluation/gradcam.py`.

---

## 🌐 Deployment
The interactive web application is designed with `Streamlit` and can be easily deployed to:
* **Streamlit Community Cloud:** Connect this GitHub repository directly to Streamlit for cloud execution.
* **Docker Container:** Package this application for local server environments or Kubernetes.

---

## ⚠️ Limitations & Future Work
* **Generalization:** Model is trained on skull-stripped and aligned BraTS datasets; raw clinical scans might need skull-stripping preprocessing (e.g., using HD-BET) before inference.
* **Context Limit:** Standard patch training ($128^3$) may fail to capture global anatomical positioning. 
* **Next Steps:** Export weights to **ONNX** format for faster inference times and explore **Swin-UNETR** (Transformer-based 3D segmentation).

---

## 📄 License & Maintainers
* **Project Owner / Maintainer:** RJ (rj@example.com)
* **License:** MIT License. All rights reserved.
