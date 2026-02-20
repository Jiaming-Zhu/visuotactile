# 🤖 Visuotactile Fusion for Robotic Object Property Estimation

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![Robotics](https://img.shields.io/badge/Robotics-Visuotactile-00a896.svg)

A multi-modal deep learning framework that combines **visual** and **tactile** sensing for estimating physical properties of grasped objects using a low-cost robotic manipulator.

---

## 📖 Overview

This project implements a **ResNet-Transformer fusion architecture** that predicts three physical properties from a single robotic grasp interaction. Instead of using expensive tactile sensors (like GelSight), it extracts implicit proprioceptive tactile signals directly from the low-cost servo motors' feedback.

| Property | Classes | Description |
|----------|---------|-------------|
| **Mass** | 4 | `very_low`, `low`, `medium`, `high` |
| **Stiffness** | 4 | `very_soft`, `soft`, `medium`, `rigid` |
| **Material** | 5 | `sponge`, `foam`, `wood`, `hollow_container`, `filled_container` |

### ✨ Key Features

- **Visual-Tactile Fusion**: Combines static RGB images (pre-grasp) with dynamic time-series proprioceptive signals (motor current, position, load, velocity during grasp).
- **Low-Cost Tactile Sensing**: Uses servo motor feedback as implicit tactile signals—no expensive tactile sensors required.
- **Robust OOD Generalization**: Demonstrates strong zero-shot generalization to unseen objects, leveraging tactile grounding to overcome visual "simplicity bias".

---

## 🧠 Architecture

![Fusion Model Architecture](assets/fusionModel.png)

*The fusion model combines visual features (ResNet18) and tactile features (1D-CNN) through a Transformer Encoder. The `[CLS]` token is then passed through 3 independent MLP heads to output predictions for mass, stiffness, and material.*

---

## 📊 Experimental Results & Key Findings

Our extensive evaluation across multiple random seeds (n=5) revealed several critical insights regarding multimodal learning for physical property estimation.

### 1. Superior Out-Of-Distribution (OOD) Generalization
The Fusion model vastly outperforms single-modality baselines when encountering **novel, unseen objects**:

| Model | Test Acc (In-Distribution) | OOD Test Acc (Novel Objects) |
|---|:---:|:---:|
| **Fusion (Visual + Tactile)** | **99.51%** | **89.94%** |
| Tactile Only | 95.69% | 79.28% |
| Vision Only | 95.29% | 20.50% |

### 2. Synergistic Complementarity (1 + 1 > 2)
While the Vision-only model completely collapses on OOD data (20.50%, roughly random chance), combining it with Tactile data (79.28%) yields a Fusion performance of **89.94%**. Vision features, though useless independently for novel objects, provide crucial disambiguation cues that resolve tactile ambiguities through cross-modal attention.

### 3. The "Simplicity Bias" & Attention Masking
We discovered that Vision suffers from severe *simplicity bias*—it memorizes object appearances (colors/textures) rather than physics. 
Counterintuitively, during inference on OOD data, **masking out the visual tokens** in the Transformer attention mechanism actually **improves** the Fusion model's accuracy from `89.94%` to `96.44%`. This indicates that for unseen objects, visual features can act as deceptive noise, and the model benefits from falling back entirely on tactile grounding.

---

## 🛠️ Hardware Setup

- **Robot**: SO-101 6-DOF Manipulator
- **Actuators**: Feetech STS3215 servo motors (providing 24-dim tactile feedback)
- **Camera**: USB webcam (640×480)
- **Controller**: Raspberry Pi / Linux PC

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install torch torchvision
pip install opencv-python pillow numpy pandas
pip install scikit-learn seaborn matplotlib
pip install streamlit  # for dataset cleaner UI
```

### Training & Evaluation

```bash
# Train fusion model (full modalities)
python scripts/train_fusion.py --mode train \
    --data_root /path/to/Plaintextdataset \
    --epochs 50 --device cuda

# Ablation-style training (block one modality)
python scripts/train_fusion.py --mode train \
    --data_root /path/to/Plaintextdataset \
    --block_modality visual   # or tactile / none

# Evaluate saved checkpoint on a split
python scripts/train_fusion.py --mode eval \
    --data_root /path/to/Plaintextdataset \
    --checkpoint outputs/fusion_model_clean/best_model.pth \
    --eval_split test
```

### Inference-Time Attention Ablation
```bash
# Run attention masking ablation on fusion checkpoints
python scripts/infer_fusion_multiseed_ablation.py
```

### Data Collection & Cleaning

```bash
# Collect multimodal grasping data using the real robot
python collect_custom_multimodal.py \
    --log-file outputs/logs/position_logs.json \
    --dataset-root ../Plaintextdataset/train

# Launch Streamlit UI for dataset inspection
streamlit run scripts/clean_dataset_ui.py
```

---

## 📂 Dataset Format

```text
Plaintextdataset/
├── train/
│   ├── physical_properties.json    # Labels for training objects
│   ├── WoodBlock_Native/
│   │   ├── episode_xxx/
│   │   │   ├── visual_anchor.jpg   # RGB image before grasp
│   │   │   ├── tactile_data.pkl    # Time-series sensor data (Pickle)
│   │   │   └── metadata.json       # Episode metadata
│   │   └── ...
│   └── ...
└── val/
    ├── physical_properties.json    # Labels for validation objects
    └── ...
```

### 📡 Tactile Data Channels (24-dim)

The implicit tactile feedback is collected as a 24-dimensional time series from the 6 joints:

| Channel Index | Feedback Type |
|---------------|---------------|
| `0 - 5` | Joint Positions (6 DOF) |
| `6 - 11` | Joint Loads |
| `12 - 17` | Joint Currents |
| `18 - 23` | Joint Velocities |

---

## 🗂️ Project Structure

```text
visuotactile/
├── scripts/                    # Training & utility scripts
│   ├── train_fusion.py         # Main training/evaluation entrypoint
│   ├── train_vision.py         # Vision-only baseline
│   ├── train_tactile.py        # Tactile-only baseline
│   ├── infer_fusion_multiseed_ablation.py # Attention masking ablation
│   ├── clean_dataset_ui.py     # Streamlit dataset cleaner
│   ├── run_multi_seed.sh       # Multi-seed training automation
│   └── ...
├── outputs/                    # Model checkpoints & results
├── collect_custom_multimodal.py# Robot teleoperation & Data collection script
├── interactive_control_oop.py  # Robot control interface
├── replay_position_logs.py     # Motion replay utility
├── assets/                     # SO-101 robot CAD & architecture diagrams
├── docs/                       # Documentation
└── so101_new_calib.urdf        # Robot URDF model
```

---

## 📄 License

This project is for academic research purposes.
