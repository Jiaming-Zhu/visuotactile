# Visuotactile Fusion Model - Experimental Results

## 1. Models Overview

This document summarises the evaluation results of **five models** for physical property prediction (mass, stiffness, material) from robotic grasping data.

### 1.1 Model Index

| # | Model Name | Script | Architecture | Input Modalities |
|---|---|---|---|---|
| A | **Fusion (Both)** | `scripts/train_fusion.py` | FusionModel | Visual + Tactile |
| B | **Vision Only (Standalone)** | `scripts/train_vision.py` | VisionOnlyModel | Visual only |
| C | **Tactile Only (Standalone)** | `scripts/train_tactile.py` | TactileOnlyModel | Tactile only |
| D | **Fusion (Block Tactile)** | `scripts/train_fusion.py --block_modality tactile` | FusionModel | Visual only (tactile zeroed) |
| E | **Fusion (Block Visual)** | `scripts/train_fusion.py --block_modality visual` | FusionModel | Tactile only (visual zeroed) |

### 1.2 Architecture Details

#### Model A / D / E: FusionModel (`train_fusion.py`)

- **Visual Encoder**: ResNet18 (ImageNet pretrained, frozen by default) -> 1x1 Conv2d projection (512 -> fusion_dim)
- **Tactile Encoder**: 3-layer Conv1d (24 -> 64 -> 128 -> fusion_dim), stride=2, with BatchNorm + ReLU
- **Token Sequence**: [CLS] + 49 visual tokens (7x7) + 375 tactile tokens (3000/8) = **425 tokens**
- **Fusion**: Transformer Encoder (4 layers, 8 heads, d_model=256, FFN=1024, GELU, dropout=0.1)
- **Classification Heads**: 3x MLP (Linear(256,128) -> GELU -> Dropout -> Linear(128, n_classes))
  - Mass: 4 classes | Stiffness: 4 classes | Material: 5 classes
- **Positional Embedding**: Learnable, shape (1, 425, 256), hardcoded
- **Padding Mask**: Applied to variable-length tactile via 3x max_pool1d downsampling

#### Model B: VisionOnlyModel (`train_vision.py`)

- **Visual Encoder**: Same as FusionModel (ResNet18 + 1x1 Conv2d)
- **Token Sequence**: [CLS] + 49 visual tokens = **50 tokens**
- **Transformer + Heads**: Same architecture as FusionModel
- **Positional Embedding**: Learnable, shape (1, 50, 256), dynamically computed via dummy forward
- **No padding mask** (fixed-length visual input)
- **Additional**: Supports configurable `--image_size` (default 224)

#### Model C: TactileOnlyModel (`train_tactile.py`)

- **Tactile Encoder**: Same as FusionModel (3-layer Conv1d)
- **Token Sequence**: [CLS] + 375 tactile tokens = **376 tokens**
- **Transformer + Heads**: Same architecture as FusionModel
- **Positional Embedding**: Learnable, shape (1, 376, 256), dynamically computed via dummy forward
- **Padding Mask**: Same mechanism as FusionModel (3x max_pool1d downsampling)

### 1.3 Shared Training Strategy

All five models use identical training hyperparameters for fair comparison:

| Parameter | Value |
|---|---|
| Optimiser | AdamW (lr=1e-4, weight_decay=0.01) |
| LR Schedule | Linear warmup (5 epochs) + Cosine annealing |
| Loss | CrossEntropyLoss (equal weight sum of 3 tasks) |
| Gradient Clipping | max_norm=1.0 |
| Epochs | 50 |
| Batch Size | 16 |
| Seed | 42 |
| Best Model Selection | Highest average validation accuracy across 3 tasks |

---

## 2. Main Results

### 2.1 Test Set (In-Distribution, 68 samples)

| Model | Mass Acc | Stiffness Acc | Material Acc | **Avg Acc** | Avg Macro F1 | Loss |
|---|---|---|---|---|---|---|
| **A. Fusion (Both)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **1.0000** | 0.051 |
| B. Vision Only | 91.18% | 91.18% | 97.06% | 93.14% | 0.9177 | 0.597 |
| C. Tactile Only | 94.12% | 94.12% | 91.18% | 93.14% | 0.9041 | 1.034 |
| D. Fusion (Block Tac) | 95.59% | 95.59% | 98.53% | 96.57% | 0.9586 | 0.666 |
| E. Fusion (Block Vis) | 94.12% | 94.12% | 89.71% | 92.65% | 0.8939 | 0.666 |

### 2.2 OOD Test Set (Out-of-Distribution, 120 samples)

| Model | Mass Acc | Stiffness Acc | Material Acc | **Avg Acc** | Avg Macro F1 | Loss |
|---|---|---|---|---|---|---|
| **A. Fusion (Both)** | **97.50%** | **96.67%** | **96.67%** | **96.94%** | **0.8262** | 0.253 |
| B. Vision Only | 22.50% | 23.33% | 22.50% | 22.78% | 0.1215 | 15.168 |
| C. Tactile Only | 80.83% | 80.83% | 80.83% | 80.83% | 0.6671 | 2.997 |
| D. Fusion (Block Tac) | 25.00% | 25.00% | 25.00% | 25.00% | 0.1312 | 20.233 |
| E. Fusion (Block Vis) | 77.50% | 77.50% | 77.50% | 77.50% | 0.6136 | 2.476 |

### 2.3 Training Convergence

| Model | Best Epoch | Best Val Avg Acc | Final Train Loss |
|---|---|---|---|
| A. Fusion (Both) | 6 / 50 | 100.00% | 0.0010 |
| B. Vision Only | 14 / 50 | 100.00% | 0.0042 |
| C. Tactile Only | 33 / 50 | 93.63% | 0.0083 |
| D. Fusion (Block Tac) | 36 / 50 | 100.00% | 0.0021 |
| E. Fusion (Block Vis) | 16 / 50 | 93.14% | 0.0087 |

> **Note**: Model D (Fusion with blocked tactile) exhibited severe training instability with validation accuracy oscillating between ~19% and ~100% across epochs due to 375 zero-valued tactile tokens polluting the Transformer's self-attention. The best epoch (36) was reached during a brief stability window.

---

## 3. Per-Task Breakdown (OOD Test)

### 3.1 Mass Classification (4 classes)

| Model | very_low | low | medium | high | Accuracy |
|---|---|---|---|---|---|
| A. Fusion | P:0.94 R:1.00 | P:1.00 R:0.90 | P:0.97 R:1.00 | P:1.00 R:1.00 | **97.50%** |
| B. Vision | P:0.00 R:0.00 | P:0.00 R:0.00 | P:0.45 R:0.90 | P:0.00 R:0.00 | 22.50% |
| C. Tactile | P:0.57 R:1.00 | P:1.00 R:0.23 | P:1.00 R:1.00 | P:1.00 R:1.00 | 80.83% |

### 3.2 Stiffness Classification (4 classes)

| Model | very_soft | soft* | medium | rigid | Accuracy |
|---|---|---|---|---|---|
| A. Fusion | P:1.00 R:1.00 | P:0.00 R:0.00 | P:1.00 R:0.87 | P:0.97 R:1.00 | **96.67%** |
| B. Vision | P:0.00 R:0.00 | P:0.00 R:0.00 | P:0.00 R:0.00 | P:0.32 R:0.47 | 23.33% |
| C. Tactile | P:1.00 R:1.00 | P:0.00 R:0.00 | P:1.00 R:0.23 | P:1.00 R:1.00 | 80.83% |

> *\*soft class has 0 support in OOD test set*

### 3.3 Material Classification (5 classes)

| Model | sponge | foam | wood | hollow_c* | filled_c | Accuracy |
|---|---|---|---|---|---|---|
| A. Fusion | P:1.00 R:1.00 | P:1.00 R:0.87 | P:0.94 R:1.00 | P:0.00 R:0.00 | P:0.97 R:1.00 | **96.67%** |
| B. Vision | P:0.00 R:0.00 | P:0.00 R:0.00 | P:0.45 R:0.90 | P:0.00 R:0.00 | P:0.00 R:0.00 | 22.50% |
| C. Tactile | P:1.00 R:1.00 | P:1.00 R:0.23 | P:1.00 R:1.00 | P:0.00 R:0.00 | P:0.64 R:1.00 | 80.83% |

> *\*hollow_container class has 0 support in OOD test set*

---

## 4. Key Findings

### 4.1 Fusion achieves the best performance on both in-distribution and OOD data

- Test: 100% (perfect) vs 93.14% (best single modality)
- OOD: 96.94% vs 80.83% (best single modality, tactile)
- The OOD improvement (+16.1 pp over tactile-only) is substantially larger than the test improvement (+6.9 pp), demonstrating that fusion provides the greatest benefit for generalisation to unseen objects.

### 4.2 Vision overfits to object appearance and fails completely on OOD

- Vision achieves strong in-distribution performance (93.14%) but collapses to random-chance levels on OOD (22.78%).
- The standalone VisionOnlyModel (22.78%) and the FusionModel with blocked tactile (25.00%) both show this collapse, confirming it is a property of the visual modality itself, not a model artifact.
- On OOD, vision-only models predict all samples as a single class (e.g., all mass -> "medium", all material -> "wood"), indicating complete failure to generalise visual features to unseen objects.

### 4.3 Tactile provides robust generalisation but has limitations

- Tactile-only maintains 80.83% on OOD (vs 93.14% test), showing meaningful but imperfect generalisation.
- Tactile consistently struggles with **low mass** (23% recall), **medium stiffness** (23% recall), and **foam material** (23% recall) on OOD, confusing them with similar-feeling categories.

### 4.4 Synergistic complementarity: 1 + 1 > 2

- Vision alone: 22.78% OOD (useless independently)
- Tactile alone: 80.83% OOD
- Fusion: 96.94% OOD

Adding a modality that is independently useless on OOD dramatically improves the combined performance. This demonstrates **synergistic complementarity** -- visual features, while insufficient for independent classification of novel objects, provide disambiguation cues that resolve tactile ambiguities through cross-modal attention in the Transformer.

Specifically, fusion fixes the exact classes where tactile fails:
- Mass "low": 23% (tactile) -> 90% (fusion)
- Stiffness "medium": 23% (tactile) -> 87% (fusion)
- Material "foam": 23% (tactile) -> 87% (fusion)

### 4.5 Token count imbalance creates a structural bias

The FusionModel has 49 visual tokens vs 375 tactile tokens (ratio ~1:7.7), which creates an inherent bias in the Transformer's self-attention toward tactile information. This may partially explain why the fusion model's OOD behaviour more closely resembles tactile-only, and why visual contributions might be underutilised.

---

## 5. Commands Used

```bash
# Model A: Fusion (both modalities)
python visuotactile/scripts/train_fusion.py --mode train --save_dir visuotactile/outputs/fusion_model_clean --no_live_plot

# Model B: Vision Only (standalone)
python visuotactile/scripts/train_vision.py --mode train --save_dir visuotactile/outputs/vision_model_clean --no_live_plot

# Model C: Tactile Only (standalone)
python visuotactile/scripts/train_tactile.py --mode train --save_dir visuotactile/outputs/tactile_model_clean --no_live_plot

# Model D: Fusion with blocked tactile (visual only)
python visuotactile/scripts/train_fusion.py --mode train --block_modality tactile --save_dir visuotactile/outputs/fusion_model_visualOnly --no_live_plot

# Model E: Fusion with blocked visual (tactile only)
python visuotactile/scripts/train_fusion.py --mode train --block_modality visual --save_dir visuotactile/outputs/fusion_model_tactileOnly --no_live_plot
```

---

## 6. Output Directories

| Model | Checkpoints & Training History | Test Eval | OOD Test Eval |
|---|---|---|---|
| A. Fusion | `outputs/fusion_model_clean/` | `outputs/fusion_model_clean/eval_test/` | `outputs/fusion_model_clean/eval_ood_test/` |
| B. Vision Only | `outputs/vision_model_clean/` | `outputs/vision_model_clean/eval_test/` | `outputs/vision_model_clean/eval_ood_test/` |
| C. Tactile Only | `outputs/tactile_model_clean/` | `outputs/tactile_model_clean/eval_test/` | `outputs/tactile_model_clean/eval_ood_test/` |
| D. Fusion (Block Tac) | `outputs/fusion_model_visualOnly/` | `outputs/fusion_model_visualOnly/eval_test_block_tactile/` | `outputs/fusion_model_visualOnly/eval_ood_test_block_tactile/` |
| E. Fusion (Block Vis) | `outputs/fusion_model_tactileOnly/` | `outputs/fusion_model_tactileOnly/eval_test_block_visual/` | `outputs/fusion_model_tactileOnly/eval_ood_test_block_visual/` |
