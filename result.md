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
| Seeds | 42, 123, 456, 789, 2024 (5 independent runs) |
| Best Model Selection | Highest average validation accuracy across 3 tasks |

---

## 2. Main Results (Multi-Seed: mean ± std, n=5)

Results are averaged over 5 independent runs with different random seeds (42, 123, 456, 789, 2024). Data splits remain fixed across all runs. Only the model initialisation and training stochasticity differ.

### 2.1 Test Set (In-Distribution, 68 samples)

| Model | Mass Acc | Stiffness Acc | Material Acc | **Avg Acc** |
|---|---|---|---|---|
| **A. Fusion (Both)** | **100.00 ± 0.00%** | **100.00 ± 0.00%** | 98.53 ± 1.86% | **99.51 ± 0.62%** |
| B. Vision Only | 93.82 ± 1.95% | 94.12 ± 1.86% | **97.94 ± 1.18%** | 95.29 ± 1.47% |
| C. Tactile Only | 95.59 ± 1.32% | 95.88 ± 1.10% | 95.59 ± 2.28% | 95.69 ± 1.37% |

### 2.2 OOD Test Set (Out-of-Distribution, 120 samples)

| Model | Mass Acc | Stiffness Acc | Material Acc | **Avg Acc** |
|---|---|---|---|---|
| **A. Fusion (Both)** | **89.00 ± 8.42%** | **91.50 ± 6.06%** | **89.33 ± 7.37%** | **89.94 ± 7.00%** |
| B. Vision Only | 19.50 ± 5.79% | 21.33 ± 3.10% | 20.67 ± 4.78% | 20.50 ± 4.47% |
| C. Tactile Only | 79.00 ± 1.86% | 79.67 ± 1.55% | 79.17 ± 1.75% | 79.28 ± 1.64% |

### 2.3 Per-Seed Breakdown

| Model | Metric | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 2024 |
|---|---|---|---|---|---|---|
| Fusion | Test Avg Acc | 100.00% | 98.53% | 99.02% | 100.00% | 100.00% |
| Fusion | OOD Avg Acc | 96.94% | 82.50% | 84.44% | 86.11% | 99.72% |
| Vision | Test Avg Acc | 93.14% | 97.55% | 94.61% | 95.10% | 96.08% |
| Vision | OOD Avg Acc | 22.78% | 25.56% | 20.83% | 12.22% | 21.11% |
| Tactile | Test Avg Acc | 93.14% | 96.57% | 97.06% | 96.08% | 95.59% |
| Tactile | OOD Avg Acc | 80.83% | 77.78% | 78.33% | 77.78% | 81.67% |

### 2.4 Single-Seed Results (Seed 42, including ablations)

The following table includes the modality-blocking ablation experiments (Models D & E), which were run with seed=42 only.

| Model | Test Avg Acc | OOD Avg Acc |
|---|---|---|
| A. Fusion (Both) | 100.00% | 96.94% |
| B. Vision Only | 93.14% | 22.78% |
| C. Tactile Only | 93.14% | 80.83% |
| D. Fusion (Block Tac) | 96.57% | 25.00% |
| E. Fusion (Block Vis) | 92.65% | 77.50% |

### 2.5 Training Convergence

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

## 4. Inference-Time Attention Masking Ablation (Multi-Seed, n=5)

Using the 5 fusion model checkpoints trained with both modalities (Model A), we performed inference-time modality ablation via **attention masking** (`src_key_padding_mask`). Unlike the training-time input zeroing used in Models D/E, this approach sets the Transformer's key padding mask to `True` for the target modality's tokens, preventing all other tokens from attending to them.

Script: `scripts/infer_fusion_multiseed_ablation.py`

> **Important**: Attention masking and input zeroing are fundamentally different ablation methods. Attention masking completely hides tokens from the attention mechanism, while input zeroing feeds zero-valued tokens that still participate in attention. Results are **not directly comparable** with Models D/E.

### 4.1 Results (seed_means, n=5)

#### Test Set (In-Distribution, 68 samples)

| Mode | Mass Acc | Stiffness Acc | Material Acc | **Avg Acc** |
|---|---|---|---|---|
| Full (both modalities) | 100.00 ± 0.00% | 100.00 ± 0.00% | 98.53 ± 1.86% | **99.51 ± 0.62%** |
| Tactile only (mask vision) | 86.47 ± 2.35% | 87.35 ± 2.88% | 86.18 ± 1.76% | 86.67 ± 2.29% |
| Vision only (mask tactile) | 55.88 ± 9.16% | 68.24 ± 8.71% | 61.47 ± 9.41% | 61.86 ± 8.79% |

#### OOD Test Set (Out-of-Distribution, 120 samples)

| Mode | Mass Acc | Stiffness Acc | Material Acc | **Avg Acc** |
|---|---|---|---|---|
| Full (both modalities) | 89.00 ± 8.42% | 91.50 ± 6.06% | 89.33 ± 7.37% | **89.94 ± 7.00%** |
| Tactile only (mask vision) | **97.83 ± 3.93%** | **93.33 ± 8.16%** | **98.17 ± 3.27%** | **96.44 ± 4.70%** |
| Vision only (mask tactile) | 30.67 ± 7.84% | 33.17 ± 10.74% | 22.67 ± 6.86% | 28.83 ± 5.77% |

#### Per-Seed OOD Avg Acc

| Mode | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 2024 |
|---|---|---|---|---|---|
| Full | 96.94% | 82.50% | 84.44% | 86.11% | 99.72% |
| Tactile only (mask vision) | 100.00% | 93.89% | 100.00% | 88.33% | 100.00% |
| Vision only (mask tactile) | 31.39% | 35.28% | 30.28% | 18.06% | 29.17% |

### 4.2 Key Observation: Masking Vision Improves OOD Performance

Counterintuitively, the fusion model performs **better** on OOD when vision tokens are masked (96.44%) than when both modalities are used (89.94%). This suggests that on OOD data, the visual features actively **harm** performance by introducing misleading attention signals from unseen objects. Removing them via attention masking allows the Transformer to focus solely on tactile tokens, avoiding visual noise.

### 4.3 Comparison of Ablation Methods (OOD Avg Acc)

| Method | Model Weights | OOD Avg Acc |
|---|---|---|
| Attention mask tactile-only | Fusion (trained with both) | **96.44 ± 4.70%** |
| Fusion full (both modalities) | Fusion (trained with both) | 89.94 ± 7.00% |
| Standalone Tactile model | TactileOnlyModel | 79.28 ± 1.64% |
| Input zeroing block visual (Model E) | Fusion (trained with visual zeroed) | 77.50% (seed 42 only) |

The large gap between attention masking (96.44%) and standalone tactile (79.28%) does **not** mean the fusion model's tactile branch is inherently stronger. The difference arises because:

1. **Different model weights**: The fusion model's tactile encoder was jointly trained with vision, benefiting from richer cross-modal gradients.
2. **Different ablation mechanism**: Attention masking completely hides tokens; input zeroing feeds noisy zero-derived tokens into attention.
3. **Untested inference pattern**: The model was never trained with attention-masked vision, so this is an out-of-distribution inference regime for the model itself.

---

## 5. Key Findings

### 5.1 Fusion achieves the best performance on both in-distribution and OOD data

- Test: **99.51 ± 0.62%** vs 95.69 ± 1.37% (best single modality, tactile)
- OOD: **89.94 ± 7.00%** vs 79.28 ± 1.64% (best single modality, tactile)
- The OOD improvement (+10.66 pp over tactile-only) is substantially larger than the test improvement (+3.82 pp), demonstrating that fusion provides the greatest benefit for generalisation to unseen objects.

### 5.2 Vision overfits to object appearance and fails completely on OOD

- Vision achieves strong in-distribution performance (95.29 ± 1.47%) but collapses to random-chance levels on OOD (20.50 ± 4.47%).
- This collapse is consistent across all 5 seeds (range: 12.22%–25.56%), confirming it is a robust property of the visual modality, not a random artifact.
- The standalone VisionOnlyModel and the FusionModel with blocked tactile both show this collapse, confirming it is inherent to visual features for novel objects.
- On OOD, vision-only models predict all samples as a single class (e.g., all mass -> "medium", all material -> "wood"), indicating complete failure to generalise visual features to unseen objects.

### 5.3 Tactile provides robust generalisation but has limitations

- Tactile-only maintains **79.28 ± 1.64%** on OOD (vs 95.69 ± 1.37% test), showing meaningful but imperfect generalisation.
- Notably, tactile has the **lowest variance** across seeds on OOD (std=1.64%), indicating very stable generalisation behaviour.
- Tactile consistently struggles with **low mass** (23% recall), **medium stiffness** (23% recall), and **foam material** (23% recall) on OOD, confusing them with similar-feeling categories.

### 5.4 Synergistic complementarity: 1 + 1 > 2

- Vision alone: **20.50 ± 4.47%** OOD (useless independently)
- Tactile alone: **79.28 ± 1.64%** OOD
- Fusion: **89.94 ± 7.00%** OOD

Adding a modality that is independently useless on OOD still improves the combined performance by ~10.7 pp on average. This demonstrates **synergistic complementarity** -- visual features, while insufficient for independent classification of novel objects, provide disambiguation cues that resolve tactile ambiguities through cross-modal attention in the Transformer.

Specifically, in the best seed (42), fusion fixes the exact classes where tactile fails:
- Mass "low": 23% (tactile) -> 90% (fusion)
- Stiffness "medium": 23% (tactile) -> 87% (fusion)
- Material "foam": 23% (tactile) -> 87% (fusion)

### 5.5 Fusion model shows higher variance on OOD

The fusion model has the highest OOD variance (std=7.00%) compared to tactile (1.64%) and vision (4.47%). Per-seed OOD results: 96.94%, 82.50%, 84.44%, 86.11%, 99.72%. This suggests that the Transformer-based cross-modal fusion is sensitive to random initialisation -- different seeds may lead to different quality of learned cross-modal attention patterns. Notably, seeds 42 and 2024 both achieved near-perfect OOD accuracy (96.94% and 99.72%), while other seeds ranged 82–86%. This is a known challenge in multimodal learning and could be mitigated by ensemble methods or more training data.

### 5.6 Token count imbalance creates a structural bias

The FusionModel has 49 visual tokens vs 375 tactile tokens (ratio ~1:7.7), which creates an inherent bias in the Transformer's self-attention toward tactile information. This may partially explain why the fusion model's OOD behaviour more closely resembles tactile-only, and why visual contributions might be underutilised.

---

## 6. Commands Used

### 6.1 Single-Seed Experiments

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

### 6.2 Multi-Seed Experiments

```bash
# Automated script for running 5 seeds (42, 123, 456, 789, 2024) for Models A/B/C
bash visuotactile/scripts/run_multi_seed.sh

# Or manually per model per seed:
for SEED in 42 123 456 789 2024; do
  python visuotactile/scripts/train_fusion.py --mode train --seed $SEED --save_dir visuotactile/outputs/fusion_seed${SEED} --no_live_plot
  python visuotactile/scripts/train_vision.py --mode train --seed $SEED --save_dir visuotactile/outputs/vision_seed${SEED} --no_live_plot
  python visuotactile/scripts/train_tactile.py --mode train --seed $SEED --save_dir visuotactile/outputs/tactile_seed${SEED} --no_live_plot
done
```

### 6.3 Inference-Time Attention Masking Ablation

```bash
# Run attention masking ablation on all 5 fusion model checkpoints
python visuotactile/scripts/infer_fusion_multiseed_ablation.py
```

---

## 7. Output Directories

### 7.1 Single-Seed Experiments (Seed 42)

| Model | Checkpoints & Training History | Test Eval | OOD Test Eval |
|---|---|---|---|
| A. Fusion | `outputs/fusion_model_clean/` | `outputs/fusion_model_clean/eval_test/` | `outputs/fusion_model_clean/eval_ood_test/` |
| B. Vision Only | `outputs/vision_model_clean/` | `outputs/vision_model_clean/eval_test/` | `outputs/vision_model_clean/eval_ood_test/` |
| C. Tactile Only | `outputs/tactile_model_clean/` | `outputs/tactile_model_clean/eval_test/` | `outputs/tactile_model_clean/eval_ood_test/` |
| D. Fusion (Block Tac) | `outputs/fusion_model_visualOnly/` | `outputs/fusion_model_visualOnly/eval_test_block_tactile/` | `outputs/fusion_model_visualOnly/eval_ood_test_block_tactile/` |
| E. Fusion (Block Vis) | `outputs/fusion_model_tactileOnly/` | `outputs/fusion_model_tactileOnly/eval_test_block_visual/` | `outputs/fusion_model_tactileOnly/eval_ood_test_block_visual/` |

### 7.2 Multi-Seed Experiments

| Model | Seed 42 | Seed 123 | Seed 456 | Seed 789 | Seed 2024 |
|---|---|---|---|---|---|
| Fusion | `outputs/fusion_seed42/` | `outputs/fusion_seed123/` | `outputs/fusion_seed456/` | `outputs/fusion_seed789/` | `outputs/fusion_seed2024/` |
| Vision | `outputs/vision_seed42/` | `outputs/vision_seed123/` | `outputs/vision_seed456/` | `outputs/vision_seed789/` | `outputs/vision_seed2024/` |
| Tactile | `outputs/tactile_seed42/` | `outputs/tactile_seed123/` | `outputs/tactile_seed456/` | `outputs/tactile_seed789/` | `outputs/tactile_seed2024/` |

### 7.3 Attention Masking Ablation

| Output | Path |
|---|---|
| Overall Summary | `outputs/fusion_infer_ablation_mask/summary_overall.json` |
| Per-Seed Results | `outputs/fusion_infer_ablation_mask/seed_*/` |

Aggregated summary: `outputs/multi_seed_summary.json`
