# Visuotactile RPDF Project Context

This file gives future coding agents the current project state. The active
paper story is no longer generic visuotactile fusion; it is Reliable
Proprioception-Dominant Fusion (RPDF) under deceptive visual cues.

## Project Goal

The system predicts physical object properties from one robotic interaction:

- `mass`: low, medium, high
- `stiffness`: soft, medium, rigid
- `material`: sponge, foam, wood, container

Inputs are:

- pre-contact RGB image from a USB camera;
- 24-channel proprioceptive sequence from six SO-101/Feetech joints:
  position, load, current, and velocity.

The main scientific question is how to keep multimodal learning robust when
visual appearance is an unreliable shortcut.

## Current Dataset

Dataset repository:
https://github.com/Jiaming-Zhu/visuotactile-rpdf-dataset

The paper-facing split is the current `Plaintextdataset` layout:

- `train`: 607 episodes
- `val`: 76 episodes
- `test`: 76 episodes
- `ood_test`: 150 deceptive-object episodes

The original 678-episode ID dataset was collected in November and December
2025. The deceptive OOD split and final hollow-box taxonomy were expanded from
February to March 2026. Use directory names plus `physical_properties.json` as
the label source; some per-episode `metadata.json` labels are not reliable.

## Core Models

Baseline scripts:

- `scripts/train_vision.py`: vision-only ResNet baseline.
- `scripts/train_tactile.py`: proprioception-only sequence baseline.
- `scripts/train_fusion.py`: standard visual-proprioceptive Transformer fusion.

RPDF/reliable scripts:

- `scripts/train_fusion_gating_online.py`: online-prefix gating model.
- `scripts/train_fusion_gating_online_reliable.py`: main RPDF training script.
- `scripts/aggregate_multi_seed_reliable.py`: five-seed RPDF aggregation.
- `scripts/train_fusion_standard_ablation.py`: standard-control variants.
- `scripts/train_fusion_qmf.py`: QMF baseline.
- `scripts/train_fusion_standard_ogmge.py`: OGM-GE baseline.
- `scripts/aggregate_fixed_gate_grid.py`: fixed visual-budget grid aggregation.

RPDF applies a learned visual budget `g`:

```text
V_tilde = V_null + g * (V - V_null)
```

The reported gate score should be interpreted as a visual budget. Lower values
mean the model is operating closer to the learned null visual token and relying
more heavily on proprioception.

## Paper-Facing Results

Use the five-seed protocol with seeds `42, 123, 456, 789, 2024`.

Main RPDF summary:

- seen test average: `99.12 +/- 0.78`
- deceptive OOD average: `94.89 +/- 3.18`
- OOD mass: `99.73 +/- 0.33`
- OOD stiffness: `93.33 +/- 4.94`
- OOD material: `91.60 +/- 4.87`
- OOD visual budget: `10.95 +/- 2.03` percent

Important baselines:

- standard fusion OOD: `89.78 +/- 4.93`
- best fixed visual budget, `g=0.05`: `90.98 +/- 4.09`
- QMF OOD: `75.96 +/- 1.69`
- OGM-GE OOD: `86.13 +/- 3.31`

Important ablations:

- RPDF w/o mismatch supervision: `77.64 +/- 6.15`, visual budget reopens to
  `42.62 +/- 1.81` percent.
- RPDF w/o auxiliary heads: `88.89 +/- 1.60`.
- RPDF w/o visual-budget regularization: `89.07 +/- 2.09`.

Mechanism interpretation:

- RPDF is not presented as a calibrated reliability oracle.
- It is a stable proprioception-dominant model with a small useful visual
  residual.
- At full sequence length, forcing visual budget to zero drops OOD accuracy
  from about `94.93` to `86.67`.

## Reproduction Commands

Main five-seed RPDF:

```bash
DATA_ROOT=/path/to/Plaintextdataset \
OUTPUT_ROOT=outputs \
PYTHON_BIN=python \
DEVICE=cuda \
bash scripts/run_multi_seed_gating_online_reliable.sh
```

Single-seed RPDF:

```bash
python scripts/train_fusion_gating_online_reliable.py \
  --mode train \
  --data_root /path/to/Plaintextdataset \
  --save_dir outputs/fusion_gating_online_reliable_seed42 \
  --seed 42 \
  --device cuda \
  --epochs 50 \
  --visual_mismatch_prob 0.25 \
  --lambda_mismatch_gate 0.2 \
  --reliable_selection_start_epoch 16 \
  --primary_checkpoint reliable \
  --no_live_plot
```

Fixed-gate, ablation, QMF, and OGM-GE scripts are shell entry points under
`scripts/run_*_202604*.sh`.

## Artifact Policy

Do not commit run directories, checkpoints, mask dumps, datasets, or generated
logs. Commit source code, tests, lightweight docs, and small paper-facing
figures/tables only. `.gitignore` should block `outputs*/`, `newOutput*/`,
`*.pth`, `*.pt`, `Plaintextdataset*/`, `runs/`, and `logs/`.
