# RPDF: Proprioception-Dominant Fusion for Robotic Object Property Prediction

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-deep%20learning-ee4c2c.svg)
![Robotics](https://img.shields.io/badge/robotics-visuo--proprioceptive-00a896.svg)

This repository contains the code for a low-cost robotic perception project on
shortcut-robust physical-property prediction. A SO-101 arm performs a single
grasp-like interaction with an object. The model predicts mass, stiffness, and
material from a pre-contact RGB image and the robot's internal motor feedback
during contact.

The paper-facing method is **Reliable Proprioception-Dominant Fusion (RPDF)**.
RPDF was designed for deceptive visual-cue settings where objects that look
similar may have different physical properties, and objects with misleading
appearance cues can break standard visual or multimodal models.

## What This Project Studies

The task is multi-task classification over three object properties:

| Property | Classes |
| --- | --- |
| Mass | `low`, `medium`, `high` |
| Stiffness | `soft`, `medium`, `rigid` |
| Material | `sponge`, `foam`, `wood`, `container` |

The tactile signal is not collected from a dedicated tactile sensor. Instead,
the project uses proprioceptive feedback from Feetech STS3215 servos:

| Channels | Signal |
| --- | --- |
| `0-5` | joint position |
| `6-11` | joint load |
| `12-17` | joint current |
| `18-23` | joint velocity |

## Method: RPDF

RPDF starts from a ResNet-18 visual encoder, a 1D-CNN proprioceptive encoder,
and a Transformer fusion module. The reliable version changes the fusion
behaviour by giving the visual pathway a learned scalar budget `g`:

```text
V_tilde = V_null + g * (V - V_null)
```

Here `V` is the visual-token sequence and `V_null` is a learned null visual
token. A small `g` keeps the model in a proprioception-dominant operating
regime while still allowing a weak residual visual contribution.

The reliable training line adds:

- online-prefix training and evaluation for partial proprioceptive sequences;
- auxiliary proprioceptive heads;
- visual-budget regularization;
- visual-mismatch gate supervision;
- reliable checkpoint selection on validation behaviour.

The main entry point is:

```text
scripts/train_fusion_gating_online_reliable.py
```

## Main Results

The paper-facing evidence uses five seeds: `42, 123, 456, 789, 2024`.

| Method | Seen test avg. | Deceptive OOD avg. | Notes |
| --- | ---: | ---: | --- |
| Vision only | 95.29 | 20.50 | collapses under deceptive visual cues |
| Proprioception only | 95.69 | 79.28 | strong physical grounding, limited disambiguation |
| Standard fusion | 99.65 +/- 0.51 | 89.78 +/- 4.93 | strong seen performance, still shortcut-sensitive |
| Best fixed visual budget `g=0.05` | - | 90.98 +/- 4.09 | strong static low-visual-budget control |
| QMF | 99.30 +/- 0.45 | 75.96 +/- 1.69 | generic robust-fusion baseline |
| OGM-GE | 99.56 +/- 0.44 | 86.13 +/- 3.31 | generic robust-fusion baseline |
| **RPDF** | **99.12 +/- 0.78** | **94.89 +/- 3.18** | reliable proprioception-dominant fusion |

RPDF keeps the average visual budget low on OOD:

| Split | Average visual budget |
| --- | ---: |
| Seen test | 15.07 +/- 0.19 percent |
| Deceptive OOD | 10.95 +/- 2.03 percent |

Component ablations show why the reliable objective matters:

| Variant | Deceptive OOD avg. | Average visual budget |
| --- | ---: | ---: |
| RPDF full | 94.89 +/- 3.18 | 10.95 +/- 2.03 percent |
| w/o mismatch supervision | 77.64 +/- 6.15 | 42.62 +/- 1.81 percent |
| w/o auxiliary heads | 88.89 +/- 1.60 | 0.35 +/- 0.14 percent |
| w/o visual-budget regularization | 89.07 +/- 2.09 | 0.01 +/- 0.01 percent |

The mechanism diagnostics support the interpretation that RPDF is not a
general-purpose visual reliability oracle. It operates as a proprioception-first
model with a small useful visual residual. At the full sequence length, forcing
the visual budget to zero reduces OOD accuracy from `94.93 +/- 3.17` to
`86.67 +/- 4.50`.

## Repository Layout

```text
visuotactile/
├── collect_custom_multimodal.py      # replay-based robot data collection
├── collect_teleop_multimodal.py      # teleoperation data collection
├── interactive_control_oop.py        # SO-101 control interface
├── replay_position_logs.py           # replay recorded motion logs
├── scripts/
│   ├── train_fusion.py               # standard fusion baseline
│   ├── train_vision.py               # vision-only baseline
│   ├── train_tactile.py              # proprioception-only baseline
│   ├── train_fusion_gating_online_reliable.py
│   ├── aggregate_multi_seed_reliable.py
│   ├── train_fusion_qmf.py
│   ├── train_fusion_standard_ogmge.py
│   └── visualization/
├── tests/                            # focused unit tests for analysis helpers
├── docs/                             # project notes and lightweight figures
├── assets/                           # SO-101 CAD and architecture image
└── outputs/                          # local artifacts, ignored by git
```

Generated checkpoints, run directories, plots, masks, and datasets are not part
of the repository. They are intentionally ignored to keep the GitHub tree small.

## Dataset Format

The training scripts expect a dataset root with `train`, `val`, `test`, and
`ood_test` splits:

```text
Plaintextdataset/
├── train/
│   ├── physical_properties.json
│   └── WoodBlock_Native/
│       └── episode_YYYYMMDD_HHMMSS/
│           ├── visual_anchor.jpg
│           ├── tactile_data.pkl
│           └── metadata.json
├── val/
├── test/
└── ood_test/
```

The active paper experiments used 909 episodes:

| Split | Episodes |
| --- | ---: |
| train | 607 |
| val | 76 |
| test | 76 |
| ood_test | 150 |

## Setup

Use Python 3.10+ and install the core dependencies:

```bash
pip install torch torchvision
pip install opencv-python pillow numpy pandas scipy scikit-learn
pip install matplotlib seaborn tqdm streamlit pytest
```

Optional tooling for mask review and saliency diagnostics may require extra
packages such as Segment Anything, depending on the script you run.

## Running the Main RPDF Experiment

Single-seed training:

```bash
python scripts/train_fusion_gating_online_reliable.py \
  --mode train \
  --data_root /path/to/Plaintextdataset \
  --save_dir outputs/fusion_gating_online_reliable_seed42 \
  --seed 42 \
  --device cuda \
  --epochs 50 \
  --batch_size 16 \
  --num_workers 8 \
  --visual_mismatch_prob 0.25 \
  --lambda_mismatch_gate 0.2 \
  --reliable_selection_start_epoch 16 \
  --primary_checkpoint reliable \
  --no_live_plot
```

Evaluate a checkpoint:

```bash
python scripts/train_fusion_gating_online_reliable.py \
  --mode eval \
  --data_root /path/to/Plaintextdataset \
  --checkpoint outputs/fusion_gating_online_reliable_seed42/best_model.pth \
  --eval_split ood_test \
  --device cuda
```

Online-prefix evaluation:

```bash
python scripts/train_fusion_gating_online_reliable.py \
  --mode online_eval \
  --data_root /path/to/Plaintextdataset \
  --checkpoint outputs/fusion_gating_online_reliable_seed42/best_model.pth \
  --eval_split ood_test \
  --prefix_ratios 0.1,0.2,0.4,0.6,0.8,1.0 \
  --device cuda
```

Five-seed run:

```bash
DATA_ROOT=/path/to/Plaintextdataset \
OUTPUT_ROOT=outputs \
PYTHON_BIN=python \
DEVICE=cuda \
bash scripts/run_multi_seed_gating_online_reliable.sh
```

Aggregate five-seed RPDF results:

```bash
python scripts/aggregate_multi_seed_reliable.py \
  --runs_root outputs/fusion_gating_online_reliable_multiseed \
  --output_dir outputs/fusion_gating_online_reliable_multiseed/meta \
  --output_name multi_seed_summary_reliable.json
```

## Baselines and Diagnostics

Standard baselines:

```bash
python scripts/train_vision.py --mode train --data_root /path/to/Plaintextdataset
python scripts/train_tactile.py --mode train --data_root /path/to/Plaintextdataset
python scripts/train_fusion.py --mode train --data_root /path/to/Plaintextdataset
```

RPDF-related controls:

```bash
bash scripts/run_standard_controls_multiseed_20260421.sh
bash scripts/run_reliable_ablations_multiseed_20260421.sh
bash scripts/run_fixed_gate_grid_20260421.sh
bash scripts/run_qmf_multiseed_20260422.sh
bash scripts/run_standard_ogmge_multiseed_20260422.sh
```

Mechanism diagnostics:

```bash
python scripts/diagnose_visual_residual_prefix_scan.py --help
python scripts/diagnose_object_background_counterfactual.py --help
python scripts/evaluate_object_level_ood.py --help
```

## Data Collection

Robot collection entry points:

```bash
python collect_custom_multimodal.py \
  --log-file outputs/logs/position_logs.json \
  --dataset-root ../Plaintextdataset/train

python collect_teleop_multimodal.py
```

Dataset inspection UI:

```bash
streamlit run scripts/clean_dataset_ui.py
```

## Tests

Focused tests cover aggregation logic, fixed-gate overrides, object-level OOD
evidence, and QMF/OGM-GE helper behaviour:

```bash
pytest tests
```

For a fast syntax check after editing scripts:

```bash
python -m py_compile scripts/train_fusion_gating_online_reliable.py
```

## Artifact Policy

Large or generated files are ignored:

- `outputs*/`, `newOutput*/`, `runs/`, `logs/`, checkpoints, and tensorboard data;
- datasets such as `Plaintextdataset/`;
- mask review outputs and generated SAM masks;
- Python caches and local environment files.

If a result needs to be cited in a paper, commit a small summary JSON, table, or
figure under `docs/` rather than committing checkpoints or full run folders.

## License

Academic research code. Check third-party package and dataset licenses before
redistribution.
