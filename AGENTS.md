# Repository Guidelines

> **⚠️ AI Agent 必读**: 在处理具体的代码或分析实验结果前，请优先阅读 [PROJECT_CONTEXT_FOR_AGENTS.md](PROJECT_CONTEXT_FOR_AGENTS.md) 以快速掌握该项目的架构设计和核心实验结论（特别是 OOD 泛化性能和模态消融实验的发现）。

## Project Structure & Module Organization
Core code is Python-first and split by workflow:
- `scripts/`: training, evaluation, visualization, and dataset tools (for example `train_fusion.py`, `clean_dataset_ui.py`, `visualize_plaintext_dataset.py`).
- Repository root: robot runtime and data collection entry points such as `collect_custom_multimodal.py`, `interactive_control_oop.py`, and `replay_position_logs.py`.
- `docs/`: design and architecture notes (see `docs/model_architecture.md`).
- `assets/`: CAD/URDF and media assets used by the SO-101 setup.
- `outputs/`: generated checkpoints, metrics, and plots. Treat as experiment artifacts, not source.

## Build, Test, and Development Commands
Use Python 3.10+ and install dependencies from `README.md` (PyTorch, OpenCV, NumPy, pandas, Streamlit, etc.).

Key commands:
- `python scripts/train_fusion.py --mode train --data_root /home/martina/Y3_Project/Plaintextdataset --epochs 50 --device cuda`: train fusion model.
- `python scripts/train_fusion.py --mode train --data_root /home/martina/Y3_Project/Plaintextdataset --block_modality visual`: train with visual modality blocked (ablation).
- `python scripts/train_fusion.py --mode eval --data_root /home/martina/Y3_Project/Plaintextdataset --checkpoint outputs/fusion_model_clean/best_model.pth --eval_split test`: run evaluation.
- `python collect_custom_multimodal.py --log-file outputs/logs/position_logs.json --dataset-root ../Plaintextdataset/train`: record multimodal episodes.
- `streamlit run scripts/clean_dataset_ui.py`: launch dataset cleaning UI.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and snake_case for functions/variables.
- Use `PascalCase` for classes (for example `FusionModel`, `RoboticGraspDataset`).
- Keep script names action-oriented and lowercase with underscores (for example `visualize_plaintext_dataset.py`).
- Prefer type hints for new public helpers and keep argument parsing explicit in entry scripts.

## Testing Guidelines
No dedicated `tests/` suite exists yet. For each change:
- Run an eval smoke test, e.g. `python scripts/train_fusion.py --mode eval --data_root /home/martina/Y3_Project/Plaintextdataset --checkpoint <path> --eval_split val`.
- If model code changed, run a short train/eval smoke test and confirm outputs are written under `outputs/`.
- Document dataset path assumptions in PR notes so reviewers can reproduce.

## Commit & Pull Request Guidelines
Recent history shows short, imperative commit subjects in both English and Chinese. Keep messages concise and task-focused, e.g.:
- `fix tactile padding mask length mismatch`
- `更新 README 中的数据集结构说明`

For PRs:
- Describe what changed and why.
- List exact commands run for validation.
- Link related issues/tasks.
- Include screenshots for UI/plot changes (for example Streamlit pages or confusion matrices).
