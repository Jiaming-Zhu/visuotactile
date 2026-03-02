#!/bin/bash
# Multi-seed training for all model variants:
# 1. Fusion Gating (entropy)
# 2. Standard Fusion
# 3. Vision Only
# 4. Tactile Only
#
# Runs seeds: 42, 123, 456, 789, 2024
# Combined summary saved to: visuotactile/outputs/meta/multi_seed_summary_all_models.json

set -euo pipefail

DATA_ROOT="/home/martina/Y3_Project/Plaintextdataset"
SCRIPT_DIR="/home/martina/Y3_Project/visuotactile/scripts"
SEEDS=(42 123 456 789 2024)

GATING_OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs/fusion/gating"
FUSION_OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs/fusion/standard"
VISION_OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs/vision/standard"
TACTILE_OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs/tactile/standard"

# Defaults (override via env vars).
# Example:
# DEVICE=cuda EPOCHS=50 BATCH_SIZE=64 bash scripts/run_multi_seed_all_models.sh
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EARLY_STOP_MIN_EPOCH="${EARLY_STOP_MIN_EPOCH:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
EARLY_STOP_ACC="${EARLY_STOP_ACC:-1.0}"

echo "==============================================================="
echo "  Multi-Seed All-Model Training"
echo "  Seeds: ${SEEDS[*]}"
echo "  Device: ${DEVICE}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Num workers: ${NUM_WORKERS}"
echo "  Early stop: min_epoch=${EARLY_STOP_MIN_EPOCH}, patience=${EARLY_STOP_PATIENCE}, acc=${EARLY_STOP_ACC}"
echo "==============================================================="

run_model() {
    local model_label="$1"
    local script_name="$2"
    local save_prefix="$3"
    local output_base="$4"
    shift 4
    local extra_args=("$@")

    echo ""
    echo "==============================================================="
    echo "  ${model_label}"
    echo "  Output base: ${output_base}"
    echo "==============================================================="

    for SEED in "${SEEDS[@]}"; do
        SAVE_DIR="${output_base}/${save_prefix}_seed${SEED}"
        if [ -f "${SAVE_DIR}/eval_ood_test/evaluation_results.json" ]; then
            echo "[SKIP] ${model_label} seed=${SEED} already evaluated: ${SAVE_DIR}"
            continue
        fi
        echo ""
        echo ">>> ${model_label} seed=${SEED} -> ${SAVE_DIR}"
        python "${SCRIPT_DIR}/${script_name}" \
            --mode train \
            --seed "$SEED" \
            --device "$DEVICE" \
            --data_root "$DATA_ROOT" \
            --save_dir "$SAVE_DIR" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --no_live_plot \
            --epochs "$EPOCHS" \
            --early_stop_min_epoch "$EARLY_STOP_MIN_EPOCH" \
            --early_stop_patience "$EARLY_STOP_PATIENCE" \
            --early_stop_acc "$EARLY_STOP_ACC" \
            "${extra_args[@]}"
        echo "<<< ${model_label} seed=${SEED} done"
    done
}

run_model \
    "Fusion Gating (entropy)" \
    "train_fusion_gating.py" \
    "fusion_gating_entropy" \
    "$GATING_OUTPUT_BASE" \
    --reg_type entropy \
    --lambda_reg 0.1 \
    --gate_reg_warmup_epochs 5 \
    --gate_reg_ramp_epochs 10

run_model \
    "Standard Fusion" \
    "train_fusion.py" \
    "fusion_standard" \
    "$FUSION_OUTPUT_BASE"

run_model \
    "Vision Only" \
    "train_vision.py" \
    "vision_standard" \
    "$VISION_OUTPUT_BASE"

run_model \
    "Tactile Only" \
    "train_tactile.py" \
    "tactile_standard" \
    "$TACTILE_OUTPUT_BASE"

echo ""
echo "==================================================="
echo "  Aggregating results..."
echo "==================================================="

python - <<'PYEOF'
import json
import numpy as np
from pathlib import Path

meta_dir = Path("/home/martina/Y3_Project/visuotactile/outputs/meta")
meta_dir.mkdir(parents=True, exist_ok=True)

seeds = [42, 123, 456, 789, 2024]
tasks = ["mass", "stiffness", "material"]
splits = ["eval_test", "eval_ood_test"]

models = {
    "fusion_gating_entropy": {
        "label": "Fusion Gating (entropy)",
        "base": Path("/home/martina/Y3_Project/visuotactile/outputs/fusion/gating"),
        "prefix": "fusion_gating_entropy",
        "extra_metric_keys": ["avg_gate_score"],
    },
    "fusion_standard": {
        "label": "Standard Fusion",
        "base": Path("/home/martina/Y3_Project/visuotactile/outputs/fusion/standard"),
        "prefix": "fusion_standard",
        "extra_metric_keys": [],
    },
    "vision_standard": {
        "label": "Vision Only",
        "base": Path("/home/martina/Y3_Project/visuotactile/outputs/vision/standard"),
        "prefix": "vision_standard",
        "extra_metric_keys": [],
    },
    "tactile_standard": {
        "label": "Tactile Only",
        "base": Path("/home/martina/Y3_Project/visuotactile/outputs/tactile/standard"),
        "prefix": "tactile_standard",
        "extra_metric_keys": [],
    },
}

summary = {}
for model_key, cfg in models.items():
    summary[model_key] = {"label": cfg["label"]}
    for split in splits:
        records = {"mass": [], "stiffness": [], "material": [], "loss": [], "avg": []}
        for extra_key in cfg["extra_metric_keys"]:
            records[extra_key] = []

        used_seeds = []
        for seed in seeds:
            result_file = cfg["base"] / f"{cfg['prefix']}_seed{seed}" / split / "evaluation_results.json"
            if not result_file.exists():
                print(f"  [WARN] Missing: {result_file}")
                continue
            result = json.loads(result_file.read_text())
            for task in tasks:
                records[task].append(float(result[task]))
            records["loss"].append(float(result["loss"]))
            records["avg"].append(float(np.mean([result[t] for t in tasks])))
            for extra_key in cfg["extra_metric_keys"]:
                records[extra_key].append(float(result.get(extra_key, 0.0)))
            used_seeds.append(seed)

        if not used_seeds:
            continue

        summary[model_key][split] = {"used_seeds": used_seeds}
        for key, values in records.items():
            arr = np.array(values, dtype=float)
            summary[model_key][split][key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "values": [float(v) for v in arr],
            }

print("\n" + "=" * 100)
print("  ALL MODELS MULTI-SEED RESULTS (mean ± std, n<=5)")
print("=" * 100)

for split in splits:
    split_label = "Test (In-Distribution)" if split == "eval_test" else "OOD Test (Out-of-Distribution)"
    print(f"\n{'-' * 100}")
    print(f"  {split_label}")
    print(f"{'-' * 100}")
    for model_key in ["fusion_gating_entropy", "fusion_standard", "vision_standard", "tactile_standard"]:
        model_summary = summary.get(model_key, {})
        if split not in model_summary:
            print(f"  {model_summary.get('label', model_key)}: no available results")
            continue
        s = model_summary[split]
        line = (
            f"  {model_summary['label']}: "
            f"Mass={s['mass']['mean']*100:.2f}±{s['mass']['std']*100:.2f}% | "
            f"Stiffness={s['stiffness']['mean']*100:.2f}±{s['stiffness']['std']*100:.2f}% | "
            f"Material={s['material']['mean']*100:.2f}±{s['material']['std']*100:.2f}% | "
            f"Avg={s['avg']['mean']*100:.2f}±{s['avg']['std']*100:.2f}%"
        )
        if "avg_gate_score" in s:
            line += f" | Gate={s['avg_gate_score']['mean']:.3f}±{s['avg_gate_score']['std']:.3f}"
        print(line)
        print(f"    seeds: {s['used_seeds']}")

out_path = meta_dir / "multi_seed_summary_all_models.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nFull results saved to: {out_path}")
PYEOF

echo ""
echo "Done!"
