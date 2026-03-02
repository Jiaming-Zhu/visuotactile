#!/bin/bash
# Multi-seed training for single-modality baselines:
# 1. Vision Only
# 2. Tactile Only
#
# Runs seeds: 42, 123, 456, 789, 2024
# Combined summary saved to: visuotactile/outputs/meta/multi_seed_summary_single_modal.json

set -euo pipefail

DATA_ROOT="/home/martina/Y3_Project/Plaintextdataset"
SCRIPT_DIR="/home/martina/Y3_Project/visuotactile/scripts"
SEEDS=(42 123 456 789 2024)

VISION_OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs/vision/standard"
TACTILE_OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs/tactile/standard"

# Defaults (override via env vars).
# Example:
# DEVICE=cuda EPOCHS=50 BATCH_SIZE=64 bash scripts/run_multi_seed_single_modal.sh
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EARLY_STOP_MIN_EPOCH="${EARLY_STOP_MIN_EPOCH:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
EARLY_STOP_ACC="${EARLY_STOP_ACC:-1.0}"

echo "==============================================================="
echo "  Multi-Seed Single-Modal Training"
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
            --early_stop_acc "$EARLY_STOP_ACC"
        echo "<<< ${model_label} seed=${SEED} done"
    done
}

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
    "vision_standard": {
        "label": "Vision Only",
        "base": Path("/home/martina/Y3_Project/visuotactile/outputs/vision/standard"),
        "prefix": "vision_standard",
    },
    "tactile_standard": {
        "label": "Tactile Only",
        "base": Path("/home/martina/Y3_Project/visuotactile/outputs/tactile/standard"),
        "prefix": "tactile_standard",
    },
}

summary = {}
for model_key, cfg in models.items():
    summary[model_key] = {"label": cfg["label"]}
    for split in splits:
        records = {"mass": [], "stiffness": [], "material": [], "loss": [], "avg": []}
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

print("\n" + "=" * 90)
print("  SINGLE-MODAL MULTI-SEED RESULTS (mean ± std, n<=5)")
print("=" * 90)

for split in splits:
    split_label = "Test (In-Distribution)" if split == "eval_test" else "OOD Test (Out-of-Distribution)"
    print(f"\n{'-' * 90}")
    print(f"  {split_label}")
    print(f"{'-' * 90}")
    for model_key in ["vision_standard", "tactile_standard"]:
        model_summary = summary.get(model_key, {})
        if split not in model_summary:
            print(f"  {model_summary.get('label', model_key)}: no available results")
            continue
        s = model_summary[split]
        print(
            f"  {model_summary['label']}: "
            f"Mass={s['mass']['mean']*100:.2f}±{s['mass']['std']*100:.2f}% | "
            f"Stiffness={s['stiffness']['mean']*100:.2f}±{s['stiffness']['std']*100:.2f}% | "
            f"Material={s['material']['mean']*100:.2f}±{s['material']['std']*100:.2f}% | "
            f"Avg={s['avg']['mean']*100:.2f}±{s['avg']['std']*100:.2f}%"
        )
        print(f"    seeds: {s['used_seeds']}")

out_path = meta_dir / "multi_seed_summary_single_modal.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nFull results saved to: {out_path}")
PYEOF

echo ""
echo "Done!"
