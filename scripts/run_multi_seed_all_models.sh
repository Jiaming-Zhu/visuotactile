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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

python "${SCRIPT_DIR}/aggregate_multi_seed_results.py" \
    --meta_dir "/home/martina/Y3_Project/visuotactile/outputs/meta" \
    --out_name "multi_seed_summary_all_models.json" \
    --title "All Models Multi-Seed Results" \
    --model "fusion_gating_entropy:Fusion Gating (entropy):${GATING_OUTPUT_BASE}:fusion_gating_entropy:avg_gate_score" \
    --model "fusion_standard:Standard Fusion:${FUSION_OUTPUT_BASE}:fusion_standard" \
    --model "vision_standard:Vision Only:${VISION_OUTPUT_BASE}:vision_standard" \
    --model "tactile_standard:Tactile Only:${TACTILE_OUTPUT_BASE}:tactile_standard"

echo ""
echo "Done!"
