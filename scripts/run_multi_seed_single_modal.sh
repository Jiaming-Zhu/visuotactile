#!/bin/bash
# Multi-seed training for single-modality baselines:
# 1. Vision Only
# 2. Tactile Only
#
# Runs seeds: 42, 123, 456, 789, 2024
# Combined summary saved to: visuotactile/outputs/meta/multi_seed_summary_single_modal.json

set -euo pipefail

DATA_ROOT="/home/martina/Y3_Project/Plaintextdataset"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=(42 123 456 789 2024)

# Defaults (override via env vars).
# Example:
# OUTPUT_ROOT=/home/martina/Y3_Project/visuotactile/newOutput \
# DEVICE=cuda EPOCHS=50 BATCH_SIZE=64 NUM_WORKERS=12 \
# bash scripts/run_multi_seed_single_modal.sh
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/martina/Y3_Project/visuotactile/outputs}"
VISION_OUTPUT_BASE="${VISION_OUTPUT_BASE:-${OUTPUT_ROOT}/singleModal/vision/standard}"
TACTILE_OUTPUT_BASE="${TACTILE_OUTPUT_BASE:-${OUTPUT_ROOT}/singleModal/tactile/standard}"
META_DIR="${META_DIR:-${OUTPUT_ROOT}/meta}"
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
echo "  Output root: ${OUTPUT_ROOT}"
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

python "${SCRIPT_DIR}/aggregate_multi_seed_results.py" \
    --meta_dir "${META_DIR}" \
    --out_name "multi_seed_summary_single_modal.json" \
    --title "Single-Modal Multi-Seed Results" \
    --model "vision_standard:Vision Only:${VISION_OUTPUT_BASE}:vision_standard" \
    --model "tactile_standard:Tactile Only:${TACTILE_OUTPUT_BASE}:tactile_standard"

echo ""
echo "Done!"
