#!/bin/bash
# Multi-seed training for standard Fusion model.
#
# Runs seeds: 42, 123, 456, 789, 2024
# Outputs under: visuotactile/outputs/fusion/standard/
# Summary saved to: visuotactile/outputs/meta/multi_seed_summary_standard.json

set -euo pipefail

DATA_ROOT="/home/martina/Y3_Project/Plaintextdataset"
OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs/fusion/standard"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=(42 123 456 789 2024)

# Defaults (override via env vars).
# Example: DEVICE=cuda EPOCHS=50 BATCH_SIZE=64 bash scripts/run_multi_seed.sh
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EARLY_STOP_MIN_EPOCH="${EARLY_STOP_MIN_EPOCH:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
EARLY_STOP_ACC="${EARLY_STOP_ACC:-1.0}"

echo "==============================================================="
echo "  Multi-Seed Standard Fusion Training"
echo "  Seeds: ${SEEDS[*]}"
echo "  Output base: ${OUTPUT_BASE}"
echo "  Device: ${DEVICE}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Early stop: min_epoch=${EARLY_STOP_MIN_EPOCH}, patience=${EARLY_STOP_PATIENCE}, acc=${EARLY_STOP_ACC}"
echo "==============================================================="

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${OUTPUT_BASE}/fusion_standard_seed${SEED}"
    if [ -f "${SAVE_DIR}/eval_ood_test/evaluation_results.json" ]; then
        echo "[SKIP] seed=${SEED} already evaluated: ${SAVE_DIR}"
        continue
    fi
    echo ""
    echo ">>> seed=${SEED} -> ${SAVE_DIR}"
    python "${SCRIPT_DIR}/train_fusion.py" \
        --mode train \
        --seed "$SEED" \
        --device "$DEVICE" \
        --data_root "$DATA_ROOT" \
        --save_dir "$SAVE_DIR" \
        --batch_size "$BATCH_SIZE" \
        --num_workers 0 \
        --no_live_plot \
        --epochs "$EPOCHS" \
        --early_stop_min_epoch "$EARLY_STOP_MIN_EPOCH" \
        --early_stop_patience "$EARLY_STOP_PATIENCE" \
        --early_stop_acc "$EARLY_STOP_ACC"
    echo "<<< seed=${SEED} done"
done

echo ""
echo "==================================================="
echo "  Aggregating results..."
echo "==================================================="

python "${SCRIPT_DIR}/aggregate_multi_seed_results.py" \
    --meta_dir "/home/martina/Y3_Project/visuotactile/outputs/meta" \
    --out_name "multi_seed_summary_standard.json" \
    --title "Standard Fusion Multi-Seed Results" \
    --model "fusion_standard:Standard Fusion:${OUTPUT_BASE}:fusion_standard"

echo ""
echo "Done!"
