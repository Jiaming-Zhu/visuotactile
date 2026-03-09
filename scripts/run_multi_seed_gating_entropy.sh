#!/bin/bash
# Multi-seed training for Fusion Gating model with entropy regularization (anti-collapse).
#
# Runs seeds: 42, 123, 456, 789, 2024
# Outputs under: visuotactile/outputs/fusion/gating/
# Summary saved to: visuotactile/outputs/meta/multi_seed_summary_gating_entropy.json

set -euo pipefail

DATA_ROOT="/home/martina/Y3_Project/Plaintextdataset"
OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs/fusion/gating2.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=(42 123 456 789 2024)

# Defaults (override via env vars).
# Example: DEVICE=cuda EPOCHS=50 bash visuotactile/scripts/run_multi_seed_gating_entropy.sh
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"

# Default hyperparams for the anti-collapse gate regularization.
REG_TYPE="entropy"
LAMBDA_REG="0.1"
GATE_WARMUP_EPOCHS="5"
GATE_RAMP_EPOCHS="10"

echo "==============================================================="
echo "  Multi-Seed Gating Training (reg=${REG_TYPE})"
echo "  Seeds: ${SEEDS[*]}"
echo "  Output base: ${OUTPUT_BASE}"
echo "  Device: ${DEVICE}"
echo "  Epochs: ${EPOCHS}"
echo "==============================================================="

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${OUTPUT_BASE}/fusion_gating_${REG_TYPE}_seed${SEED}"
    if [ -f "${SAVE_DIR}/eval_ood_test/evaluation_results.json" ]; then
        echo "[SKIP] seed=${SEED} already evaluated: ${SAVE_DIR}"
        continue
    fi
    echo ""
    echo ">>> seed=${SEED} -> ${SAVE_DIR}"
    python "${SCRIPT_DIR}/train_fusion_gating2.py" \
        --mode train \
        --seed "$SEED" \
        --device "$DEVICE" \
        --data_root "$DATA_ROOT" \
        --save_dir "$SAVE_DIR" \
        --num_workers 0 \
        --no_live_plot \
        --epochs "$EPOCHS" \
        --early_stop_min_epoch 10 \
        --early_stop_patience 3 \
        --early_stop_acc 1.0 \
        --reg_type "$REG_TYPE" \
        --lambda_reg "$LAMBDA_REG" \
        --lambda_aux "0.5" \
        --gate_reg_warmup_epochs "$GATE_WARMUP_EPOCHS" \
        --gate_reg_ramp_epochs "$GATE_RAMP_EPOCHS"
    echo "<<< seed=${SEED} done"
done

echo ""
echo "==================================================="
echo "  Aggregating results..."
echo "==================================================="

python "${SCRIPT_DIR}/aggregate_multi_seed_results.py" \
    --meta_dir "${OUTPUT_BASE}/meta" \
    --out_name "multi_seed_summary_gating_entropy.json" \
    --title "Fusion Gating (Entropy) Multi-Seed Results" \
    --model "fusion_gating_entropy:Fusion Gating (entropy):${OUTPUT_BASE}:fusion_gating_entropy:avg_gate_score"

echo ""
echo "Done!"
