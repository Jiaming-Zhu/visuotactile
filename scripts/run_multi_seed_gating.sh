#!/bin/bash
# Multi-seed training for Fusion Gating model.
# Seed 42 results are expected in fusion_model_gating.

set -euo pipefail

DATA_ROOT="/home/martina/Y3_Project/Plaintextdataset"
OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=(123 456 789 2024)

echo "==============================================================="
echo "  Multi-Seed Gating Training (seeds: 42*, 123, 456, 789, 2024)"
echo "  * seed 42 expected in fusion_model_gating"
echo "==============================================================="

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${OUTPUT_BASE}/fusion_gating_seed${SEED}"
    if [ -d "${SAVE_DIR}/eval_ood_test" ]; then
        echo "[SKIP] Fusion Gating seed=${SEED} already done"
        continue
    fi
    echo ""
    echo ">>> Fusion Gating seed=${SEED}"
    python "${SCRIPT_DIR}/train_fusion_gating.py" \
        --mode train \
        --seed "$SEED" \
        --data_root "$DATA_ROOT" \
        --save_dir "$SAVE_DIR" \
        --num_workers 0 \
        --no_live_plot
    echo "<<< Fusion Gating seed=${SEED} done"
done

echo ""
echo "==================================================="
echo "  Gating training complete. Aggregating results..."
echo "==================================================="

python3 "${SCRIPT_DIR}/aggregate_multi_seed_results.py" \
    --meta_dir "${OUTPUT_BASE}" \
    --out_name "multi_seed_summary_gating.json" \
    --title "Fusion Gating Multi-Seed Results" \
    --model "fusion_gating:Fusion Gating:${OUTPUT_BASE}:fusion_gating:avg_gate_score"
    
# Note: Seed 42 for this script is saved as fusion_model_gating instead of fusion_gating_seed42.
# We'll need a quick symlink or just note this difference.
if [ ! -d "${OUTPUT_BASE}/fusion_gating_seed42" ] && [ -d "${OUTPUT_BASE}/fusion_model_gating" ]; then
    ln -s "${OUTPUT_BASE}/fusion_model_gating" "${OUTPUT_BASE}/fusion_gating_seed42"
fi

echo ""
echo "Done!"
