#!/bin/bash
# Multi-seed training for the online-prefix gating model using the validated v2 hyperparameters.
#
# Seeds: 42, 123, 456, 789, 2024
# Reuses the existing fusion_gating_online_v2 run for seed 42 when available.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/martina/Y3_Project/Plaintextdataset}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=(42 123 456 789 2024)

# Defaults (override via env vars).
# Example:
# OUTPUT_ROOT=/home/martina/Y3_Project/visuotactile/newOutput \
# DEVICE=cuda EPOCHS=60 BATCH_SIZE=32 NUM_WORKERS=12 \
# bash scripts/run_multi_seed_gating_online_v2.sh
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/martina/Y3_Project/visuotactile/outputs}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_ROOT}/fusion_gating_online_v2_multiseed}"
META_DIR="${META_DIR:-${OUTPUT_BASE}/meta}"
REFERENCE_SEED42="${REFERENCE_SEED42:-${OUTPUT_ROOT}/fusion_gating_online_v2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_EVERY="${SAVE_EVERY:-10}"
MAX_TACTILE_LEN="${MAX_TACTILE_LEN:-3000}"
PREFIX_RATIOS="${PREFIX_RATIOS:-0.1,0.2,0.4,0.6,0.8,1.0}"
REUSE_SEED42="${REUSE_SEED42:-0}"
LAMBDA_REG="${LAMBDA_REG:-0.1}"
LAMBDA_AUX="${LAMBDA_AUX:-0.5}"
LAMBDA_SUPCON="${LAMBDA_SUPCON:-0.0}"
SUPCON_TEMPERATURE="${SUPCON_TEMPERATURE:-0.07}"
SUPCON_TASK="${SUPCON_TASK:-material}"
REG_TYPE="${REG_TYPE:-entropy}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
EARLY_STOP_ACC="${EARLY_STOP_ACC:-1.0}"
EARLY_STOP_MIN_EPOCH="${EARLY_STOP_MIN_EPOCH:-0}"
SEPARATE_CLS_TOKENS="${SEPARATE_CLS_TOKENS:-0}"

mkdir -p "${OUTPUT_BASE}"

SEED42_LINK="${OUTPUT_BASE}/fusion_gating_online_v2_seed42"
if [ "${REUSE_SEED42}" = "1" ] && [ ! -e "${SEED42_LINK}" ] && [ -d "${REFERENCE_SEED42}" ]; then
    ln -s "${REFERENCE_SEED42}" "${SEED42_LINK}"
fi

echo "==============================================================="
echo "  Multi-Seed Online Gating Training (v2 config)"
echo "  Seeds: ${SEEDS[*]}"
echo "  Output root: ${OUTPUT_ROOT}"
echo "  Output base: ${OUTPUT_BASE}"
echo "  Device: ${DEVICE}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Num workers: ${NUM_WORKERS}"
echo "  Reuse seed42: ${REUSE_SEED42}"
echo "  Gate reg: type=${REG_TYPE}, lambda_reg=${LAMBDA_REG}, lambda_aux=${LAMBDA_AUX}"
echo "  SupCon: lambda_supcon=${LAMBDA_SUPCON}, task=${SUPCON_TASK}, temp=${SUPCON_TEMPERATURE}"
echo "  Early stop: patience=${EARLY_STOP_PATIENCE}, acc=${EARLY_STOP_ACC}, min_epoch=${EARLY_STOP_MIN_EPOCH}"
echo "  Separate CLS tokens: ${SEPARATE_CLS_TOKENS}"
echo "==============================================================="

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${OUTPUT_BASE}/fusion_gating_online_v2_seed${SEED}"
    OOD_JSON="${SAVE_DIR}/eval_ood_test/evaluation_results.json"
    ONLINE_JSON="${SAVE_DIR}/online_eval_ood_test/online_evaluation_results.json"

    if [ -f "${OOD_JSON}" ] && [ -f "${ONLINE_JSON}" ]; then
        echo "[SKIP] seed=${SEED} already trained and online-evaluated: ${SAVE_DIR}"
        continue
    fi

    echo ""
    echo ">>> seed=${SEED} -> ${SAVE_DIR}"

    if [ ! -f "${OOD_JSON}" ]; then
        TRAIN_ARGS=(
            "${PYTHON_BIN}" "${SCRIPT_DIR}/train_fusion_gating_online.py"
            --mode train
            --seed "${SEED}"
            --device "${DEVICE}"
            --data_root "${DATA_ROOT}"
            --save_dir "${SAVE_DIR}"
            --batch_size "${BATCH_SIZE}"
            --epochs "${EPOCHS}"
            --lr 1e-4
            --weight_decay 0.01
            --warmup_epochs 5
            --save_every "${SAVE_EVERY}"
            --num_workers "${NUM_WORKERS}"
            --max_tactile_len "${MAX_TACTILE_LEN}"
            --fusion_dim 256
            --num_heads 8
            --dropout 0.1
            --num_layers 4
            --freeze_visual
            --visual_drop_prob 0.1
            --tactile_drop_prob 0.0
            --early_stop_patience "${EARLY_STOP_PATIENCE}"
            --early_stop_acc "${EARLY_STOP_ACC}"
            --early_stop_min_epoch "${EARLY_STOP_MIN_EPOCH}"
            --lambda_reg "${LAMBDA_REG}"
            --lambda_aux "${LAMBDA_AUX}"
            --lambda_supcon "${LAMBDA_SUPCON}"
            --supcon_temperature "${SUPCON_TEMPERATURE}"
            --supcon_task "${SUPCON_TASK}"
            --reg_type "${REG_TYPE}"
            --gate_target_mean 0.5
            --gate_entropy_eps 1e-6
            --gate_reg_warmup_epochs 5
            --gate_reg_ramp_epochs 10
            --online_train_prob 0.6
            --online_min_prefix_ratio 0.4
            --min_prefix_len 64
            --prefix_ratios "${PREFIX_RATIOS}"
            --no_live_plot
        )
        if [ "${SEPARATE_CLS_TOKENS}" = "1" ]; then
            TRAIN_ARGS+=(--separate_cls_tokens)
        fi
        "${TRAIN_ARGS[@]}"
    fi

    if [ ! -f "${ONLINE_JSON}" ]; then
        "${PYTHON_BIN}" "${SCRIPT_DIR}/train_fusion_gating_online.py" \
            --mode online_eval \
            --data_root "${DATA_ROOT}" \
            --checkpoint "${SAVE_DIR}/best_model.pth" \
            --eval_split ood_test \
            --device "${DEVICE}" \
            --batch_size "${BATCH_SIZE}" \
            --num_workers "${NUM_WORKERS}" \
            --max_tactile_len "${MAX_TACTILE_LEN}" \
            --prefix_ratios "${PREFIX_RATIOS}" \
            --output_dir "${SAVE_DIR}/online_eval_ood_test"
    fi

    echo "<<< seed=${SEED} done"
done

echo ""
echo "==================================================="
echo "  Aggregating test / ood_test results..."
echo "==================================================="

"${PYTHON_BIN}" "${SCRIPT_DIR}/aggregate_multi_seed_results.py" \
    --meta_dir "${META_DIR}" \
    --out_name "multi_seed_summary_gating_online_v2.json" \
    --title "Fusion Gating Online v2 Multi-Seed Results" \
    --model "fusion_gating_online_v2:Fusion Gating Online v2:${OUTPUT_BASE}:fusion_gating_online_v2:avg_gate_score"

echo ""
echo "Done!"
