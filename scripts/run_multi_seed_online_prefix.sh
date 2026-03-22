#!/bin/bash
# Multi-seed training for prefix-aware baselines:
# 1. Fusion Online Prefix
# 2. Tactile Online Prefix
#
# Runs seeds: 42, 123, 456, 789, 2024
# Each seed also runs online_eval on ood_test.

set -euo pipefail

DATA_ROOT="/home/martina/Y3_Project/Plaintextdataset"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=(42 123 456 789 2024)

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_EVERY="${SAVE_EVERY:-10}"
MAX_TACTILE_LEN="${MAX_TACTILE_LEN:-3000}"
EARLY_STOP_MIN_EPOCH="${EARLY_STOP_MIN_EPOCH:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
EARLY_STOP_ACC="${EARLY_STOP_ACC:-1.0}"
ONLINE_TRAIN_PROB="${ONLINE_TRAIN_PROB:-1.0}"
ONLINE_MIN_PREFIX_RATIO="${ONLINE_MIN_PREFIX_RATIO:-0.2}"
MIN_PREFIX_LEN="${MIN_PREFIX_LEN:-64}"
PREFIX_RATIOS="${PREFIX_RATIOS:-0.1,0.2,0.4,0.6,0.8,1.0}"

FUSION_OUTPUT_BASE="${FUSION_OUTPUT_BASE:-/home/martina/Y3_Project/visuotactile/outputs/fusion/online_prefix}"
TACTILE_OUTPUT_BASE="${TACTILE_OUTPUT_BASE:-/home/martina/Y3_Project/visuotactile/outputs/tactile/online_prefix}"

RUN_FUSION="${RUN_FUSION:-1}"
RUN_TACTILE="${RUN_TACTILE:-1}"
FUSION_BLOCK_MODALITY="${FUSION_BLOCK_MODALITY:-none}"
FUSION_FREEZE_VISUAL="${FUSION_FREEZE_VISUAL:-1}"
VISUAL_DROP_PROB="${VISUAL_DROP_PROB:-0.0}"
TACTILE_DROP_PROB_FUSION="${TACTILE_DROP_PROB_FUSION:-0.0}"
TACTILE_DROP_PROB_TACTILE="${TACTILE_DROP_PROB_TACTILE:-0.0}"

mkdir -p "${FUSION_OUTPUT_BASE}" "${TACTILE_OUTPUT_BASE}"

echo "==============================================================="
echo "  Multi-Seed Prefix-Aware Baselines"
echo "  Seeds: ${SEEDS[*]}"
echo "  Device: ${DEVICE}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Num workers: ${NUM_WORKERS}"
echo "  Online train: prob=${ONLINE_TRAIN_PROB}, min_ratio=${ONLINE_MIN_PREFIX_RATIO}, min_len=${MIN_PREFIX_LEN}"
echo "  Online eval ratios: ${PREFIX_RATIOS}"
echo "==============================================================="

run_fusion() {
    echo ""
    echo "==============================================================="
    echo "  Fusion Online Prefix"
    echo "  Output base: ${FUSION_OUTPUT_BASE}"
    echo "==============================================================="

    local freeze_flag="--freeze_visual"
    if [ "${FUSION_FREEZE_VISUAL}" != "1" ]; then
        freeze_flag="--unfreeze_visual"
    fi

    for SEED in "${SEEDS[@]}"; do
        SAVE_DIR="${FUSION_OUTPUT_BASE}/fusion_online_prefix_seed${SEED}"
        OOD_JSON="${SAVE_DIR}/eval_ood_test/evaluation_results.json"
        ONLINE_JSON="${SAVE_DIR}/online_eval_ood_test/online_evaluation_results.json"

        if [ -f "${OOD_JSON}" ] && [ -f "${ONLINE_JSON}" ]; then
            echo "[SKIP] Fusion seed=${SEED} already trained and online-evaluated: ${SAVE_DIR}"
            continue
        fi

        echo ""
        echo ">>> Fusion seed=${SEED} -> ${SAVE_DIR}"

        if [ ! -f "${OOD_JSON}" ]; then
            "${PYTHON_BIN}" "${SCRIPT_DIR}/train_fusion_online.py" \
                --mode train \
                --seed "${SEED}" \
                --device "${DEVICE}" \
                --data_root "${DATA_ROOT}" \
                --save_dir "${SAVE_DIR}" \
                --batch_size "${BATCH_SIZE}" \
                --num_workers "${NUM_WORKERS}" \
                --epochs "${EPOCHS}" \
                --save_every "${SAVE_EVERY}" \
                --max_tactile_len "${MAX_TACTILE_LEN}" \
                --early_stop_min_epoch "${EARLY_STOP_MIN_EPOCH}" \
                --early_stop_patience "${EARLY_STOP_PATIENCE}" \
                --early_stop_acc "${EARLY_STOP_ACC}" \
                --online_train_prob "${ONLINE_TRAIN_PROB}" \
                --online_min_prefix_ratio "${ONLINE_MIN_PREFIX_RATIO}" \
                --min_prefix_len "${MIN_PREFIX_LEN}" \
                --prefix_ratios "${PREFIX_RATIOS}" \
                --block_modality "${FUSION_BLOCK_MODALITY}" \
                --visual_drop_prob "${VISUAL_DROP_PROB}" \
                --tactile_drop_prob "${TACTILE_DROP_PROB_FUSION}" \
                "${freeze_flag}" \
                --no_live_plot
        fi

        if [ ! -f "${ONLINE_JSON}" ]; then
            "${PYTHON_BIN}" "${SCRIPT_DIR}/train_fusion_online.py" \
                --mode online_eval \
                --data_root "${DATA_ROOT}" \
                --checkpoint "${SAVE_DIR}/best_model.pth" \
                --eval_split ood_test \
                --device "${DEVICE}" \
                --batch_size "${BATCH_SIZE}" \
                --num_workers "${NUM_WORKERS}" \
                --max_tactile_len "${MAX_TACTILE_LEN}" \
                --prefix_ratios "${PREFIX_RATIOS}" \
                --block_modality "${FUSION_BLOCK_MODALITY}" \
                --output_dir "${SAVE_DIR}/online_eval_ood_test"
        fi

        echo "<<< Fusion seed=${SEED} done"
    done
}

run_tactile() {
    echo ""
    echo "==============================================================="
    echo "  Tactile Online Prefix"
    echo "  Output base: ${TACTILE_OUTPUT_BASE}"
    echo "==============================================================="

    for SEED in "${SEEDS[@]}"; do
        SAVE_DIR="${TACTILE_OUTPUT_BASE}/tactile_online_prefix_seed${SEED}"
        OOD_JSON="${SAVE_DIR}/eval_ood_test/evaluation_results.json"
        ONLINE_JSON="${SAVE_DIR}/online_eval_ood_test/online_evaluation_results.json"

        if [ -f "${OOD_JSON}" ] && [ -f "${ONLINE_JSON}" ]; then
            echo "[SKIP] Tactile seed=${SEED} already trained and online-evaluated: ${SAVE_DIR}"
            continue
        fi

        echo ""
        echo ">>> Tactile seed=${SEED} -> ${SAVE_DIR}"

        if [ ! -f "${OOD_JSON}" ]; then
            "${PYTHON_BIN}" "${SCRIPT_DIR}/train_tactile_online.py" \
                --mode train \
                --seed "${SEED}" \
                --device "${DEVICE}" \
                --data_root "${DATA_ROOT}" \
                --save_dir "${SAVE_DIR}" \
                --batch_size "${BATCH_SIZE}" \
                --num_workers "${NUM_WORKERS}" \
                --epochs "${EPOCHS}" \
                --save_every "${SAVE_EVERY}" \
                --max_tactile_len "${MAX_TACTILE_LEN}" \
                --early_stop_min_epoch "${EARLY_STOP_MIN_EPOCH}" \
                --early_stop_patience "${EARLY_STOP_PATIENCE}" \
                --early_stop_acc "${EARLY_STOP_ACC}" \
                --online_train_prob "${ONLINE_TRAIN_PROB}" \
                --online_min_prefix_ratio "${ONLINE_MIN_PREFIX_RATIO}" \
                --min_prefix_len "${MIN_PREFIX_LEN}" \
                --prefix_ratios "${PREFIX_RATIOS}" \
                --tactile_drop_prob "${TACTILE_DROP_PROB_TACTILE}" \
                --no_live_plot
        fi

        if [ ! -f "${ONLINE_JSON}" ]; then
            "${PYTHON_BIN}" "${SCRIPT_DIR}/train_tactile_online.py" \
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

        echo "<<< Tactile seed=${SEED} done"
    done
}

if [ "${RUN_FUSION}" = "1" ]; then
    run_fusion
fi

if [ "${RUN_TACTILE}" = "1" ]; then
    run_tactile
fi

echo ""
echo "==================================================="
echo "  Aggregating test / ood_test results..."
echo "==================================================="

AGG_ARGS=(
    --meta_dir "/home/martina/Y3_Project/visuotactile/outputs/meta"
    --out_name "multi_seed_summary_online_prefix.json"
    --title "Online Prefix Multi-Seed Results"
)

if [ "${RUN_FUSION}" = "1" ]; then
    AGG_ARGS+=(--model "fusion_online_prefix:Fusion Online Prefix:${FUSION_OUTPUT_BASE}:fusion_online_prefix")
fi

if [ "${RUN_TACTILE}" = "1" ]; then
    AGG_ARGS+=(--model "tactile_online_prefix:Tactile Online Prefix:${TACTILE_OUTPUT_BASE}:tactile_online_prefix")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/aggregate_multi_seed_results.py" "${AGG_ARGS[@]}"

echo ""
echo "Done!"
