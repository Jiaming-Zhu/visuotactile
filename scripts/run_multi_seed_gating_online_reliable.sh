#!/bin/bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/jiaming/Y3_Project/Plaintextdataset}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=(42 123 456 789 2024)

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/jiaming/Y3_Project/visuotactile/outputs}"
OUTPUT_BASE="${OUTPUT_BASE:-${OUTPUT_ROOT}/fusion_gating_online_reliable_multiseed}"
META_DIR="${META_DIR:-${OUTPUT_BASE}/meta}"
PLOTS_DIR="${PLOTS_DIR:-${OUTPUT_BASE}/plots_online_prefix_multisplit}"
REFERENCE_SEED42="${REFERENCE_SEED42:-${OUTPUT_ROOT}/fusion_gating_online_reliable_formal_20260412}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_EVERY="${SAVE_EVERY:-10}"
MAX_TACTILE_LEN="${MAX_TACTILE_LEN:-3000}"
REUSE_SEED42="${REUSE_SEED42:-1}"

VISUAL_MISMATCH_PROB="${VISUAL_MISMATCH_PROB:-0.25}"
LAMBDA_MISMATCH_GATE="${LAMBDA_MISMATCH_GATE:-0.2}"
RELIABLE_SELECTION_START_EPOCH="${RELIABLE_SELECTION_START_EPOCH:-16}"
PRIMARY_CHECKPOINT="${PRIMARY_CHECKPOINT:-reliable}"

mkdir -p "${OUTPUT_BASE}" "${META_DIR}" "${PLOTS_DIR}"

SEED42_LINK="${OUTPUT_BASE}/fusion_gating_online_reliable_seed42"
if [ "${REUSE_SEED42}" = "1" ] && [ ! -e "${SEED42_LINK}" ] && [ -d "${REFERENCE_SEED42}" ]; then
    ln -s "${REFERENCE_SEED42}" "${SEED42_LINK}"
fi

echo "==============================================================="
echo "  Multi-Seed Reliable Online Gating"
echo "  Seeds: ${SEEDS[*]}"
echo "  Output base: ${OUTPUT_BASE}"
echo "  Reference seed42: ${REFERENCE_SEED42}"
echo "  Reuse seed42: ${REUSE_SEED42}"
echo "  Device: ${DEVICE}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Num workers: ${NUM_WORKERS}"
echo "==============================================================="

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${OUTPUT_BASE}/fusion_gating_online_reliable_seed${SEED}"
    OOD_JSON="${SAVE_DIR}/eval_ood_test/evaluation_results.json"
    TEST_ONLINE_JSON="${SAVE_DIR}/online_eval_test/online_evaluation_results.json"
    OOD_ONLINE_JSON="${SAVE_DIR}/online_eval_ood_test/online_evaluation_results.json"
    CHECKPOINT_PATH="${SAVE_DIR}/best_model.pth"

    if [ -f "${OOD_JSON}" ] && [ -f "${TEST_ONLINE_JSON}" ] && [ -f "${OOD_ONLINE_JSON}" ]; then
        echo "[SKIP] seed=${SEED} already finished: ${SAVE_DIR}"
        continue
    fi

    echo ""
    echo ">>> seed=${SEED} -> ${SAVE_DIR}"

    if [ ! -f "${OOD_JSON}" ]; then
        "${PYTHON_BIN}" "${SCRIPT_DIR}/train_fusion_gating_online_reliable.py" \
            --mode train \
            --seed "${SEED}" \
            --device "${DEVICE}" \
            --data_root "${DATA_ROOT}" \
            --save_dir "${SAVE_DIR}" \
            --epochs "${EPOCHS}" \
            --batch_size "${BATCH_SIZE}" \
            --num_workers "${NUM_WORKERS}" \
            --max_tactile_len "${MAX_TACTILE_LEN}" \
            --save_every "${SAVE_EVERY}" \
            --no_live_plot \
            --visual_mismatch_prob "${VISUAL_MISMATCH_PROB}" \
            --lambda_mismatch_gate "${LAMBDA_MISMATCH_GATE}" \
            --reliable_selection_start_epoch "${RELIABLE_SELECTION_START_EPOCH}" \
            --primary_checkpoint "${PRIMARY_CHECKPOINT}"
    else
        echo "[SKIP TRAIN] seed=${SEED} eval outputs already exist"
    fi

    if [ ! -f "${CHECKPOINT_PATH}" ]; then
        echo "[ERROR] missing checkpoint for seed=${SEED}: ${CHECKPOINT_PATH}" >&2
        exit 1
    fi

    if [ ! -f "${TEST_ONLINE_JSON}" ]; then
        "${PYTHON_BIN}" "${SCRIPT_DIR}/train_fusion_gating_online_reliable.py" \
            --mode online_eval \
            --device "${DEVICE}" \
            --data_root "${DATA_ROOT}" \
            --checkpoint "${CHECKPOINT_PATH}" \
            --eval_split test \
            --prefix_ratios "0.1,0.2,0.4,0.6,0.8,1.0"
    else
        echo "[SKIP ONLINE TEST] seed=${SEED} already has ${TEST_ONLINE_JSON}"
    fi

    if [ ! -f "${OOD_ONLINE_JSON}" ]; then
        "${PYTHON_BIN}" "${SCRIPT_DIR}/train_fusion_gating_online_reliable.py" \
            --mode online_eval \
            --device "${DEVICE}" \
            --data_root "${DATA_ROOT}" \
            --checkpoint "${CHECKPOINT_PATH}" \
            --eval_split ood_test \
            --prefix_ratios "0.1,0.2,0.4,0.6,0.8,1.0"
    else
        echo "[SKIP ONLINE OOD] seed=${SEED} already has ${OOD_ONLINE_JSON}"
    fi

    echo "<<< seed=${SEED} done"
done

echo ""
echo "==============================================================="
echo "  Aggregating reliable multi-seed results"
echo "==============================================================="

"${PYTHON_BIN}" "${SCRIPT_DIR}/aggregate_multi_seed_reliable.py" \
    --runs_root "${OUTPUT_BASE}" \
    --output_dir "${META_DIR}" \
    --output_name "multi_seed_summary_reliable.json"

"${PYTHON_BIN}" "${SCRIPT_DIR}/visualization/plot_online_prefix_multisplit.py" \
    --runs_root "${OUTPUT_BASE}" \
    --glob "fusion_gating_online_reliable_seed*" \
    --splits "test,ood_test" \
    --eval_dir_template "online_eval_{split}" \
    --metric "average_accuracy" \
    --output_dir "${PLOTS_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/visualization/plot_online_prefix_multisplit.py" \
    --runs_root "${OUTPUT_BASE}" \
    --glob "fusion_gating_online_reliable_seed*" \
    --splits "test,ood_test" \
    --eval_dir_template "online_eval_{split}" \
    --metric "gate_score" \
    --output_dir "${PLOTS_DIR}"

echo ""
echo "Done!"
