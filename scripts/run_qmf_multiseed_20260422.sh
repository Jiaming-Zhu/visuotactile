#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/martina/Y3_Project_7z_extracted/Y3_Project"
DATA_ROOT="${ROOT}/Plaintextdataset"
SCRIPT_DIR="${ROOT}/visuotactile/scripts"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_fusion_qmf.py"
OUTPUT_ROOT="${ROOT}/visuotactile/outputs/qmf_multiseed_20260422"
LOG_ROOT="${OUTPUT_ROOT}/logs"
META_DIR="${OUTPUT_ROOT}/meta"
SEEDS=(42 123 456 789 2024)

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${META_DIR}"

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"

echo "==============================================================="
echo "  QMF Multi-Seed Training"
echo "  Seeds: ${SEEDS[*]}"
echo "  Device: ${DEVICE}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Num workers: ${NUM_WORKERS}"
echo "==============================================================="

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${OUTPUT_ROOT}/qmf_baseline_seed${SEED}"
    LOG_FILE="${LOG_ROOT}/qmf_baseline_seed${SEED}.log"

    if [ -f "${SAVE_DIR}/eval_ood_test/evaluation_results.json" ]; then
        echo "[SKIP] qmf_baseline seed=${SEED} already complete"
        continue
    fi

    echo ""
    echo ">>> qmf_baseline seed=${SEED}"
    conda run -n Y3 python "${TRAIN_SCRIPT}" \
        --mode train \
        --data_root "${DATA_ROOT}" \
        --save_dir "${SAVE_DIR}" \
        --variant_name "qmf_baseline" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --seed "${SEED}" \
        --device "${DEVICE}" >"${LOG_FILE}" 2>&1
    echo "<<< qmf_baseline seed=${SEED} done (log: ${LOG_FILE})"
done

python "${SCRIPT_DIR}/aggregate_multi_seed_results.py" \
    --meta_dir "${META_DIR}" \
    --out_name "multi_seed_summary_qmf.json" \
    --title "QMF Multi-Seed Results" \
    --model "qmf_baseline:QMF:${OUTPUT_ROOT}:qmf_baseline:average_tactile_weight"

echo ""
echo "QMF multi-seed runs and aggregation completed."
