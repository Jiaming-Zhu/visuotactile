#!/bin/bash
# Re-run fine-grained online prefix evaluation for fusion_gating_online_v2 multi-seed runs
# and generate mean±std-only plots.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_ROOT="${DATA_ROOT:-/home/martina/Y3_Project/Plaintextdataset}"
RUNS_ROOT="${RUNS_ROOT:-/home/martina/Y3_Project/visuotactile/outputs/fusion_gating_online_v2_multiseed}"
GLOB_PATTERN="${GLOB_PATTERN:-fusion_gating_online_v2_seed*}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_TACTILE_LEN="${MAX_TACTILE_LEN:-3000}"
EVAL_SPLIT="${EVAL_SPLIT:-ood_test}"
EVAL_DIR_NAME="${EVAL_DIR_NAME:-online_eval_${EVAL_SPLIT}_fine}"
PLOT_DIR="${PLOT_DIR:-${RUNS_ROOT}/plots_online_prefix_fine}"
FINE_PREFIX_RATIOS="${FINE_PREFIX_RATIOS:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.0}"
METRICS_RAW="${METRICS:-average_accuracy}"
FORCE_RERUN="${FORCE_RERUN:-0}"

IFS=',' read -r -a METRICS <<< "${METRICS_RAW}"

echo "==============================================================="
echo "  Fine Prefix Multi-Seed Online Eval + Mean±Std Plot"
echo "  Runs root : ${RUNS_ROOT}"
echo "  Split     : ${EVAL_SPLIT}"
echo "  Device    : ${DEVICE}"
echo "  Ratios    : ${FINE_PREFIX_RATIOS}"
echo "  Metrics   : ${METRICS[*]}"
echo "==============================================================="

for RUN_DIR in "${RUNS_ROOT}"/${GLOB_PATTERN}; do
    if [ ! -d "${RUN_DIR}" ]; then
        continue
    fi

    CHECKPOINT="${RUN_DIR}/best_model.pth"
    OUTPUT_JSON="${RUN_DIR}/${EVAL_DIR_NAME}/online_evaluation_results.json"

    if [ ! -f "${CHECKPOINT}" ]; then
        echo "[WARN] missing checkpoint: ${CHECKPOINT}"
        continue
    fi

    if [ "${FORCE_RERUN}" != "1" ] && [ -f "${OUTPUT_JSON}" ]; then
        echo "[SKIP] fine online_eval already exists: ${RUN_DIR}/${EVAL_DIR_NAME}"
        continue
    fi

    echo ""
    echo ">>> fine online_eval -> ${RUN_DIR}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/train_fusion_gating_online.py" \
        --mode online_eval \
        --data_root "${DATA_ROOT}" \
        --checkpoint "${CHECKPOINT}" \
        --eval_split "${EVAL_SPLIT}" \
        --device "${DEVICE}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --max_tactile_len "${MAX_TACTILE_LEN}" \
        --prefix_ratios "${FINE_PREFIX_RATIOS}" \
        --output_dir "${RUN_DIR}/${EVAL_DIR_NAME}"
    echo "<<< done: ${RUN_DIR}"
done

mkdir -p "${PLOT_DIR}"

for METRIC in "${METRICS[@]}"; do
    METRIC="$(echo "${METRIC}" | xargs)"
    if [ -z "${METRIC}" ]; then
        continue
    fi
    echo ""
    echo ">>> plotting metric=${METRIC}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/visualization/plot_online_prefix_multiseed.py" \
        --runs_root "${RUNS_ROOT}" \
        --glob "${GLOB_PATTERN}" \
        --split "${EVAL_SPLIT}" \
        --eval_dir_name "${EVAL_DIR_NAME}" \
        --metric "${METRIC}" \
        --output_dir "${PLOT_DIR}" \
        --hide_seed_lines \
        --show_mean
done

echo ""
echo "Done!"
