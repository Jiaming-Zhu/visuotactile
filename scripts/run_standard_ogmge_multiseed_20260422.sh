#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/martina/Y3_Project_7z_extracted/Y3_Project"
DATA_ROOT="${ROOT}/Plaintextdataset"
TRAIN_SCRIPT="${ROOT}/visuotactile/scripts/train_fusion_standard_ogmge.py"
OUTPUT_ROOT="${ROOT}/visuotactile/outputs/standard_ogmge_multiseed_20260422"
LOG_ROOT="${OUTPUT_ROOT}/logs"
SEEDS=(42 123 456 789 2024)

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

for seed in "${SEEDS[@]}"; do
    save_dir="${OUTPUT_ROOT}/fusion_standard_ogmge_seed${seed}"
    log_file="${LOG_ROOT}/fusion_standard_ogmge_seed${seed}.log"

    if [ -f "${save_dir}/eval_ood_test/evaluation_results.json" ]; then
        echo "[SKIP] standard_ogmge seed=${seed} already complete"
        continue
    fi

    echo ""
    echo ">>> standard_ogmge seed=${seed}"
    conda run -n Y3 python "${TRAIN_SCRIPT}" \
        --mode train \
        --data_root "${DATA_ROOT}" \
        --save_dir "${save_dir}" \
        --variant_name "standard_ogmge" \
        --epochs 50 \
        --batch_size 16 \
        --num_workers 8 \
        --seed "${seed}" \
        --device cuda \
        --lambda_proxy 0.5 \
        --ogmge_alpha 1.0 \
        --ogmge_min_scale 0.05 \
        --ogmge_noise_std 0.0 >"${log_file}" 2>&1
    echo "<<< standard_ogmge seed=${seed} done (log: ${log_file})"
done

echo ""
echo "All OGM-GE multi-seed runs have been launched/completed."
