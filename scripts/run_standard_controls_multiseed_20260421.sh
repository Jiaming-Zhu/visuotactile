#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/martina/Y3_Project_7z_extracted/Y3_Project"
DATA_ROOT="${ROOT}/Plaintextdataset"
TRAIN_SCRIPT="${ROOT}/visuotactile/scripts/train_fusion_standard_ablation.py"
OUTPUT_ROOT="${ROOT}/visuotactile/outputs/standard_controls_multiseed_20260421"
LOG_ROOT="${OUTPUT_ROOT}/logs"
SEEDS=(123 456 789 2024)

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

link_seed42() {
    local family_dir="$1"
    local prefix="$2"
    local formal_dir="$3"
    mkdir -p "${family_dir}"
    if [ ! -e "${family_dir}/${prefix}_seed42" ]; then
        ln -s "${formal_dir}" "${family_dir}/${prefix}_seed42"
    fi
}

run_variant() {
    local family_dir="$1"
    local prefix="$2"
    local formal_dir="$3"
    shift 3
    local extra_args=("$@")

    link_seed42 "${family_dir}" "${prefix}" "${formal_dir}"

    for seed in "${SEEDS[@]}"; do
        local save_dir="${family_dir}/${prefix}_seed${seed}"
        local log_file="${LOG_ROOT}/${prefix}_seed${seed}.log"
        if [ -f "${save_dir}/eval_ood_test/evaluation_results.json" ]; then
            echo "[SKIP] ${prefix} seed=${seed} already complete"
            continue
        fi

        echo ""
        echo ">>> ${prefix} seed=${seed}"
        conda run -n Y3 python "${TRAIN_SCRIPT}" \
            --mode train \
            --data_root "${DATA_ROOT}" \
            --save_dir "${save_dir}" \
            --variant_name "${prefix}" \
            --epochs 50 \
            --batch_size 16 \
            --num_workers 8 \
            --seed "${seed}" \
            --device cuda \
            "${extra_args[@]}" >"${log_file}" 2>&1
        echo "<<< ${prefix} seed=${seed} done (log: ${log_file})"
    done
}

run_variant \
    "${OUTPUT_ROOT}/fusion_standard_baseline" \
    "fusion_standard_baseline" \
    "${ROOT}/visuotactile/outputs/fusion_standard_baseline_formal_20260413"

run_variant \
    "${OUTPUT_ROOT}/fusion_standard_online_prefix" \
    "fusion_standard_online_prefix" \
    "${ROOT}/visuotactile/outputs/fusion_standard_online_prefix_formal_20260413" \
    --online_train_prob 1.0 \
    --online_min_prefix_ratio 0.2 \
    --min_prefix_len 64

run_variant \
    "${OUTPUT_ROOT}/fusion_standard_aux_supcon" \
    "fusion_standard_aux_supcon" \
    "${ROOT}/visuotactile/outputs/fusion_standard_aux_supcon_formal_20260413" \
    --lambda_aux 0.5 \
    --lambda_supcon 0.1

run_variant \
    "${OUTPUT_ROOT}/fusion_standard_visual_aug" \
    "fusion_standard_visual_aug" \
    "${ROOT}/visuotactile/outputs/fusion_standard_visual_aug_formal_20260413" \
    --augment_policy classical

echo ""
echo "All missing standard-control multi-seed runs have been launched/completed."
