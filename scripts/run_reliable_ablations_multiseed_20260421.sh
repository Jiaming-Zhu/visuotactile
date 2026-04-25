#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/martina/Y3_Project_7z_extracted/Y3_Project"
DATA_ROOT="${ROOT}/Plaintextdataset"
TRAIN_SCRIPT="${ROOT}/visuotactile/scripts/train_fusion_gating_online_reliable.py"
OUTPUT_ROOT="${ROOT}/visuotactile/outputs/reliable_ablations_multiseed_20260421"
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
            --epochs 50 \
            --batch_size 16 \
            --num_workers 8 \
            --seed "${seed}" \
            --device cuda \
            --no_live_plot \
            "${extra_args[@]}" >"${log_file}" 2>&1
        echo "<<< ${prefix} seed=${seed} done (log: ${log_file})"
    done
}

run_variant \
    "${OUTPUT_ROOT}/fusion_gating_online_reliable_ablation_nomismatch" \
    "fusion_gating_online_reliable_ablation_nomismatch" \
    "${ROOT}/visuotactile/outputs/fusion_gating_online_reliable_ablation_nomismatch_seed42_20260419" \
    --visual_mismatch_prob 0.0 \
    --lambda_mismatch_gate 0.0

run_variant \
    "${OUTPUT_ROOT}/fusion_gating_online_reliable_ablation_noaux" \
    "fusion_gating_online_reliable_ablation_noaux" \
    "${ROOT}/visuotactile/outputs/fusion_gating_online_reliable_ablation_noaux_seed42_20260419" \
    --lambda_aux 0.0

run_variant \
    "${OUTPUT_ROOT}/fusion_gating_online_reliable_ablation_noreg" \
    "fusion_gating_online_reliable_ablation_noreg" \
    "${ROOT}/visuotactile/outputs/fusion_gating_online_reliable_ablation_noreg_seed42_20260419" \
    --lambda_reg 0.0 \
    --reg_type none

echo ""
echo "All missing reliable ablation multi-seed runs have been launched/completed."
