#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/martina/Y3_Project_7z_extracted/Y3_Project"
DATA_ROOT="${ROOT}/Plaintextdataset"
TRAIN_SCRIPT="${ROOT}/visuotactile/scripts/train_fusion_gating_online_reliable.py"
OUTPUT_ROOT="${ROOT}/visuotactile/outputs/fixed_gate_grid_20260421"
LOG_ROOT="${OUTPUT_ROOT}/logs"

SEEDS=("${@:-42}")
GATES=(0.00 0.01 0.02 0.05 0.10 0.15 0.20 0.30)

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

run_variant() {
    local seed="$1"
    local gate="$2"
    local gate_tag
    gate_tag="$(python3 -c 'import sys; print(f"{int(round(float(sys.argv[1]) * 100)):03d}")' "${gate}")"
    local family_dir="${OUTPUT_ROOT}/fusion_fixed_gate_g${gate_tag}"
    local save_dir="${family_dir}/fusion_fixed_gate_g${gate_tag}_seed${seed}"
    local log_file="${LOG_ROOT}/fusion_fixed_gate_g${gate_tag}_seed${seed}.log"

    mkdir -p "${family_dir}"

    if [ -f "${save_dir}/eval_ood_test/evaluation_results.json" ]; then
        echo "[SKIP] fixed_gate=${gate} seed=${seed} already complete"
        return
    fi

    echo ""
    echo ">>> fixed_gate=${gate} seed=${seed}"
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
        --fixed_gate_value "${gate}" \
        --primary_checkpoint acc \
        --visual_mismatch_prob 0.0 \
        --lambda_mismatch_gate 0.0 \
        --lambda_reg 0.0 \
        --reg_type none >"${log_file}" 2>&1
    echo "<<< fixed_gate=${gate} seed=${seed} done (log: ${log_file})"
}

for seed in "${SEEDS[@]}"; do
    for gate in "${GATES[@]}"; do
        run_variant "${seed}" "${gate}"
    done
done

echo ""
echo "All requested fixed-gate runs have been launched/completed."
