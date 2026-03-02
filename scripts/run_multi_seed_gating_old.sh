#!/bin/bash
set -euo pipefail

DATA_ROOT="/home/martina/Y3_Project/Plaintextdataset"
OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs/fusion/gating_old"
SCRIPT_DIR="/home/martina/Y3_Project/visuotactile/scripts"
SEEDS=(42 123 456 789 2024)

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
REG_TYPE="entropy"
LAMBDA_REG="0.1"
GATE_WARMUP_EPOCHS="5"
GATE_RAMP_EPOCHS="10"

echo "==============================================================="
echo "  Multi-Seed Gating Training OLD MODEL (reg=${REG_TYPE})"
echo "==============================================================="

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${OUTPUT_BASE}/fusion_gating_${REG_TYPE}_seed${SEED}"
    if [ -f "${SAVE_DIR}/eval_ood_test/evaluation_results.json" ]; then
        echo "[SKIP] seed=${SEED} already evaluated: ${SAVE_DIR}"
        continue
    fi
    echo ">>> seed=${SEED} -> ${SAVE_DIR}"
    python "${SCRIPT_DIR}/train_fusion_gating.py" \
        --mode train \
        --seed "$SEED" \
        --device "$DEVICE" \
        --data_root "$DATA_ROOT" \
        --save_dir "$SAVE_DIR" \
        --num_workers 0 \
        --no_live_plot \
        --epochs "$EPOCHS" \
        --early_stop_min_epoch 10 \
        --early_stop_patience 3 \
        --early_stop_acc 1.0 \
        --reg_type "$REG_TYPE" \
        --lambda_reg "$LAMBDA_REG" \
        --gate_reg_warmup_epochs "$GATE_WARMUP_EPOCHS" \
        --gate_reg_ramp_epochs "$GATE_RAMP_EPOCHS"
done

python - <<'PYEOF'
import json
import numpy as np
from pathlib import Path

output_base = Path("/home/martina/Y3_Project/visuotactile/outputs/fusion/gating_old")
meta_dir = Path("/home/martina/Y3_Project/visuotactile/outputs/meta")
meta_dir.mkdir(parents=True, exist_ok=True)

seeds = [42, 123, 456, 789, 2024]
seed_dirs = {s: output_base / f"fusion_gating_entropy_seed{s}" for s in seeds}
splits = ["eval_test", "eval_ood_test"]
tasks = ["mass", "stiffness", "material"]

summary = {}
for split in splits:
    records = {"mass": [], "stiffness": [], "material": [], "loss": [], "avg": [], "avg_gate_score": []}
    used_seeds = []
    for seed, base_dir in sorted(seed_dirs.items()):
        result_file = base_dir / split / "evaluation_results.json"
        if result_file.exists():
            result = json.loads(result_file.read_text())
            for task in tasks:
                records[task].append(result[task])
            records["loss"].append(result["loss"])
            records["avg"].append(np.mean([result[t] for t in tasks]))
            records["avg_gate_score"].append(result.get("avg_gate_score", 0.0))
            used_seeds.append(seed)

    if used_seeds:
        summary[split] = {"used_seeds": used_seeds}
        for key in tasks + ["loss", "avg", "avg_gate_score"]:
            arr = np.array(records[key], dtype=float)
            summary[split][key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "values": [float(v) for v in arr]}

out_path = meta_dir / "multi_seed_summary_gating_entropy_old.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
PYEOF
