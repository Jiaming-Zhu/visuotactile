#!/bin/bash
# Multi-seed training for Fusion Gating model with entropy regularization (anti-collapse).
#
# Runs seeds: 42, 123, 456, 789, 2024
# Outputs under: visuotactile/outputs/fusion/gating/
# Summary saved to: visuotactile/outputs/meta/multi_seed_summary_gating_entropy.json

set -euo pipefail

DATA_ROOT="/home/martina/Y3_Project/Plaintextdataset"
OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs/fusion/gating2.0"
SCRIPT_DIR="/home/martina/Y3_Project/visuotactile/scripts"
SEEDS=(42 123 456 789 2024)

# Defaults (override via env vars).
# Example: DEVICE=cuda EPOCHS=50 bash visuotactile/scripts/run_multi_seed_gating_entropy.sh
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"

# Default hyperparams for the anti-collapse gate regularization.
REG_TYPE="entropy"
LAMBDA_REG="0.1"
GATE_WARMUP_EPOCHS="5"
GATE_RAMP_EPOCHS="10"

echo "==============================================================="
echo "  Multi-Seed Gating Training (reg=${REG_TYPE})"
echo "  Seeds: ${SEEDS[*]}"
echo "  Output base: ${OUTPUT_BASE}"
echo "  Device: ${DEVICE}"
echo "  Epochs: ${EPOCHS}"
echo "==============================================================="

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${OUTPUT_BASE}/fusion_gating_${REG_TYPE}_seed${SEED}"
    if [ -f "${SAVE_DIR}/eval_ood_test/evaluation_results.json" ]; then
        echo "[SKIP] seed=${SEED} already evaluated: ${SAVE_DIR}"
        continue
    fi
    echo ""
    echo ">>> seed=${SEED} -> ${SAVE_DIR}"
    python "${SCRIPT_DIR}/train_fusion_gating copy.py" \
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
        --lambda_aux "0.5" \
        --gate_reg_warmup_epochs "$GATE_WARMUP_EPOCHS" \
        --gate_reg_ramp_epochs "$GATE_RAMP_EPOCHS"
    echo "<<< seed=${SEED} done"
done

echo ""
echo "==================================================="
echo "  Aggregating results..."
echo "==================================================="

python - <<'PYEOF'
import json
import numpy as np
from pathlib import Path

output_base = Path("/home/martina/Y3_Project/visuotactile/outputs/fusion/gating2.0")
meta_dir = Path("/home/martina/Y3_Project/visuotactile/outputs/fusion/gating2.0/meta")
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
        if not result_file.exists():
            print(f"  [WARN] Missing: {result_file}")
            continue
        result = json.loads(result_file.read_text())
        for task in tasks:
            records[task].append(result[task])
        records["loss"].append(result["loss"])
        records["avg"].append(np.mean([result[t] for t in tasks]))
        records["avg_gate_score"].append(result.get("avg_gate_score", 0.0))
        used_seeds.append(seed)

    if not used_seeds:
        continue

    summary[split] = {"used_seeds": used_seeds}
    for key in tasks + ["loss", "avg", "avg_gate_score"]:
        arr = np.array(records[key], dtype=float)
        summary[split][key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "values": [float(v) for v in arr],
        }

print("\n" + "=" * 90)
print("  FUSION GATING (entropy) MULTI-SEED RESULTS (mean ± std, n<=5)")
print("=" * 90)

for split in splits:
    if split not in summary:
        print(f"\n{split}: no available results")
        continue
    split_label = "Test (In-Distribution)" if split == "eval_test" else "OOD Test (Out-of-Distribution)"
    s = summary[split]
    print(f"\n{'-' * 90}")
    print(f"  {split_label}")
    print(f"  Seeds used: {s['used_seeds']}")
    print(f"{'-' * 90}")
    print(
        "  "
        f"Mass={s['mass']['mean']*100:.2f}±{s['mass']['std']*100:.2f}% | "
        f"Stiffness={s['stiffness']['mean']*100:.2f}±{s['stiffness']['std']*100:.2f}% | "
        f"Material={s['material']['mean']*100:.2f}±{s['material']['std']*100:.2f}% | "
        f"Avg={s['avg']['mean']*100:.2f}±{s['avg']['std']*100:.2f}% | "
        f"Gate={s['avg_gate_score']['mean']:.3f}±{s['avg_gate_score']['std']:.3f}"
    )

out_path = meta_dir / "multi_seed_summary_gating_entropy.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nFull results saved to: {out_path}")
PYEOF

echo ""
echo "Done!"
