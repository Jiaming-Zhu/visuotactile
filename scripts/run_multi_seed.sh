#!/bin/bash
# Multi-seed training for standard Fusion model.
#
# Runs seeds: 42, 123, 456, 789, 2024
# Outputs under: visuotactile/outputs/fusion/standard/
# Summary saved to: visuotactile/outputs/meta/multi_seed_summary_standard.json

set -euo pipefail

DATA_ROOT="/home/martina/Y3_Project/Plaintextdataset"
OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs/fusion/standard"
SCRIPT_DIR="/home/martina/Y3_Project/visuotactile/scripts"
SEEDS=(42 123 456 789 2024)

# Defaults (override via env vars).
# Example: DEVICE=cuda EPOCHS=50 BATCH_SIZE=64 bash scripts/run_multi_seed.sh
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EARLY_STOP_MIN_EPOCH="${EARLY_STOP_MIN_EPOCH:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
EARLY_STOP_ACC="${EARLY_STOP_ACC:-1.0}"

echo "==============================================================="
echo "  Multi-Seed Standard Fusion Training"
echo "  Seeds: ${SEEDS[*]}"
echo "  Output base: ${OUTPUT_BASE}"
echo "  Device: ${DEVICE}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Early stop: min_epoch=${EARLY_STOP_MIN_EPOCH}, patience=${EARLY_STOP_PATIENCE}, acc=${EARLY_STOP_ACC}"
echo "==============================================================="

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${OUTPUT_BASE}/fusion_standard_seed${SEED}"
    if [ -f "${SAVE_DIR}/eval_ood_test/evaluation_results.json" ]; then
        echo "[SKIP] seed=${SEED} already evaluated: ${SAVE_DIR}"
        continue
    fi
    echo ""
    echo ">>> seed=${SEED} -> ${SAVE_DIR}"
    python "${SCRIPT_DIR}/train_fusion.py" \
        --mode train \
        --seed "$SEED" \
        --device "$DEVICE" \
        --data_root "$DATA_ROOT" \
        --save_dir "$SAVE_DIR" \
        --batch_size "$BATCH_SIZE" \
        --num_workers 0 \
        --no_live_plot \
        --epochs "$EPOCHS" \
        --early_stop_min_epoch "$EARLY_STOP_MIN_EPOCH" \
        --early_stop_patience "$EARLY_STOP_PATIENCE" \
        --early_stop_acc "$EARLY_STOP_ACC"
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

output_base = Path("/home/martina/Y3_Project/visuotactile/outputs/fusion/standard")
meta_dir = Path("/home/martina/Y3_Project/visuotactile/outputs/meta")
meta_dir.mkdir(parents=True, exist_ok=True)

seeds = [42, 123, 456, 789, 2024]
seed_dirs = {s: output_base / f"fusion_standard_seed{s}" for s in seeds}
splits = ["eval_test", "eval_ood_test"]
tasks = ["mass", "stiffness", "material"]

summary = {}
for split in splits:
    records = {"mass": [], "stiffness": [], "material": [], "loss": [], "avg": []}
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
        used_seeds.append(seed)

    if not used_seeds:
        continue

    summary[split] = {"used_seeds": used_seeds}
    for key in tasks + ["loss", "avg"]:
        arr = np.array(records[key], dtype=float)
        summary[split][key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "values": [float(v) for v in arr],
        }

print("\n" + "=" * 90)
print("  STANDARD FUSION MULTI-SEED RESULTS (mean ± std, n<=5)")
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
        f"Avg={s['avg']['mean']*100:.2f}±{s['avg']['std']*100:.2f}%"
    )

out_path = meta_dir / "multi_seed_summary_standard.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nFull results saved to: {out_path}")
PYEOF

echo ""
echo "Done!"
