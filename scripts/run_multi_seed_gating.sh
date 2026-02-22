#!/bin/bash
# Multi-seed training for Fusion Gating model.
# Seed 42 results are expected in fusion_model_gating.

set -euo pipefail

DATA_ROOT="/home/martina/Y3_Project/Plaintextdataset"
OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs"
SCRIPT_DIR="/home/martina/Y3_Project/visuotactile/scripts"
SEEDS=(123 456 789 2024)

echo "==============================================================="
echo "  Multi-Seed Gating Training (seeds: 42*, 123, 456, 789, 2024)"
echo "  * seed 42 expected in fusion_model_gating"
echo "==============================================================="

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${OUTPUT_BASE}/fusion_gating_seed${SEED}"
    if [ -d "${SAVE_DIR}/eval_ood_test" ]; then
        echo "[SKIP] Fusion Gating seed=${SEED} already done"
        continue
    fi
    echo ""
    echo ">>> Fusion Gating seed=${SEED}"
    python "${SCRIPT_DIR}/train_fusion_gating.py" \
        --mode train \
        --seed "$SEED" \
        --data_root "$DATA_ROOT" \
        --save_dir "$SAVE_DIR" \
        --num_workers 0 \
        --no_live_plot
    echo "<<< Fusion Gating seed=${SEED} done"
done

echo ""
echo "==================================================="
echo "  Gating training complete. Aggregating results..."
echo "==================================================="

python3 - <<'PYEOF'
import json
import numpy as np
from pathlib import Path

output_base = Path("/home/martina/Y3_Project/visuotactile/outputs")
seed_dirs = {
    42: output_base / "fusion_model_gating",
    123: output_base / "fusion_gating_seed123",
    456: output_base / "fusion_gating_seed456",
    789: output_base / "fusion_gating_seed789",
    2024: output_base / "fusion_gating_seed2024",
}
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
print("  FUSION GATING MULTI-SEED RESULTS (mean ± std, n<=5)")
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

out_path = output_base / "multi_seed_summary_gating.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nFull results saved to: {out_path}")
PYEOF

echo ""
echo "Done!"
