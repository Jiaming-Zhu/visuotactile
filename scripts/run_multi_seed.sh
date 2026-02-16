#!/bin/bash
# Multi-seed training for Fusion / Vision / Tactile models
# Seed 42 results already exist in *_model_clean directories.
# This script runs additional seeds, then aggregates all results.

set -e

DATA_ROOT="/home/martina/Y3_Project/Plaintextdataset"
OUTPUT_BASE="/home/martina/Y3_Project/visuotactile/outputs"
SCRIPT_DIR="/home/martina/Y3_Project/visuotactile/scripts"
SEEDS=(123 456 789 2024)

echo "======================================================="
echo "  Multi-Seed Training (seeds: 42*, 123, 456, 789, 2024)"
echo "  * seed 42 already exists in *_model_clean dirs"
echo "======================================================="

# ---- Fusion ----
for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${OUTPUT_BASE}/fusion_seed${SEED}"
    if [ -d "${SAVE_DIR}/eval_ood_test" ]; then
        echo "[SKIP] Fusion seed=${SEED} already done"
        continue
    fi
    echo ""
    echo ">>> Fusion seed=${SEED}"
    python "${SCRIPT_DIR}/train_fusion.py" \
        --mode train \
        --seed "$SEED" \
        --data_root "$DATA_ROOT" \
        --save_dir "$SAVE_DIR" \
        --no_live_plot
    echo "<<< Fusion seed=${SEED} done"
done

# ---- Vision Only ----
for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${OUTPUT_BASE}/vision_seed${SEED}"
    if [ -d "${SAVE_DIR}/eval_ood_test" ]; then
        echo "[SKIP] Vision seed=${SEED} already done"
        continue
    fi
    echo ""
    echo ">>> Vision seed=${SEED}"
    python "${SCRIPT_DIR}/train_vision.py" \
        --mode train \
        --seed "$SEED" \
        --data_root "$DATA_ROOT" \
        --save_dir "$SAVE_DIR" \
        --no_live_plot
    echo "<<< Vision seed=${SEED} done"
done

# ---- Tactile Only ----
for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${OUTPUT_BASE}/tactile_seed${SEED}"
    if [ -d "${SAVE_DIR}/eval_ood_test" ]; then
        echo "[SKIP] Tactile seed=${SEED} already done"
        continue
    fi
    echo ""
    echo ">>> Tactile seed=${SEED}"
    python "${SCRIPT_DIR}/train_tactile.py" \
        --mode train \
        --seed "$SEED" \
        --data_root "$DATA_ROOT" \
        --save_dir "$SAVE_DIR" \
        --no_live_plot
    echo "<<< Tactile seed=${SEED} done"
done

echo ""
echo "============================================"
echo "  All training complete. Aggregating results..."
echo "============================================"

# ---- Aggregate results ----
python3 - <<'PYEOF'
import json
import numpy as np
from pathlib import Path

output_base = Path("/home/martina/Y3_Project/visuotactile/outputs")

models = {
    "Fusion": {
        42:   output_base / "fusion_model_clean",
        123:  output_base / "fusion_seed123",
        456:  output_base / "fusion_seed456",
        789:  output_base / "fusion_seed789",
        2024: output_base / "fusion_seed2024",
    },
    "Vision": {
        42:   output_base / "vision_model_clean",
        123:  output_base / "vision_seed123",
        456:  output_base / "vision_seed456",
        789:  output_base / "vision_seed789",
        2024: output_base / "vision_seed2024",
    },
    "Tactile": {
        42:   output_base / "tactile_model_clean",
        123:  output_base / "tactile_seed123",
        456:  output_base / "tactile_seed456",
        789:  output_base / "tactile_seed789",
        2024: output_base / "tactile_seed2024",
    },
}

splits = ["eval_test", "eval_ood_test"]
tasks = ["mass", "stiffness", "material"]

summary = {}

for model_name, seed_dirs in models.items():
    summary[model_name] = {}
    for split in splits:
        records = {"mass": [], "stiffness": [], "material": [], "loss": [], "avg": []}
        for seed, base_dir in sorted(seed_dirs.items()):
            result_file = base_dir / split / "evaluation_results.json"
            if not result_file.exists():
                print(f"  [WARN] Missing: {result_file}")
                continue
            r = json.loads(result_file.read_text())
            for t in tasks:
                records[t].append(r[t])
            records["loss"].append(r["loss"])
            records["avg"].append(np.mean([r[t] for t in tasks]))

        if not records["avg"]:
            continue

        summary[model_name][split] = {}
        for key in tasks + ["loss", "avg"]:
            arr = np.array(records[key])
            summary[model_name][split][key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "values": [float(v) for v in arr],
            }

# Print table
print("\n" + "=" * 80)
print("  MULTI-SEED RESULTS (mean ± std, n=5 seeds)")
print("=" * 80)

for split in splits:
    split_label = "Test (In-Distribution)" if split == "eval_test" else "OOD Test (Out-of-Distribution)"
    print(f"\n{'─' * 80}")
    print(f"  {split_label}")
    print(f"{'─' * 80}")
    print(f"  {'Model':<12} {'Mass':<18} {'Stiffness':<18} {'Material':<18} {'Avg Acc':<18}")
    print(f"  {'─'*12} {'─'*18} {'─'*18} {'─'*18} {'─'*18}")

    for model_name in ["Fusion", "Vision", "Tactile"]:
        if split not in summary.get(model_name, {}):
            print(f"  {model_name:<12} (results missing)")
            continue
        s = summary[model_name][split]
        row = f"  {model_name:<12}"
        for key in tasks + ["avg"]:
            m = s[key]["mean"] * 100
            sd = s[key]["std"] * 100
            row += f" {m:5.2f}±{sd:4.2f}%    "
        print(row)

    # Print per-seed detail
    print(f"\n  Per-seed breakdown:")
    for model_name in ["Fusion", "Vision", "Tactile"]:
        if split not in summary.get(model_name, {}):
            continue
        vals = summary[model_name][split]["avg"]["values"]
        seeds_used = sorted(models[model_name].keys())
        detail = ", ".join(f"s{seed}={v*100:.2f}%" for seed, v in zip(seeds_used, vals))
        print(f"    {model_name}: {detail}")

# Save to JSON
out_path = output_base / "multi_seed_summary.json"
out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nFull results saved to: {out_path}")
PYEOF

echo ""
echo "Done!"
