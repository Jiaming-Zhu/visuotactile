import argparse
import json
import numpy as np
from pathlib import Path
import sys

def aggregate_single_model(model_label, base_dir, prefix, seeds, splits, tasks, extra_metric_keys):
    """Aggregates results for a single model across multiple seeds."""
    summary = {"label": model_label}
    has_any_data = False
    
    for split in splits:
        records = {task: [] for task in tasks}
        records["loss"] = []
        records["avg"] = []
        for extra_key in extra_metric_keys:
            records[extra_key] = []

        used_seeds = []
        for seed in seeds:
            result_file = base_dir / f"{prefix}_seed{seed}" / split / "evaluation_results.json"
            if not result_file.exists():
                print(f"  [WARN] Missing: {result_file}", file=sys.stderr)
                continue
            
            try:
                result = json.loads(result_file.read_text())
                for task in tasks:
                    records[task].append(float(result[task]))
                records["loss"].append(float(result["loss"]))
                records["avg"].append(float(np.mean([result[t] for t in tasks])))
                for extra_key in extra_metric_keys:
                    records[extra_key].append(float(result.get(extra_key, 0.0)))
                used_seeds.append(seed)
                has_any_data = True
            except Exception as e:
                print(f"  [ERROR] Failed to read/parse {result_file}: {e}", file=sys.stderr)

        if not used_seeds:
            continue

        summary[split] = {"used_seeds": used_seeds}
        for key, values in records.items():
            arr = np.array(values, dtype=float)
            summary[split][key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "values": [float(v) for v in arr],
            }
            
    return summary if has_any_data else None


def print_single_model_summary(model_key, model_summary, splits):
    """Prints the aggregated summary for a specific model."""
    for split in splits:
        if split not in model_summary:
            print(f"  {model_summary.get('label', model_key)}: no available results")
            continue
        
        s = model_summary[split]
        line = (
             f"  {model_summary['label']}: "
             f"Mass={s['mass']['mean']*100:.2f}±{s['mass']['std']*100:.2f}% | "
             f"Stiffness={s['stiffness']['mean']*100:.2f}±{s['stiffness']['std']*100:.2f}% | "
             f"Material={s['material']['mean']*100:.2f}±{s['material']['std']*100:.2f}% | "
             f"Avg={s['avg']['mean']*100:.2f}±{s['avg']['std']*100:.2f}%"
        )
        if "avg_gate_score" in s:
             line += f" | Gate={s['avg_gate_score']['mean']:.3f}±{s['avg_gate_score']['std']:.3f}"
        
        print(line)
        print(f"    seeds: {s['used_seeds']}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed evaluation results.")
    parser.add_argument("--meta_dir", type=str, required=True, help="Directory to save the summary JSON.")
    parser.add_argument("--out_name", type=str, required=True, help="Filename for the summary JSON.")
    parser.add_argument("--title", type=str, default="MULTI-SEED RESULTS", help="Title for the printed summary.")
    
    # Arguments to define one or more models to aggregate
    # Format: --model "model_key:label:base_dir:prefix:[extra_keys]"
    parser.add_argument("--model", action="append", required=True, 
                        help="Model definition: key:label:base_dir:prefix:[extra,keys]")
                        
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024", help="Comma-separated list of seeds")
    parser.add_argument("--tasks", type=str, default="mass,stiffness,material", help="Comma-separated list of tasks")
    parser.add_argument("--splits", type=str, default="eval_test,eval_ood_test", help="Comma-separated list of splits")
    
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]
    
    meta_dir = Path(args.meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    for mod_idx, mod_def in enumerate(args.model):
        parts = mod_def.split(":")
        if len(parts) < 4:
            print(f"Invalid model definition: {mod_def}. Expected key:label:base_dir:prefix[:extra,keys]", file=sys.stderr)
            continue
            
        key = parts[0]
        label = parts[1]
        base_dir = Path(parts[2])
        prefix = parts[3]
        extra_keys = []
        if len(parts) > 4 and parts[4].strip():
            extra_keys = [k.strip() for k in parts[4].split(",")]
            
        mod_summ = aggregate_single_model(label, base_dir, prefix, seeds, splits, tasks, extra_keys)
        if mod_summ:
            summary[key] = mod_summ
            
    # Print unified output to console
    print("\n" + "=" * 100)
    print(f"  {args.title.upper()} (mean ± std, n<={len(seeds)})")
    print("=" * 100)

    for split in splits:
        split_label = "Test (In-Distribution)" if split == "eval_test" else "OOD Test (Out-of-Distribution)"
        print(f"\n{'-' * 100}")
        print(f"  {split_label}")
        print(f"{'-' * 100}")
        
        for key in summary.keys():
             print_single_model_summary(key, summary[key], [split])

    # Save to JSON
    if summary:
        out_path = meta_dir / args.out_name
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nFull results saved to: {out_path}")
    else:
        print("\nNo results were found to aggregate.")


if __name__ == "__main__":
    main()
