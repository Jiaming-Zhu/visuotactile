import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


TASKS = ["mass", "stiffness", "material"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate reliable multi-seed results.")
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="fusion_gating_online_reliable_seed*")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", type=str, default="multi_seed_summary_reliable.json")
    return parser.parse_args()


def parse_seed_labels(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def summarize(values: List[float]) -> Dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "values": [float(v) for v in arr],
    }


def load_eval_file(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "average_accuracy": float(data["summary"]["average_accuracy"]),
        "loss": float(data["loss"]),
        "avg_gate_score": float(data.get("avg_gate_score", 0.0)),
    }
    for task in TASKS:
        task_block = data["tasks"][task]
        result[task] = float(task_block["accuracy"])
    return result


def load_online_file(path: Path) -> Dict[str, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    curves = {}
    for point in data.get("prefix_curves", []):
        ratio_key = f"{float(point['prefix_ratio']):.4f}"
        curves[ratio_key] = {
            "average_accuracy": float(point["average_accuracy"]),
            "mass": float(point["mass"]),
            "stiffness": float(point["stiffness"]),
            "material": float(point["material"]),
            "gate_score": float(point["gate_score"]),
            "loss": float(point["loss"]),
        }
    return curves


def aggregate_split(run_dirs: Dict[int, Path], split_name: str) -> Dict[str, object]:
    used_seeds = []
    records = {metric: [] for metric in ["average_accuracy", "loss", "avg_gate_score", *TASKS]}
    for seed, run_dir in run_dirs.items():
        path = run_dir / f"eval_{split_name}" / "evaluation_results.json"
        if not path.exists():
            continue
        payload = load_eval_file(path)
        used_seeds.append(seed)
        for metric, value in payload.items():
            records[metric].append(value)
    if not used_seeds:
        return {}
    summary = {"used_seeds": used_seeds}
    for metric, values in records.items():
        summary[metric] = summarize(values)
    return summary


def aggregate_online_split(run_dirs: Dict[int, Path], split_name: str) -> Dict[str, object]:
    used_seeds = []
    seed_curves: Dict[int, Dict[str, Dict[str, float]]] = {}
    for seed, run_dir in run_dirs.items():
        path = run_dir / f"online_eval_{split_name}" / "online_evaluation_results.json"
        if not path.exists():
            continue
        seed_curves[seed] = load_online_file(path)
        used_seeds.append(seed)
    if not used_seeds:
        return {}

    ratio_keys = sorted({ratio for curves in seed_curves.values() for ratio in curves.keys()}, key=float)
    metric_names = ["average_accuracy", "mass", "stiffness", "material", "gate_score", "loss"]

    summary = {
        "used_seeds": used_seeds,
        "ratios": [float(r) for r in ratio_keys],
        "metrics": {},
    }
    for metric in metric_names:
        rows = []
        for seed in used_seeds:
            row = [seed_curves[seed][ratio][metric] for ratio in ratio_keys]
            rows.append(row)
        arr = np.asarray(rows, dtype=np.float64)
        summary["metrics"][metric] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "values": {
                str(seed): {
                    ratio: float(seed_curves[seed][ratio][metric]) for ratio in ratio_keys
                }
                for seed in used_seeds
            },
        }
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_seeds = parse_seed_labels(args.seeds)
    run_dirs: Dict[int, Path] = {}
    for seed in requested_seeds:
        matches = list(args.runs_root.glob(f"*seed{seed}"))
        if matches:
            run_dirs[seed] = matches[0]

    payload = {
        "runs_root": str(args.runs_root),
        "requested_seeds": requested_seeds,
        "available_seeds": sorted(run_dirs.keys()),
        "eval": {},
        "online": {},
    }

    for split in ["test", "ood_test"]:
        payload["eval"][split] = aggregate_split(run_dirs, split)
        payload["online"][split] = aggregate_online_split(run_dirs, split)

    out_path = args.output_dir / args.output_name
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved reliable multi-seed summary to {out_path}")


if __name__ == "__main__":
    main()
