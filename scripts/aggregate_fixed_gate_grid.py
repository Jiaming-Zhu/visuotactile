import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np


TASKS = ["mass", "stiffness", "material"]
OBJECT_LEVEL_METRICS = ["object_macro_mass", "object_macro_stiffness", "object_macro_material", "object_macro_avg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate fixed-gate training results.")
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--gates", type=str, default="0.00,0.01,0.02,0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_name", type=str, default="fixed_gate_grid_summary.json")
    parser.add_argument("--markdown_name", type=str, default="")
    parser.add_argument("--object_level_subdir", type=str, default="object_level_ood_test")
    parser.add_argument("--reference_summary_json", type=Path, default=None)
    parser.add_argument("--reference_label", type=str, default="RPDF")
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_seed_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def gate_tag(gate: float) -> str:
    return f"{int(round(gate * 100)):03d}"


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
        result[task] = float(data["tasks"][task]["accuracy"])
    return result


def load_object_level_file(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    object_macro = data["object_macro"]
    bootstrap = data["grouped_bootstrap_avg"]
    return {
        "object_macro_mass": float(object_macro["mass"]),
        "object_macro_stiffness": float(object_macro["stiffness"]),
        "object_macro_material": float(object_macro["material"]),
        "object_macro_avg": float(object_macro["avg"]),
        "grouped_bootstrap_mean": float(bootstrap["mean"]),
        "grouped_bootstrap_ci_low": float(bootstrap["ci_low"]),
        "grouped_bootstrap_ci_high": float(bootstrap["ci_high"]),
        "num_objects": int(bootstrap["num_objects"]),
        "num_resamples": int(bootstrap["num_resamples"]),
        "bootstrap_seed": int(bootstrap["seed"]),
    }


def aggregate_gate_split(run_dirs: Dict[int, Path], split_name: str) -> Dict[str, object]:
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


def aggregate_gate_object_level(
    run_dirs: Dict[int, Path],
    split_name: str,
    object_level_subdir: str | None = None,
) -> Dict[str, object]:
    used_seeds = []
    records = {metric: [] for metric in OBJECT_LEVEL_METRICS}
    bootstrap_records = {
        "mean": [],
        "ci_low": [],
        "ci_high": [],
        "num_objects": [],
        "num_resamples": [],
        "seed": [],
    }
    target_subdir = object_level_subdir or f"object_level_{split_name}"

    for seed, run_dir in run_dirs.items():
        path = run_dir / target_subdir / "object_level_results.json"
        if not path.exists():
            continue
        payload = load_object_level_file(path)
        used_seeds.append(seed)
        for metric in OBJECT_LEVEL_METRICS:
            records[metric].append(payload[metric])
        bootstrap_records["mean"].append(payload["grouped_bootstrap_mean"])
        bootstrap_records["ci_low"].append(payload["grouped_bootstrap_ci_low"])
        bootstrap_records["ci_high"].append(payload["grouped_bootstrap_ci_high"])
        bootstrap_records["num_objects"].append(payload["num_objects"])
        bootstrap_records["num_resamples"].append(payload["num_resamples"])
        bootstrap_records["seed"].append(payload["bootstrap_seed"])

    if not used_seeds:
        return {}

    summary = {"used_seeds": used_seeds}
    for metric, values in records.items():
        summary[metric] = summarize(values)
    summary["grouped_bootstrap_avg"] = bootstrap_records
    return summary


def find_run_dirs(runs_root: Path, gate: float, seeds: List[int]) -> Dict[int, Path]:
    family_dir = runs_root / f"fusion_fixed_gate_g{gate_tag(gate)}"
    if not family_dir.exists():
        return {}
    found: Dict[int, Path] = {}
    for seed in seeds:
        run_dir = family_dir / f"fusion_fixed_gate_g{gate_tag(gate)}_seed{seed}"
        if run_dir.exists():
            found[seed] = run_dir
    return found


def select_best_gate(
    payload: Mapping[str, object],
    split_name: str,
    metric: str,
    source: str = "eval",
) -> Dict[str, object]:
    best_gate_key = None
    best_mean = None
    best_std = None
    for gate_key, variant_block in payload.get("variants", {}).items():
        block = variant_block.get(source, {}).get(split_name, {})
        metric_block = block.get(metric, {})
        if "mean" not in metric_block:
            continue
        mean = float(metric_block["mean"])
        if best_mean is None or mean > best_mean:
            best_gate_key = gate_key
            best_mean = mean
            best_std = float(metric_block.get("std", 0.0))
    if best_gate_key is None:
        return {}
    return {
        "gate_key": best_gate_key,
        "mean": best_mean,
        "std": best_std,
        "split": split_name,
        "metric": metric,
        "source": source,
    }


def build_paired_delta_summary(
    reference_summary: Mapping[str, object],
    variant_block: Mapping[str, object],
    split_name: str,
    metric: str,
    source: str = "eval",
) -> Dict[str, object]:
    ref_split = reference_summary.get(source, {}).get(split_name, {})
    tgt_split = variant_block.get(source, {}).get(split_name, {})
    if metric not in ref_split or metric not in tgt_split:
        return {}

    ref_seeds = list(ref_split.get("used_seeds", []))
    tgt_seeds = list(tgt_split.get("used_seeds", []))
    ref_values = list(ref_split[metric].get("values", []))
    tgt_values = list(tgt_split[metric].get("values", []))
    ref_map = {int(seed): float(value) for seed, value in zip(ref_seeds, ref_values)}
    tgt_map = {int(seed): float(value) for seed, value in zip(tgt_seeds, tgt_values)}

    common_seeds = sorted(set(ref_map.keys()) & set(tgt_map.keys()))
    if not common_seeds:
        return {}

    deltas = [ref_map[seed] - tgt_map[seed] for seed in common_seeds]
    return {
        "used_seeds": common_seeds,
        "reference_values": [ref_map[seed] for seed in common_seeds],
        "target_values": [tgt_map[seed] for seed in common_seeds],
        "deltas": deltas,
        "mean_delta": float(np.mean(np.asarray(deltas, dtype=np.float64))),
        "std_delta": float(np.std(np.asarray(deltas, dtype=np.float64))),
    }


def format_pct_block(metric_block: Mapping[str, object]) -> str:
    return f"{float(metric_block['mean']) * 100:.2f} ± {float(metric_block['std']) * 100:.2f}"


def build_markdown_summary(payload: Mapping[str, object], reference_label: str) -> str:
    lines = [
        f"# Fixed-Gate Multi-Seed Summary ({payload.get('date_label', 'dense-grid')})",
        "",
        "Runs root:",
        f"`{payload['runs_root']}`",
        "",
        "Seeds:",
        "`" + ", ".join(str(seed) for seed in payload.get("requested_seeds", [])) + "`",
        "",
        "## Episode-level OOD (`ood_test`) five-seed results",
        "",
        "| Method | OOD Avg. | Mass | Stiffness | Material |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for gate_key, variant_block in payload.get("variants", {}).items():
        eval_block = variant_block.get("eval", {}).get("ood_test", {})
        if not eval_block:
            continue
        lines.append(
            f"| Fixed gate `g={gate_key}` | "
            f"`{format_pct_block(eval_block['average_accuracy'])}` | "
            f"`{eval_block['mass']['mean'] * 100:.2f}` | "
            f"`{eval_block['stiffness']['mean'] * 100:.2f}` | "
            f"`{eval_block['material']['mean'] * 100:.2f}` |"
        )

    best_eval = payload.get("best_gate", {}).get("eval_ood_test_average_accuracy", {})
    if best_eval:
        lines.extend(
            [
                "",
                "Best fixed gate by episode-level OOD mean:",
                f"- `g={best_eval['gate_key']}` with `{best_eval['mean'] * 100:.2f} ± {best_eval['std'] * 100:.2f}`",
            ]
        )

    best_object = payload.get("best_gate", {}).get("object_level_ood_test_macro_avg", {})
    if best_object:
        lines.extend(
            [
                "",
                "Best fixed gate by object-level OOD macro average:",
                f"- `g={best_object['gate_key']}` with `{best_object['mean'] * 100:.2f} ± {best_object['std'] * 100:.2f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Object-level OOD macro results",
            "",
            "| Method | Object-macro Avg. | Object-macro Mass | Object-macro Stiffness | Object-macro Material |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for gate_key, variant_block in payload.get("variants", {}).items():
        object_block = variant_block.get("object_level", {}).get("ood_test", {})
        if not object_block:
            continue
        lines.append(
            f"| Fixed gate `g={gate_key}` | "
            f"`{format_pct_block(object_block['object_macro_avg'])}` | "
            f"`{object_block['object_macro_mass']['mean'] * 100:.2f}` | "
            f"`{object_block['object_macro_stiffness']['mean'] * 100:.2f}` | "
            f"`{object_block['object_macro_material']['mean'] * 100:.2f}` |"
        )

    reference = payload.get("reference", {})
    paired_best = payload.get("paired_vs_reference", {}).get("best_eval_gate", {})
    if reference and paired_best:
        lines.extend(
            [
                "",
                f"## Paired seed-wise comparison against {reference_label}",
                "",
                f"Best fixed gate by episode-level OOD mean: `g={paired_best['gate_key']}`",
                f"- Mean paired delta (`{reference_label} - fixed gate`) = `{paired_best['mean_delta'] * 100:.2f} ± {paired_best['std_delta'] * 100:.2f}`",
                f"- Seeds: `{', '.join(str(seed) for seed in paired_best['used_seeds'])}`",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_gates = parse_float_list(args.gates)
    requested_seeds = parse_seed_list(args.seeds)

    payload: Dict[str, object] = {
        "runs_root": str(args.runs_root),
        "requested_gates": requested_gates,
        "requested_seeds": requested_seeds,
        "variants": {},
    }

    for gate in requested_gates:
        gate_key = f"{gate:.2f}"
        run_dirs = find_run_dirs(args.runs_root, gate, requested_seeds)
        variant_block = {
            "gate": gate,
            "available_seeds": sorted(run_dirs.keys()),
            "eval": {},
            "object_level": {},
        }
        for split in ["test", "ood_test"]:
            variant_block["eval"][split] = aggregate_gate_split(run_dirs, split)
        variant_block["object_level"]["ood_test"] = aggregate_gate_object_level(
            run_dirs,
            "ood_test",
            object_level_subdir=args.object_level_subdir,
        )
        payload["variants"][gate_key] = variant_block

    payload["best_gate"] = {
        "eval_ood_test_average_accuracy": select_best_gate(
            payload,
            split_name="ood_test",
            metric="average_accuracy",
            source="eval",
        ),
        "object_level_ood_test_macro_avg": select_best_gate(
            payload,
            split_name="ood_test",
            metric="object_macro_avg",
            source="object_level",
        ),
    }

    if args.reference_summary_json and args.reference_summary_json.exists():
        reference_summary = json.loads(args.reference_summary_json.read_text(encoding="utf-8"))
        payload["reference"] = {
            "label": args.reference_label,
            "summary_json": str(args.reference_summary_json),
        }
        payload["paired_vs_reference"] = {}
        for gate_key, variant_block in payload["variants"].items():
            paired = build_paired_delta_summary(
                reference_summary=reference_summary,
                variant_block=variant_block,
                split_name="ood_test",
                metric="average_accuracy",
                source="eval",
            )
            if paired:
                variant_block["paired_vs_reference"] = {"ood_test_average_accuracy": paired}
        best_eval_gate = payload["best_gate"].get("eval_ood_test_average_accuracy", {})
        if best_eval_gate:
            gate_key = best_eval_gate["gate_key"]
            best_variant = payload["variants"].get(gate_key, {})
            best_paired = best_variant.get("paired_vs_reference", {}).get("ood_test_average_accuracy", {})
            if best_paired:
                payload["paired_vs_reference"]["best_eval_gate"] = {
                    "gate_key": gate_key,
                    **best_paired,
                }

    out_path = args.output_dir / args.output_name
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved fixed-gate summary to {out_path}")

    if args.markdown_name:
        markdown_path = args.output_dir / args.markdown_name
        markdown_path.write_text(
            build_markdown_summary(payload, reference_label=args.reference_label),
            encoding="utf-8",
        )
        print(f"Saved fixed-gate markdown summary to {markdown_path}")


if __name__ == "__main__":
    main()
