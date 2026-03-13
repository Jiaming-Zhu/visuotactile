import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("matplotlib is required to generate plots") from exc


DEFAULT_RUNS_ROOT = Path("/home/martina/Y3_Project/visuotactile/outputs/fusion_gating_online_v2_multiseed")
DEFAULT_GLOB = "fusion_gating_online_v2_seed*"
DEFAULT_SPLITS = "test,ood_test"
DEFAULT_EVAL_DIR_TEMPLATE = "online_eval_{split}_fine"
DEFAULT_METRIC = "gate_score"

SPLIT_STYLES = {
    "test": {"label": "Test", "color": "#0072B2", "marker": "o"},
    "ood_test": {"label": "OOD", "color": "#D55E00", "marker": "s"},
    "val": {"label": "Val", "color": "#009E73", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot multi-seed online prefix curves for multiple splits on one figure."
    )
    parser.add_argument("--runs_root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--glob", type=str, default=DEFAULT_GLOB)
    parser.add_argument("--splits", type=str, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--eval_dir_template",
        type=str,
        default=DEFAULT_EVAL_DIR_TEMPLATE,
        help="Template used to resolve per-split online eval dir, e.g. online_eval_{split}_fine",
    )
    parser.add_argument(
        "--metric",
        choices=["average_accuracy", "mass", "stiffness", "material", "gate_score", "loss"],
        default=DEFAULT_METRIC,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Defaults to <runs_root>/plots_online_prefix_multisplit",
    )
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--show_error_bars", action="store_true", default=True)
    parser.add_argument("--hide_error_bars", dest="show_error_bars", action="store_false")
    return parser.parse_args()


def parse_seed_label(name: str) -> str:
    match = re.search(r"seed(\d+)", name)
    return match.group(1) if match else name


def sort_seed_labels(labels: Sequence[str]) -> List[str]:
    def key_fn(label: str):
        return (0, int(label)) if label.isdigit() else (1, label)

    return sorted(labels, key=key_fn)


def parse_splits(raw: str) -> List[str]:
    splits = [item.strip() for item in raw.split(",") if item.strip()]
    if not splits:
        raise ValueError("at least one split is required")
    return splits


def load_curves_for_split(
    runs_root: Path,
    glob_pattern: str,
    split: str,
    eval_dir_template: str,
) -> Dict[str, Dict[float, Dict[str, float]]]:
    curves: Dict[str, Dict[float, Dict[str, float]]] = {}
    eval_dir_name = eval_dir_template.format(split=split)
    for run_dir in sorted(runs_root.glob(glob_pattern)):
        if not run_dir.is_dir():
            continue
        result_path = run_dir / eval_dir_name / "online_evaluation_results.json"
        if not result_path.exists():
            continue
        data = json.loads(result_path.read_text(encoding="utf-8"))
        curve: Dict[float, Dict[str, float]] = {}
        for point in data.get("prefix_curves", []):
            ratio = float(point["prefix_ratio"])
            curve[ratio] = {
                "average_accuracy": float(point["average_accuracy"]),
                "mass": float(point["mass"]),
                "stiffness": float(point["stiffness"]),
                "material": float(point["material"]),
                "gate_score": float(point["gate_score"]),
                "loss": float(point["loss"]),
            }
        if curve:
            curves[parse_seed_label(run_dir.name)] = curve
    return curves


def collect_ratios(split_curves: Dict[str, Dict[str, Dict[float, Dict[str, float]]]]) -> List[float]:
    ratios = set()
    for curves in split_curves.values():
        for curve in curves.values():
            ratios.update(curve.keys())
    return sorted(ratios)


def metric_label(metric: str) -> str:
    return {
        "average_accuracy": "Average Accuracy",
        "mass": "Mass Accuracy",
        "stiffness": "Stiffness Accuracy",
        "material": "Material Accuracy",
        "gate_score": "Gate Score",
        "loss": "Loss",
    }[metric]


def y_label(metric: str) -> str:
    if metric in {"average_accuracy", "mass", "stiffness", "material"}:
        return "Accuracy (%)"
    if metric == "gate_score":
        return "Gate Score"
    return "Loss"


def maybe_scale(metric: str, values: np.ndarray) -> np.ndarray:
    if metric in {"average_accuracy", "mass", "stiffness", "material"}:
        return values * 100.0
    return values


def build_title(args: argparse.Namespace, splits: Sequence[str]) -> str:
    if args.title:
        return args.title
    split_names = " vs ".join(SPLIT_STYLES.get(split, {}).get("label", split) for split in splits)
    return f"{split_names}: Prefix Ratio vs {metric_label(args.metric)} (Mean ± Std)"


def summarise_split(
    curves: Dict[str, Dict[float, Dict[str, float]]],
    ratios: Sequence[float],
    metric: str,
) -> Dict[str, object]:
    seeds = sort_seed_labels(list(curves.keys()))
    matrix = []
    for seed in seeds:
        matrix.append([curves[seed].get(ratio, {}).get(metric, np.nan) for ratio in ratios])
    arr = np.asarray(matrix, dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    return {
        "seeds": seeds,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "values": {
            seed: {f"{ratio:.4f}": float(curves[seed][ratio][metric]) for ratio in ratios if ratio in curves[seed]}
            for seed in seeds
        },
    }


def write_csv(
    output_dir: Path,
    split_curves: Dict[str, Dict[str, Dict[float, Dict[str, float]]]],
    ratios: Sequence[float],
    metric: str,
) -> Path:
    out_path = output_dir / f"online_prefix_multisplit_{metric}.csv"
    all_seeds = sorted({seed for curves in split_curves.values() for seed in curves})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "prefix_ratio", *[f"seed_{seed}" for seed in all_seeds], "mean", "std"])
        for split, curves in split_curves.items():
            for ratio in ratios:
                row = []
                for seed in all_seeds:
                    row.append(curves.get(seed, {}).get(ratio, {}).get(metric, np.nan))
                arr = np.asarray(row, dtype=np.float64)
                writer.writerow([split, ratio, *row, float(np.nanmean(arr)), float(np.nanstd(arr))])
    return out_path


def write_manifest(
    output_dir: Path,
    split_curves: Dict[str, Dict[str, Dict[float, Dict[str, float]]]],
    ratios: Sequence[float],
    metric: str,
) -> Path:
    payload = {
        "metric": metric,
        "ratios": list(ratios),
        "splits": {},
    }
    for split, curves in split_curves.items():
        payload["splits"][split] = summarise_split(curves, ratios, metric)
    out_path = output_dir / f"online_prefix_multisplit_{metric}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def plot_curves(
    output_dir: Path,
    split_curves: Dict[str, Dict[str, Dict[float, Dict[str, float]]]],
    ratios: Sequence[float],
    args: argparse.Namespace,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    ordered_splits = parse_splits(args.splits)
    for split in ordered_splits:
        curves = split_curves.get(split, {})
        if not curves:
            continue
        summary = summarise_split(curves, ratios, args.metric)
        mean = maybe_scale(args.metric, np.asarray(summary["mean"], dtype=np.float64))
        std = maybe_scale(args.metric, np.asarray(summary["std"], dtype=np.float64))
        style = SPLIT_STYLES.get(split, {"label": split, "color": None, "marker": "o"})
        if args.show_error_bars:
            ax.errorbar(
                ratios,
                mean,
                yerr=std,
                label=f"{style['label']} mean ± std",
                color=style["color"],
                ecolor=style["color"],
                linewidth=2.4,
                marker=style["marker"],
                markersize=5,
                capsize=4,
                elinewidth=1.3,
            )
        else:
            ax.plot(
                ratios,
                mean,
                label=f"{style['label']} mean",
                color=style["color"],
                linewidth=2.4,
                marker=style["marker"],
                markersize=5,
            )

    ax.set_title(build_title(args, ordered_splits))
    ax.set_xlabel("Prefix Ratio")
    ax.set_ylabel(y_label(args.metric))
    ax.set_xticks(list(ratios))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    if args.ymin is not None or args.ymax is not None:
        ymin, ymax = ax.get_ylim()
        if args.ymin is not None:
            ymin = args.ymin
        if args.ymax is not None:
            ymax = args.ymax
        ax.set_ylim(ymin, ymax)
    elif args.metric in {"average_accuracy", "mass", "stiffness", "material"}:
        ax.set_ylim(0, 105)
    elif args.metric == "gate_score":
        ax.set_ylim(0.35, 0.85)

    fig.tight_layout()
    out_path = output_dir / f"online_prefix_multisplit_{args.metric}.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.runs_root / "plots_online_prefix_multisplit")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = parse_splits(args.splits)
    split_curves = {
        split: load_curves_for_split(args.runs_root, args.glob, split, args.eval_dir_template)
        for split in splits
    }
    split_curves = {split: curves for split, curves in split_curves.items() if curves}
    if not split_curves:
        raise FileNotFoundError("No online evaluation files found for any requested split.")

    ratios = collect_ratios(split_curves)
    if not ratios:
        raise RuntimeError("No prefix ratios found in the loaded online evaluation files.")

    png_path = plot_curves(output_dir, split_curves, ratios, args)
    csv_path = write_csv(output_dir, split_curves, ratios, args.metric)
    json_path = write_manifest(output_dir, split_curves, ratios, args.metric)

    print("=" * 70)
    print("online prefix multi-split plot generated")
    print("=" * 70)
    print(f"runs_root : {args.runs_root}")
    print(f"splits    : {', '.join(split_curves.keys())}")
    print(f"eval_dir  : {args.eval_dir_template}")
    print(f"metric    : {args.metric}")
    print(f"ratios    : {', '.join(str(r) for r in ratios)}")
    print(f"PNG       : {png_path}")
    print(f"CSV       : {csv_path}")
    print(f"JSON      : {json_path}")


if __name__ == "__main__":
    main()
