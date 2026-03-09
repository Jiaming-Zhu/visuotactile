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
DEFAULT_SPLIT = "ood_test"
DEFAULT_METRIC = "average_accuracy"

SEED_COLORS = (
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
)
MEAN_COLOR = "#0072B2"
MEAN_MARKER_FACE = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot multi-seed online prefix curves (accuracy vs prefix ratio)"
    )
    parser.add_argument("--runs_root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--glob", type=str, default=DEFAULT_GLOB)
    parser.add_argument("--split", choices=["val", "test", "ood_test"], default=DEFAULT_SPLIT)
    parser.add_argument(
        "--eval_dir_name",
        type=str,
        default="",
        help="Custom online eval directory name under each seed run, e.g. online_eval_ood_test_fine. "
        "Defaults to online_eval_<split>.",
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
        help="Defaults to <runs_root>/plots_online_prefix",
    )
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show_seed_lines", action="store_true", default=True)
    parser.add_argument("--hide_seed_lines", dest="show_seed_lines", action="store_false")
    parser.add_argument("--show_mean", action="store_true", default=True)
    parser.add_argument("--hide_mean", dest="show_mean", action="store_false")
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    return parser.parse_args()


def parse_seed_label(name: str) -> str:
    match = re.search(r"seed(\d+)", name)
    return match.group(1) if match else name


def sort_seed_labels(labels: Sequence[str]) -> List[str]:
    def key_fn(label: str):
        return (0, int(label)) if label.isdigit() else (1, label)

    return sorted(labels, key=key_fn)


def load_curves(
    runs_root: Path,
    glob_pattern: str,
    split: str,
    eval_dir_name: str = "",
) -> Dict[str, Dict[float, Dict[str, float]]]:
    curves: Dict[str, Dict[float, Dict[str, float]]] = {}
    resolved_eval_dir = eval_dir_name or f"online_eval_{split}"
    for run_dir in sorted(runs_root.glob(glob_pattern)):
        if not run_dir.is_dir():
            continue
        result_path = run_dir / resolved_eval_dir / "online_evaluation_results.json"
        if not result_path.exists():
            continue
        data = json.loads(result_path.read_text(encoding="utf-8"))
        seed_label = parse_seed_label(run_dir.name)
        curve = {}
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
            curves[seed_label] = curve
    if not curves:
        raise FileNotFoundError(
            f"No online evaluation files found under {runs_root} with glob '{glob_pattern}' for split '{split}'"
        )
    return curves


def collect_ratios(curves: Dict[str, Dict[float, Dict[str, float]]]) -> List[float]:
    ratios = set()
    for curve in curves.values():
        ratios.update(curve.keys())
    return sorted(ratios)


def build_title(args: argparse.Namespace) -> str:
    if args.title:
        return args.title
    metric_name = {
        "average_accuracy": "Average Accuracy",
        "mass": "Mass Accuracy",
        "stiffness": "Stiffness Accuracy",
        "material": "Material Accuracy",
        "gate_score": "Gate Score",
        "loss": "Loss",
    }[args.metric]
    split_name = "OOD Test" if args.split == "ood_test" else ("Test" if args.split == "test" else "Val")
    return f"{split_name}: Prefix Ratio vs {metric_name} (Multi-Seed)"


def write_csv(
    output_dir: Path,
    curves: Dict[str, Dict[float, Dict[str, float]]],
    ratios: List[float],
    metric: str,
) -> Path:
    out_path = output_dir / f"online_prefix_multiseed_{metric}.csv"
    seeds = sort_seed_labels(list(curves.keys()))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prefix_ratio", *[f"seed_{seed}" for seed in seeds], "mean", "std"])
        for ratio in ratios:
            values = [curves[seed].get(ratio, {}).get(metric, np.nan) for seed in seeds]
            arr = np.asarray(values, dtype=np.float64)
            writer.writerow(
                [ratio, *values, float(np.nanmean(arr)), float(np.nanstd(arr))]
            )
    return out_path


def plot_curves(
    output_dir: Path,
    curves: Dict[str, Dict[float, Dict[str, float]]],
    ratios: List[float],
    args: argparse.Namespace,
) -> Path:
    seeds = sort_seed_labels(list(curves.keys()))
    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    if args.metric in {"average_accuracy", "mass", "stiffness", "material"}:
        ylabel = "Accuracy (%)"
    elif args.metric == "gate_score":
        ylabel = "Gate Score"
    else:
        ylabel = "Loss"

    if args.show_seed_lines:
        for idx, seed in enumerate(seeds):
            y = [curves[seed].get(ratio, {}).get(args.metric, np.nan) for ratio in ratios]
            y = np.asarray(y, dtype=np.float64)
            y_plot = y * 100.0 if args.metric in {"average_accuracy", "mass", "stiffness", "material"} else y

            ax.plot(
                ratios,
                y_plot,
                marker="o",
                linewidth=1.8,
                markersize=5,
                color=SEED_COLORS[idx % len(SEED_COLORS)],
                alpha=0.9,
                label=f"seed {seed}",
            )

    if args.show_mean:
        matrix = []
        for seed in seeds:
            row = [curves[seed].get(ratio, {}).get(args.metric, np.nan) for ratio in ratios]
            matrix.append(row)
        arr = np.asarray(matrix, dtype=np.float64)
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        if args.metric in {"average_accuracy", "mass", "stiffness", "material"}:
            mean = mean * 100.0
            std = std * 100.0
        ax.errorbar(
            ratios,
            mean,
            yerr=std,
            color=MEAN_COLOR,
            ecolor=MEAN_COLOR,
            linewidth=2.6,
            marker="s",
            markersize=5,
            markerfacecolor=MEAN_MARKER_FACE,
            markeredgecolor=MEAN_COLOR,
            markeredgewidth=1.4,
            capsize=4,
            elinewidth=1.4,
            label="mean ± std",
        )

    ax.set_title(build_title(args))
    ax.set_xlabel("Prefix Ratio")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ratios)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)

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
        ax.set_ylim(0.4, 0.8)

    fig.tight_layout()
    out_path = output_dir / f"online_prefix_multiseed_{args.metric}.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_manifest(
    output_dir: Path,
    curves: Dict[str, Dict[float, Dict[str, float]]],
    ratios: List[float],
    metric: str,
) -> Path:
    seeds = sort_seed_labels(list(curves.keys()))
    payload = {
        "metric": metric,
        "seeds": seeds,
        "ratios": ratios,
        "values": {
            seed: {f"{ratio:.4f}": float(curves[seed][ratio][metric]) for ratio in ratios if ratio in curves[seed]}
            for seed in seeds
        },
    }
    out_path = output_dir / f"online_prefix_multiseed_{metric}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.runs_root / "plots_online_prefix")
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = load_curves(args.runs_root, args.glob, args.split, args.eval_dir_name)
    ratios = collect_ratios(curves)

    png_path = plot_curves(output_dir, curves, ratios, args)
    csv_path = write_csv(output_dir, curves, ratios, args.metric)
    json_path = write_manifest(output_dir, curves, ratios, args.metric)

    print("=" * 70)
    print("online prefix multi-seed plot generated")
    print("=" * 70)
    print(f"runs_root : {args.runs_root}")
    print(f"split     : {args.split}")
    print(f"eval_dir  : {args.eval_dir_name or f'online_eval_{args.split}'}")
    print(f"metric    : {args.metric}")
    print(f"seeds     : {', '.join(sort_seed_labels(list(curves.keys())))}")
    print(f"ratios    : {', '.join(str(r) for r in ratios)}")
    print(f"PNG       : {png_path}")
    print(f"CSV       : {csv_path}")
    print(f"JSON      : {json_path}")


if __name__ == "__main__":
    main()
