import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("matplotlib is required to generate plots") from exc

try:
    from plot_online_prefix_multiseed import sort_seed_labels
except ImportError:  # pragma: no cover
    from visuotactile.scripts.visualization.plot_online_prefix_multiseed import sort_seed_labels  # type: ignore


DEFAULT_OUTPUT_DIR = Path("/home/martina/Y3_Project/paperWorkSpace/ieeeconf/figures")
DEFAULT_OUTPUT_NAME = "online_prefix_multiseed_accuracy_gate_dual_axis"

MODEL_SPECS = {
    "fusion": {
        "label": "Vanilla Fusion",
        "runs_root": Path("/home/martina/Y3_Project/visuotactile/outputs/fusion/online_prefix"),
        "glob": "fusion_online_prefix_seed*",
        "eval_dir_name": "online_eval_ood_test",
        "color": "#0B6EBD",
        "marker": "o",
    },
    "tactile": {
        "label": "Proprio-only",
        "runs_root": Path("/home/martina/Y3_Project/visuotactile/outputs/tactile/online_prefix"),
        "glob": "tactile_online_prefix_seed*",
        "eval_dir_name": "online_eval_ood_test",
        "color": "#D1495B",
        "marker": "s",
    },
    "gating": {
        "label": "Ours (Gated Fusion)",
        "runs_root": Path("/home/martina/Y3_Project/visuotactile/outputs/fusion_gating_online_v2_multiseed"),
        "glob": "fusion_gating_online_v2_seed*",
        "eval_dir_name": "online_eval_ood_test",
        "color": "#2A9D8F",
        "marker": "^",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot online-prefix average accuracy curves for three multi-seed models"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="fusion,tactile,gating",
        help="Comma-separated model keys. Available: fusion,tactile,gating",
    )
    parser.add_argument("--split", choices=["val", "test", "ood_test"], default="ood_test")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--min_ratio", type=float, default=0.1)
    parser.add_argument("--max_ratio", type=float, default=1.0)
    parser.add_argument(
        "--common_ratios_only",
        action="store_true",
        default=True,
        help="Restrict plotting to prefix ratios available for every model and seed.",
    )
    parser.add_argument(
        "--allow_union_ratios",
        dest="common_ratios_only",
        action="store_false",
        help="Use the union of available prefix ratios and ignore missing points with NaN-aware stats.",
    )
    return parser.parse_args()


def validate_model_keys(keys: Sequence[str]) -> List[str]:
    resolved = []
    for key in keys:
        norm = key.strip().lower()
        if not norm:
            continue
        if norm not in MODEL_SPECS:
            raise ValueError(f"Unknown model key: {norm}")
        resolved.append(norm)
    if not resolved:
        raise ValueError("At least one model key is required")
    return resolved


def parse_seed_label(name: str) -> str:
    lower = name.lower()
    if "seed" not in lower:
        return name
    return lower.split("seed", 1)[1]


def resolve_eval_dir_name(runs_root: Path, glob_pattern: str, split: str) -> str:
    fine_name = f"online_eval_{split}_fine"
    for run_dir in sorted(runs_root.glob(glob_pattern)):
        if (run_dir / fine_name / "online_evaluation_results.json").exists():
            return fine_name
    return f"online_eval_{split}"


def load_model_curves(
    runs_root: Path,
    glob_pattern: str,
    eval_dir_name: str,
) -> Dict[str, Dict[float, Dict[str, float]]]:
    curves: Dict[str, Dict[float, Dict[str, float]]] = {}
    for run_dir in sorted(runs_root.glob(glob_pattern)):
        if not run_dir.is_dir():
            continue
        result_path = run_dir / eval_dir_name / "online_evaluation_results.json"
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
                "loss": float(point["loss"]),
                "gate_score": float(point.get("gate_score", np.nan)),
            }
        if curve:
            curves[seed_label] = curve
    if not curves:
        raise FileNotFoundError(f"No online evaluation files found under {runs_root} with glob '{glob_pattern}'")
    return curves


def collect_seed_common_ratios(curves: Dict[str, Dict[float, Dict[str, float]]]) -> set[float]:
    common = None
    for seed_curve in curves.values():
        ratios = set(seed_curve.keys())
        common = ratios if common is None else (common & ratios)
    return common or set()


def collect_model_ratios(
    all_curves: Dict[str, Dict[str, Dict[float, Dict[str, float]]]],
    common_only: bool,
) -> List[float]:
    if common_only:
        common = None
        for curves in all_curves.values():
            seed_common = collect_seed_common_ratios(curves)
            common = seed_common if common is None else (common & seed_common)
        return sorted(common) if common else []

    union = set()
    for curves in all_curves.values():
        for seed_curve in curves.values():
            union.update(seed_curve.keys())
    return sorted(union)


def filter_ratios(ratios: List[float], min_ratio: float, max_ratio: float) -> List[float]:
    filtered = [ratio for ratio in ratios if min_ratio <= ratio <= max_ratio]
    if not filtered:
        raise ValueError(f"No prefix ratios remain after filtering to [{min_ratio}, {max_ratio}]")
    return filtered


def compute_metric_stats(
    curves: Dict[str, Dict[float, Dict[str, float]]],
    ratios: List[float],
    metric: str,
) -> Dict[str, object]:
    seeds = sort_seed_labels(list(curves.keys()))
    matrix = []
    for seed in seeds:
        matrix.append([curves[seed].get(ratio, {}).get(metric, np.nan) for ratio in ratios])
    values = np.asarray(matrix, dtype=np.float64)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    return {
        "seeds": seeds,
        "values": values,
        "mean": mean,
        "std": std,
    }


def xticks_for_range(min_ratio: float, max_ratio: float) -> List[float]:
    start = max(0.1, np.ceil(min_ratio * 10.0) / 10.0)
    stop = np.floor(max_ratio * 10.0) / 10.0
    ticks = np.arange(start, stop + 1e-9, 0.1)
    return [float(f"{tick:.1f}") for tick in ticks]


def plot_curves(
    model_keys: List[str],
    all_curves: Dict[str, Dict[str, Dict[float, Dict[str, float]]]],
    ratios: List[float],
    output_dir: Path,
    output_name: str,
    title: str,
    dpi: int,
) -> Dict[str, Path]:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    json_payload = {
        "ratios": ratios,
        "models": {},
    }

    for key in model_keys:
        spec = MODEL_SPECS[key]
        stats = compute_metric_stats(all_curves[key], ratios, "average_accuracy")
        mean = np.asarray(stats["mean"]) * 100.0
        std = np.asarray(stats["std"]) * 100.0
        lower = np.clip(mean - std, 0.0, 100.0)
        upper = np.clip(mean + std, 0.0, 100.0)

        ax.fill_between(ratios, lower, upper, color=spec["color"], alpha=0.14, linewidth=0)
        ax.plot(
            ratios,
            mean,
            color=spec["color"],
            linewidth=2.8,
            marker=spec["marker"],
            markersize=6.0,
            markerfacecolor="white",
            markeredgewidth=1.4,
            label=spec["label"],
        )

        json_payload["models"][key] = {
            "label": spec["label"],
            "seeds": stats["seeds"],
            "mean_accuracy_pct": [float(x) for x in mean],
            "std_accuracy_pct": [float(x) for x in std],
        }

    ax.set_xlim(min(ratios), max(ratios))
    ax.set_xticks(xticks_for_range(min(ratios), max(ratios)))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Prefix Ratio")
    ax.set_ylabel("Average Accuracy (%)")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.55, color="#CAD5E2")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")

    if title:
        ax.set_title(title)

    fig.tight_layout()

    png_path = output_dir / f"{output_name}.png"
    pdf_path = output_dir / f"{output_name}.pdf"
    json_path = output_dir / f"{output_name}.json"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"png": png_path, "pdf": pdf_path, "json": json_path}


def main() -> None:
    args = parse_args()
    model_keys = validate_model_keys(args.models.split(","))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_curves = {}
    for key in model_keys:
        spec = MODEL_SPECS[key]
        eval_dir_name = resolve_eval_dir_name(
            runs_root=spec["runs_root"],
            glob_pattern=spec["glob"],
            split=args.split,
        )
        curves = load_model_curves(
            runs_root=spec["runs_root"],
            glob_pattern=spec["glob"],
            eval_dir_name=eval_dir_name,
        )
        all_curves[key] = curves

    ratios = collect_model_ratios(all_curves, args.common_ratios_only)
    ratios = filter_ratios(ratios, args.min_ratio, args.max_ratio)
    title = args.title or "OOD Test: Prefix Ratio vs Average Accuracy"
    paths = plot_curves(
        model_keys=model_keys,
        all_curves=all_curves,
        ratios=ratios,
        output_dir=output_dir,
        output_name=args.output_name,
        title=title,
        dpi=args.dpi,
    )

    print("=" * 72)
    print("three-model online prefix plot generated")
    print("=" * 72)
    print(f"models    : {', '.join(model_keys)}")
    print(f"split     : {args.split}")
    print(f"ratios    : {', '.join(f'{ratio:.1f}' for ratio in ratios)}")
    for key in model_keys:
        spec = MODEL_SPECS[key]
        print(f"{spec['label']}: {len(all_curves[key])} seeds from {spec['runs_root']}")
    print(f"png       : {paths['png']}")
    print(f"pdf       : {paths['pdf']}")
    print(f"json      : {paths['json']}")


if __name__ == "__main__":
    main()
