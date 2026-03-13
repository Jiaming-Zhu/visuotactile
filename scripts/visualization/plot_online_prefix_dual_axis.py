import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("matplotlib is required to generate plots") from exc

try:
    from plot_online_prefix_multiseed import collect_ratios, load_curves, sort_seed_labels
except ImportError:  # pragma: no cover
    from visuotactile.scripts.visualization.plot_online_prefix_multiseed import (  # type: ignore
        collect_ratios,
        load_curves,
        sort_seed_labels,
    )


DEFAULT_RUNS_ROOT = Path("/home/martina/Y3_Project/visuotactile/outputs/fusion_gating_online_v2_multiseed")
DEFAULT_GLOB = "fusion_gating_online_v2_seed*"
DEFAULT_SPLIT = "ood_test"
DEFAULT_OUTPUT_NAME = "online_prefix_multiseed_accuracy_gate_dual_axis.png"

ACCURACY_COLOR = "#0B6EBD"
GATE_COLOR = "#E67E22"
GRID_COLOR = "#CAD5E2"
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 15
TICK_FONT_SIZE = 13
LEGEND_FONT_SIZE = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a dual-axis multi-seed online prefix curve with mean/std confidence regions"
    )
    parser.add_argument("--runs_root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--glob", type=str, default=DEFAULT_GLOB)
    parser.add_argument("--split", choices=["val", "test", "ood_test"], default=DEFAULT_SPLIT)
    parser.add_argument(
        "--eval_dir_name",
        type=str,
        default="",
        help="Optional online eval directory name. Defaults to online_eval_<split>_fine and falls back to online_eval_<split>.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--output_name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--min_ratio", type=float, default=0.1)
    parser.add_argument("--max_ratio", type=float, default=1.0)
    parser.add_argument(
        "--manual_curve_json",
        type=Path,
        default=None,
        help="Optional JSON file with supplemental per-seed prefix curves. Expected format: "
        "{seed_label: {ratio: {average_accuracy: ..., gate_score: ...}}}",
    )
    parser.add_argument(
        "--common_ratios_only",
        action="store_true",
        help="Restrict plotting to prefix ratios available for every seed after merging supplemental curves.",
    )
    return parser.parse_args()


def load_curves_with_fallback(args: argparse.Namespace) -> Tuple[Dict[str, Dict[float, Dict[str, float]]], str]:
    if args.eval_dir_name:
        return load_curves(args.runs_root, args.glob, args.split, args.eval_dir_name), args.eval_dir_name

    candidates = [f"online_eval_{args.split}_fine", f"online_eval_{args.split}"]
    errors: List[str] = []
    for candidate in candidates:
        try:
            curves = load_curves(args.runs_root, args.glob, args.split, candidate)
            return curves, candidate
        except FileNotFoundError as exc:
            errors.append(str(exc))

    raise FileNotFoundError("\n".join(errors))


def filter_ratios(ratios: List[float], min_ratio: float, max_ratio: float) -> List[float]:
    filtered = [ratio for ratio in ratios if min_ratio <= ratio <= max_ratio]
    if not filtered:
        raise ValueError(f"No prefix ratios remain after filtering to [{min_ratio}, {max_ratio}]")
    return filtered


def merge_manual_curves(
    curves: Dict[str, Dict[float, Dict[str, float]]],
    manual_curve_json: Path | None,
) -> Dict[str, Dict[float, Dict[str, float]]]:
    if manual_curve_json is None:
        return curves

    payload = json.loads(manual_curve_json.read_text(encoding="utf-8"))
    merged = {seed: dict(points) for seed, points in curves.items()}
    for seed_label, seed_points in payload.items():
        seed_curve = dict(merged.get(seed_label, {}))
        for ratio_raw, metrics in seed_points.items():
            ratio = float(ratio_raw)
            seed_curve[ratio] = {key: float(value) for key, value in metrics.items()}
        merged[str(seed_label)] = seed_curve
    return merged


def collect_common_ratios(curves: Dict[str, Dict[float, Dict[str, float]]]) -> List[float]:
    common: set[float] | None = None
    for curve in curves.values():
        ratios = set(curve.keys())
        common = ratios if common is None else (common & ratios)
    return sorted(common) if common else []


def metric_stats(
    curves: Dict[str, Dict[float, Dict[str, float]]],
    seeds: List[str],
    ratios: List[float],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = []
    for seed in seeds:
        matrix.append([curves[seed].get(ratio, {}).get(metric, np.nan) for ratio in ratios])
    values = np.asarray(matrix, dtype=np.float64)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    return values, mean, std


def xticks_for_range(min_ratio: float, max_ratio: float) -> List[float]:
    start = max(0.1, np.ceil(min_ratio * 10.0) / 10.0)
    stop = np.floor(max_ratio * 10.0) / 10.0
    ticks = np.arange(start, stop + 1e-9, 0.1)
    return [float(f"{tick:.1f}") for tick in ticks]


def plot_dual_axis(
    output_dir: Path,
    curves: Dict[str, Dict[float, Dict[str, float]]],
    ratios: List[float],
    args: argparse.Namespace,
) -> Tuple[Path, Path]:
    seeds = sort_seed_labels(list(curves.keys()))

    _, acc_mean, acc_std = metric_stats(curves, seeds, ratios, "average_accuracy")
    _, gate_mean, gate_std = metric_stats(curves, seeds, ratios, "gate_score")

    acc_mean_pct = acc_mean * 100.0
    acc_std_pct = acc_std * 100.0
    acc_lower = np.clip(acc_mean_pct - acc_std_pct, 0.0, 100.0)
    acc_upper = np.clip(acc_mean_pct + acc_std_pct, 0.0, 100.0)
    gate_lower = np.clip(gate_mean - gate_std, 0.0, 1.0)
    gate_upper = np.clip(gate_mean + gate_std, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax2 = ax.twinx()

    ax.fill_between(ratios, acc_lower, acc_upper, color=ACCURACY_COLOR, alpha=0.20, linewidth=0)
    acc_line = ax.plot(
        ratios,
        acc_mean_pct,
        color=ACCURACY_COLOR,
        linewidth=3.1,
        marker="o",
        markersize=5.8,
        markerfacecolor="white",
        markeredgewidth=1.4,
        label="Mean Accuracy",
        zorder=3,
    )[0]

    ax2.fill_between(ratios, gate_lower, gate_upper, color=GATE_COLOR, alpha=0.20, linewidth=0)
    gate_line = ax2.plot(
        ratios,
        gate_mean,
        color=GATE_COLOR,
        linewidth=2.8,
        linestyle="--",
        marker="s",
        markersize=5.2,
        markerfacecolor="white",
        markeredgewidth=1.3,
        label="Mean Gate",
        zorder=3,
    )[0]

    ax.set_xlim(args.min_ratio, args.max_ratio)
    ax.set_xticks(xticks_for_range(args.min_ratio, args.max_ratio))
    ax.set_ylim(0, 100)
    ax2.set_ylim(0.0, 1.0)

    ax.set_xlabel("Prefix Ratio", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Accuracy (%)", color=ACCURACY_COLOR, fontsize=LABEL_FONT_SIZE)
    ax2.set_ylabel("Average Gate", color=GATE_COLOR, fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", colors=ACCURACY_COLOR, labelsize=TICK_FONT_SIZE)
    ax2.tick_params(axis="y", colors=GATE_COLOR, labelsize=TICK_FONT_SIZE)
    ax.spines["left"].set_color(ACCURACY_COLOR)
    ax2.spines["right"].set_color(GATE_COLOR)

    if args.title:
        ax.set_title(args.title, fontsize=TITLE_FONT_SIZE, pad=10)

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6, color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.legend(
        [acc_line, gate_line],
        ["Mean Accuracy", "Mean Gate"],
        frameon=False,
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.4,
    )

    fig.tight_layout()
    png_path = output_dir / args.output_name
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    json_payload = {
        "split": args.split,
        "eval_dir_name": args.eval_dir_name or "",
        "seeds": seeds,
        "ratios": ratios,
        "accuracy_mean": [float(x) for x in acc_mean],
        "accuracy_std": [float(x) for x in acc_std],
        "gate_mean": [float(x) for x in gate_mean],
        "gate_std": [float(x) for x in gate_std],
    }
    json_path = output_dir / f"{Path(args.output_name).stem}.json"
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return png_path, json_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.runs_root / "plots_online_prefix")
    output_dir.mkdir(parents=True, exist_ok=True)

    curves, resolved_eval_dir = load_curves_with_fallback(args)
    curves = merge_manual_curves(curves, args.manual_curve_json)
    ratio_source = collect_common_ratios(curves) if args.common_ratios_only else collect_ratios(curves)
    ratios = filter_ratios(ratio_source, args.min_ratio, args.max_ratio)
    png_path, json_path = plot_dual_axis(output_dir, curves, ratios, args)

    print("=" * 70)
    print("dual-axis online prefix multi-seed plot generated")
    print("=" * 70)
    print(f"runs_root : {args.runs_root}")
    print(f"split     : {args.split}")
    print(f"eval_dir  : {resolved_eval_dir}")
    print(f"seeds     : {', '.join(sort_seed_labels(list(curves.keys())))}")
    print(f"ratios    : {', '.join(str(r) for r in ratios)}")
    print(f"PNG       : {png_path}")
    print(f"JSON      : {json_path}")


if __name__ == "__main__":
    main()
