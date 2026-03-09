import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("matplotlib is required to generate plots") from exc


DEFAULT_ACCURACY_JSON = Path(
    "/home/martina/Y3_Project/visuotactile/outputs/fusion_gating_online_v2_multiseed/plots_online_prefix_fine/online_prefix_multiseed_average_accuracy.json"
)
DEFAULT_LATENCY_JSON = Path(
    "/home/martina/Y3_Project/visuotactile/outputs/fusion_gating_online_v2/latency_benchmark/latency_results.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/martina/Y3_Project/visuotactile/outputs/fusion_gating_online_v2_multiseed/plots_online_prefix_fine"
)

LINE_COLOR = "#355C7D"
ERROR_COLOR = "#4C78A8"
POINT_CMAP = "viridis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prefix accuracy vs latency trade-off")
    parser.add_argument("--accuracy_json", type=Path, default=DEFAULT_ACCURACY_JSON)
    parser.add_argument("--latency_json", type=Path, default=DEFAULT_LATENCY_JSON)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--control_hz", type=float, default=100.0)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_accuracy_summary(path: Path) -> Dict[float, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    seeds = data["seeds"]
    ratio_to_values: Dict[float, List[float]] = {}
    for seed in seeds:
        for ratio_str, value in data["values"][seed].items():
            ratio = round(float(ratio_str), 4)
            ratio_to_values.setdefault(ratio, []).append(float(value))

    summary = {}
    for ratio, values in ratio_to_values.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[ratio] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": int(arr.size),
        }
    return summary


def load_latency_summary(path: Path) -> Dict[float, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = {}
    for item in data["per_prefix"]:
        ratio = round(float(item["prefix_ratio"]), 4)
        summary[ratio] = {
            "mean_ms": float(item["mean_ms"]),
            "std_ms": float(item["std_ms"]),
            "hz": float(item["hz"]),
            "margin_x": float(item["control_margin_x"]),
        }
    return summary


def build_title(args: argparse.Namespace) -> str:
    if args.title:
        return args.title
    return "OOD Prefix Accuracy vs Inference Latency"


def plot_tradeoff(
    output_dir: Path,
    accuracy_summary: Dict[float, Dict[str, float]],
    latency_summary: Dict[float, Dict[str, float]],
    args: argparse.Namespace,
) -> Dict[str, Path]:
    ratios = sorted(set(accuracy_summary.keys()) & set(latency_summary.keys()))
    if not ratios:
        raise ValueError("No overlapping prefix ratios between accuracy and latency results")

    x = np.asarray([latency_summary[r]["mean_ms"] for r in ratios], dtype=np.float64)
    y = np.asarray([accuracy_summary[r]["mean"] * 100.0 for r in ratios], dtype=np.float64)
    yerr = np.asarray([accuracy_summary[r]["std"] * 100.0 for r in ratios], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="none",
        ecolor=ERROR_COLOR,
        elinewidth=1.4,
        capsize=4,
        alpha=0.95,
        label="mean ± std",
        zorder=2,
    )

    scatter = ax.scatter(
        x,
        y,
        c=ratios,
        cmap=POINT_CMAP,
        s=72,
        edgecolors="white",
        linewidths=0.9,
        zorder=3,
    )

    label_offsets = [
        (6, 6),
        (6, 10),
        (8, 6),
        (8, 10),
        (-28, 6),
        (-28, 10),
        (6, -12),
        (8, -14),
    ]
    for idx, (ratio, xi, yi) in enumerate(zip(ratios, x, y)):
        offset = label_offsets[idx % len(label_offsets)]
        ax.annotate(
            f"{ratio:.2f}",
            (xi, yi),
            textcoords="offset points",
            xytext=offset,
            fontsize=9,
            color="#333333",
        )

    ax.set_title(build_title(args))
    ax.set_xlabel("Inference Latency per Query (ms)")
    ax.set_ylabel("OOD Average Accuracy (%)")
    ax.set_ylim(0, 105)
    x_lo = float(np.min(x))
    x_hi = float(np.max(x))
    x_span = max(x_hi - x_lo, 0.15)
    x_pad = max(0.08, 0.18 * x_span)
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Prefix Ratio")

    fig.tight_layout()
    png_path = output_dir / "prefix_accuracy_latency_tradeoff.png"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "ratios": ratios,
        "latency_ms_mean": {f"{r:.4f}": latency_summary[r]["mean_ms"] for r in ratios},
        "latency_ms_std": {f"{r:.4f}": latency_summary[r]["std_ms"] for r in ratios},
        "accuracy_mean": {f"{r:.4f}": accuracy_summary[r]["mean"] for r in ratios},
        "accuracy_std": {f"{r:.4f}": accuracy_summary[r]["std"] for r in ratios},
        "control_hz": args.control_hz,
        "control_budget_ms": 1000.0 / max(args.control_hz, 1e-8),
    }
    json_path = output_dir / "prefix_accuracy_latency_tradeoff.json"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"png": png_path, "json": json_path}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_summary = load_accuracy_summary(args.accuracy_json)
    latency_summary = load_latency_summary(args.latency_json)
    outputs = plot_tradeoff(args.output_dir, accuracy_summary, latency_summary, args)

    print("=" * 70)
    print("prefix accuracy-latency trade-off plot generated")
    print("=" * 70)
    print(f"accuracy_json : {args.accuracy_json}")
    print(f"latency_json  : {args.latency_json}")
    print(f"PNG           : {outputs['png']}")
    print(f"JSON          : {outputs['json']}")


if __name__ == "__main__":
    main()
