import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("matplotlib is required to generate the TABLE II visualization") from exc


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "docs/figures/table2"
DEFAULT_OUTPUT_NAME = "table2_main_multiseed_results"

SPLITS = [
    "Test",
    "OOD",
    "OOD Mass",
    "OOD Stiff.",
    "OOD Mater.",
]

METHOD_RESULTS = [
    {
        "label": "Vision-only",
        "color": "#9ca3af",
        "edgecolor": "#374151",
        "hatch": "",
        "values": {
            "Test": {"mean": 95.39, "std": 0.73},
            "OOD": {"mean": 18.00, "std": 6.16},
            "OOD Mass": {"mean": 17.17, "std": 6.55},
            "OOD Stiff.": {"mean": 17.83, "std": 6.84},
            "OOD Mater.": {"mean": 19.00, "std": 5.15},
        },
    },
    {
        "label": "Internal-feedback-only",
        "color": "#8ecae6",
        "edgecolor": "#1d3557",
        "hatch": "",
        "values": {
            "Test": {"mean": 94.02, "std": 1.50},
            "OOD": {"mean": 88.89, "std": 2.26},
            "OOD Mass": {"mean": 100.00, "std": 0.00},
            "OOD Stiff.": {"mean": 79.50, "std": 2.56},
            "OOD Mater.": {"mean": 87.17, "std": 5.23},
        },
    },
    {
        "label": "Standard fusion",
        "color": "#a7c957",
        "edgecolor": "#386641",
        "hatch": "",
        "values": {
            "Test": {"mean": 99.65, "std": 0.51},
            "OOD": {"mean": 89.78, "std": 4.93},
            "OOD Mass": {"mean": 99.87, "std": 0.27},
            "OOD Stiff.": {"mean": 85.33, "std": 7.30},
            "OOD Mater.": {"mean": 84.13, "std": 7.36},
        },
    },
    {
        "label": "RPDF (ours)",
        "color": "#f28482",
        "edgecolor": "#9d0208",
        "hatch": "///",
        "values": {
            "Test": {"mean": 99.12, "std": 0.78},
            "OOD": {"mean": 94.89, "std": 3.18},
            "OOD Mass": {"mean": 99.73, "std": 0.33},
            "OOD Stiff.": {"mean": 93.33, "std": 4.94},
            "OOD Mater.": {"mean": 91.60, "std": 4.87},
        },
    },
]


@dataclass(frozen=True)
class PlotArtifactPaths:
    png_path: Path
    pdf_path: Path
    json_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TABLE II visualization for seen and deceptive-object OOD splits."
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def build_artifact_paths(output_dir: Path, output_name: str) -> PlotArtifactPaths:
    return PlotArtifactPaths(
        png_path=output_dir / f"{output_name}.png",
        pdf_path=output_dir / f"{output_name}.pdf",
        json_path=output_dir / f"{output_name}.json",
    )


def write_manifest(json_path: Path) -> None:
    payload: Dict[str, object] = {
        "title": "TABLE II Main Multi-Seed Results on the Seen and Deceptive-Object OOD Splits",
        "note": "Values are mean ± std across five seeds.",
        "splits": SPLITS,
        "methods": METHOD_RESULTS,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_table_figure(paths: PlotArtifactPaths, dpi: int) -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 9

    x = np.arange(len(SPLITS), dtype=float)
    width = 0.18

    fig, ax = plt.subplots(figsize=(8.4, 4.4))

    # Separate the seen split from the OOD splits to make the comparison read faster.
    ax.axvspan(0.5, len(SPLITS) - 0.5, color="#f8f4ea", alpha=0.85, zorder=0)
    ax.text(0.02, 1.02, "Seen", transform=ax.transAxes, fontsize=9, weight="bold")
    ax.text(0.25, 1.02, "Deceptive-object OOD", transform=ax.transAxes, fontsize=9, weight="bold")

    for idx, method in enumerate(METHOD_RESULTS):
        means = [method["values"][split]["mean"] for split in SPLITS]
        offset = (idx - (len(METHOD_RESULTS) - 1) / 2.0) * width

        bars = ax.bar(
            x + offset,
            means,
            width=width,
            color=method["color"],
            edgecolor=method["edgecolor"],
            linewidth=1.0,
            hatch=method["hatch"],
            alpha=0.95,
            label=method["label"],
            zorder=3,
        )

        for bar, value in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(102.6, value + 2.1),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
            )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(SPLITS)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.2,
        handlelength=1.8,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24, top=0.92)
    fig.savefig(paths.png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(paths.pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = build_artifact_paths(args.output_dir, args.output_name)
    plot_table_figure(artifact_paths, args.dpi)
    write_manifest(artifact_paths.json_path)

    print("=" * 72)
    print("table ii visualization generated")
    print("=" * 72)
    print(f"PNG : {artifact_paths.png_path}")
    print(f"PDF : {artifact_paths.pdf_path}")
    print(f"JSON: {artifact_paths.json_path}")


if __name__ == "__main__":
    main()
