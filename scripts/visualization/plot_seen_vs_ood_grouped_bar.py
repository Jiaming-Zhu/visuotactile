import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise ImportError("matplotlib is required to generate plots") from exc

import numpy as np


ROOT = Path("/home/martina/Y3_Project/visuotactile")
DEFAULT_OUTPUT_DIR = ROOT / "docs/figures"
DEFAULT_OUTPUT_NAME = "seen_vs_ood_grouped_bar.pdf"

ID_BAR_COLOR = "#79A2D1"
OOD_BAR_COLOR = "#D1A879"
EDGE_COLOR = "#202020"
GRID_COLOR = "#D6DEE8"

TITLE_FONT_SIZE = 10
LABEL_FONT_SIZE = 9
TICK_FONT_SIZE = 7.5
LEGEND_FONT_SIZE = 7.5

@dataclass(frozen=True)
class ModelSpec:
    label: str
    display_label: str
    summary_path: Path
    root_key: str | None


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        label="Vision-only (A)",
        display_label="A",
        summary_path=ROOT / "outputs/singleModal/meta/multi_seed_summary_single_modal.json",
        root_key="vision_standard",
    ),
    ModelSpec(
        label="Tactile-only (B)",
        display_label="B",
        summary_path=ROOT / "outputs/singleModal/meta/multi_seed_summary_single_modal.json",
        root_key="tactile_standard",
    ),
    ModelSpec(
        label="Early fusion (C)",
        display_label="C",
        summary_path=ROOT / "outputs/fusion/standard/meta/multi_seed_summary_standard.json",
        root_key=None,
    ),
    ModelSpec(
        label="Gated + aux. (G1)",
        display_label="G1",
        summary_path=ROOT / "outputs/fusion/gating/meta/multi_seed_summary_gating_entropy.json",
        root_key="fusion_gating_entropy",
    ),
    ModelSpec(
        label="Prefix-aware (G2)",
        display_label="G2",
        summary_path=ROOT / "outputs/fusion_gating_online_v2_multiseed/meta/multi_seed_summary_gating_online_v2.json",
        root_key="fusion_gating_online_v2",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a grouped bar chart comparing seen-object and unseen-object accuracy."
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--title", type=str, default="")
    return parser.parse_args()


def load_summary(spec: ModelSpec) -> Dict:
    payload = json.loads(spec.summary_path.read_text(encoding="utf-8"))
    return payload if spec.root_key is None else payload[spec.root_key]


def average_accuracy_pct(summary: Dict, split_key: str) -> float:
    return float(summary[split_key]["avg"]["mean"]) * 100.0


def collect_plot_data() -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for spec in MODEL_SPECS:
        summary = load_summary(spec)
        rows.append(
            {
                "label": spec.label,
                "display_label": spec.display_label,
                "id_accuracy": average_accuracy_pct(summary, "eval_test"),
                "ood_accuracy": average_accuracy_pct(summary, "eval_ood_test"),
            }
        )
    return rows


def build_plot(rows: List[Dict[str, float | str]], output_path: Path, title: str) -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    labels = [str(row["display_label"]) for row in rows]
    id_values = np.array([float(row["id_accuracy"]) for row in rows], dtype=float)
    ood_values = np.array([float(row["ood_accuracy"]) for row in rows], dtype=float)

    x = np.arange(len(rows), dtype=float)
    width = 0.34

    fig, ax = plt.subplots(figsize=(3.45, 2.7))

    ax.bar(
        x - width / 2,
        id_values,
        width,
        color=ID_BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.1,
        label="Seen-object (ID)",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        ood_values,
        width,
        color=OOD_BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=1.1,
        hatch="///",
        label="Unseen-object (OOD)",
        zorder=3,
    )

    ax.set_ylabel("Accuracy (%)", fontsize=LABEL_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle="--", linewidth=0.8, color=GRID_COLOR, alpha=0.9)
    ax.set_axisbelow(True)

    if title:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=10)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        ncol=2,
        handlelength=1.6,
        columnspacing=1.0,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_manifest(rows: List[Dict[str, float | str]], output_path: Path) -> Path:
    payload = {
        "models": rows,
        "sources": [
            {
                "label": spec.label,
                "summary_path": str(spec.summary_path),
                "root_key": spec.root_key,
            }
            for spec in MODEL_SPECS
        ],
    }
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name

    rows = collect_plot_data()
    build_plot(rows, output_path, args.title)
    manifest_path = write_manifest(rows, output_path)

    print("=" * 72)
    print("seen-vs-ood grouped bar chart generated")
    print("=" * 72)
    print(f"PDF  : {output_path}")
    print(f"JSON : {manifest_path}")
    for row in rows:
        print(
            f"{row['label']}: ID={float(row['id_accuracy']):.2f}% | "
            f"OOD={float(row['ood_accuracy']):.2f}%"
        )


if __name__ == "__main__":
    main()
