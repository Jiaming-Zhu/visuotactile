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


ROOT = Path("/home/martina/Y3_Project/visuotactile")
DEFAULT_INPUTS = [
    ROOT / "outputs/backup4mass/meta/multi_seed_summary.json",
    ROOT / "outputs/backup4mass/meta/multi_seed_summary_gating_entropy.json",
    ROOT / "outputs/meta/multi_seed_summary_all_models.json",
    ROOT / "outputs/meta/multi_seed_summary_standard.json",
    ROOT / "outputs/meta/multi_seed_summary_gating_entropy.json",
    ROOT / "outputs/meta/multi_seed_summary_gating_entropy_old.json",
    ROOT / "outputs/singleModal/meta/multi_seed_summary_single_modal.json",
]
DEFAULT_OUTPUT_DIR = ROOT / "outputs/plots/multi_seed"
SPLITS = ("eval_test", "eval_ood_test")
METRICS = ("mass", "stiffness", "material", "avg")
MODEL_PALETTE = ("#7fc97f", "#beaed4", "#fdc086", "#ffff99", "#386cb0")
METRIC_PALETTE = MODEL_PALETTE[:4]
GATE_COLOR = MODEL_PALETTE[-1]
PREFERRED_ORDER = (
    "Fusion Gating (entropy)",
    "Fusion Gating (entropy, old)",
    "Standard Fusion",
    "Vision Only",
    "Tactile Only",
)
KNOWN_LABELS = {
    "multi_seed_summary_gating_entropy": "Fusion Gating (entropy)",
    "multi_seed_summary_gating_entropy_old": "Fusion Gating (entropy, old)",
    "multi_seed_summary_standard": "Standard Fusion",
}
MODEL_KEY_LABELS = {
    "Fusion": "Standard Fusion",
    "Vision": "Vision Only",
    "Tactile": "Tactile Only",
    "fusion_standard": "Standard Fusion",
    "vision_standard": "Vision Only",
    "tactile_standard": "Tactile Only",
    "fusion_gating_entropy": "Fusion Gating (entropy)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multi-seed experiment summaries from JSON")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Summary JSON files. If omitted, the script auto-discovers common summary files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save generated plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
    return parser.parse_args()


def infer_label(path: Path) -> str:
    stem = path.stem
    if stem in KNOWN_LABELS:
        return KNOWN_LABELS[stem]
    if "gating" in stem and "entropy" in stem and "old" in stem:
        return "Fusion Gating (entropy, old)"
    if "gating" in stem and "entropy" in stem:
        return "Fusion Gating (entropy)"
    if "standard" in stem and "fusion" in stem:
        return "Standard Fusion"
    if "vision" in stem:
        return "Vision Only"
    if "tactile" in stem:
        return "Tactile Only"
    return stem.replace("_", " ").title()


def discover_inputs(cli_inputs: List[str]) -> List[Path]:
    if cli_inputs:
        paths = [Path(p) for p in cli_inputs]
    else:
        paths = [p for p in DEFAULT_INPUTS if p.exists()]
    if not paths:
        raise FileNotFoundError("No summary JSON files found. Pass --inputs explicitly.")
    return paths


def _has_split_payload(payload: Dict) -> bool:
    return any(split in payload for split in SPLITS)


def _extract_entry(payload: Dict) -> Dict[str, Dict]:
    entry: Dict[str, Dict] = {}
    for split in SPLITS:
        if split in payload and isinstance(payload[split], dict):
            entry[split] = payload[split]
    return entry


def load_models(paths: List[Path]) -> Dict[str, Dict[str, Dict]]:
    models: Dict[str, Dict[str, Dict]] = {}
    for path in paths:
        data = json.loads(path.read_text())

        if _has_split_payload(data):
            label = infer_label(path)
            models[label] = _extract_entry(data)
            continue

        for model_key, payload in data.items():
            if not isinstance(payload, dict) or not _has_split_payload(payload):
                continue
            label = payload.get("label") or MODEL_KEY_LABELS.get(model_key) or model_key.replace("_", " ").title()
            models[label] = _extract_entry(payload)
    if not models:
        raise ValueError("No plottable model entries found in the provided JSON files.")
    return models


def sort_labels(labels: List[str]) -> List[str]:
    order = {label: idx for idx, label in enumerate(PREFERRED_ORDER)}
    return sorted(labels, key=lambda label: (order.get(label, 999), label))


def metric_series(model_payload: Dict, split: str, metric: str) -> Dict[str, float]:
    metric_payload = model_payload[split].get(metric)
    if not isinstance(metric_payload, dict):
        raise KeyError(f"Missing metric '{metric}' in split '{split}'")
    return {
        "mean": float(metric_payload["mean"]),
        "std": float(metric_payload.get("std", 0.0)),
    }


def optional_metric_series(model_payload: Dict, split: str, metric: str) -> Dict[str, float]:
    metric_payload = model_payload[split].get(metric)
    if not isinstance(metric_payload, dict):
        return {}
    return {
        "mean": float(metric_payload["mean"]),
        "std": float(metric_payload.get("std", 0.0)),
    }


def plot_metric_bars(models: Dict[str, Dict], split: str, output_dir: Path, dpi: int) -> Path:
    labels = [label for label in sort_labels(list(models.keys())) if split in models[label]]
    if not labels:
        raise ValueError(f"No models contain split '{split}'")

    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, metric in enumerate(METRICS):
        means = [metric_series(models[label], split, metric)["mean"] * 100.0 for label in labels]
        stds = [metric_series(models[label], split, metric)["std"] * 100.0 for label in labels]
        offset = (idx - (len(METRICS) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            label=metric.upper(),
            color=METRIC_PALETTE[idx % len(METRIC_PALETTE)],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
        )

    split_title = "Test (In-Distribution)" if split == "eval_test" else "OOD Test (Out-of-Distribution)"
    ax.set_title(f"{split_title} - Per-Task Accuracy")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / f"{split}_metric_bars.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_avg_overview(models: Dict[str, Dict], split: str, output_dir: Path, dpi: int) -> Path:
    labels = [label for label in sort_labels(list(models.keys())) if split in models[label]]
    if not labels:
        raise ValueError(f"No models contain split '{split}'")

    avg_means = [metric_series(models[label], split, "avg")["mean"] * 100.0 for label in labels]
    avg_stds = [metric_series(models[label], split, "avg")["std"] * 100.0 for label in labels]
    gate_means = [optional_metric_series(models[label], split, "avg_gate_score").get("mean") for label in labels]
    has_gate = any(value is not None for value in gate_means)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bar_colors = [MODEL_PALETTE[idx % len(MODEL_PALETTE)] for idx in range(len(labels))]
    ax.bar(
        x,
        avg_means,
        yerr=avg_stds,
        capsize=5,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.8,
        alpha=0.95,
    )
    ax.set_title(("Test" if split == "eval_test" else "OOD Test") + " - Average Accuracy by Model")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)

    for idx, value in enumerate(avg_means):
        ax.text(idx, min(103.0, value + 2.0), f"{value:.1f}%", ha="center", va="bottom", fontsize=9)

    if has_gate:
        ax2 = ax.twinx()
        gate_x = [idx for idx, value in enumerate(gate_means) if value is not None]
        gate_y = [value for value in gate_means if value is not None]
        ax2.plot(gate_x, gate_y, color=GATE_COLOR, marker="o", linewidth=2, label="Avg Gate Score")
        ax2.set_ylabel("Average Gate Score")
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="upper right")

    fig.tight_layout()
    out_path = output_dir / f"{split}_avg_overview.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_manifest(models: Dict[str, Dict], output_dir: Path) -> Path:
    manifest = {}
    for label in sort_labels(list(models.keys())):
        manifest[label] = {}
        for split in SPLITS:
            if split not in models[label]:
                continue
            manifest[label][split] = {}
            for metric in METRICS + ("avg_gate_score",):
                metric_payload = models[label][split].get(metric)
                if isinstance(metric_payload, dict):
                    manifest[label][split][metric] = {
                        "mean": float(metric_payload["mean"]),
                        "std": float(metric_payload.get("std", 0.0)),
                    }
    out_path = output_dir / "plot_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return out_path


def main() -> None:
    args = parse_args()
    input_paths = discover_inputs(args.inputs)
    models = load_models(input_paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: List[Path] = []
    for split in SPLITS:
        split_labels = [label for label in models if split in models[label]]
        if not split_labels:
            continue
        generated.append(plot_metric_bars(models, split, output_dir, args.dpi))
        generated.append(plot_avg_overview(models, split, output_dir, args.dpi))

    generated.append(write_manifest(models, output_dir))

    print("Loaded summaries:")
    for path in input_paths:
        print(f"  - {path}")
    print("Generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
