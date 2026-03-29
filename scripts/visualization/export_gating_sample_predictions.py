import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from torch.utils.data import DataLoader


THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parents[1]
REPO_DIR = THIS_FILE.parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(REPO_DIR.parent))

from train_fusion_gating_online import build_model, effective_padding_mask  # noqa: E402
from train_fusion_gating2 import RoboticGraspDataset, resolve_device  # noqa: E402


TASKS = ("mass", "stiffness", "material")
DEFAULT_DATA_ROOT = "/home/martina/Y3_Project/Plaintextdataset"
DEFAULT_RUN_ROOT = "/home/martina/Y3_Project/visuotactile/outputs/fusion_gating_online_v2_multiseed"

HYBRID_LABEL_CONFIG_TRAINLIKE = {
    "properties": {
        "WoodBlock_Native": {"mass": "medium", "stiffness": "rigid", "material": "wood"},
        "WoodBlock_Foil": {"mass": "medium", "stiffness": "rigid", "material": "wood"},
        "WoodBlock_Red": {"mass": "medium", "stiffness": "rigid", "material": "wood"},
        "YogaBrick_Native": {"mass": "low", "stiffness": "medium", "material": "foam"},
        "YogaBrick_Blue": {"mass": "low", "stiffness": "medium", "material": "foam"},
        "CardboardBox_Hollow": {"mass": "low", "stiffness": "soft", "material": "hollow_container"},
        "CardboardBox_SpongeFilled": {"mass": "low", "stiffness": "soft", "material": "filled_container"},
        "CardboardBox_RockFilled": {"mass": "high", "stiffness": "rigid", "material": "filled_container"},
        "CardboardBox_RockFilled_Red": {"mass": "high", "stiffness": "rigid", "material": "filled_container"},
        "Sponge_Blue": {"mass": "low", "stiffness": "very_soft", "material": "sponge"},
    },
    "mass_to_idx": {"low": 0, "medium": 1, "high": 2},
    "stiffness_to_idx": {"very_soft": 0, "soft": 1, "medium": 2, "rigid": 3},
    "material_to_idx": {
        "sponge": 0,
        "foam": 1,
        "wood": 2,
        "hollow_container": 3,
        "filled_container": 4,
    },
}

HYBRID_LABEL_CONFIG_OOD = {
    "properties": {
        "YogaBrick_Foil_ANCHOR": {"mass": "low", "stiffness": "medium", "material": "foam"},
        "WoodBlock_noise1": {"mass": "medium", "stiffness": "rigid", "material": "wood"},
        "WoodBlock_noise2": {"mass": "medium", "stiffness": "rigid", "material": "wood"},
        "Sponge_Red": {"mass": "low", "stiffness": "very_soft", "material": "sponge"},
        "Box_Rock_Blue": {"mass": "high", "stiffness": "rigid", "material": "filled_container"},
        "Cardbox_hollow_noise": {"mass": "low", "stiffness": "soft", "material": "hollow_container"},
    },
    "mass_to_idx": {"low": 0, "medium": 1, "high": 2},
    "stiffness_to_idx": {"very_soft": 0, "soft": 1, "medium": 2, "rigid": 3},
    "material_to_idx": {
        "sponge": 0,
        "foam": 1,
        "wood": 2,
        "hollow_container": 3,
        "filled_container": 4,
    },
}


class AnalysisRoboticGraspDataset(RoboticGraspDataset):
    def __init__(
        self,
        split_dir: Path,
        max_tactile_len: int = 3000,
        override_props_config: Dict[str, object] | None = None,
    ) -> None:
        super().__init__(split_dir=split_dir, max_tactile_len=max_tactile_len)
        if override_props_config is not None:
            self.properties = override_props_config["properties"]
            self.mass_to_idx = override_props_config["mass_to_idx"]
            self.stiffness_to_idx = override_props_config["stiffness_to_idx"]
            self.material_to_idx = override_props_config["material_to_idx"]
            self.samples = self._collect_samples()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        sample = self.samples[idx]
        item["obj_class"] = sample.img_path.parent.parent.name
        item["episode_name"] = sample.img_path.parent.name
        item["episode_path"] = str(sample.img_path.parent)
        item["sample_index"] = idx
        return item


def resolve_override_props_config(split_name: str, cfg: Dict[str, object]) -> Dict[str, object] | None:
    signature = (
        int(cfg.get("mass_classes", -1)),
        int(cfg.get("stiffness_classes", -1)),
        int(cfg.get("material_classes", -1)),
    )
    if signature == (3, 4, 5):
        if split_name == "ood_test":
            return HYBRID_LABEL_CONFIG_OOD
        return HYBRID_LABEL_CONFIG_TRAINLIKE
    return None


def build_loader(
    split_dir: Path,
    batch_size: int,
    max_tactile_len: int,
    num_workers: int,
    override_props_config: Dict[str, object] | None = None,
) -> DataLoader:
    dataset = AnalysisRoboticGraspDataset(
        split_dir=split_dir,
        max_tactile_len=max_tactile_len,
        override_props_config=override_props_config,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def ordered_label_names(dataset: AnalysisRoboticGraspDataset) -> Dict[str, List[str]]:
    mappings = {
        "mass": dataset.mass_to_idx,
        "stiffness": dataset.stiffness_to_idx,
        "material": dataset.material_to_idx,
    }
    result: Dict[str, List[str]] = {}
    for task, mapping in mappings.items():
        names = [None] * len(mapping)
        for name, idx in mapping.items():
            names[idx] = name
        result[task] = names
    return result


def parse_seed(run_name: str) -> int:
    match = re.search(r"seed(\d+)", run_name)
    if not match:
        raise ValueError(f"Cannot parse seed from {run_name}")
    return int(match.group(1))


def iter_run_dirs(run_root: Path) -> List[Path]:
    run_dirs = [
        p
        for p in run_root.iterdir()
        if p.is_dir()
        and re.fullmatch(r"fusion_gating_online_v2_seed\d+", p.name) is not None
        and (p / "best_model.pth").exists()
    ]
    return sorted(run_dirs, key=lambda p: parse_seed(p.name))


def build_runtime_args(cfg: Dict, cli_args: argparse.Namespace) -> argparse.Namespace:
    runtime = argparse.Namespace()
    runtime.device = cli_args.device
    runtime.batch_size = cli_args.batch_size
    runtime.max_tactile_len = cfg.get("max_tactile_len", cli_args.max_tactile_len)
    runtime.num_workers = cli_args.num_workers
    runtime.online_train_prob = cfg.get("online_train_prob", 0.0)
    runtime.online_min_prefix_ratio = cfg.get("online_min_prefix_ratio", 0.2)
    runtime.min_prefix_len = cfg.get("min_prefix_len", 16)
    runtime.block_modality = "none"
    runtime.fusion_dim = cfg.get("fusion_dim", 256)
    runtime.num_heads = cfg.get("num_heads", 8)
    runtime.dropout = cfg.get("dropout", 0.1)
    runtime.num_layers = cfg.get("num_layers", 4)
    runtime.lambda_reg = cfg.get("lambda_reg", 0.0)
    return runtime


def collect_rows_for_run(
    run_dir: Path,
    split_name: str,
    data_root: Path,
    cli_args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], Dict[str, List[str]]]:
    checkpoint_path = run_dir / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = resolve_device(cli_args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    runtime_args = build_runtime_args(cfg, cli_args)
    override_props_config = resolve_override_props_config(split_name, cfg)

    split_dir = data_root / split_name
    loader = build_loader(
        split_dir,
        runtime_args.batch_size,
        runtime_args.max_tactile_len,
        runtime_args.num_workers,
        override_props_config=override_props_config,
    )
    dataset = loader.dataset
    label_names = ordered_label_names(dataset)

    model = build_model(
        cfg,
        runtime_args,
        mass_classes=len(dataset.mass_to_idx),
        stiffness_classes=len(dataset.stiffness_to_idx),
        material_classes=len(dataset.material_to_idx),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tactile = batch["tactile"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            prefix_mask = effective_padding_mask(
                padding_mask=padding_mask,
                train_mode=False,
                online_train_prob=0.0,
                online_min_prefix_ratio=runtime_args.online_min_prefix_ratio,
                min_prefix_len=runtime_args.min_prefix_len,
                fixed_ratio=None,
            )
            outputs = model(images, tactile, padding_mask=prefix_mask)

            pred_indices = {
                task: outputs[task].argmax(dim=1).cpu().tolist()
                for task in TASKS
            }
            gate_scores = outputs["gate_score"].cpu().tolist()

            batch_size = len(gate_scores)
            for i in range(batch_size):
                row = {
                    "split": split_name,
                    "seed": parse_seed(run_dir.name),
                    "run_name": run_dir.name,
                    "checkpoint_path": str(checkpoint_path),
                    "sample_index": int(batch["sample_index"][i]),
                    "obj_class": batch["obj_class"][i],
                    "episode_name": batch["episode_name"][i],
                    "episode_path": batch["episode_path"][i],
                    "gate_score": float(gate_scores[i]),
                }
                for task in TASKS:
                    true_idx = int(batch[task][i].item())
                    pred_idx = int(pred_indices[task][i])
                    row[f"true_{task}_idx"] = true_idx
                    row[f"pred_{task}_idx"] = pred_idx
                    row[f"true_{task}"] = label_names[task][true_idx]
                    row[f"pred_{task}"] = label_names[task][pred_idx]
                    row[f"{task}_correct"] = int(true_idx == pred_idx)
                rows.append(row)
    return rows, label_names


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_rows_by_sample(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[int, str, str], Dict[str, object]] = {}
    for row in rows:
        key = (int(row["sample_index"]), str(row["obj_class"]), str(row["episode_name"]))
        if key not in grouped:
            grouped[key] = {
                "sample_index": row["sample_index"],
                "obj_class": row["obj_class"],
                "episode_name": row["episode_name"],
                "episode_path": row["episode_path"],
                "true_mass": row["true_mass"],
                "true_stiffness": row["true_stiffness"],
                "true_material": row["true_material"],
            }
        seed = int(row["seed"])
        grouped[key][f"seed_{seed}_gate_score"] = row["gate_score"]
        for task in TASKS:
            grouped[key][f"seed_{seed}_pred_{task}"] = row[f"pred_{task}"]
            grouped[key][f"seed_{seed}_{task}_correct"] = row[f"{task}_correct"]

    wide_rows = [grouped[key] for key in sorted(grouped.keys())]
    for row in wide_rows:
        gate_values = [float(v) for k, v in row.items() if k.endswith("_gate_score")]
        row["mean_gate_score"] = float(np.mean(gate_values)) if gate_values else float("nan")
        for task in TASKS:
            correct_values = [float(v) for k, v in row.items() if k.endswith(f"_{task}_correct")]
            row[f"{task}_correct_rate"] = float(np.mean(correct_values)) if correct_values else float("nan")
    return wide_rows


def object_class_boundaries(sample_labels: Sequence[str]) -> List[int]:
    boundaries = []
    prev = None
    for idx, name in enumerate(sample_labels):
        if prev is not None and name != prev:
            boundaries.append(idx - 0.5)
        prev = name
    return boundaries


def make_categorical_colormap(num_classes: int) -> ListedColormap:
    base = plt.cm.get_cmap("tab10", max(num_classes, 3))
    colors = [base(i) for i in range(num_classes)]
    return ListedColormap(colors[:num_classes])


def render_task_prediction_heatmaps(
    split_name: str,
    output_dir: Path,
    seeds: Sequence[int],
    sample_rows: Sequence[Dict[str, object]],
    label_names: Dict[str, List[str]],
) -> None:
    num_samples = len(sample_rows)
    fig_height = max(10.0, num_samples * 0.22)
    fig, axes = plt.subplots(1, len(TASKS), figsize=(18, fig_height), constrained_layout=True)
    if len(TASKS) == 1:
        axes = [axes]

    sample_labels = [f"{row['obj_class']}/{row['episode_name']}" for row in sample_rows]
    boundaries = object_class_boundaries([str(row["obj_class"]) for row in sample_rows])

    for ax, task in zip(axes, TASKS):
        names = label_names[task]
        matrix = np.full((num_samples, len(seeds) + 1), -1, dtype=np.int64)
        for i, row in enumerate(sample_rows):
            matrix[i, 0] = names.index(str(row[f"true_{task}"]))
            for j, seed in enumerate(seeds, start=1):
                matrix[i, j] = names.index(str(row[f"seed_{seed}_pred_{task}"]))

        cmap = make_categorical_colormap(len(names))
        norm = BoundaryNorm(np.arange(-0.5, len(names) + 0.5, 1), cmap.N)
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(f"{split_name} | {task} predictions")
        ax.set_xticks(range(len(seeds) + 1))
        ax.set_xticklabels(["GT", *[str(seed) for seed in seeds]], rotation=45, ha="right")
        ax.set_yticks(range(num_samples))
        ax.set_yticklabels(sample_labels, fontsize=5)
        for boundary in boundaries:
            ax.axhline(boundary, color="white", linewidth=1.0, alpha=0.8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_ticks(range(len(names)))
        cbar.set_ticklabels(names)

    fig.suptitle("Per-sample predicted labels (GT column + 5 seed predictions)")
    fig.savefig(output_dir / f"{split_name}_sample_prediction_labels.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_task_correctness_heatmaps(
    split_name: str,
    output_dir: Path,
    seeds: Sequence[int],
    sample_rows: Sequence[Dict[str, object]],
) -> None:
    num_samples = len(sample_rows)
    fig_height = max(10.0, num_samples * 0.20)
    fig, axes = plt.subplots(1, len(TASKS), figsize=(16, fig_height), constrained_layout=True)
    if len(TASKS) == 1:
        axes = [axes]

    sample_labels = [f"{row['obj_class']}/{row['episode_name']}" for row in sample_rows]
    boundaries = object_class_boundaries([str(row["obj_class"]) for row in sample_rows])
    cmap = ListedColormap(["#c73e1d", "#2a9d8f"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    for ax, task in zip(axes, TASKS):
        matrix = np.full((num_samples, len(seeds)), -1, dtype=np.int64)
        for i, row in enumerate(sample_rows):
            for j, seed in enumerate(seeds):
                matrix[i, j] = int(row[f"seed_{seed}_{task}_correct"])
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(f"{split_name} | {task} correctness")
        ax.set_xticks(range(len(seeds)))
        ax.set_xticklabels([str(seed) for seed in seeds], rotation=45, ha="right")
        ax.set_yticks(range(num_samples))
        ax.set_yticklabels(sample_labels, fontsize=5)
        for boundary in boundaries:
            ax.axhline(boundary, color="white", linewidth=1.0, alpha=0.8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["wrong", "correct"])

    fig.suptitle("Per-sample correctness by seed")
    fig.savefig(output_dir / f"{split_name}_sample_prediction_correctness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_object_accuracy_heatmap(
    split_name: str,
    output_dir: Path,
    seeds: Sequence[int],
    rows: Sequence[Dict[str, object]],
) -> None:
    obj_classes = sorted({str(row["obj_class"]) for row in rows})
    columns = [f"{seed}:{task}" for seed in seeds for task in TASKS]
    matrix = np.zeros((len(obj_classes), len(columns)), dtype=np.float64)
    for i, obj_class in enumerate(obj_classes):
        for j, seed in enumerate(seeds):
            subset = [row for row in rows if row["obj_class"] == obj_class and int(row["seed"]) == seed]
            for k, task in enumerate(TASKS):
                value = float(np.mean([row[f"{task}_correct"] for row in subset])) if subset else np.nan
                matrix[i, j * len(TASKS) + k] = value

    fig_height = max(4.0, len(obj_classes) * 0.6)
    fig, ax = plt.subplots(figsize=(max(10, len(columns) * 0.65), fig_height), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"{split_name} | object-class accuracy by seed/task")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticks(range(len(obj_classes)))
    ax.set_yticklabels(obj_classes)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("accuracy")
    fig.savefig(output_dir / f"{split_name}_object_accuracy_by_seed_task.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(
    split_name: str,
    output_dir: Path,
    seeds: Sequence[int],
    rows: Sequence[Dict[str, object]],
    sample_rows: Sequence[Dict[str, object]],
) -> None:
    lines = []
    lines.append(f"# {split_name} Sample-Level Prediction Summary")
    lines.append("")
    lines.append(f"- Seeds: `{', '.join(str(seed) for seed in seeds)}`")
    lines.append(f"- Num rows (seed x sample): `{len(rows)}`")
    lines.append(f"- Num samples: `{len(sample_rows)}`")
    lines.append("")

    lines.append("## Mean Accuracy Across Seeds")
    lines.append("")
    for task in TASKS:
        acc = float(np.mean([row[f"{task}_correct"] for row in rows]))
        lines.append(f"- `{task}`: `{acc * 100:.2f}%`")
    lines.append("")

    lines.append("## Hardest Samples")
    lines.append("")
    hardest = sorted(
        sample_rows,
        key=lambda row: (
            row["mass_correct_rate"] + row["stiffness_correct_rate"] + row["material_correct_rate"]
        ),
    )[:15]
    lines.append("| Sample | Mass | Stiffness | Material | Mean Gate |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in hardest:
        sample_name = f"{row['obj_class']}/{row['episode_name']}"
        lines.append(
            f"| `{sample_name}` | "
            f"{row['mass_correct_rate'] * 100:.0f}% | "
            f"{row['stiffness_correct_rate'] * 100:.0f}% | "
            f"{row['material_correct_rate'] * 100:.0f}% | "
            f"{row['mean_gate_score']:.3f} |"
        )
    lines.append("")

    lines.append("## Output Files")
    lines.append("")
    lines.append(f"- `{split_name}_sample_predictions_long.csv`")
    lines.append(f"- `{split_name}_sample_predictions_wide.csv`")
    lines.append(f"- `{split_name}_sample_prediction_labels.png`")
    lines.append(f"- `{split_name}_sample_prediction_correctness.png`")
    lines.append(f"- `{split_name}_object_accuracy_by_seed_task.png`")
    lines.append("")

    (output_dir / f"{split_name}_summary.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_split(
    run_root: Path,
    data_root: Path,
    output_root: Path,
    split_name: str,
    args: argparse.Namespace,
) -> None:
    run_dirs = iter_run_dirs(run_root)
    all_rows: List[Dict[str, object]] = []
    label_names_ref: Dict[str, List[str]] | None = None
    for run_dir in run_dirs:
        rows, label_names = collect_rows_for_run(run_dir, split_name, data_root, args)
        all_rows.extend(rows)
        if label_names_ref is None:
            label_names_ref = label_names

    if label_names_ref is None:
        raise RuntimeError(f"No runs found under {run_root}")

    split_output_dir = output_root / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)

    long_fieldnames = [
        "split",
        "seed",
        "run_name",
        "checkpoint_path",
        "sample_index",
        "obj_class",
        "episode_name",
        "episode_path",
        "gate_score",
    ]
    for task in TASKS:
        long_fieldnames.extend(
            [
                f"true_{task}_idx",
                f"pred_{task}_idx",
                f"true_{task}",
                f"pred_{task}",
                f"{task}_correct",
            ]
        )
    write_csv(split_output_dir / f"{split_name}_sample_predictions_long.csv", all_rows, long_fieldnames)

    sample_rows = group_rows_by_sample(all_rows)
    wide_fieldnames = list(sample_rows[0].keys())
    write_csv(split_output_dir / f"{split_name}_sample_predictions_wide.csv", sample_rows, wide_fieldnames)

    seeds = sorted({int(row["seed"]) for row in all_rows})
    render_task_prediction_heatmaps(split_name, split_output_dir, seeds, sample_rows, label_names_ref)
    render_task_correctness_heatmaps(split_name, split_output_dir, seeds, sample_rows)
    render_object_accuracy_heatmap(split_name, split_output_dir, seeds, all_rows)
    write_summary_markdown(split_name, split_output_dir, seeds, all_rows, sample_rows)

    meta = {
        "run_root": str(run_root),
        "data_root": str(data_root),
        "split": split_name,
        "seeds": seeds,
        "num_runs": len(run_dirs),
        "num_rows": len(all_rows),
        "num_samples": len(sample_rows),
    }
    (split_output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-sample predictions for fusion_gating_online_v2 multi-seed runs and visualize them."
    )
    parser.add_argument("--run-root", type=Path, default=Path(DEFAULT_RUN_ROOT))
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--splits",
        type=str,
        default="ood_test",
        help="Comma-separated splits to analyze, e.g. 'test,ood_test'.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tactile-len", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    data_root = args.data_root.resolve()
    if args.output_dir is None:
        output_root = run_root / "sample_prediction_analysis"
    else:
        output_root = args.output_dir.resolve()
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    if not splits:
        raise ValueError("No split specified")

    for split_name in splits:
        analyze_split(run_root=run_root, data_root=data_root, output_root=output_root, split_name=split_name, args=args)
        print(f"Saved sample-level analysis for {split_name} to {output_root / split_name}")


if __name__ == "__main__":
    main()
