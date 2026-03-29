import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader


THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parents[1]
REPO_DIR = THIS_FILE.parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(REPO_DIR.parent))

from train_fusion_gating_online import build_model, effective_padding_mask  # noqa: E402
from train_fusion_gating2 import resolve_device  # noqa: E402
from visualization.export_gating_sample_predictions import (  # noqa: E402
    AnalysisRoboticGraspDataset,
    iter_run_dirs,
    ordered_label_names,
    resolve_override_props_config,
)


DEFAULT_RUN_ROOT = "/home/martina/Y3_Project/visuotactile/outputs/fusion_gating_online_v2_multiseed"
DEFAULT_DATA_ROOT = "/home/martina/Y3_Project/Plaintextdataset"
DEFAULT_CLASSES = ("Cardbox_hollow_noise", "YogaBrick_Foil_ANCHOR")


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


@torch.no_grad()
def collect_two_class_embeddings(
    run_dir: Path,
    split_name: str,
    data_root: Path,
    target_classes: Sequence[str],
    cli_args: argparse.Namespace,
) -> Dict[str, object]:
    device = resolve_device(cli_args.device)
    checkpoint_path = run_dir / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    runtime_args = build_runtime_args(cfg, cli_args)
    override_props_config = resolve_override_props_config(split_name, cfg)

    loader = build_loader(
        data_root / split_name,
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

    hook_cache: List[torch.Tensor] = []

    def hook_fn(_module, _inputs, output):
        hook_cache.append(output[:, 0, :].detach().cpu())

    handle = model.transformer_encoder.register_forward_hook(hook_fn)

    rows: List[Dict[str, object]] = []
    embeddings: List[np.ndarray] = []
    target_classes = set(target_classes)

    for batch in loader:
        hook_cache.clear()
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
        if not hook_cache:
            raise RuntimeError("Transformer hook did not capture CLS embeddings.")

        cls_embeddings = hook_cache[-1].numpy()
        pred_indices = {task: outputs[task].argmax(dim=1).cpu().tolist() for task in ("mass", "stiffness", "material")}
        gate_scores = outputs["gate_score"].cpu().tolist()

        for i, obj_class in enumerate(batch["obj_class"]):
            if obj_class not in target_classes:
                continue
            row = {
                "seed": int(cfg.get("seed")),
                "run_name": run_dir.name,
                "obj_class": obj_class,
                "episode_name": batch["episode_name"][i],
                "episode_path": batch["episode_path"][i],
                "gate_score": float(gate_scores[i]),
            }
            for task in ("mass", "stiffness", "material"):
                true_idx = int(batch[task][i].item())
                pred_idx = int(pred_indices[task][i])
                row[f"true_{task}"] = label_names[task][true_idx]
                row[f"pred_{task}"] = label_names[task][pred_idx]
            rows.append(row)
            embeddings.append(cls_embeddings[i])

    handle.remove()

    if not rows:
        raise RuntimeError(f"No matching rows found for {target_classes}")

    return {
        "rows": rows,
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "label_names": label_names,
    }


def run_tsne(embeddings: np.ndarray, perplexity: int, seed: int) -> np.ndarray:
    n_samples = embeddings.shape[0]
    safe_perplexity = max(2, min(perplexity, n_samples - 1))
    reducer = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=1000,
    )
    return reducer.fit_transform(embeddings)


def safe_silhouette(coords: np.ndarray, labels: np.ndarray) -> float | None:
    unique = np.unique(labels)
    if len(unique) < 2:
        return None
    counts = [int(np.sum(labels == u)) for u in unique]
    if min(counts) < 2:
        return None
    return float(silhouette_score(coords, labels))


def write_long_csv(path: Path, rows: Sequence[Dict[str, object]], coords: np.ndarray) -> None:
    fieldnames = list(rows[0].keys()) + ["tsne_x", "tsne_y"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, xy in zip(rows, coords):
            out = dict(row)
            out["tsne_x"] = float(xy[0])
            out["tsne_y"] = float(xy[1])
            writer.writerow(out)


def plot_by_true_object(seed_results: Sequence[Dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(seed_results), figsize=(5.5 * len(seed_results), 5), constrained_layout=True)
    if len(seed_results) == 1:
        axes = [axes]

    colors = {
        "Cardbox_hollow_noise": "#d1495b",
        "YogaBrick_Foil_ANCHOR": "#00798c",
    }
    for ax, result in zip(axes, seed_results):
        rows = result["rows"]
        coords = result["coords"]
        for obj_class in DEFAULT_CLASSES:
            mask = np.array([row["obj_class"] == obj_class for row in rows], dtype=bool)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=40,
                alpha=0.9,
                color=colors[obj_class],
                label=obj_class,
                edgecolors="none",
            )
        score = result["silhouette_object"]
        score_text = "n/a" if score is None else f"{score:.3f}"
        ax.set_title(f"seed {result['seed']} | silhouette={score_text}")
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")
        ax.grid(alpha=0.2, linestyle="--")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="True object")
    fig.suptitle("Gating CLS embeddings | true object labels")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_by_pred_material(seed_results: Sequence[Dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(seed_results), figsize=(5.5 * len(seed_results), 5), constrained_layout=True)
    if len(seed_results) == 1:
        axes = [axes]

    material_colors = {
        "foam": "#577590",
        "hollow_container": "#d1495b",
        "filled_container": "#f8961e",
        "wood": "#6a994e",
        "sponge": "#9c89b8",
    }
    marker_of = {
        "Cardbox_hollow_noise": "o",
        "YogaBrick_Foil_ANCHOR": "^",
    }

    for ax, result in zip(axes, seed_results):
        rows = result["rows"]
        coords = result["coords"]
        for material in sorted({row["pred_material"] for row in rows}):
            for obj_class in DEFAULT_CLASSES:
                mask = np.array(
                    [row["pred_material"] == material and row["obj_class"] == obj_class for row in rows],
                    dtype=bool,
                )
                if not np.any(mask):
                    continue
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=46,
                    alpha=0.9,
                    color=material_colors.get(material, "#444444"),
                    marker=marker_of[obj_class],
                    label=f"{material} | {obj_class}",
                    edgecolors="none",
                )
        ax.set_title(f"seed {result['seed']} | pred material")
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")
        ax.grid(alpha=0.2, linestyle="--")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="Pred material | object")
    fig.suptitle("Gating CLS embeddings | predicted material")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, seed_results: Sequence[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Hollow vs YogaBrick Foil t-SNE Summary")
    lines.append("")
    lines.append("Compared classes:")
    for name in DEFAULT_CLASSES:
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("| Seed | Silhouette (true object) | Hollow pred stiffness | Hollow pred material | Yoga pred stiffness | Yoga pred material | Hollow mean gate | Yoga mean gate |")
    lines.append("| --- | ---: | --- | --- | --- | --- | ---: | ---: |")
    for result in seed_results:
        hollow_rows = [row for row in result["rows"] if row["obj_class"] == "Cardbox_hollow_noise"]
        yoga_rows = [row for row in result["rows"] if row["obj_class"] == "YogaBrick_Foil_ANCHOR"]
        hollow_stiff = Counter(row["pred_stiffness"] for row in hollow_rows)
        hollow_mat = Counter(row["pred_material"] for row in hollow_rows)
        yoga_stiff = Counter(row["pred_stiffness"] for row in yoga_rows)
        yoga_mat = Counter(row["pred_material"] for row in yoga_rows)
        sil = result["silhouette_object"]
        sil_text = "n/a" if sil is None else f"{sil:.3f}"
        lines.append(
            f"| {result['seed']} | {sil_text} | "
            f"`{dict(hollow_stiff)}` | `{dict(hollow_mat)}` | "
            f"`{dict(yoga_stiff)}` | `{dict(yoga_mat)}` | "
            f"{np.mean([row['gate_score'] for row in hollow_rows]):.3f} | "
            f"{np.mean([row['gate_score'] for row in yoga_rows]):.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize t-SNE embeddings for Cardbox_hollow_noise vs YogaBrick_Foil_ANCHOR using gating checkpoints."
    )
    parser.add_argument("--run-root", type=Path, default=Path(DEFAULT_RUN_ROOT))
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--split", type=str, default="ood_test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tactile-len", type=int, default=3000)
    parser.add_argument("--perplexity", type=int, default=20)
    parser.add_argument("--tsne-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    data_root = args.data_root.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir else run_root / "tsne_hollow_vs_yogabrick_foil")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_results: List[Dict[str, object]] = []
    for run_dir in iter_run_dirs(run_root):
        payload = collect_two_class_embeddings(
            run_dir=run_dir,
            split_name=args.split,
            data_root=data_root,
            target_classes=DEFAULT_CLASSES,
            cli_args=args,
        )
        coords = run_tsne(payload["embeddings"], perplexity=args.perplexity, seed=args.tsne_seed)
        labels = np.asarray([0 if row["obj_class"] == DEFAULT_CLASSES[0] else 1 for row in payload["rows"]], dtype=np.int64)
        silhouette = safe_silhouette(coords, labels)
        seed = int(payload["rows"][0]["seed"])
        seed_results.append(
            {
                "seed": seed,
                "rows": payload["rows"],
                "coords": coords,
                "silhouette_object": silhouette,
            }
        )
        write_long_csv(output_dir / f"seed{seed}_tsne_points.csv", payload["rows"], coords)

    seed_results.sort(key=lambda item: item["seed"])
    plot_by_true_object(seed_results, output_dir / "tsne_by_true_object.png")
    plot_by_pred_material(seed_results, output_dir / "tsne_by_pred_material.png")
    write_summary(output_dir / "summary.md", seed_results)

    meta = {
        "run_root": str(run_root),
        "data_root": str(data_root),
        "split": args.split,
        "classes": list(DEFAULT_CLASSES),
        "seeds": [item["seed"] for item in seed_results],
        "perplexity": args.perplexity,
        "tsne_seed": args.tsne_seed,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved t-SNE analysis to {output_dir}")


if __name__ == "__main__":
    main()
