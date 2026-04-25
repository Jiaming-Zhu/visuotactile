from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch

try:
    from train_fusion_gating_online import build_model, effective_padding_mask
    from train_fusion_gating2 import apply_modality_block, build_loader, resolve_device
    from train_fusion_standard_ablation import (
        apply_modality_block as standard_apply_modality_block,
        build_loader_for_eval as standard_build_loader_for_eval,
        build_model_from_config as standard_build_model_from_config,
        effective_padding_mask as standard_effective_padding_mask,
    )
    from train_tactile import TactileOnlyModel, build_loader as tactile_build_loader
    from train_vision import VisionOnlyModel, build_loader as vision_build_loader
except ImportError:  # pragma: no cover
    from visuotactile.scripts.train_fusion_gating_online import build_model, effective_padding_mask  # type: ignore
    from visuotactile.scripts.train_fusion_gating2 import (  # type: ignore
        apply_modality_block,
        build_loader,
        resolve_device,
    )
    from visuotactile.scripts.train_fusion_standard_ablation import (  # type: ignore
        apply_modality_block as standard_apply_modality_block,
        build_loader_for_eval as standard_build_loader_for_eval,
        build_model_from_config as standard_build_model_from_config,
        effective_padding_mask as standard_effective_padding_mask,
    )
    from visuotactile.scripts.train_tactile import (  # type: ignore
        TactileOnlyModel,
        build_loader as tactile_build_loader,
    )
    from visuotactile.scripts.train_vision import (  # type: ignore
        VisionOnlyModel,
        build_loader as vision_build_loader,
    )


TASKS = ("mass", "stiffness", "material")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate object-level OOD evidence for a checkpoint.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_split", choices=["val", "test", "ood_test"], default="ood_test")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--model_family",
        choices=["fusion", "standard_fusion", "vision", "tactile"],
        default="fusion",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_tactile_len", type=int, default=3000)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.set_defaults(separate_cls_tokens=False)
    parser.add_argument("--separate_cls_tokens", dest="separate_cls_tokens", action="store_true")
    parser.add_argument("--shared_cls_token", dest="separate_cls_tokens", action="store_false")
    parser.add_argument("--fixed_gate_value", type=float, default=None)
    parser.add_argument("--online_min_prefix_ratio", type=float, default=0.2)
    parser.add_argument("--min_prefix_len", type=int, default=64)
    parser.add_argument("--block_modality", choices=["none", "visual", "tactile"], default="none")
    parser.add_argument("--num_resamples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=7)
    return parser.parse_args()


def object_id_from_episode_dir(episode_dir: Path) -> str:
    return episode_dir.parent.name


def episode_dir_from_sample(sample) -> Path:
    if hasattr(sample, "img_path"):
        return Path(sample.img_path).parent
    if hasattr(sample, "tactile_path"):
        return Path(sample.tactile_path).parent
    raise AttributeError("Sample must expose either img_path or tactile_path")


def optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def summarize_object_metrics(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    grouped: MutableMapping[str, Dict[str, List[bool]]] = {}
    for row in rows:
        object_id = str(row["object_id"])
        bucket = grouped.setdefault(object_id, {task: [] for task in TASKS})
        for task in TASKS:
            bucket[task].append(int(row[f"{task}_pred"]) == int(row[f"{task}_label"]))

    per_object: Dict[str, Dict[str, float]] = {}
    macro_values = {task: [] for task in TASKS}
    macro_values["avg"] = []
    for object_id, task_hits in grouped.items():
        task_scores = {
            task: float(np.mean(np.asarray(hits, dtype=np.float64))) if hits else 0.0
            for task, hits in task_hits.items()
        }
        task_scores["avg"] = float(np.mean([task_scores[task] for task in TASKS]))
        per_object[object_id] = task_scores
        for key, value in task_scores.items():
            macro_values[key].append(value)

    object_macro = {
        key: float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0
        for key, values in macro_values.items()
    }
    return {
        "num_objects": len(per_object),
        "per_object": per_object,
        "object_macro": object_macro,
    }


def grouped_bootstrap_mean(
    per_object_avg: Mapping[str, float],
    num_resamples: int = 2000,
    seed: int = 7,
) -> Dict[str, float]:
    object_ids = sorted(per_object_avg.keys())
    values = np.asarray([float(per_object_avg[obj]) for obj in object_ids], dtype=np.float64)
    if values.size == 0:
        return {
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "num_objects": 0,
            "num_resamples": num_resamples,
            "seed": seed,
        }

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(num_resamples):
        indices = rng.integers(0, len(values), size=len(values))
        boot.append(float(np.mean(values[indices])))
    boot_arr = np.asarray(boot, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(boot_arr, 2.5)),
        "ci_high": float(np.percentile(boot_arr, 97.5)),
        "num_objects": int(len(values)),
        "num_resamples": int(num_resamples),
        "seed": int(seed),
    }


def _episode_dirs_in_loader_order(dataset) -> List[Path]:
    return [episode_dir_from_sample(sample) for sample in dataset.samples]


def _label_name_maps(dataset) -> Dict[str, Dict[int, str]]:
    return {
        "mass": {idx: name for name, idx in dataset.mass_to_idx.items()},
        "stiffness": {idx: name for name, idx in dataset.stiffness_to_idx.items()},
        "material": {idx: name for name, idx in dataset.material_to_idx.items()},
    }


def _build_loader_for_family(args: argparse.Namespace, split_dir: Path):
    if args.model_family == "fusion":
        return build_loader(split_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)
    if args.model_family == "standard_fusion":
        return standard_build_loader_for_eval(args, split_dir)
    if args.model_family == "vision":
        return vision_build_loader(split_dir, args.batch_size, args.num_workers, shuffle=False, image_size=args.image_size)
    return tactile_build_loader(split_dir, args.batch_size, args.max_tactile_len, args.num_workers, shuffle=False)


def _build_model_for_family(args: argparse.Namespace, checkpoint: Dict[str, object], dataset, device: torch.device):
    cfg = checkpoint.get("config", {})
    if args.model_family == "fusion":
        model = build_model(
            cfg,
            args,
            mass_classes=len(dataset.mass_to_idx),
            stiffness_classes=len(dataset.stiffness_to_idx),
            material_classes=len(dataset.material_to_idx),
        ).to(device)
    elif args.model_family == "standard_fusion":
        model = standard_build_model_from_config(cfg, dataset).to(device)
    elif args.model_family == "vision":
        model = VisionOnlyModel(
            fusion_dim=cfg.get("fusion_dim", args.fusion_dim),
            num_heads=cfg.get("num_heads", args.num_heads),
            dropout=cfg.get("dropout", args.dropout),
            num_layers=cfg.get("num_layers", args.num_layers),
            freeze_visual=True,
            image_size=cfg.get("image_size", args.image_size),
            mass_classes=len(dataset.mass_to_idx),
            stiffness_classes=len(dataset.stiffness_to_idx),
            material_classes=len(dataset.material_to_idx),
        ).to(device)
    else:
        model = TactileOnlyModel(
            fusion_dim=cfg.get("fusion_dim", args.fusion_dim),
            num_heads=cfg.get("num_heads", args.num_heads),
            dropout=cfg.get("dropout", args.dropout),
            num_layers=cfg.get("num_layers", args.num_layers),
            max_tactile_len=cfg.get("max_tactile_len", args.max_tactile_len),
            mass_classes=len(dataset.mass_to_idx),
            stiffness_classes=len(dataset.stiffness_to_idx),
            material_classes=len(dataset.material_to_idx),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_object_level(args: argparse.Namespace) -> Dict[str, object]:
    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=resolve_device(args.device), weights_only=False)

    device = resolve_device(args.device)
    split_dir = Path(args.data_root) / args.eval_split
    loader = _build_loader_for_family(args, split_dir)
    dataset = loader.dataset

    model = _build_model_for_family(args, checkpoint, dataset, device)

    episode_dirs = _episode_dirs_in_loader_order(dataset)
    label_names = _label_name_maps(dataset)
    rows: List[Dict[str, object]] = []
    episode_offset = 0

    with torch.no_grad():
        for batch in loader:
            if args.model_family == "fusion":
                images = batch["image"].to(device)
                tactile = batch["tactile"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                prefix_mask = effective_padding_mask(
                    padding_mask=padding_mask,
                    train_mode=False,
                    online_train_prob=0.0,
                    online_min_prefix_ratio=args.online_min_prefix_ratio,
                    min_prefix_len=args.min_prefix_len,
                    fixed_ratio=None,
                )
                images, tactile = apply_modality_block(images, tactile, args.block_modality)
                outputs = model(images, tactile, padding_mask=prefix_mask)
                gate_scores = outputs["gate_score"].view(-1).cpu().tolist()
                batch_size = images.size(0)
            elif args.model_family == "standard_fusion":
                images = batch["image"].to(device)
                tactile = batch["tactile"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                prefix_mask = standard_effective_padding_mask(
                    padding_mask=padding_mask,
                    train_mode=False,
                    online_train_prob=0.0,
                    online_min_prefix_ratio=args.online_min_prefix_ratio,
                    min_prefix_len=args.min_prefix_len,
                    fixed_ratio=None,
                )
                images, tactile = standard_apply_modality_block(images, tactile, args.block_modality)
                outputs = model(images, tactile, padding_mask=prefix_mask)
                gate_scores = [None] * images.size(0)
                batch_size = images.size(0)
            elif args.model_family == "vision":
                images = batch["image"].to(device)
                outputs = model(images)
                gate_scores = [None] * images.size(0)
                batch_size = images.size(0)
            else:
                tactile = batch["tactile"].to(device, non_blocking=True)
                padding_mask = batch["padding_mask"].to(device, non_blocking=True)
                outputs = model(tactile, padding_mask=padding_mask)
                gate_scores = [None] * tactile.size(0)
                batch_size = tactile.size(0)

            batch_episode_dirs = episode_dirs[episode_offset : episode_offset + batch_size]
            episode_offset += batch_size

            preds = {task: outputs[task].argmax(dim=1).cpu().tolist() for task in TASKS}
            labels = {task: batch[task].cpu().tolist() for task in TASKS}

            for idx, episode_dir in enumerate(batch_episode_dirs):
                record: Dict[str, object] = {
                    "episode_dir": str(episode_dir),
                    "object_id": object_id_from_episode_dir(episode_dir),
                    "gate_score": optional_float(gate_scores[idx]),
                }
                for task in TASKS:
                    label_idx = int(labels[task][idx])
                    pred_idx = int(preds[task][idx])
                    record[f"{task}_label"] = label_idx
                    record[f"{task}_pred"] = pred_idx
                    record[f"{task}_label_name"] = label_names[task][label_idx]
                    record[f"{task}_pred_name"] = label_names[task][pred_idx]
                rows.append(record)

    summary = summarize_object_metrics(rows)
    bootstrap = grouped_bootstrap_mean(
        {obj: metrics["avg"] for obj, metrics in summary["per_object"].items()},
        num_resamples=args.num_resamples,
        seed=args.bootstrap_seed,
    )

    result = {
        "split": args.eval_split,
        "checkpoint": str(ckpt_path),
        "model_family": args.model_family,
        "block_modality": args.block_modality,
        "num_episodes": len(rows),
        "num_objects": summary["num_objects"],
        "object_macro": summary["object_macro"],
        "grouped_bootstrap_avg": bootstrap,
        "per_object": summary["per_object"],
        "rows": rows,
    }
    return result


def main() -> None:
    args = parse_args()
    result = evaluate_object_level(args)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = "" if args.block_modality == "none" else f"_block_{args.block_modality}"
        output_dir = Path(args.checkpoint).parent / f"object_level_{args.eval_split}{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "object_level_results.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved object-level summary to {out_path}")


if __name__ == "__main__":
    main()
