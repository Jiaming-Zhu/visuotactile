import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_fusion import FusionModel, build_loader, resolve_device, set_seed


DEFAULT_MODEL_DIRS = [
    "/home/martina/Y3_Project/visuotactile/outputs/fusion_model_clean",
    "/home/martina/Y3_Project/visuotactile/outputs/fusion_seed123",
    "/home/martina/Y3_Project/visuotactile/outputs/fusion_seed456",
    "/home/martina/Y3_Project/visuotactile/outputs/fusion_seed789",
    "/home/martina/Y3_Project/visuotactile/outputs/fusion_seed2024",
]

MODE_KEYS = ["full", "tactile_only", "vision_only"]
METRIC_KEYS = ["loss", "mass", "stiffness", "material", "avg_acc"]


def normalize_split_name(split_name: str) -> str:
    normalized = split_name.strip().lower()
    if normalized in {"id", "id_test", "test"}:
        return "test"
    if normalized in {"ood", "ood_test"}:
        return "ood_test"
    raise ValueError(f"Unsupported split '{split_name}'. Use one of: id_test/test, ood_test/ood.")


def infer_model_tag(model_dir: Path) -> str:
    match = re.search(r"seed(\d+)", model_dir.name)
    if match:
        return f"seed{match.group(1)}"
    if model_dir.name.endswith("_model_clean"):
        return "seed42"
    return model_dir.name


def _downsample_tactile_padding_mask(padding_mask: torch.Tensor, num_tac_tokens: int) -> torch.Tensor:
    tac_mask = padding_mask.float().unsqueeze(1)
    tac_mask = F.max_pool1d(tac_mask, kernel_size=2, stride=2)
    tac_mask = F.max_pool1d(tac_mask, kernel_size=2, stride=2)
    tac_mask = F.max_pool1d(tac_mask, kernel_size=2, stride=2)
    tac_mask = tac_mask.squeeze(1) > 0.5
    return tac_mask[:, :num_tac_tokens]


def forward_with_modality_mask(
    model: FusionModel,
    images: torch.Tensor,
    tactile: torch.Tensor,
    padding_mask: torch.Tensor,
    mode: str,
) -> Dict[str, torch.Tensor]:
    if mode not in MODE_KEYS:
        raise ValueError(f"Unsupported mode: {mode}")

    bsz = images.shape[0]
    device = images.device

    v = model.vis_backbone(images)
    v = model.vis_proj(v)
    v_tokens = v.flatten(2).transpose(1, 2)
    num_vis_tokens = v_tokens.shape[1]

    t = model.tac_encoder(tactile)
    t_tokens = t.transpose(1, 2)
    num_tac_tokens = t_tokens.shape[1]

    cls_token = model.cls_token.expand(bsz, -1, -1)
    x = torch.cat([cls_token, v_tokens, t_tokens], dim=1)
    seq_len = x.shape[1]
    x = x + model.pos_emb[:, :seq_len, :]

    cls_mask = torch.zeros(bsz, 1, dtype=torch.bool, device=device)
    vis_mask = torch.zeros(bsz, num_vis_tokens, dtype=torch.bool, device=device)
    tac_mask = _downsample_tactile_padding_mask(padding_mask, num_tac_tokens)

    if mode == "tactile_only":
        vis_mask[:] = True
    elif mode == "vision_only":
        tac_mask[:] = True

    full_mask = torch.cat([cls_mask, vis_mask, tac_mask], dim=1)
    x = model.transformer_encoder(x, src_key_padding_mask=full_mask)
    cls_out = x[:, 0, :]

    return {
        "mass": model.head_mass(cls_out),
        "stiffness": model.head_stiffness(cls_out),
        "material": model.head_material(cls_out),
    }


@torch.no_grad()
def evaluate_once(
    model: FusionModel,
    loader,
    device: torch.device,
    mode: str,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    correct = {"mass": 0, "stiffness": 0, "material": 0}

    for batch in loader:
        images = batch["image"].to(device)
        tactile = batch["tactile"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        labels = {k: batch[k].to(device) for k in ["mass", "stiffness", "material"]}

        outputs = forward_with_modality_mask(model, images, tactile, padding_mask, mode=mode)

        loss = (
            criterion(outputs["mass"], labels["mass"])
            + criterion(outputs["stiffness"], labels["stiffness"])
            + criterion(outputs["material"], labels["material"])
        )

        bsz = images.size(0)
        total_loss += loss.item() * bsz
        total_samples += bsz

        for task in correct:
            correct[task] += (outputs[task].argmax(dim=1) == labels[task]).sum().item()

    mass_acc = correct["mass"] / total_samples
    stiff_acc = correct["stiffness"] / total_samples
    material_acc = correct["material"] / total_samples

    return {
        "loss": total_loss / total_samples,
        "mass": mass_acc,
        "stiffness": stiff_acc,
        "material": material_acc,
        "avg_acc": float(np.mean([mass_acc, stiff_acc, material_acc])),
    }


def _metric_stats(metric_values: List[float]) -> Dict[str, object]:
    arr = np.asarray(metric_values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "values": [float(v) for v in arr.tolist()],
    }


def _summarize_metric_dict_list(metric_dicts: List[Dict[str, float]]) -> Dict[str, object]:
    summary = {"num_records": len(metric_dicts)}
    for key in METRIC_KEYS:
        summary[key] = _metric_stats([m[key] for m in metric_dicts])
    return summary


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
    fallback_cfg: argparse.Namespace,
) -> Tuple[FusionModel, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})

    model = FusionModel(
        fusion_dim=cfg.get("fusion_dim", fallback_cfg.fusion_dim),
        num_heads=cfg.get("num_heads", fallback_cfg.num_heads),
        dropout=cfg.get("dropout", fallback_cfg.dropout),
        num_layers=cfg.get("num_layers", fallback_cfg.num_layers),
        freeze_visual=True,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-seed Fusion inference ablations with token-level masking. "
            "Modes: full, tactile_only(mask vision tokens), vision_only(mask tactile tokens)."
        )
    )
    parser.add_argument("--data_root", type=str, default="/home/martina/Y3_Project/Plaintextdataset")
    parser.add_argument("--model_dirs", nargs="+", default=DEFAULT_MODEL_DIRS)
    parser.add_argument("--checkpoint_name", type=str, default="best_model.pth")
    parser.add_argument("--splits", nargs="+", default=["id_test", "ood_test"])
    parser.add_argument("--runs_per_setting", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_tactile_len", type=int, default=3000)
    parser.add_argument("--seed_base", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_root", type=str, default="outputs/fusion_infer_ablation_mask")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic CuDNN settings.")

    # Fallbacks if old checkpoints do not include full config.
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device: {device}")

    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print("deterministic cudnn: enabled")

    normalized_splits = [normalize_split_name(name) for name in args.splits]
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    model_dir_by_tag: Dict[str, str] = {}

    for model_idx, model_dir_raw in enumerate(args.model_dirs):
        model_dir = Path(model_dir_raw)
        checkpoint_path = model_dir / args.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model_tag = infer_model_tag(model_dir)
        model_dir_by_tag[model_tag] = str(model_dir)
        print(f"\n=== Model {model_tag} | {checkpoint_path} ===")

        model, checkpoint = _load_model(checkpoint_path, device, args)
        print(f"checkpoint epoch: {checkpoint.get('epoch')}")

        split_loaders = {}
        for split_name in normalized_splits:
            split_dir = data_root / split_name
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Split directory not found: {split_dir}")
            split_loaders[split_name] = build_loader(
                split_dir=split_dir,
                batch_size=args.batch_size,
                max_tactile_len=args.max_tactile_len,
                num_workers=args.num_workers,
                shuffle=False,
            )

        for split_idx, split_name in enumerate(normalized_splits):
            loader = split_loaders[split_name]
            print(f"\n[{model_tag}] split={split_name} samples={len(loader.dataset)}")

            for mode_idx, mode in enumerate(MODE_KEYS):
                for run_idx in range(1, args.runs_per_setting + 1):
                    run_seed = args.seed_base + model_idx * 1000 + split_idx * 100 + mode_idx * 10 + run_idx
                    set_seed(run_seed)
                    metrics = evaluate_once(model=model, loader=loader, device=device, mode=mode)

                    record = {
                        "model_tag": model_tag,
                        "model_dir": str(model_dir),
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_epoch": checkpoint.get("epoch"),
                        "split": split_name,
                        "mode": mode,
                        "run_idx": run_idx,
                        "run_seed": run_seed,
                        "metrics": metrics,
                    }
                    records.append(record)

                    print(
                        f"  mode={mode:<12} run={run_idx} "
                        f"loss={metrics['loss']:.4f} "
                        f"mass={metrics['mass']:.2%} "
                        f"stiff={metrics['stiffness']:.2%} "
                        f"mat={metrics['material']:.2%} "
                        f"avg={metrics['avg_acc']:.2%}"
                    )

    per_run_path = output_root / "per_run.jsonl"
    with per_run_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    grouped_by_seed: Dict[str, Dict[str, Dict[str, List[Dict[str, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for rec in records:
        grouped_by_seed[rec["model_tag"]][rec["split"]][rec["mode"]].append(rec["metrics"])

    summary_by_seed = {"model_dir_by_tag": model_dir_by_tag, "results": {}}
    for model_tag, split_map in grouped_by_seed.items():
        summary_by_seed["results"][model_tag] = {}
        for split_name, mode_map in split_map.items():
            summary_by_seed["results"][model_tag][split_name] = {}
            for mode, metric_list in mode_map.items():
                summary_by_seed["results"][model_tag][split_name][mode] = _summarize_metric_dict_list(metric_list)

    summary_by_seed_path = output_root / "summary_by_seed.json"
    summary_by_seed_path.write_text(json.dumps(summary_by_seed, indent=2, ensure_ascii=False), encoding="utf-8")

    grouped_all_runs: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    grouped_seed_means: Dict[str, Dict[str, List[Dict[str, float]]]] = defaultdict(lambda: defaultdict(list))

    for rec in records:
        grouped_all_runs[rec["split"]][rec["mode"]].append(rec["metrics"])

    for model_tag, split_map in grouped_by_seed.items():
        for split_name, mode_map in split_map.items():
            for mode, metric_list in mode_map.items():
                mean_metrics = {}
                for metric_key in METRIC_KEYS:
                    mean_metrics[metric_key] = float(np.mean([m[metric_key] for m in metric_list]))
                grouped_seed_means[split_name][mode].append(mean_metrics)

    summary_overall = {}
    for split_name, mode_map in grouped_all_runs.items():
        summary_overall[split_name] = {}
        for mode, metric_list in mode_map.items():
            seed_mean_list = grouped_seed_means[split_name][mode]
            summary_overall[split_name][mode] = {
                "num_total_runs": len(metric_list),
                "num_models": len(seed_mean_list),
                "all_runs": _summarize_metric_dict_list(metric_list),
                "seed_means": _summarize_metric_dict_list(seed_mean_list),
            }

    summary_overall_path = output_root / "summary_overall.json"
    summary_overall_path.write_text(json.dumps(summary_overall, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSaved:")
    print(f"  per-run records: {per_run_path}")
    print(f"  by-seed summary: {summary_by_seed_path}")
    print(f"  overall summary: {summary_overall_path}")


if __name__ == "__main__":
    main()
