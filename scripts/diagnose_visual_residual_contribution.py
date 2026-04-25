import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

try:
    from train_fusion_gating_online import TASKS, build_model, effective_padding_mask
    from train_fusion_gating2 import RoboticGraspDataset, resolve_device, set_seed
except ImportError:  # pragma: no cover
    from visuotactile.scripts.train_fusion_gating_online import TASKS, build_model, effective_padding_mask  # type: ignore
    from visuotactile.scripts.train_fusion_gating2 import RoboticGraspDataset, resolve_device, set_seed  # type: ignore


class IndexedRoboticGraspDataset(RoboticGraspDataset):
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        item["dataset_index"] = torch.tensor(idx, dtype=torch.long)
        return item


class VisualMismatchDataset(Dataset):
    def __init__(self, base_dataset: IndexedRoboticGraspDataset, mismatch_indices: List[int]) -> None:
        self.base_dataset = base_dataset
        self.mismatch_indices = mismatch_indices
        self.mass_to_idx = base_dataset.mass_to_idx
        self.stiffness_to_idx = base_dataset.stiffness_to_idx
        self.material_to_idx = base_dataset.material_to_idx

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.base_dataset[idx]
        source_idx = self.mismatch_indices[idx]
        source_item = self.base_dataset[source_idx]
        item["image"] = source_item["image"]
        item["mismatch_source_index"] = torch.tensor(source_idx, dtype=torch.long)
        return item


def build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def build_label_signature(dataset: IndexedRoboticGraspDataset) -> List[int]:
    signatures: List[int] = []
    for sample in dataset.samples:
        labels = sample.labels
        signatures.append(int(labels["mass"]) * 100 + int(labels["stiffness"]) * 10 + int(labels["material"]))
    return signatures


def build_global_mismatch_indices(dataset: IndexedRoboticGraspDataset, seed: int) -> List[int]:
    rng = random.Random(seed)
    signatures = build_label_signature(dataset)
    all_indices = list(range(len(dataset)))
    mismatch_indices: List[int] = []

    for target_idx, target_sig in enumerate(signatures):
        diff_candidates = [idx for idx in all_indices if idx != target_idx and signatures[idx] != target_sig]
        if not diff_candidates:
            diff_candidates = [idx for idx in all_indices if idx != target_idx]
        if not diff_candidates:
            mismatch_indices.append(target_idx)
            continue
        mismatch_indices.append(rng.choice(diff_candidates))
    return mismatch_indices


def make_prefix_mask(
    padding_mask: torch.Tensor,
    prefix_ratio: float,
    min_prefix_len: int,
) -> torch.Tensor:
    fixed_ratio = None if prefix_ratio >= 1.0 else prefix_ratio
    return effective_padding_mask(
        padding_mask=padding_mask,
        train_mode=False,
        online_train_prob=0.0,
        online_min_prefix_ratio=0.1,
        min_prefix_len=min_prefix_len,
        fixed_ratio=fixed_ratio,
    )


def model_forward_with_forced_gate(
    model: nn.Module,
    img: torch.Tensor,
    tac: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
    force_gate: Optional[float],
) -> Dict[str, torch.Tensor]:
    bsz = img.shape[0]
    device = img.device

    v = model.vis_backbone(img)
    v = model.vis_proj(v)
    v_tokens = v.flatten(2).transpose(1, 2)
    num_vis_tokens = v_tokens.shape[1]

    t = model.tac_encoder(tac)
    t_tokens = t.transpose(1, 2)
    num_tac_tokens = t_tokens.shape[1]

    v_global = v_tokens.mean(dim=1)
    full_mask = None
    if padding_mask is not None:
        tac_mask = padding_mask.float().unsqueeze(1)
        tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
        tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
        tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
        tac_mask = tac_mask.squeeze(1) > 0.5
        tac_mask = tac_mask[:, :num_tac_tokens]

        tac_mask_float = (~tac_mask).unsqueeze(-1).float()
        t_global = (t_tokens * tac_mask_float).sum(dim=1) / (tac_mask_float.sum(dim=1) + 1e-8)
        cls_vis_mask = torch.zeros(
            bsz,
            (3 if model.separate_cls_tokens else 1) + num_vis_tokens,
            dtype=torch.bool,
            device=device,
        )
        full_mask = torch.cat([cls_vis_mask, tac_mask], dim=1)
    else:
        t_global = t_tokens.mean(dim=1)

    if force_gate is None:
        vt_global = torch.cat([v_global, t_global], dim=-1)
        g = model.gate_mlp(vt_global)
    else:
        g = v_tokens.new_full((bsz, 1), float(force_gate))

    g_expand = g.unsqueeze(1)
    v_tokens_gated = g_expand * v_tokens + (1.0 - g_expand) * model.t_null

    if model.separate_cls_tokens:
        cls_tokens = model.task_cls_tokens.expand(bsz, -1, -1)
    else:
        cls_tokens = model.cls_token.expand(bsz, -1, -1)

    x = torch.cat([cls_tokens, v_tokens_gated, t_tokens], dim=1)
    seq_len = x.shape[1]
    x = x + model.pos_emb[:, :seq_len, :]
    x = model.transformer_encoder(x, src_key_padding_mask=full_mask)

    mass_cls = x[:, model.cls_indices["mass"], :]
    stiffness_cls = x[:, model.cls_indices["stiffness"], :]
    material_cls = x[:, model.cls_indices["material"], :]
    return {
        "mass": model.head_mass(mass_cls),
        "stiffness": model.head_stiffness(stiffness_cls),
        "material": model.head_material(material_cls),
        "gate_score": g.squeeze(-1),
    }


@torch.no_grad()
def evaluate_condition(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    prefix_ratio: float,
    min_prefix_len: int,
    force_gate: Optional[float],
) -> Dict[str, object]:
    model.eval()
    total_samples = 0
    total_gate = 0.0
    correct = {task: 0 for task in TASKS}
    all_gate_scores: List[float] = []

    for batch in loader:
        images = batch["image"].to(device)
        tactile = batch["tactile"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        prefix_mask = make_prefix_mask(padding_mask, prefix_ratio=prefix_ratio, min_prefix_len=min_prefix_len)
        outputs = model_forward_with_forced_gate(
            model=model,
            img=images,
            tac=tactile,
            padding_mask=prefix_mask,
            force_gate=force_gate,
        )
        batch_size = images.size(0)
        total_samples += batch_size
        gate_scores = outputs["gate_score"].detach().cpu().tolist()
        all_gate_scores.extend(gate_scores)
        total_gate += float(np.sum(gate_scores))
        for task in TASKS:
            preds = outputs[task].argmax(dim=1)
            labels = batch[task].to(device)
            correct[task] += int((preds == labels).sum().item())

    task_acc = {task: correct[task] / max(1, total_samples) for task in TASKS}
    avg_acc = float(np.mean(list(task_acc.values())))
    return {
        "num_samples": total_samples,
        "avg_gate_score": total_gate / max(1, total_samples),
        "summary": {
            "average_accuracy": avg_acc,
        },
        "tasks": {task: {"accuracy": float(task_acc[task])} for task in TASKS},
        "gate_scores": all_gate_scores,
    }


def load_model_and_dataset(
    checkpoint_path: Path,
    data_root: Path,
    split_name: str,
    device: torch.device,
    batch_size: int,
    max_tactile_len: int,
    num_workers: int,
    cli_args: argparse.Namespace,
) -> tuple[nn.Module, IndexedRoboticGraspDataset, DataLoader]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    split_dir = data_root / split_name
    dataset = IndexedRoboticGraspDataset(split_dir=split_dir, max_tactile_len=max_tactile_len)
    loader = build_loader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)

    runtime_args = argparse.Namespace(
        fusion_dim=cli_args.fusion_dim,
        num_heads=cli_args.num_heads,
        dropout=cli_args.dropout,
        num_layers=cli_args.num_layers,
        separate_cls_tokens=cli_args.separate_cls_tokens,
    )
    for key, value in cfg.items():
        setattr(runtime_args, key, value)

    model = build_model(
        cfg=cfg,
        args=runtime_args,
        mass_classes=len(dataset.mass_to_idx),
        stiffness_classes=len(dataset.stiffness_to_idx),
        material_classes=len(dataset.material_to_idx),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, dataset, loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose residual visual contribution in the reliable gating model.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/jiaming/Y3_Project/Plaintextdataset",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/best_model.pth",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="test,ood_test",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_tactile_len", type=int, default=3000)
    parser.add_argument("--prefix_ratio", type=float, default=1.0)
    parser.add_argument("--min_prefix_len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--separate_cls_tokens", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.device)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = [item.strip() for item in args.splits.split(",") if item.strip()]
    result: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "prefix_ratio": args.prefix_ratio,
        "conditions": {},
    }

    for split_name in split_names:
        model, dataset, base_loader = load_model_and_dataset(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            split_name=split_name,
            device=device,
            batch_size=args.batch_size,
            max_tactile_len=args.max_tactile_len,
            num_workers=args.num_workers,
            cli_args=args,
        )
        mismatch_indices = build_global_mismatch_indices(dataset, seed=args.seed)
        mismatch_dataset = VisualMismatchDataset(base_dataset=dataset, mismatch_indices=mismatch_indices)
        mismatch_loader = build_loader(
            dataset=mismatch_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        split_result = {
            "original": evaluate_condition(
                model=model,
                loader=base_loader,
                device=device,
                prefix_ratio=args.prefix_ratio,
                min_prefix_len=args.min_prefix_len,
                force_gate=None,
            ),
            "force_gate_zero": evaluate_condition(
                model=model,
                loader=base_loader,
                device=device,
                prefix_ratio=args.prefix_ratio,
                min_prefix_len=args.min_prefix_len,
                force_gate=0.0,
            ),
            "visual_mismatch": evaluate_condition(
                model=model,
                loader=mismatch_loader,
                device=device,
                prefix_ratio=args.prefix_ratio,
                min_prefix_len=args.min_prefix_len,
                force_gate=None,
            ),
            "num_samples": len(dataset),
            "mismatch_same_index_count": int(sum(int(i == j) for i, j in enumerate(mismatch_indices))),
        }
        result["conditions"][split_name] = split_result

    output_path = output_dir / "visual_residual_diagnostic.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved diagnostic results to {output_path}")


if __name__ == "__main__":
    main()
