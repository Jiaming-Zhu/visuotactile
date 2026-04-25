import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

try:
    from diagnose_visual_residual_contribution import (
        VisualMismatchDataset,
        build_global_mismatch_indices,
        build_loader,
        evaluate_condition,
        load_model_and_dataset,
    )
    from train_fusion_gating2 import resolve_device, set_seed
except ImportError:  # pragma: no cover
    from visuotactile.scripts.diagnose_visual_residual_contribution import (  # type: ignore
        VisualMismatchDataset,
        build_global_mismatch_indices,
        build_loader,
        evaluate_condition,
        load_model_and_dataset,
    )
    from visuotactile.scripts.train_fusion_gating2 import resolve_device, set_seed  # type: ignore


DEFAULT_PREFIX_RATIOS = "0.1,0.2,0.4,0.6,0.8,1.0"


def parse_ratio_list(raw: str) -> List[float]:
    ratios = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not ratios:
        raise ValueError("At least one prefix ratio is required.")
    for ratio in ratios:
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"prefix ratio must be in (0, 1], got {ratio}")
    return ratios


def summarize_gap(
    original_avg: float,
    gate_zero_avg: float,
    mismatch_avg: float,
) -> Dict[str, float]:
    return {
        "delta_vs_gate_zero": float(original_avg - gate_zero_avg),
        "delta_vs_mismatch": float(original_avg - mismatch_avg),
        "mismatch_vs_gate_zero": float(mismatch_avg - gate_zero_avg),
    }


def scan_prefixes_for_split(
    model: nn.Module,
    base_loader,
    mismatch_loader,
    device: torch.device,
    prefix_ratios: List[float],
    min_prefix_len: int,
) -> List[Dict[str, object]]:
    curves: List[Dict[str, object]] = []
    for ratio in prefix_ratios:
        original = evaluate_condition(
            model=model,
            loader=base_loader,
            device=device,
            prefix_ratio=ratio,
            min_prefix_len=min_prefix_len,
            force_gate=None,
        )
        force_gate_zero = evaluate_condition(
            model=model,
            loader=base_loader,
            device=device,
            prefix_ratio=ratio,
            min_prefix_len=min_prefix_len,
            force_gate=0.0,
        )
        visual_mismatch = evaluate_condition(
            model=model,
            loader=mismatch_loader,
            device=device,
            prefix_ratio=ratio,
            min_prefix_len=min_prefix_len,
            force_gate=None,
        )
        curves.append(
            {
                "prefix_ratio": ratio,
                "original": original,
                "force_gate_zero": force_gate_zero,
                "visual_mismatch": visual_mismatch,
                "summary_gap": summarize_gap(
                    original_avg=float(original["summary"]["average_accuracy"]),
                    gate_zero_avg=float(force_gate_zero["summary"]["average_accuracy"]),
                    mismatch_avg=float(visual_mismatch["summary"]["average_accuracy"]),
                ),
            }
        )
    return curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan prefix-wise residual visual contribution for reliable gating checkpoints."
    )
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
    parser.add_argument("--splits", type=str, default="test,ood_test")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_tactile_len", type=int, default=3000)
    parser.add_argument("--prefix_ratios", type=str, default=DEFAULT_PREFIX_RATIOS)
    parser.add_argument("--min_prefix_len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--separate_cls_tokens", action="store_true")
    return parser.parse_args()


def build_compact_summary(result: Dict[str, object]) -> Dict[str, object]:
    compact: Dict[str, object] = {
        "checkpoint": result["checkpoint"],
        "device": result["device"],
        "prefix_ratios": result["prefix_ratios"],
        "splits": {},
    }
    for split_name, split_result in result["splits"].items():
        rows = []
        for curve in split_result["prefix_curves"]:
            rows.append(
                {
                    "prefix_ratio": curve["prefix_ratio"],
                    "original_avg": curve["original"]["summary"]["average_accuracy"],
                    "force_gate_zero_avg": curve["force_gate_zero"]["summary"]["average_accuracy"],
                    "visual_mismatch_avg": curve["visual_mismatch"]["summary"]["average_accuracy"],
                    "original_gate": curve["original"]["avg_gate_score"],
                    "force_gate_zero_gate": curve["force_gate_zero"]["avg_gate_score"],
                    "visual_mismatch_gate": curve["visual_mismatch"]["avg_gate_score"],
                    "delta_vs_gate_zero": curve["summary_gap"]["delta_vs_gate_zero"],
                    "delta_vs_mismatch": curve["summary_gap"]["delta_vs_mismatch"],
                    "mismatch_vs_gate_zero": curve["summary_gap"]["mismatch_vs_gate_zero"],
                }
            )
        compact["splits"][split_name] = rows
    return compact


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.device)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = [item.strip() for item in args.splits.split(",") if item.strip()]
    prefix_ratios = parse_ratio_list(args.prefix_ratios)

    result: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "prefix_ratios": prefix_ratios,
        "splits": {},
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
        prefix_curves = scan_prefixes_for_split(
            model=model,
            base_loader=base_loader,
            mismatch_loader=mismatch_loader,
            device=device,
            prefix_ratios=prefix_ratios,
            min_prefix_len=args.min_prefix_len,
        )
        result["splits"][split_name] = {
            "num_samples": len(dataset),
            "mismatch_same_index_count": int(sum(int(i == j) for i, j in enumerate(mismatch_indices))),
            "prefix_curves": prefix_curves,
        }

    detailed_path = output_dir / "visual_residual_prefix_scan.json"
    summary_path = output_dir / "visual_residual_prefix_scan_summary.json"
    detailed_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    summary_path.write_text(json.dumps(build_compact_summary(result), indent=2, ensure_ascii=False))
    print(f"Saved detailed prefix scan to {detailed_path}")
    print(f"Saved compact prefix scan summary to {summary_path}")


if __name__ == "__main__":
    main()
