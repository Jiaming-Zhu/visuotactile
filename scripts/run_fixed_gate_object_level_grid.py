from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List

try:
    from evaluate_object_level_ood import evaluate_object_level
except ImportError:  # pragma: no cover
    from visuotactile.scripts.evaluate_object_level_ood import evaluate_object_level  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run object-level evaluation for fixed-gate checkpoints.")
    parser.add_argument("--runs_root", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--gates", type=str, default="0.00,0.01,0.02,0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--eval_split", choices=["val", "test", "ood_test"], default="ood_test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_tactile_len", type=int, default=3000)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--online_min_prefix_ratio", type=float, default=0.2)
    parser.add_argument("--min_prefix_len", type=int, default=64)
    parser.add_argument("--num_resamples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=7)
    return parser.parse_args()


def parse_float_list(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_seed_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def gate_tag(gate: float) -> str:
    return f"{int(round(gate * 100)):03d}"


def main() -> None:
    args = parse_args()
    requested_gates = parse_float_list(args.gates)
    requested_seeds = parse_seed_list(args.seeds)

    for gate in requested_gates:
        tag = gate_tag(gate)
        family_dir = args.runs_root / f"fusion_fixed_gate_g{tag}"
        for seed in requested_seeds:
            run_dir = family_dir / f"fusion_fixed_gate_g{tag}_seed{seed}"
            checkpoint = run_dir / "best_model.pth"
            output_dir = run_dir / f"object_level_{args.eval_split}"
            output_file = output_dir / "object_level_results.json"

            if not run_dir.exists() or not checkpoint.exists():
                print(f"[WARN] Missing run or checkpoint for gate={gate:.2f} seed={seed}: {run_dir}")
                continue

            if output_file.exists():
                print(f"[SKIP] object-level gate={gate:.2f} seed={seed} already complete")
                continue

            eval_args = SimpleNamespace(
                data_root=str(args.data_root),
                checkpoint=str(checkpoint),
                eval_split=args.eval_split,
                output_dir=str(output_dir),
                device=args.device,
                model_family="fusion",
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_tactile_len=args.max_tactile_len,
                image_size=args.image_size,
                fusion_dim=args.fusion_dim,
                num_heads=args.num_heads,
                dropout=args.dropout,
                num_layers=args.num_layers,
                separate_cls_tokens=False,
                fixed_gate_value=None,
                online_min_prefix_ratio=args.online_min_prefix_ratio,
                min_prefix_len=args.min_prefix_len,
                block_modality="none",
                num_resamples=args.num_resamples,
                bootstrap_seed=args.bootstrap_seed,
            )

            print(f">>> object-level gate={gate:.2f} seed={seed}")
            result = evaluate_object_level(eval_args)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"<<< saved {output_file}")


if __name__ == "__main__":
    main()
