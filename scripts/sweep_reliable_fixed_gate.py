import argparse
import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    from diagnose_visual_residual_contribution import (
        evaluate_condition,
        load_model_and_dataset,
    )
    from train_fusion_gating2 import resolve_device, set_seed
except ImportError:  # pragma: no cover
    from visuotactile.scripts.diagnose_visual_residual_contribution import (  # type: ignore
        evaluate_condition,
        load_model_and_dataset,
    )
    from visuotactile.scripts.train_fusion_gating2 import resolve_device, set_seed  # type: ignore


GateValue = Union[str, float]


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_gate_values(raw: str) -> List[GateValue]:
    values: List[GateValue] = []
    for item in parse_csv_list(raw):
        if item.lower() == "learned":
            values.append("learned")
        else:
            values.append(float(item))
    if not values:
        raise ValueError("at least one gate value is required")
    return values


def parse_prefix_ratios(raw: str) -> List[float]:
    ratios = [float(item) for item in parse_csv_list(raw)]
    for ratio in ratios:
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"prefix ratio must be in (0, 1], got {ratio}")
    if not ratios:
        raise ValueError("at least one prefix ratio is required")
    return ratios


def gate_label(value: GateValue) -> str:
    if value == "learned":
        return "learned"
    return f"{float(value):.2f}"


def maybe_plot_results(result: dict, output_dir: Path) -> None:
    if plt is None:
        return

    for split_name, split_data in result["splits"].items():
        for prefix_entry in split_data["prefix_results"]:
            prefix_ratio = prefix_entry["prefix_ratio"]
            labels = [entry["gate_label"] for entry in prefix_entry["gate_sweep"]]
            avg_acc = [entry["summary"]["average_accuracy"] for entry in prefix_entry["gate_sweep"]]
            mass = [entry["tasks"]["mass"]["accuracy"] for entry in prefix_entry["gate_sweep"]]
            stiff = [entry["tasks"]["stiffness"]["accuracy"] for entry in prefix_entry["gate_sweep"]]
            material = [entry["tasks"]["material"]["accuracy"] for entry in prefix_entry["gate_sweep"]]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(labels, avg_acc, marker="o", label="Average")
            ax.plot(labels, mass, marker="o", label="Mass")
            ax.plot(labels, stiff, marker="o", label="Stiffness")
            ax.plot(labels, material, marker="o", label="Material")
            ax.set_title(f"{split_name} | prefix={prefix_ratio}")
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("Forced gate")
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"{split_name}_prefix_{prefix_ratio:.2f}_fixed_gate_sweep.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep fixed gate values on a reliable gating checkpoint."
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412/fixed_gate_sweep",
    )
    parser.add_argument("--splits", type=str, default="test,ood_test")
    parser.add_argument("--prefix_ratios", type=str, default="0.1,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument(
        "--gate_values",
        type=str,
        default="learned,0.00,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.80,1.00",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_tactile_len", type=int, default=3000)
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = parse_csv_list(args.splits)
    prefix_ratios = parse_prefix_ratios(args.prefix_ratios)
    gate_values = parse_gate_values(args.gate_values)

    result = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "prefix_ratios": prefix_ratios,
        "gate_values": [gate_label(value) for value in gate_values],
        "splits": {},
    }

    for split_name in split_names:
        model, dataset, loader = load_model_and_dataset(
            checkpoint_path=Path(args.checkpoint),
            data_root=Path(args.data_root),
            split_name=split_name,
            device=device,
            batch_size=args.batch_size,
            max_tactile_len=args.max_tactile_len,
            num_workers=args.num_workers,
            cli_args=args,
        )

        split_result = {"num_samples": len(dataset), "prefix_results": []}
        for prefix_ratio in prefix_ratios:
            gate_sweep = []
            for gate_value in gate_values:
                forced_gate: Optional[float]
                if gate_value == "learned":
                    forced_gate = None
                else:
                    forced_gate = float(gate_value)
                metrics = evaluate_condition(
                    model=model,
                    loader=loader,
                    device=device,
                    prefix_ratio=prefix_ratio,
                    min_prefix_len=args.min_prefix_len,
                    force_gate=forced_gate,
                )
                gate_sweep.append(
                    {
                        "gate_label": gate_label(gate_value),
                        "forced_gate": None if forced_gate is None else float(forced_gate),
                        "avg_gate_score": float(metrics["avg_gate_score"]),
                        "summary": metrics["summary"],
                        "tasks": metrics["tasks"],
                    }
                )
                print(
                    f"[{split_name}] prefix={prefix_ratio:.2f} gate={gate_label(gate_value)} "
                    f"avg={metrics['summary']['average_accuracy']:.2%}"
                )

            best_entry = max(gate_sweep, key=lambda item: item["summary"]["average_accuracy"])
            split_result["prefix_results"].append(
                {
                    "prefix_ratio": prefix_ratio,
                    "best_gate_label": best_entry["gate_label"],
                    "best_average_accuracy": best_entry["summary"]["average_accuracy"],
                    "gate_sweep": gate_sweep,
                }
            )
        result["splits"][split_name] = split_result

    (output_dir / "fixed_gate_sweep_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    maybe_plot_results(result, output_dir)
    print(f"Saved fixed-gate sweep to {output_dir / 'fixed_gate_sweep_results.json'}")


if __name__ == "__main__":
    main()
