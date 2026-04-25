import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError("analyze_reliable_visual_delta.py requires scikit-learn") from exc

try:
    from diagnose_visual_residual_contribution import (
        TASKS,
        evaluate_condition,
        load_model_and_dataset,
        make_prefix_mask,
        model_forward_with_forced_gate,
    )
    from train_fusion_gating2 import resolve_device, set_seed
except ImportError:  # pragma: no cover
    from visuotactile.scripts.diagnose_visual_residual_contribution import (  # type: ignore
        TASKS,
        evaluate_condition,
        load_model_and_dataset,
        make_prefix_mask,
        model_forward_with_forced_gate,
    )
    from visuotactile.scripts.train_fusion_gating2 import resolve_device, set_seed  # type: ignore


def inverse_label_names(label_to_idx: Dict[str, int]) -> List[str]:
    return [name for name, _ in sorted(label_to_idx.items(), key=lambda item: item[1])]


def compute_task_report(
    labels: List[int],
    preds: List[int],
    label_names: List[str],
) -> Dict[str, object]:
    all_class_labels = list(range(len(label_names)))
    acc = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=all_class_labels,
        average=None,
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=all_class_labels,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=all_class_labels,
        average="weighted",
        zero_division=0,
    )
    report_text = classification_report(
        labels,
        preds,
        labels=all_class_labels,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=all_class_labels)
    return {
        "accuracy": float(acc),
        "macro": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1": float(f1_macro),
        },
        "weighted": {
            "precision": float(precision_weighted),
            "recall": float(recall_weighted),
            "f1": float(f1_weighted),
        },
        "per_class": {
            label_names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(label_names))
        },
        "classification_report": report_text,
        "confusion_matrix": {
            "labels": label_names,
            "matrix": cm.tolist(),
        },
    }


@torch.no_grad()
def evaluate_condition_detailed(
    model,
    dataset,
    loader,
    device: torch.device,
    prefix_ratio: float,
    min_prefix_len: int,
    force_gate: float | None,
) -> Dict[str, object]:
    model.eval()
    all_gate_scores: List[float] = []
    all_preds = {task: [] for task in TASKS}
    all_labels = {task: [] for task in TASKS}
    sample_records: List[Dict[str, object]] = []

    for batch in loader:
        images = batch["image"].to(device)
        tactile = batch["tactile"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        prefix_mask = make_prefix_mask(
            padding_mask=padding_mask,
            prefix_ratio=prefix_ratio,
            min_prefix_len=min_prefix_len,
        )
        outputs = model_forward_with_forced_gate(
            model=model,
            img=images,
            tac=tactile,
            padding_mask=prefix_mask,
            force_gate=force_gate,
        )
        gate_scores = outputs["gate_score"].detach().cpu().tolist()
        batch_indices = batch["dataset_index"].tolist()
        for local_idx, dataset_index in enumerate(batch_indices):
            sample = dataset.samples[int(dataset_index)]
            record = {
                "dataset_index": int(dataset_index),
                "image_path": str(sample.img_path),
                "episode_dir": str(sample.img_path.parent),
                "object_name": sample.img_path.parent.parent.name,
                "gate_score": float(gate_scores[local_idx]),
                "labels": {},
                "preds": {},
                "correct": {},
            }
            for task in TASKS:
                pred = int(outputs[task][local_idx].argmax().item())
                label = int(batch[task][local_idx].item())
                all_preds[task].append(pred)
                all_labels[task].append(label)
                record["labels"][task] = label
                record["preds"][task] = pred
                record["correct"][task] = bool(pred == label)
            sample_records.append(record)
        all_gate_scores.extend(gate_scores)

    label_names = {
        "mass": inverse_label_names(dataset.mass_to_idx),
        "stiffness": inverse_label_names(dataset.stiffness_to_idx),
        "material": inverse_label_names(dataset.material_to_idx),
    }
    task_reports = {
        task: compute_task_report(all_labels[task], all_preds[task], label_names[task]) for task in TASKS
    }
    avg_acc = float(np.mean([task_reports[task]["accuracy"] for task in TASKS]))

    return {
        "num_samples": len(dataset),
        "avg_gate_score": float(np.mean(all_gate_scores)) if all_gate_scores else 0.0,
        "summary": {
            "average_accuracy": avg_acc,
        },
        "tasks": task_reports,
        "gate_scores": all_gate_scores,
        "sample_records": sample_records,
    }


def top_confusion_deltas(
    original_matrix: List[List[int]],
    force_gate_zero_matrix: List[List[int]],
    label_names: List[str],
    top_k: int,
) -> List[Dict[str, object]]:
    rows = []
    for i, source_name in enumerate(label_names):
        for j, target_name in enumerate(label_names):
            if i == j:
                continue
            original_count = int(original_matrix[i][j])
            zero_count = int(force_gate_zero_matrix[i][j])
            delta = zero_count - original_count
            rows.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "original_count": original_count,
                    "force_gate_zero_count": zero_count,
                    "delta": delta,
                }
            )
    rows.sort(key=lambda item: (item["delta"], item["force_gate_zero_count"]), reverse=True)
    return rows[:top_k]


def build_delta_summary(
    original: Dict[str, object],
    force_gate_zero: Dict[str, object],
    top_k_confusions: int,
) -> Dict[str, object]:
    task_accuracy_delta = {}
    class_recall_delta = {}
    class_precision_delta = {}
    top_confusions = {}

    for task in TASKS:
        original_task = original["tasks"][task]
        zero_task = force_gate_zero["tasks"][task]
        task_accuracy_delta[task] = float(
            original_task["accuracy"] - zero_task["accuracy"]
        )
        class_recall_delta[task] = {}
        class_precision_delta[task] = {}
        for class_name, metrics in original_task["per_class"].items():
            class_recall_delta[task][class_name] = float(
                metrics["recall"] - zero_task["per_class"][class_name]["recall"]
            )
            class_precision_delta[task][class_name] = float(
                metrics["precision"] - zero_task["per_class"][class_name]["precision"]
            )
        top_confusions[task] = top_confusion_deltas(
            original_matrix=original_task["confusion_matrix"]["matrix"],
            force_gate_zero_matrix=zero_task["confusion_matrix"]["matrix"],
            label_names=original_task["confusion_matrix"]["labels"],
            top_k=top_k_confusions,
        )

    return {
        "average_accuracy_delta": float(
            original["summary"]["average_accuracy"] - force_gate_zero["summary"]["average_accuracy"]
        ),
        "task_accuracy_delta": task_accuracy_delta,
        "class_recall_delta": class_recall_delta,
        "class_precision_delta": class_precision_delta,
        "top_confusion_pair_deltas": top_confusions,
    }


def build_compact_summary(result: Dict[str, object]) -> Dict[str, object]:
    compact = {
        "checkpoint": result["checkpoint"],
        "checkpoint_epoch": result["checkpoint_epoch"],
        "prefix_ratio": result["prefix_ratio"],
        "splits": {},
    }
    for split_name, split_result in result["splits"].items():
        compact["splits"][split_name] = {
            "original_avg": split_result["original"]["summary"]["average_accuracy"],
            "force_gate_zero_avg": split_result["force_gate_zero"]["summary"]["average_accuracy"],
            "average_accuracy_delta": split_result["delta_summary"]["average_accuracy_delta"],
            "task_accuracy_delta": split_result["delta_summary"]["task_accuracy_delta"],
            "class_recall_delta": split_result["delta_summary"]["class_recall_delta"],
            "top_confusion_pair_deltas": split_result["delta_summary"]["top_confusion_pair_deltas"],
        }
    return compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze original-vs-gate-zero visual contribution for reliable gating.")
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
    parser.add_argument("--prefix_ratio", type=float, default=1.0)
    parser.add_argument("--min_prefix_len", type=int, default=16)
    parser.add_argument("--top_k_confusions", type=int, default=5)
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
        "checkpoint_epoch": None,
        "device": str(device),
        "prefix_ratio": args.prefix_ratio,
        "splits": {},
    }

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    result["checkpoint_epoch"] = checkpoint.get("epoch")

    for split_name in split_names:
        model, dataset, loader = load_model_and_dataset(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            split_name=split_name,
            device=device,
            batch_size=args.batch_size,
            max_tactile_len=args.max_tactile_len,
            num_workers=args.num_workers,
            cli_args=args,
        )
        original = evaluate_condition_detailed(
            model=model,
            dataset=dataset,
            loader=loader,
            device=device,
            prefix_ratio=args.prefix_ratio,
            min_prefix_len=args.min_prefix_len,
            force_gate=None,
        )
        force_gate_zero = evaluate_condition_detailed(
            model=model,
            dataset=dataset,
            loader=loader,
            device=device,
            prefix_ratio=args.prefix_ratio,
            min_prefix_len=args.min_prefix_len,
            force_gate=0.0,
        )
        # Keep a lightweight reference to the already-existing aggregate diagnostic.
        aggregate_reference = {
            "original": evaluate_condition(
                model=model,
                loader=loader,
                device=device,
                prefix_ratio=args.prefix_ratio,
                min_prefix_len=args.min_prefix_len,
                force_gate=None,
            ),
            "force_gate_zero": evaluate_condition(
                model=model,
                loader=loader,
                device=device,
                prefix_ratio=args.prefix_ratio,
                min_prefix_len=args.min_prefix_len,
                force_gate=0.0,
            ),
        }
        delta_summary = build_delta_summary(
            original=original,
            force_gate_zero=force_gate_zero,
            top_k_confusions=args.top_k_confusions,
        )
        result["splits"][split_name] = {
            "original": original,
            "force_gate_zero": force_gate_zero,
            "delta_summary": delta_summary,
            "aggregate_reference": aggregate_reference,
        }

    detailed_path = output_dir / "visual_delta_analysis.json"
    summary_path = output_dir / "visual_delta_analysis_summary.json"
    detailed_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    summary_path.write_text(json.dumps(build_compact_summary(result), indent=2, ensure_ascii=False))
    print(f"Saved detailed analysis to {detailed_path}")
    print(f"Saved compact summary to {summary_path}")


if __name__ == "__main__":
    main()
