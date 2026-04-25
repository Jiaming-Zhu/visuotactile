import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

try:
    from diagnose_visual_residual_contribution import (
        IndexedRoboticGraspDataset,
        build_loader,
        load_model_and_dataset,
        make_prefix_mask,
    )
    from train_fusion_gating_online import TASKS
    from train_fusion_gating2 import resolve_device, set_seed
except ImportError:  # pragma: no cover
    from visuotactile.scripts.diagnose_visual_residual_contribution import (  # type: ignore
        IndexedRoboticGraspDataset,
        build_loader,
        load_model_and_dataset,
        make_prefix_mask,
    )
    from visuotactile.scripts.train_fusion_gating_online import TASKS  # type: ignore
    from visuotactile.scripts.train_fusion_gating2 import resolve_device, set_seed  # type: ignore


TARGETS = ["gate", *TASKS]


class VisTokenGradCAM:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handle = self.model.vis_proj.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output
        self.gradients = None
        if isinstance(output, torch.Tensor) and output.requires_grad:
            output.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def remove(self) -> None:
        self._handle.remove()

    def _target_scalar(
        self,
        outputs: Dict[str, torch.Tensor],
        target_name: str,
        target_index: int | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, float | int]]:
        if target_name == "gate":
            score = outputs["gate_score"][0]
            return score, {
                "target_score": float(score.detach().cpu().item()),
                "target_index": -1,
            }

        logits = outputs[target_name][0]
        probs = F.softmax(logits, dim=0)
        pred_idx = int(torch.argmax(probs).detach().cpu().item())
        chosen_idx = pred_idx if target_index is None else int(target_index)
        score = probs[chosen_idx]
        return score, {
            "pred_index": pred_idx,
            "target_index": chosen_idx,
            "target_score": float(score.detach().cpu().item()),
        }

    def generate(
        self,
        image: torch.Tensor,
        tactile: torch.Tensor,
        prefix_mask: torch.Tensor,
        target_name: str,
    ) -> Tuple[np.ndarray, Dict[str, float | int]]:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(image, tactile, padding_mask=prefix_mask)
        scalar, metadata = self._target_scalar(outputs, target_name)
        scalar.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture vis_proj activations/gradients for saliency.")

        activations = self.activations[0].detach()
        gradients = self.gradients[0].detach()
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=0))
        if float(cam.max().item()) > 0.0:
            cam = cam / cam.max().clamp(min=1e-8)
        return cam.cpu().numpy(), metadata


@torch.no_grad()
def score_records(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    prefix_ratio: float,
    min_prefix_len: int,
) -> List[Dict[str, float | int]]:
    records: List[Dict[str, float | int]] = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        tactile = batch["tactile"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        prefix_mask = make_prefix_mask(
            padding_mask=padding_mask,
            prefix_ratio=prefix_ratio,
            min_prefix_len=min_prefix_len,
        )
        outputs = model(images, tactile, padding_mask=prefix_mask)
        for sample_idx in range(images.size(0)):
            records.append(
                {
                    "dataset_index": int(batch["dataset_index"][sample_idx].item()),
                    "gate_score": float(outputs["gate_score"][sample_idx].detach().cpu().item()),
                }
            )
    return records


def parse_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_ratio_list(raw: str) -> List[float]:
    ratios = [float(item.strip()) for item in raw.split(",") if item.strip()]
    ordered = sorted(set(ratios))
    if 0.0 not in ordered:
        ordered = [0.0] + ordered
    return ordered


def select_top_gate_indices(records: Sequence[Dict[str, float | int]], num_samples: int) -> List[int]:
    ordered = sorted(records, key=lambda item: float(item["gate_score"]), reverse=True)
    return [int(item["dataset_index"]) for item in ordered[:num_samples]]


def patch_slice(image: torch.Tensor, patch_idx: int, grid_size: int) -> Tuple[slice, slice]:
    _, height, width = image.shape
    patch_h = height // grid_size
    patch_w = width // grid_size
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    return (
        slice(row * patch_h, (row + 1) * patch_h),
        slice(col * patch_w, (col + 1) * patch_w),
    )


def apply_patch_mask(
    image: torch.Tensor,
    patch_order: Sequence[int],
    patch_ratio: float,
    mode: str,
    grid_size: int,
) -> torch.Tensor:
    total_patches = grid_size * grid_size
    active_count = int(round(patch_ratio * total_patches))
    active_count = min(max(active_count, 0), total_patches)
    selected = set(int(idx) for idx in patch_order[:active_count])

    if mode == "deletion":
        out = image.clone()
        for patch_idx in selected:
            hs, ws = patch_slice(image, patch_idx, grid_size)
            out[:, hs, ws] = 0.0
        return out

    if mode == "insertion":
        out = torch.zeros_like(image)
        for patch_idx in selected:
            hs, ws = patch_slice(image, patch_idx, grid_size)
            out[:, hs, ws] = image[:, hs, ws]
        return out

    raise ValueError(f"Unsupported mode: {mode}")


@torch.no_grad()
def evaluate_target_scores(
    model: torch.nn.Module,
    image_variants: torch.Tensor,
    tactile: torch.Tensor,
    prefix_mask: torch.Tensor,
    target_name: str,
    target_index: int,
    batch_size: int,
) -> List[float]:
    scores: List[float] = []
    total = image_variants.size(0)
    tactile_batch = tactile.expand(min(batch_size, total), -1, -1)
    prefix_batch = prefix_mask.expand(min(batch_size, total), -1)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        current_images = image_variants[start:end]
        current_count = end - start
        current_tactile = tactile_batch[:current_count]
        current_prefix = prefix_batch[:current_count]
        outputs = model(current_images, current_tactile, padding_mask=current_prefix)
        if target_name == "gate":
            current_scores = outputs["gate_score"].detach().cpu().tolist()
        else:
            probs = F.softmax(outputs[target_name], dim=1)
            current_scores = probs[:, target_index].detach().cpu().tolist()
        scores.extend(float(item) for item in current_scores)
    return scores


def build_plot(
    split_name: str,
    summary: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
    ratios: Sequence[float],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, len(TARGETS), figsize=(4.0 * len(TARGETS), 7.2), constrained_layout=True)

    for col, target_name in enumerate(TARGETS):
        for row, mode in enumerate(["deletion", "insertion"]):
            ax = axes[row, col]
            for order_name, color in [("high", "tab:red"), ("low", "tab:blue"), ("random", "tab:gray")]:
                ax.plot(
                    ratios,
                    summary[target_name][mode][order_name]["normalized_mean"],
                    marker="o",
                    linewidth=2.0,
                    label=order_name,
                    color=color,
                )
            ax.set_title(f"{target_name} / {mode}")
            ax.set_xlabel("Patch Ratio")
            ax.set_ylabel("Normalized Score")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25)
            if row == 0 and col == len(TARGETS) - 1:
                ax.legend(loc="upper right", frameon=False)

    fig.suptitle(f"Causal Visual Saliency ({split_name}, top-gate samples)", fontsize=14)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal patch deletion/insertion validation for reliable-gating visual saliency."
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
    parser.add_argument("--prefix_ratio", type=float, default=1.0)
    parser.add_argument("--min_prefix_len", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=24)
    parser.add_argument("--ratios", type=str, default="0.1,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--variant_batch_size", type=int, default=32)
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
    ratios = parse_ratio_list(args.ratios)

    overall_result: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "prefix_ratio": args.prefix_ratio,
        "num_samples": args.num_samples,
        "ratios": ratios,
        "splits": {},
    }

    split_names = parse_list(args.splits)
    extractor_refs: List[VisTokenGradCAM] = []

    for split_name in split_names:
        print(f"[{split_name}] loading model and scoring gate values...", flush=True)
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
        model.eval()
        records = score_records(
            model=model,
            loader=loader,
            device=device,
            prefix_ratio=args.prefix_ratio,
            min_prefix_len=args.min_prefix_len,
        )
        extractor = VisTokenGradCAM(model)
        extractor_refs.append(extractor)
        selected_indices = select_top_gate_indices(records, min(args.num_samples, len(records)))
        print(
            f"[{split_name}] selected {len(selected_indices)} top-gate samples "
            f"(max gate={max(float(r['gate_score']) for r in records):.4f})",
            flush=True,
        )

        sample_results: List[Dict[str, object]] = []
        summary_store: DefaultDict[str, DefaultDict[str, DefaultDict[str, List[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)

        for sample_rank, dataset_index in enumerate(selected_indices, start=1):
            sample = dataset[int(dataset_index)]
            image = sample["image"].unsqueeze(0).to(device)
            tactile = sample["tactile"].unsqueeze(0).to(device)
            padding_mask = sample["padding_mask"].unsqueeze(0).to(device)
            prefix_mask = make_prefix_mask(
                padding_mask=padding_mask,
                prefix_ratio=args.prefix_ratio,
                min_prefix_len=args.min_prefix_len,
            )

            print(
                f"[{split_name}] sample {sample_rank}/{len(selected_indices)} "
                f"(dataset_index={dataset_index})",
                flush=True,
            )

            sample_result: Dict[str, object] = {
                "dataset_index": int(dataset_index),
                "targets": {},
            }

            random_order = list(range(49))
            random.Random(args.seed + int(dataset_index)).shuffle(random_order)

            for target_name in TARGETS:
                cam_map, metadata = extractor.generate(
                    image=image,
                    tactile=tactile,
                    prefix_mask=prefix_mask,
                    target_name=target_name,
                )
                flat_scores = cam_map.reshape(-1)
                high_order = [int(idx) for idx in np.argsort(-flat_scores)]
                low_order = list(reversed(high_order))
                target_index = int(metadata["target_index"])
                variant_images: List[torch.Tensor] = []
                variant_meta: List[Tuple[str, str, float]] = []

                for mode in ["deletion", "insertion"]:
                    for order_name, order_values in [
                        ("high", high_order),
                        ("low", low_order),
                        ("random", random_order),
                    ]:
                        for ratio in ratios:
                            variant_images.append(
                                apply_patch_mask(
                                    image=image[0],
                                    patch_order=order_values,
                                    patch_ratio=ratio,
                                    mode=mode,
                                    grid_size=7,
                                )
                            )
                            variant_meta.append((mode, order_name, float(ratio)))

                image_variants = torch.stack(variant_images, dim=0)
                variant_scores = evaluate_target_scores(
                    model=model,
                    image_variants=image_variants,
                    tactile=tactile,
                    prefix_mask=prefix_mask,
                    target_name=target_name,
                    target_index=target_index,
                    batch_size=args.variant_batch_size,
                )

                grouped_scores: DefaultDict[str, DefaultDict[str, Dict[str, float]]] = defaultdict(
                    lambda: defaultdict(dict)
                )
                for (mode, order_name, ratio), score in zip(variant_meta, variant_scores):
                    grouped_scores[mode][order_name][f"{ratio:.4f}"] = float(score)

                original_score = float(metadata["target_score"])
                baseline_score = float(grouped_scores["insertion"]["high"]["0.0000"])
                target_result = {
                    "original_score": original_score,
                    "baseline_score": baseline_score,
                    "target_index": target_index,
                    "cam_map": cam_map.tolist(),
                    "scores": grouped_scores,
                }
                sample_result["targets"][target_name] = target_result

                for mode in ["deletion", "insertion"]:
                    for order_name in ["high", "low", "random"]:
                        curve = [float(grouped_scores[mode][order_name][f"{ratio:.4f}"]) for ratio in ratios]
                        if mode == "deletion":
                            denom = max(original_score, 1e-8)
                            normalized_curve = [value / denom for value in curve]
                        else:
                            denom = max(original_score - baseline_score, 1e-8)
                            normalized_curve = [(value - baseline_score) / denom for value in curve]
                        for ratio, value in zip(ratios, normalized_curve):
                            summary_store[target_name][mode][order_name].append(float(value))

            sample_results.append(sample_result)

        summary: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {}
        num_points = len(ratios)
        for target_name in TARGETS:
            summary[target_name] = {}
            for mode in ["deletion", "insertion"]:
                summary[target_name][mode] = {}
                for order_name in ["high", "low", "random"]:
                    raw_values = summary_store[target_name][mode][order_name]
                    matrix = np.asarray(raw_values, dtype=np.float32).reshape(len(selected_indices), num_points)
                    summary[target_name][mode][order_name] = {
                        "normalized_mean": matrix.mean(axis=0).tolist(),
                        "normalized_std": matrix.std(axis=0).tolist(),
                    }

        split_result = {
            "selected_indices": selected_indices,
            "samples": sample_results,
            "summary": summary,
        }
        overall_result["splits"][split_name] = split_result

        split_json = split_output_dir / "causal_saliency_results.json"
        split_json.write_text(json.dumps(split_result, indent=2, ensure_ascii=False))
        build_plot(
            split_name=split_name,
            summary=summary,
            ratios=ratios,
            out_path=split_output_dir / "causal_saliency_curves.png",
        )
        print(f"[{split_name}] saved causal saliency outputs to {split_output_dir}", flush=True)

    for extractor in extractor_refs:
        extractor.remove()

    output_path = output_dir / "causal_saliency_summary.json"
    output_path.write_text(json.dumps(overall_result, indent=2, ensure_ascii=False))
    print(f"Saved causal saliency summary to {output_path}", flush=True)


if __name__ == "__main__":
    main()
