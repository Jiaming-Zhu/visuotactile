import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

try:
    from diagnose_visual_residual_contribution import (
        IndexedRoboticGraspDataset,
        evaluate_condition,
        load_model_and_dataset,
    )
    from train_fusion_gating2 import resolve_device, set_seed
except ImportError:  # pragma: no cover
    from visuotactile.scripts.diagnose_visual_residual_contribution import (  # type: ignore
        IndexedRoboticGraspDataset,
        evaluate_condition,
        load_model_and_dataset,
    )
    from visuotactile.scripts.train_fusion_gating2 import resolve_device, set_seed  # type: ignore


def build_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def severity_value(corruption: str, severity: int):
    severity = int(severity)
    if corruption == "color_jitter":
        vals = [0.1, 0.2, 0.3, 0.4, 0.5]
        return {
            "brightness": vals[severity - 1],
            "contrast": vals[severity - 1],
            "saturation": vals[severity - 1],
            "hue": [0.02, 0.04, 0.06, 0.08, 0.10][severity - 1],
        }
    if corruption == "gaussian_blur":
        return [1, 2, 3, 4, 5][severity - 1]
    if corruption == "gaussian_noise":
        return [0.02, 0.04, 0.06, 0.08, 0.10][severity - 1]
    if corruption == "cutout":
        return [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    if corruption == "pixelate":
        return [0.9, 0.7, 0.5, 0.35, 0.25][severity - 1]
    raise ValueError(f"Unsupported corruption: {corruption}")


def clamp_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def apply_color_jitter(img: Image.Image, params: Dict[str, float]) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    brightness = params["brightness"]
    contrast = params["contrast"]
    saturation = params["saturation"]
    hue = params["hue"]

    arr = arr * (1.0 + brightness)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    arr = (arr - mean) * (1.0 + contrast) + mean

    gray = arr.mean(axis=2, keepdims=True)
    arr = gray + (arr - gray) * (1.0 + saturation)

    hsv = Image.fromarray(clamp_uint8(arr)).convert("HSV")
    hsv_arr = np.asarray(hsv).copy()
    hsv_arr[..., 0] = (hsv_arr[..., 0].astype(np.int16) + int(hue * 255)) % 255
    return Image.fromarray(hsv_arr, mode="HSV").convert("RGB")


def apply_gaussian_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_gaussian_noise(img: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img).astype(np.float32) / 255.0
    noisy = arr + rng.normal(0.0, sigma, size=arr.shape)
    return Image.fromarray(clamp_uint8(noisy * 255.0))


def apply_cutout(img: Image.Image, area_ratio: float, seed: int) -> Image.Image:
    rng = random.Random(seed)
    arr = np.asarray(img).copy()
    height, width = arr.shape[:2]
    cut_area = max(1, int(height * width * area_ratio))
    side = max(1, int(math.sqrt(cut_area)))
    side = min(side, height, width)
    x0 = rng.randint(0, max(0, width - side))
    y0 = rng.randint(0, max(0, height - side))
    arr[y0 : y0 + side, x0 : x0 + side] = np.array([124, 116, 104], dtype=np.uint8)
    return Image.fromarray(arr)


def apply_pixelate(img: Image.Image, scale: float) -> Image.Image:
    width, height = img.size
    small_w = max(1, int(width * scale))
    small_h = max(1, int(height * scale))
    return img.resize((small_w, small_h), Image.Resampling.BILINEAR).resize(
        (width, height),
        Image.Resampling.NEAREST,
    )


def corrupt_image(img: Image.Image, corruption: str, severity: int, sample_seed: int) -> Image.Image:
    if corruption == "clean":
        return img
    value = severity_value(corruption, severity)
    if corruption == "color_jitter":
        return apply_color_jitter(img, value)
    if corruption == "gaussian_blur":
        return apply_gaussian_blur(img, value)
    if corruption == "gaussian_noise":
        return apply_gaussian_noise(img, value, seed=sample_seed)
    if corruption == "cutout":
        return apply_cutout(img, value, seed=sample_seed)
    if corruption == "pixelate":
        return apply_pixelate(img, value)
    raise ValueError(f"Unsupported corruption: {corruption}")


class CorruptedImageDataset(Dataset):
    def __init__(
        self,
        base_dataset: IndexedRoboticGraspDataset,
        corruption: str,
        severity: int,
        base_seed: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.corruption = corruption
        self.severity = severity
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset.samples[idx]
        image = Image.open(sample.img_path).convert("RGB")
        image = corrupt_image(
            img=image,
            corruption=self.corruption,
            severity=self.severity,
            sample_seed=self.base_seed + idx,
        )
        image = self.base_dataset.transform(image)
        tactile, padding_mask = self.base_dataset._load_tactile(sample.tactile_path)
        return {
            "image": image,
            "tactile": tactile,
            "padding_mask": padding_mask,
            "mass": torch.tensor(sample.labels["mass"], dtype=torch.long),
            "stiffness": torch.tensor(sample.labels["stiffness"], dtype=torch.long),
            "material": torch.tensor(sample.labels["material"], dtype=torch.long),
            "dataset_index": torch.tensor(idx, dtype=torch.long),
        }


def parse_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reliable gating under visual corruption severity sweeps.")
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
    parser.add_argument(
        "--corruptions",
        type=str,
        default="color_jitter,gaussian_blur,gaussian_noise,cutout,pixelate",
    )
    parser.add_argument("--max_severity", type=int, default=5)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
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

    split_names = parse_list(args.splits)
    corruptions = ["clean"] + parse_list(args.corruptions)
    severities = list(range(1, args.max_severity + 1))
    total_conditions = sum(1 if corruption == "clean" else len(severities) for corruption in corruptions)

    result: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "prefix_ratio": args.prefix_ratio,
        "splits": {},
    }

    for split_name in split_names:
        print(
            f"[{split_name}] starting corruption scan with {total_conditions} conditions",
            flush=True,
        )
        model, base_dataset, _ = load_model_and_dataset(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            split_name=split_name,
            device=device,
            batch_size=args.batch_size,
            max_tactile_len=args.max_tactile_len,
            num_workers=args.num_workers,
            cli_args=args,
        )
        split_rows = []
        clean_original_avg = None
        clean_gate_zero_avg = None
        condition_idx = 0
        for corruption in corruptions:
            active_severities = [1] if corruption == "clean" else severities
            for severity in active_severities:
                condition_idx += 1
                print(
                    f"[{split_name}] running {condition_idx}/{total_conditions}: "
                    f"{corruption}@{severity}",
                    flush=True,
                )
                dataset = CorruptedImageDataset(
                    base_dataset=base_dataset,
                    corruption=corruption,
                    severity=severity,
                    base_seed=args.seed,
                )
                loader = build_loader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
                fusion_metrics = evaluate_condition(
                    model=model,
                    loader=loader,
                    device=device,
                    prefix_ratio=args.prefix_ratio,
                    min_prefix_len=args.min_prefix_len,
                    force_gate=None,
                )
                tactile_ref_metrics = evaluate_condition(
                    model=model,
                    loader=loader,
                    device=device,
                    prefix_ratio=args.prefix_ratio,
                    min_prefix_len=args.min_prefix_len,
                    force_gate=0.0,
                )
                fusion_avg = float(fusion_metrics["summary"]["average_accuracy"])
                tactile_avg = float(tactile_ref_metrics["summary"]["average_accuracy"])
                fusion_gate = float(fusion_metrics["avg_gate_score"])
                if corruption == "clean":
                    clean_original_avg = fusion_avg
                    clean_gate_zero_avg = tactile_avg
                row = {
                    "split": split_name,
                    "corruption": corruption,
                    "severity": severity,
                    "fusion": fusion_metrics,
                    "tactile_ref": tactile_ref_metrics,
                    "delta_vs_tactile": float(fusion_avg - tactile_avg),
                    "delta_vs_clean": None if clean_original_avg is None else float(fusion_avg - clean_original_avg),
                    "gate_delta_vs_clean": None,
                }
                if corruption != "clean" and clean_original_avg is not None:
                    row["delta_vs_clean"] = float(fusion_avg - clean_original_avg)
                if corruption == "clean":
                    row["gate_delta_vs_clean"] = 0.0
                else:
                    clean_gate = split_rows[0]["fusion"]["avg_gate_score"] if split_rows else fusion_gate
                    row["gate_delta_vs_clean"] = float(fusion_gate - clean_gate)
                split_rows.append(row)
                print(
                    f"[{split_name}] {corruption}@{severity}: "
                    f"fusion={fusion_avg:.4f}, tactile_ref={tactile_avg:.4f}, gate={fusion_gate:.4f}",
                    flush=True,
                )
        result["splits"][split_name] = split_rows

    output_path = output_dir / "corruption_scan_results.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved corruption scan results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
