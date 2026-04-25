import argparse
import concurrent.futures
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

try:
    from diagnose_visual_residual_contribution import (
        IndexedRoboticGraspDataset,
        TASKS,
        build_global_mismatch_indices,
        build_loader,
        evaluate_condition,
        load_model_and_dataset,
    )
    from train_fusion_gating2 import resolve_device, set_seed
except ImportError:  # pragma: no cover
    from visuotactile.scripts.diagnose_visual_residual_contribution import (  # type: ignore
        IndexedRoboticGraspDataset,
        TASKS,
        build_global_mismatch_indices,
        build_loader,
        evaluate_condition,
        load_model_and_dataset,
    )
    from visuotactile.scripts.train_fusion_gating2 import resolve_device, set_seed  # type: ignore


IMAGENET_MEAN_RGB = np.array([124, 116, 104], dtype=np.uint8)
PROJECT_ROOT = SCRIPT_ROOT.parent.parent
DATASET_SPLIT_NAMES = {"train", "val", "test", "ood_test"}


@dataclass
class ImageArtifacts:
    original_rgb: np.ndarray
    mask_uint8: np.ndarray
    object_only_rgb: np.ndarray
    background_only_rgb: np.ndarray
    mask_area_ratio: float
    used_fallback: bool


def load_rgb_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def normalize_project_path(path_like: str | Path) -> Path:
    raw_path = Path(str(path_like))
    if raw_path.exists():
        return raw_path.resolve()

    for anchor in ("Plaintextdataset", "visuotactile", "paperWorkSpace"):
        if anchor in raw_path.parts:
            anchor_idx = raw_path.parts.index(anchor)
            candidate = (PROJECT_ROOT / Path(*raw_path.parts[anchor_idx:])).resolve()
            if candidate.exists():
                return candidate

    return raw_path.resolve()


def canonical_dataset_image_key(path_like: str | Path) -> str:
    raw_path = Path(str(path_like))
    candidates = [raw_path]
    normalized_path = normalize_project_path(raw_path)
    if normalized_path != raw_path:
        candidates.insert(0, normalized_path)

    for candidate in candidates:
        for idx, part in enumerate(candidate.parts):
            if part in DATASET_SPLIT_NAMES:
                return "/".join(candidate.parts[idx:])

    return str(normalized_path)


def load_mask_registry(metadata_dirs: List[Path]) -> Dict[str, Path]:
    registry: Dict[str, Path] = {}
    for metadata_dir in metadata_dirs:
        if not metadata_dir.exists():
            continue
        for path in metadata_dir.glob("*.json"):
            obj = json.loads(path.read_text())
            source_image_path = obj.get("source_image_path")
            mask_path = obj.get("mask_path")
            if not source_image_path or not mask_path:
                continue
            registry[canonical_dataset_image_key(source_image_path)] = normalize_project_path(mask_path)
    return registry


def load_saved_mask(mask_path: Path, expected_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert("L"))
    if mask.shape != expected_shape:
        raise ValueError(
            f"Mask shape mismatch for {mask_path}: expected {expected_shape}, got {mask.shape}"
        )
    return (mask > 127).astype(np.uint8)


def default_grabcut_rect(width: int, height: int) -> Tuple[int, int, int, int]:
    x = int(width * 0.10)
    y = int(height * 0.08)
    w = int(width * 0.78)
    h = int(height * 0.84)
    return x, y, w, h


def fallback_center_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    x0 = int(width * 0.15)
    x1 = int(width * 0.85)
    y0 = int(height * 0.10)
    y1 = int(height * 0.92)
    mask[y0:y1, x0:x1] = 1
    return mask


def extract_foreground_mask(
    rgb_image: np.ndarray,
    grabcut_iters: int,
    min_mask_ratio: float,
    max_mask_ratio: float,
) -> Tuple[np.ndarray, bool]:
    height, width = rgb_image.shape[:2]
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    rect = default_grabcut_rect(width, height)
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    used_fallback = False
    try:
        cv2.grabCut(
            bgr,
            mask,
            rect,
            bgd_model,
            fgd_model,
            grabcut_iters,
            cv2.GC_INIT_WITH_RECT,
        )
        foreground = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            1,
            0,
        ).astype(np.uint8)
    except cv2.error:
        foreground = fallback_center_mask(height, width)
        used_fallback = True

    ratio = float(foreground.mean())
    if ratio < min_mask_ratio or ratio > max_mask_ratio:
        foreground = fallback_center_mask(height, width)
        used_fallback = True

    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    if float(foreground.mean()) < min_mask_ratio:
        foreground = fallback_center_mask(height, width)
        used_fallback = True

    return foreground.astype(np.uint8), used_fallback


def build_artifacts_for_image(
    img_path: Path,
    grabcut_iters: int,
    min_mask_ratio: float,
    max_mask_ratio: float,
    sam_mask_registry: Optional[Dict[str, Path]] = None,
    allow_grabcut_fallback: bool = False,
) -> ImageArtifacts:
    original_rgb = load_rgb_image(img_path)
    resolved_path = img_path.resolve()
    registry_key = canonical_dataset_image_key(resolved_path)
    mask_uint8: np.ndarray
    used_fallback = False
    if sam_mask_registry is not None and registry_key in sam_mask_registry:
        mask_uint8 = load_saved_mask(
            mask_path=sam_mask_registry[registry_key],
            expected_shape=original_rgb.shape[:2],
        )
    else:
        if sam_mask_registry is not None and not allow_grabcut_fallback:
            raise FileNotFoundError(f"No reviewed SAM mask found for {resolved_path}")
        mask_uint8, used_fallback = extract_foreground_mask(
            rgb_image=original_rgb,
            grabcut_iters=grabcut_iters,
            min_mask_ratio=min_mask_ratio,
            max_mask_ratio=max_mask_ratio,
        )
    mask_bool = mask_uint8.astype(bool)
    object_only_rgb = np.full_like(original_rgb, IMAGENET_MEAN_RGB)
    object_only_rgb[mask_bool] = original_rgb[mask_bool]
    background_only_bgr = cv2.inpaint(
        cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR),
        (mask_uint8 * 255).astype(np.uint8),
        3,
        cv2.INPAINT_TELEA,
    )
    background_only_rgb = cv2.cvtColor(background_only_bgr, cv2.COLOR_BGR2RGB)
    return ImageArtifacts(
        original_rgb=original_rgb,
        mask_uint8=mask_uint8,
        object_only_rgb=object_only_rgb,
        background_only_rgb=background_only_rgb,
        mask_area_ratio=float(mask_uint8.mean()),
        used_fallback=used_fallback,
    )


class CounterfactualDataset(Dataset):
    def __init__(
        self,
        base_dataset: IndexedRoboticGraspDataset,
        mode: str,
        donor_indices: Optional[List[int]] = None,
        grabcut_iters: int = 5,
        min_mask_ratio: float = 0.03,
        max_mask_ratio: float = 0.90,
        artifact_cache: Optional[Dict[Path, ImageArtifacts]] = None,
        sam_mask_registry: Optional[Dict[Path, Path]] = None,
        allow_grabcut_fallback: bool = False,
    ) -> None:
        if mode not in {"original", "object_only", "background_only", "background_swapped"}:
            raise ValueError(f"Unsupported counterfactual mode: {mode}")
        if mode == "background_swapped" and donor_indices is None:
            raise ValueError("donor_indices are required for background_swapped mode")
        self.base_dataset = base_dataset
        self.mode = mode
        self.donor_indices = donor_indices
        self.grabcut_iters = grabcut_iters
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self._artifact_cache = artifact_cache if artifact_cache is not None else {}
        self.sam_mask_registry = sam_mask_registry
        self.allow_grabcut_fallback = allow_grabcut_fallback

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _get_artifacts(self, img_path: Path) -> ImageArtifacts:
        if img_path not in self._artifact_cache:
            self._artifact_cache[img_path] = build_artifacts_for_image(
                img_path=img_path,
                grabcut_iters=self.grabcut_iters,
                min_mask_ratio=self.min_mask_ratio,
                max_mask_ratio=self.max_mask_ratio,
                sam_mask_registry=self.sam_mask_registry,
                allow_grabcut_fallback=self.allow_grabcut_fallback,
            )
        return self._artifact_cache[img_path]

    def _compose_background_swapped(self, idx: int, source_artifacts: ImageArtifacts) -> np.ndarray:
        assert self.donor_indices is not None
        donor_idx = self.donor_indices[idx]
        donor_sample = self.base_dataset.samples[donor_idx]
        donor_artifacts = self._get_artifacts(donor_sample.img_path)
        mask_bool = source_artifacts.mask_uint8.astype(bool)
        composed = donor_artifacts.background_only_rgb.copy()
        composed[mask_bool] = source_artifacts.original_rgb[mask_bool]
        return composed

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset.samples[idx]
        artifacts = self._get_artifacts(sample.img_path)
        if self.mode == "original":
            rgb = artifacts.original_rgb
        elif self.mode == "object_only":
            rgb = artifacts.object_only_rgb
        elif self.mode == "background_only":
            rgb = artifacts.background_only_rgb
        else:
            rgb = self._compose_background_swapped(idx, artifacts)

        image = self.base_dataset.transform(Image.fromarray(rgb))
        tactile, padding_mask = self.base_dataset._load_tactile(sample.tactile_path)
        donor_index = idx if self.donor_indices is None else int(self.donor_indices[idx])
        return {
            "image": image,
            "tactile": tactile,
            "padding_mask": padding_mask,
            "mass": torch.tensor(sample.labels["mass"], dtype=torch.long),
            "stiffness": torch.tensor(sample.labels["stiffness"], dtype=torch.long),
            "material": torch.tensor(sample.labels["material"], dtype=torch.long),
            "dataset_index": torch.tensor(idx, dtype=torch.long),
            "donor_index": torch.tensor(donor_index, dtype=torch.long),
            "mask_area_ratio": torch.tensor(artifacts.mask_area_ratio, dtype=torch.float32),
            "used_fallback": torch.tensor(1 if artifacts.used_fallback else 0, dtype=torch.long),
        }


def summarize_dataset_counterfactual(
    dataset: CounterfactualDataset,
    donor_indices: Optional[List[int]],
) -> Dict[str, object]:
    mask_area_ratios = []
    fallback_count = 0
    for idx, sample in enumerate(dataset.base_dataset.samples):
        artifacts = dataset._get_artifacts(sample.img_path)
        mask_area_ratios.append(artifacts.mask_area_ratio)
        fallback_count += int(artifacts.used_fallback)
    summary = {
        "num_samples": len(dataset),
        "mask_area_ratio_mean": float(np.mean(mask_area_ratios)) if mask_area_ratios else 0.0,
        "mask_area_ratio_std": float(np.std(mask_area_ratios)) if mask_area_ratios else 0.0,
        "fallback_count": int(fallback_count),
    }
    if donor_indices is not None:
        summary["background_swap_same_index_count"] = int(
            sum(int(i == donor_idx) for i, donor_idx in enumerate(donor_indices))
        )
    return summary


def precompute_artifacts_for_dataset(dataset: CounterfactualDataset) -> None:
    for sample in dataset.base_dataset.samples:
        dataset._get_artifacts(sample.img_path)


def precompute_artifacts_for_dataset_parallel(
    dataset: CounterfactualDataset,
    num_workers: int,
    split_name: str,
) -> None:
    unique_paths = []
    seen = set()
    for sample in dataset.base_dataset.samples:
        if sample.img_path not in seen:
            seen.add(sample.img_path)
            unique_paths.append(sample.img_path)

    total = len(unique_paths)
    if total == 0:
        return

    print(
        f"[{split_name}] precomputing foreground artifacts for {total} images "
        f"with {max(1, num_workers)} CPU workers..."
    )

    if num_workers <= 1:
        for idx, img_path in enumerate(unique_paths, start=1):
            dataset._artifact_cache[img_path] = build_artifacts_for_image(
                img_path=img_path,
                grabcut_iters=dataset.grabcut_iters,
                min_mask_ratio=dataset.min_mask_ratio,
                max_mask_ratio=dataset.max_mask_ratio,
                sam_mask_registry=dataset.sam_mask_registry,
                allow_grabcut_fallback=dataset.allow_grabcut_fallback,
            )
            if idx == total or idx % 10 == 0:
                print(f"[{split_name}] artifact progress: {idx}/{total}")
        return

    def _worker(path: Path) -> Tuple[Path, ImageArtifacts]:
        return (
            path,
            build_artifacts_for_image(
                img_path=path,
                grabcut_iters=dataset.grabcut_iters,
                min_mask_ratio=dataset.min_mask_ratio,
                max_mask_ratio=dataset.max_mask_ratio,
                sam_mask_registry=dataset.sam_mask_registry,
                allow_grabcut_fallback=dataset.allow_grabcut_fallback,
            ),
        )

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker, img_path) for img_path in unique_paths]
        for future in concurrent.futures.as_completed(futures):
            img_path, artifacts = future.result()
            dataset._artifact_cache[img_path] = artifacts
            completed += 1
            if completed == total or completed % 10 == 0:
                print(f"[{split_name}] artifact progress: {completed}/{total}")


def save_preview_grid(
    dataset: CounterfactualDataset,
    donor_indices: List[int],
    output_dir: Path,
    count: int,
) -> None:
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    count = min(count, len(dataset))
    for idx in range(count):
        sample = dataset.base_dataset.samples[idx]
        donor_sample = dataset.base_dataset.samples[donor_indices[idx]]
        source_artifacts = dataset._get_artifacts(sample.img_path)
        donor_artifacts = dataset._get_artifacts(donor_sample.img_path)
        mask_rgb = np.repeat((source_artifacts.mask_uint8 * 255)[:, :, None], 3, axis=2)
        swapped = dataset._compose_background_swapped(idx, source_artifacts)
        panel = np.concatenate(
            [
                source_artifacts.original_rgb,
                mask_rgb,
                source_artifacts.object_only_rgb,
                source_artifacts.background_only_rgb,
                donor_artifacts.background_only_rgb,
                swapped,
            ],
            axis=1,
        )
        Image.fromarray(panel).save(preview_dir / f"{idx:03d}_{sample.img_path.parent.name}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Object/background counterfactual diagnosis for reliable gating."
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grabcut_iters", type=int, default=5)
    parser.add_argument("--min_mask_ratio", type=float, default=0.03)
    parser.add_argument("--max_mask_ratio", type=float, default=0.90)
    parser.add_argument("--save_preview_count", type=int, default=12)
    parser.add_argument("--precompute_workers", type=int, default=8)
    parser.add_argument("--fusion_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--separate_cls_tokens", action="store_true")
    parser.add_argument(
        "--sam_mask_metadata_dirs",
        type=str,
        default=(
            "/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/sam_outputs/mask_metadata,"
            "/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/mask_metadata"
        ),
    )
    parser.add_argument("--allow_grabcut_fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.device)
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_metadata_dirs = [Path(item) for item in parse_csv_list(args.sam_mask_metadata_dirs)]
    sam_mask_registry = load_mask_registry(mask_metadata_dirs)
    print(f"Loaded {len(sam_mask_registry)} reviewed SAM masks from {len(mask_metadata_dirs)} metadata dirs")

    split_names = parse_csv_list(args.splits)
    result: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "prefix_ratio": args.prefix_ratio,
        "sam_mask_metadata_dirs": [str(path) for path in mask_metadata_dirs],
        "reviewed_sam_mask_count": len(sam_mask_registry),
        "allow_grabcut_fallback": bool(args.allow_grabcut_fallback),
        "splits": {},
    }

    for split_name in split_names:
        model, base_dataset, base_loader = load_model_and_dataset(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            split_name=split_name,
            device=device,
            batch_size=args.batch_size,
            max_tactile_len=args.max_tactile_len,
            num_workers=args.num_workers,
            cli_args=args,
        )
        donor_indices = build_global_mismatch_indices(base_dataset, seed=args.seed)
        shared_artifact_cache: Dict[Path, ImageArtifacts] = {}

        object_only_dataset = CounterfactualDataset(
            base_dataset=base_dataset,
            mode="object_only",
            grabcut_iters=args.grabcut_iters,
            min_mask_ratio=args.min_mask_ratio,
            max_mask_ratio=args.max_mask_ratio,
            artifact_cache=shared_artifact_cache,
            sam_mask_registry=sam_mask_registry,
            allow_grabcut_fallback=args.allow_grabcut_fallback,
        )
        background_only_dataset = CounterfactualDataset(
            base_dataset=base_dataset,
            mode="background_only",
            grabcut_iters=args.grabcut_iters,
            min_mask_ratio=args.min_mask_ratio,
            max_mask_ratio=args.max_mask_ratio,
            artifact_cache=shared_artifact_cache,
            sam_mask_registry=sam_mask_registry,
            allow_grabcut_fallback=args.allow_grabcut_fallback,
        )
        background_swapped_dataset = CounterfactualDataset(
            base_dataset=base_dataset,
            mode="background_swapped",
            donor_indices=donor_indices,
            grabcut_iters=args.grabcut_iters,
            min_mask_ratio=args.min_mask_ratio,
            max_mask_ratio=args.max_mask_ratio,
            artifact_cache=shared_artifact_cache,
            sam_mask_registry=sam_mask_registry,
            allow_grabcut_fallback=args.allow_grabcut_fallback,
        )

        precompute_artifacts_for_dataset_parallel(
            dataset=background_swapped_dataset,
            num_workers=args.precompute_workers,
            split_name=split_name,
        )

        object_only_loader = build_loader(
            dataset=object_only_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        background_only_loader = build_loader(
            dataset=background_only_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        background_swapped_loader = build_loader(
            dataset=background_swapped_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        split_output_dir = output_dir / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        save_preview_grid(
            dataset=background_swapped_dataset,
            donor_indices=donor_indices,
            output_dir=split_output_dir,
            count=args.save_preview_count,
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
            "object_only": evaluate_condition(
                model=model,
                loader=object_only_loader,
                device=device,
                prefix_ratio=args.prefix_ratio,
                min_prefix_len=args.min_prefix_len,
                force_gate=None,
            ),
            "background_only": evaluate_condition(
                model=model,
                loader=background_only_loader,
                device=device,
                prefix_ratio=args.prefix_ratio,
                min_prefix_len=args.min_prefix_len,
                force_gate=None,
            ),
            "background_swapped": evaluate_condition(
                model=model,
                loader=background_swapped_loader,
                device=device,
                prefix_ratio=args.prefix_ratio,
                min_prefix_len=args.min_prefix_len,
                force_gate=None,
            ),
            "mask_summary": summarize_dataset_counterfactual(
                dataset=background_swapped_dataset,
                donor_indices=donor_indices,
            ),
        }
        result["splits"][split_name] = split_result

    output_path = output_dir / "object_background_counterfactual.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved counterfactual diagnosis to {output_path}")


if __name__ == "__main__":
    main()
