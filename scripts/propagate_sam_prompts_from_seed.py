import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

from diagnose_visual_residual_contribution import IndexedRoboticGraspDataset
from sam_prompt_mask_utils import (
    DEFAULT_MODEL_ID,
    get_cached_sam,
    overlay_mask,
    predict_mask,
    save_mask_outputs,
)


@dataclass
class TemplateStats:
    bbox_norm: Tuple[float, float, float, float]
    seed_score_q1: float
    seed_score_q3: float
    seed_score_min: float
    area_low: float
    area_high: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Propagate SAM prompts from a manually labeled seed set to the remaining samples."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/jiaming/Y3_Project/Plaintextdataset",
    )
    parser.add_argument(
        "--seed_manifest",
        type=str,
        default="/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/manifest.json",
    )
    parser.add_argument(
        "--seed_sam_output_dir",
        type=str,
        default="/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/sam_outputs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation",
    )
    parser.add_argument("--splits", type=str, default="test,ood_test")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def quantiles(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float32)
    return float(np.quantile(arr, 0.25)), float(np.quantile(arr, 0.75))


def compute_seed_templates(
    manifest: Dict[str, object],
    seed_sam_output_dir: Path,
) -> Dict[str, TemplateStats]:
    by_split_bbox: Dict[str, List[Tuple[float, float, float, float]]] = {}
    by_split_score: Dict[str, List[float]] = {}
    by_split_area: Dict[str, List[float]] = {}

    for record in manifest["records"]:
        ann_path = Path(record["annotation_json_path"])
        if not ann_path.exists():
            continue
        ann = json.loads(ann_path.read_text())
        bbox = ann.get("bbox_xyxy")
        if bbox is None:
            continue

        image = Image.open(record["copied_image_path"])
        width, height = image.size
        x1, y1, x2, y2 = bbox
        split = str(record["split"])
        by_split_bbox.setdefault(split, []).append((x1 / width, y1 / height, x2 / width, y2 / height))

        metadata_path = seed_sam_output_dir / "mask_metadata" / f"{record['record_id']}.json"
        mask_path = seed_sam_output_dir / "masks" / f"{record['record_id']}.png"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            by_split_score.setdefault(split, []).append(float(metadata["score"]))
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
            by_split_area.setdefault(split, []).append(float((mask > 0).mean()))

    templates: Dict[str, TemplateStats] = {}
    for split, bbox_rows in by_split_bbox.items():
        cols = list(zip(*bbox_rows))
        bbox_norm = tuple(float(statistics.median(col)) for col in cols)

        score_values = by_split_score.get(split, [])
        area_values = by_split_area.get(split, [])
        score_q1, score_q3 = quantiles(score_values) if score_values else (0.97, 1.0)
        area_q1, area_q3 = quantiles(area_values) if area_values else (0.05, 0.8)
        area_iqr = area_q3 - area_q1
        area_low = max(0.0, area_q1 - 1.5 * area_iqr)
        area_high = min(1.0, area_q3 + 1.5 * area_iqr)

        templates[split] = TemplateStats(
            bbox_norm=bbox_norm,
            seed_score_q1=score_q1,
            seed_score_q3=score_q3,
            seed_score_min=min(score_values) if score_values else 0.97,
            area_low=area_low,
            area_high=area_high,
        )
    return templates


def template_bbox_for_image(template: TemplateStats, width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = template.bbox_norm
    return [
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    ]


def seed_image_paths(manifest: Dict[str, object]) -> set[str]:
    paths = set()
    for record in manifest["records"]:
        if Path(record["annotation_json_path"]).exists():
            paths.add(str(Path(record["source_image_path"]).resolve()))
    return paths


def build_auto_record(sample, split: str, idx: int, output_dir: Path) -> Dict[str, object]:
    auto_id = f"{split}_{idx:04d}"
    annotation_path = output_dir / "auto_prompt_annotations" / f"{auto_id}.json"
    preview_path = output_dir / "auto_prompt_previews" / f"{auto_id}.png"
    return {
        "record_id": auto_id,
        "split": split,
        "selection_group": "auto_propagated",
        "dataset_index": idx,
        "source_image_path": str(sample.img_path),
        "copied_image_path": str(sample.img_path),
        "annotation_json_path": str(annotation_path),
        "preview_image_path": str(preview_path),
        "object_class": str(sample.img_path.parent.parent.name),
        "episode_name": str(sample.img_path.parent.name),
        "labels": {
            "mass": int(sample.labels["mass"]),
            "stiffness": int(sample.labels["stiffness"]),
            "material": int(sample.labels["material"]),
        },
    }


def touch_border(mask_uint8: np.ndarray) -> bool:
    mask = mask_uint8 > 0
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def review_reasons(score: float, area_ratio: float, touches_border: bool, template: TemplateStats) -> List[str]:
    reasons: List[str] = []
    score_threshold = min(template.seed_score_q1, max(0.96, template.seed_score_min - 0.01))
    if score < score_threshold:
        reasons.append("low_score")
    if area_ratio < template.area_low:
        reasons.append("mask_too_small")
    if area_ratio > template.area_high:
        reasons.append("mask_too_large")
    if touches_border:
        reasons.append("touches_border")
    return reasons


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    seed_manifest_path = Path(args.seed_manifest)
    seed_sam_output_dir = Path(args.seed_sam_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(seed_manifest_path.read_text())
    templates = compute_seed_templates(manifest=manifest, seed_sam_output_dir=seed_sam_output_dir)
    seed_paths = seed_image_paths(manifest)

    processor, model, device = get_cached_sam(model_id=args.model_id, device=args.device)
    print(f"Using SAM model `{args.model_id}` on device `{device}`")

    generated = []
    review = []
    split_names = parse_list(args.splits)
    for split in split_names:
        if split not in templates:
            print(f"Skipping split `{split}` because no seed template is available.")
            continue
        template = templates[split]
        dataset = IndexedRoboticGraspDataset(split_dir=data_root / split, max_tactile_len=3000)
        split_total = 0
        split_generated = 0
        for idx, sample in enumerate(dataset.samples):
            split_total += 1
            source_path = str(sample.img_path.resolve())
            if source_path in seed_paths:
                continue

            auto_record = build_auto_record(sample=sample, split=split, idx=idx, output_dir=output_dir)
            annotation_path = Path(auto_record["annotation_json_path"])
            mask_path = output_dir / "masks" / f"{auto_record['record_id']}.png"
            if mask_path.exists() and not args.force:
                continue

            image = Image.open(sample.img_path).convert("RGB")
            width, height = image.size
            bbox = template_bbox_for_image(template=template, width=width, height=height)
            annotation = {
                "record_id": auto_record["record_id"],
                "bbox_xyxy": bbox,
                "positive_points": [],
                "negative_points": [],
            }
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.write_text(json.dumps(annotation, indent=2, ensure_ascii=False))

            prediction = predict_mask(
                image=image,
                annotation=annotation,
                processor=processor,
                model=model,
                device=device,
            )
            overlay = overlay_mask(image=image, mask_uint8=prediction.mask_uint8)
            saved = save_mask_outputs(
                record=auto_record,
                mask_uint8=prediction.mask_uint8,
                overlay_image=overlay,
                score=prediction.score,
                output_root=output_dir,
            )

            area_ratio = float((prediction.mask_uint8 > 0).mean())
            touches = touch_border(prediction.mask_uint8)
            reasons = review_reasons(
                score=prediction.score,
                area_ratio=area_ratio,
                touches_border=touches,
                template=template,
            )

            row = {
                "record_id": auto_record["record_id"],
                "split": split,
                "dataset_index": idx,
                "source_image_path": str(sample.img_path),
                "object_class": auto_record["object_class"],
                "episode_name": auto_record["episode_name"],
                "bbox_xyxy": bbox,
                "score": float(prediction.score),
                "mask_area_ratio": area_ratio,
                "touches_border": touches,
                "review_needed": bool(reasons),
                "review_reasons": reasons,
                **saved,
            }
            generated.append(row)
            if reasons:
                review.append(row)
            split_generated += 1
            if split_generated == 1 or split_generated % 20 == 0:
                print(f"[{split}] generated {split_generated} propagated masks")

        print(f"[{split}] finished propagated masks for remaining samples")

    summary = {
        "seed_manifest": str(seed_manifest_path),
        "seed_sam_output_dir": str(seed_sam_output_dir),
        "output_dir": str(output_dir),
        "model_id": args.model_id,
        "device": str(device),
        "templates": {
            split: {
                "bbox_norm": list(template.bbox_norm),
                "seed_score_q1": template.seed_score_q1,
                "seed_score_q3": template.seed_score_q3,
                "seed_score_min": template.seed_score_min,
                "area_low": template.area_low,
                "area_high": template.area_high,
            }
            for split, template in templates.items()
        },
        "generated_count": len(generated),
        "review_needed_count": len(review),
        "generated": generated,
        "review_needed": review,
    }
    summary_path = output_dir / "auto_propagation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved auto propagation summary to {summary_path}")


if __name__ == "__main__":
    main()
