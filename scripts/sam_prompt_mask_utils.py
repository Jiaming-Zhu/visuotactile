from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor


DEFAULT_MODEL_ID = "facebook/sam-vit-base"


@dataclass
class SamPrediction:
    mask_uint8: np.ndarray
    score: float
    selected_index: int


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def load_annotation_file(annotation_path: Path) -> Dict[str, object]:
    return json.loads(annotation_path.read_text())


def annotation_has_prompt(annotation: Dict[str, object]) -> bool:
    bbox = annotation.get("bbox_xyxy")
    return bbox is not None or bool(annotation.get("positive_points")) or bool(annotation.get("negative_points"))


def build_sam_inputs(annotation: Dict[str, object]) -> Dict[str, object]:
    kwargs: Dict[str, object] = {}
    bbox = annotation.get("bbox_xyxy")
    if bbox is not None:
        kwargs["input_boxes"] = [[[list(map(float, bbox))]]]

    positive_points = [[float(x), float(y)] for x, y in annotation.get("positive_points", [])]
    negative_points = [[float(x), float(y)] for x, y in annotation.get("negative_points", [])]
    if positive_points or negative_points:
        all_points = positive_points + negative_points
        all_labels = [1] * len(positive_points) + [0] * len(negative_points)
        kwargs["input_points"] = [[[point for point in all_points]]]
        kwargs["input_labels"] = [[[int(label) for label in all_labels]]]
    return kwargs


def _to_numpy_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().numpy().astype(np.float32)
    return (mask > 0.0).astype(np.uint8)


def predict_mask(
    image: Image.Image,
    annotation: Dict[str, object],
    processor: SamProcessor,
    model: SamModel,
    device: torch.device,
) -> SamPrediction:
    if not annotation_has_prompt(annotation):
        raise ValueError("Annotation has no prompts. Need at least one box or point.")

    sam_kwargs = build_sam_inputs(annotation)
    inputs = processor(
        image,
        return_tensors="pt",
        **sam_kwargs,
    )
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.detach().cpu(),
        inputs["original_sizes"].detach().cpu(),
        inputs["reshaped_input_sizes"].detach().cpu(),
        binarize=False,
    )
    scores = outputs.iou_scores.detach().cpu().numpy()

    mask_candidates = masks[0][0]
    score_candidates = scores[0][0]
    best_idx = int(np.argmax(score_candidates))
    best_mask = _to_numpy_mask(mask_candidates[best_idx])
    return SamPrediction(
        mask_uint8=(best_mask * 255).astype(np.uint8),
        score=float(score_candidates[best_idx]),
        selected_index=best_idx,
    )


def overlay_mask(
    image: Image.Image,
    mask_uint8: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
) -> Image.Image:
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    mask = (mask_uint8 > 0).astype(np.float32)[..., None]
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    blended = rgb * (1.0 - alpha * mask) + color_arr * (alpha * mask)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def save_mask_outputs(
    record: Dict[str, object],
    mask_uint8: np.ndarray,
    overlay_image: Image.Image,
    score: float,
    output_root: Path,
) -> Dict[str, str]:
    masks_dir = output_root / "masks"
    overlays_dir = output_root / "mask_previews"
    metadata_dir = output_root / "mask_metadata"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    record_id = str(record["record_id"])
    mask_path = masks_dir / f"{record_id}.png"
    overlay_path = overlays_dir / f"{record_id}.png"
    metadata_path = metadata_dir / f"{record_id}.json"

    Image.fromarray(mask_uint8).save(mask_path)
    overlay_image.save(overlay_path)
    metadata_path.write_text(
        json.dumps(
            {
                "record_id": record_id,
                "source_image_path": record["source_image_path"],
                "copied_image_path": record["copied_image_path"],
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "score": float(score),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return {
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "metadata_path": str(metadata_path),
    }


_MODEL_CACHE: Dict[Tuple[str, str], Tuple[SamProcessor, SamModel]] = {}


def get_cached_sam(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
) -> Tuple[SamProcessor, SamModel, torch.device]:
    resolved_device = resolve_device(device)
    cache_key = (model_id, str(resolved_device))
    if cache_key not in _MODEL_CACHE:
        processor = SamProcessor.from_pretrained(model_id)
        model = SamModel.from_pretrained(model_id)
        model.to(resolved_device)
        model.eval()
        _MODEL_CACHE[cache_key] = (processor, model)
    processor, model = _MODEL_CACHE[cache_key]
    return processor, model, resolved_device
