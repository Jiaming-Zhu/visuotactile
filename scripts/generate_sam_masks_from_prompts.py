import argparse
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image

from sam_prompt_mask_utils import (
    DEFAULT_MODEL_ID,
    annotation_has_prompt,
    get_cached_sam,
    load_annotation_file,
    overlay_mask,
    predict_mask,
    save_mask_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SAM masks from saved manual box/point annotations."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/manifest.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/sam_outputs",
    )
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    manifest = json.loads(manifest_path.read_text())
    records: List[Dict[str, object]] = manifest["records"]

    processor, model, device = get_cached_sam(model_id=args.model_id, device=args.device)
    print(f"Using SAM model `{args.model_id}` on device `{device}`")

    generated = []
    skipped = []
    for idx, record in enumerate(records, start=1):
        annotation_path = Path(record["annotation_json_path"])
        record_id = str(record["record_id"])
        mask_path = output_dir / "masks" / f"{record_id}.png"

        if not annotation_path.exists():
            skipped.append({"record_id": record_id, "reason": "missing_annotation"})
            continue
        if mask_path.exists() and not args.force:
            skipped.append({"record_id": record_id, "reason": "already_exists"})
            continue

        annotation = load_annotation_file(annotation_path)
        if not annotation_has_prompt(annotation):
            skipped.append({"record_id": record_id, "reason": "empty_prompt"})
            continue

        image = Image.open(record["copied_image_path"]).convert("RGB")
        prediction = predict_mask(
            image=image,
            annotation=annotation,
            processor=processor,
            model=model,
            device=device,
        )
        overlay = overlay_mask(image=image, mask_uint8=prediction.mask_uint8)
        saved = save_mask_outputs(
            record=record,
            mask_uint8=prediction.mask_uint8,
            overlay_image=overlay,
            score=prediction.score,
            output_root=output_dir,
        )
        generated.append(
            {
                "record_id": record_id,
                "score": float(prediction.score),
                "selected_index": int(prediction.selected_index),
                **saved,
            }
        )
        print(f"[{idx}/{len(records)}] generated mask for {record_id} score={prediction.score:.4f}")

    summary = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "model_id": args.model_id,
        "device": str(device),
        "generated_count": len(generated),
        "skipped_count": len(skipped),
        "generated": generated,
        "skipped": skipped,
    }
    summary_path = output_dir / "sam_generation_summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved SAM generation summary to {summary_path}")


if __name__ == "__main__":
    main()
