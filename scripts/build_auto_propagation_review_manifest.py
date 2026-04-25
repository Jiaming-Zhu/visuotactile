import argparse
import json
from pathlib import Path
from typing import Dict, List

SCRIPT_ROOT = Path(__file__).resolve().parent
import sys

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

from diagnose_visual_residual_contribution import IndexedRoboticGraspDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Streamlit review manifest for all auto-propagated SAM prompts."
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/auto_propagation_summary.json",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/jiaming/Y3_Project/Plaintextdataset",
    )
    parser.add_argument(
        "--output_manifest",
        type=str,
        default="/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13/auto_propagation/review_all_manifest.json",
    )
    return parser.parse_args()


def load_datasets(data_root: Path, splits: List[str]) -> Dict[str, IndexedRoboticGraspDataset]:
    datasets: Dict[str, IndexedRoboticGraspDataset] = {}
    for split in splits:
        datasets[split] = IndexedRoboticGraspDataset(split_dir=data_root / split, max_tactile_len=3000)
    return datasets


def build_record(
    item: Dict[str, object],
    dataset: IndexedRoboticGraspDataset,
    output_root: Path,
) -> Dict[str, object]:
    dataset_index = int(item["dataset_index"])
    sample = dataset.samples[dataset_index]
    source_path = str(Path(item["source_image_path"]).resolve())
    sample_path = str(Path(sample.img_path).resolve())
    if source_path != sample_path:
        raise ValueError(
            f"Source path mismatch for {item['record_id']}: summary={source_path} dataset={sample_path}"
        )

    auto_id = str(item["record_id"])
    return {
        "split": str(item["split"]),
        "selection_group": "auto_propagated_review_all",
        "order_rank": dataset_index,
        "dataset_index": dataset_index,
        "source_image_path": source_path,
        "object_class": str(item["object_class"]),
        "episode_name": str(item["episode_name"]),
        "labels": {
            "mass": int(sample.labels["mass"]),
            "stiffness": int(sample.labels["stiffness"]),
            "material": int(sample.labels["material"]),
        },
        "record_id": auto_id,
        "copied_image_path": source_path,
        "annotation_json_path": str(output_root / "auto_prompt_annotations" / f"{auto_id}.json"),
        "preview_image_path": str(output_root / "review_previews" / f"{auto_id}.png"),
        "review_needed": bool(item.get("review_needed", False)),
        "review_reasons": list(item.get("review_reasons", [])),
        "existing_mask_preview_path": str(output_root / "mask_previews" / f"{auto_id}.png"),
    }


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    output_manifest = Path(args.output_manifest)
    output_root = summary_path.parent
    summary = json.loads(summary_path.read_text())

    generated = list(summary["generated"])
    splits = sorted({str(item["split"]) for item in generated})
    datasets = load_datasets(data_root=Path(args.data_root), splits=splits)

    records = [
        build_record(item=item, dataset=datasets[str(item["split"])], output_root=output_root)
        for item in generated
    ]
    records.sort(key=lambda record: (record["split"], int(record["dataset_index"])))

    manifest = {
        "data_root": str(Path(args.data_root)),
        "source_summary_path": str(summary_path),
        "sam_output_root": str(output_root),
        "num_records": len(records),
        "selection_specs": [
            {
                "name": "auto_propagated_review_all",
                "description": "All auto-propagated samples for manual review and correction.",
            }
        ],
        "records": records,
    }

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Saved review manifest to {output_manifest}")
    print(f"Num records: {len(records)}")


if __name__ == "__main__":
    main()
