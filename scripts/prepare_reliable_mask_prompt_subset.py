import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

try:
    from diagnose_visual_residual_contribution import IndexedRoboticGraspDataset
except ImportError:  # pragma: no cover
    from visuotactile.scripts.diagnose_visual_residual_contribution import IndexedRoboticGraspDataset  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a compact image subset for manual box/point annotation on reliable-gating samples."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/jiaming/Y3_Project/Plaintextdataset",
    )
    parser.add_argument(
        "--reliable_dir",
        type=str,
        default="/home/jiaming/Y3_Project/visuotactile/outputs/fusion_gating_online_reliable_formal_20260412",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/jiaming/Y3_Project/visuotactile/outputs/mask_prompt_annotation_2026-04-13",
    )
    parser.add_argument("--ood_top_n", type=int, default=20)
    parser.add_argument("--ood_random_n", type=int, default=10)
    parser.add_argument("--test_top_n", type=int, default=5)
    parser.add_argument("--test_random_n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_top_indices(reliable_dir: Path, split_name: str) -> List[int]:
    candidate = reliable_dir / "causal_saliency_validation" / split_name / "causal_saliency_results.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Missing top-gate source file: {candidate}")
    payload = json.loads(candidate.read_text())
    indices = [int(item) for item in payload["selected_indices"]]
    if not indices:
        raise ValueError(f"No selected_indices found in {candidate}")
    return indices


def build_record(
    dataset: IndexedRoboticGraspDataset,
    split_name: str,
    selection_group: str,
    order_rank: int,
    dataset_index: int,
) -> Dict[str, object]:
    sample = dataset.samples[dataset_index]
    return {
        "split": split_name,
        "selection_group": selection_group,
        "order_rank": int(order_rank),
        "dataset_index": int(dataset_index),
        "source_image_path": str(sample.img_path),
        "object_class": str(sample.img_path.parent.parent.name),
        "episode_name": str(sample.img_path.parent.name),
        "labels": {
            "mass": int(sample.labels["mass"]),
            "stiffness": int(sample.labels["stiffness"]),
            "material": int(sample.labels["material"]),
        },
    }


def pick_random_indices(
    dataset_size: int,
    excluded: List[int],
    count: int,
    seed: int,
) -> List[int]:
    if count <= 0:
        return []
    candidates = [idx for idx in range(dataset_size) if idx not in set(excluded)]
    if count > len(candidates):
        raise ValueError(f"Requested {count} random indices, but only {len(candidates)} candidates remain.")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def copy_image(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    reliable_dir = Path(args.reliable_dir)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    previews_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "test": IndexedRoboticGraspDataset(split_dir=data_root / "test", max_tactile_len=3000),
        "ood_test": IndexedRoboticGraspDataset(split_dir=data_root / "ood_test", max_tactile_len=3000),
    }

    selection_specs = [
        ("ood_test", "top_gate", args.ood_top_n),
        ("ood_test", "random", args.ood_random_n),
        ("test", "top_gate", args.test_top_n),
        ("test", "random", args.test_random_n),
    ]

    top_gate_indices = {
        split_name: load_top_indices(reliable_dir=reliable_dir, split_name=split_name)
        for split_name in ("test", "ood_test")
    }

    selected_records: List[Dict[str, object]] = []
    random_indices = {
        "ood_test": pick_random_indices(
            dataset_size=len(datasets["ood_test"]),
            excluded=top_gate_indices["ood_test"],
            count=args.ood_random_n,
            seed=args.seed + 1,
        ),
        "test": pick_random_indices(
            dataset_size=len(datasets["test"]),
            excluded=top_gate_indices["test"],
            count=args.test_random_n,
            seed=args.seed + 2,
        ),
    }

    global_rank = 0
    for split_name, selection_group, count in selection_specs:
        if count <= 0:
            continue
        if selection_group == "top_gate":
            indices = top_gate_indices[split_name][:count]
        else:
            indices = random_indices[split_name][:count]

        for order_rank, dataset_index in enumerate(indices, start=1):
            global_rank += 1
            record = build_record(
                dataset=datasets[split_name],
                split_name=split_name,
                selection_group=selection_group,
                order_rank=order_rank,
                dataset_index=dataset_index,
            )
            src_path = Path(record["source_image_path"])
            dst_name = (
                f"{global_rank:03d}_{split_name}_{selection_group}_idx{dataset_index}_"
                f"{record['object_class']}_{record['episode_name']}{src_path.suffix.lower()}"
            )
            dst_path = images_dir / dst_name
            copy_image(src_path=src_path, dst_path=dst_path)
            record["record_id"] = f"{global_rank:03d}"
            record["copied_image_path"] = str(dst_path)
            record["annotation_json_path"] = str(annotations_dir / f"{global_rank:03d}.json")
            record["preview_image_path"] = str(previews_dir / f"{global_rank:03d}.png")
            selected_records.append(record)

    manifest = {
        "data_root": str(data_root),
        "reliable_dir": str(reliable_dir),
        "num_records": len(selected_records),
        "selection_specs": {
            "ood_test_top_n": args.ood_top_n,
            "ood_test_random_n": args.ood_random_n,
            "test_top_n": args.test_top_n,
            "test_random_n": args.test_random_n,
        },
        "records": selected_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    summary_lines = [
        f"Prepared {len(selected_records)} images for annotation.",
        f"Manifest: {manifest_path}",
        f"Images directory: {images_dir}",
        f"Annotations directory: {annotations_dir}",
    ]
    (output_dir / "README.txt").write_text("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
