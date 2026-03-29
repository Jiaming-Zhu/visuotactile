from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parents[1]
REPO_DIR = THIS_FILE.parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(REPO_DIR.parent))

from train_fusion_gating2 import TACTILE_STATS


DEFAULT_DATA_ROOT = "/home/martina/Y3_Project/Plaintextdataset"
DEFAULT_TARGET_SPECS = (
    "ood_test:Cardbox_hollow_noise",
    "ood_test:YogaBrick_Foil_ANCHOR",
)
DEFAULT_REFERENCE_SPECS = (
    "train:CardboardBox_Hollow",
    "train:CardboardBox_SpongeFilled",
)
DEFAULT_OUTPUT_DIR = (
    "/home/martina/Y3_Project/visuotactile/outputs/analysis"
    "/raw_tactile_probe_cardbox_vs_yogabrick_2026-03-29"
)
DEFAULT_RESAMPLED_LEN = 256
DEFAULT_MAX_TACTILE_LEN = 3000


@dataclass(frozen=True)
class ClassSpec:
    split_name: str
    obj_class: str
    group: str


@dataclass
class EpisodeRecord:
    split_name: str
    obj_class: str
    group: str
    episode_name: str
    episode_path: str
    valid_len: int
    raw_feature: np.ndarray
    summary_feature: np.ndarray


def parse_class_spec(raw: str, group: str) -> ClassSpec:
    if ":" not in raw:
        raise ValueError(f"Invalid class spec '{raw}'. Expected split:class_name")
    split_name, obj_class = raw.split(":", 1)
    split_name = split_name.strip()
    obj_class = obj_class.strip()
    if not split_name or not obj_class:
        raise ValueError(f"Invalid class spec '{raw}'. Expected split:class_name")
    return ClassSpec(split_name=split_name, obj_class=obj_class, group=group)


def normalize_block(arr: np.ndarray, key: str) -> np.ndarray:
    stats = TACTILE_STATS[key]
    return (np.asarray(arr, dtype=np.float32) - stats["mean"]) / (stats["std"] + 1e-8)


def load_normalized_tactile(tactile_path: Path, max_tactile_len: int = DEFAULT_MAX_TACTILE_LEN) -> np.ndarray:
    with tactile_path.open("rb") as f:
        data = pickle.load(f)

    joint_pos = normalize_block(data["joint_position_profile"], "joint_position")
    joint_load = normalize_block(data["joint_load_profile"], "joint_load")
    joint_current = normalize_block(data["joint_current_profile"], "joint_current")
    joint_vel = normalize_block(data["joint_velocity_profile"], "joint_velocity")
    tactile = np.concatenate([joint_pos, joint_load, joint_current, joint_vel], axis=1).T
    valid_len = min(int(tactile.shape[1]), max_tactile_len)
    return np.asarray(tactile[:, :valid_len], dtype=np.float32)


def resample_tactile(tactile: np.ndarray, target_len: int) -> np.ndarray:
    tactile = np.asarray(tactile, dtype=np.float32)
    channels, source_len = tactile.shape
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if source_len == 0:
        raise ValueError("tactile sequence must be non-empty")
    if source_len == target_len:
        return tactile.copy()
    if source_len == 1:
        return np.repeat(tactile, target_len, axis=1)

    src_x = np.linspace(0.0, 1.0, source_len, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    resampled = np.empty((channels, target_len), dtype=np.float32)
    for channel_idx in range(channels):
        resampled[channel_idx] = np.interp(dst_x, src_x, tactile[channel_idx]).astype(np.float32)
    return resampled


def build_summary_features(tactile: np.ndarray) -> np.ndarray:
    tactile = np.asarray(tactile, dtype=np.float32)
    mean = tactile.mean(axis=1)
    std = tactile.std(axis=1)
    minimum = tactile.min(axis=1)
    maximum = tactile.max(axis=1)
    value_range = maximum - minimum
    final_value = tactile[:, -1]
    return np.concatenate([mean, std, minimum, maximum, value_range, final_value], axis=0).astype(np.float32)


def flatten_resampled_trace(tactile: np.ndarray, target_len: int) -> np.ndarray:
    return resample_tactile(tactile, target_len=target_len).reshape(-1).astype(np.float32)


def collect_records(
    data_root: Path,
    class_specs: Sequence[ClassSpec],
    resampled_len: int,
    max_tactile_len: int,
) -> List[EpisodeRecord]:
    records: List[EpisodeRecord] = []
    for spec in class_specs:
        class_dir = data_root / spec.split_name / spec.obj_class
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for episode_dir in sorted(class_dir.iterdir()):
            if not episode_dir.is_dir():
                continue
            tactile_path = episode_dir / "tactile_data.pkl"
            if not tactile_path.exists():
                continue
            tactile = load_normalized_tactile(tactile_path, max_tactile_len=max_tactile_len)
            records.append(
                EpisodeRecord(
                    split_name=spec.split_name,
                    obj_class=spec.obj_class,
                    group=spec.group,
                    episode_name=episode_dir.name,
                    episode_path=str(episode_dir),
                    valid_len=int(tactile.shape[1]),
                    raw_feature=flatten_resampled_trace(tactile, target_len=resampled_len),
                    summary_feature=build_summary_features(tactile),
                )
            )
    if not records:
        raise RuntimeError("No tactile episodes collected for the requested classes.")
    return records


def features_and_labels(
    records: Sequence[EpisodeRecord],
    feature_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.stack([getattr(record, feature_name) for record in records], axis=0).astype(np.float32)
    labels = np.asarray([record.obj_class for record in records], dtype=object)
    return features, labels


def scaled_features(features: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(features)


def evaluate_linear_probe(
    features: np.ndarray,
    labels: Sequence[str],
    random_state: int = 42,
    max_splits: int = 5,
) -> Dict[str, object]:
    labels = np.asarray(labels)
    class_counts = Counter(labels.tolist())
    min_count = min(class_counts.values())
    n_splits = min(max_splits, min_count)
    if n_splits < 2:
        raise ValueError("Need at least two samples per class for linear probe.")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    )
    scores: List[float] = []
    encoded_y = labels
    for train_idx, test_idx in splitter.split(features, encoded_y):
        clf.fit(features[train_idx], encoded_y[train_idx])
        score = float(clf.score(features[test_idx], encoded_y[test_idx]))
        scores.append(score)

    return {
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "fold_accuracies": [float(score) for score in scores],
        "num_samples": int(len(labels)),
        "num_classes": int(len(class_counts)),
        "n_splits": int(n_splits),
        "class_counts": {str(k): int(v) for k, v in sorted(class_counts.items())},
    }


def compute_centroid_distance_matrix(
    features: np.ndarray,
    labels: Sequence[str],
) -> tuple[List[str], np.ndarray]:
    labels = np.asarray(labels)
    class_names = sorted(set(labels.tolist()))
    centroids = []
    for class_name in class_names:
        centroids.append(features[labels == class_name].mean(axis=0))
    centroid_matrix = np.stack(centroids, axis=0)
    diffs = centroid_matrix[:, None, :] - centroid_matrix[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    return class_names, distances.astype(np.float32)


def compute_target_reference_distances(
    features: np.ndarray,
    labels: Sequence[str],
    target_names: Sequence[str],
    reference_names: Sequence[str],
) -> List[Dict[str, object]]:
    labels = np.asarray(labels)
    centroids = {
        class_name: features[labels == class_name].mean(axis=0)
        for class_name in sorted(set(labels.tolist()))
    }
    rows: List[Dict[str, object]] = []
    for target_name in target_names:
        if target_name not in centroids:
            continue
        target_centroid = centroids[target_name]
        best_reference = None
        best_distance = math.inf
        for reference_name in reference_names:
            if reference_name not in centroids:
                continue
            distance = float(np.linalg.norm(target_centroid - centroids[reference_name]))
            rows.append(
                {
                    "target_class": target_name,
                    "reference_class": reference_name,
                    "distance": distance,
                }
            )
            if distance < best_distance:
                best_distance = distance
                best_reference = reference_name
        rows.append(
            {
                "target_class": target_name,
                "reference_class": "__nearest_reference__",
                "distance": float(best_distance),
                "nearest_reference": best_reference,
            }
        )
    return rows


def safe_silhouette(features: np.ndarray, labels: Sequence[str]) -> float | None:
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None
    counts = [int(np.sum(labels == item)) for item in unique_labels]
    if min(counts) < 2:
        return None
    return float(silhouette_score(features, labels))


def safe_davies_bouldin(features: np.ndarray, labels: Sequence[str]) -> float | None:
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return None
    counts = [int(np.sum(labels == item)) for item in unique_labels]
    if min(counts) < 2:
        return None
    return float(davies_bouldin_score(features, labels))


def evaluate_pairwise_probes(
    features: np.ndarray,
    labels: Sequence[str],
    random_state: int,
) -> List[Dict[str, object]]:
    labels = np.asarray(labels)
    class_names = sorted(set(labels.tolist()))
    rows: List[Dict[str, object]] = []
    for idx, left_name in enumerate(class_names):
        for right_name in class_names[idx + 1 :]:
            mask = np.isin(labels, [left_name, right_name])
            result = evaluate_linear_probe(features[mask], labels[mask], random_state=random_state)
            rows.append(
                {
                    "class_a": left_name,
                    "class_b": right_name,
                    "mean_accuracy": result["mean_accuracy"],
                    "std_accuracy": result["std_accuracy"],
                    "n_splits": result["n_splits"],
                    "num_samples": result["num_samples"],
                }
            )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    if fieldnames is not None:
        resolved_fields = list(fieldnames)
    else:
        resolved_fields = []
        for row in rows:
            for key in row.keys():
                if key not in resolved_fields:
                    resolved_fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=resolved_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    value_fmt: str = ".2f",
) -> None:
    fig, ax = plt.subplots(figsize=(1.2 * len(col_labels) + 2.5, 1.0 * len(row_labels) + 2.0))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                format(value, value_fmt),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > np.nanmax(matrix) * 0.65 else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pairwise_rows_to_matrix(rows: Sequence[Dict[str, object]]) -> tuple[List[str], np.ndarray]:
    class_names = sorted({row["class_a"] for row in rows} | {row["class_b"] for row in rows})
    matrix = np.full((len(class_names), len(class_names)), np.nan, dtype=np.float32)
    np.fill_diagonal(matrix, 1.0)
    index = {name: idx for idx, name in enumerate(class_names)}
    for row in rows:
        left_idx = index[str(row["class_a"])]
        right_idx = index[str(row["class_b"])]
        value = float(row["mean_accuracy"])
        matrix[left_idx, right_idx] = value
        matrix[right_idx, left_idx] = value
    return class_names, matrix


def binary_target_probe(
    features: np.ndarray,
    labels: Sequence[str],
    target_names: Sequence[str],
    random_state: int,
) -> Dict[str, object]:
    labels = np.asarray(labels)
    mask = np.isin(labels, list(target_names))
    subset_features = features[mask]
    subset_labels = labels[mask]
    result = evaluate_linear_probe(subset_features, subset_labels, random_state=random_state)
    result["classes"] = list(target_names)
    return result


def build_sample_metadata(records: Sequence[EpisodeRecord]) -> List[Dict[str, object]]:
    return [
        {
            "split": record.split_name,
            "group": record.group,
            "obj_class": record.obj_class,
            "episode_name": record.episode_name,
            "episode_path": record.episode_path,
            "valid_len": record.valid_len,
        }
        for record in records
    ]


def summarize_probe_rows(rows: Sequence[Dict[str, object]]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"- {row['class_a']} vs {row['class_b']}: "
            f"{100.0 * float(row['mean_accuracy']):.2f}% ± {100.0 * float(row['std_accuracy']):.2f}%"
        )
    return "\n".join(lines)


def write_summary(
    output_dir: Path,
    sample_rows: Sequence[Dict[str, object]],
    target_specs: Sequence[ClassSpec],
    reference_specs: Sequence[ClassSpec],
    analyses: Dict[str, Dict[str, object]],
) -> None:
    target_names = [spec.obj_class for spec in target_specs]
    reference_names = [spec.obj_class for spec in reference_specs]
    lines = [
        "# Raw Tactile Separability Summary",
        "",
        "## Dataset slices",
        "",
        f"- Targets: {', '.join(target_names)}",
        f"- References: {', '.join(reference_names)}",
        f"- Total episodes: {len(sample_rows)}",
    ]

    counts = Counter((row["split"], row["obj_class"]) for row in sample_rows)
    for (split_name, obj_class), count in sorted(counts.items()):
        lines.append(f"- {split_name}/{obj_class}: {count}")

    for feature_name, result in analyses.items():
        binary_result = result["binary_probe"]
        multiclass_result = result["multiclass_probe"]
        silhouette = result["silhouette"]
        davies = result["davies_bouldin"]
        target_reference_rows = result["target_reference_rows"]
        nearest_reference = [
            row
            for row in target_reference_rows
            if row["reference_class"] == "__nearest_reference__"
        ]

        lines.extend(
            [
                "",
                f"## {feature_name}",
                "",
                f"- Binary probe ({target_names[0]} vs {target_names[1]}): "
                f"{100.0 * float(binary_result['mean_accuracy']):.2f}% ± "
                f"{100.0 * float(binary_result['std_accuracy']):.2f}% "
                f"(n_splits={binary_result['n_splits']})",
                f"- Four-class probe: "
                f"{100.0 * float(multiclass_result['mean_accuracy']):.2f}% ± "
                f"{100.0 * float(multiclass_result['std_accuracy']):.2f}% "
                f"(n_splits={multiclass_result['n_splits']})",
                f"- Silhouette: {'n/a' if silhouette is None else f'{silhouette:.3f}'}",
                f"- Davies-Bouldin: {'n/a' if davies is None else f'{davies:.3f}'}",
                "",
                "Nearest reference per target centroid:",
            ]
        )
        for row in nearest_reference:
            lines.append(
                f"- {row['target_class']} -> {row['nearest_reference']} "
                f"(distance={float(row['distance']):.3f})"
            )

        lines.extend(
            [
                "",
                "Pairwise linear probe accuracy:",
                summarize_probe_rows(result["pairwise_rows"]),
            ]
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantify raw tactile separability with linear probes and centroid distance analysis.",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-spec",
        action="append",
        dest="target_specs",
        default=None,
        help="Class spec in split:class_name form. Repeat for multiple targets.",
    )
    parser.add_argument(
        "--reference-spec",
        action="append",
        dest="reference_specs",
        default=None,
        help="Reference class spec in split:class_name form. Repeat for multiple references.",
    )
    parser.add_argument("--resampled-len", type=int, default=DEFAULT_RESAMPLED_LEN)
    parser.add_argument("--max-tactile-len", type=int, default=DEFAULT_MAX_TACTILE_LEN)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def run_analysis(args: argparse.Namespace) -> Path:
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_specs = [
        parse_class_spec(raw, group="target")
        for raw in (args.target_specs or list(DEFAULT_TARGET_SPECS))
    ]
    reference_specs = [
        parse_class_spec(raw, group="reference")
        for raw in (args.reference_specs or list(DEFAULT_REFERENCE_SPECS))
    ]
    all_specs = [*target_specs, *reference_specs]

    records = collect_records(
        data_root=data_root,
        class_specs=all_specs,
        resampled_len=args.resampled_len,
        max_tactile_len=args.max_tactile_len,
    )
    sample_rows = build_sample_metadata(records)
    write_csv(output_dir / "sample_metadata.csv", sample_rows)

    analyses: Dict[str, Dict[str, object]] = {}
    target_names = [spec.obj_class for spec in target_specs]
    reference_names = [spec.obj_class for spec in reference_specs]

    feature_configs = {
        "resampled_raw_trace": "raw_feature",
        "summary_stats": "summary_feature",
    }
    for feature_label, feature_attr in feature_configs.items():
        features, labels = features_and_labels(records, feature_attr)
        scaled = scaled_features(features)
        binary_probe = binary_target_probe(
            features,
            labels,
            target_names=target_names,
            random_state=args.random_state,
        )
        multiclass_probe = evaluate_linear_probe(
            features,
            labels,
            random_state=args.random_state,
        )
        pairwise_rows = evaluate_pairwise_probes(
            features,
            labels,
            random_state=args.random_state,
        )
        centroid_names, centroid_matrix = compute_centroid_distance_matrix(scaled, labels)
        target_reference_rows = compute_target_reference_distances(
            scaled,
            labels,
            target_names=target_names,
            reference_names=reference_names,
        )
        silhouette = safe_silhouette(scaled, labels)
        davies = safe_davies_bouldin(scaled, labels)

        write_csv(output_dir / f"{feature_label}_pairwise_probe.csv", pairwise_rows)
        write_csv(
            output_dir / f"{feature_label}_centroid_distance.csv",
            [
                {"row_class": row_name, **{col_name: float(value) for col_name, value in zip(centroid_names, row)}}
                for row_name, row in zip(centroid_names, centroid_matrix)
            ],
        )
        write_csv(output_dir / f"{feature_label}_target_reference_distance.csv", target_reference_rows)
        pairwise_names, pairwise_matrix = pairwise_rows_to_matrix(pairwise_rows)
        plot_heatmap(
            centroid_matrix,
            centroid_names,
            centroid_names,
            title=f"{feature_label} centroid distance",
            output_path=output_dir / f"{feature_label}_centroid_distance.png",
            cmap="magma",
            value_fmt=".2f",
        )
        plot_heatmap(
            pairwise_matrix,
            pairwise_names,
            pairwise_names,
            title=f"{feature_label} pairwise linear probe accuracy",
            output_path=output_dir / f"{feature_label}_pairwise_probe_accuracy.png",
            cmap="viridis",
            value_fmt=".2f",
        )

        analyses[feature_label] = {
            "binary_probe": binary_probe,
            "multiclass_probe": multiclass_probe,
            "pairwise_rows": pairwise_rows,
            "silhouette": silhouette,
            "davies_bouldin": davies,
            "target_reference_rows": target_reference_rows,
        }

    summary_json = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "target_specs": [spec.__dict__ for spec in target_specs],
        "reference_specs": [spec.__dict__ for spec in reference_specs],
        "num_records": len(records),
        "analyses": analyses,
    }
    (output_dir / "meta.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    write_summary(output_dir, sample_rows, target_specs, reference_specs, analyses)
    return output_dir


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_dir = run_analysis(args)
    print(f"Saved raw tactile separability analysis to: {output_dir}")


if __name__ == "__main__":
    main()
