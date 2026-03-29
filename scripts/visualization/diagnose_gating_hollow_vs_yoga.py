from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parents[1]
REPO_DIR = THIS_FILE.parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(REPO_DIR.parent))

from train_fusion_gating_online import build_model, effective_padding_mask  # noqa: E402
from train_fusion_gating2 import resolve_device  # noqa: E402
from visualization.analyze_raw_tactile_separation import (  # noqa: E402
    build_summary_features,
    evaluate_linear_probe,
    flatten_resampled_trace,
)
from visualization.export_gating_sample_predictions import (  # noqa: E402
    AnalysisRoboticGraspDataset,
    iter_run_dirs,
    ordered_label_names,
    resolve_override_props_config,
)


DEFAULT_RUN_ROOT = "/home/martina/Y3_Project/visuotactile/outputs/fusion_gating_online_v2_multiseed"
DEFAULT_DATA_ROOT = "/home/martina/Y3_Project/Plaintextdataset"
DEFAULT_CLASSES = ("Cardbox_hollow_noise", "YogaBrick_Foil_ANCHOR")
DEFAULT_OUTPUT_DIR = (
    "/home/martina/Y3_Project/visuotactile/outputs/fusion_gating_online_v2_multiseed"
    "/diagnosis_hollow_vs_yoga_chain"
)
TASKS = ("mass", "stiffness", "material")


def build_runtime_args(cfg: Dict, cli_args: argparse.Namespace) -> argparse.Namespace:
    runtime = argparse.Namespace()
    runtime.device = cli_args.device
    runtime.batch_size = cli_args.batch_size
    runtime.max_tactile_len = cfg.get("max_tactile_len", cli_args.max_tactile_len)
    runtime.num_workers = cli_args.num_workers
    runtime.online_train_prob = cfg.get("online_train_prob", 0.0)
    runtime.online_min_prefix_ratio = cfg.get("online_min_prefix_ratio", 0.2)
    runtime.min_prefix_len = cfg.get("min_prefix_len", 16)
    runtime.block_modality = "none"
    runtime.fusion_dim = cfg.get("fusion_dim", 256)
    runtime.num_heads = cfg.get("num_heads", 8)
    runtime.dropout = cfg.get("dropout", 0.1)
    runtime.num_layers = cfg.get("num_layers", 4)
    runtime.lambda_reg = cfg.get("lambda_reg", 0.0)
    return runtime


def build_loader(
    split_dir: Path,
    batch_size: int,
    max_tactile_len: int,
    num_workers: int,
    override_props_config: Dict[str, object] | None = None,
) -> DataLoader:
    dataset = AnalysisRoboticGraspDataset(
        split_dir=split_dir,
        max_tactile_len=max_tactile_len,
        override_props_config=override_props_config,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def compute_true_class_margin(logits: np.ndarray, true_idx: int) -> float:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 1:
        raise ValueError("logits must be a 1D array")
    if not (0 <= true_idx < logits.shape[0]):
        raise IndexError(f"true_idx {true_idx} out of bounds for logits of length {logits.shape[0]}")
    true_score = float(logits[true_idx])
    if logits.shape[0] == 1:
        return true_score
    mask = np.ones(logits.shape[0], dtype=bool)
    mask[true_idx] = False
    competitor = float(np.max(logits[mask]))
    return true_score - competitor


def decompose_final_linear_logits(
    hidden: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.asarray(hidden, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    projection = weight @ hidden
    logits = projection + bias
    return projection.astype(np.float32), logits.astype(np.float32)


def leave_one_out_probe_predictions(
    features: np.ndarray,
    labels: Sequence[str],
    sample_rows: Sequence[Dict[str, object]],
) -> tuple[List[Dict[str, object]], float]:
    labels = np.asarray(labels)
    if len(features) != len(labels) or len(labels) != len(sample_rows):
        raise ValueError("features, labels, and sample_rows must have the same length")

    classes = sorted(set(labels.tolist()))
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    )
    loo = LeaveOneOut()
    rows: List[Dict[str, object]] = []
    correct = 0
    for train_idx, test_idx in loo.split(features):
        pipeline.fit(features[train_idx], labels[train_idx])
        pred_label = str(pipeline.predict(features[test_idx])[0])
        proba = pipeline.predict_proba(features[test_idx])[0]
        class_to_prob = {
            class_name: float(prob)
            for class_name, prob in zip(pipeline.classes_, proba)
        }
        row = dict(sample_rows[int(test_idx[0])])
        row["true_label"] = str(labels[test_idx[0]])
        row["pred_label"] = pred_label
        row["correct"] = int(pred_label == labels[test_idx[0]])
        row["pred_confidence"] = float(np.max(proba))
        for class_name in classes:
            row[f"prob_{class_name}"] = class_to_prob.get(class_name, 0.0)
        rows.append(row)
        correct += int(row["correct"])
    accuracy = correct / len(sample_rows)
    rows.sort(key=lambda item: (item["correct"], item["pred_confidence"], item["episode_name"]))
    return rows, float(accuracy)


def build_stage_summary_row(
    seed: int | str,
    stage_name: str,
    probe_result: Dict[str, object],
    pair_accuracy: float,
    stiffness_accuracy: float,
    material_accuracy: float,
) -> Dict[str, object]:
    return {
        "seed": seed,
        "stage": stage_name,
        "probe_mean_accuracy": float(probe_result["mean_accuracy"]),
        "probe_std_accuracy": float(probe_result["std_accuracy"]),
        "probe_n_splits": int(probe_result["n_splits"]),
        "pair_accuracy": float(pair_accuracy),
        "stiffness_accuracy": float(stiffness_accuracy),
        "material_accuracy": float(material_accuracy),
    }


def train_small_mlp_head_predictions(
    features: np.ndarray,
    labels: Sequence[str],
    sample_rows: Sequence[Dict[str, object]],
    random_state: int = 42,
    hidden_dim: int = 32,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_splits: int = 5,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    labels = np.asarray(labels)
    if len(features) != len(labels) or len(labels) != len(sample_rows):
        raise ValueError("features, labels, and sample_rows must have the same length")

    class_names = sorted(set(labels.tolist()))
    if len(class_names) != 2:
        raise ValueError("Small MLP head experiment currently expects exactly two classes.")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = np.asarray([class_to_idx[str(label)] for label in labels], dtype=np.int64)
    class_counts = Counter(y.tolist())
    min_count = min(class_counts.values())
    n_splits = min(max_splits, min_count)
    if n_splits < 2:
        raise ValueError("Need at least two samples per class for cross-validation.")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: List[Dict[str, object]] = []
    fold_accuracies: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(features, y)):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx]).astype(np.float32)
        x_test = scaler.transform(features[test_idx]).astype(np.float32)
        y_train = y[train_idx]
        y_test = y[test_idx]

        torch.manual_seed(random_state + fold_idx)
        model = torch.nn.Sequential(
            torch.nn.Linear(x_train.shape[1], hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, len(class_names)),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        x_train_tensor = torch.from_numpy(x_train)
        y_train_tensor = torch.from_numpy(y_train)
        for _ in range(num_epochs):
            optimizer.zero_grad()
            logits = model(x_train_tensor)
            loss = criterion(logits, y_train_tensor)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(model(torch.from_numpy(x_test)), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        fold_accuracies.append(float(np.mean(preds == y_test)))

        for local_idx, sample_idx in enumerate(test_idx):
            row = dict(sample_rows[int(sample_idx)])
            pred_idx = int(preds[local_idx])
            row["true_label"] = str(labels[sample_idx])
            row["pred_label"] = class_names[pred_idx]
            row["correct"] = int(pred_idx == y[sample_idx])
            row["pred_confidence"] = float(np.max(probs[local_idx]))
            row["fold"] = fold_idx
            for class_name, class_idx in class_to_idx.items():
                row[f"prob_{class_name}"] = float(probs[local_idx, class_idx])
            rows.append(row)

    rows.sort(key=lambda item: (item["correct"], item["pred_confidence"], item["episode_name"]))
    metrics = {
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "fold_accuracies": [float(item) for item in fold_accuracies],
        "n_splits": int(n_splits),
        "num_samples": int(len(labels)),
        "class_counts": {class_names[idx]: int(count) for idx, count in sorted(class_counts.items())},
    }
    return rows, metrics


def train_frozen_task_head_predictions(
    features: np.ndarray,
    labels: Sequence[str],
    sample_rows: Sequence[Dict[str, object]],
    task_name: str,
    random_state: int = 42,
    hidden_dim: int = 32,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_splits: int = 5,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    labels = np.asarray(labels)
    if len(features) != len(labels) or len(labels) != len(sample_rows):
        raise ValueError("features, labels, and sample_rows must have the same length")

    class_names = sorted(set(labels.tolist()))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    y = np.asarray([class_to_idx[str(label)] for label in labels], dtype=np.int64)
    class_counts = Counter(y.tolist())
    min_count = min(class_counts.values())
    n_splits = min(max_splits, min_count)
    if n_splits < 2:
        raise ValueError("Need at least two samples per class for cross-validation.")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: List[Dict[str, object]] = []
    fold_accuracies: List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(features, y)):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx]).astype(np.float32)
        x_test = scaler.transform(features[test_idx]).astype(np.float32)
        y_train = y[train_idx]
        y_test = y[test_idx]

        torch.manual_seed(random_state + fold_idx)
        model = torch.nn.Sequential(
            torch.nn.Linear(x_train.shape[1], hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, len(class_names)),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        x_train_tensor = torch.from_numpy(x_train)
        y_train_tensor = torch.from_numpy(y_train)
        for _ in range(num_epochs):
            optimizer.zero_grad()
            logits = model(x_train_tensor)
            loss = criterion(logits, y_train_tensor)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(model(torch.from_numpy(x_test)), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        fold_accuracies.append(float(np.mean(preds == y_test)))

        for local_idx, sample_idx in enumerate(test_idx):
            row = dict(sample_rows[int(sample_idx)])
            pred_idx = int(preds[local_idx])
            row["task"] = task_name
            row["true_label"] = str(labels[sample_idx])
            row["pred_label"] = class_names[pred_idx]
            row["correct"] = int(pred_idx == y[sample_idx])
            row["pred_confidence"] = float(np.max(probs[local_idx]))
            row["fold"] = fold_idx
            for class_name, class_idx in class_to_idx.items():
                row[f"prob_{class_name}"] = float(probs[local_idx, class_idx])
            rows.append(row)

    rows.sort(key=lambda item: (item["correct"], item["pred_confidence"], item["episode_name"]))
    metrics = {
        "task": task_name,
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "fold_accuracies": [float(item) for item in fold_accuracies],
        "n_splits": int(n_splits),
        "num_samples": int(len(labels)),
        "class_counts": {class_names[idx]: int(count) for idx, count in sorted(class_counts.items())},
    }
    return rows, metrics


def train_frozen_task_head_train_eval_split(
    train_features: np.ndarray,
    train_labels: Sequence[str],
    eval_features: np.ndarray,
    eval_labels: Sequence[str],
    eval_rows: Sequence[Dict[str, object]],
    task_name: str,
    random_state: int = 42,
    hidden_dim: int = 32,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    train_labels = np.asarray(train_labels)
    eval_labels = np.asarray(eval_labels)
    if len(train_features) != len(train_labels):
        raise ValueError("train_features and train_labels must have the same length")
    if len(eval_features) != len(eval_labels) or len(eval_labels) != len(eval_rows):
        raise ValueError("eval_features, eval_labels, and eval_rows must have the same length")

    class_names = sorted(set(train_labels.tolist()))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    missing_eval = sorted(set(str(label) for label in eval_labels.tolist()) - set(class_to_idx))
    if missing_eval:
        raise ValueError(f"Eval labels missing from training labels for task {task_name}: {missing_eval}")

    y_train = np.asarray([class_to_idx[str(label)] for label in train_labels], dtype=np.int64)
    y_eval = np.asarray([class_to_idx[str(label)] for label in eval_labels], dtype=np.int64)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features).astype(np.float32)
    x_eval = scaler.transform(eval_features).astype(np.float32)

    torch.manual_seed(random_state)
    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, len(class_names)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)
    for _ in range(num_epochs):
        optimizer.zero_grad()
        logits = model(x_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x_eval)), dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)

    rows: List[Dict[str, object]] = []
    for idx, sample_row in enumerate(eval_rows):
        pred_idx = int(preds[idx])
        row = dict(sample_row)
        row["task"] = task_name
        row["true_label"] = str(eval_labels[idx])
        row["pred_label"] = class_names[pred_idx]
        row["correct"] = int(pred_idx == y_eval[idx])
        row["pred_confidence"] = float(np.max(probs[idx]))
        for class_name, class_idx in class_to_idx.items():
            row[f"prob_{class_name}"] = float(probs[idx, class_idx])
        rows.append(row)

    rows.sort(key=lambda item: (item["correct"], item["pred_confidence"], item["episode_name"]))
    metrics = {
        "task": task_name,
        "eval_accuracy": float(np.mean(preds == y_eval)),
        "num_train_samples": int(len(train_labels)),
        "num_eval_samples": int(len(eval_labels)),
        "num_classes": int(len(class_names)),
    }
    return rows, metrics


def train_frozen_task_head_train_eval_split(
    train_features: np.ndarray,
    train_labels: Sequence[str],
    eval_features: np.ndarray,
    eval_labels: Sequence[str],
    eval_rows: Sequence[Dict[str, object]],
    task_name: str,
    random_state: int = 42,
    hidden_dim: int = 32,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    train_labels = np.asarray(train_labels)
    eval_labels = np.asarray(eval_labels)
    if len(train_features) != len(train_labels):
        raise ValueError("train_features and train_labels must have the same length")
    if len(eval_features) != len(eval_labels) or len(eval_labels) != len(eval_rows):
        raise ValueError("eval_features, eval_labels, and eval_rows must have the same length")

    class_names = sorted(set(train_labels.tolist()))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    if any(str(label) not in class_to_idx for label in eval_labels.tolist()):
        missing = sorted(set(str(label) for label in eval_labels.tolist()) - set(class_to_idx))
        raise ValueError(f"Eval labels missing from training labels for task {task_name}: {missing}")

    y_train = np.asarray([class_to_idx[str(label)] for label in train_labels], dtype=np.int64)
    y_eval = np.asarray([class_to_idx[str(label)] for label in eval_labels], dtype=np.int64)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features).astype(np.float32)
    x_eval = scaler.transform(eval_features).astype(np.float32)

    torch.manual_seed(random_state)
    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, len(class_names)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)
    for _ in range(num_epochs):
        optimizer.zero_grad()
        logits = model(x_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(x_eval)), dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)

    rows: List[Dict[str, object]] = []
    for idx, sample_row in enumerate(eval_rows):
        pred_idx = int(preds[idx])
        row = dict(sample_row)
        row["task"] = task_name
        row["true_label"] = str(eval_labels[idx])
        row["pred_label"] = class_names[pred_idx]
        row["correct"] = int(pred_idx == y_eval[idx])
        row["pred_confidence"] = float(np.max(probs[idx]))
        for class_name, class_idx in class_to_idx.items():
            row[f"prob_{class_name}"] = float(probs[idx, class_idx])
        rows.append(row)

    rows.sort(key=lambda item: (item["correct"], item["pred_confidence"], item["episode_name"]))
    metrics = {
        "task": task_name,
        "eval_accuracy": float(np.mean(preds == y_eval)),
        "num_train_samples": int(len(train_labels)),
        "num_eval_samples": int(len(eval_labels)),
        "num_classes": int(len(class_names)),
    }
    return rows, metrics


@torch.no_grad()
def forward_with_intermediates(
    model,
    img: torch.Tensor,
    tac: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> Dict[str, torch.Tensor]:
    bsz = img.shape[0]
    device = img.device

    v = model.vis_backbone(img)
    v = model.vis_proj(v)
    v_tokens = v.flatten(2).transpose(1, 2)
    num_vis_tokens = v_tokens.shape[1]

    t = model.tac_encoder(tac)
    t_tokens = t.transpose(1, 2)
    num_tac_tokens = t_tokens.shape[1]

    v_global = v_tokens.mean(dim=1)

    full_mask = None
    if padding_mask is not None:
        tac_mask = padding_mask.float().unsqueeze(1)
        tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
        tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
        tac_mask = torch.nn.functional.max_pool1d(tac_mask, kernel_size=2, stride=2)
        tac_mask = tac_mask.squeeze(1) > 0.5
        tac_mask = tac_mask[:, :num_tac_tokens]
        tac_mask_float = (~tac_mask).unsqueeze(-1).float()
        t_global = (t_tokens * tac_mask_float).sum(dim=1) / (tac_mask_float.sum(dim=1) + 1e-8)
        cls_vis_mask = torch.zeros(bsz, 1 + num_vis_tokens, dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_vis_mask, tac_mask], dim=1)
    else:
        t_global = t_tokens.mean(dim=1)

    vt_global = torch.cat([v_global, t_global], dim=-1)
    g = model.gate_mlp(vt_global)
    g_expand = g.unsqueeze(1)
    v_tokens_gated = g_expand * v_tokens + (1 - g_expand) * model.t_null

    cls_token = model.cls_token.expand(bsz, -1, -1)
    x = torch.cat([cls_token, v_tokens_gated, t_tokens], dim=1)
    seq_len = x.shape[1]
    x = x + model.pos_emb[:, :seq_len, :]
    x = model.transformer_encoder(x, src_key_padding_mask=full_mask)
    cls_out = x[:, 0, :]

    outputs = {
        "mass": model.head_mass(cls_out),
        "stiffness": model.head_stiffness(cls_out),
        "material": model.head_material(cls_out),
        "gate_score": g.squeeze(-1),
        "t_global": t_global,
        "cls_out": cls_out,
        "stiffness_hidden": model.head_stiffness[2](model.head_stiffness[1](model.head_stiffness[0](cls_out))),
        "material_hidden": model.head_material[2](model.head_material[1](model.head_material[0](cls_out))),
    }
    return outputs


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_accuracy(rows: Sequence[Dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def build_head_bias_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[int, str], Dict[str, np.ndarray]] = {}
    for row in rows:
        seed = int(row["seed"])
        if (seed, "stiffness") not in grouped:
            grouped[(seed, "stiffness")] = {"bias": np.asarray(row["stiffness_bias"], dtype=np.float32)}
            grouped[(seed, "material")] = {"bias": np.asarray(row["material_bias"], dtype=np.float32)}

    label_orders = {
        "stiffness": ["very_soft", "soft", "medium", "rigid"],
        "material": ["sponge", "foam", "wood", "hollow_container", "filled_container"],
    }
    out: List[Dict[str, object]] = []
    for (seed, task), payload in sorted(grouped.items()):
        for class_name, bias_value in zip(label_orders[task], payload["bias"]):
            out.append(
                {
                    "seed": seed,
                    "task": task,
                    "class_name": class_name,
                    "bias": float(bias_value),
                }
            )
    return out


def build_logit_decomposition_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    label_orders = {
        "stiffness": ["very_soft", "soft", "medium", "rigid"],
        "material": ["sponge", "foam", "wood", "hollow_container", "filled_container"],
    }
    grouped: Dict[tuple[int, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), str(row["obj_class"]), "stiffness")].append(row)
        grouped[(int(row["seed"]), str(row["obj_class"]), "material")].append(row)

    out: List[Dict[str, object]] = []
    for (seed, obj_class, task), subset in sorted(grouped.items()):
        logits_key = f"{task}_logits"
        projection_key = f"{task}_projection"
        bias_key = f"{task}_bias"
        mean_logits = np.mean(np.stack([np.asarray(row[logits_key], dtype=np.float32) for row in subset], axis=0), axis=0)
        mean_projection = np.mean(
            np.stack([np.asarray(row[projection_key], dtype=np.float32) for row in subset], axis=0),
            axis=0,
        )
        bias = np.asarray(subset[0][bias_key], dtype=np.float32)
        for class_name, logit_value, projection_value, bias_value in zip(
            label_orders[task],
            mean_logits,
            mean_projection,
            bias,
        ):
            out.append(
                {
                    "seed": seed,
                    "obj_class": obj_class,
                    "task": task,
                    "class_name": class_name,
                    "mean_logit": float(logit_value),
                    "mean_projection": float(projection_value),
                    "bias": float(bias_value),
                }
            )
    return out


def build_bias_ablation_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(row)

    out: List[Dict[str, object]] = []
    for seed, subset in sorted(grouped.items()):
        out.append(
            {
                "seed": seed,
                "stiffness_accuracy": aggregate_accuracy(subset, "stiffness_correct"),
                "stiffness_accuracy_no_bias": aggregate_accuracy(subset, "stiffness_correct_no_bias"),
                "material_accuracy": aggregate_accuracy(subset, "material_correct"),
                "material_accuracy_no_bias": aggregate_accuracy(subset, "material_correct_no_bias"),
                "pair_accuracy": aggregate_accuracy(subset, "pair_correct"),
                "pair_accuracy_no_bias": aggregate_accuracy(subset, "pair_correct_no_bias"),
            }
        )
    return out


def plot_task_head_retrain(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    tasks = ["stiffness", "material"]
    seeds = [int(row["seed"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, task in zip(axes, tasks):
        subset = [row for row in rows if row["task"] == task]
        seed_labels = [str(int(row["seed"])) for row in subset]
        original = [float(row["original_accuracy"]) for row in subset]
        retrained = [float(row["retrained_accuracy"]) for row in subset]
        x = np.arange(len(seed_labels))
        width = 0.38
        ax.bar(x - width / 2, original, width=width, label="original head", color="#577590")
        ax.bar(x + width / 2, retrained, width=width, label="retrained small head", color="#90be6d")
        ax.set_title(f"CLS frozen head retrain | {task}")
        ax.set_xticks(x)
        ax.set_xticklabels(seed_labels)
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_ylabel("accuracy")
    axes[-1].legend(loc="lower right")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_strict_task_head_retrain(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    tasks = ["stiffness", "material"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, task in zip(axes, tasks):
        subset = [row for row in rows if row["task"] == task]
        seed_labels = [str(int(row["seed"])) for row in subset]
        original = [float(row["original_accuracy"]) for row in subset]
        retrained = [float(row["retrained_accuracy"]) for row in subset]
        x = np.arange(len(seed_labels))
        width = 0.38
        ax.bar(x - width / 2, original, width=width, label="original head", color="#577590")
        ax.bar(x + width / 2, retrained, width=width, label="train+val retrained head", color="#f8961e")
        ax.set_title(f"Frozen CLS | train+val -> OOD | {task}")
        ax.set_xticks(x)
        ax.set_xticklabels(seed_labels)
        ax.set_ylim(0.0, 1.02)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_ylabel("OOD target accuracy")
    axes[-1].legend(loc="lower right")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stage_accuracy(stage_rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in stage_rows:
        if row["stage"] == "final_pair":
            continue
        grouped[str(row["stage"])].append(float(row["probe_mean_accuracy"]))

    ordered_stages = [
        "raw_summary_probe",
        "raw_resampled_probe",
        "t_global_probe",
        "t_global_small_mlp",
        "cls_out_probe",
        "logits_sm_probe",
    ]
    means = [float(np.mean(grouped[stage])) for stage in ordered_stages]
    stds = [float(np.std(grouped[stage])) for stage in ordered_stages]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(
        range(len(ordered_stages)),
        means,
        yerr=stds,
        color=["#577590", "#4d908e", "#43aa8b", "#90be6d", "#f8961e", "#f94144"],
    )
    ax.set_xticks(range(len(ordered_stages)))
    ax.set_xticklabels(ordered_stages, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("binary probe accuracy")
    ax.set_title("Hollow vs Yoga separability across stages")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_margin_boxplots(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    class_order = list(DEFAULT_CLASSES)
    margin_keys = [
        ("stiffness_true_margin", "Stiffness true-class margin"),
        ("material_true_margin", "Material true-class margin"),
    ]
    for ax, (margin_key, title) in zip(axes, margin_keys):
        series = [
            [float(row[margin_key]) for row in rows if row["obj_class"] == class_name]
            for class_name in class_order
        ]
        ax.boxplot(series, tick_labels=class_order, showfliers=True)
        ax.axhline(0.0, color="#c1121f", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel("logit margin")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mean_logit_bars(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    tasks = ("stiffness", "material")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, task in zip(axes, tasks):
        subset = [row for row in rows if row["task"] == task]
        class_names = []
        for row in subset:
            if row["class_name"] not in class_names:
                class_names.append(row["class_name"])
        obj_classes = list(DEFAULT_CLASSES)
        width = 0.35
        x = np.arange(len(class_names))
        for idx, obj_class in enumerate(obj_classes):
            values = []
            for class_name in class_names:
                class_rows = [row for row in subset if row["obj_class"] == obj_class and row["class_name"] == class_name]
                values.append(float(np.mean([row["mean_logit"] for row in class_rows])))
            ax.bar(x + (idx - 0.5) * width, values, width=width, label=obj_class)
        ax.set_title(f"{task} mean logits by true object")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=20, ha="right")
        ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    axes[-1].legend(loc="best")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def collect_seed_rows(
    run_dir: Path,
    split_name: str,
    data_root: Path,
    cli_args: argparse.Namespace,
    target_classes: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    device = resolve_device(cli_args.device)
    checkpoint_path = run_dir / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    runtime_args = build_runtime_args(cfg, cli_args)
    override_props_config = resolve_override_props_config(split_name, cfg)

    loader = build_loader(
        data_root / split_name,
        runtime_args.batch_size,
        runtime_args.max_tactile_len,
        runtime_args.num_workers,
        override_props_config=override_props_config,
    )
    dataset = loader.dataset
    label_names = ordered_label_names(dataset)

    model = build_model(
        cfg,
        runtime_args,
        mass_classes=len(dataset.mass_to_idx),
        stiffness_classes=len(dataset.stiffness_to_idx),
        material_classes=len(dataset.material_to_idx),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stiff_weight = model.head_stiffness[3].weight.detach().cpu().numpy().astype(np.float32)
    stiff_bias = model.head_stiffness[3].bias.detach().cpu().numpy().astype(np.float32)
    material_weight = model.head_material[3].weight.detach().cpu().numpy().astype(np.float32)
    material_bias = model.head_material[3].bias.detach().cpu().numpy().astype(np.float32)

    rows: List[Dict[str, object]] = []
    target_set = set(target_classes) if target_classes is not None else None
    for batch in loader:
        images = batch["image"].to(device)
        tactile = batch["tactile"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        prefix_mask = effective_padding_mask(
            padding_mask=padding_mask,
            train_mode=False,
            online_train_prob=0.0,
            online_min_prefix_ratio=runtime_args.online_min_prefix_ratio,
            min_prefix_len=runtime_args.min_prefix_len,
            fixed_ratio=None,
        )
        outputs = forward_with_intermediates(model, images, tactile, prefix_mask)

        for idx, obj_class in enumerate(batch["obj_class"]):
            if target_set is not None and obj_class not in target_set:
                continue
            valid_len = int((~batch["padding_mask"][idx]).sum().item())
            tactile_valid = batch["tactile"][idx, :, :valid_len].cpu().numpy()
            stiffness_logits = outputs["stiffness"][idx].detach().cpu().numpy()
            material_logits = outputs["material"][idx].detach().cpu().numpy()
            stiffness_hidden = outputs["stiffness_hidden"][idx].detach().cpu().numpy()
            material_hidden = outputs["material_hidden"][idx].detach().cpu().numpy()
            stiffness_projection, stiffness_rebuilt_logits = decompose_final_linear_logits(
                stiffness_hidden,
                stiff_weight,
                stiff_bias,
            )
            material_projection, material_rebuilt_logits = decompose_final_linear_logits(
                material_hidden,
                material_weight,
                material_bias,
            )
            pred_stiff_idx = int(np.argmax(stiffness_logits))
            pred_mat_idx = int(np.argmax(material_logits))
            pred_stiff_no_bias_idx = int(np.argmax(stiffness_projection))
            pred_mat_no_bias_idx = int(np.argmax(material_projection))
            true_stiff_idx = int(batch["stiffness"][idx].item())
            true_mat_idx = int(batch["material"][idx].item())

            rows.append(
                {
                    "seed": int(cfg.get("seed")),
                    "run_name": run_dir.name,
                    "obj_class": obj_class,
                    "episode_name": batch["episode_name"][idx],
                    "episode_path": batch["episode_path"][idx],
                    "sample_index": int(batch["sample_index"][idx]),
                    "true_stiffness": label_names["stiffness"][true_stiff_idx],
                    "true_material": label_names["material"][true_mat_idx],
                    "pred_stiffness": label_names["stiffness"][pred_stiff_idx],
                    "pred_material": label_names["material"][pred_mat_idx],
                    "pred_stiffness_no_bias": label_names["stiffness"][pred_stiff_no_bias_idx],
                    "pred_material_no_bias": label_names["material"][pred_mat_no_bias_idx],
                    "stiffness_correct": int(true_stiff_idx == pred_stiff_idx),
                    "material_correct": int(true_mat_idx == pred_mat_idx),
                    "stiffness_correct_no_bias": int(true_stiff_idx == pred_stiff_no_bias_idx),
                    "material_correct_no_bias": int(true_mat_idx == pred_mat_no_bias_idx),
                    "pair_correct": int((true_stiff_idx == pred_stiff_idx) and (true_mat_idx == pred_mat_idx)),
                    "pair_correct_no_bias": int(
                        (true_stiff_idx == pred_stiff_no_bias_idx) and (true_mat_idx == pred_mat_no_bias_idx)
                    ),
                    "gate_score": float(outputs["gate_score"][idx].item()),
                    "stiffness_true_margin": compute_true_class_margin(stiffness_logits, true_stiff_idx),
                    "material_true_margin": compute_true_class_margin(material_logits, true_mat_idx),
                    "stiffness_true_margin_no_bias": compute_true_class_margin(stiffness_projection, true_stiff_idx),
                    "material_true_margin_no_bias": compute_true_class_margin(material_projection, true_mat_idx),
                    "raw_resampled_feature": flatten_resampled_trace(tactile_valid, target_len=cli_args.resampled_len),
                    "raw_summary_feature": build_summary_features(tactile_valid),
                    "t_global_feature": outputs["t_global"][idx].detach().cpu().numpy().astype(np.float32),
                    "cls_out_feature": outputs["cls_out"][idx].detach().cpu().numpy().astype(np.float32),
                    "logits_sm_feature": np.concatenate([stiffness_logits, material_logits], axis=0).astype(np.float32),
                    "stiffness_logits": stiffness_rebuilt_logits.astype(np.float32),
                    "stiffness_projection": stiffness_projection.astype(np.float32),
                    "stiffness_bias": stiff_bias.copy(),
                    "material_logits": material_rebuilt_logits.astype(np.float32),
                    "material_projection": material_projection.astype(np.float32),
                    "material_bias": material_bias.copy(),
                }
            )
    rows.sort(key=lambda item: (item["seed"], item["obj_class"], item["episode_name"]))
    return rows


def summarize_stage_for_seed(seed_rows: Sequence[Dict[str, object]], seed: int, random_state: int) -> List[Dict[str, object]]:
    labels = np.asarray([row["obj_class"] for row in seed_rows], dtype=object)
    out: List[Dict[str, object]] = []
    stage_map = {
        "t_global_probe": "t_global_feature",
        "cls_out_probe": "cls_out_feature",
        "logits_sm_probe": "logits_sm_feature",
    }
    pair_accuracy = aggregate_accuracy(seed_rows, "pair_correct")
    stiffness_accuracy = aggregate_accuracy(seed_rows, "stiffness_correct")
    material_accuracy = aggregate_accuracy(seed_rows, "material_correct")
    for stage_name, feature_key in stage_map.items():
        features = np.stack([row[feature_key] for row in seed_rows], axis=0).astype(np.float32)
        probe_result = evaluate_linear_probe(features, labels, random_state=random_state)
        out.append(
            build_stage_summary_row(
                seed=seed,
                stage_name=stage_name,
                probe_result=probe_result,
                pair_accuracy=pair_accuracy,
                stiffness_accuracy=stiffness_accuracy,
                material_accuracy=material_accuracy,
            )
        )
    out.append(
        {
            "seed": seed,
            "stage": "final_pair",
            "probe_mean_accuracy": float("nan"),
            "probe_std_accuracy": float("nan"),
            "probe_n_splits": 0,
            "pair_accuracy": pair_accuracy,
            "stiffness_accuracy": stiffness_accuracy,
            "material_accuracy": material_accuracy,
        }
    )
    return out


def write_summary(
    output_dir: Path,
    probe_rows_by_name: Dict[str, List[Dict[str, object]]],
    probe_accuracy_by_name: Dict[str, float],
    stage_rows: Sequence[Dict[str, object]],
    logits_rows: Sequence[Dict[str, object]],
    bias_ablation_rows: Sequence[Dict[str, object]],
    head_bias_rows: Sequence[Dict[str, object]],
    task_head_rows: Sequence[Dict[str, object]],
    strict_task_head_rows: Sequence[Dict[str, object]],
) -> None:
    lines: List[str] = []
    lines.append("# Gating Hollow vs Yoga Diagnosis")
    lines.append("")
    lines.append("## Small tactile-only binary probe")
    lines.append("")
    for probe_name in ("raw_summary_probe", "raw_resampled_probe"):
        rows = probe_rows_by_name[probe_name]
        errors = [row for row in rows if int(row["correct"]) == 0]
        lines.append(
            f"- `{probe_name}` LOO accuracy: `{100.0 * probe_accuracy_by_name[probe_name]:.2f}%` "
            f"(errors: `{len(errors)}` / `{len(rows)}`)"
        )
    mlp_rows = [row for row in stage_rows if row["stage"] == "t_global_small_mlp"]
    if mlp_rows:
        mlp_mean = float(np.mean([row["probe_mean_accuracy"] for row in mlp_rows]))
        mlp_std = float(np.std([row["probe_mean_accuracy"] for row in mlp_rows]))
        lines.append(
            f"- `t_global_small_mlp` 5-seed mean CV accuracy: "
            f"`{100.0 * mlp_mean:.2f}% ± {100.0 * mlp_std:.2f}%`"
        )
    lines.append("")

    lines.append("## Gating stage separability")
    lines.append("")
    shared_rows = [row for row in stage_rows if row["seed"] == "shared"]
    for row in shared_rows:
        lines.append(
            f"- `{row['stage']}` CV accuracy: `{100.0 * float(row['probe_mean_accuracy']):.2f}% ± "
            f"{100.0 * float(row['probe_std_accuracy']):.2f}%`"
        )
    if shared_rows:
        lines.append("")
    lines.append("| Seed | t_global probe | t_global small head | CLS probe | Logits probe | Final pair acc | Stiffness acc | Material acc |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    by_seed: Dict[int, Dict[str, Dict[str, object]]] = defaultdict(dict)
    final_rows: Dict[int, Dict[str, object]] = {}
    for row in stage_rows:
        if row["seed"] == "shared":
            continue
        if row["stage"] == "final_pair":
            final_rows[int(row["seed"])] = row
        else:
            by_seed[int(row["seed"])][str(row["stage"])] = row
    for seed in sorted(by_seed):
        t_row = by_seed[seed]["t_global_probe"]
        m_row = by_seed[seed]["t_global_small_mlp"]
        c_row = by_seed[seed]["cls_out_probe"]
        l_row = by_seed[seed]["logits_sm_probe"]
        f_row = final_rows[seed]
        lines.append(
            f"| {seed} | "
            f"{100.0 * float(t_row['probe_mean_accuracy']):.2f}% | "
            f"{100.0 * float(m_row['probe_mean_accuracy']):.2f}% | "
            f"{100.0 * float(c_row['probe_mean_accuracy']):.2f}% | "
            f"{100.0 * float(l_row['probe_mean_accuracy']):.2f}% | "
            f"{100.0 * float(f_row['pair_accuracy']):.2f}% | "
            f"{100.0 * float(f_row['stiffness_accuracy']):.2f}% | "
            f"{100.0 * float(f_row['material_accuracy']):.2f}% |"
        )
    lines.append("")

    lines.append("## CLS Frozen Task-Head Retrain")
    lines.append("")
    lines.append("| Seed | Task | Original acc | Retrained acc | Splits |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in task_head_rows:
        lines.append(
            f"| {row['seed']} | {row['task']} | "
            f"{100.0 * float(row['original_accuracy']):.2f}% | "
            f"{100.0 * float(row['retrained_accuracy']):.2f}% | "
            f"{int(row['n_splits'])} |"
        )
    lines.append("")

    lines.append("## CLS Frozen Task-Head Retrain (Train/Val -> OOD)")
    lines.append("")
    lines.append("| Seed | Task | Original OOD acc | Retrained OOD acc | Train samples | Eval samples |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in strict_task_head_rows:
        lines.append(
            f"| {row['seed']} | {row['task']} | "
            f"{100.0 * float(row['original_accuracy']):.2f}% | "
            f"{100.0 * float(row['retrained_accuracy']):.2f}% | "
            f"{int(row['num_train_samples'])} | "
            f"{int(row['num_eval_samples'])} |"
        )
    lines.append("")

    lines.append("## Bias Ablation")
    lines.append("")
    lines.append("| Seed | Stiffness acc | Stiffness acc no bias | Material acc | Material acc no bias | Pair acc | Pair acc no bias |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in bias_ablation_rows:
        lines.append(
            f"| {row['seed']} | "
            f"{100.0 * float(row['stiffness_accuracy']):.2f}% | "
            f"{100.0 * float(row['stiffness_accuracy_no_bias']):.2f}% | "
            f"{100.0 * float(row['material_accuracy']):.2f}% | "
            f"{100.0 * float(row['material_accuracy_no_bias']):.2f}% | "
            f"{100.0 * float(row['pair_accuracy']):.2f}% | "
            f"{100.0 * float(row['pair_accuracy_no_bias']):.2f}% |"
        )
    lines.append("")

    lines.append("## Final-Layer Bias Snapshot")
    lines.append("")
    for task in ("stiffness", "material"):
        task_rows = [row for row in head_bias_rows if row["task"] == task]
        mean_by_class: Dict[str, List[float]] = defaultdict(list)
        for row in task_rows:
            mean_by_class[str(row["class_name"])].append(float(row["bias"]))
        bias_summary = ", ".join(
            f"{class_name}={np.mean(values):.3f}"
            for class_name, values in mean_by_class.items()
        )
        lines.append(f"- `{task}` final-layer bias mean across seeds: {bias_summary}")
    lines.append("")

    lines.append("## Logit-margin observations")
    lines.append("")
    for class_name in DEFAULT_CLASSES:
        subset = [row for row in logits_rows if row["obj_class"] == class_name]
        lines.append(
            f"- `{class_name}`: mean stiffness margin `{np.mean([row['stiffness_true_margin'] for row in subset]):.3f}`, "
            f"mean material margin `{np.mean([row['material_true_margin'] for row in subset]):.3f}`, "
            f"mean gate `{np.mean([row['gate_score'] for row in subset]):.3f}`"
        )
    lines.append("")

    lines.append("## Hardest final-model samples")
    lines.append("")
    aggregated: Dict[tuple[str, str], Dict[str, object]] = {}
    for row in logits_rows:
        key = (str(row["obj_class"]), str(row["episode_name"]))
        if key not in aggregated:
            aggregated[key] = {
                "obj_class": row["obj_class"],
                "episode_name": row["episode_name"],
                "pair_correct_values": [],
                "stiffness_margin_values": [],
                "material_margin_values": [],
            }
        aggregated[key]["pair_correct_values"].append(int(row["pair_correct"]))
        aggregated[key]["stiffness_margin_values"].append(float(row["stiffness_true_margin"]))
        aggregated[key]["material_margin_values"].append(float(row["material_true_margin"]))
    hardest = sorted(
        aggregated.values(),
        key=lambda item: (
            np.mean(item["pair_correct_values"]),
            np.mean(item["stiffness_margin_values"]) + np.mean(item["material_margin_values"]),
        ),
    )[:12]
    lines.append("| Sample | Pair acc | Mean stiff margin | Mean material margin |")
    lines.append("| --- | ---: | ---: | ---: |")
    for item in hardest:
        lines.append(
            f"| `{item['obj_class']}/{item['episode_name']}` | "
            f"{100.0 * np.mean(item['pair_correct_values']):.0f}% | "
            f"{np.mean(item['stiffness_margin_values']):.3f} | "
            f"{np.mean(item['material_margin_values']):.3f} |"
        )
    lines.append("")

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why gating predicts Cardbox_hollow_noise and YogaBrick_Foil_ANCHOR similarly.",
    )
    parser.add_argument("--run-root", type=Path, default=Path(DEFAULT_RUN_ROOT))
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--split", type=str, default="ood_test")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tactile-len", type=int, default=3000)
    parser.add_argument("--resampled-len", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_seed_rows: List[Dict[str, object]] = []
    stage_rows: List[Dict[str, object]] = []
    small_head_prediction_rows: List[Dict[str, object]] = []
    task_head_rows: List[Dict[str, object]] = []
    task_head_prediction_rows: List[Dict[str, object]] = []
    strict_task_head_rows: List[Dict[str, object]] = []
    strict_task_head_prediction_rows: List[Dict[str, object]] = []
    for run_dir in iter_run_dirs(run_root):
        seed_rows = collect_seed_rows(
            run_dir=run_dir,
            split_name=args.split,
            data_root=data_root,
            cli_args=args,
            target_classes=DEFAULT_CLASSES,
        )
        if not seed_rows:
            continue
        all_seed_rows.extend(seed_rows)
        seed = int(seed_rows[0]["seed"])
        stage_rows.extend(summarize_stage_for_seed(seed_rows, seed=seed, random_state=args.random_state))
        t_global_features = np.stack([row["t_global_feature"] for row in seed_rows], axis=0).astype(np.float32)
        t_global_labels = np.asarray([row["obj_class"] for row in seed_rows], dtype=object)
        t_global_sample_rows = [
            {
                "seed": seed,
                "obj_class": row["obj_class"],
                "episode_name": row["episode_name"],
                "episode_path": row["episode_path"],
            }
            for row in seed_rows
        ]
        seed_small_head_rows, seed_small_head_metrics = train_small_mlp_head_predictions(
            t_global_features,
            t_global_labels,
            sample_rows=t_global_sample_rows,
            random_state=args.random_state + seed,
        )
        small_head_prediction_rows.extend(seed_small_head_rows)
        pair_accuracy = aggregate_accuracy(seed_rows, "pair_correct")
        stiffness_accuracy = aggregate_accuracy(seed_rows, "stiffness_correct")
        material_accuracy = aggregate_accuracy(seed_rows, "material_correct")
        stage_rows.append(
            build_stage_summary_row(
                seed=seed,
                stage_name="t_global_small_mlp",
                probe_result=seed_small_head_metrics,
                pair_accuracy=pair_accuracy,
                stiffness_accuracy=stiffness_accuracy,
                material_accuracy=material_accuracy,
            )
        )
        cls_out_features = np.stack([row["cls_out_feature"] for row in seed_rows], axis=0).astype(np.float32)
        task_targets = (
            ("stiffness", [str(row["true_stiffness"]) for row in seed_rows], "stiffness_correct"),
            ("material", [str(row["true_material"]) for row in seed_rows], "material_correct"),
        )
        for task_name, task_labels, original_key in task_targets:
            task_sample_rows = [
                {
                    "seed": seed,
                    "obj_class": row["obj_class"],
                    "episode_name": row["episode_name"],
                    "episode_path": row["episode_path"],
                }
                for row in seed_rows
            ]
            task_pred_rows, task_metrics = train_frozen_task_head_predictions(
                cls_out_features,
                np.asarray(task_labels, dtype=object),
                sample_rows=task_sample_rows,
                task_name=task_name,
                random_state=args.random_state + seed * 17,
            )
            task_head_prediction_rows.extend(task_pred_rows)
            task_head_rows.append(
                {
                    "seed": seed,
                    "task": task_name,
                    "original_accuracy": aggregate_accuracy(seed_rows, original_key),
                    "retrained_accuracy": float(task_metrics["mean_accuracy"]),
                    "retrained_std": float(task_metrics["std_accuracy"]),
                    "n_splits": int(task_metrics["n_splits"]),
                }
            )
        train_rows = collect_seed_rows(
            run_dir=run_dir,
            split_name="train",
            data_root=data_root,
            cli_args=args,
            target_classes=None,
        )
        val_rows = collect_seed_rows(
            run_dir=run_dir,
            split_name="val",
            data_root=data_root,
            cli_args=args,
            target_classes=None,
        )
        trainval_rows = [*train_rows, *val_rows]
        trainval_cls = np.stack([row["cls_out_feature"] for row in trainval_rows], axis=0).astype(np.float32)
        ood_cls = np.stack([row["cls_out_feature"] for row in seed_rows], axis=0).astype(np.float32)
        strict_targets = (
            ("stiffness", [str(row["true_stiffness"]) for row in trainval_rows], [str(row["true_stiffness"]) for row in seed_rows], "stiffness_correct"),
            ("material", [str(row["true_material"]) for row in trainval_rows], [str(row["true_material"]) for row in seed_rows], "material_correct"),
        )
        strict_eval_rows = [
            {
                "seed": seed,
                "obj_class": row["obj_class"],
                "episode_name": row["episode_name"],
                "episode_path": row["episode_path"],
            }
            for row in seed_rows
        ]
        for task_name, train_labels, eval_labels, original_key in strict_targets:
            strict_pred_rows, strict_metrics = train_frozen_task_head_train_eval_split(
                trainval_cls,
                np.asarray(train_labels, dtype=object),
                ood_cls,
                np.asarray(eval_labels, dtype=object),
                strict_eval_rows,
                task_name=task_name,
                random_state=args.random_state + seed * 31,
            )
            strict_task_head_prediction_rows.extend(strict_pred_rows)
            strict_task_head_rows.append(
                {
                    "seed": seed,
                    "task": task_name,
                    "original_accuracy": aggregate_accuracy(seed_rows, original_key),
                    "retrained_accuracy": float(strict_metrics["eval_accuracy"]),
                    "num_train_samples": int(strict_metrics["num_train_samples"]),
                    "num_eval_samples": int(strict_metrics["num_eval_samples"]),
                }
            )

    if not all_seed_rows:
        raise RuntimeError("No rows collected for the requested classes.")

    shared_rows = [row for row in all_seed_rows if int(row["seed"]) == min(int(item["seed"]) for item in all_seed_rows)]
    sample_rows = [
        {
            "obj_class": row["obj_class"],
            "episode_name": row["episode_name"],
            "episode_path": row["episode_path"],
        }
        for row in shared_rows
    ]
    labels = np.asarray([row["obj_class"] for row in shared_rows], dtype=object)

    probe_rows_by_name: Dict[str, List[Dict[str, object]]] = {}
    probe_accuracy_by_name: Dict[str, float] = {}
    for probe_name, feature_key in (
        ("raw_summary_probe", "raw_summary_feature"),
        ("raw_resampled_probe", "raw_resampled_feature"),
    ):
        features = np.stack([row[feature_key] for row in shared_rows], axis=0).astype(np.float32)
        loo_rows, loo_acc = leave_one_out_probe_predictions(features, labels, sample_rows=sample_rows)
        cv_probe = evaluate_linear_probe(features, labels, random_state=args.random_state)
        probe_rows_by_name[probe_name] = loo_rows
        probe_accuracy_by_name[probe_name] = loo_acc
        write_csv(output_dir / f"{probe_name}_sample_predictions.csv", loo_rows)
        stage_rows.append(
            build_stage_summary_row(
                seed="shared",
                stage_name=probe_name,
                probe_result=cv_probe,
                pair_accuracy=float("nan"),
                stiffness_accuracy=float("nan"),
                material_accuracy=float("nan"),
            )
        )

    logits_rows = []
    for row in all_seed_rows:
        logits_rows.append(
            {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "raw_resampled_feature",
                    "raw_summary_feature",
                    "t_global_feature",
                    "cls_out_feature",
                    "logits_sm_feature",
                }
            }
        )
    head_bias_rows = build_head_bias_rows(all_seed_rows)
    logit_decomposition_rows = build_logit_decomposition_rows(all_seed_rows)
    bias_ablation_rows = build_bias_ablation_rows(all_seed_rows)
    write_csv(output_dir / "gating_logits_sample_rows.csv", logits_rows)
    write_csv(output_dir / "gating_stage_probe_by_seed.csv", stage_rows)
    write_csv(output_dir / "t_global_small_mlp_oof_predictions.csv", small_head_prediction_rows)
    write_csv(output_dir / "cls_out_task_head_retrain_accuracy.csv", task_head_rows)
    write_csv(output_dir / "cls_out_task_head_oof_predictions.csv", task_head_prediction_rows)
    write_csv(output_dir / "cls_out_task_head_trainval_to_ood_accuracy.csv", strict_task_head_rows)
    write_csv(output_dir / "cls_out_task_head_trainval_to_ood_predictions.csv", strict_task_head_prediction_rows)
    write_csv(output_dir / "head_bias_rows.csv", head_bias_rows)
    write_csv(output_dir / "logit_decomposition_by_object.csv", logit_decomposition_rows)
    write_csv(output_dir / "bias_ablation_accuracy.csv", bias_ablation_rows)
    plot_stage_accuracy(stage_rows, output_dir / "stage_probe_accuracy.png")
    plot_margin_boxplots(logits_rows, output_dir / "logit_margin_boxplots.png")
    plot_mean_logit_bars(logit_decomposition_rows, output_dir / "mean_logits_by_object.png")
    plot_task_head_retrain(task_head_rows, output_dir / "cls_out_task_head_retrain_accuracy.png")
    plot_strict_task_head_retrain(strict_task_head_rows, output_dir / "cls_out_task_head_trainval_to_ood_accuracy.png")
    write_summary(
        output_dir,
        probe_rows_by_name,
        probe_accuracy_by_name,
        stage_rows,
        logits_rows,
        bias_ablation_rows,
        head_bias_rows,
        task_head_rows,
        strict_task_head_rows,
    )

    meta = {
        "run_root": str(run_root),
        "data_root": str(data_root),
        "split": args.split,
        "classes": list(DEFAULT_CLASSES),
        "num_rows": len(all_seed_rows),
        "num_unique_samples": len(shared_rows),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved diagnosis to {output_dir}")


if __name__ == "__main__":
    main()
