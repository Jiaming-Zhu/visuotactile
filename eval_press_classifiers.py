#!/usr/bin/env python
"""
评估已训练的三类分类器（帧级 MLP、序列 Transformer、序列 LSTM）在 sponge/woodblock 数据上的表现。

使用方式（默认尝试三个常用 checkpoint）：
  python learn_PyBullet/eval_press_classifiers.py \
    --sponge_dir data/so101_press/sponge_20251107_201409 \
    --woodblock_dir data/so101_press/woodBlock_20251107_204040

也可以指定模型路径：
  python learn_PyBullet/eval_press_classifiers.py --models outputs/press_classifier.pt outputs/press_transformer_classifier.pt

说明：
- MLP 使用帧级数据集（单帧输入）；Transformer/LSTM 使用序列数据集（窗口长度取自 ckpt 或 --context）。
- 可选：按 episode 投票评估（对同一 episode 的所有窗口/帧进行多数投票聚合）。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys
import time

import torch
from torch.utils.data import DataLoader

# 确保仓库根目录在 sys.path 中
sys.path.append(str(Path(__file__).resolve().parents[1]))

from learn_PyBullet.datasets.parquet_frame_dataset import ParquetFrameDataset as FrameDS, SourceSpec as FrameSrc
from learn_PyBullet.datasets.parquet_seq_dataset import ParquetSeqDataset as SeqDS, SourceSpec as SeqSrc
from learn_PyBullet.models.mlp_classifier import MLPClassifier, MLPConfig
from learn_PyBullet.models.transformer_classifier import TransformerClassifier, TransformerClassifierConfig
from learn_PyBullet.models.lstm_classifier import LSTMClassifier, LSTMClassifierConfig


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained classifiers on sponge vs woodblock")
    p.add_argument("--models", nargs="*", type=str, default=[
        "outputs/press_classifier.pt",
        "outputs/press_transformer_classifier.pt",
        "outputs/press_lstm_classifier.pt",
    ])
    p.add_argument("--sponge_dir", type=str, default="data/so101_press/sponge_20251107_201409")
    p.add_argument("--woodblock_dir", type=str, default="data/so101_press/woodBlock_20251107_204040")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # 对序列模型生效：若 ckpt 未包含 context_len，则使用该值
    p.add_argument("--context", type=int, default=16)
    # 可覆盖 ckpt 中的 state_keys（一般保持默认即可）
    p.add_argument("--state_keys", nargs="*", type=str, default=None)
    # 是否启用按 episode 投票评估
    p.add_argument("--episode_voting", action="store_true", help="对同一 episode 的所有窗口/帧进行多数投票，计算 episode 级准确率")
    return p.parse_args()


def read_features(dataset_dir: Path) -> Dict:
    with open(dataset_dir / "meta/info.json") as f:
        info = json.load(f)
    return info["features"]


def compute_input_dim(state_keys: List[str], features: Dict[str, Dict]) -> int:
    dim = 0
    for k in state_keys:
        if k in features:
            dim += int(features[k]["shape"][0])
    return dim


def collate_frame(batch: List[dict]):
    if len(batch) == 0:
        return {"x": torch.empty(0), "y": torch.empty(0, dtype=torch.long)}
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["label"] for b in batch], dim=0)
    return {"x": x, "y": y}


def collate_seq(batch: List[dict]):
    if len(batch) == 0:
        return {"x": torch.empty(0), "y": torch.empty(0, dtype=torch.long)}
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["label"] for b in batch], dim=0)
    return {"x": x, "y": y}


def load_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    required = ["state_dict", "config", "state_keys", "label_map"]
    for k in required:
        if k not in ckpt:
            raise ValueError(f"Checkpoint {path} missing key: {k}")
    return ckpt


def infer_model_type(cfg: dict) -> str:
    if "d_model" in cfg and "n_heads" in cfg:
        return "transformer"
    if "num_layers" in cfg and "bidirectional" in cfg:
        return "lstm"
    return "mlp"


# 返回 (dataset, dataloader)

def build_dataloader(model_type: str, sponge_dir: Path, wood_dir: Path, state_keys: List[str], batch_size: int, device: str, context_len: int | None, features: Dict):
    if model_type == "mlp":
        ds = FrameDS(
            sources=[FrameSrc(sponge_dir, 0), FrameSrc(wood_dir, 1)],
            state_keys=state_keys,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=device != "cpu", collate_fn=collate_frame)
        return ds, loader
    else:
        T = int(context_len) if context_len is not None else 16
        ds = SeqDS(
            sources=[SeqSrc(sponge_dir, 0), SeqSrc(wood_dir, 1)],
            state_keys=state_keys,
            context_len=T,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=device != "cpu", collate_fn=collate_seq)
        return ds, loader


def build_model(model_type: str, cfg: dict, input_dim: int, num_classes: int, device: str):
    if model_type == "mlp":
        mcfg = MLPConfig(input_dim=input_dim, num_classes=num_classes, hidden_dim=cfg.get("hidden_dim", 256), hidden_layers=cfg.get("hidden_layers", 2), dropout=cfg.get("dropout", 0.1))
        model = MLPClassifier(mcfg)
    elif model_type == "transformer":
        mcfg = TransformerClassifierConfig(
            input_dim=input_dim,
            num_classes=num_classes,
            context_len=cfg.get("context_len", 16),
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 4),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
        )
        model = TransformerClassifier(mcfg)
    elif model_type == "lstm":
        mcfg = LSTMClassifierConfig(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=cfg.get("hidden_dim", 256),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.1),
            bidirectional=cfg.get("bidirectional", False),
        )
        model = LSTMClassifier(mcfg)
    else:
        raise ValueError(f"Unknown model_type={model_type}")
    return model.to(device)


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    return cm


def evaluate_model(model, loader: DataLoader, device: str) -> Dict:
    model.eval()
    all_preds = []
    all_labels = []
    total = len(loader.dataset) if hasattr(loader, "dataset") else 0
    processed = 0
    # 每 10% 打印一次进度（至少每个批次打印一次）
    step = max(1, total // 10) if total > 0 else 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch["x"].numel() == 0:
                continue
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            processed += y.numel()
            if total > 0 and (processed >= step and processed % step < y.numel()):
                pct = processed / total * 100.0
                elapsed = time.time() - t0
                print(f"[Eval] processed {processed}/{total} ({pct:.1f}%), elapsed {elapsed:.1f}s")

    if len(all_preds) == 0:
        return {"accuracy": 0.0, "cm": torch.zeros((2, 2), dtype=torch.long), "n": 0}

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = (preds == labels).float().mean().item()
    cm = confusion_matrix(preds, labels, num_classes=2)
    return {"accuracy": acc, "cm": cm, "n": labels.numel()}


def evaluate_model_episode_voting(model, model_type: str, ds, loader: DataLoader, device: str) -> Dict:
    """
    对同一 episode 的所有窗口/帧进行多数投票，得到 episode 级别的预测并计算准确率。
    要求 loader 的 shuffle=False 以保证顺序为“按文件依次展开”。
    """
    model.eval()
    # 计算每个 episode 的样本数边界：序列模型使用窗口数，帧级使用行数
    if model_type == "mlp":
        # ParquetFrameDataset: file_lengths 为每个 episode 的帧数
        counts_per_episode = list(getattr(ds, "file_lengths"))
        labels_per_episode = [lbl for (lbl, _path) in getattr(ds, "files")]
    else:
        # ParquetSeqDataset: file_win 为每个 episode 的窗口数（可能为 0）
        counts_per_episode = list(getattr(ds, "file_win"))
        labels_per_episode = [lbl for (lbl, _path) in getattr(ds, "files")]

    total_samples = sum(c for c in counts_per_episode if c > 0)
    zero_ep = sum(1 for c in counts_per_episode if c <= 0)
    print(f"[Voting] aggregating {total_samples} samples across {len(counts_per_episode)-zero_ep} episodes (skipped {zero_ep})")

    # 顺序遍历 loader，累计预测（要求 shuffle=False，与 ds 的顺序一致）
    all_preds = []
    t0 = time.time()
    processed = 0
    with torch.no_grad():
        for batch in loader:
            if batch["x"].numel() == 0:
                continue
            x = batch["x"].to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            processed += preds.numel()
    print(f"[Voting] inference done, collected {processed} predictions in {time.time()-t0:.1f}s")

    preds_flat = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    # 依据 counts_per_episode 划分区间做多数投票
    cm = torch.zeros((2, 2), dtype=torch.long)
    correct = 0
    total_ep = 0

    offset = 0
    for idx, cnt in enumerate(counts_per_episode):
        if cnt <= 0:
            continue
        true_label = int(labels_per_episode[idx])
        ep_preds = preds_flat[offset: offset + cnt]
        offset += cnt
        if ep_preds.numel() == 0:
            continue
        # 多数投票（简单 argmax 计数，平票时选择较小索引的类别）
        votes0 = int((ep_preds == 0).sum().item())
        votes1 = int((ep_preds == 1).sum().item())
        pred_label = 0 if votes0 >= votes1 else 1
        cm[true_label, pred_label] += 1
        correct += int(pred_label == true_label)
        total_ep += 1

    episode_acc = correct / total_ep if total_ep > 0 else 0.0
    return {"episode_accuracy": episode_acc, "episode_cm": cm, "episodes": total_ep}


def main():
    args = parse_args()
    sponge_dir = Path(args.sponge_dir)
    wood_dir = Path(args.woodblock_dir)
    if not (sponge_dir / "meta/info.json").exists() or not (wood_dir / "meta/info.json").exists():
        raise FileNotFoundError("Please provide valid sponge_dir and woodblock_dir containing meta/info.json")

    print("Loading features & preparing evaluation...")
    features = read_features(sponge_dir)

    results = []
    for mpath in args.models:
        p = Path(mpath)
        if not p.exists():
            print(f"[Skip] {p} not found")
            continue
        print(f"\n==== Evaluating {p} ====")
        ckpt = load_checkpoint(p)
        cfg = ckpt["config"]
        model_type = infer_model_type(cfg)
        state_keys = args.state_keys if args.state_keys else ckpt.get("state_keys", [])
        if not state_keys:
            # fallback to defaults
            state_keys = [
                "observation.position",
                "observation.velocity",
                "observation.load",
                "observation.current",
            ]
        input_dim = compute_input_dim(state_keys, features)
        num_classes = len(ckpt.get("label_map", {0: "sponge", 1: "woodblock"}))
        context_len = cfg.get("context_len", None) if model_type == "transformer" else (args.context if model_type != "mlp" else None)
        print(f"ModelType={model_type}, input_dim={input_dim}, context={context_len if model_type!='mlp' else 'N/A'}")

        ds, loader = build_dataloader(model_type, sponge_dir, wood_dir, state_keys, args.batch_size, args.device, context_len, features)
        num_files = len(getattr(ds, "files", []))
        print(f"Dataset: files={num_files}, samples={len(ds)} (batch_size={args.batch_size})")
        if model_type == "mlp":
            pass
        else:
            zero_win = sum(1 for c in getattr(ds, "file_win") if c <= 0)
            if zero_win > 0:
                print(f"Note: {zero_win} episodes have < context windows and will be skipped in voting")

        model = build_model(model_type, cfg, input_dim, num_classes, args.device)
        model.load_state_dict(ckpt["state_dict"], strict=True)

        print("Running sample-level evaluation...")
        metrics = evaluate_model(model, loader, args.device)
        cm = metrics["cm"].tolist()
        print("Sample-level Result:")
        print(f"Samples: {metrics['n']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        print(f"[[{cm[0][0]}, {cm[0][1]}],\n [{cm[1][0]}, {cm[1][1]}]]")

        # episode 投票准确率（可选）
        if args.episode_voting:
            print("Running episode-level voting evaluation...")
            epi_metrics = evaluate_model_episode_voting(model, model_type, ds, loader, args.device)
            ecm = epi_metrics["episode_cm"].tolist()
            print("Episode-level Result:")
            print(f"Episodes: {epi_metrics['episodes']}")
            print(f"Episode Accuracy: {epi_metrics['episode_accuracy']:.4f}")
            print("Episode Confusion Matrix (rows=true, cols=pred):")
            print(f"[[{ecm[0][0]}, {ecm[0][1]}],\n [{ecm[1][0]}, {ecm[1][1]}]]")

        results.append((str(p), model_type, metrics["accuracy"]))

    # 汇总
    if results:
        print("\nSummary:")
        for path, mtype, acc in results:
            print(f"- {mtype:12s} {acc:.4f}  <- {path}")


if __name__ == "__main__":
    main()
