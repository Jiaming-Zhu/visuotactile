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
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

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


def build_dataloader(model_type: str, sponge_dir: Path, wood_dir: Path, state_keys: List[str], batch_size: int, device: str, context_len: int | None, features: Dict) -> DataLoader:
    if model_type == "mlp":
        ds = FrameDS(
            sources=[FrameSrc(sponge_dir, 0), FrameSrc(wood_dir, 1)],
            state_keys=state_keys,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=device != "cpu", collate_fn=collate_frame)
    else:
        T = int(context_len) if context_len is not None else 16
        ds = SeqDS(
            sources=[SeqSrc(sponge_dir, 0), SeqSrc(wood_dir, 1)],
            state_keys=state_keys,
            context_len=T,
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=device != "cpu", collate_fn=collate_seq)


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
    with torch.no_grad():
        for batch in loader:
            if batch["x"].numel() == 0:
                continue
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    if len(all_preds) == 0:
        return {"accuracy": 0.0, "cm": torch.zeros((2, 2), dtype=torch.long), "n": 0}

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = (preds == labels).float().mean().item()
    cm = confusion_matrix(preds, labels, num_classes=2)
    return {"accuracy": acc, "cm": cm, "n": labels.numel()}


def main():
    args = parse_args()
    sponge_dir = Path(args.sponge_dir)
    wood_dir = Path(args.woodblock_dir)
    if not (sponge_dir / "meta/info.json").exists() or not (wood_dir / "meta/info.json").exists():
        raise FileNotFoundError("Please provide valid sponge_dir and woodblock_dir containing meta/info.json")

    features = read_features(sponge_dir)

    results = []
    for mpath in args.models:
        p = Path(mpath)
        if not p.exists():
            print(f"[Skip] {p} not found")
            continue
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

        loader = build_dataloader(model_type, sponge_dir, wood_dir, state_keys, args.batch_size, args.device, context_len, features)
        model = build_model(model_type, cfg, input_dim, num_classes, args.device)
        model.load_state_dict(ckpt["state_dict"], strict=True)

        metrics = evaluate_model(model, loader, args.device)
        cm = metrics["cm"].tolist()
        print("\nModel:", p)
        print("Type:", model_type)
        print(f"Samples: {metrics['n']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Confusion Matrix (rows=true, cols=pred):")
        print(f"[[{cm[0][0]}, {cm[0][1]}],\n [{cm[1][0]}, {cm[1][1]}]]")
        results.append((str(p), model_type, metrics["accuracy"]))

    # 汇总
    if results:
        print("\nSummary:")
        for path, mtype, acc in results:
            print(f"- {mtype:12s} {acc:.4f}  <- {path}")


if __name__ == "__main__":
    main()

