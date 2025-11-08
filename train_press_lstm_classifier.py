#!/usr/bin/env python
"""
使用 LSTM 做海绵 vs 木块的序列二分类：
  - 数据：两个数据集根目录（sponge / woodblock），在 episode parquet 内按窗口截取序列；
  - 模型：LSTMClassifier；
  - 目标：二分类（0/1）。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# 确保仓库根目录在 sys.path 中
sys.path.append(str(Path(__file__).resolve().parents[1]))

from learn_PyBullet.datasets.parquet_seq_dataset import ParquetSeqDataset, SourceSpec
from learn_PyBullet.models.lstm_classifier import LSTMClassifier, LSTMClassifierConfig


def parse_args():
    p = argparse.ArgumentParser(description="LSTM sequence classifier: sponge vs woodblock")
    p.add_argument("--sponge_dir", type=str, default="data/so101_press/sponge_20251107_201409")
    p.add_argument("--woodblock_dir", type=str, default="data/so101_press/woodBlock_20251107_204040")
    p.add_argument("--state_keys", nargs="*", type=str, default=[
        "observation.position",
        "observation.velocity",
        "observation.load",
        "observation.current",
    ])
    p.add_argument("--context", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", type=str, default="outputs/press_lstm_classifier.pt")
    # LSTM 结构参数
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--bidirectional", action="store_true")
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


def collate(batch: List[dict]):
    if len(batch) == 0:
        return {"x": torch.empty(0), "y": torch.empty(0, dtype=torch.long)}
    x = torch.stack([b["x"] for b in batch], dim=0)  # (B, T, D)
    y = torch.stack([b["label"] for b in batch], dim=0)
    return {"x": x, "y": y}


def main():
    args = parse_args()

    sponge_root = Path(args.sponge_dir)
    wood_root = Path(args.woodblock_dir)
    if not (sponge_root / "meta/info.json").exists():
        raise FileNotFoundError(f"{sponge_root}/meta/info.json not found")
    if not (wood_root / "meta/info.json").exists():
        raise FileNotFoundError(f"{wood_root}/meta/info.json not found")

    features = read_features(sponge_root)
    input_dim = compute_input_dim(args.state_keys, features)

    dataset = ParquetSeqDataset(
        sources=[
            SourceSpec(root=sponge_root, label=0),
            SourceSpec(root=wood_root, label=1),
        ],
        state_keys=args.state_keys,
        context_len=args.context,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=args.device != "cpu",
        collate_fn=collate,
    )

    cfg = LSTMClassifierConfig(
        input_dim=input_dim,
        num_classes=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )
    model = LSTMClassifier(cfg).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for batch in loader:
            if batch["x"].numel() == 0:
                continue
            x = batch["x"].to(args.device, non_blocking=True)
            y = batch["y"].to(args.device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()

        acc = correct / total if total > 0 else 0.0
        avg_loss = running_loss / max(1, total)
        print(f"epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            out = Path(args.save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "state_keys": args.state_keys,
                "label_map": {0: "sponge", 1: "woodblock"},
            }, out)
            print(f"Saved best model to {out} (acc={best_acc:.4f})")


if __name__ == "__main__":
    main()

