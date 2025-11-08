#!/usr/bin/env python
"""
帧级二分类训练脚本：区分海绵（sponge）与木块（woodblock）。

特点：
- 直接用 pyarrow 读取 parquet，避免 datasets/HF 缓存权限问题；
- 使用 position/velocity/load/current 等低维状态拼接为输入向量；
- 模型为轻量 MLP 分类器，训练速度快，适合基线与验证流程。
"""

import argparse
import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

# 确保将仓库根目录加入 sys.path，避免直接运行子目录脚本时包导入失败
sys.path.append(str(Path(__file__).resolve().parents[1]))

from learn_PyBullet.models.mlp_classifier import MLPClassifier, MLPConfig
from learn_PyBullet.datasets.parquet_frame_dataset import ParquetFrameDataset, SourceSpec


def parse_args():
    """命令行参数定义"""
    p = argparse.ArgumentParser(description="Train a frame-level classifier: sponge vs woodblock")
    p.add_argument("--sponge_dir", type=str, required=False,
                   default="data/so101_press/sponge_20251107_201409")
    p.add_argument("--woodblock_dir", type=str, required=False,
                   default="data/so101_press/woodBlock_20251107_204040")
    p.add_argument("--state_keys", nargs="*", type=str, default=[
        "observation.position",
        "observation.velocity",
        "observation.load",
        "observation.current",
    ])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_path", type=str, default="outputs/press_classifier.pt")
    return p.parse_args()


def read_features(dataset_dir: Path) -> Dict:
    """读取 meta/info.json 的 features 字段（用于推断维度）"""
    with open(dataset_dir / "meta/info.json") as f:
        info = json.load(f)
    return info["features"]


def compute_input_dim(state_keys: List[str], features: Dict[str, Dict]) -> int:
    """根据给定状态键，计算拼接后的输入维度"""
    dim = 0
    for k in state_keys:
        if k not in features:
            continue
        shape = features[k]["shape"]
        dim += int(shape[0])
    return dim


def collate(batch: List[dict], state_keys: List[str]) -> Dict[str, torch.Tensor]:
    """将样本打包为批次张量（x: [B, D], y: [B]）"""
    if len(batch) == 0:
        return {"x": torch.empty(0), "y": torch.empty(0, dtype=torch.long)}
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["label"] for b in batch], dim=0)
    return {"x": x, "y": y}


@dataclass
class Split:
    train: Dataset
    val: Dataset


def train_val_split(ds: Dataset, val_ratio: float, seed: int) -> Split:
    """随机划分训练/验证集（样本级别切分）"""
    n = len(ds)
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    n_val = max(1, int(n * val_ratio))
    val_idx = idxs[:n_val]
    tr_idx = idxs[n_val:]
    return Split(train=Subset(ds, tr_idx), val=Subset(ds, val_idx))


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """计算分类准确率"""
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def main():
    """主流程：构建数据、模型，训练并验证，保存最佳模型"""
    args = parse_args()
    # 使用工作区本地 HF 缓存目录，避免权限问题
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path("outputs/hf_cache").resolve()))
    torch.manual_seed(args.seed)

    sponge_dir = Path(args.sponge_dir)
    wood_dir = Path(args.woodblock_dir)
    if not (sponge_dir / "meta/info.json").exists():
        raise FileNotFoundError(f"{sponge_dir}/meta/info.json not found")
    if not (wood_dir / "meta/info.json").exists():
        raise FileNotFoundError(f"{wood_dir}/meta/info.json not found")

    # 从任一数据集推断输入维度（假设两者 shape 一致）
    feat = read_features(sponge_dir)
    input_dim = compute_input_dim(args.state_keys, feat)

    # 构建帧级数据集（直接读取 parquet）
    full = ParquetFrameDataset(
        sources=[
            SourceSpec(root=sponge_dir, label=0),
            SourceSpec(root=wood_dir, label=1),
        ],
        state_keys=args.state_keys,
    )
    split = train_val_split(full, args.val_ratio, args.seed)

    collate_fn = lambda b: collate(b, args.state_keys)
    train_loader = DataLoader(
        split.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=args.device != "cpu",
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        split.val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=args.device != "cpu",
        collate_fn=collate_fn,
    )

    # 模型与优化器
    cfg = MLPConfig(input_dim=input_dim, num_classes=2)
    model = MLPClassifier(cfg).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for batch in train_loader:
            if batch["x"].numel() == 0:
                continue
            x = batch["x"].to(args.device)
            y = batch["y"].to(args.device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
        avg_loss = total_loss / max(1, n_samples)

        # 验证评估
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch["x"].numel() == 0:
                    continue
                x = batch["x"].to(args.device)
                y = batch["y"].to(args.device)
                logits = model(x)
                val_correct += (logits.argmax(dim=-1) == y).sum().item()
                val_total += y.numel()
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out = Path(args.save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "state_keys": args.state_keys,
                "label_map": {0: "sponge", 1: "woodblock"},
            }, out)
            print(f"Saved best model to {out} (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
