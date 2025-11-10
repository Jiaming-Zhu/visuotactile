#!/usr/bin/env python
"""
使用 Transformer 做海绵 vs 木块的序列二分类：
  - 数据：两个数据集根目录（sponge / woodblock），从每个 episode 的 parquet 内按窗口截取序列；
  - 模型：TransformerClassifier（含 CLS token）；
  - 目标：二分类（0/1）。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from learn_PyBullet.datasets.parquet_seq_dataset import ParquetSeqDataset, SourceSpec
from learn_PyBullet.models.transformer_classifier import (
    TransformerClassifier,
    TransformerClassifierConfig,
)


def parse_args():
    p = argparse.ArgumentParser(description="Transformer sequence classifier: sponge vs woodblock")
    # --- 数据集参数 ---
    p.add_argument("--sponge_dir", type=str, default="data/so101_press/sponge_20251107_201409", help="海绵数据集的根目录")
    p.add_argument("--woodblock_dir", type=str, default="data/so101_press/woodBlock_20251107_204040", help="木块数据集的根目录")
    # --- 模型输入参数 ---
    p.add_argument("--state_keys", nargs="*", type=str, default=[
        "observation.position",
        "observation.velocity",
        "observation.load",
        "observation.current",
    ], help="从数据集中选择用作模型输入的状态")
    p.add_argument("--context", type=int, default=16, help="输入序列的上下文长度 (T)")
    # --- 训练超参数 ---
    p.add_argument("--batch_size", type=int, default=128, help="训练时的批量大小 (B)")
    p.add_argument("--epochs", type=int, default=5, help="训练的总轮数")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    # --- 其他参数 ---
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    p.add_argument("--save_path", type=str, default="outputs/press_transformer_classifier.pt", help="最佳模型的保存路径")
    return p.parse_args()


def read_features(dataset_dir: Path) -> Dict:
    """从数据集的 meta/info.json 文件中读取特征信息"""
    with open(dataset_dir / "meta/info.json") as f:
        info = json.load(f)
    return info["features"]


def compute_input_dim(state_keys: List[str], features: Dict[str, Dict]) -> int:
    """根据选择的 state_keys 和特征信息计算模型的输入维度 D"""
    dim = 0
    for k in state_keys:
        if k in features:
            # 累加每个状态的维度
            dim += int(features[k]["shape"][0])
    return dim


def collate(batch: List[dict]):
    """
    自定义的 collate 函数，用于将 DataLoader 中的样本列表打包成一个批次。
    - 将多个样本的 'x' 和 'label' 堆叠成一个批次张量。
    """
    if len(batch) == 0:
        return {"x": torch.empty(0), "y": torch.empty(0, dtype=torch.long)}
    # x 的形状是 (B, T, D)
    x = torch.stack([b["x"] for b in batch], dim=0)  # (B, T, D)
    # y 的形状是 (B,)
    y = torch.stack([b["label"] for b in batch], dim=0)
    return {"x": x, "y": y}


def main():
    args = parse_args()

    # --- 1. 验证并准备数据集路径 ---
    sponge_root = Path(args.sponge_dir)
    wood_root = Path(args.woodblock_dir)
    # 确保数据集元信息文件存在
    if not (sponge_root / "meta/info.json").exists():
        raise FileNotFoundError(f"{sponge_root}/meta/info.json not found")
    if not (wood_root / "meta/info.json").exists():
        raise FileNotFoundError(f"{wood_root}/meta/info.json not found")

    # --- 2. 计算模型输入维度 ---
    features = read_features(sponge_root)
    input_dim = compute_input_dim(args.state_keys, features)
    print(f"模型输入维度 (D): {input_dim}")

    # --- 3. 创建数据集和数据加载器 ---
    dataset = ParquetSeqDataset(
        sources=[
            SourceSpec(root=sponge_root, label=0),  # 海绵数据集，标签为 0
            SourceSpec(root=wood_root, label=1),    # 木块数据集，标签为 1
        ],
        state_keys=args.state_keys,
        context_len=args.context,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 每个 epoch 开始时打乱数据
        num_workers=2,
        pin_memory=args.device != "cpu",  # 如果使用 GPU，开启内存锁定加速数据转移
        collate_fn=collate,
    )
    print(f"数据集样本总数: {len(dataset)}")

    # --- 4. 初始化模型、损失函数和优化器 ---
    cfg = TransformerClassifierConfig(
        input_dim=input_dim,
        num_classes=2,  # 二分类：海绵 vs 木块
        context_len=args.context,
    )
    model = TransformerClassifier(cfg).to(args.device)
    # 交叉熵损失，适用于分类任务
    criterion = nn.CrossEntropyLoss()
    # AdamW 优化器，加入权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # --- 5. 训练循环 ---
    best_acc = 0.0
    print(f"开始在 {args.device} 上训练...")
    for epoch in range(args.epochs):
        model.train()  # 设置为训练模式
        total, correct, running_loss = 0, 0, 0.0

        for batch in loader:
            # 如果批次为空则跳过
            if batch["x"].numel() == 0:
                continue

            # 将数据移动到指定设备
            x = batch["x"].to(args.device)
            y = batch["y"].to(args.device)

            # 前向传播
            logits = model(x)

            # 计算损失
            loss = criterion(logits, y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()

            # --- 统计损失和准确率 ---
            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()

        # 计算当前 epoch 的平均损失和准确率
        acc = correct / total if total > 0 else 0.0
        avg_loss = running_loss / max(1, total)
        print(f"epoch {epoch+1}/{args.epochs} loss={avg_loss:.4f} acc={acc:.4f}")

        # --- 6. 保存最佳模型 ---
        if acc > best_acc:
            best_acc = acc
            out = Path(args.save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            # 保存模型权重、配置、使用的状态键和标签映射
            torch.save({
                "state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "state_keys": args.state_keys,
                "label_map": {0: "sponge", 1: "woodblock"},
            }, out)
            print(f"已保存最佳模型到 {out} (acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
