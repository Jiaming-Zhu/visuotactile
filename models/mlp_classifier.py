"""
简单的帧级 MLP 分类器：
    - 输入：单帧低维状态向量（多键拼接）
    - 输出：类别 logits（例如海绵 sponge vs 木块 woodblock）
适合作为基线，训练快、实现简单。
"""

from dataclasses import dataclass

import torch
from torch import nn, Tensor


@dataclass
class MLPConfig:
    """MLP 配置参数"""
    input_dim: int
    num_classes: int = 2
    hidden_dim: int = 256
    hidden_layers: int = 2
    dropout: float = 0.1


class MLPClassifier(nn.Module):
    """多层感知机，用于帧级分类。"""
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        layers = []
        dim = cfg.input_dim
        # 对输入做 LayerNorm，有助于稳定训练
        self.in_norm = nn.LayerNorm(dim)
        for _ in range(cfg.hidden_layers):
            layers += [nn.Linear(dim, cfg.hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout)]
            dim = cfg.hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dim, cfg.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x 形状：(B, D)
        x = self.in_norm(x)
        z = self.backbone(x)
        return self.head(z)
