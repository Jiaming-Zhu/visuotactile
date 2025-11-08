"""
Transformer 二分类模型（时序）：
  - 输入：时间窗口内的低维状态序列，形状 (B, T, D)
  - 输出：两类 logits（如 0=sponge, 1=woodblock）

设计要点：
  - 使用可学习的 [CLS] token 聚合全局信息；
  - 标准正弦位置编码 + TransformerEncoder；
  - 仅处理低维状态（不含图像）。
"""

from dataclasses import dataclass
import math
import torch
from torch import nn, Tensor


@dataclass
class TransformerClassifierConfig:
    """模型配置（超参数）"""
    input_dim: int          # 单步状态维度（多键拼接后的维度）
    num_classes: int = 2
    context_len: int = 16   # 时间窗口长度 T
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1


class PositionalEncoding(nn.Module):
    """正弦位置编码，为序列引入位置信息"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10_000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """时序 Transformer 二分类器（使用 [CLS] token 聚合）"""

    def __init__(self, cfg: TransformerClassifierConfig):
        super().__init__()
        self.cfg = cfg

        # 输入线性映射到 d_model
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        # 可学习的 CLS token（用于全局聚合）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        # 位置编码
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )

    def forward(self, x_seq: Tensor) -> Tensor:
        """前向：
        参数：x_seq (B, T, D)
        返回：logits (B, C)
        """
        B, T, _ = x_seq.shape
        x = self.input_proj(x_seq)  # (B, T, d)
        # 拼接 CLS token 到序列前端
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)
        x = torch.cat([cls, x], dim=1)  # (B, 1+T, d)
        x = self.pos_enc(x)
        z = self.encoder(x)  # (B, 1+T, d)
        cls_out = z[:, 0, :]  # 取 CLS
        return self.head(cls_out)

