"""
基于 LSTM 的时序二分类模型：
  - 输入：时间窗口内的低维状态序列，形状 (B, T, D)
  - 输出：两类 logits（0/1）

设计要点：
  - 使用多层 LSTM 编码序列；
  - 取最后一步的隐藏状态（或池化）作为全局表示；
  - 仅处理低维状态（不含图像）。
"""

from dataclasses import dataclass
import torch
from torch import nn, Tensor


@dataclass
class LSTMClassifierConfig:
    """模型配置（超参数）"""
    input_dim: int          # 单步状态维度（多键拼接后的维度）
    num_classes: int = 2
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False


class LSTMClassifier(nn.Module):
    """时序 LSTM 二分类器"""

    def __init__(self, cfg: LSTMClassifierConfig):
        super().__init__()
        self.cfg = cfg

        # LSTM 主体
        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )

        out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(out_dim, cfg.num_classes),
        )

    def forward(self, x_seq: Tensor) -> Tensor:
        """前向：
        参数：x_seq (B, T, D)
        返回：logits (B, C)
        """
        # 取最后时间步的隐藏状态作为序列表示
        out, (h_n, c_n) = self.lstm(x_seq)  # out: (B, T, H[*2])
        last = out[:, -1, :]                # (B, H[*2])
        return self.head(last)

