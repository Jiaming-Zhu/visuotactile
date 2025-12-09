#!/usr/bin/env python3
"""
Vanilla Transformer 多任务分类器（仅使用感知信号，无视觉）

任务：同时预测 4 个属性
  - class: 物体类别 (10 类)
  - mass: 质量 (very_low, low, medium, high)
  - stiffness: 硬度 (very_soft, soft, medium, rigid)
  - material: 材料 (sponge, foam, wood, hollow_container, filled_container)

输入: 触觉时间序列 (6 个关节: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
  - joint_position_profile: (T, 6) 关节位置
  - joint_load_profile: (T, 6) 关节负载
  - joint_current_profile: (T, 6) 关节电流
  - joint_velocity_profile: (T, 6) 关节速度
  => 总特征维度: 6*4 = 24

作者: Martina
日期: 2025-11-30
"""

import os
import json
import pickle
import random
import logging
import argparse
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# =============================================================================
# Logging Setup
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# 物理属性定义
# =============================================================================
PHYSICAL_PROPERTIES = {
    'WoodBlock_Native':         {'mass': 'medium',   'stiffness': 'rigid',     'material': 'wood'},
    'WoodBlock_Foil':           {'mass': 'medium',   'stiffness': 'rigid',     'material': 'wood'},
    'WoodBlock_Red':            {'mass': 'medium',   'stiffness': 'rigid',     'material': 'wood'},
    'YogaBrick_Native':         {'mass': 'low',      'stiffness': 'medium',    'material': 'foam'},
    'YogaBrick_Blue':           {'mass': 'low',      'stiffness': 'medium',    'material': 'foam'},
    'CardboardBox_Hollow':      {'mass': 'very_low', 'stiffness': 'soft',      'material': 'hollow_container'},
    'CardboardBox_SpongeFilled':{'mass': 'very_low', 'stiffness': 'soft',      'material': 'filled_container'},
    'CardboardBox_RockFilled':  {'mass': 'high',     'stiffness': 'rigid',     'material': 'filled_container'},
    'CardboardBox_RockFilled_Red':{'mass': 'high',   'stiffness': 'rigid',     'material': 'filled_container'},
    'Sponge_Blue':              {'mass': 'very_low', 'stiffness': 'very_soft', 'material': 'sponge'},
}

MASS_TO_IDX = {'very_low': 0, 'low': 1, 'medium': 2, 'high': 3}
STIFFNESS_TO_IDX = {'very_soft': 0, 'soft': 1, 'medium': 2, 'rigid': 3}
MATERIAL_TO_IDX = {'sponge': 0, 'foam': 1, 'wood': 2, 'hollow_container': 3, 'filled_container': 4}

IDX_TO_MASS = {v: k for k, v in MASS_TO_IDX.items()}
IDX_TO_STIFFNESS = {v: k for k, v in STIFFNESS_TO_IDX.items()}
IDX_TO_MATERIAL = {v: k for k, v in MATERIAL_TO_IDX.items()}


# =============================================================================
# 模型配置
# =============================================================================
@dataclass
class MultiTaskTransformerConfig:
    """多任务 Transformer 配置"""
    input_dim: int = 24          # 单步特征维度 (6 joints × 4 channels)
    context_len: int = 256       # 时间窗口长度
    d_model: int = 128           # Transformer 隐藏维度
    n_heads: int = 4             # 多头注意力头数
    n_layers: int = 4            # Transformer 层数
    dim_feedforward: int = 256   # FFN 中间维度
    dropout: float = 0.1
    # 各任务类别数
    num_classes: int = 10
    num_mass: int = 4
    num_stiffness: int = 4
    num_material: int = 5


# =============================================================================
# 模型定义
# =============================================================================
class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiTaskTransformer(nn.Module):
    """
    多任务 Vanilla Transformer
    使用 [CLS] token 聚合全局信息，输出多个分类头
    """
    
    def __init__(self, cfg: MultiTaskTransformerConfig):
        super().__init__()
        self.cfg = cfg
        
        # 输入投影层
        self.input_proj = nn.Linear(cfg.input_dim, cfg.d_model)
        
        # 可学习的 CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 位置编码
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.dropout, max_len=cfg.context_len + 1)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        
        # Layer Norm (用于 CLS 输出)
        self.norm = nn.LayerNorm(cfg.d_model)
        
        # 多任务分类头
        self.head_class = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.num_classes),
        )
        self.head_mass = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_mass),
        )
        self.head_stiffness = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_stiffness),
        )
        self.head_material = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_material),
        )
    
    def forward(self, x_seq: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        前向传播
        
        Args:
            x_seq: (B, T, D) 输入序列
            mask: (B, T) 可选的 padding mask (True = 忽略)
        
        Returns:
            dict: 各任务的 logits
        """
        B, T, _ = x_seq.shape
        
        # 投影到 d_model
        x = self.input_proj(x_seq)  # (B, T, d_model)
        
        # 拼接 CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, 1+T, d_model)
        
        # 位置编码
        x = self.pos_enc(x)
        
        # 处理 mask (如果有的话，需要扩展为 CLS + T)
        if mask is not None:
            # 添加 CLS 位置 (不 mask)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
        
        # Transformer Encoder
        z = self.encoder(x, src_key_padding_mask=mask)  # (B, 1+T, d_model)
        
        # 取 CLS token 输出
        cls_out = self.norm(z[:, 0, :])  # (B, d_model)
        
        # 多任务输出
        return {
            'class': self.head_class(cls_out),
            'mass': self.head_mass(cls_out),
            'stiffness': self.head_stiffness(cls_out),
            'material': self.head_material(cls_out),
        }


# =============================================================================
# 数据集
# =============================================================================
class TactileSequenceDataset(Dataset):
    """
    触觉序列数据集（滑动窗口模式）
    从 Plaintextdataset 加载 tactile_data.pkl 文件
    使用滑动窗口生成多个样本，充分利用时序数据
    
    重要：必须先按 Episode 划分 Train/Val，再对各自的 Episode 切窗口！
    避免数据泄露（同一 Episode 的重叠窗口进入不同集合）
    """
    
    def __init__(
        self,
        root_dir: str,
        context_len: int = 256,
        stride: int = 64,
        normalize: bool = True,
        augment: bool = False,
        episode_indices: Optional[List[int]] = None,  # 指定使用哪些 episode
        stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # 外部传入的 (mean, std)
    ):
        """
        Args:
            root_dir: 数据集根目录 (Plaintextdataset)
            context_len: 时间窗口长度
            stride: 滑动窗口步长
            normalize: 是否标准化
            augment: 是否启用数据增强
            episode_indices: 指定使用哪些 episode 的索引（用于 train/val 划分）
            stats: 外部传入的标准化统计量 (mean, std)，验证集应使用训练集的统计量
        """
        self.root_dir = Path(root_dir)
        self.context_len = context_len
        self.stride = stride
        self.normalize = normalize
        self.augment = augment
        self.episode_indices = episode_indices  # None 表示使用全部
        
        self.all_episodes: List[Tuple[Path, int]] = []  # 全部 (pkl_path, class_idx)
        self.episodes: List[Tuple[Path, int]] = []  # 实际使用的 episodes
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        
        # 滑动窗口索引: (episode_idx, window_start_idx)
        self.window_indices: List[Tuple[int, int]] = []
        
        # 缓存每个 episode 的长度
        self.episode_lengths: List[int] = []
        
        # 用于标准化的统计量
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        
        # 缓存所有 episode 的特征数据（避免重复读取磁盘）
        self.cached_features: List[np.ndarray] = []
        
        self._find_classes_and_samples()
        self._select_episodes()  # 根据 episode_indices 选择 episode
        self._preload_all_data()  # 预加载所选 episode 数据到内存
        self._build_window_indices()
        
        # 标准化统计量
        if stats is not None:
            self.mean, self.std = stats
            logger.info(f"Using provided stats: mean shape={self.mean.shape}")
        elif self.normalize:
            self._compute_stats()
    
    def _find_classes_and_samples(self):
        """扫描数据集目录结构"""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root_dir}")
        
        # 获取类别
        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        logger.info(f"Found {len(self.classes)} classes: {self.classes}")
        
        # 收集所有 episode
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            cls_idx = self.class_to_idx[cls_name]
            
            for ep_dir in sorted(cls_dir.iterdir()):  # 排序保证可复现性
                if not ep_dir.is_dir():
                    continue
                pkl_path = ep_dir / "tactile_data.pkl"
                if pkl_path.exists():
                    self.all_episodes.append((pkl_path, cls_idx))
        
        logger.info(f"Found {len(self.all_episodes)} tactile episodes.")
    
    def _select_episodes(self):
        """根据 episode_indices 选择要使用的 episode"""
        if self.episode_indices is None:
            # 使用全部 episode
            self.episodes = self.all_episodes.copy()
            logger.info(f"Using all {len(self.episodes)} episodes")
        else:
            # 使用指定的 episode
            self.episodes = [self.all_episodes[i] for i in self.episode_indices]
            logger.info(f"Using {len(self.episodes)} / {len(self.all_episodes)} episodes (indices provided)")
    
    def _preload_all_data(self):
        """预加载所有 episode 数据到内存，避免训练时重复读取磁盘"""
        logger.info("Preloading all tactile data into memory...")
        
        self.cached_features = []
        total_memory = 0
        
        for pkl_path, _ in tqdm(self.episodes, desc="Preloading"):
            features = self._load_and_extract_features(pkl_path)
            self.cached_features.append(features)
            total_memory += features.nbytes
        
        logger.info(f"Preloaded {len(self.cached_features)} episodes")
        logger.info(f"Total memory usage: {total_memory / 1024 / 1024:.1f} MB")
    
    def _build_window_indices(self):
        """构建滑动窗口索引（使用已缓存的数据）"""
        logger.info("Building sliding window indices...")
        
        self.window_indices = []
        self.episode_lengths = []
        
        for ep_idx, features in enumerate(self.cached_features):
            T = features.shape[0]
            self.episode_lengths.append(T)
            
            # 计算该 episode 可以生成的窗口数
            if T >= self.context_len:
                num_windows = (T - self.context_len) // self.stride + 1
                for w_idx in range(num_windows):
                    start = w_idx * self.stride
                    self.window_indices.append((ep_idx, start))
            else:
                # 长度不足，只生成一个填充样本
                self.window_indices.append((ep_idx, 0))
        
        logger.info(f"Generated {len(self.window_indices)} sliding windows from {len(self.episodes)} episodes")
        logger.info(f"  Average windows per episode: {len(self.window_indices) / len(self.episodes):.1f}")
    
    def _compute_stats(self):
        """计算标准化统计量（使用已缓存的全部数据）"""
        logger.info("Computing normalization statistics from cached data...")
        
        # 直接使用所有缓存的数据计算统计量
        all_features = np.concatenate(self.cached_features, axis=0)  # (N, D)
        self.mean = all_features.mean(axis=0)
        self.std = all_features.std(axis=0) + 1e-8  # 防止除零
        
        logger.info(f"Computed stats from {all_features.shape[0]} samples: mean shape={self.mean.shape}")
    
    def get_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取标准化统计量，用于传递给验证集"""
        return self.mean, self.std
    
    @staticmethod
    def get_all_episode_indices(root_dir: str) -> Tuple[List[int], List[str], int]:
        """
        静态方法：扫描数据集，返回所有 episode 的索引列表
        用于在创建数据集之前进行 episode 级别的划分
        
        Returns:
            episode_indices: 所有 episode 的索引 [0, 1, 2, ...]
            classes: 类别名称列表
            num_episodes: episode 总数
        """
        root_path = Path(root_dir)
        classes = sorted([
            d.name for d in root_path.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        num_episodes = 0
        for cls_name in classes:
            cls_dir = root_path / cls_name
            for ep_dir in sorted(cls_dir.iterdir()):
                if not ep_dir.is_dir():
                    continue
                pkl_path = ep_dir / "tactile_data.pkl"
                if pkl_path.exists():
                    num_episodes += 1
        
        return list(range(num_episodes)), classes, num_episodes
    
    def _load_and_extract_features(self, pkl_path: Path) -> np.ndarray:
        """
        加载 pkl 文件并提取特征
        
        Returns:
            features: (T, D) 特征矩阵, D=24 (6 joints × 4 channels)
        """
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 提取各通道 (6 个关节: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
        joint_position = np.array(data['joint_position_profile'])     # (T, 6)
        joint_load = np.array(data['joint_load_profile'])             # (T, 6)
        joint_current = np.array(data['joint_current_profile'])       # (T, 6)
        joint_velocity = np.array(data['joint_velocity_profile'])     # (T, 6)
        
        # 拼接为 (T, 24)
        features = np.concatenate([
            joint_position,
            joint_load,
            joint_current,
            joint_velocity,
        ], axis=1).astype(np.float32)
        
        return features
    
    def _extract_window(self, features: np.ndarray, start: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        从序列中提取固定长度窗口
        
        Args:
            features: (T, D) 原始特征
            start: 窗口起始位置
        
        Returns:
            window: (context_len, D) 固定长度窗口
            mask: (context_len,) padding mask (True=padding/ignore, False=valid)
        """
        T = features.shape[0]
        mask = np.zeros(self.context_len, dtype=bool)  # False = valid
        
        if T >= self.context_len:
            # 从指定位置提取窗口
            end = start + self.context_len
            if end > T:
                # 如果超出边界，从末尾往前取
                start = T - self.context_len
                end = T
            window = features[start:end]
        else:
            # 长度不足，填充
            pad_len = self.context_len - T
            window = np.pad(features, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
            mask[T:] = True  # Mark padding as ignored
        
        return window, mask
    
    def _augment_window(self, window: np.ndarray) -> np.ndarray:
        """
        窗口数据增强（不改变长度）
        
        Args:
            window: (context_len, D) 固定长度窗口
        
        Returns:
            augmented: (context_len, D) 增强后的窗口
        """
        # 1. 随机噪声
        if random.random() < 0.5:
            noise_scale = 0.02 * (np.abs(window).mean() + 1e-8)
            noise = np.random.randn(*window.shape) * noise_scale
            window = window + noise.astype(np.float32)
        
        # 2. 随机幅度缩放
        if random.random() < 0.3:
            scale = random.uniform(0.95, 1.05)
            window = window * scale
        
        # 3. 随机通道 dropout (模拟传感器故障)
        if random.random() < 0.1:
            num_channels = window.shape[1]
            drop_idx = random.randint(0, num_channels - 1)
            window[:, drop_idx] = 0
        
        return window
    
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        ep_idx, window_start = self.window_indices[idx]
        _, class_idx = self.episodes[ep_idx]
        
        # 从缓存获取特征（无需读取磁盘）
        features = self.cached_features[ep_idx]
        
        # 提取窗口和 mask
        window, mask = self._extract_window(features, window_start)
        
        # 数据增强（仅对窗口内数据）
        if self.augment:
            window = self._augment_window(window)
        
        # 标准化
        if self.normalize and self.mean is not None:
            window = (window - self.mean) / self.std
        
        # 获取物理属性标签
        class_name = self.classes[class_idx]
        props = PHYSICAL_PROPERTIES.get(class_name, {})
        mass_idx = MASS_TO_IDX.get(props.get('mass', 'medium'), 2)
        stiffness_idx = STIFFNESS_TO_IDX.get(props.get('stiffness', 'medium'), 2)
        material_idx = MATERIAL_TO_IDX.get(props.get('material', 'wood'), 2)
        
        return (
            torch.from_numpy(window.astype(np.float32)),
            torch.from_numpy(mask),
            class_idx,
            mass_idx,
            stiffness_idx,
            material_idx,
            ep_idx,  # 返回 episode 索引，用于 episode 级别聚合
        )


# =============================================================================
# 实时可视化
# =============================================================================
class MultiTaskLivePlotter:
    """实时绘制多任务训练曲线"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc_class': [], 'val_acc_class': [],
            'train_acc_mass': [], 'val_acc_mass': [],
            'train_acc_stiffness': [], 'val_acc_stiffness': [],
            'train_acc_material': [], 'val_acc_material': [],
        }
        
        plt.ion()
        
        # 2x3 布局
        self.fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Tactile Transformer Multi-Task Training', fontsize=14)
        
        # Loss
        self.ax_loss = axes[0, 0]
        self.line_train_loss, = self.ax_loss.plot([], [], 'b-o', label='Train', markersize=3)
        self.line_val_loss, = self.ax_loss.plot([], [], 'r-o', label='Val', markersize=3)
        self.ax_loss.set_title('Total Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.legend(loc='upper right')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Class Accuracy
        self.ax_class = axes[0, 1]
        self.line_train_class, = self.ax_class.plot([], [], 'b-o', label='Train', markersize=3)
        self.line_val_class, = self.ax_class.plot([], [], 'r-o', label='Val', markersize=3)
        self.ax_class.set_title('Class Accuracy')
        self.ax_class.set_xlabel('Epoch')
        self.ax_class.set_ylim(0, 1.05)
        self.ax_class.legend(loc='lower right')
        self.ax_class.grid(True, alpha=0.3)
        
        # Mass Accuracy
        self.ax_mass = axes[0, 2]
        self.line_train_mass, = self.ax_mass.plot([], [], 'b-o', label='Train', markersize=3)
        self.line_val_mass, = self.ax_mass.plot([], [], 'r-o', label='Val', markersize=3)
        self.ax_mass.set_title('Mass Accuracy')
        self.ax_mass.set_xlabel('Epoch')
        self.ax_mass.set_ylim(0, 1.05)
        self.ax_mass.legend(loc='lower right')
        self.ax_mass.grid(True, alpha=0.3)
        
        # Stiffness Accuracy
        self.ax_stiffness = axes[1, 0]
        self.line_train_stiffness, = self.ax_stiffness.plot([], [], 'b-o', label='Train', markersize=3)
        self.line_val_stiffness, = self.ax_stiffness.plot([], [], 'r-o', label='Val', markersize=3)
        self.ax_stiffness.set_title('Stiffness Accuracy')
        self.ax_stiffness.set_xlabel('Epoch')
        self.ax_stiffness.set_ylim(0, 1.05)
        self.ax_stiffness.legend(loc='lower right')
        self.ax_stiffness.grid(True, alpha=0.3)
        
        # Material Accuracy
        self.ax_material = axes[1, 1]
        self.line_train_material, = self.ax_material.plot([], [], 'b-o', label='Train', markersize=3)
        self.line_val_material, = self.ax_material.plot([], [], 'r-o', label='Val', markersize=3)
        self.ax_material.set_title('Material Accuracy')
        self.ax_material.set_xlabel('Epoch')
        self.ax_material.set_ylim(0, 1.05)
        self.ax_material.legend(loc='lower right')
        self.ax_material.grid(True, alpha=0.3)
        
        # Val Accuracy 对比
        self.ax_compare = axes[1, 2]
        self.line_cmp_class, = self.ax_compare.plot([], [], 'o-', label='Class', markersize=3, color='#1f77b4')
        self.line_cmp_mass, = self.ax_compare.plot([], [], 's-', label='Mass', markersize=3, color='#ff7f0e')
        self.line_cmp_stiffness, = self.ax_compare.plot([], [], '^-', label='Stiffness', markersize=3, color='#2ca02c')
        self.line_cmp_material, = self.ax_compare.plot([], [], 'd-', label='Material', markersize=3, color='#d62728')
        self.ax_compare.set_title('Val Accuracy Comparison')
        self.ax_compare.set_xlabel('Epoch')
        self.ax_compare.set_ylim(0, 1.05)
        self.ax_compare.legend(loc='lower right')
        self.ax_compare.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(block=False)
    
    def update(self, **kwargs):
        """更新所有图表"""
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)
        
        epochs = list(range(1, len(self.history['train_loss']) + 1))
        
        # 更新 Loss
        self.line_train_loss.set_data(epochs, self.history['train_loss'])
        self.line_val_loss.set_data(epochs, self.history['val_loss'])
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()
        
        # 更新 Class Acc
        self.line_train_class.set_data(epochs, self.history['train_acc_class'])
        self.line_val_class.set_data(epochs, self.history['val_acc_class'])
        self.ax_class.relim()
        self.ax_class.autoscale_view()
        self.ax_class.set_ylim(0, 1.05)
        
        # 更新 Mass Acc
        self.line_train_mass.set_data(epochs, self.history['train_acc_mass'])
        self.line_val_mass.set_data(epochs, self.history['val_acc_mass'])
        self.ax_mass.relim()
        self.ax_mass.autoscale_view()
        self.ax_mass.set_ylim(0, 1.05)
        
        # 更新 Stiffness Acc
        self.line_train_stiffness.set_data(epochs, self.history['train_acc_stiffness'])
        self.line_val_stiffness.set_data(epochs, self.history['val_acc_stiffness'])
        self.ax_stiffness.relim()
        self.ax_stiffness.autoscale_view()
        self.ax_stiffness.set_ylim(0, 1.05)
        
        # 更新 Material Acc
        self.line_train_material.set_data(epochs, self.history['train_acc_material'])
        self.line_val_material.set_data(epochs, self.history['val_acc_material'])
        self.ax_material.relim()
        self.ax_material.autoscale_view()
        self.ax_material.set_ylim(0, 1.05)
        
        # 更新对比图
        self.line_cmp_class.set_data(epochs, self.history['val_acc_class'])
        self.line_cmp_mass.set_data(epochs, self.history['val_acc_mass'])
        self.line_cmp_stiffness.set_data(epochs, self.history['val_acc_stiffness'])
        self.line_cmp_material.set_data(epochs, self.history['val_acc_material'])
        self.ax_compare.relim()
        self.ax_compare.autoscale_view()
        self.ax_compare.set_ylim(0, 1.05)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.05)
    
    def save(self):
        if self.output_dir:
            save_path = self.output_dir / 'tactile_transformer_training.png'
            self.fig.savefig(save_path, dpi=150)
            logger.info(f"Saved training plot to {save_path}")
    
    def close(self):
        plt.ioff()
        plt.close(self.fig)


# =============================================================================
# 训练循环
# =============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """训练一个 epoch"""
    model.train()
    
    running_loss = 0.0
    running_corrects = {'class': 0, 'mass': 0, 'stiffness': 0, 'material': 0}
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        features, mask, labels_class, labels_mass, labels_stiffness, labels_material, _ = batch  # 忽略 ep_idx
        features = features.to(device)
        mask = mask.to(device)
        labels_class = labels_class.to(device)
        labels_mass = labels_mass.to(device)
        labels_stiffness = labels_stiffness.to(device)
        labels_material = labels_material.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(features, mask=mask)
        
        # 计算各任务损失
        loss_class = criterion(outputs['class'], labels_class)
        loss_mass = criterion(outputs['mass'], labels_mass)
        loss_stiffness = criterion(outputs['stiffness'], labels_stiffness)
        loss_material = criterion(outputs['material'], labels_material)
        
        # 总损失（加权）
        loss = loss_class + 0.5 * (loss_mass + loss_stiffness + loss_material)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # 统计
        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        
        running_corrects['class'] += (outputs['class'].argmax(1) == labels_class).sum().item()
        running_corrects['mass'] += (outputs['mass'].argmax(1) == labels_mass).sum().item()
        running_corrects['stiffness'] += (outputs['stiffness'].argmax(1) == labels_stiffness).sum().item()
        running_corrects['material'] += (outputs['material'].argmax(1) == labels_material).sum().item()
        total_samples += batch_size
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / total_samples
    epoch_acc = {k: v / total_samples for k, v in running_corrects.items()}
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """
    评估模型
    
    返回:
        epoch_loss: 平均损失
        epoch_acc: 窗口级别准确率 (window-level)
        all_preds: 所有窗口的预测
        all_labels: 所有窗口的标签
        episode_acc: Episode 级别准确率 (episode-level, 多数投票)
    """
    model.eval()
    
    running_loss = 0.0
    running_corrects = {'class': 0, 'mass': 0, 'stiffness': 0, 'material': 0}
    total_samples = 0
    
    all_preds = {'class': [], 'mass': [], 'stiffness': [], 'material': []}
    all_labels = {'class': [], 'mass': [], 'stiffness': [], 'material': []}
    all_ep_indices = []  # 记录每个窗口属于哪个 episode
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        features, mask, labels_class, labels_mass, labels_stiffness, labels_material, ep_indices = batch
        features = features.to(device)
        mask = mask.to(device)
        labels_class = labels_class.to(device)
        labels_mass = labels_mass.to(device)
        labels_stiffness = labels_stiffness.to(device)
        labels_material = labels_material.to(device)
        
        outputs = model(features, mask=mask)
        
        loss_class = criterion(outputs['class'], labels_class)
        loss_mass = criterion(outputs['mass'], labels_mass)
        loss_stiffness = criterion(outputs['stiffness'], labels_stiffness)
        loss_material = criterion(outputs['material'], labels_material)
        loss = loss_class + 0.5 * (loss_mass + loss_stiffness + loss_material)
        
        batch_size = features.size(0)
        running_loss += loss.item() * batch_size
        
        preds_class = outputs['class'].argmax(1)
        preds_mass = outputs['mass'].argmax(1)
        preds_stiffness = outputs['stiffness'].argmax(1)
        preds_material = outputs['material'].argmax(1)
        
        running_corrects['class'] += (preds_class == labels_class).sum().item()
        running_corrects['mass'] += (preds_mass == labels_mass).sum().item()
        running_corrects['stiffness'] += (preds_stiffness == labels_stiffness).sum().item()
        running_corrects['material'] += (preds_material == labels_material).sum().item()
        total_samples += batch_size
        
        all_preds['class'].extend(preds_class.cpu().numpy())
        all_preds['mass'].extend(preds_mass.cpu().numpy())
        all_preds['stiffness'].extend(preds_stiffness.cpu().numpy())
        all_preds['material'].extend(preds_material.cpu().numpy())
        
        all_labels['class'].extend(labels_class.cpu().numpy())
        all_labels['mass'].extend(labels_mass.cpu().numpy())
        all_labels['stiffness'].extend(labels_stiffness.cpu().numpy())
        all_labels['material'].extend(labels_material.cpu().numpy())
        
        all_ep_indices.extend(ep_indices.numpy())
    
    epoch_loss = running_loss / total_samples
    window_acc = {k: v / total_samples for k, v in running_corrects.items()}
    
    # =========================================================================
    # Episode 级别准确率（多数投票）
    # =========================================================================
    episode_acc = compute_episode_accuracy(all_preds, all_labels, all_ep_indices)
    
    return epoch_loss, window_acc, all_preds, all_labels, episode_acc


def compute_episode_accuracy(all_preds, all_labels, all_ep_indices):
    """
    计算 Episode 级别的准确率（多数投票聚合）
    
    对于每个 episode 的所有窗口预测，取出现次数最多的类别作为该 episode 的最终预测。
    """
    from collections import defaultdict, Counter
    
    # 按 episode 分组收集预测和标签
    episode_preds = defaultdict(lambda: {'class': [], 'mass': [], 'stiffness': [], 'material': []})
    episode_labels = {}  # 每个 episode 的真实标签（所有窗口标签相同）
    
    for i, ep_idx in enumerate(all_ep_indices):
        for task in ['class', 'mass', 'stiffness', 'material']:
            episode_preds[ep_idx][task].append(all_preds[task][i])
        
        # 记录标签（每个 episode 的标签是固定的）
        if ep_idx not in episode_labels:
            episode_labels[ep_idx] = {
                'class': all_labels['class'][i],
                'mass': all_labels['mass'][i],
                'stiffness': all_labels['stiffness'][i],
                'material': all_labels['material'][i],
            }
    
    # 多数投票得到每个 episode 的最终预测
    episode_corrects = {'class': 0, 'mass': 0, 'stiffness': 0, 'material': 0}
    num_episodes = len(episode_preds)
    
    for ep_idx, preds in episode_preds.items():
        true_labels = episode_labels[ep_idx]
        
        for task in ['class', 'mass', 'stiffness', 'material']:
            # 多数投票
            majority_pred = Counter(preds[task]).most_common(1)[0][0]
            if majority_pred == true_labels[task]:
                episode_corrects[task] += 1
    
    episode_acc = {k: v / num_episodes for k, v in episode_corrects.items()}
    
    return episode_acc


def print_classification_report(all_preds, all_labels, class_names):
    """打印分类报告"""
    try:
        from sklearn.metrics import classification_report
        
        # Class
        logger.info("\n📊 Classification Report - CLASS:")
        logger.info(classification_report(
            all_labels['class'], all_preds['class'],
            target_names=class_names, digits=4, zero_division=0
        ))
        
        # Mass
        mass_names = list(MASS_TO_IDX.keys())
        logger.info("\n📊 Classification Report - MASS:")
        logger.info(classification_report(
            all_labels['mass'], all_preds['mass'],
            target_names=mass_names, digits=4, zero_division=0
        ))
        
        # Stiffness
        stiffness_names = list(STIFFNESS_TO_IDX.keys())
        logger.info("\n📊 Classification Report - STIFFNESS:")
        logger.info(classification_report(
            all_labels['stiffness'], all_preds['stiffness'],
            target_names=stiffness_names, digits=4, zero_division=0
        ))
        
        # Material
        material_names = list(MATERIAL_TO_IDX.keys())
        logger.info("\n📊 Classification Report - MATERIAL:")
        logger.info(classification_report(
            all_labels['material'], all_preds['material'],
            target_names=material_names, digits=4, zero_division=0
        ))
        
    except ImportError:
        logger.warning("scikit-learn not installed, skipping classification report.")


def plot_confusion_matrices(all_preds, all_labels, class_names, output_dir: Path):
    """
    绘制多任务混淆矩阵热力图
    
    Args:
        all_preds: 预测结果字典
        all_labels: 真实标签字典
        class_names: 类别名称列表
        output_dir: 输出目录
    """
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
    except ImportError:
        logger.warning("sklearn or seaborn not installed, skipping confusion matrix plot.")
        return
    
    # 准备各任务的标签名
    task_configs = {
        'class': {
            'labels': class_names,
            'title': 'Class Confusion Matrix',
            'figsize': (14, 12),
            'fontsize': 8,
        },
        'mass': {
            'labels': list(MASS_TO_IDX.keys()),
            'title': 'Mass Confusion Matrix',
            'figsize': (8, 6),
            'fontsize': 12,
        },
        'stiffness': {
            'labels': list(STIFFNESS_TO_IDX.keys()),
            'title': 'Stiffness Confusion Matrix',
            'figsize': (8, 6),
            'fontsize': 12,
        },
        'material': {
            'labels': list(MATERIAL_TO_IDX.keys()),
            'title': 'Material Confusion Matrix',
            'figsize': (10, 8),
            'fontsize': 10,
        },
    }
    
    # 创建 2x2 子图布局
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle('Tactile Transformer - Confusion Matrices', fontsize=16, fontweight='bold')
    
    task_list = ['class', 'mass', 'stiffness', 'material']
    
    for idx, task in enumerate(task_list):
        ax = axes[idx // 2, idx % 2]
        config = task_configs[task]
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_labels[task], all_preds[task])
        
        # 计算归一化混淆矩阵（按行归一化，即按真实标签归一化）
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        
        # 绘制热力图
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=config['labels'],
            yticklabels=config['labels'],
            ax=ax,
            cbar=True,
            square=True,
            annot_kws={'fontsize': config['fontsize']},
        )
        
        ax.set_title(config['title'], fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        
        # 旋转标签以便阅读
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=config['fontsize'])
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=config['fontsize'])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    save_path = output_dir / 'confusion_matrices.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"📊 Saved confusion matrices to {save_path}")
    
    plt.close(fig)
    
    # 同时生成单独的 Class 混淆矩阵（因为类别较多，单独保存一个大图）
    fig_class, ax_class = plt.subplots(figsize=(16, 14))
    
    cm_class = confusion_matrix(all_labels['class'], all_preds['class'])
    cm_class_norm = cm_class.astype('float') / (cm_class.sum(axis=1, keepdims=True) + 1e-8)
    
    # 同时显示数量和比例
    annot_labels = []
    for i in range(cm_class.shape[0]):
        row = []
        for j in range(cm_class.shape[1]):
            count = cm_class[i, j]
            pct = cm_class_norm[i, j]
            row.append(f'{count}\n({pct:.1%})')
        annot_labels.append(row)
    annot_labels = np.array(annot_labels)
    
    sns.heatmap(
        cm_class_norm,
        annot=annot_labels,
        fmt='',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax_class,
        cbar=True,
        square=True,
        annot_kws={'fontsize': 9},
    )
    
    ax_class.set_title('Class Confusion Matrix (Tactile Transformer)', fontsize=16, fontweight='bold')
    ax_class.set_xlabel('Predicted', fontsize=14)
    ax_class.set_ylabel('True', fontsize=14)
    ax_class.set_xticklabels(ax_class.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax_class.set_yticklabels(ax_class.get_yticklabels(), rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    save_path_class = output_dir / 'confusion_matrix_class.png'
    fig_class.savefig(save_path_class, dpi=150, bbox_inches='tight')
    logger.info(f"📊 Saved class confusion matrix to {save_path_class}")
    
    plt.close(fig_class)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    output_dir: Path,
    class_names: List[str],
    resume: bool = False,
    checkpoint_path: Optional[Path] = None,
):
    """完整训练流程"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Cosine annealing scheduler -> OneCycleLR (with Warmup)
    # Transformer 训练通常需要 Warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=25.0,  # init_lr = max_lr / 25
        final_div_factor=10000.0,  # min_lr = init_lr / 10000
    )
    
    # 实时可视化
    plotter = MultiTaskLivePlotter(output_dir=output_dir)
    
    best_acc = 0.0
    start_epoch = 0
    save_path = output_dir / 'tactile_transformer_best.pth'
    
    # 断点续训
    if resume:
        ckpt_path = checkpoint_path if checkpoint_path else save_path
        if ckpt_path.exists():
            logger.info(f"📂 Loading checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            
            # 恢复模型权重
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 恢复优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 恢复调度器状态（如果有）
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复起始 epoch 和最佳准确率
            start_epoch = checkpoint['epoch'] + 1
            # 优先使用 episode_acc（新格式），兼容旧 checkpoint
            if 'episode_acc' in checkpoint:
                best_acc = checkpoint['episode_acc']['class']
            else:
                best_acc = checkpoint['val_acc']['class']
            
            # 恢复训练历史（用于可视化）
            if 'history' in checkpoint:
                plotter.history = checkpoint['history']
            
            logger.info(f"✅ Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")
        else:
            logger.warning(f"⚠️ Checkpoint not found at {ckpt_path}, starting from scratch")
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        logger.info("-" * 40)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"  Class: {train_acc['class']:.4f} | Mass: {train_acc['mass']:.4f} | "
                   f"Stiffness: {train_acc['stiffness']:.4f} | Material: {train_acc['material']:.4f}")
        
        # 验证
        val_loss, val_acc, all_preds, all_labels, episode_acc = evaluate(model, val_loader, criterion, device)
        
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"  [Window]  Class: {val_acc['class']:.4f} | Mass: {val_acc['mass']:.4f} | "
                   f"Stiffness: {val_acc['stiffness']:.4f} | Material: {val_acc['material']:.4f}")
        logger.info(f"  [Episode] Class: {episode_acc['class']:.4f} | Mass: {episode_acc['mass']:.4f} | "
                   f"Stiffness: {episode_acc['stiffness']:.4f} | Material: {episode_acc['material']:.4f}")
        
        # 更新可视化
        plotter.update(
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc_class=train_acc['class'],
            val_acc_class=val_acc['class'],
            train_acc_mass=train_acc['mass'],
            val_acc_mass=val_acc['mass'],
            train_acc_stiffness=train_acc['stiffness'],
            val_acc_stiffness=val_acc['stiffness'],
            train_acc_material=train_acc['material'],
            val_acc_material=val_acc['material'],
        )
        
        # 保存最佳模型 (基于 episode-level class accuracy，更可靠)
        if episode_acc['class'] > best_acc:
            best_acc = episode_acc['class']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'episode_acc': episode_acc,
                'val_loss': val_loss,
                'history': plotter.history,  # 保存训练历史用于可视化恢复
            }, save_path)
            logger.info(f"✅ Saved best model with Episode Class Acc: {best_acc:.4f}")
        
        # 最后一个 epoch 打印详细报告
        if epoch == num_epochs - 1:
            logger.info("\n" + "=" * 60)
            logger.info("Final Multi-Task Performance Summary:")
            logger.info("-" * 60)
            logger.info("Window-Level Accuracy (每个窗口独立计算):")
            logger.info(f"  Class:     {val_acc['class']:.4f}")
            logger.info(f"  Mass:      {val_acc['mass']:.4f}")
            logger.info(f"  Stiffness: {val_acc['stiffness']:.4f}")
            logger.info(f"  Material:  {val_acc['material']:.4f}")
            logger.info("-" * 60)
            logger.info("Episode-Level Accuracy (多数投票聚合，更可靠):")
            logger.info(f"  Class:     {episode_acc['class']:.4f}")
            logger.info(f"  Mass:      {episode_acc['mass']:.4f}")
            logger.info(f"  Stiffness: {episode_acc['stiffness']:.4f}")
            logger.info(f"  Material:  {episode_acc['material']:.4f}")
            logger.info("=" * 60)
            print_classification_report(all_preds, all_labels, class_names)
            
            # 绘制混淆矩阵
            plot_confusion_matrices(all_preds, all_labels, class_names, output_dir)
    
    logger.info(f"\nBest validation Class Accuracy: {best_acc:.4f}")
    
    plotter.save()
    plotter.close()
    
    return model


# =============================================================================
# 主函数
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train a Vanilla Transformer for multi-task tactile classification"
    )
    parser.add_argument('--dataset_root', type=str, default='Plaintextdataset',
                        help='Path to Plaintextdataset root')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--output_dir', type=str, default='learn_PyBullet/outputs',
                        help='Directory to save outputs')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--context_len', type=int, default=256, help='Sequence length')
    parser.add_argument('--d_model', type=int, default=128, help='Transformer hidden dim')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--no-augment', action='store_true', help='Disable data augmentation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (default: output_dir/tactile_transformer_best.pth)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate the model and generate confusion matrices (no training)')
    parser.add_argument('--export-onnx', action='store_true', help='Export the best model to ONNX format')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 路径
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # ==========================================================================
    # ✅ 正确的划分方式：先按 Episode 划分，再分别切窗口
    # 避免数据泄露（同一 Episode 的重叠窗口进入不同集合）
    # ==========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("⚠️  IMPORTANT: Splitting at EPISODE level (not window level)")
    logger.info("    This prevents data leakage from overlapping windows!")
    logger.info("=" * 60 + "\n")
    
    # Step 1: 获取所有 episode 索引
    all_ep_indices, classes, num_episodes = TactileSequenceDataset.get_all_episode_indices(str(dataset_root))
    
    # Step 2: 按 Episode 划分 Train/Val (80/20)
    num_train_eps = int(0.8 * num_episodes)
    
    # 使用固定随机种子打乱 episode 顺序
    rng = random.Random(args.seed)
    shuffled_indices = all_ep_indices.copy()
    rng.shuffle(shuffled_indices)
    
    train_ep_indices = sorted(shuffled_indices[:num_train_eps])
    val_ep_indices = sorted(shuffled_indices[num_train_eps:])
    
    logger.info(f"Episode split: {len(train_ep_indices)} train / {len(val_ep_indices)} val")
    
    # Step 3: 创建训练集（使用训练 episodes，计算标准化统计量）
    logger.info("\nLoading training dataset...")
    train_dataset = TactileSequenceDataset(
        root_dir=str(dataset_root),
        context_len=args.context_len,
        normalize=True,
        augment=not args.no_augment,
        episode_indices=train_ep_indices,
        stats=None,  # 训练集自己计算 stats
    )
    
    # Step 4: 创建验证集（使用验证 episodes，复用训练集的统计量）
    logger.info("\nLoading validation dataset...")
    val_dataset = TactileSequenceDataset(
        root_dir=str(dataset_root),
        context_len=args.context_len,
        normalize=True,
        augment=False,  # 验证集不增强
        episode_indices=val_ep_indices,
        stats=train_dataset.get_stats(),  # 使用训练集的 mean/std
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),  # 保持 worker 进程存活
        prefetch_factor=4 if args.num_workers > 0 else None,  # 预取更多批次
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    
    logger.info(f"\nTraining windows: {len(train_dataset)} (from {len(train_ep_indices)} episodes)")
    logger.info(f"Validation windows: {len(val_dataset)} (from {len(val_ep_indices)} episodes)")
    logger.info(f"Classes: {train_dataset.classes}")
    
    # 模型配置
    cfg = MultiTaskTransformerConfig(
        input_dim=24,  # 6 joints × 4 channels (position, load, current, velocity)
        context_len=args.context_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.d_model * 2,
        dropout=0.1,
        num_classes=len(train_dataset.classes),
        num_mass=len(MASS_TO_IDX),
        num_stiffness=len(STIFFNESS_TO_IDX),
        num_material=len(MATERIAL_TO_IDX),
    )
    
    logger.info(f"\nModel Configuration:")
    logger.info(f"  Input dim: {cfg.input_dim}")
    logger.info(f"  Context length: {cfg.context_len}")
    logger.info(f"  d_model: {cfg.d_model}")
    logger.info(f"  n_heads: {cfg.n_heads}")
    logger.info(f"  n_layers: {cfg.n_layers}")
    logger.info(f"  Tasks: class({cfg.num_classes}), mass({cfg.num_mass}), "
               f"stiffness({cfg.num_stiffness}), material({cfg.num_material})")
    
    model = MultiTaskTransformer(cfg).to(device)
    
    # 计算模型参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total trainable parameters: {num_params:,}")
    
    # 确定检查点路径
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / 'tactile_transformer_best.pth'
    
    # Eval-only 模式：仅加载权重并评估
    if args.eval_only:
        logger.info("\n📊 Eval-only mode: Loading checkpoint and generating confusion matrices...")
        
        if not checkpoint_path.exists():
            logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
            logger.error("Please train a model first or specify --checkpoint path")
            return
        
        # 加载模型权重
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ Loaded checkpoint from {checkpoint_path}")
        logger.info(f"   Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"   Checkpoint val_acc: {checkpoint.get('val_acc', 'N/A')}")
        
        # 评估
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, all_preds, all_labels, episode_acc = evaluate(model, val_loader, criterion, device)
        
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Results:")
        logger.info(f"  Val Loss: {val_loss:.4f}")
        logger.info("-" * 60)
        logger.info("Window-Level Accuracy (每个窗口独立计算):")
        logger.info(f"  Class:     {val_acc['class']:.4f}")
        logger.info(f"  Mass:      {val_acc['mass']:.4f}")
        logger.info(f"  Stiffness: {val_acc['stiffness']:.4f}")
        logger.info(f"  Material:  {val_acc['material']:.4f}")
        logger.info("-" * 60)
        logger.info("Episode-Level Accuracy (多数投票聚合，更可靠):")
        logger.info(f"  Class:     {episode_acc['class']:.4f}")
        logger.info(f"  Mass:      {episode_acc['mass']:.4f}")
        logger.info(f"  Stiffness: {episode_acc['stiffness']:.4f}")
        logger.info(f"  Material:  {episode_acc['material']:.4f}")
        logger.info("=" * 60)
        
        # 打印分类报告
        print_classification_report(all_preds, all_labels, train_dataset.classes)
        
        # 生成混淆矩阵
        plot_confusion_matrices(all_preds, all_labels, train_dataset.classes, output_dir)
        
        # Eval-only 模式下的 ONNX 导出
        if args.export_onnx:
            logger.info("\nExporting model to ONNX (Eval-only mode)...")
            onnx_path = output_dir / 'tactile_transformer.onnx'
            
            model.eval()
            # 创建 dummy input: (batch=1, seq_len, input_dim=24)
            dummy_input = torch.randn(1, args.context_len, 24, device=device)
            
            output_names = ['class_logits', 'mass_logits', 'stiffness_logits', 'material_logits']
            
            try:
                torch.onnx.export(
                    model, 
                    dummy_input, 
                    onnx_path, 
                    verbose=False,
                    input_names=['tactile_sequence'],
                    output_names=output_names,
                    dynamic_axes={'tactile_sequence': {0: 'batch_size'}}
                )
                logger.info(f"✅ Model exported to: {onnx_path.resolve()}")
            except Exception as e:
                logger.error(f"❌ Failed to export to ONNX: {e}")
        
        logger.info("\n✅ Evaluation complete!")
        return
    
    # 正常训练模式
    if args.resume:
        logger.info("\n🔄 Resume mode enabled")
    else:
        logger.info("\nStarting training...")
    
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_dir=output_dir,
        class_names=train_dataset.classes,
        resume=args.resume,
        checkpoint_path=checkpoint_path if args.resume else None,
    )
    
    # 保存类别名和配置
    class_names_path = output_dir / 'tactile_class_names.txt'
    with open(class_names_path, 'w') as f:
        for cls in train_dataset.classes:
            f.write(f"{cls}\n")
    logger.info(f"Saved class names to {class_names_path}")
    
    # 保存配置
    config_path = output_dir / 'tactile_transformer_config.json'
    config_dict = {
        'input_dim': cfg.input_dim,
        'context_len': cfg.context_len,
        'd_model': cfg.d_model,
        'n_heads': cfg.n_heads,
        'n_layers': cfg.n_layers,
        'dim_feedforward': cfg.dim_feedforward,
        'dropout': cfg.dropout,
        'num_classes': cfg.num_classes,
        'num_mass': cfg.num_mass,
        'num_stiffness': cfg.num_stiffness,
        'num_material': cfg.num_material,
        'classes': train_dataset.classes,
        'normalization': {
            'mean': train_dataset.mean.tolist(),
            'std': train_dataset.std.tolist(),
        }
    }
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved config to {config_path}")
    
    # Export to ONNX
    if args.export_onnx:
        logger.info("Exporting model to ONNX...")
        onnx_path = output_dir / 'tactile_transformer.onnx'
        
        # Load best weights
        if checkpoint_path.exists():
             try:
                 checkpoint = torch.load(checkpoint_path, map_location=device)
                 if 'model_state_dict' in checkpoint:
                     model.load_state_dict(checkpoint['model_state_dict'])
                 else:
                     model.load_state_dict(checkpoint) # In case it's just state dict
             except Exception as e:
                 logger.warning(f"Could not load checkpoint for export, using current weights: {e}")
        
        model.eval()
        # Input shape: (Batch, Context_Len, Input_Dim)
        dummy_input = torch.randn(1, cfg.context_len, cfg.input_dim, device=device)
        
        # Output order in forward: class, mass, stiffness, material
        output_names = ['class_logits', 'mass_logits', 'stiffness_logits', 'material_logits']
        
        try:
             # We only provide the first argument (x_seq), mask is optional and defaults to None
             torch.onnx.export(
                model, 
                dummy_input, 
                onnx_path, 
                verbose=False,
                input_names=['tactile_sequence'],
                output_names=output_names,
                dynamic_axes={'tactile_sequence': {0: 'batch_size'}}
            )
             logger.info(f"✅ Model exported to: {onnx_path.resolve()}")
        except Exception as e:
            logger.error(f"❌ Failed to export to ONNX: {e}")

    logger.info("\n✅ Training complete!")


if __name__ == '__main__':
    main()

