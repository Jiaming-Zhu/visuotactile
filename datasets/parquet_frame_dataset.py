from __future__ import annotations

"""
ParquetFrameDataset：
    直接使用 pyarrow 读取 LeRobot v2.1 数据集中的 parquet 文件，构建帧级样本。
    目的：绕过 datasets/HF 缓存权限与版本问题，保持轻量与可控。

设计：
    - sources: 多个数据源（根目录 + 标签），例如 sponge 与 woodblock
    - 按文件级建立索引（episode_*.parquet），并预读各文件行数，形成全局长度
    - __getitem__ 根据全局 idx 定位到具体文件与行，仅读取该行需要的列
"""

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


@dataclass
class SourceSpec:
    """单个数据源描述：根目录 + 标签（类别 id）"""
    root: Path
    label: int


class ParquetFrameDataset(Dataset):
    """轻量帧级数据集，按需从 parquet 文件逐行读取。"""

    def __init__(self, sources: Sequence[SourceSpec], state_keys: Sequence[str], action_key: str = None):
        self.sources = list(sources)
        self.state_keys = list(state_keys)
        self.action_key = action_key

        # 建立“文件 -> 标签”的索引列表
        self.files: List[Tuple[int, Path]] = []  # (label, path)
        for src in self.sources:
            pattern = str(src.root / 'data' / 'chunk-*' / 'episode_*.parquet')
            for f in sorted(glob.glob(pattern)):
                self.files.append((src.label, Path(f)))

        # 预计算每个文件的样本数（行数）
        self.file_lengths: List[int] = []
        for _, f in self.files:
            pf = pq.ParquetFile(f)
            self.file_lengths.append(pf.metadata.num_rows)

        # 前缀和便于 O(logN) 查找全局 idx 所属文件
        self.cum_lengths = np.cumsum([0] + self.file_lengths)

    def __len__(self) -> int:
        return int(self.cum_lengths[-1])

    def _locate(self, idx: int) -> Tuple[int, int]:
        # 返回 (文件索引, 行索引)，使用前缀和二分定位
        file_idx = int(np.searchsorted(self.cum_lengths, idx, side='right') - 1)
        row_idx = int(idx - self.cum_lengths[file_idx])
        return file_idx, row_idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, row_idx = self._locate(idx)
        label, fpath = self.files[file_idx]

        cols = list(self.state_keys)
        if self.action_key:
            cols.append(self.action_key)

        table = pq.read_table(fpath, columns=cols)
        row = table.slice(row_idx, 1).to_pydict()

        parts = []
        for k in self.state_keys:
            arr = row[k][0]  # 固定长度 list[float]
            parts.append(torch.tensor(arr, dtype=torch.float32))
        x = torch.cat(parts, dim=0)

        item: Dict[str, torch.Tensor] = {"x": x, "label": torch.tensor(label, dtype=torch.long)}
        if self.action_key:
            item["action"] = torch.tensor(row[self.action_key][0], dtype=torch.float32)
        return item
