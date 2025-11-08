"""
ParquetSeqDataset：基于 parquet 的时序分类数据集。

- 输入：在每个 episode_*.parquet 内按滑动窗口截取长度为 T 的序列；
        对窗口内每一帧，按给定 state_keys 拼接成单步向量，再堆叠为 (T, D)。
- 输出：字典 { "x": (T, D) float32, "label": int64 }。

注意：读取为懒执行，每次 __getitem__ 只读取一个文件的一小段行，简洁易用。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import glob

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


@dataclass
class SourceSpec:
    root: Path
    label: int


class ParquetSeqDataset(Dataset):
    def __init__(self, sources: Sequence[SourceSpec], state_keys: Sequence[str], context_len: int = 16):
        self.sources = list(sources)
        self.state_keys = list(state_keys)
        self.T = int(context_len)

        # 收集所有 episode parquet 文件
        self.files: List[Tuple[int, Path]] = []  # (label, path)
        for src in self.sources:
            for f in sorted(glob.glob(str(src.root / 'data' / 'chunk-*' / 'episode_*.parquet'))):
                self.files.append((src.label, Path(f)))

        # 每个文件的行数与可用窗口数（N - T + 1）
        self.file_rows: List[int] = []
        self.file_win: List[int] = []
        for _, fpath in self.files:
            pf = pq.ParquetFile(fpath)
            n = pf.metadata.num_rows
            self.file_rows.append(n)
            self.file_win.append(max(0, n - self.T + 1))

        # 前缀和：用于快速定位全局窗口索引属于哪个文件
        self.cum_win = np.cumsum([0] + self.file_win)

    def __len__(self) -> int:
        return int(self.cum_win[-1])

    def _locate(self, idx: int) -> Tuple[int, int]:
        # 返回 (文件索引, 窗口起点行号)
        file_idx = int(np.searchsorted(self.cum_win, idx, side='right') - 1)
        start = int(idx - self.cum_win[file_idx])
        return file_idx, start

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, start = self._locate(idx)
        label, fpath = self.files[file_idx]

        cols = list(self.state_keys)
        table = pq.read_table(fpath, columns=cols)
        # 取 T 行窗口
        window = table.slice(start, self.T).to_pydict()

        # 堆叠为 (T, D)
        steps = []
        for t in range(self.T):
            parts = []
            for k in self.state_keys:
                parts.append(torch.tensor(window[k][t], dtype=torch.float32))  # (d_k,)
            steps.append(torch.cat(parts, dim=0))  # (D,)
        x = torch.stack(steps, dim=0)  # (T, D)

        return {"x": x, "label": torch.tensor(label, dtype=torch.long)}

