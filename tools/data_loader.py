"""数据加载与时间窗口工具，对应示例项目tools/data_loader.py。

Sequence保存一段数据及其来源；load_sequences校验并读取准备好的NPZ。
WindowDataset将全局样本编号映射到某段内的窗口，按需切片而不预存全部窗口。
窗口始终位于所属段内部，标签与输入窗口最后一个点对齐。
"""
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path):
    """计算文件内容校验值，用于发现数据或权重版本变化。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Sequence:
    # x为[N,3]或[N,4]，y为[N]；offset是本段第一点在源文件中的行号。
    """保存一段输入、标签及来源信息；x/y长度相同，offset表示源行号偏移。"""
    name: str
    x: np.ndarray
    y: np.ndarray
    offset: int = 0
    source: str = ""


def load_sequences(cfg, split):
    """读取指定split的准备结果，逐文件核对哈希；smoke时截取少量序列和点。"""
    root = Path(cfg.processed_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = [r for r in manifest["records"] if r["split"] == split]
    if cfg.smoke:
        records = records[: cfg.smoke_sequences]
    sequences = []
    for record in records:
        # 兼容 Windows 和 Linux 路径
        relative_path = record["file"].replace("\\", "/")
        path = root / relative_path

        if sha256(path) != record["sha256"]:
            raise ValueError(f"数据校验失败：{path}")
        with np.load(path, allow_pickle=False) as d:
            # copy后数组不再依赖已关闭的NPZ文件句柄。
            x = d["X"].copy()
            y = d["Y"].copy()
        if cfg.smoke:
            x = x[: cfg.smoke_points]
            y = y[: cfg.smoke_points]
        sequences.append(Sequence(record["case"], x, y, record["start"], record["source"]))
    return sequences


class WindowDataset(Dataset):
    """在各段内部按需构造重叠窗口，避免预先复制出全部窗口数组。"""

    def __init__(self, sequences, length, stride=1, feature_indices=None):
        """保存窗口参数，可选地只返回指定输入列。"""
        if length < 1 or stride < 1:
            raise ValueError("窗口和步长必须为正")
        self.sequences = sequences
        self.length = length
        self.stride = stride
        self.feature_indices = None if feature_indices is None else tuple(feature_indices)
        if self.feature_indices is not None:
            if not self.feature_indices:
                raise ValueError("feature_indices不能为空")
            if any(index < 0 for index in self.feature_indices):
                raise ValueError("feature_indices不能包含负数")
            if sequences and any(
                index >= sequence.x.shape[1]
                for sequence in sequences
                for index in self.feature_indices
            ):
                raise ValueError("feature_indices超出序列输入列范围")
        self.cumulative = [0]
        for sequence in sequences:
            # N个点、长度L、步长stride，可构造floor((N-L)/stride)+1个窗口。
            window_count = max(0, (len(sequence.y) - length) // stride + 1)
            self.cumulative.append(self.cumulative[-1] + window_count)

    def __len__(self):
        """返回全部序列可生成的窗口总数，而不是原始时间点总数。"""
        return self.cumulative[-1]

    def locate(self, index):
        """把全局窗口index转换为所属序列编号，以及该序列内的目标end位置。"""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        # cumulative是各段窗口数的前缀和。例如[0,5,10]对应两段各5个窗口。
        # 二分查找把全局窗口编号映射到所属段；不会从相邻段补取历史。
        seq = bisect_right(self.cumulative, index) - 1
        start = (index - self.cumulative[seq]) * self.stride
        # end是目标点，且包含在输入窗口中；它不是“下一个时刻”。
        return seq, start + self.length - 1

    def __getitem__(self, index):
        """返回输入窗口、末端SOC标签、序列编号和段内目标行号。"""
        sequence_index, end = self.locate(index)
        sequence = self.sequences[sequence_index]
        # Python切片右端不包含，因此使用end+1取到目标时刻这一行。
        values = sequence.x[end - self.length + 1 : end + 1]
        if self.feature_indices is not None:
            values = values[:, self.feature_indices]
        window = torch.from_numpy(values.copy())
        target = torch.tensor(sequence.y[end], dtype=torch.float32)
        return window, target, sequence_index, end
