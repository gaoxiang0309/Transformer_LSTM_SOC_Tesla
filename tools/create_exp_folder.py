"""实验目录管理，对应示例项目tools/create_exp_folder.py。

创建run/train/exp、exp1等互不覆盖的目录，并建立weights和cache子目录。
本文件只负责目录组织，不加载数据、不训练模型。
"""

from pathlib import Path


def create_exp_folder(root):
    """寻找未使用的exp编号，新建weights/cache子目录并返回实验路径。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for i in range(100000):
        path = root / ("exp" if i == 0 else f"exp{i}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        (path / "weights").mkdir()
        (path / "cache").mkdir()
        return path
    raise RuntimeError("无法创建新实验目录")
