"""Tesla Model 3 2170 行车SOC数据准备：原始驱动循环 MAT 转换为本项目 NPZ。

本轮任务只做“行车状态”SOC（见 docs/窗口与数据说明.md）：
- 只保留 Drive Cycles 中的行驶循环（REORDERED / HWCUST / HWGRADE）；
- CC-CV 充电文件整体排除，不进入训练/验证/测试；
- 行车内部的正负电流、回馈、短停车全部保留，不按电流符号切子状态；
- 超过 max_parking_seconds 的长驻车静置段从序列中切掉，避免把“长停车”当成行车历史。
训练/验证/测试按电芯划分（默认 m80 / m448-N / m1000），三个集合不共享文件。
输入为 V、I、T（Voltage/Current/Battery_Temp_degC），并可拼接第4维安时积分
Ah = ∫I dt / 额定容量（由 config.use_ah_feature 控制，每段从0开始）。
SOC 为作者按容量插值得到的0～1参考值。
窗口按物理秒定义，点数由采样间隔换算（Tesla为1 Hz）。
归一化只在训练划分上拟合：每通道 p0.01–p99.99 截断后 min-max 到[0,1]（可改为 z-score）。
原始 MAT 文件不修改。
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from config import Config, load_config

# 六个测试温度目录名；每格包含 Drive Cycles 子目录。
TEMPERATURES = ("-20C", "-10C", "0C", "10C", "25C", "40C")
# 原始文件中的特征列顺序；写入NPZ时保持这个顺序，模型按V/I/T读取。
FEATURE_NAMES = ["Voltage", "Current", "Battery_Temp_degC"]
# 第4维安时积分特征的名称（由电流积分得到，不在原始MAT列中）。
AH_FEATURE_NAME = "Ah_integral"
# 判定“静置/驻车”的电流阈值（安培）。只用于切长驻车，不用于状态分类。
PARKING_CURRENT_A = 0.05
# 1 Hz重采样的最小间隔（秒）：时间差小于它的行视为重复突发，只保留第一个。
MIN_SAMPLE_INTERVAL_S = 0.5


def sha256(path):
    """计算文件内容的SHA-256，记录准备结果是否发生变化。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cycle_label(meas):
    """从meas结构中读取逐点循环标签，取第一个非空值作为序列名。"""
    # squeeze_me=True时，Cycle_Label可能是字符串、字符串数组或object数组。
    raw = getattr(meas, "Cycle_Label", "")
    values = np.asarray(raw).reshape(-1)
    for value in values:
        text = str(value).strip()
        if text:
            return text
    return "unknown"


def read_meas(path):
    """读取一个MAT文件的meas结构，返回[N,3]输入、[N]标签、循环标签和[N]秒时间。

    取Voltage/Current/Battery_Temp_degC/SOC四列，并把Time一并返回用于重采样；
    原始文件并非严格1 Hz，个别文件存在亚秒级重复突发，必须按Time清理。
    """
    from scipy.io import loadmat  # 只在准备阶段需要SciPy

    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "meas" not in data:
        raise ValueError(f"{path} 缺少meas结构")
    meas = data["meas"]
    columns = {}
    for name in FEATURE_NAMES + ["SOC", "Time"]:
        if not hasattr(meas, name):
            raise ValueError(f"{path} 缺少meas.{name}")
        columns[name] = np.asarray(getattr(meas, name), dtype=np.float64).reshape(-1)
    samples = min(len(values) for values in columns.values())
    # 三列依次为V、I、T；转置为[N,3]，标签为[N]，时间用于1 Hz重采样。
    x = np.column_stack([columns[name][:samples] for name in FEATURE_NAMES])
    y = columns["SOC"][:samples]
    time = columns["Time"][:samples]
    return x, y, cycle_label(meas), time


def is_charge(label, filename):
    """按循环标签或文件名判断是否为CC-CV充电文件。"""
    text = f"{label} {filename}".upper()
    return "CC_CV" in text or "CHARGE" in text


def filter_rows(x, y):
    """去掉非有限值和明显越界的采样点，返回清洗结果、删除行数和有效mask。"""
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    valid &= (x[:, 0] > 1.0) & (x[:, 0] < 6.0)  # 单体电压
    valid &= np.abs(x[:, 1]) < 50.0  # 电流
    valid &= (x[:, 2] > -60.0) & (x[:, 2] < 90.0)  # 温度
    valid &= (y > -0.1) & (y < 1.1)  # SOC可略越界，但不接受明显错误
    return x[valid], y[valid], int((~valid).sum()), valid


def split_long_parking(x, y, max_parking_seconds, sample_interval_s):
    """切掉超过阈值的连续静置段，返回保留区间和删除点数。

    只按电流绝对值接近0判断驻车，不按电流符号区分“充电/回馈”；
    区间为左闭右开，坐标相对于传入的x/y。
    """
    if max_parking_seconds <= 0:
        return [(0, len(y))], 0
    rest = np.abs(x[:, 1]) <= PARKING_CURRENT_A
    # 找连续静置段的起止：diff(0, rest, 0) 中 1->start, -1->end。
    edges = np.diff(np.concatenate(([0], rest.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    max_points = max_parking_seconds / sample_interval_s
    segments = []
    cursor = 0
    removed = 0
    for start, end in zip(starts, ends):
        if end - start > max_points:
            if start > cursor:
                segments.append((cursor, start))
            removed += int(end - start)
            cursor = int(end)
    if cursor < len(y):
        segments.append((cursor, len(y)))
    return segments, removed


def fit_normalizer(x_train, cfg):
    """只用训练数据拟合归一化参数，返回可写入manifest的字典。"""
    low = np.percentile(x_train, cfg.clip_percentile, axis=0)
    high = np.percentile(x_train, 100.0 - cfg.clip_percentile, axis=0)
    if cfg.use_ah_feature:
        # 安时积分用固定物理尺度（额定容量比例），避免测试段深放电/回馈被训练分位数截断。
        low[-1] = cfg.ah_norm_low
        high[-1] = cfg.ah_norm_high
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    # 常数通道的标准差置1，避免z-score除零。
    std[std < 1e-8] = 1.0
    return dict(
        method=cfg.normalization,
        percentile=cfg.clip_percentile,
        low=low.tolist(),
        high=high.tolist(),
        mean=mean.tolist(),
        std=std.tolist(),
    )


def apply_normalizer(x, normalizer):
    """按训练集参数转换输入，返回转换结果和每通道被截断的比例。"""
    low = np.asarray(normalizer["low"], dtype=np.float64)
    high = np.asarray(normalizer["high"], dtype=np.float64)
    if normalizer["method"] == "zscore":
        mean = np.asarray(normalizer["mean"], dtype=np.float64)
        std = np.asarray(normalizer["std"], dtype=np.float64)
        return (x - mean) / std, np.zeros(x.shape[1], dtype=np.float64)
    clipped_fraction = np.mean((x < low) | (x > high), axis=0)
    clipped = np.clip(x, low, high)
    return (clipped - low) / (high - low), clipped_fraction


def cell_files(root, cell, include_charges):
    """列出一个电芯在所有温度下的原始文件，返回(path, temperature, kind)列表。"""
    cell_root = root / cell
    if not cell_root.is_dir():
        raise FileNotFoundError(f"找不到电芯目录：{cell_root}")
    entries = []
    for temperature in TEMPERATURES:
        folder = cell_root / temperature / "Drive Cycles"
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到驱动循环目录：{folder}")
        # 文件名以顺序号开头；排序保证切分顺序稳定。
        for path in sorted(folder.glob("*.mat")):
            charge = is_charge("", path.name)
            if charge and not include_charges:
                # 本轮行车SOC：充电文件整体隔离。
                continue
            entries.append(
                dict(path=path, temperature=temperature, kind="charge" if charge else "drive")
            )
    return entries


def resample_mask(time, min_interval_s):
    """返回按Time重采样到约1 Hz的保留mask。

    原始文件并非严格1 Hz，个别文件存在亚秒级重复突发（例如5e-5 s间隔）。
    规则：保留第一个点；之后与上一个保留点的时间差>=min_interval_s才保留。
    这样只去掉重复/突发行，不改变真实时间轴。
    """
    keep = np.zeros(len(time), dtype=bool)
    last = None
    for i in range(len(time)):
        if last is None or time[i] - last >= min_interval_s:
            keep[i] = True
            last = time[i]
    return keep


def split_time_gaps(time, max_gap_seconds):
    """按真实时间差切开序列，返回左闭右开区间；避免跨越数据缺口构窗。"""
    if max_gap_seconds <= 0 or len(time) < 2:
        return [(0, len(time))]
    cuts = np.flatnonzero(np.diff(time) > max_gap_seconds) + 1
    bounds = [0] + cuts.tolist() + [len(time)]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def load_segments(entry, cfg):
    """读取一个原始文件，按时间重采样、缺口和长驻车切分，返回行驶段列表。"""
    x, y, label, time = read_meas(entry["path"])
    x, y, dropped, valid = filter_rows(x, y)
    time = time[: len(valid)][valid]
    # 先压掉亚秒级重复行，再按真实时间缺口切分；窗口不跨缺口、不跨长驻车。
    keep = resample_mask(time, MIN_SAMPLE_INTERVAL_S)
    resampled = int((~keep).sum())
    x, y, time = x[keep], y[keep], time[keep]
    original_index = np.flatnonzero(valid)[keep]

    parts = []
    for gap_start, gap_stop in split_time_gaps(time, cfg.max_gap_seconds):
        gap_x = x[gap_start:gap_stop]
        gap_y = y[gap_start:gap_stop]
        gap_time = time[gap_start:gap_stop]
        gap_index = original_index[gap_start:gap_stop]
        if entry["kind"] == "drive":
            segments, parking_removed = split_long_parking(
                gap_x, gap_y, cfg.max_parking_seconds, cfg.sample_interval_s
            )
        else:
            # 充电文件（仅当显式 include_charges=true 时才会走到这里）不做驻车切分。
            segments, parking_removed = [(0, len(gap_y))], 0
        for start, stop in segments:
            if stop - start < cfg.long_window:
                continue
            segment_x = gap_x[start:stop]
            segment_y = gap_y[start:stop]
            segment_time = gap_time[start:stop]
            if cfg.use_ah_feature:
                # 安时积分：∫I dt / 额定容量；用真实时间差，序列起点为0。
                # 电流符号沿用Tesla（正=充电、负=放电），不读取真实SOC。
                dt = np.diff(segment_time)
                steps = segment_x[:-1, 1] * dt / 3600.0 / cfg.capacity_ah
                ah = np.concatenate(([0.0], np.cumsum(steps)))
                segment_x = np.column_stack([segment_x, ah])
            parts.append(
                dict(
                    x=segment_x,
                    y=segment_y,
                    start=int(gap_index[start]),
                    label=label,
                    part=len(parts) + 1,
                    part_count=0,  # 全部段收集完后再填
                    dropped=dropped if not parts else 0,
                    parking_removed=parking_removed if not parts else 0,
                    resampled=resampled if not parts else 0,
                )
            )
    # 只有真正被切成多段时才在文件名里加 _p 后缀。
    for part in parts:
        part["part_count"] = len(parts)
    return parts


def prepare(cfg, output=None):
    """读取Tesla原始MAT，生成按电芯划分的行车NPZ与manifest，返回来源清单。"""
    root = Path(cfg.source_dir)
    out = Path(output) if output else Path(cfg.processed_dir)
    if (out / "manifest.json").exists():
        raise FileExistsError("已存在prepared manifest；请指定 --output，避免覆盖实验数据")

    splits = dict(
        train=list(cfg.train_cells),
        validation=list(cfg.validation_cells),
        test=list(cfg.test_cells),
    )
    feature_names = list(FEATURE_NAMES)
    if cfg.use_ah_feature:
        feature_names.append(AH_FEATURE_NAME)
    # 第一步：加载训练段并拟合归一化参数。
    train_parts = []
    pending = []  # 暂存训练段，拟合完成后再统一写出
    dropped_rows = 0
    parking_removed = 0
    resampled_rows = 0
    for cell in splits["train"]:
        for entry in cell_files(root, cell, cfg.include_charges):
            parts = load_segments(entry, cfg)
            for part in parts:
                dropped_rows += part["dropped"]
                parking_removed += part["parking_removed"]
                resampled_rows += part["resampled"]
                pending.append(
                    dict(
                        cell=cell,
                        temperature=entry["temperature"],
                        kind=entry["kind"],
                        source=entry["path"],
                        **part,
                    )
                )
                train_parts.append(part["x"].astype(np.float32))
    if not pending:
        raise ValueError("没有可用的训练序列")
    train_x = np.vstack(train_parts)
    normalizer = fit_normalizer(train_x, cfg)
    del train_parts, train_x
    print(
        "训练集归一化参数（p%.2f–p%.2f）：" % (cfg.clip_percentile, 100 - cfg.clip_percentile),
        {
            name: (round(lo, 4), round(hi, 4))
            for name, lo, hi in zip(
                feature_names, normalizer["low"], normalizer["high"]
            )
        },
        flush=True,
    )

    # 第二步：统一写文件；训练段已在内存，验证/测试逐文件处理以控制内存。
    records = []
    counts = {split: 0 for split in splits}
    clipped = {split: np.zeros(cfg.input_features) for split in splits}

    def save_sequence(split, cell, temperature, kind, source, part):
        x = part["x"]
        y = part["y"]
        x_scaled, clipped_fraction = apply_normalizer(x, normalizer)
        # 同一文件被长驻车切成多段时，用 _p1/_p2 区分；重名时再追加序号。
        base = f"{cell}_{temperature}_{part['label'].replace(' ', '_')}"
        if part["part_count"] > 1:
            base += f"_p{part['part']}"
        name = base
        index = 1
        while any(record["case"] == name for record in records):
            index += 1
            name = f"{base}_{index:02d}"
        path = out / split / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=x_scaled.astype("float32"), Y=y.astype("float32"))
        records.append(
            dict(
                split=split,
                case=name,
                file=(Path(split) / f"{name}.npz").as_posix(),
                source=str(source.relative_to(root)).replace("\\", "/"),
                start=int(part["start"]),
                stop=int(part["start"] + len(y)),
                samples=int(len(y)),
                boundary_status="source_file",
                state="driving" if kind == "drive" else "charging",
                cell=cell,
                temperature=temperature,
                cycle=part["label"],
                kind=kind,
                part=int(part["part"]),
                sha256=sha256(path),
            )
        )
        clipped[split] += clipped_fraction * len(y)
        counts[split] += len(y)

    for item in pending:
        save_sequence(
            "train", item["cell"], item["temperature"], item["kind"], item["source"], item
        )
    del pending

    for split in ("validation", "test"):
        for cell in splits[split]:
            for entry in cell_files(root, cell, cfg.include_charges):
                for part in load_segments(entry, cfg):
                    dropped_rows += part["dropped"]
                    parking_removed += part["parking_removed"]
                    resampled_rows += part["resampled"]
                    save_sequence(
                        split, cell, entry["temperature"], entry["kind"], entry["path"], part
                    )

    # 第三步：汇总各split的点数、截断比例和来源。
    summary = {}
    for split in splits:
        summary[split] = dict(
            cells=splits[split],
            state="driving_only" if not cfg.include_charges else "mixed",
            sequences=sum(1 for record in records if record["split"] == split),
            samples=counts[split],
            clipped_fraction=[
                float(value) for value in (clipped[split] / max(counts[split], 1))
            ],
        )
    manifest = dict(
        schema_version=2,
        dataset="Tesla_Model3_2170",
        task="driving_soc",
        state_filter="driving_only" if not cfg.include_charges else "mixed",
        source_root=str(root),
        features=feature_names,
        feature_order="V,I,T,Ah" if cfg.use_ah_feature else "V,I,T",
        ah_feature=dict(
            used=cfg.use_ah_feature,
            capacity_ah=cfg.capacity_ah,
            norm_low=cfg.ah_norm_low,
            norm_high=cfg.ah_norm_high,
        ),
        target="author_SOC_fraction",
        sampling_interval_s=cfg.sample_interval_s,
        window_seconds=dict(long=cfg.long_window_seconds, short=cfg.short_window_seconds),
        window_points=dict(long=cfg.long_window, short=cfg.short_window),
        max_parking_seconds=cfg.max_parking_seconds,
        parking_current_a=PARKING_CURRENT_A,
        max_gap_seconds=cfg.max_gap_seconds,
        min_sample_interval_s=MIN_SAMPLE_INTERVAL_S,
        normalization=normalizer,
        split_rule=dict(
            train_cells=splits["train"],
            validation_cells=splits["validation"],
            test_cells=splits["test"],
            temperatures=list(TEMPERATURES),
            include_charges=cfg.include_charges,
        ),
        boundary_status="source_file",
        dropped_rows=dropped_rows,
        resampled_rows=resampled_rows,
        parking_removed_points=parking_removed,
        summary=summary,
        records=records,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Prepared {len(records)} sequences -> {out}")
    for split, info in summary.items():
        print(
            f"  {split:>10}: {info['sequences']} sequences, {info['samples']} points, "
            f"clipped={[round(v, 5) for v in info['clipped_fraction']]}",
            flush=True,
        )
    print("Dropped rows:", dropped_rows, "| parking removed points:", parking_removed)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.source:
        cfg.source_dir = str(args.source.resolve())
    prepare(cfg, args.output)
