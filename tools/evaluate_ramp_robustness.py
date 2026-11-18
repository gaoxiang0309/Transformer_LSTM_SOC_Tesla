"""评估现有实验在验证集或测试集上的ramp偏置鲁棒性。

脚本只在内存中修改输入，不改变data/processed中的NPZ。支持两种扰动路径。

  clean              保持输入不变
  ah_only            只给累计Ah加入电流零偏的积分误差
  current_and_ah     同时给电流I和由同一偏置积分得到的Ah加入误差

每个条件都在相同的目标时刻上输出predictions.csv和metrics.json，
并在顶层写出robustness_summary.csv，便于比较q_Ah和q_v实验。
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

# 直接以“python tools/xxx.py”运行时，Python默认只把tools目录加入sys.path。
# 把项目根目录显式加入，保证config、model和tools包都能从服务器命令行导入。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from model.lstm_model import LSTMSOC
from model.tf_model import make_model
from model.train_utils import get_device, make_fused_sequences, restore, seed_everything
from tools.data_loader import Sequence, WindowDataset, load_sequences, sha256
from tools.evaluation import evaluate
from plot_results import plot_results
from tools.summarize_predictions import summarize


def _next_output_root(run_dir, split, requested):
    """创建不覆盖旧结果的分析目录。"""
    if requested:
        root = Path(requested)
        if root.exists():
            raise FileExistsError(f"输出目录已存在，请换一个路径: {root}")
        root.mkdir(parents=True)
        return root
    parent = Path(run_dir) / "analysis"
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(100000):
        name = f"ramp_robustness_{split}" if index == 0 else f"ramp_robustness_{split}{index}"
        root = parent / name
        try:
            root.mkdir()
            return root
        except FileExistsError:
            continue
    raise RuntimeError("无法创建鲁棒性分析目录")


def _normalization_bounds(cfg):
    """读取manifest中的训练侧归一化边界，返回low/high数组。"""
    manifest_path = Path(cfg.processed_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalization = manifest["normalization"]
    low = np.asarray(normalization["low"], dtype=np.float64)
    high = np.asarray(normalization["high"], dtype=np.float64)
    if low.shape != (4,) or high.shape != (4,):
        raise ValueError("当前ramp评估要求manifest包含[V,I,T,Ah]四个通道的边界")
    return manifest, low, high


def _inject_ramp(sequence, cfg, low, high, bias_a, fault_mode, temperature_coefficient):
    """按物理单位给一条序列注入ramp偏置并重新归一化。"""
    values = sequence.x.astype(np.float64, copy=True)
    ah_raw = values[:, 3] * (high[3] - low[3]) + low[3]
    temperature_c = values[:, 2] * (high[2] - low[2]) + low[2]
    bias = bias_a + temperature_coefficient * (temperature_c - 25.0)
    ah_delta = np.cumsum(bias) * cfg.sample_interval_s / 3600.0 / cfg.capacity_ah
    ah_raw = ah_raw + ah_delta
    values[:, 3] = (ah_raw - low[3]) / (high[3] - low[3])

    if fault_mode == "current_and_ah":
        current_raw = values[:, 1] * (high[1] - low[1]) + low[1]
        current_raw = current_raw + bias
        values[:, 1] = (current_raw - low[1]) / (high[1] - low[1])
    elif fault_mode != "ah_only":
        raise ValueError(f"未知fault_mode: {fault_mode}")

    return Sequence(
        sequence.name,
        values.astype(np.float32),
        sequence.y.copy(),
        sequence.offset,
        sequence.source,
    )


def _condition_sequences(sequences, cfg, low, high, fault_mode, bias_a, temperature_coefficient):
    if fault_mode == "clean":
        return [
            Sequence(s.name, s.x.copy(), s.y.copy(), s.offset, s.source)
            for s in sequences
        ]
    return [
        _inject_ramp(s, cfg, low, high, bias_a, fault_mode, temperature_coefficient)
        for s in sequences
    ]


def _condition_name(fault_mode, bias_a):
    if fault_mode == "clean":
        return "clean"
    sign = "neg" if bias_a < 0 else "pos"
    magnitude = str(abs(bias_a)).replace(".", "p")
    return f"{fault_mode}_{sign}{magnitude}A"


def run(args):
    run_dir = Path(args.run).resolve()
    cfg = load_config(run_dir / "config.json")
    if args.device:
        cfg.device = args.device
    cfg.validate()
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    manifest_path = Path(cfg.processed_dir) / "manifest.json"
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    manifest_sha = sha256(manifest_path)
    if manifest_sha != environment["manifest_sha256"]:
        raise ValueError("数据manifest与训练时不同")
    manifest, low, high = _normalization_bounds(cfg)

    transformer = make_model(cfg)
    transformer_state = restore(
        transformer, run_dir / "weights" / "transformer_best.pt", device
    )
    del transformer_state
    transformer_sha = sha256(run_dir / "weights" / "transformer_best.pt")
    lstm = LSTMSOC(cfg)
    lstm_state = restore(lstm, run_dir / "weights" / "lstm_best.pt", device)
    if lstm_state.get("transformer_sha256") != transformer_sha:
        raise ValueError("LSTM和Transformer checkpoint不配套")

    sequences = load_sequences(cfg, args.split)
    output_root = _next_output_root(run_dir, args.split, args.output)
    summary_rows = []
    conditions = [("clean", 0.0)]
    fault_modes = args.fault_mode
    for fault_mode in fault_modes:
        for bias_a in args.biases:
            conditions.append((fault_mode, float(bias_a)))

    for fault_mode, bias_a in conditions:
        condition_sequences = _condition_sequences(
            sequences,
            cfg,
            low,
            high,
            fault_mode,
            bias_a,
            args.temperature_coefficient,
        )
        fused = make_fused_sequences(
            transformer,
            condition_sequences,
            cfg,
            device,
            checkpoint_sha=transformer_sha,
            perturb=False,
        )
        dataset = WindowDataset(fused, cfg.short_window, cfg.eval_stride)
        condition_dir = output_root / _condition_name(fault_mode, bias_a)
        metadata = dict(
            split=args.split,
            fault_mode=fault_mode,
            bias_A=bias_a,
            temperature_coefficient_A_per_C=args.temperature_coefficient,
            sample_interval_s=cfg.sample_interval_s,
            window_seconds={
                "long": cfg.long_window_seconds,
                "short": cfg.short_window_seconds,
            },
            manifest_sha256=manifest_sha,
            transformer_sha256=transformer_sha,
            lstm_sha256=sha256(run_dir / "weights" / "lstm_best.pt"),
            q_input_features=cfg.q_input_features,
            base_input_features=cfg.input_features,
            use_history_stats=cfg.use_history_stats,
            history_window_seconds=cfg.history_window_seconds,
            prediction_windows=len(dataset),
            first_valid_local_target=cfg.long_window + cfg.short_window - 2,
        )
        report = evaluate(dataset, lstm, cfg, device, condition_dir, metadata)
        if not args.no_plot:
            plot_results(condition_dir, history_root=run_dir)
            summarize(condition_dir)
        coll = report["aggregate_collaborative"]
        trans = report["aggregate_transformer"]
        summary_rows.append(
            dict(
                fault_mode=fault_mode,
                bias_A=bias_a,
                collaborative_mae_pp=coll["mae_pp"],
                collaborative_rmse_pp=coll["rmse_pp"],
                collaborative_mean_signed_error_pp=coll["mean_signed_error_pp"],
                collaborative_p95_abs_pp=coll["p95_abs_pp"],
                collaborative_max_abs_pp=coll["max_abs_pp"],
                transformer_mae_pp=trans["mae_pp"],
                transformer_p95_abs_pp=trans["p95_abs_pp"],
                output_dir=str(condition_dir),
            )
        )

    summary_path = output_root / "robustness_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    metadata = dict(
        split=args.split,
        fault_modes=args.fault_mode,
        biases_A=args.biases,
        temperature_coefficient_A_per_C=args.temperature_coefficient,
        manifest_sha256=manifest_sha,
        q_input_features=cfg.q_input_features,
        base_input_features=cfg.input_features,
        use_history_stats=cfg.use_history_stats,
        history_window_seconds=cfg.history_window_seconds,
    )
    (output_root / "robustness_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("ROBUSTNESS_OUTPUT:", output_root)
    print("SUMMARY:", summary_path)
    return output_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--fault-mode",
        nargs="+",
        choices=["ah_only", "current_and_ah"],
        default=["ah_only", "current_and_ah"],
    )
    parser.add_argument(
        "--biases",
        nargs="+",
        type=float,
        default=[-0.12, -0.05, 0.05, 0.12],
    )
    parser.add_argument("--temperature-coefficient", type=float, default=0.0)
    parser.add_argument("--no-plot", action="store_true", help="只输出CSV和指标，不生成PNG")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    run(parser.parse_args())
