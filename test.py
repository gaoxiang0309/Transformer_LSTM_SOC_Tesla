"""独立测试入口：加载已训练的两级权重并比较SOC估计结果。

先检查数据版本和权重是否配套，再生成q和短窗，最后导出CSV/JSON。
本脚本不更新模型、不选择超参数，也不重新拟合归一化参数。
运行示例：python test.py --run run/train/exp。
"""
import argparse
import json
from pathlib import Path
from config import load_config
from model.tf_model import make_model
from model.lstm_model import LSTMSOC
from model.train_utils import restore, get_device, make_fused_sequences, seed_everything
from tools.data_loader import load_sequences, WindowDataset, sha256
from tools.evaluation import evaluate


def run(args):
    # 第一步：使用训练时保存的配置，而不是当前可能已修改的默认配置。
    """校验并加载已有实验，生成测试预测和指标，返回新结果目录。

    :param args: 包含已有实验run路径，以及可选device覆盖项。
    """
    run_dir = Path(args.run).resolve()
    cfg = load_config(run_dir / "config.json")
    if args.device:
        cfg.device = args.device
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    manifest = Path(cfg.processed_dir) / "manifest.json"
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    # 校验数据版本，防止准备数据改变后仍直接引用旧实验的权重。
    if sha256(manifest) != environment["manifest_sha256"]:
        raise ValueError("数据manifest与训练时不同")
    transformer = make_model(cfg)
    restore(transformer, run_dir / "weights" / "transformer_best.pt", device)
    t_sha = sha256(run_dir / "weights" / "transformer_best.pt")
    lstm = LSTMSOC(cfg)
    state = restore(lstm, run_dir / "weights" / "lstm_best.pt", device)
    # LSTM训练时依赖特定的Transformer初估，不能任意换成其他初估模型。
    if state["transformer_sha256"] != t_sha:
        raise ValueError("LSTM和Transformer checkpoint不配套")
    # 第二步：测试集单独加载，冻结Transformer生成q并拼成短窗特征。
    sequences = load_sequences(cfg, "test")
    fused = make_fused_sequences(
        transformer, sequences, cfg, device, run_dir / "cache" / "test", t_sha
    )
    dataset = WindowDataset(fused, cfg.short_window, cfg.eval_stride)
    # 第三步：选择尚不存在的结果目录，避免覆盖已有预测CSV和指标。
    for i in range(100000):
        output = run_dir / ("test" if i == 0 else f"test{i}")
        if not output.exists():
            break
    # 覆盖率按原始目标点计：每个评估窗口代表 stride 个点；冷启动前缀与长驻车切分不产生输出。
    raw_points = sum(len(s.y) for s in sequences)
    full_windows = sum(max(0, len(s.y) - cfg.short_window + 1) for s in fused)
    coverage_ratio = (
        min(1.0, len(dataset) * cfg.eval_stride / raw_points) if raw_points else 0.0
    )
    metadata = dict(
        smoke=cfg.smoke,
        boundary_status=environment["boundary_status"],
        state_filter=environment.get("state_filter"),
        sampling_interval_s=environment.get("sampling_interval_s"),
        window_seconds=environment.get("window_seconds"),
        manifest_sha256=environment["manifest_sha256"],
        transformer_sha256=t_sha,
        lstm_sha256=sha256(run_dir / "weights" / "lstm_best.pt"),
        raw_points_in_loaded_test_sequences=raw_points,
        prediction_windows=len(dataset),
        full_windows_at_stride_1=full_windows,
        coverage_ratio=coverage_ratio,
        long_window=cfg.long_window,
        short_window=cfg.short_window,
        base_input_features=cfg.input_features,
        q_input_features=cfg.q_input_features,
        stride=cfg.eval_stride,
        first_valid_local_target=cfg.long_window + cfg.short_window - 2,
        sequence_names=[s.name for s in sequences],
    )
    report = evaluate(dataset, lstm, cfg, device, output, metadata)
    print("TEST_OUTPUT:", output)
    print("Transformer:", report["aggregate_transformer"])
    print("Collaborative:", report["aggregate_collaborative"])
    if cfg.smoke:
        print("SMOKE结果：只验证程序流程，不能作为论文复现成绩。")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    run(args)
