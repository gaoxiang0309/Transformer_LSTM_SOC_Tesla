"""SOC估计的训练主入口，对应示例项目main.py。

运行顺序：读取配置和数据 -> 训练Transformer -> 冻结生成初估 -> 训练LSTM。
训练只读取train/validation；最终test由独立test.py执行。
可先阅读底部命令行参数，再按run函数中的步骤了解整个流程。
"""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import torch
from config import load_config
from model.tf_model import make_model
from model.lstm_model import LSTMSOC
from model.train_utils import (
    get_device,
    seed_everything,
    train_model,
    restore,
    make_fused_sequences,
)
from tools.data_loader import load_sequences, WindowDataset, sha256
from tools.create_exp_folder import create_exp_folder


def run(args):
    # 第一步：选择配置。接续LSTM时，必须读取原实验快照，防止网络设置不一致。
    """按args指定的阶段执行训练，并返回本次实验目录。

    :param args: 命令行解析得到的对象，包含config、stage、run、smoke和device。
    """
    if args.run and args.stage != "lstm":
        raise ValueError("--run仅用于接续已有Transformer阶段：--stage lstm --run ...")
    if args.run:
        config_path = Path(args.run) / "config.json"
    else:
        config_path = args.config
    cfg = load_config(config_path)
    if args.smoke and not args.run:
        cfg.smoke_config()
    if args.device:
        cfg.device = args.device
    cfg.validate()
    # 第二步：检查准备好的数据。序列边界来自原始测试文件，
    # 不再需要SOC跳变的人工复核标记。
    manifest_path = Path(cfg.processed_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("先运行 prepare_data.py，见README")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["boundary_status"] != "source_file":
        raise ValueError("Tesla数据应为source_file边界；请重新运行prepare_data.py")
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    # 第三步：新建实验目录或接续已有实验，保存配置、环境和数据校验值。
    if args.run:
        run_dir = Path(args.run).resolve()
    else:
        run_dir = create_exp_folder(cfg.run_dir)
    data_sha = sha256(manifest_path)
    if args.run:
        old = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
        if old["manifest_sha256"] != data_sha:
            raise ValueError("准备数据已改变，不允许混用实验")
    else:
        cfg.save(run_dir / "config.json")
        environment = dict(
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            device=str(device),
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            manifest_sha256=data_sha,
            boundary_status=manifest["boundary_status"],
            state_filter=manifest.get("state_filter"),
            sampling_interval_s=cfg.sample_interval_s,
            window_seconds=dict(
                long=cfg.long_window_seconds, short=cfg.short_window_seconds
            ),
            smoke=cfg.smoke,
        )
        (run_dir / "environment.json").write_text(
            json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("RUN_DIR:", run_dir, flush=True)
    print(
        "DEVICE:",
        device,
        "| BOUNDARY:",
        manifest["boundary_status"],
        "| STATE:",
        manifest.get("state_filter"),
        "| SMOKE:",
        cfg.smoke,
        flush=True,
    )
    # 第四步：只加载训练和验证集。测试集由test.py在选模完成后读取。
    train = load_sequences(cfg, "train")
    val = load_sequences(cfg, "validation")
    transformer = make_model(cfg).to(device)
    if args.stage in ("all", "transformer"):
        # 长窗输入[B,L,3]，目标是窗口最后一个时刻的SOC。
        train_dataset = WindowDataset(train, cfg.long_window, cfg.train_stride)
        validation_dataset = WindowDataset(val, cfg.long_window, cfg.eval_stride)
        train_model(
            transformer,
            train_dataset,
            validation_dataset,
            cfg,
            run_dir,
            "transformer",
            device,
        )
    else:
        restore(transformer, run_dir / "weights" / "transformer_best.pt", device)
    if args.stage in ("all", "lstm"):
        # 第五步：固定最佳Transformer生成q，再训练短窗LSTM（输入V/I/T/Ah/q）。
        # t_sha用于记录q由哪个权重生成，测试时检查两阶段权重是否配套。
        t_sha = sha256(run_dir / "weights" / "transformer_best.pt")
        fused_train = make_fused_sequences(
            transformer, train, cfg, device, run_dir / "cache" / "train", t_sha,
            perturb=cfg.perturb_train,
        )
        fused_val = make_fused_sequences(
            transformer, val, cfg, device, run_dir / "cache" / "validation", t_sha
        )
        lstm = LSTMSOC(cfg)
        if cfg.init_lstm_from:
            # 从旧权重warm-start（strict=False忽略缺失的gate_head，保持其近恒等初始化）。
            print("WARM-START LSTM from:", cfg.init_lstm_from, flush=True)
            restore(lstm, Path(cfg.init_lstm_from), device, strict=False)
        train_dataset = WindowDataset(fused_train, cfg.short_window, cfg.train_stride)
        validation_dataset = WindowDataset(fused_val, cfg.short_window, cfg.eval_stride)
        train_model(
            lstm,
            train_dataset,
            validation_dataset,
            cfg,
            run_dir,
            "lstm",
            device,
            t_sha,
        )
    # 训练结束后打印本次实验目录与后续命令；测试必须单独运行，不会自动执行。
    print("=" * 72, flush=True)
    print("训练阶段完成。RUN_DIR =", run_dir, flush=True)
    print("下一步（在同一 Python 环境执行）：", flush=True)
    print(f'  python test.py --run "{run_dir}"', flush=True)
    print(f'  python plot_results.py --test-dir "{run_dir}/test"', flush=True)
    print(f'  python tools/summarize_predictions.py --test-dir "{run_dir}/test"', flush=True)
    print("=" * 72, flush=True)
    return run_dir


if __name__ == "__main__":
    # parser定义程序接受的命令行选项；parse_args将它们转换成args对象。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--stage", choices=["all", "transformer", "lstm"], default="all")
    parser.add_argument("--run")
    parser.add_argument("--smoke", action="store_true", help="小模型/少量真实数据/每阶段1轮，非正式实验")
    parser.add_argument("--device", help="auto、cpu或cuda:0")
    args = parser.parse_args()
    run(args)
