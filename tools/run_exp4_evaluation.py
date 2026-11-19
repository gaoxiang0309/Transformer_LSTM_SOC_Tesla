"""一键训练并评价实验四。

默认流程会复用已有实验四权重或先训练实验四，然后运行clean测试、
q_Ah控制组和实验四的validation ramp，最后生成配对比较CSV和评价索引。
所有评价产出写入实验四run目录下的analysis/exp4_evaluation子目录。
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP4_RUN = PROJECT_ROOT / "run" / "train" / "exp_batch4_qah_loadstats60"
CONTROL_RUN = PROJECT_ROOT / "run" / "train" / "exp_batch1_qah_control"
CONFIG = PROJECT_ROOT / "configs" / "exp_batch4_qah_loadstats60.json"


def _run(command):
    display = " ".join(str(item) for item in command)
    print(f"\n>>> {display}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _new_evaluation_root(run_dir):
    parent = run_dir / "analysis"
    parent.mkdir(parents=True, exist_ok=True)
    base = parent / "exp4_evaluation"
    for index in range(100000):
        candidate = base if index == 0 else parent / f"exp4_evaluation_{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError("无法创建实验四评价目录")


def _latest_test_dir(run_dir):
    candidates = [path for path in run_dir.glob("test*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"实验四没有找到test输出目录: {run_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _check_exp4_run(run_dir):
    config_path = run_dir / "config.json"
    weights = run_dir / "weights"
    if not config_path.exists() or not (weights / "transformer_best.pt").exists() or not (
        weights / "lstm_best.pt"
    ).exists():
        raise FileNotFoundError(
            "实验四目录缺少完整配置或最佳权重，请先运行训练，或删除不完整目录后重试"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_name") != "exp_batch4_qah_loadstats60":
        raise ValueError(f"实验四目录配置身份不匹配: {config.get('experiment_name')}")
    if not config.get("use_load_stats", False):
        raise ValueError("实验四配置没有启用use_load_stats")


def _write_manifest(path, values):
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args):
    exp4_run = EXP4_RUN
    if exp4_run.exists():
        if not args.skip_train:
            print("检测到已有完整实验四目录，将复用已有权重并跳过重复训练。", flush=True)
        _check_exp4_run(exp4_run)
    else:
        if args.skip_train:
            raise FileNotFoundError(f"--skip-train要求实验四权重已存在: {exp4_run}")
        _run([sys.executable, "main.py", "--config", str(CONFIG), "--device", args.device])
        _check_exp4_run(exp4_run)

    if not CONTROL_RUN.exists():
        raise FileNotFoundError(f"缺少q_Ah控制组目录: {CONTROL_RUN}")
    evaluation_root = _new_evaluation_root(exp4_run)

    clean_test_dir = evaluation_root / "clean_test"
    if not args.skip_test:
        _run(
            [
                sys.executable,
                "test.py",
                "--run",
                str(exp4_run),
                "--output",
                str(clean_test_dir),
                "--device",
                args.device,
            ]
        )
    elif clean_test_dir.exists():
        pass
    else:
        clean_test_dir = _latest_test_dir(exp4_run)

    common_ramp = [
        "--split",
        args.split,
        "--fault-mode",
        "ah_only",
        "current_and_ah",
        "--biases",
        *[str(value) for value in args.biases],
        "--temperature-coefficient",
        str(args.temperature_coefficient),
        "--device",
        args.device,
    ]
    control_ramp = evaluation_root / "qah_control_ramp_robustness_validation"
    exp4_ramp = evaluation_root / "exp4_ramp_robustness_validation"

    if not args.skip_baseline_ramp:
        _run(
            [
                sys.executable,
                "tools/evaluate_ramp_robustness.py",
                "--run",
                str(CONTROL_RUN),
                *common_ramp,
                "--output",
                str(control_ramp),
            ]
        )
    elif not control_ramp.exists():
        raise FileNotFoundError(
            "--skip-baseline-ramp要求当前评价目录中已有q_Ah控制组ramp结果"
        )
    _run(
        [
            sys.executable,
            "tools/evaluate_ramp_robustness.py",
            "--run",
            str(exp4_run),
            *common_ramp,
            "--output",
            str(exp4_ramp),
        ]
    )

    comparison = evaluation_root / "qah_control_vs_exp4_robustness.csv"
    _run(
        [
            sys.executable,
            "tools/compare_robustness.py",
            "--first",
            str(control_ramp),
            "--second",
            str(exp4_ramp),
            "--output",
            str(comparison),
        ]
    )

    manifest = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        experiment="exp_batch4_qah_loadstats60",
        baseline="exp_batch1_qah_control",
        split=args.split,
        biases_A=args.biases,
        temperature_coefficient_A_per_C=args.temperature_coefficient,
        exp4_run=str(exp4_run),
        clean_test_dir=str(clean_test_dir),
        evaluation_root=str(evaluation_root),
        qah_control_ramp_dir=str(control_ramp),
        exp4_ramp_dir=str(exp4_ramp),
        robustness_comparison=str(comparison),
        features=["I_rms_60", "I_pos_ratio_60"],
    )
    manifest_path = evaluation_root / "evaluation_manifest.json"
    _write_manifest(manifest_path, manifest)
    print("\n实验四一键评价完成。", flush=True)
    print("EVALUATION_ROOT:", evaluation_root, flush=True)
    print("CLEAN_TEST:", clean_test_dir, flush=True)
    print("ROBUSTNESS_COMPARISON:", comparison, flush=True)
    print("EVALUATION_MANIFEST:", manifest_path, flush=True)
    return evaluation_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument(
        "--biases",
        nargs="+",
        type=float,
        default=[-0.12, -0.05, 0.05, 0.12],
    )
    parser.add_argument("--temperature-coefficient", type=float, default=0.0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-baseline-ramp", action="store_true")
    run(parser.parse_args())
