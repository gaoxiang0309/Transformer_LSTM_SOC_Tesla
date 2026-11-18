"""结果绘图脚本：读取已有CSV，保存SOC、误差和训练loss曲线。

输入为某次实验的test目录；输出PNG保存在同一目录。
仅依赖绘图库和CSV，不需要加载模型或启动训练，可使用Anaconda base环境。
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def plot_results(test_dir, history_root=None):
    """按工况绘制预测/误差曲线和训练loss，返回生成的PNG路径。"""
    test_dir = Path(test_dir)
    history_root = test_dir.parent if history_root is None else Path(history_root)
    if not (test_dir / "predictions.csv").exists():
        raise FileNotFoundError(f"缺少预测文件: {test_dir / 'predictions.csv'}")
    import matplotlib

    # Agg后端只保存图像，不在训练机器上弹出交互窗口。
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 一个预测CSV可包含多个温度/工况；分组后逐个绘图，避免曲线跨case连接。
    groups = defaultdict(list)
    with (test_dir / "predictions.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            groups[row["case"]].append(row)
    outputs = []
    for case, rows in groups.items():
        rows.sort(key=lambda row: int(row["source_row"]))
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        index = [int(r["source_row"]) for r in rows]
        # 计算仍使用SOC小数；仅在显示时换成百分比刻度。
        truth = [100 * float(r["reference_soc"]) for r in rows]
        axes[0].plot(index, truth, label="Reference", color="black")
        for col, label in [
            ("transformer_soc", "Transformer"),
            ("collaborative_soc", "Transformer-LSTM"),
        ]:
            values = [100 * float(r[col]) for r in rows]
            axes[0].plot(index, values, label=label, alpha=0.8)
            axes[1].plot(index, [v - y for v, y in zip(values, truth)], label=label)
        axes[0].set_ylabel("SOC (%)")
        axes[1].set_ylabel("Error (pp)")
        axes[1].set_xlabel("Source row index")
        axes[0].set_title(case)
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        output = test_dir / f"{case}.png"
        fig.savefig(output, dpi=160)
        outputs.append(output)
        plt.close(fig)
    # 预测结果目录的上一级是实验目录，其中保存了两个阶段的训练历史。
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    found = False
    for ax, stage in zip(axes, ["transformer", "lstm"]):
        path = history_root / f"{stage}_history.csv"
        if not path.exists():
            ax.text(
                0.5,
                0.5,
                "reused checkpoint",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(stage)
            ax.set_axis_off()
            continue
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for column in ["train_mse", "validation_mse"]:
            ax.plot(
                [int(r["epoch"]) for r in rows],
                [float(r[column]) for r in rows],
                marker="o",
                label=column,
            )
        ax.set_title(stage)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.grid(alpha=0.3)
        found = True
    if found:
        fig.tight_layout()
        output = test_dir / "training_losses.png"
        fig.savefig(output, dpi=160)
        outputs.append(output)
    plt.close(fig)
    print("Plots:", test_dir)
    return outputs


def main():
    """解析结果目录，按工况绘制预测/误差曲线，再绘制两阶段训练loss。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", required=True, type=Path)
    args = parser.parse_args()
    plot_results(args.test_dir)


if __name__ == "__main__":
    main()
