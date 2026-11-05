"""按工况类型汇总测试预测：把静置/充电占主导的总指标拆开看。

输入为某次测试输出目录中的 predictions.csv；输出为按 case 类型（drive/charge）
和温度分组的 MAE/RMSE，写入 summary_by_kind.csv 并打印。
不重新训练、不改变原始预测，只做分组统计。
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np


def parse_case(case):
    """从case名中拆出电芯、温度和循环名，例如 m1000_25C_REORDERED1。"""
    parts = case.split("_", 2)
    if len(parts) < 3:
        return case, "unknown", "unknown"
    return parts[0], parts[1], parts[2]


def is_charge(cycle):
    """按循环名判断是否为充电文件。"""
    text = cycle.upper()
    return "CC_CV" in text or "CHARGE" in text


def metrics(reference, prediction):
    """计算MAE/RMSE/最大误差，单位百分点。"""
    error = np.asarray(prediction) - np.asarray(reference)
    return dict(
        n=int(error.size),
        mae_pp=float(np.mean(np.abs(error)) * 100),
        rmse_pp=float(np.sqrt(np.mean(error**2)) * 100),
        max_abs_pp=float(np.max(np.abs(error)) * 100),
    )


def summarize(test_dir):
    """读取predictions.csv，按类型、温度、case分组写出指标。"""
    test_dir = Path(test_dir)
    references = defaultdict(list)
    predictions = defaultdict(lambda: [[], []])  # key -> [transformer, collaborative]
    with (test_dir / "predictions.csv").open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            cell, temperature, cycle = parse_case(row["case"])
            kind = "charge" if is_charge(cycle) else "drive"
            reference = float(row["reference_soc"])
            values = [float(row["transformer_soc"]), float(row["collaborative_soc"])]
            # 同一批目标时刻分别按类型、温度和case累计，保证三组统计口径一致。
            for key in (("kind", kind), ("temperature", temperature), ("case", row["case"])):
                references[key].append(reference)
                predictions[key][0].append(values[0])
                predictions[key][1].append(values[1])

    rows = []
    for key, reference in references.items():
        for index, model_name in enumerate(["transformer", "collaborative"]):
            stat = metrics(reference, predictions[key][index])
            rows.append(dict(group=key[0], key=key[1], model=model_name, **stat))

    output = test_dir / "summary_by_kind.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["group", "key", "model", "n", "mae_pp", "rmse_pp", "max_abs_pp"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", output)
    for row in rows:
        if row["group"] != "case":
            print(
                f"{row['group']:>11} {row['key']:>10} {row['model']:>14}: "
                f"n={row['n']:>7} MAE={row['mae_pp']:.3f} RMSE={row['rmse_pp']:.3f} "
                f"MAX={row['max_abs_pp']:.2f}"
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", required=True, type=Path)
    args = parser.parse_args()
    summarize(args.test_dir)
