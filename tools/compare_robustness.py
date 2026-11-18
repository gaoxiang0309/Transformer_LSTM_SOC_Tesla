"""配对比较q_Ah与q_v的ramp鲁棒性汇总。

输入可以是evaluate_ramp_robustness.py生成的目录，或直接指定其中的
robustness_summary.csv。脚本按fault_mode和bias_A配对，不按窗口数重新加权。
"""
import argparse
import csv
from pathlib import Path


def _summary_path(value):
    path = Path(value)
    if path.is_dir():
        path = path / "robustness_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {(row["fault_mode"], row["bias_A"]): row for row in rows}


def compare(first, second, output):
    first_path = _summary_path(first)
    second_path = _summary_path(second)
    left = _read(first_path)
    right = _read(second_path)
    keys = sorted(set(left) & set(right))
    if not keys:
        raise ValueError("两份汇总没有相同的fault_mode和bias_A条件")

    rows = []
    for key in keys:
        fault_mode, bias_a = key
        a = left[key]
        b = right[key]
        a_mae = float(a["collaborative_mae_pp"])
        b_mae = float(b["collaborative_mae_pp"])
        a_p95 = float(a["collaborative_p95_abs_pp"])
        b_p95 = float(b["collaborative_p95_abs_pp"])
        rows.append(
            {
                "fault_mode": fault_mode,
                "bias_A": bias_a,
                "first_mae_pp": a_mae,
                "second_mae_pp": b_mae,
                "second_minus_first_mae_pp": b_mae - a_mae,
                "first_p95_abs_pp": a_p95,
                "second_p95_abs_pp": b_p95,
                "second_minus_first_p95_abs_pp": b_p95 - a_p95,
            }
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", output)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path, help="q_Ah汇总目录或CSV")
    parser.add_argument("--second", required=True, type=Path, help="q_v汇总目录或CSV")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    compare(args.first, args.second, args.output)
