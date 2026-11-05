"""评估与结果导出工具：在相同目标时刻比较Transformer和协作模型。

输入为短窗Dataset、LSTM和实验元数据；输出预测CSV及分工况/汇总指标。
SOC内部为0～1，误差报告单位为百分点；原始预测不进行静默裁剪。
阅读时先看metrics的公式，再看evaluate中的逐batch预测和逐case汇总。
"""

import csv
import json
from pathlib import Path
import numpy as np
import torch
from model.train_utils import loader


def metrics(reference, predicted):
    # 内部SOC采用0～1；乘100后得到百分点，而非MAPE相对百分比。
    """计算样本数、MAE/RMSE/最大误差和预测超界数，误差以百分点报告。"""
    error = np.asarray(predicted) - np.asarray(reference)
    if error.size == 0:
        raise ValueError("没有可评估样本")
    return dict(
        n=int(error.size),
        mae_pp=float(np.mean(np.abs(error)) * 100),
        rmse_pp=float(np.sqrt(np.mean(error**2)) * 100),
        max_abs_pp=float(np.max(np.abs(error)) * 100),
        outside_0_1=int(np.sum((np.asarray(predicted) < 0) | (np.asarray(predicted) > 1))),
    )


def evaluate(dataset, model, cfg, device, output, metadata):
    """在相同目标时刻比较q[t]和LSTM输出，写出预测CSV与分工况/汇总指标。

    :param dataset: 短窗Dataset，每个窗口最后一列即q[t]。
    :param metadata: 数据版本、窗口、smoke及边界状态等实验信息。
    :return: 同步保存到metrics.json的指标字典。
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    groups = {}
    model.to(device).eval()
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case",
                "source",
                "source_row",
                "reference_soc",
                "transformer_soc",
                "collaborative_soc",
            ]
        )
        with torch.inference_mode():
            for x, y, seq_ids, ends in loader(dataset, cfg.lstm_batch, False, cfg):
                final = model(x.to(device)).cpu().numpy()
                # q追加在输入特征之后，索引为cfg.input_features（3维输入时为3，4维时为4）。
                # 它与LSTM输出共享同一个参考y[t]，不能分别选不同的评估区间。
                baseline = x[:, -1, cfg.input_features].numpy()
                reference = y.numpy()
                for i, seq_id in enumerate(seq_ids.tolist()):
                    s = dataset.sequences[seq_id]
                    writer.writerow(
                        [
                            s.name,
                            s.source,
                            s.offset + int(ends[i]),
                            float(reference[i]),
                            float(baseline[i]),
                            float(final[i]),
                        ]
                    )
                    g = groups.setdefault(s.name, [[], [], []])
                    # 每个工况的三个列表依次保存：参考、Transformer、协作模型。
                    for target, value in zip(g, [reference[i], baseline[i], final[i]]):
                        target.append(float(value))
    # 第一步：分别计算每个温度/工况的两种模型指标。
    per_case = {}
    for name, case_values in groups.items():
        reference = case_values[0]
        transformer_prediction = case_values[1]
        collaborative_prediction = case_values[2]
        per_case[name] = {
            "transformer": metrics(reference, transformer_prediction),
            "collaborative": metrics(reference, collaborative_prediction),
        }

    # 第二步：按相同的case顺序合并三个数组，用全部有效点计算加权汇总指标。
    # 先按工况计算RMSE再求平均，与这里的“全部点RMSE”不是同一个统计量。
    all_values = []
    for value_index in range(3):
        case_arrays = []
        for case_values in groups.values():
            case_arrays.append(np.asarray(case_values[value_index]))
        all_values.append(np.concatenate(case_arrays))
    report = dict(
        **metadata,
        units="percentage_points",
        clipped=False,
        aggregate_transformer=metrics(all_values[0], all_values[1]),
        aggregate_collaborative=metrics(all_values[0], all_values[2]),
        per_case=per_case
    )
    # macro MAE给每个工况相同权重，另存一项以区别于按点数加权的总MAE。
    macro_mae = {}
    for model_name in ["transformer", "collaborative"]:
        case_mae_values = []
        for case_metrics in per_case.values():
            case_mae_values.append(case_metrics[model_name]["mae_pp"])
        macro_mae[model_name] = float(np.mean(case_mae_values))
    report["macro_mae_pp"] = macro_mae
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "metrics_by_case.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "model", "n", "mae_pp", "rmse_pp", "max_abs_pp", "outside_0_1"])
        for name, case_metrics in per_case.items():
            for model_name, model_metrics in case_metrics.items():
                # 显式写出字段顺序，避免依赖字典values的隐含顺序来理解CSV列。
                writer.writerow(
                    [
                        name,
                        model_name,
                        model_metrics["n"],
                        model_metrics["mae_pp"],
                        model_metrics["rmse_pp"],
                        model_metrics["max_abs_pp"],
                        model_metrics["outside_0_1"],
                    ]
                )
    return report
