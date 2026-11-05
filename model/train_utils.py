"""训练与两阶段衔接工具，对应示例项目model/train_utils.py。

包括随机种子、设备选择、DataLoader、单轮训练、checkpoint和特征融合。
train_model只用验证损失选模型；make_fused_sequences冻结Transformer生成q维。
所有训练张量均采用batch在前的格式，标签为窗口末端SOC小数。
"""
import csv
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tools.data_loader import Sequence, WindowDataset, sha256


def seed_everything(seed):
    """固定Python、NumPy和PyTorch随机种子，并设置cuDNN以减少运行差异。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(value):
    """把auto/cpu/cuda选项转换为torch.device；显式要求不可用CUDA时抛出错误。"""
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用；使用 --device cpu")
    return device


def loader(dataset, batch, shuffle, cfg):
    """为窗口Dataset建立DataLoader，并在没有有效窗口时给出明确错误。"""
    if len(dataset) == 0:
        raise ValueError("没有有效窗口，请检查序列长度、边界和窗口配置")
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=cfg.num_workers)


def run_epoch(batches, model, device, optimizer=None, max_batches=0, grad_clip=1.0):
    # 传入优化器表示训练；不传优化器表示验证，只做前向计算。
    """完成一轮训练或验证，返回按样本数加权的MSE和实际样本数。

    :param optimizer: 非空时执行反向传播；为None时关闭梯度。
    :param max_batches: 大于0时只处理前若干batch，主要用于smoke检查。
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    # 上下文管理器统一控制是否建立反向传播计算图，不同于model.eval的作用。
    with torch.set_grad_enabled(training):
        for i, (x, y, _, _) in enumerate(batches):
            if max_batches and i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            # prediction与y都为[B]。使用MSE直接监督当前时刻SOC。
            loss = torch.mean((prediction - y) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("发现非有限loss")
            if training:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            # 按batch样本数加权，避免最后一个小batch与完整batch权重相同。
            total_loss += loss.detach().item() * len(y)
            total_samples += len(y)
    if not total_samples:
        raise ValueError("本轮无样本")
    return total_loss / total_samples, total_samples


def save_checkpoint(path, payload):
    """先保存临时checkpoint，再替换目标文件，降低中断导致文件不完整的风险。"""
    path = Path(path)
    temp = path.with_suffix(".tmp")
    # 先写临时文件再替换，避免训练意外中断时留下半个checkpoint。
    torch.save(payload, temp)
    temp.replace(path)


def restore(model, path, device):
    """读取checkpoint，将权重载入模型并设置推理模式，返回checkpoint中的其他信息。"""
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return state


def train_model(model, train, validation, cfg, run, stage, device, transformer_sha=None):
    """训练一个阶段，根据验证MSE保存best/last并执行早停，最后恢复best权重。

    :param train: 对应阶段的训练窗口Dataset。
    :param validation: 只用于验证和选模的窗口Dataset。
    :param stage: transformer或lstm，决定采用哪组batch、epoch和学习率。
    :param transformer_sha: LSTM训练时记录其初估来源权重，供测试核对。
    """
    is_transformer = stage == "transformer"
    batch = cfg.transformer_batch if is_transformer else cfg.lstm_batch
    epochs = cfg.transformer_epochs if is_transformer else cfg.lstm_epochs
    lr = cfg.transformer_lr if is_transformer else cfg.lstm_lr
    train_loader = loader(train, batch, True, cfg)
    val_loader = loader(validation, batch, False, cfg)
    model = model.to(device)
    # 两个阶段分别创建优化器。LSTM阶段不会更新Transformer参数。
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=lr,
        alpha=cfg.rmsprop_alpha,
        eps=cfg.rmsprop_eps,
        weight_decay=cfg.weight_decay,
    )
    best = math.inf
    bad_epochs = 0
    weights = Path(run) / "weights"
    with (Path(run) / f"{stage}_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["epoch", "train_mse", "validation_mse", "train_samples", "validation_samples"]
        )
        for epoch in range(1, epochs + 1):
            train_loss, ntrain = run_epoch(
                train_loader, model, device, optimizer, cfg.max_train_batches, cfg.grad_clip
            )
            val_loss, nval = run_epoch(val_loader, model, device, None, cfg.max_eval_batches)
            # CSV同时记录实际参与计算的点数，smoke运行可能只处理前几个batch。
            writer.writerow([epoch, train_loss, val_loss, ntrain, nval])
            stream.flush()
            print(
                f"{stage} epoch={epoch} train_MSE={train_loss:.7f} val_MSE={val_loss:.7f}",
                flush=True,
            )
            state = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                validation_mse=val_loss,
                stage=stage,
                smoke=cfg.smoke,
                transformer_sha256=transformer_sha,
            )
            save_checkpoint(weights / f"{stage}_last.pt", state)
            # 只依据验证损失选择best，测试数据不参与选模。
            if val_loss < best:
                best = val_loss
                bad_epochs = 0
                save_checkpoint(weights / f"{stage}_best.pt", state)
            else:
                bad_epochs += 1
            if cfg.patience and bad_epochs >= cfg.patience:
                break
    restore(model, weights / f"{stage}_best.pt", device)
    return model


def make_fused_sequences(transformer, sequences, cfg, device, cache_dir=None, checkpoint_sha=""):
    """冻结Transformer，按时间顺序生成q并拼成[输入特征,q]序列，返回Sequence列表。

    每个q[s]独立使用[s-L+1,s]，不一次编码整段后回取历史位置。
    新序列会裁去L-1个无初估的点，offset同步更新。
    """
    transformer.to(device).eval()
    # eval关闭dropout；requires_grad_(False)冻结参数，两者作用不同。
    transformer.requires_grad_(False)
    result = []
    skipped = []
    for sequence in sequences:
        dataset = WindowDataset([sequence], cfg.long_window)
        if len(dataset) == 0:
            skipped.append(sequence.name)
            continue
        predictions = []
        with torch.inference_mode():
            # Dataset在单条记录内按步长1生成全部长窗，按时间顺序预测q。
            for x, _, _, _ in loader(dataset, cfg.inference_batch, False, cfg):
                predictions.append(transformer(x.to(device)).cpu().numpy())
        q = np.concatenate(predictions).astype("float32")
        first = cfg.long_window - 1
        # q[0]实际对应原记录的first点，故V/I/T与标签也必须从first裁切。
        x = np.column_stack([sequence.x[first:], q]).astype("float32")
        # offset记录裁切前的原行号，便于最后把预测映射回原记录。
        fused = Sequence(
            sequence.name, x, sequence.y[first:].copy(), sequence.offset + first, sequence.source
        )
        result.append(fused)
        if cache_dir:
            path = Path(cache_dir)
            path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path / f"{sequence.name}.npz",
                X=x,
                Y=fused.y,
                first_original_row=np.asarray(fused.offset),
                transformer_sha256=np.asarray(checkpoint_sha),
            )
    if skipped:
        print("因长窗口历史不足而跳过：", skipped, flush=True)
    return result
