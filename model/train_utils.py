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


def restore(model, path, device, strict=True):
    """读取checkpoint，将权重载入模型并设置推理模式，返回checkpoint中的其他信息。

    :param strict: False时允许缺失/多余键，用于从旧结构（如无gate_head的hard锚）warm-start。
    """
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"], strict=strict)
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


def _capacity_for_sequence(name, cfg):
    """按序列名中的温度标签查容量表；找不到时回退到额定容量。

    长键优先匹配，避免"0C"误匹配"10C"/"40C"/"-10C"。
    """
    for key in sorted(cfg.capacity_by_temperature, key=len, reverse=True):
        if key in name:
            return float(cfg.capacity_by_temperature[key])
    return float(cfg.capacity_ah)


def _rolling_delta(values, window):
    """返回 values[t]-values[t-window+1]；前 window-1 个点置0。"""
    out = np.zeros(len(values), dtype=np.float64)
    if window > 1 and len(values) >= window:
        out[window - 1 :] = values[window - 1 :] - values[: -window + 1]
    return out


def _perturb_ah(x, cfg, rng):
    """给输入第3列(Ah)注入 step/ramp 偏置并重归一化；V/I/T 与标签不变。

    step：在 [0.3n, 0.7n] 随机时刻加 ±perturb_amp 的固定偏移；
    ramp：电流零偏 b(t)=b0+tc·(T−25℃)，b0∈[-perturb_amp,perturb_amp](A)、
          tc∈[-perturb_tc,perturb_tc](A/℃)，Ah误差=∫b dt/3600/C（线性+温漂）。
    """
    x = np.array(x, dtype=np.float64)
    ah = x[:, 3] * (cfg.ah_norm_high - cfg.ah_norm_low) + cfg.ah_norm_low
    n = len(ah)
    if cfg.perturb_mode == "step":
        lo = max(1, int(0.3 * n))
        hi = max(lo + 1, int(0.7 * n))
        t0 = int(rng.integers(lo, hi))
        delta = float(rng.choice(np.array([-1.0, 1.0]))) * cfg.perturb_amp
        ah[t0:] += delta
    else:  # ramp
        b0 = float(rng.uniform(-cfg.perturb_amp, cfg.perturb_amp))
        tc = float(rng.uniform(-cfg.perturb_tc, cfg.perturb_tc))
        T = x[:, 2] * (cfg.t_norm_high - cfg.t_norm_low) + cfg.t_norm_low  # 还原℃
        b = b0 + tc * (T - 25.0)  # 温漂：25℃处零偏=b0
        ah += np.cumsum(b) / 3600.0 / cfg.capacity_ah  # 1 Hz，dt=1s
    x[:, 3] = (ah - cfg.ah_norm_low) / (cfg.ah_norm_high - cfg.ah_norm_low)
    return x.astype(np.float32)


def make_fused_sequences(transformer, sequences, cfg, device, cache_dir=None, checkpoint_sha="", perturb=False):
    """冻结Transformer生成q，并拼接物理锚与ΔAh特征，返回Sequence列表。

    列顺序：[V,I,T,Ah, q, anchor, dAh_K, dAh_L]（按开关拼接）。
    每个q[s]独立使用[s-L+1,s]，不一次编码整段后回取历史位置。
    新序列会裁去L-1个无初估的点，offset同步更新。
    perturb=True 时对训练集给Ah注入偏置（方案一），label保持干净，供扰动训练。
    """
    transformer.to(device).eval()
    # eval关闭dropout；requires_grad_(False)冻结参数，两者作用不同。
    transformer.requires_grad_(False)
    rng = np.random.default_rng(cfg.perturb_seed or cfg.seed)
    result = []
    skipped = []
    for sequence in sequences:
        x_arr = sequence.x
        if perturb and cfg.use_ah_feature and rng.random() < cfg.perturb_prob:
            x_arr = _perturb_ah(sequence.x, cfg, rng)
        seq = Sequence(sequence.name, x_arr, sequence.y, sequence.offset, sequence.source)
        dataset = WindowDataset([seq], cfg.long_window)
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
        base_x = seq.x[first:]
        x = np.column_stack([base_x, q]).astype("float32")
        if cfg.use_physics_anchor:
            # Ah在prepare中按固定范围归一化，这里反算回容量比例。
            ah_raw = base_x[:, 3] * (cfg.ah_norm_high - cfg.ah_norm_low) + cfg.ah_norm_low
            capacity = _capacity_for_sequence(sequence.name, cfg)
            scale = cfg.capacity_ah / capacity
            coulomb = scale * ah_raw
            # 用段首q反推段首SOC，再沿Ah累计给出整段物理锚。
            anchor = (float(q[0]) - float(coulomb[0])) + coulomb
            x = np.column_stack([x, anchor]).astype("float32")
        if cfg.use_delta_ah:
            # 在裁切前计算完整序列的历史差分，再同步裁切到q的有效起点，
            # 避免边界处ΔAh（尤其ΔAh_L）被截断为0。
            full_ah_raw = seq.x[:, 3] * (cfg.ah_norm_high - cfg.ah_norm_low) + cfg.ah_norm_low
            d_short_full = _rolling_delta(full_ah_raw, cfg.short_window)
            d_long_full = _rolling_delta(full_ah_raw, cfg.long_window)
            d_short = d_short_full[first:]
            d_long = d_long_full[first:]
            x = np.column_stack([x, d_short, d_long]).astype("float32")
        if x.shape[1] != cfg.lstm_input_features:
            raise ValueError(
                f"融合特征列数{x.shape[1]}与lstm_input_features={cfg.lstm_input_features}不一致"
            )
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
