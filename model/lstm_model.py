"""第二阶段模型：用短窗口估计当前SOC（物理锚 + 残差）。

输入[B,K,F]的列依次为V、I、T、q，以及物理锚anchor和短/长窗ΔAh（按配置拼接）。
若启用物理锚，输出 = anchor[t] + 残差；否则输出 = 残差（即直接回归SOC）。
真实SOC仅用于计算监督损失；每个窗口独立初始化隐藏状态。
"""
import torch
from torch import nn


class LSTMSOC(nn.Module):
    """用[V,I,T,q,anchor,dAh]短窗口预测SOC（锚+残差）的第二阶段模型。"""

    def __init__(self, cfg):
        """建立单向LSTM和两层隐藏FC，再接一个标量残差输出层。"""
        super().__init__()
        # LSTM输入维度由config派生：输入特征 + q + 可选anchor + 2个ΔAh。
        self.input_features = cfg.input_features
        self.lstm_input_features = cfg.lstm_input_features
        self.use_physics_anchor = cfg.use_physics_anchor
        # anchor紧跟q之后；q在input_features位置。
        self.anchor_index = cfg.input_features + 1
        self.lstm = nn.LSTM(
            self.lstm_input_features,
            cfg.lstm_hidden,
            cfg.lstm_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.lstm_hidden, cfg.fc1),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fc1, cfg.fc2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fc2, 1),
        )
        self.output_activation = cfg.output_activation

    def forward(self, x):
        """读取短窗口，返回anchor+残差（或纯残差）的[B]的SOC预测。"""
        if x.ndim != 3 or x.shape[-1] != self.lstm_input_features:
            raise ValueError(f"LSTM输入必须为[B,K,{self.lstm_input_features}]")
        # out为[B,K,H]。不传入上一个窗口的状态，每个窗口独立从零状态开始。
        out, _ = self.lstm(x)
        last_hidden = out[:, -1]  # [B,H]，短窗口最后一个时刻的隐藏状态
        residual = self.head(last_hidden).squeeze(-1)  # [B]，锚之外的修正量
        if self.use_physics_anchor:
            anchor = x[:, -1, self.anchor_index]  # [B]，窗口末端的物理锚
            prediction = anchor + residual
        else:
            prediction = residual
        if self.output_activation == "sigmoid":
            prediction = torch.sigmoid(prediction)
        return prediction
