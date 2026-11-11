"""第二阶段模型：用短窗口估计当前SOC（物理锚 + 可学习门控残差）。

输入[B,K,F]的列依次为V、I、T、Ah、q，以及物理锚anchor和短/长窗ΔAh（按配置拼接）。
物理锚的使用方式由 cfg.anchor_mode 决定：
  hard  = SOC = anchor + residual（锚权重恒为1，旧版）
  soft  = SOC = residual（锚只作为输入特征）
  gated = SOC = w·anchor + residual（方案B：可学习门控 w，w 初始化≈1）
真实SOC仅用于计算监督损失；每个窗口独立初始化隐藏状态。
"""
import torch
from torch import nn


class LSTMSOC(nn.Module):
    """用[V,I,T,Ah,q,anchor,dAh]短窗口预测SOC的第二阶段模型（锚+门控残差）。"""

    def __init__(self, cfg):
        """建立单向LSTM、残差回归头和可选门控头。"""
        super().__init__()
        # LSTM输入维度由config派生：输入特征 + q + 可选anchor + 2个ΔAh。
        self.input_features = cfg.input_features
        self.lstm_input_features = cfg.lstm_input_features
        self.use_physics_anchor = cfg.use_physics_anchor
        self.anchor_mode = cfg.anchor_mode if self.use_physics_anchor else "hard"
        # q紧跟输入特征之后，anchor紧跟q之后。
        self.q_index = cfg.input_features
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
        # 门控头：用[LSTM末端隐藏状态; anchor与q的差异]学一个 0~1 的锚权重 w。
        # 初始化 w≈1：权重置0、偏置置+4（sigmoid(4)≈0.982），保证warm-start起步≈旧版行为。
        self.gate_head = None
        if self.anchor_mode == "gated":
            self.gate_head = nn.Linear(cfg.lstm_hidden + 1, 1)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, 4.0)
        self.output_activation = cfg.output_activation

    def forward(self, x):
        """读取短窗口，返回按anchor_mode组合的[B]的SOC预测。"""
        if x.ndim != 3 or x.shape[-1] != self.lstm_input_features:
            raise ValueError(f"LSTM输入必须为[B,K,{self.lstm_input_features}]")
        out, _ = self.lstm(x)
        last_hidden = out[:, -1]  # [B,H]，短窗口最后一个时刻的隐藏状态
        residual = self.head(last_hidden).squeeze(-1)  # [B]，锚之外的修正量

        if self.use_physics_anchor:
            anchor = x[:, -1, self.anchor_index]  # [B]，窗口末端的物理锚
            if self.anchor_mode == "hard":
                prediction = anchor + residual
            elif self.anchor_mode == "soft":
                prediction = residual
            else:  # gated
                q = x[:, -1, self.q_index]  # [B]，窗口末端的Transformer初估
                mismatch = (anchor - q).unsqueeze(-1)  # [B,1]，锚与电压估计的差异
                gate_in = torch.cat([last_hidden, mismatch], dim=-1)  # [B,H+1]
                w = torch.sigmoid(self.gate_head(gate_in)).squeeze(-1)  # [B]，0~1
                prediction = w * anchor + residual
        else:
            prediction = residual

        if self.output_activation == "sigmoid":
            prediction = torch.sigmoid(prediction)
        return prediction
