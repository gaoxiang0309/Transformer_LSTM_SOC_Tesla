"""第二阶段模型：用短窗口估计当前SOC。

输入[B,K,F+1]的列依次为V、I、T、可选安时积分Ah，以及冻结Transformer的初估q。
LSTM末端隐藏状态经全连接层输出[B]；真实SOC仅用于计算监督损失。
每个窗口独立初始化隐藏状态，不强制将输出写成q的残差。
"""
import torch
from torch import nn


class LSTMSOC(nn.Module):
    """用[V,I,T,(Ah),q]短窗口预测当前SOC的第二阶段模型。"""

    def __init__(self, cfg):
        """建立单向LSTM和两层隐藏FC，再接一个标量输出层。"""
        super().__init__()
        # 输入维度 = 输入特征数（3或4）+ 1维q；单层LSTM没有可设置的层间dropout。
        self.input_features = cfg.input_features
        self.lstm = nn.LSTM(
            self.input_features + 1,
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
        """读取[B,K,F+1]的短窗口，取LSTM末端隐藏状态，返回[B]的SOC预测。"""
        if x.ndim != 3 or x.shape[-1] != self.input_features + 1:
            raise ValueError(f"LSTM输入必须为[B,K,{self.input_features + 1}]")
        # out为[B,K,H]。不传入上一个窗口的状态，每个窗口独立从零状态开始。
        out, _ = self.lstm(x)
        last_hidden = out[:, -1]  # [B,H]，短窗口最后一个时刻的隐藏状态
        prediction = self.head(last_hidden)  # [B,1]，直接回归SOC而非强制学习残差
        if self.output_activation == "sigmoid":
            prediction = torch.sigmoid(prediction)
        return prediction.squeeze(-1)
