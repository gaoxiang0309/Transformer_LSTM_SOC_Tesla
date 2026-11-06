"""自动检查脚本：验证时间索引、两阶段融合、因果性和权重重载。

使用人工构造的小数组与小模型，不运行400轮训练，也不依赖正式测试集调参。
运行方式：在项目目录执行python -m unittest discover -s tests -v。
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from config import Config
from model.tf_model import make_model
from model.lstm_model import LSTMSOC
from model.train_utils import make_fused_sequences, save_checkpoint, restore
from tools.data_loader import Sequence, WindowDataset


class PipelineTests(unittest.TestCase):
    """使用小型模型检查时序索引与训练管线的关键约束。"""

    def setUp(self):
        """每个用例开始前固定线程、随机种子和小模型配置。"""
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.cfg = Config().smoke_config()

    def sequence(self, n=40, value=0):
        """构造N行V/I/T输入、内部Ah辅助量与递增SOC标签。"""
        x = np.full((n, self.cfg.input_features), value, dtype="float32")
        y = np.arange(n, dtype="float32") / n
        ah = np.arange(n, dtype="float32") * 0.001
        return Sequence("example", x, y, ah=ah)

    def test_windows_never_cross_sequences(self):
        """检查两个不同数值的序列不会被同一个窗口混合，且标签对应末端。"""
        a, b = self.sequence(20, 1), self.sequence(20, 2)
        ds = WindowDataset([a, b], 16)
        self.assertEqual(len(ds), 10)
        self.assertTrue(torch.all(ds[4][0] == 1))
        self.assertTrue(torch.all(ds[5][0] == 2))
        self.assertEqual(ds[0][3], 15)
        self.assertAlmostEqual(ds[0][1].item(), float(a.y[15]))

    def test_fusion_alignment_and_no_truth_in_fourth_feature(self):
        """检查两级有效起点与样本数，并证明改变标签不会改变第四维q。"""

        class Dummy(torch.nn.Module):
            """可预测固定初估的模拟模型，便于检查第四维来源。"""

            def forward(self, x):
                """只从输入计算模拟q，不读取真实SOC标签。"""
                return x[:, -1, 0] + 0.125

        sequence = self.sequence()
        fused = make_fused_sequences(Dummy(), [sequence], self.cfg, torch.device("cpu"))[0]
        ds = WindowDataset([fused], self.cfg.short_window)
        self.assertEqual(self.cfg.input_features, 3)
        self.assertEqual(self.cfg.lstm_input_features, 7)
        self.assertEqual(fused.x.shape[1], 7)
        self.assertEqual(len(ds), 40 - 16 - 4 + 2)
        self.assertEqual(fused.offset + ds[0][3], 16 + 4 - 2)
        self.assertTrue(np.all(fused.x[:, self.cfg.input_features] == 0.125))
        changed = Sequence(sequence.name, sequence.x, sequence.y + 100, ah=sequence.ah)
        other = make_fused_sequences(Dummy(), [changed], self.cfg, torch.device("cpu"))[0]
        np.testing.assert_array_equal(fused.x, other.x)

    def test_future_input_does_not_change_past_estimate(self):
        """修改第25点之后的数据，确认更早的q不变且初估模型参数被冻结。"""
        model = make_model(self.cfg).eval()
        sequence = self.sequence(36)
        original = make_fused_sequences(model, [sequence], self.cfg, torch.device("cpu"))[0]
        altered = sequence.x.copy()
        altered[25:] = 100
        changed = make_fused_sequences(
            model,
            [Sequence("changed", altered, sequence.y, ah=sequence.ah)],
            self.cfg,
            torch.device("cpu"),
        )[0]
        # q追加在输入特征之后，索引为cfg.input_features。
        np.testing.assert_array_equal(
            original.x[:10, self.cfg.input_features],
            changed.x[:10, self.cfg.input_features],
        )
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_model_gradients_and_variable_window_length(self):
        """检查不同长度输入的输出形状、梯度有限性和每头key维度。"""
        model = make_model(self.cfg)
        for length in (8, 16):
            out = model(torch.randn(2, length, self.cfg.input_features))
            self.assertEqual(tuple(out.shape), (2,))
            out.square().mean().backward()
        self.assertTrue(
            all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        )
        # 每头key维度必须等于 d_model/n_heads；基线为3，本轮容量提升后为8。
        self.assertEqual(
            model.encoder.layers[0].self_attn.d_k, self.cfg.d_model // self.cfg.n_heads
        )
        lstm = LSTMSOC(self.cfg)
        self.assertEqual(
            tuple(lstm(torch.randn(2, 4, self.cfg.lstm_input_features)).shape), (2,)
        )

    def test_checkpoint_roundtrip(self):
        """保存并重载同一模型，要求固定输入的预测逐值一致。"""
        model = make_model(self.cfg).eval()
        x = torch.randn(2, 16, self.cfg.input_features)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best.pt"
            save_checkpoint(path, {"model": model.state_dict()})
            restored = make_model(self.cfg)
            restore(restored, path, torch.device("cpu"))
            torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
