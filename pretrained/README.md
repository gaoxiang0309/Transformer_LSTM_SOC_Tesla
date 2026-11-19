# 项目内嵌权重

`exp_batch1_qah_control/` 保存实验三复用的 q_Ah 控制组最佳权重。

| 文件 | 用途 | SHA-256 |
|---|---|---|
| `exp_batch1_qah_control/transformer_best.pt` | q_Ah Transformer，输入 `[V,I,T,Ah]` | `e3cc67df311ec92c1610a041763ea1aee73e7fd9c0d6704e52e5e809a391e064` |
| `exp_batch1_qah_control/lstm_best.pt` | q_Ah dual LSTM，原输入8列 | `119d8208354b3eca086c28d783faa04ee4e32d64f3e3dfbd810191ac68ad5a50` |

实验三配置从这个目录加载权重，不依赖 `run/train/exp_batch1_qah_control` 是否存在。实验三扩展LSTM输入时会复用原LSTM输入投影的前8列，并初始化新增历史统计列。

实验四同样从这个目录加载 q_Ah 控制组权重。它扩展LSTM输入的前8列，新增的 `I_rms_60` 和 `I_pos_ratio_60` 列使用当前模型初始化。

项目根目录下的 `transformer_best.pt` 和 `lstm_best.pt` 保留为早期实验配置使用的权重。新实验应在配置中明确指定对应的权重目录。
