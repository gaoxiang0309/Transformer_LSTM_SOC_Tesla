# Transformer–LSTM SOC：Tesla Model 3 2170 行车SOC

本项目沿用论文《Collaborative framework of Transformer and LSTM for enhanced state-of-charge estimation》的两阶段特征融合结构，数据使用 McMaster University 公开的 **Tesla Model 3 2170**，本轮**只做行车状态 SOC**。这是方法适配，不是论文 CALCE A123 实验数字的复刻。

```text
长窗[V,I,T,Ah] → Transformer Encoder → 初估q[s]
短窗[V,I,T,Ah,q,anchor,ΔAh_60,ΔAh_300] → LSTM → α·anchor + (1−α)·q + 残差 → 当前SOC
```

## 1. 本轮任务边界（对齐《BIT_SOC_数据窗口与状态划分》）

| 决策 | 本项目的做法 |
|---|---|
| 行车 vs 充电 | **只训练行车**。Tesla 的 `Drive Cycles` 中 REORDERED/HWCUST/HWGRADE 文件进入数据集；CC-CV 充电文件整体排除，不进入训练/验证/测试 |
| 状态判据 | 按文件/循环类型（充电事件）划分，**不按电流符号**。行车中的正电流（回馈）、负电流（放电）、0 A、SOC 局部回升都保留 |
| 行车内部 | **融合**：不按“放电/回馈/短停车”再切子状态，单一行车模型 |
| 短停车 | 保留在序列内 |
| 长驻车 | 超过 `max_parking_seconds=1800 s` 的连续静置段从序列中切掉；短停车保留 |
| 时间清洗 | 原始文件并非严格 1 Hz：6 个文件存在亚秒级重复突发（共 44,271 行，最大两处为 m448-N 40℃ REORDERED1 和 m1000 40℃ HWGRADE1）。按 `Time` 重采样到约 1 Hz，并在时间差 >60 s 的缺口处切开序列 |
| 输入特征 | **V、I、T、Ah** 四维。第 4 维是安时积分 `Ah = ∫I dt / 4.5 Ah`，每段从 0 开始，只用电流、不读取真实 SOC |
| 物理锚（A） | 第二阶段输出 `SOC = α·anchor + (1−α)·q + residual`。`anchor = Q0 + a(T)·Ah`（积分侧），`q`（电压侧），`α` 是**可学习混合比例**（`anchor_mode=dual`，初始化≈1），让网络在锚与电压冲突时自动偏向电压侧 |
| 短窗 ΔAh（C） | LSTM 额外输入最近 60 s 与 300 s 的 ΔAh，直接看到近期充放电吞吐，缓解锯齿段平滑 |
| 扰动训练（方案一） | 训练集融合时给累计 Ah 注入电流零偏（默认 `ramp`：`b(t)=b0+tc·(T−25℃)`，`b0=±0.05 A`、`tc=±1 mA/℃` 温漂），标签保持作者 SOC 不变；验证/测试不注入。让模型学会用 V/I/T 纠正被污染的 Ah |
| 窗口 | **按物理秒定义**：长窗 300 s、短窗 60 s。Tesla 为 1 Hz，即 300 点/60 点；冷启动前 `L+K-2=358` 点无输出。上一版基线为 120 s/30 s |
| 窗口是否冻结 | 否。`long_window_seconds / short_window_seconds` 是配置项，等小会确认后再改 |

## 2. 数据

- 来源：`公开数据集/Tesla_Model3_2170/解压数据与工具`（McMaster Tesla 2170 开放数据，V3）。
- 电芯：`m80`（训练）、`m448-N`（验证）、`m1000`（测试）；`m448`（Ch2）未随包提供。
- 温度：−20、−10、0、10、25、40 ℃。
- 序列：每个行驶循环文件一条序列；被长驻车切开时拆成多条。窗口不跨文件、不跨切分边界。

| 集合 | 电芯 | 序列 | 点数 | 静置 | SOC上升（回馈） | SOC下降（放电） |
|---|---|---:|---:|---:|---:|---:|
| train | m80 | 60 | 1,208,237 | 9.0% | 20.1% | 70.9% |
| validation | m448-N | 60 | 1,362,581 | 14.7% | 25.0% | 60.3% |
| test | m1000 | 60 | 1,072,901 | 9.1% | 23.9% | 67.0% |

- 输入：`Voltage`、`Current`、`Battery_Temp_degC`，以及由电流积分得到的安时积分 `Ah`，顺序 **V, I, T, Ah**，1 Hz。
- 标签：作者按容量插值得到的 SOC 小数（0～1）。
- 归一化：V/I/T 只在 m80 训练集拟合 `robust_minmax`（p0.01–p99.99 截断后 min-max）；Ah 用固定物理范围 `[-1.2, 0.2]`（额定容量比例），避免测试段深放电被截断：

| 通道 | low | high | 训练截断 | 验证截断 | 测试截断 |
|---|---|---|---|---|---|
| V | 2.800 V | 4.201 V | 0.02% | 0.03% | 0.03% |
| I | −9.227 A | 7.542 A | 0.02% | 0.11% | 0.60% |
| T | −20.18 ℃ | 44.61 ℃ | 0.02% | 0.05% | 0.12% |
| Ah | −1.2 | 0.2 | 0.00% | 0.00% | 0.00% |

> 说明：行车数据的电流尾部比“含充电”数据窄；若用 p0.1，m1000 约 3% 的高电流样本会被截断，因此改用 p0.01（测试饱和约 0.6%），仍能剔除极端离群点。Ah 通道不再按训练分位数拟合，避免测试段深放电被截断。

## 3. 网络与训练

- Transformer：输入 4 维 `[V,I,T,Ah]`；`d_model=16`、2 头（每头 8 维）、3 层、`d_ff=128`、dropout 0.15、Post-LN、正弦位置编码，读取窗口末端，约 16,273 参数（基线为 `d_model=6`、2 层、`d_ff=64`、2,091 参数）。
- LSTM：输入 8 维 `[V,I,T,Ah,q,anchor,ΔAh_60,ΔAh_300]`，隐藏 64、1 层，末端接 FC 64→32→1 输出残差，约 25,217 参数（不含门控头，门控头仅 `H+1→1`）。
- 物理锚：`anchor[t] = (q[0] - a(T)·Ah[0]) + a(T)·Ah[t]`，`a(T)=4.5/C(T)`；C(T) 由 m80 训练集按温度拟合（−20℃ 4.209、−10℃ 4.241、0℃ 4.284、10℃ 4.361、25℃ 4.479、40℃ 4.583 Ah）。
- 锚的使用方式 `anchor_mode`：`hard`（`SOC=anchor+残差`，旧版）、`soft`（`SOC=残差`）、`gated`（`SOC=w·anchor+残差`）、`dual`（`SOC=α·anchor+(1−α)·q+残差`，双支路混合，α 由 `sigmoid(MLP([hidden, anchor−q]))` 门控、初始化≈1）。默认 **dual**。
- 两阶段冻结训练；RMSprop、MSE、Transformer batch 1024、LSTM batch 256、Transformer lr 5e-4、LSTM lr 1e-3、最多 400 轮、patience 20、梯度裁剪 1.0。
- 输出 `linear`，不做 sigmoid 截断；`use_physics_anchor`/`use_delta_ah`/`anchor_mode` 可分别配置做消融。
- 扰动训练（方案一，与门控软锚配合）：默认 `perturb_train=true`、`perturb_mode=ramp`——注入电流零偏 `b(t)=b0+tc·(T−25℃)`，`b0∈±perturb_amp(0.05 A)`、温漂系数 `tc∈±perturb_tc(0.001 A/℃)`，`Ah误差=∫b dt/3600/C`（约 0.6~1.6 pp / 约 2 h 行车）；`perturb_seed` 控制随机源，`t_norm_low/high` 用于把归一化 T 还原成℃。也可切 `step` 做固定 Ah 偏移。
- 权重复用：`init_transformer_from` / `init_lstm_from` 分别给 Transformer / LSTM 提供**预训练初始权重，随后仍继续训练/微调**（仅缩短训练，不是直接复用）。预训练权重已复制到项目内 `pretrained/`；LSTM 从旧 hard 权重 warm-start 用 `strict=False`（gate_head 保持 α≈1 初始化）。

## 4. 文件结构

```text
Transformer_LSTM_SOC_Tesla/
├── config.py                 配置（含秒制窗口、状态过滤、长驻车阈值）
├── configs/tesla_start.json  本轮正式实验配置
├── prepare_data.py           原始 MAT → 行车 NPZ + manifest（含归一化参数）
├── main.py                   两阶段训练入口
├── test.py                   独立测试入口（含覆盖率统计）
├── plot_results.py           测试曲线与训练 loss 绘图
├── model/                    Transformer、LSTM、训练工具
├── tools/                    数据加载、实验目录、评估、按状态汇总
├── data/processed/           已准备的行车 NPZ 与 manifest
├── docs/                     数据窗口、复现假设、代码复用与运行记录
├── tests/test_pipeline.py    时间对齐、因果性等检查
└── run/train/exp*/           训练产物（当前为空，服务器训练后生成）
```

## 5. 环境

- 训练/测试：PyTorch（`torch>=2.0`）；数据准备另需 SciPy，绘图需 matplotlib。
- 服务器上可使用现成的 conda 环境（例如 `Transformer_LSTM_SOC`）。

## 6. 数据准备（本机已执行，仓库内已带 `data/processed`）

```bash
python prepare_data.py --config configs/tesla_start.json
```

只有需要重新切分/改窗口/改归一化百分位时才需要重跑；已有 manifest 会拒绝覆盖，需先移走 `data/processed` 或用 `--output`。

## 7. 小规模检查

```bash
python -m unittest discover -s tests -v
python main.py --smoke
python test.py --run run/train/exp
```

`--smoke` 用 2 条截短序列、每阶段 1 轮，只验证链路，不能作为精度结果。

## 8. 正式训练、测试、绘图

服务器上的完整命令见本文末尾；流程是：

```bash
python main.py --config configs/tesla_start.json --device cuda:0
python test.py --run <RUN_DIR> --device cuda:0
python plot_results.py --test-dir <RUN_DIR>/test
python tools/summarize_predictions.py --test-dir <RUN_DIR>/test
```

`RUN_DIR` 以 `main.py` 打印的为准（例如 `run/train/exp`；若先跑过 smoke，则正式训练会新建 `exp1`）。两阶段各最多 400 轮，验证 MSE 连续 20 轮不改善即早停。

## 9. 主要输出

- `config.json`、`environment.json`：参数、环境、数据校验值、状态过滤与窗口秒数。
- `weights/transformer_best.pt`、`lstm_best.pt`：按验证 MSE 选择。
- `transformer_history.csv`、`lstm_history.csv`：逐轮训练/验证 MSE。
- `cache/`：冻结 Transformer 生成的 q 维（附权重 SHA-256）。
- `test*/predictions.csv`、`metrics.json`、`metrics_by_case.csv`、`summary_by_kind.csv`、`*.png`。
- `metrics.json` 含 `state_filter`、`window_seconds`、`first_valid_local_target=358`、`coverage_ratio` 等字段，便于如实报告冷启动覆盖。
- 指标单位为百分点（pp），原始预测不裁剪。

## 10. 已知限制

1. **窗口秒数未冻结**：300 s/60 s 是本轮实验值（上一版基线为 120 s/30 s）；确认后改 `long_window_seconds/short_window_seconds` 并重跑 prepare 即可。
2. **数据时间戳**：原始文件并非严格 1 Hz；6 个文件有亚秒级重复突发（共 44,271 行，主要两处为 m448-N/m1000 的 40℃ REORDERED1/HWGRADE1），已按 `Time` 重采样并在 >60 s 缺口处切开。此前版本按行号当 1 Hz，相关 case 的旧结果（含 exp1/基线）不能与新数据直接对比。
3. **跨电芯泛化**：m80→m448-N/m1000 严格不共享文件，但三颗电芯来自同一台架、同一套测试流程与同一 SOC 计算方式，不等于独立车辆/批次验证。
4. **公开循环有限**：UDDS/HWFET/LA92/US06 属于官方盲测集，本包只有 REORDERED/HWCUST/HWGRADE，不能与论文 US06/FUDS 数字直接对照。
5. **第二阶段 q 为样本内预测**：训练集 q 由冻结 Transformer 在同一训练集上生成，未做折外预测。
6. **安时积分与物理锚**：Ah 只由电流积分得到（不读取真实 SOC），但会显著降低“按安时计数跟踪 SOC”的难度；物理锚把 `Q0+a(T)·Ah` 作为基线，LSTM 学门控与残差。若要对照论文的纯 V/I/T 输入，把 `use_ah_feature` 设为 false（锚与ΔAh会自动不可用）并重跑 prepare。
7. **锚的初值与门控**：`Q0` 由段首 q 反推，段首 q 的偏差会以常数形式进入锚；`gated` 模式下门控 `w` 可视锚与 q 的差异而降权。当前模型的纠偏能力靠 `perturb_train` 扰动训练（step/ramp）来获得，验收用 P0-2 偏移注入（见 `docs/运行检查记录.md`）。
8. **未做超参搜索、多种子与更多基线**；所有数字为单次固定配置结果。

论文事实与假设见[复现假设](docs/复现假设.md)，数据与窗口细节见[窗口与数据说明](docs/窗口与数据说明.md)，代码来源见[代码复用说明](docs/代码复用说明.md)，本次运行记录见[运行检查记录](docs/运行检查记录.md)。

## 11. 服务器命令（Linux / conda）

```bash
# 0) 进入项目并激活环境
cd ~/projects/Transformer_LSTM_SOC_Tesla
conda activate Transformer_LSTM_SOC

# 1) 可选：检查依赖（matplotlib/ scipy 只在准备与绘图时需要）
python -c "import torch, numpy; print(torch.__version__, torch.cuda.is_available())"

# 2) 可选：重新准备数据（仓库已带 data/processed，一般可跳过）
#    若原始数据不在 ../公开数据集/Tesla_Model3_2170/解压数据与工具，
#    在 config.py 改 source_dir 或追加 --source "<Tesla_解压数据与工具路径>"
python prepare_data.py --config configs/tesla_start.json

# 3) 流程自检
python -m unittest discover -s tests -v
python main.py --smoke
python test.py --run run/train/exp

# 4) 正式训练（GPU；输出目录看 RUN_DIR 打印，例如 run/train/exp1）
python main.py --config configs/tesla_start.json --device cuda:0 2>&1 | tee run_train.log

# 5) 独立测试（把 <RUN_DIR> 换成上一步打印的目录）
python test.py --run <RUN_DIR> --device cuda:0 2>&1 | tee run_test.log

# 6) 绘图与分组汇总（<TEST_DIR> = <RUN_DIR>/test）
python plot_results.py --test-dir <TEST_DIR>
python tools/summarize_predictions.py --test-dir <TEST_DIR>
```

没有 GPU 时把 `--device cuda:0` 改为 `--device cpu`；设备也可以不写，默认 `auto`。
