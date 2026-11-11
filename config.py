"""训练与数据配置，Tesla Model 3 2170 行车SOC版本。

本轮任务只做“行车状态”SOC：充电文件整体隔离，不进入训练/验证/测试；
行车内部（放电、回馈、短停车）整体保留，不按电流符号再切分。
窗口按物理秒定义，点数由采样间隔换算；Tesla数据为1 Hz，300 s/60 s对应300点/60点。
论文事实与实现假设仍见 docs/复现假设.md，已有实验以各自 config.json 快照为准。
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent


# dataclass自动生成初始化方法，使JSON字段能直接对应同名配置属性。
@dataclass
class Config:
    # 数据与运行位置：修改路径不会改变原始MAT文件。
    """集中保存项目参数；cfg变量就是本类的实例。字段名称与已保存的配置兼容。"""
    source_dir: str = str(
        ROOT.parent / "公开数据集" / "Tesla_Model3_2170" / "解压数据与工具"
    )
    processed_dir: str = str(ROOT / "data" / "processed")
    run_dir: str = str(ROOT / "run" / "train")
    seed: int = 42
    device: str = "auto"
    # 回归输出的范围约束。Tesla参考SOC可略超1，因此默认linear、允许越界。
    output_activation: str = "linear"
    # 数据划分：三个集合来自不同电芯，不共享任何原始文件。
    train_cells: list = field(default_factory=lambda: ["m80"])
    validation_cells: list = field(default_factory=lambda: ["m448-N"])
    test_cells: list = field(default_factory=lambda: ["m1000"])
    # 本轮只训练行车SOC：充电文件（CC-CV）不进入数据集。
    # 行车内部的正负电流、短停车全部保留，不按电流符号切子状态。
    include_charges: bool = False
    # 长驻车处理：按秒计，超过该长度的静置段从序列中切掉（0表示不处理）。
    # 短停车（例如10分钟）保留在行车序列内。
    max_parking_seconds: float = 1800.0
    # 原始文件并非严格1 Hz（个别文件有亚秒级重复突发）；按Time重采样到约1 Hz，
    # 相邻保留点时间差超过该秒数时切开，不跨缺口构窗（0表示不切）。
    max_gap_seconds: float = 60.0
    # 安时积分特征：把 ∫I dt / 额定容量 作为第4个输入通道，输入变为 [V, I, T, Ah]。
    # 该特征只由电流积分得到，不读取真实SOC；每条序列从0开始积分。
    use_ah_feature: bool = True
    capacity_ah: float = 4.5  # Tesla 2170 额定容量
    input_features: int = 4  # 由 use_ah_feature 在 validate() 中确定（3或4）
    # 温度相关容量表（由m80训练集按 SOC~Ah 斜率拟合），用于把Ah换算到SOC单位。
    # capacity_by_temperature 为实际可用容量(Ah)；scale=capacity_ah/C(T)。
    capacity_by_temperature: dict = field(
        default_factory=lambda: {
            "-20C": 4.209,
            "-10C": 4.241,
            "0C": 4.284,
            "10C": 4.361,
            "25C": 4.479,
            "40C": 4.583,
        }
    )
    # Ah特征的固定归一化范围（容量比例），prepare与融合阶段共用。
    ah_norm_low: float = -1.2
    ah_norm_high: float = 0.2
    # A：物理锚 + 残差输出。anchor = Q0 + scale(T)*Ah，Q0由段首q反推；
    # LSTM只学习 anchor 之外的残差。
    use_physics_anchor: bool = True
    # 物理锚的使用方式（仅在 use_physics_anchor=True 时生效）：
    # hard  = SOC = anchor + residual（旧版，锚权重恒为1）
    # soft  = SOC = residual（锚只作为输入特征，不再写死进输出）
    # gated = SOC = w·anchor + residual（方案B：可学习门控 w，w 初始化≈1）
    # dual  = SOC = α·anchor + (1−α)·q + residual（方案C：双支路混合，
    #         在“积分侧锚”与“电压侧估计q”之间学混合比例 α，α 初始化≈1）
    anchor_mode: str = "dual"
    # 第二阶段LSTM的warm-start权重路径（可选；留空则随机初始化）。
    # 从旧的hard锚权重继续训练时用 strict=False 载入，gate_head 保持近恒等初始化。
    init_lstm_from: str = ""
    # 第一阶段Transformer的预训练初始化路径（可选；留空则随机初始化）。
    # 各版本共用同一Transformer（T MAE 均1.298），设置后用它做初始权重并继续训练/微调（缩短训练）。
    init_transformer_from: str = ""
    # 扰动训练（方案一，与门控软锚配合）：仅在训练集融合时给累计Ah注入偏置，label保持作者SOC，
    # 让模型学会用V/I/T纠正被污染的Ah；验证/测试不注入（保持干净）。
    perturb_train: bool = True
    perturb_mode: str = "ramp"  # ramp=电流零偏(带温漂)；step=固定Ah偏移
    perturb_amp: float = 0.05  # ramp：基础电流零偏幅值(A, ±)；step：Ah比例幅值(±)
    perturb_tc: float = 0.001  # ramp：温漂系数(A/℃)，b(t)=b0+tc·(T−25℃)
    perturb_seed: int = 0  # 0=用cfg.seed；>0=固定随机源（复现）
    # T通道归一化范围（robust_minmax，与data/processed一致），用于把归一化T还原成℃算温漂。
    t_norm_low: float = -20.178
    t_norm_high: float = 44.607
    # C：给LSTM增加短窗ΔAh特征（短窗60s与长窗300s的安时变化）。
    use_delta_ah: bool = True
    # LSTM输入维度 = 输入特征 + q + (锚) + (2个ΔAh)，由validate()派生。
    lstm_input_features: int = 0
    # 归一化只在训练划分上拟合。
    # robust_minmax：每通道p0.01–p99.99截断后min-max到[0,1]；
    # zscore：训练集均值/标准差标准化，不截断。
    normalization: str = "robust_minmax"
    # 行车数据电流尾部比含充电数据更窄；p0.1会让m1000约3%的高电流样本饱和，
    # 因此用p0.01（测试饱和约0.9%），仍能剔除极端离群点。
    clip_percentile: float = 0.01
    # 窗口以物理秒定义。Tesla为1 Hz，300 s/60 s即300点/60点；
    # 若换采样率，只需改 sample_interval_s，点数由 validate() 换算。
    # 300/60 是本轮“加长窗口”的实验值；120/30 是上一版基线。
    sample_interval_s: float = 1.0
    long_window_seconds: float = 300.0
    short_window_seconds: float = 60.0
    # 以下两个点数字段由秒换算得到，validate() 会覆盖；保存快照时仅作记录。
    long_window: int = 300
    short_window: int = 60
    train_stride: int = 5  # 稀疏采样以控制单轮时间
    eval_stride: int = 20  # 验证/测试每20秒取一个目标时刻
    # Transformer结构：总宽度必须能被注意力头数整除。
    # 基线为d_model=6（2头×3），本轮为提升第一阶段容量改为16（2头×8）。
    d_model: int = 16
    n_heads: int = 2
    n_layers: int = 3  # 缺项：假设；本轮由2层加到3层
    d_ff: int = 128
    dropout: float = 0.15  # 容量增大后略提高dropout以抑制过拟合
    norm_order: str = "post"  # 论文图；pre 可复用示例项目的顺序
    norm_eps: float = 1e-6  # 将论文epsilon用于LN是实现选择
    # LSTM与回归头：两个FC之后还有一个输出维度为1的线性层。
    lstm_hidden: int = 64
    lstm_layers: int = 1
    fc1: int = 64
    fc2: int = 32
    # 训练设置：两阶段各自使用batch、学习率和epoch上限。
    transformer_batch: int = 1024
    lstm_batch: int = 256
    inference_batch: int = 256
    transformer_epochs: int = 400
    lstm_epochs: int = 400
    transformer_lr: float = 5e-4
    lstm_lr: float = 1e-3
    rmsprop_alpha: float = 0.99
    rmsprop_eps: float = 1e-8  # 与LN epsilon独立，不伪称论文确定值
    weight_decay: float = 0.0
    patience: int = 20  # 早停属于适配选择，0表示禁用
    grad_clip: float = 1.0
    num_workers: int = 0  # Windows先采用单进程加载
    # 小规模检查设置：用于验证程序链路，不用于报告估计精度。
    smoke: bool = False
    smoke_points: int = 256
    smoke_sequences: int = 2
    max_train_batches: int = 0
    max_eval_batches: int = 0

    def validate(self):
        """统一路径、由秒换算窗口点数并检查参数约束，返回当前配置对象。"""
        for name in ("source_dir", "processed_dir", "run_dir"):
            # getattr按字符串取属性；例如name为source_dir时，取出self.source_dir。
            path = Path(getattr(self, name))
            # 相对路径始终相对于项目，而不是命令行当前目录。
            if not path.is_absolute():
                absolute_path = (ROOT / path).resolve()
            else:
                absolute_path = path.resolve()
            setattr(self, name, str(absolute_path))
        if self.init_lstm_from:
            warm = Path(self.init_lstm_from)
            if not warm.is_absolute():
                warm = (ROOT / warm).resolve()
            self.init_lstm_from = str(warm)
        if self.init_transformer_from:
            warm_t = Path(self.init_transformer_from)
            if not warm_t.is_absolute():
                warm_t = (ROOT / warm_t).resolve()
            self.init_transformer_from = str(warm_t)
        if self.sample_interval_s <= 0:
            raise ValueError("sample_interval_s必须为正")
        if self.max_parking_seconds < 0:
            raise ValueError("max_parking_seconds不能为负")
        if self.max_gap_seconds < 0:
            raise ValueError("max_gap_seconds不能为负")
        if self.capacity_ah <= 0:
            raise ValueError("capacity_ah必须为正")
        if self.ah_norm_high <= self.ah_norm_low:
            raise ValueError("ah_norm_high必须大于ah_norm_low")
        if any(value <= 0 for value in self.capacity_by_temperature.values()):
            raise ValueError("capacity_by_temperature中的容量必须为正")
        if (self.use_physics_anchor or self.use_delta_ah) and not self.use_ah_feature:
            raise ValueError("物理锚和ΔAh特征要求use_ah_feature=True")
        if self.anchor_mode not in ("hard", "soft", "gated", "dual"):
            raise ValueError("anchor_mode需要是hard/soft/gated/dual")
        if self.perturb_mode not in ("step", "ramp"):
            raise ValueError("perturb_mode需要是step/ramp")
        if self.perturb_amp <= 0:
            raise ValueError("perturb_amp必须为正")
        if self.perturb_tc < 0:
            raise ValueError("perturb_tc不能为负")
        if self.t_norm_high <= self.t_norm_low:
            raise ValueError("t_norm_high必须大于t_norm_low")
        # 输入特征数由是否启用安时积分决定：3=[V,I,T]，4=[V,I,T,Ah]。
        self.input_features = 4 if self.use_ah_feature else 3
        # LSTM列顺序：[输入特征..., q, anchor, dAh_K, dAh_L]，按开关拼列。
        self.lstm_input_features = (
            self.input_features
            + 1
            + (1 if self.use_physics_anchor else 0)
            + (2 if self.use_delta_ah else 0)
        )
        # 窗口按秒定义：点数 = 秒数 / 采样间隔，四舍五入到整数点。
        self.long_window = max(1, int(round(self.long_window_seconds / self.sample_interval_s)))
        self.short_window = max(1, int(round(self.short_window_seconds / self.sample_interval_s)))
        if not (1 <= self.short_window < self.long_window):
            raise ValueError("需要 1 <= short_window < long_window")
        if self.d_model % self.n_heads or self.norm_order not in ("pre", "post"):
            raise ValueError("检查 d_model/n_heads 和 norm_order")
        if (
            min(
                self.train_stride,
                self.eval_stride,
                self.n_layers,
                self.transformer_batch,
                self.lstm_batch,
                self.inference_batch,
                self.transformer_epochs,
                self.lstm_epochs,
            )
            < 1
        ):
            raise ValueError("窗口步长、层数、batch、epoch必须为正")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout需要位于[0,1)")
        if self.output_activation not in ("linear", "sigmoid"):
            raise ValueError("output_activation需要是linear或sigmoid")
        if self.normalization not in ("robust_minmax", "zscore"):
            raise ValueError("normalization需要是robust_minmax或zscore")
        if not (0 <= self.clip_percentile < 50):
            raise ValueError("clip_percentile需要位于[0,50)")
        # 三个集合必须都非空且互不重叠，否则会破坏跨电芯验证的含义。
        splits = [set(self.train_cells), set(self.validation_cells), set(self.test_cells)]
        if any(not s for s in splits):
            raise ValueError("train/validation/test电芯不能为空")
        if splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2]:
            raise ValueError("train/validation/test电芯不能重叠")
        return self

    def smoke_config(self):
        """将当前配置改为少量真实数据的小规模检查设置，随后执行参数检查。"""
        self.smoke = True
        # smoke窗口同样用秒表示；1 Hz下16 s/4 s对应16点/4点。
        self.long_window_seconds = 16.0
        self.short_window_seconds = 4.0
        self.train_stride = 1
        self.eval_stride = 1
        self.n_layers = 1
        self.d_ff = 12
        self.lstm_hidden = 8
        self.fc1 = 8
        self.fc2 = 4
        self.transformer_batch = 8
        self.lstm_batch = 8
        self.inference_batch = 64
        self.transformer_epochs = 1
        self.lstm_epochs = 1
        self.max_train_batches = 2
        self.max_eval_batches = 2
        return self.validate()

    def save(self, path):
        """将实际配置保存为UTF-8 JSON快照，供后续接续训练和测试读取。"""
        Path(path).write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def load_config(path=None):
    """读取JSON覆盖默认参数；path为空时使用默认配置，并检查得到的配置。"""
    if path:
        # JSON里只需填写要覆盖的字段，其余字段沿用Config默认值。
        config_text = Path(path).read_text(encoding="utf-8")
        config_values = json.loads(config_text)
        # **把字典展开为命名参数，例如{"long_window": 100}对应long_window=100。
        cfg = Config(**config_values)
    else:
        cfg = Config()
    return cfg.validate()
