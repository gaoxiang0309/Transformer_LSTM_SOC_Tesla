"""第一阶段模型：复用示例tf_model.py构建Encoder SOC回归网络。

输入[B,L,3]为长窗V/I/T，输出[B]为窗口末端SOC初估。
B是batch大小，L是窗口长度，D是模型宽度，h是注意力头数。
保留示例的注意力和FFN计算；适配连续输入、可配置归一化顺序和回归头。
代码来源与修改范围见docs/代码复用说明.md；本文件不包含Decoder。
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    """计算缩放点积注意力，返回加权value和注意力权重。

    query/key/value形状为[B,h,L,d_k]；mask为可广播到分数矩阵的有效位置标记。
    """
    d_k = query.size(-1)

    # 将key的最后两个维度互换(转置)，才能与query矩阵相乘，乘完了还要除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 此处的乘法规定了仅有后两个维度进行矩阵相乘，其余不变

    """
        Encoder：主要使用padding mask处理不等长序列
        Decoder：同时使用padding mask和sequence mask
        padding mask处理填充部分
        sequence mask防止看到未来信息
    """
    # 如果存在要进行mask的内容，则将那些为0的部分替换成一个很大的负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 将mask后的attention矩阵按照最后一个维度进行softmax，即行不变，对一行的所有列进行总和为1的分配
    p_attn = F.softmax(scores, dim=-1)

    # 如果dropout参数设置为非空，则进行dropout操作
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后返回注意力矩阵跟value的乘积，以及注意力矩阵
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """多头注意力模块，保留示例项目的Q/K/V投影与多头拼接实现。"""

    def __init__(self, h, d_model, dropout=0.1):
        """创建h个头的投影。d_model是总宽度，每头宽度d_k=d_model/h。"""
        super(MultiHeadedAttention, self).__init__()
        # 保证可以整除
        assert d_model % h == 0
        # 得到一个head的attention表示维度，一个头能够分到的维度
        self.d_k = d_model // h
        # head数量
        self.h = h
        # 定义4个全连接函数，供后续作为WQ，WK，WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 与课件中不同的地方，这里面多头注意力，每一个头用到的维度是d_model/h，对于w^q的一个头，维度是d_model*d_model/h，即dk=d_model/h
        # 本质上是一个全连接层，将全部头的全连接层拼成d_model*(d_model/h*h)的一个全连接层
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """将[B,L,D]拆成[B,h,L,d_k]计算注意力，再合并回[B,L,D]。"""
        if mask is not None:
            mask = mask.unsqueeze(1)
        # 增加一个head维度
        # query的第一个维度值为batch size
        nbatches = query.size(0)  # 读取一个batch的句子数
        # zip依次把Q/K/V输入与前三个Linear配对；第4个Linear保留用于合并后的输出。
        # view拆出head维，transpose把它移到时间轴之前：[B,L,D] -> [B,h,L,d_k]。
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换(具体过程见上述解析)
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # -1代表自动调整大小，全连接层运算后特征维度分配：d_model将其分为h*d_model/h
        # 交换维度变成batchsize*head*L*输入特征
        # 调用上述定义的attention函数计算得到h个注意力矩阵跟value的乘积，以及注意力矩阵
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # x的维度为batchsize*head*L*d_model/h，互换维度，contiguous用来调整内存，调整维度为batchsize*L*d_model
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """逐时间点应用的两层全连接网络，不改变时间轴长度。"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        位置前馈神经网络初始化函数
        参数:
            d_model: 模型的输入维度
            d_ff: 前馈神经网络中间层的维度
            dropout: dropout概率，默认为0.1
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一个线性层，将维度从d_model扩展到d_ff
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二个线性层，将维度从d_ff压缩回d_model
        self.dropout = nn.Dropout(dropout)  # dropout层，用于防止过拟合

    def forward(self, x):
        """依次执行Linear、ReLU、Dropout和Linear，将宽度从D扩展到d_ff后还原。"""
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def clones(module, N):
    """深拷贝N份模块形成ModuleList；各份参数独立，不共享同一个权重对象。"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    """正弦位置编码，为窗口内不同位置提供顺序信息；pe是非训练buffer。"""

    def __init__(self, d_model, dropout, max_len):
        """按最大窗口长度和模型宽度建立位置编码，并设置dropout。"""
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """把相同位置编码广播加到每个batch的输入上，保持[B,L,D]形状。"""
        if x.size(1) > self.pe.size(1):
            raise ValueError("输入超过位置编码长度")
        return self.dropout(x + self.pe[:, : x.size(1)])


class LayerNorm(nn.Module):
    """对最后一个特征维进行层归一化，a_2和b_2为可学习缩放与偏移。"""

    def __init__(self, features, eps=1e-6):
        """初始化特征维上的缩放、偏移和防止分母为零的epsilon。"""
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 在每个时间点的特征维上归一化，保留维度便于广播到[B,L,D]。
        """计算每个时间点特征的均值和总体方差，保持输入形状不变。"""
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True, unbiased=False)
        denominator = torch.sqrt(variance + self.eps)
        # 保留原来的乘除运算次序，兼容已有权重的数值结果。
        scaled_difference = self.a_2 * (x - mean)
        return scaled_difference / denominator + self.b_2


class SublayerConnection(nn.Module):
    """包装注意力或FFN子层，统一执行残差、dropout与层归一化。"""

    def __init__(self, size, dropout, order="post", eps=1e-6):
        """设置归一化宽度、dropout和pre/post连接顺序。"""
        super().__init__()
        self.norm = LayerNorm(size, eps)
        self.dropout = nn.Dropout(dropout)
        self.order = order

    def forward(self, x, sublayer):
        """调用传入的sublayer函数，按配置在残差连接之前或之后归一化。"""
        if self.order == "pre":
            # Pre-LN：先归一化，再执行注意力或FFN，最后加残差。
            normalized = self.norm(x)
            sublayer_output = sublayer(normalized)
            return x + self.dropout(sublayer_output)
        # Post-LN：先执行子层并加残差，再归一化；默认按论文图使用此顺序。
        sublayer_output = sublayer(x)
        residual_output = x + self.dropout(sublayer_output)
        return self.norm(residual_output)


class EncoderLayer(nn.Module):
    """一层Encoder：先多头自注意力，再逐位置FFN，每个子层都有残差连接。"""

    def __init__(self, size, self_attn, feed_forward, dropout, order, eps):
        """保存注意力与FFN，并为两个子层分别创建残差和归一化包装。"""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, order, eps), 2)

    def forward(self, x, mask=None):
        # lambda把注意力包装成“接收一个张量的函数”，让残差模块控制归一化位置。
        """让Q、K、V来自同一个窗口表示，顺序执行两个子层。"""
        x = self.sublayer[0](x, lambda a: self.self_attn(a, a, a, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """顺序堆叠多个独立EncoderLayer，保留pre/post两种归一化配置。"""

    def __init__(self, layer, count, order, eps):
        """复制count个编码层；Pre-LN时另外保留末端LayerNorm。"""
        super().__init__()
        self.layers = clones(layer, count)
        self.norm = LayerNorm(layer.size, eps) if order == "pre" else nn.Identity()

    def forward(self, x, mask=None):
        """逐层更新窗口表示，返回[B,L,D]的编码结果。"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerSOC(nn.Module):
    """第一阶段SOC回归网络：[B,L,F]输入转换为[B]初步SOC估计，F=3或4。"""

    def __init__(self, cfg):
        """创建连续输入投影、位置编码、Encoder和标量回归头，并初始化权重。"""
        super().__init__()
        # F=3为[V,I,T]，F=4为[V,I,T,Ah]；由config.input_features决定。
        self.input_features = cfg.input_features
        self.input_projection = nn.Linear(cfg.input_features, cfg.d_model)
        self.position = PositionalEncoding(cfg.d_model, cfg.dropout, cfg.long_window)
        layer = EncoderLayer(
            cfg.d_model,
            MultiHeadedAttention(cfg.n_heads, cfg.d_model, cfg.dropout),
            PositionwiseFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout),
            cfg.dropout,
            cfg.norm_order,
            cfg.norm_eps,
        )
        self.encoder = Encoder(layer, cfg.n_layers, cfg.norm_order, cfg.norm_eps)
        self.regression = nn.Linear(cfg.d_model, 1)
        self.output_activation = cfg.output_activation
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """编码只截止当前时刻的长窗口，读取窗口末端表示回归当前SOC。"""
        if x.ndim != 3 or x.shape[-1] != self.input_features:
            raise ValueError(f"Transformer输入必须为[B,L,{self.input_features}]")
        # [B,L,F] -> [B,L,D]：连续V/I/T(/Ah)先映射到模型宽度。
        projected = self.input_projection(x)
        positioned = self.position(projected)
        hidden = self.encoder(positioned)
        # 窗口只到当前目标时刻，取末端表示回归SOC，不使用未来输入。
        last_hidden = hidden[:, -1]
        prediction = self.regression(last_hidden)
        if self.output_activation == "sigmoid":
            prediction = torch.sigmoid(prediction)
        return prediction.squeeze(-1)  # [B,1] -> [B]，与标签形状一致


def make_model(cfg):
    """根据Config创建TransformerSOC；设备移动由训练/测试入口统一完成。"""
    return TransformerSOC(cfg)
