"""GPT 模型的配置。

用 dataclass 把超参数集中在一处：改配置只动这里，模型代码不用碰。
默认值就是文档推荐的 debug 配置（约 5M 参数）。
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 8192  # 词表大小（和 Stage 2 训出的 8K 词表一致）
    n_layer: int = 4        # 多少个 Transformer Block 堆叠
    n_head: int = 4         # 注意力头数
    n_embd: int = 256       # 每个 token 的向量维度（隐藏层宽度）
    block_size: int = 256   # 最大上下文长度（一个样本最多多少 token）
    dropout: float = 0.0    # debug 版先不开 dropout
    bias: bool = True       # GPT-2 的 Linear/LayerNorm 都带 bias
