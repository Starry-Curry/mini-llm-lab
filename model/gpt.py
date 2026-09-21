"""Stage 4：手写 GPT-2 风格的 Decoder-only Transformer。

整体结构（数据从下往上流）：

    [B, T] token ids
      -> Token Embedding + Position Embedding -> [B, T, C]
      -> Block x N（每个 Block：LayerNorm -> 注意力 -> 残差；
                     LayerNorm -> MLP -> 残差）
      -> LayerNorm
      -> LM Head -> [B, T, vocab_size] logits

形状记号：B=batch 数，T=序列长度，C=n_embd，nh=头数，hs=每个头的维度。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import GPTConfig


def check_nan(tensor: torch.Tensor) -> bool:
    """张量里有没有 NaN（NaN 会让训练悄悄坏掉，必须能查出来）。"""
    return bool(torch.isnan(tensor).any().item())


class CausalSelfAttention(nn.Module):
    """因果自注意力：每个位置只能"看"自己和它左边的位置。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        # 一次线性变换同时算出 Q、K、V（3 个 C 宽的矩阵拼在一起）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 多头结果拼回来后，再投影回 C
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果掩码：下三角全 1、上三角全 0，形状 [1,1,block_size,block_size]。
        # register_buffer 表示它随模型一起 .to(device)，但不算可训练参数。
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # [B, T, C]

        # 1) 一次算出 QKV，再沿最后一维切成三段，各 [B, T, C]
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2) 切头：把 C 拆成 nh x hs，然后把"头"提到第 2 维。
        #    view   : [B, T, C] -> [B, T, nh, hs]
        #    transpose(1,2): -> [B, nh, T, hs]（每个头独立做注意力）
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 3) 注意力分数：Q 和 K 做内积，除以 sqrt(head_size) 防数值过大。
        #    k.transpose(-2,-1)：[B,nh,hs,T]；结果 att：[B,nh,T,T]，
        #    att[b,h,i,j] = "位置 i 对位置 j 的关注程度"。
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))

        # 4) 因果掩码：把"未来位置"的分数设成 -inf，softmax 后它们变成 0。
        #    self.bias 的下三角是 1、上三角是 0；==0 的位置就是要屏蔽的未来。
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # 5) 归一化成概率权重（每行和为 1），再对 V 加权求和：
        #    y[b,h,i,:] = sum_j softmax(att)[i,j] * v[b,h,j,:]
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # [B, nh, T, hs]

        # 6) 把多个头拼回 [B, T, C]，再过一层投影。
        #    transpose 之后内存不连续，view 前先 .contiguous()。
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return self.resid_dropout(y)


class MLP(nn.Module):
    """前馈网络：先把维度放大 4 倍，过激活函数，再缩回原维度。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # GPT-2 用的是 tanh 近似的 GELU
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """一个 Transformer Block：pre-norm + 注意力 + 残差，pre-norm + MLP + 残差。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # "x + f(ln(x))" 是残差连接：f 只学"需要修正的增量"，
        # 梯度能沿 x 这条短路直通回前面，深层网络才好训练。
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """把 Embedding、N 个 Block、最后的 LayerNorm 和 LM Head 串起来。"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token 词嵌入
                wpe=nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重绑定：输入嵌入和输出头共用同一张表（GPT-2 的做法）。
        # 直觉：一个 token 的"进入向量"和"预测分数向量"应该共享语义。
        self.transformer.wte.weight = self.lm_head.weight

        # 参数初始化
        self.apply(self._init_weights)
        # 残差投影层额外缩小：保证深网络起点方差稳定
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """输入 token ids [B,T]；输出 (logits [B,T,V], loss 或 None)。"""
        B, T = idx.shape
        assert T <= self.config.block_size, "序列长度超过 block_size"

        # 位置编号 0..T-1，两个嵌入相加
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)        # [B,T,C]
        pos_emb = self.transformer.wpe(pos)        # [T,C]，广播到 [B,T,C]
        x = self.transformer.drop(tok_emb + pos_emb)

        # 依次过 N 个 Block
        for block in self.transformer.h:
            x = block(x)

        # 最后的归一化和"翻译层"：把 C 维映射成词表分数
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # [B,T,V]

        loss = None
        if targets is not None:
            # 交叉熵要求 (N, 类别数) vs (N,)：把 [B,T,V] 拉平成 [B*T,V]
            # ignore_index=-1 表示 targets 里的 -1 位置不计 loss（预留的填充位）
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        """朴素自回归采样：每次生成一个 token，拼到末尾再继续。"""
        for _ in range(max_new_tokens):
            # 只喂最后 block_size 个 token
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            # 取最后一个位置的分数 [B,V]，除以温度（越大越随机）
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                # 只保留分数最高的 top_k 个，其余设为 -inf
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            # 按概率抽一个（top_k=1 时就是贪心解码）
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def count_parameters(self) -> int:
        """数一数总共有多少个可训练参数。numel() = 张量里元素的个数。"""
        return sum(p.numel() for p in self.parameters())

    def check_gradient_norm(self) -> float:
        """所有参数梯度的总 L2 范数，监控训练健康度的常用指标。"""
        total = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total += float((p.grad.detach() ** 2).sum())
        return math.sqrt(total)
