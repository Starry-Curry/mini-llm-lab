"""Stage 3：预训练数据管线的"打包 / 洗牌 / 批处理"部分。"""

import random

import numpy as np
import torch


def shuffled_order(n_chunks: int, buffer_size: int, seed: int = 0) -> list[int]:
    """返回 0..n_chunks-1 的一个"近似打乱"的访问顺序。

    为什么需要它：理想情况下每个 epoch 应该把全部数据彻底洗牌，
    但真实预训练数据是流式读进来的（装不下内存），无法全局洗牌。
    工程上的折中是"洗牌缓冲"：只在大小为 buffer_size 的窗口内打乱。
    窗口越大越接近全局随机，但越占内存。

    参数：
        n_chunks    一共多少块
        buffer_size 窗口大小（组内打乱的范围）
        seed        随机种子，固定后每次运行得到同样的顺序（可复现）
    """
    order = list(range(n_chunks))
    rng = random.Random(seed)  # 独立的随机源，不影响全局 random

    # 每 buffer_size 个索引为一组，组内打乱，组与组之间保持原顺序
    for start in range(0, n_chunks, buffer_size):
        window = order[start : start + buffer_size]  # 切出一个窗口（是拷贝）
        rng.shuffle(window)                          # 组内洗牌
        order[start : start + buffer_size] = window  # 把洗好的窗口放回去
    return order


class BatchLoader:
    """把一条长 token 流切成固定长度的训练样本，再组批，产出 (x, y)。"""

    def __init__(
        self,
        data: np.ndarray,  # 一维 uint16 token 流
        seq_len: int,      # 每样本的 token 数（上下文长度）
        batch_size: int,   # 每批多少样本
        buffer_size: int,  # 洗牌窗口大小
        seed: int = 0,
    ):
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.seed = seed

    def n_chunks(self) -> int:
        # 最后一个块的 y 要比 x 多往后读 1 位，所以用 (len-1)//seq_len，
        # 保证永远不会越界；尾部不足一块的零头直接丢弃（浪费 < seq_len 个 token）。
        return (len(self.data) - 1) // self.seq_len

    def __iter__(self):
        order = shuffled_order(self.n_chunks(), self.buffer_size, self.seed)

        # 每 batch_size 个块组成一批；最后不足一批的丢弃，保持张量形状整齐
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            if len(idx) < self.batch_size:
                break

            # x[j] = 第 j 块；y[j] = 同一块整体右移 1 位（"预测下一个 token"）
            x = np.stack(
                [self.data[j * self.seq_len : (j + 1) * self.seq_len] for j in idx]
            )
            y = np.stack(
                [
                    self.data[j * self.seq_len + 1 : (j + 1) * self.seq_len + 1]
                    for j in idx
                ]
            )

            # from_numpy 和 numpy 数组共享内存、零拷贝；.long() 转成 int64 张量
            yield torch.from_numpy(x).long(), torch.from_numpy(y).long()
