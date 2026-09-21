"""Stage 3 数据管线的单元测试。"""

import numpy as np

from data.dataloader import BatchLoader, shuffled_order
from data.pipeline import clean_text, filter_story


def test_clean_text():
    raw = "  hello   world\n\n\nbye\tbye "
    assert clean_text(raw) == "hello world\n\nbye bye"


def test_filter_story():
    assert filter_story("") is False
    assert filter_story("short") is False
    assert filter_story("a" * 30) is True


def test_shuffled_order_is_permutation_and_deterministic():
    order = shuffled_order(20, buffer_size=4, seed=7)
    # 0..19 每个块恰好出现一次
    assert sorted(order) == list(range(20))
    # 同样 seed 两次结果完全一致（可复现）
    assert order == shuffled_order(20, buffer_size=4, seed=7)


def test_batch_loader_shapes_and_shift():
    stream = np.arange(1000, dtype=np.uint16)
    loader = BatchLoader(stream, seq_len=16, batch_size=4, buffer_size=8, seed=0)
    n = loader.n_chunks()
    assert n == (1000 - 1) // 16

    seen_starts = set()
    n_samples = 0
    for x, y in loader:
        assert x.shape == (4, 16) and y.shape == (4, 16)
        # y 是 x 右移一位：y[:, :-1] 必须等于 x[:, 1:]
        assert bool((y[:, :-1] == x[:, 1:]).all())
        seen_starts.update(x[:, 0].tolist())
        n_samples += x.shape[0]

    # 数据值互不相同，所以每个块的开头 token 都是唯一的，能据此查重
    assert len(seen_starts) == n_samples
