"""Stage 2b 简化 BPE 的单元测试。"""

from tokenizer.bpe_tokenizer import SimpleBPE, get_stats, merge


def test_get_stats():
    assert get_stats([1, 2, 1, 2, 1, 2]) == {(1, 2): 3, (2, 1): 2}


def test_merge_replaces_pairs():
    assert merge([1, 2, 3, 1, 2], (1, 2), 7) == [7, 3, 7]
    # 三个 1 里只有前两个成对，第三个保持原样
    assert merge([1, 1, 1], (1, 1), 7) == [7, 1]


def test_bpe_roundtrip_and_vocab_growth():
    corpus = "low lower lowest"
    bpe = SimpleBPE(corpus, num_merges=10)
    # encode 再 decode 必须能无损还原
    assert bpe.decode(bpe.encode("lower")) == "lower"
    # 语料太短：9 次合并后整句已成一个 token，第 10 次没有相邻对可合并，
    # 循环提前退出。所以词表 = 8 个初始字符 + 实际完成的 9 次合并。
    assert len(bpe.merges) == 9
    assert bpe.vocab_size == len(set(corpus)) + len(bpe.merges)


def test_bpe_learns_merge_and_compresses():
    bpe = SimpleBPE("aaaa", num_merges=1)
    # 初始只有 'a'，合并一次后多了 'aa'，词表变 2
    assert bpe.vocab_size == 2
    # "aa" 两个字符合并成了一个 token，id 是 1
    assert bpe.encode("aa") == [1]
    assert bpe.decode([1]) == "aa"
