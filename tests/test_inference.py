"""Stage 6 推理的单元测试。"""

import torch

from inference.sample import sample_next_token
from model.config import GPTConfig
from model.gpt import GPT


def make_mini_config():
    return GPTConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=32, block_size=16)


def test_greedy_is_argmax():
    torch.manual_seed(0)
    logits = torch.tensor([[0.1, 0.5, 0.4, 0.9, 0.2]])
    out = sample_next_token(logits, temperature=1.0, top_k=1)
    assert out.item() == 3  # 分数最高的是第 3 号


def test_top_p_shapes_and_cumulative():
    torch.manual_seed(0)
    logits = torch.tensor([[5.0, 1.0, 1.0, 1.0, 0.1]])
    out = sample_next_token(logits, temperature=1.0, top_k=None, top_p=0.9)
    assert out.shape == (1, 1)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = GPT(make_mini_config())
    model.eval()
    T = model.config.block_size
    idx = torch.randint(0, 64, (1, T))

    with torch.no_grad():
        # 一次前向整个序列
        logits_full, _ = model(idx)
        # 增量：先喂前 T-1 个，再喂最后一个
        caches = [() for _ in range(model.config.n_layer)]
        _, _, caches = model(idx[:, :-1], kv_caches=caches)
        logits_last, _, _ = model(idx[:, -1:], kv_caches=caches)

    # 两种路径在最后位置的预测必须一致
    assert torch.allclose(logits_full[:, -1, :], logits_last[:, -1, :], atol=1e-5)


def test_cached_and_uncached_generation_agree():
    torch.manual_seed(0)
    model = GPT(make_mini_config())
    model.eval()
    from tokenizer.char_tokenizer import CharTokenizer

    tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz")

    def gen(use_cache):
        torch.manual_seed(0)
        ids = tokenizer.encode("ab")
        idx = torch.tensor([ids], dtype=torch.long)
        caches = [() for _ in range(model.config.n_layer)]
        for _ in range(8):
            if use_cache:
                logits, _, caches = model(idx[:, -1:], kv_caches=caches)
            else:
                logits, _ = model(idx)
            nxt = sample_next_token(logits[:, -1, :], temperature=1.0, top_k=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    with torch.no_grad():
        a, b = gen(False), gen(True)
    assert torch.equal(a, b)
