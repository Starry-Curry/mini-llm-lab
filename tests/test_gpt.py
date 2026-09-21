"""Stage 4 手写 Transformer 的单元测试。"""

import math

import torch

from model.config import GPTConfig
from model.gpt import GPT, check_nan


def make_mini_config() -> GPTConfig:
    # 测试用小模型，CPU 上秒跑
    return GPTConfig(vocab_size=128, n_layer=2, n_head=2, n_embd=32, block_size=16)


def test_forward_shape_and_finite():
    torch.manual_seed(0)
    model = GPT(make_mini_config())
    idx = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 8, 128)
    assert loss.ndim == 0
    assert math.isfinite(loss.item())
    assert not check_nan(logits)


def test_causal_mask():
    torch.manual_seed(0)
    model = GPT(make_mini_config())
    model.eval()
    T = model.config.block_size
    idx = torch.randint(0, 128, (1, T))
    with torch.no_grad():
        logits1, _ = model(idx)
    # 把第 8 位之后的输入全部改掉
    idx2 = idx.clone()
    idx2[0, 8:] = torch.randint(0, 128, (T - 8,))
    with torch.no_grad():
        logits2, _ = model(idx2)
    # 位置 8 能看到"第 8 位本身"，而它被改掉了，所以 8 也会变；
    # 真正不受影响的是 0..7：它们只能看到自己之前的 token。
    assert torch.allclose(logits1[:, :8], logits2[:, :8], atol=1e-6)


def test_backward_flow():
    torch.manual_seed(0)
    model = GPT(make_mini_config())
    idx = torch.randint(0, 128, (2, 8))
    targets = torch.randint(0, 128, (2, 8))
    _, loss = model(idx, targets)
    model.zero_grad()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
    assert math.isfinite(model.check_gradient_norm())


def test_generate_greedy():
    torch.manual_seed(0)
    model = GPT(make_mini_config())
    model.eval()
    idx = torch.randint(0, 128, (1, 3))
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=5, top_k=1)
    assert out.shape == (1, 8)
    assert bool((out[:, :3] == idx).all())  # 前缀保持不变，只在末尾追加


def test_count_parameters():
    model = GPT(make_mini_config())
    expected = sum(p.numel() for p in model.parameters())
    assert model.count_parameters() == expected
