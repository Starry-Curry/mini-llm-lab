"""Stage 7 SFT 的单元测试。"""

import torch

from data.sft_dataset import build_sample
from tokenizer.char_tokenizer import CharTokenizer
from train.sft import masked_cross_entropy


def make_tokenizer():
    return CharTokenizer("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?:")


def test_build_sample_mask():
    tok = make_tokenizer()
    bos = tok.token_to_id["<bos>"]
    eos = tok.token_to_id["<eos>"]
    pad = tok.token_to_id["<pad>"]

    messages = [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello you"},
    ]
    ids, mask = build_sample(messages, tok, max_len=40, bos_id=bos, eos_id=eos, pad_id=pad)

    assert len(ids) == len(mask) == 40
    # assistant 的"角色标记 + 正文 + <eos>"都要算 loss
    expected_sum = len(tok.encode("Assistant: ")) + len(tok.encode("hello you")) + 1
    assert sum(mask) == expected_sum

    # 助手正文 "hello you" 的 mask 全为 1
    hello_ids = tok.encode("hello you")
    for i in range(len(ids) - len(hello_ids) + 1):
        if ids[i : i + len(hello_ids)] == hello_ids:
            assert mask[i : i + len(hello_ids)] == [1] * len(hello_ids)

    # 用户正文 "hi there" 的 mask 全为 0
    hi_ids = tok.encode("hi there")
    for i in range(len(ids) - len(hi_ids) + 1):
        if ids[i : i + len(hi_ids)] == hi_ids:
            assert mask[i : i + len(hi_ids)] == [0] * len(hi_ids)

    # assistant 的角色标记也要 mask=1
    role_ids = tok.encode("Assistant: ")
    for i in range(len(ids) - len(role_ids) + 1):
        if ids[i : i + len(role_ids)] == role_ids:
            assert mask[i : i + len(role_ids)] == [1] * len(role_ids)

    # 尾部 pad 的 mask 必须为 0
    assert ids[-1] == pad and mask[-1] == 0


def test_masked_cross_entropy_ignores_masked_positions():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 10)
    targets = torch.randint(0, 10, (2, 5))
    mask = torch.zeros(2, 5)
    mask[:, 0] = 1  # 只有第 0 列参与 loss

    loss1 = masked_cross_entropy(logits, targets, mask)
    # 把 mask=0 的位置的目标全部改掉，loss 不应有任何变化
    targets2 = targets.clone()
    targets2[:, 1:] = (targets2[:, 1:] + 1) % 10
    loss2 = masked_cross_entropy(logits, targets2, mask)
    assert torch.allclose(loss1, loss2)


def test_targets_must_be_shifted():
    """回归测试：SFT 的 targets 必须右移一位，否则模型学的是"抄自己"。

    构造一个"完美自拷贝"的 logits（位置 i 给类别 x[i] 打极高分数）：
    - 用未移位的 targets，loss 几乎为 0（错的目标）；
    - 用右移的 targets，loss 巨大（这才暴露错误）。
    """
    torch.manual_seed(0)
    V, T = 10, 6
    x = torch.randint(0, V, (1, T))
    logits = torch.full((1, T, V), -100.0)
    for i in range(T):
        logits[0, i, x[0, i]] = 100.0
    mask = torch.ones(1, T)

    loss_unshifted = masked_cross_entropy(logits[:, :-1], x[:, :-1], mask[:, :-1])
    loss_shifted = masked_cross_entropy(logits[:, :-1], x[:, 1:], mask[:, :-1])
    assert loss_unshifted.item() < 1e-3
    assert loss_shifted.item() > 10
