"""SFT 的核心数据逻辑：chat template 序列化 + assistant loss mask。

为什么需要 loss mask：
    一条训练样本里既有 user 的提问、也有 assistant 的回答。
    模型应该学会"如何回答"，而不是"如何提问"——提问是输入，不该学。
    所以每个 token 都配一个 mask：assistant 的"角色标记+正文+结尾<eos>"=1，
    其余（user 整轮、<bos>、<pad>）=0（不算 loss）。
"""

# 角色 -> 序列化时写进文本的标记
ROLE_TEXT = {"user": "User", "assistant": "Assistant", "system": "User"}


def _encode(tokenizer, text: str) -> list[int]:
    """统一编码接口：HF tokenizer 返回 Encoding（有 .ids），char tokenizer 返回列表。"""
    result = tokenizer.encode(text)
    return result.ids if hasattr(result, "ids") else result


def build_sample(messages: list, tokenizer, max_len: int, bos_id: int, eos_id: int, pad_id: int):
    """把一段对话变成 (token_ids, loss_mask)，两者长度相同且逐位对齐。

    序列化格式（每轮一条）：
        <bos> User: 内容 <eos>
        <bos> Assistant: 内容 <eos>
    mask 对 assistant 的"角色标记 + 内容 + 结尾 <eos>"置 1。
    """
    ids: list[int] = []
    mask: list[int] = []

    for message in messages:
        role = message.get("role", "user")
        is_assistant = role == "assistant"
        role_ids = _encode(tokenizer, f"{ROLE_TEXT.get(role, 'User')}: ")
        content_ids = _encode(tokenizer, str(message.get("content", "")))

        start = len(ids)
        ids += [bos_id] + role_ids + content_ids + [eos_id]
        mask += [0] * (len(ids) - start)

        if is_assistant:
            # 关键：角色标记本身也要置 1。最容易踩的坑是只 mask 正文——
            # 那样"标记的最后一个 token -> 正文第一个 token"这个衔接位置
            # 没人训练，模型在生成时就不知道该怎么开口（会乱吐空格）。
            role_lo = start + 1  # 跳过 <bos>
            content_lo = role_lo + len(role_ids)
            content_hi = content_lo + len(content_ids)
            # +1 把结尾 <eos> 也算上：教会模型"答完要闭嘴"
            for j in range(role_lo, content_hi + 1):
                mask[j] = 1

    # 超长截断（ids 和 mask 同步截，保持对齐）
    if len(ids) > max_len:
        return ids[:max_len], mask[:max_len]
    # 不足则用 <pad> 补齐，pad 的 mask 恒为 0
    n_pad = max_len - len(ids)
    return ids + [pad_id] * n_pad, mask + [0] * n_pad


def format_prompt_ids(question: str, tokenizer, bos_id: int, eos_id: int) -> list[int]:
    """推理时的提问模板：给出"用户问题 + 空白的助理开头"，让模型接着生成。"""
    user_ids = _encode(tokenizer, f"User: {question}")
    assistant_marker = _encode(tokenizer, "Assistant: ")
    return [bos_id] + user_ids + [eos_id] + [bos_id] + assistant_marker
