"""Stage 2a char tokenizer 的单元测试。

实现 tokenizer/char_tokenizer.py 里的三个 TODO 后，这里应全部通过。
运行方式（仓库根目录）：

    python -m pytest tests/test_char_tokenizer.py -v
"""

from tokenizer.char_tokenizer import SPECIAL_TOKENS, CharTokenizer


def test_roundtrip():
    # encode 再 decode 应该还原原文（词表里有这些字符时）
    tok = CharTokenizer("hello world")
    text = "hello"
    assert tok.decode(tok.encode(text)) == text


def test_vocab_learned_and_stable():
    tok = CharTokenizer("abracadabra")
    # 词表 = 语料里不重复的字符数 + 4 个特殊 token
    expected = len(set("abracadabra")) + len(SPECIAL_TOKENS)
    assert tok.vocab_size == expected

    # 同样的语料训练两次，词表必须完全一致（可复现）
    tok2 = CharTokenizer("abracadabra")
    assert tok.token_to_id == tok2.token_to_id


def test_unknown_char_maps_to_unk():
    tok = CharTokenizer("abc")
    unk_id = tok.token_to_id["<unk>"]
    # "z" 不在语料里，应该落成 <unk> 的 id
    assert tok.encode("z") == [unk_id]
    # 空字符串应该编码成空列表
    assert tok.encode("") == []
