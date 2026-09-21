"""Stage 2a：最简单的 char tokenizer。

回答一个问题：模型只能吃数字，文本怎么变成数字？

思路：把每个"字符"当作一个 token，给每个字符发一个编号（id）。

    "hello" -> ['h', 'e', 'l', 'l', 'o'] -> [id(h), id(e), id(l), id(l), id(o)]

两个方向：
    encode(text)  文本 -> id 列表
    decode(ids)   id 列表 -> 文本

特殊 token 的 id 固定（顺序就是 id）：
    0 = <bos> 开头   1 = <eos> 结尾   2 = <pad> 填充   3 = <unk> 未知
普通字符从 4 开始编号。

约定（测试按这个检查）：
    - 这一版 encode 不加 <bos>/<eos>，只做逐字符转 id；
    - 遇到词表里没有的字符，一律映射成 <unk>；
    - 同样的语料训练两次，词表必须完全一样（可复现）。
"""

# 特殊 token 列表。顺序就是它们的 id：<bos>=0, <eos>=1, <pad>=2, <unk>=3
SPECIAL_TOKENS = ["<bos>", "<eos>", "<pad>", "<unk>"]


class CharTokenizer:
    """把每个字符当作一个 token 的最小 tokenizer。"""

    def __init__(self, corpus: str):
        """从语料里"学习"词表：收集语料中出现过的所有字符。"""
        # 步骤 1：收集语料里出现过的所有字符，去重 + 排序
        #   set(corpus)：把字符串转成集合，重复字符只留一个；
        #   sorted(...)：排成固定顺序，保证两次训练结果一致（可复现）。
        chars = sorted(set(corpus))

        # 步骤 2：完整词表 = 4 个特殊 token + 所有普通字符
        #   列表相加就是拼接，所以特殊 token 固定占 0~3 号，普通字符从 4 开始。
        vocab = SPECIAL_TOKENS + chars

        # 步骤 3：建两张互相相反的"对照表"（都是字典）：
        #   一张：token -> id（编码时用，字符查编号）
        #   一张：id -> token（解码时用，编号查字符）
        #   enumerate(vocab) 会逐个产出 (编号, 元素)，例如 (0, "<bos>")、(4, "a")...
        #   {键: 值 for ...} 是字典推导式，一行造出一整个字典。
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for i, token in enumerate(vocab)}

    @property
    def vocab_size(self) -> int:
        """词表大小 = 对照表里有多少项。（这部分已填好，不用动。）"""
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        """把文本转成 id 列表。"""
        # 逐个字符查它的 id，组成列表返回。
        #
        # dict.get(键, 默认值)：
        #   键在字典里   -> 返回它对应的值；
        #   键不在字典里 -> 返回默认值（这里就是 <unk> 的 id），绝不报错。
        #
        # 列表推导式：[对 ch 做某事 for ch in text]，
        # 意思是"把 text 里每个字符都处理一遍，结果装进一个新列表"。
        return [
            self.token_to_id.get(ch, self.token_to_id["<unk>"])
            for ch in text
        ]

    def decode(self, ids: list[int]) -> str:
        """把 id 列表转回文本。"""
        # 每个 id 换回字符，再用空字符串把它们一个个直接拼起来。
        #   self.id_to_token[i] 是"编号 i 对应的字符"；
        #   "".join(一串字符) 把这一串字符拼成一个字符串：
        #   "".join(["h", "i"]) -> "hi"，前面的 "" 是拼接分隔符（空=直接相连）。
        return "".join(self.id_to_token[i] for i in ids)
