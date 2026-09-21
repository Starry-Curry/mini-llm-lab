"""Stage 2b：简化版 BPE tokenizer。

一句话原理：
    反复把语料里"出现次数最多的相邻 token 对"合并成一个新 token，
    重复 N 次之后，常见组合（the、ing、tion...）就慢慢长出来了。

它和 2a（char tokenizer）的关系：
    - 2a 的每个 token 是一个字符；
    - 2b 的 token 可以是多个字符组成的块，所以同样的文本用更少的 token 就能表示。

工作流程：
    学习阶段（__init__）：把所有字符当作初始 token -> 循环 N 次：
        统计相邻对频率 -> 选出最高频的一对 -> 合并 -> 新 token 进词表
    使用阶段：encode 把文本按学到的合并规则转成 id；decode 反着拼回去。
"""


def get_stats(ids: list[int]) -> dict:
    """统计"相邻 id 对"各出现了多少次。

    参数：ids —— 一串 token 编号，例如 [1, 2, 1, 2, 1, 2]
    返回：{(左id, 右id): 次数}，例如 {(1, 2): 3, (2, 1): 2}
    """
    counts = {}
    # range(len(ids) - 1)：从 0 数到"长度减 2"。
    # 因为要看 ids[i] 和 ids[i+1] 两个相邻元素，最后一个元素后面没人了，不参与。
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])  # 把相邻两个 id 打包成一个元组 (左, 右)

        # 高级写法：
        counts[pair] = counts.get(pair, 0) + 1

        # 上面一行的低级等价写法（先判断在不在，再决定从 0 还是从旧值加）：
        #   if pair in counts:
        #       counts[pair] = counts[pair] + 1
        #   else:
        #       counts[pair] = 1
    return counts


def merge(ids: list[int], pair: tuple, new_id: int) -> list[int]:
    """把序列里所有连续出现的 pair 替换成 new_id。

    参数（三个，都是"位置参数"，按顺序传入）：
        ids    —— 要处理的 id 序列
        pair   —— 要合并的 (左id, 右id)，用 pair[0] 取左、pair[1] 取右
        new_id —— 合并后代表这一对的新 id

    例：merge([1,2,3,1,2], (1,2), 7) -> [7, 3, 7]
    """
    new_ids = []
    i = 0
    while i < len(ids):
        # 条件由 and 串起来，从左到右依次判断，一旦某个是 False 后面就不再看
        # （这叫短路），所以 ids[i+1] 不会越界。
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)  # 两个旧 id 合成一个
            i += 2                  # 跳过这两个被合并的位置
        else:
            new_ids.append(ids[i])  # 不成对，原样保留
            i += 1                  # 只看下一个位置
    return new_ids


class SimpleBPE:
    """用给定语料学习 num_merges 次合并的最简 BPE。"""

    def __init__(self, corpus: str, num_merges: int = 10):
        # ---- 第一步：初始词表 = 每个字符一个 id（和 2a 一样）----
        # 这一版先不加特殊 token；语料外的字符会在 encode 时报错（2c 再处理 unk）。
        chars = sorted(set(corpus))
        self.token_to_id = {}
        self.id_to_token = {}

        # enumerate(chars) 会逐个产出 (编号, 字符)，例如 (0, "a"), (1, "b")...
        for i, ch in enumerate(chars):
            self.token_to_id[ch] = i
            self.id_to_token[i] = ch

        # ---- 第二步：把语料整体转成 id 序列 ----
        # 列表推导式：[对 ch 做某事 for ch in corpus]，
        # 逐个字符查 id，结果收集成一个新列表。
        ids = [self.token_to_id[ch] for ch in corpus]

        # ---- 第三步：合并规则表 (左id, 右id) -> 新id，按学习顺序存放 ----
        self.merges = {}

        # ---- 第四步：主循环，合并 num_merges 次 ----
        for _ in range(num_merges):
            stats = get_stats(ids)
            if not stats:  # 语料已短到没有相邻对（或为空），提前结束
                break

            # 选出"出现次数最多"的 pair；
            # 次数相同时选编号更小的一对，保证每次运行结果完全一样（可复现）。
            # 元组比较规则：(a,b) < (c,d) 先比 a 和 c，相等再比 b 和 d。
            best_pair = None
            best_count = -1
            for pair, count in stats.items():
                # 高级写法（后面正文有讲解）：
                #   best_pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
                # 这里是它的低级写法：
                if count > best_count or (count == best_count and pair < best_pair):
                    best_pair = pair
                    best_count = count

            # 给这个新 token 发一个"目前还没人用过"的新 id
            new_id = len(self.token_to_id)
            self.merges[best_pair] = new_id

            # 新 token 的文字 = 左 token 文字 + 右 token 文字
            new_token = (
                self.id_to_token[best_pair[0]] + self.id_to_token[best_pair[1]]
            )
            self.token_to_id[new_token] = new_id
            self.id_to_token[new_id] = new_token

            # 在语料里执行这次合并，供下一轮继续统计
            ids = merge(ids, best_pair, new_id)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        """文本 -> id 序列。"""
        # 先逐字符转 id。不在词表里的字符直接报错并告诉你哪个字符有问题。
        ids = []
        for ch in text:
            if ch not in self.token_to_id:
                # f-string：f"..." 里的花括号会被替换成变量的实际值
                raise ValueError(f"字符 {ch} 不在词表里（2b 简化版不处理 unk）")
            ids.append(self.token_to_id[ch])

        # 再按"学习的先后顺序"依次应用每一条合并规则。
        # Python 的字典会记住插入顺序，所以 self.merges.items() 是按学习顺序出的。
        # items() 逐个产出 (键, 值)，这里 for 一次性拆成 pair 和 new_id 两个变量。
        for pair, new_id in self.merges.items():
            ids = merge(ids, pair, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """id 序列 -> 文本。"""
        # 每个 id 查回 token 文字，再用空字符串把它们直接拼起来
        return "".join(self.id_to_token[i] for i in ids)
