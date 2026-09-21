"""Stage 3：预训练数据管线的"清洗 / 过滤"部分。

真实语料从网上抓下来是脏的：有多余空白、奇怪换行、太短或太长的文本。
模型对这些东西很敏感（它会把噪声当规律去学），所以进模型前必须先洗一遍。
"""


def clean_text(text: str) -> str:
    """把文本洗成统一、干净的格式，返回新字符串（不修改原字符串）。"""
    # 1) 统一换行：Windows 的 \r\n、老 Mac 的 \r 都变成 \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2) 把每行内的连续空白（空格、制表符）折叠成单个空格。
    #    关键区别：split(" ") 按"单个空格"切，连续空格会留下空片段、不会折叠；
    #    而 split() 不带参数时按"任意连续空白"切，且自动丢掉空片段。
    #    所以先按换行切成行，行内用 " ".join(line.split()) 折叠，再按换行拼回。
    lines = [" ".join(line.split()) for line in text.split("\n")]
    text = "\n".join(lines)

    # 3) 把连续多个空行折叠成最多一个空行。
    #    while ... in ... 的意思是：只要还存在 "\n\n\n" 就一直替换。
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    # 4) strip()：去掉首尾空白
    return text.strip()


def filter_story(text: str) -> bool:
    """判断一篇文本要不要保留。返回 True 保留、False 丢弃。"""
    cleaned = clean_text(text)
    if len(cleaned) < 20:
        return False  # 太短，学不到什么
    if len(cleaned) > 2000:
        return False  # 太长（本数据集基本没有，但规则先立着）
    return True
