"""Stage 2c：char / 手写BPE / HF 8K / HF 16K 对比实验。

对比指标：
    avg_tokens       平均每个故事被切成多少个 token（越少越好）
    chars_per_token  平均一个 token 覆盖几个字符（越大压缩越狠）
    failed_stories   无法编码的故事数（词表覆盖问题）
    unk_tokens       落到 <unk> 的 token 数
"""

from pathlib import Path

from tokenizers import Tokenizer as HfTokenizer

from tokenizer.bpe_tokenizer import SimpleBPE
from tokenizer.char_tokenizer import CharTokenizer


def load_stories(n: int, offset: int):
    """取 TinyStories 里 offset 之后的 n 个故事。"""
    from tokenizer.tinystories import load_stories as _load

    return _load(n, offset)


def measure(name, encode, texts, unk_id=None):
    """对一组文本统计编码指标。encode 是"文本 -> id 列表"的函数。"""
    total_chars = 0
    total_tokens = 0
    failures = 0
    unk_count = 0
    for t in texts:
        total_chars += len(t)
        try:
            ids = encode(t)
        except ValueError:
            failures += 1  # 词表里没有的字符：char 会落 unk，手写 BPE 会直接报错
            continue
        total_tokens += len(ids)
        if unk_id is not None:
            unk_count += ids.count(unk_id)
    n_ok = len(texts) - failures
    return {
        "name": name,
        "stories_ok": n_ok,
        "failed_stories": failures,
        "avg_tokens": round(total_tokens / n_ok, 1) if n_ok else None,
        "chars_per_token": round(total_chars / total_tokens, 2) if total_tokens else None,
        "unk_tokens": unk_count,
    }


def main() -> None:
    # 前 100 篇当"训练语料"，第 2000 篇起的 100 篇当"测试语料"（互相不重叠）
    # （手写 BPE 是纯 Python，语料和合并次数调小一点，跑得快；结论不变）
    train_texts = load_stories(100, offset=0)
    test_texts = load_stories(100, offset=2000)
    corpus = "\n".join(train_texts)

    # 2a 的 char tokenizer（在训练语料上学字符表）
    char = CharTokenizer(corpus)
    char_unk = char.token_to_id["<unk>"]

    # 2b 的手写 BPE（合并 300 次）
    simple = SimpleBPE(corpus, num_merges=300)

    # 2c 的 HF ByteLevel BPE（8K / 16K，需先跑 train_hf_bpe.py 生成）
    hf8 = HfTokenizer.from_file("data/tokenizers/bpe_8k.json")
    hf16 = HfTokenizer.from_file("data/tokenizers/bpe_16k.json")

    # lambda 是"一次性小函数"：输入 t，返回 hf8.encode(t).ids
    rows = [
        measure("char", char.encode, test_texts, unk_id=char_unk),
        measure("simple_bpe", simple.encode, test_texts),
        measure("hf_8k", lambda t: hf8.encode(t).ids, test_texts),
        measure("hf_16k", lambda t: hf16.encode(t).ids, test_texts),
    ]

    print(f"{'tokenizer':<12} {'ok':>4} {'fail':>5} {'avg_tokens':>11} {'chars/token':>12} {'unk':>6}")
    for r in rows:
        print(
            f"{r['name']:<12} {r['stories_ok']:>4} {r['failed_stories']:>5} "
            f"{str(r['avg_tokens']):>11} {str(r['chars_per_token']):>12} {r['unk_tokens']:>6}"
        )

    # 中文行为对比：训练语料是纯英文，看看各家对没见过的文字怎么办
    zh = "从前有一只小猫，它住在森林里。"
    print("\n中文句子:", zh)
    print("char      :", char.encode(zh))
    try:
        print("simple_bpe:", simple.encode(zh))
    except ValueError as e:
        print("simple_bpe: 直接报错 ->", e)
    print("hf_8k     :", hf8.encode(zh).ids)


if __name__ == "__main__":
    main()
