"""Stage 2c：用 HuggingFace tokenizers 在 TinyStories 子集上训练 ByteLevel BPE。

对比 2b：2b 是我们手写的"教学版"；这里用成熟的 tokenizers 库，两个关键升级：
    1. ByteLevel：在"字节"级别工作；配合 initial_alphabet 把全部 256 个字节
       预先固定进词表，任何 Unicode 文本都能无损编码（这是 GPT-2 的做法）；
    2. add_prefix_space：把单词开头的空格和单词本身粘在一起，
       避免像 2b 那样跨单词边界乱合并。

用法：
    python -m tokenizer.train_hf_bpe --vocab-size 8192 --max-stories 20000 \
        --out data/tokenizers/bpe_8k.json
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def story_stream(max_stories: int):
    """逐条产出 TinyStories 的故事文本（流式读取，只下载需要的分片）。"""
    from tokenizer.tinystories import load_stories

    for text in load_stories(max_stories):
        yield text


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ByteLevel BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--max-stories", type=int, default=20000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # 1) 底层算法选 BPE；unk_token 指定未知 token 的名字
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # 2) 预分词用 ByteLevel：把文本按"字节"切分，任意 Unicode 字符都能表示
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    # 3) 解码器也用 ByteLevel，和预分词配套
    tokenizer.decoder = decoders.ByteLevel()

    # 4) 训练器：规定目标词表大小、特殊 token、最低出现次数
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=args.min_frequency,
        # 关键：把 256 个字节全部预置进词表，保证对训练语料之外的文字也无损
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # 5) 从迭代器喂数据训练；story_stream 一条条吐出故事文本
    tokenizer.train_from_iterator(story_stream(args.max_stories), trainer=trainer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out))
    print(f"saved {out} | vocab_size={tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
