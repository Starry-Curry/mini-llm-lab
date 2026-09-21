"""Stage 3：把 TinyStories 变成可训练的 token 流。

流程：下载(已缓存) -> 清洗 -> 过滤 -> tokenize -> 拼接 -> 切块信息入库。

用法：
    python -m data.build_tinystories \
        --train-stories 20000 --val-stories 500 \
        --tokenizer data/tokenizers/bpe_8k.json \
        --seq-len 256 --out-dir data/tinystories
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer as HfTokenizer

from data.pipeline import clean_text, filter_story
from tokenizer.tinystories import iter_stories_full, load_stories


def build_stream(stories, tokenizer, eos_id: int):
    """清洗过滤 -> 逐篇 tokenize -> 每篇末尾补 <eos> -> 拼成一条长流。

    分块累积（攒够一批再拼进 numpy），避免把上亿个 id 的 Python 列表
    一次性塞进内存。
    """
    chunks = []
    kept = 0
    buf = []
    for text in stories:
        if not filter_story(text):
            continue
        buf.extend(tokenizer.encode(clean_text(text)).ids)
        buf.append(eos_id)  # 用 <eos> 标记"一篇故事到此结束"
        kept += 1
        if len(buf) > 2_000_000:
            chunks.append(np.array(buf, dtype=np.uint16))
            buf = []
    if buf:
        chunks.append(np.array(buf, dtype=np.uint16))
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.uint16), kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TinyStories token stream")
    parser.add_argument("--train-stories", type=int, default=20000)
    parser.add_argument("--val-stories", type=int, default=500)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--val-offset", type=int, default=30000)
    parser.add_argument("--tokenizer", type=str, default="data/tokenizers/bpe_8k.json")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default="data/tinystories")
    parser.add_argument("--full", action="store_true", help="用全部 4 个分片（约 210 万篇）")
    args = parser.parse_args()

    tokenizer = HfTokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<eos>")

    # 训练集和验证集用"不重叠的故事区间"，从源头杜绝数据泄漏
    if args.full:
        # 前 200 万篇训练，最后 10 万篇验证
        train_texts = iter_stories_full(2_000_000, 0)
        val_texts = iter_stories_full(args.val_stories, 2_000_000)
    else:
        train_texts = load_stories(args.train_stories, args.train_offset)
        val_texts = load_stories(args.val_stories, args.val_offset)

    train_ids, kept_train = build_stream(train_texts, tokenizer, eos_id)
    val_ids, kept_val = build_stream(val_texts, tokenizer, eos_id)

    train_arr = np.array(train_ids, dtype=np.uint16)  # 8192 词表，16 位足够
    val_arr = np.array(val_ids, dtype=np.uint16)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_arr.tofile(out / "train.bin")  # 原始二进制流，加载时用 fromfile
    val_arr.tofile(out / "val.bin")

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "seq_len": args.seq_len,
        "eos_id": eos_id,
        "train_tokens": int(len(train_arr)),
        "train_chunks": (len(train_arr) - 1) // args.seq_len,
        "train_stories_kept": kept_train,
        "val_tokens": int(len(val_arr)),
        "val_chunks": (len(val_arr) - 1) // args.seq_len,
        "val_stories_kept": kept_val,
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    # 抽查：把训练流开头 30 个 token 解码回文本，肉眼确认没问题
    print("开头 30 个 token 解码预览:")
    print(repr(tokenizer.decode(train_arr[:30].tolist())))


if __name__ == "__main__":
    main()
