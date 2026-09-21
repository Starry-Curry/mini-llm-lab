"""TinyStories 数据的加载工具。

直接下载 Parquet 分片，按需切片读取，不依赖 datasets 库。
"""

# 全部 4 个训练分片（合计约 210 万篇故事）
TINYSTORIES_FILES = [
    "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
    "data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
    "data/train-00002-of-00004-a26307300439e943.parquet",
    "data/train-00003-of-00004-d243063613e5a057.parquet",
]


def _read_shard(filename: str, offset: int, n: int | None):
    """读一个分片的 [offset, offset+n) 行。"""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        repo_id="roneneldan/TinyStories",
        filename=filename,
        repo_type="dataset",
    )
    table = pq.read_table(path).slice(offset, n)
    return table.column("text").to_pylist()


def load_stories(n: int, offset: int = 0) -> list[str]:
    """返回 TinyStories 里第 offset 篇开始的 n 篇故事文本。"""
    return _read_shard(TINYSTORIES_FILES[0], offset, n)


def iter_stories_full(n: int, offset: int = 0):
    """跨全部分片，逐篇产出第 offset 篇开始的 n 篇故事（生成器，省内存）。"""
    produced = 0
    seen = 0
    for filename in TINYSTORIES_FILES:
        if produced >= n:
            break
        texts = _read_shard(filename, 0, None)  # 一个分片约 50 万篇，一次读入可接受
        shard_len = len(texts)
        lo = max(0, offset - seen)
        hi = min(shard_len, offset + n - seen)
        for text in texts[lo:hi]:
            yield text
            produced += 1
        seen += shard_len
