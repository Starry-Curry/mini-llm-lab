"""TinyStories 数据的加载工具。

直接下载第一个 Parquet 分片（约 52 万篇故事），按需切片读取，
不依赖 datasets 库、也不会下载整库。
"""


def load_stories(n: int, offset: int = 0) -> list[str]:
    """返回 TinyStories 里第 offset 篇开始的 n 篇故事文本。"""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        repo_id="roneneldan/TinyStories",
        filename="data/train-00000-of-00004-2d5a1467fff1081b.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(path).slice(offset, n)
    return table.column("text").to_pylist()
