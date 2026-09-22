"""SmolTalk 的 everyday-conversations 子集加载器。"""

import os


def load_conversations(n: int = 2000) -> list:
    """返回 n 段对话，每段是 [{role, content}, ...] 的消息列表。"""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(
        repo_id="HuggingFaceTB/smoltalk",
        filename="data/everyday-conversations/train-00000-of-00001.parquet",
        repo_type="dataset",
        # 设置 HF_HUB_OFFLINE=1 时只读本地缓存，不做网络检查
        local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
    )
    table = pq.read_table(path)
    out = []
    for row in table.to_pylist():
        messages = row.get("messages") or []
        if len(messages) >= 2:
            out.append(messages)
        if len(out) >= n:
            break
    return out
