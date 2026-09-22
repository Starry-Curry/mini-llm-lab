"""Stage 6：推理与采样。

四个采样旋钮（所有 LLM 通用）：
    greedy      永远选分数最高的 token（最保守，容易复读）
    temperature 把 logits 除以温度：>1 更随机，<1 更确定
    top-k       只在分数最高的 k 个 token 里抽
    top-p       按分数从高到低累加，只在概率和达到 p 的那一小撮里抽

用法：
    python -m inference.sample --checkpoint runs/m5_4L256/checkpoint.pt \
        --prompt "Once upon a time" --max-new-tokens 200 \
        --temperature 0.8 --top-k 50
"""

import argparse
import time

import torch
import torch.nn.functional as F

from model.config import GPTConfig
from model.gpt import GPT


def load_pretrained(ckpt_path: str, device: str = "cpu"):
    """从 Stage 5 的 checkpoint 重建模型（配置也一并恢复）。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = ckpt["args"]
    config = GPTConfig(
        vocab_size=a["vocab_size"],
        n_layer=a["n_layer"],
        n_head=a["n_head"],
        n_embd=a["n_embd"],
        block_size=a["block_size"],
    )
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    """从最后一位置的 logits [B,V] 里按规则抽下一个 token id。"""
    logits = logits / max(temperature, 1e-8)

    if top_k is not None:
        # 只保留前 top_k 个高分，其余设为 -inf（softmax 后概率为 0）
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

    if top_p is not None:
        # top-p（核采样）：按分数从高到低排序，累加概率，
        # 保留"概率和刚好达到 p"的那批，其余清零
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = cumsum - probs > top_p
        sorted_logits[mask] = float("-inf")
        logits.scatter_(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)  # [B, 1]


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature=1.0,
             top_k=None, top_p=None, use_cache=True, device="cuda", seed=0) -> str:
    """自回归生成，返回解码后的文本。"""
    torch.manual_seed(seed)
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    # 每层一个"空缓存"，第一次增量解码时据此返回真实缓存
    caches = [() for _ in range(model.config.n_layer)]
    for _ in range(max_new_tokens):
        if use_cache:
            # 增量解码：只喂最后一个 token，其余靠 KV cache 复用
            idx_in = idx[:, -1:]
            logits, _, caches = model(idx_in, kv_caches=caches)
        else:
            # 朴素方式：每次重算整个前缀
            idx_in = idx[:, -model.config.block_size :]
            logits, _ = model(idx_in)

        next_id = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
        idx = torch.cat([idx, next_id], dim=1)
        if int(next_id) == tokenizer.token_to_id("<eos>"):
            break

    text = tokenizer.decode(idx[0].tolist())
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from a pretrained GPT")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true", help="对比有无 KV cache 的速度")
    args = parser.parse_args()

    from tokenizers import Tokenizer as HfTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_pretrained(args.checkpoint, device)
    tokenizer = HfTokenizer.from_file("data/tokenizers/bpe_8k.json")

    if args.benchmark:
        for use_cache in (False, True):
            t0 = time.time()
            generate(model, tokenizer, args.prompt, 300, temperature=1.0, top_k=1,
                     use_cache=use_cache, device=device, seed=0)
            dt = time.time() - t0
            print(f"use_cache={use_cache}: {dt:.2f}s -> {300/dt:.1f} tokens/s")

    text = generate(model, tokenizer, args.prompt, args.max_new_tokens,
                    args.temperature, args.top_k, args.top_p, use_cache=True,
                    device=device, seed=args.seed)
    print("=" * 60)
    print(text)


if __name__ == "__main__":
    main()
