"""Stage 7：SFT（监督微调）。

目标：把"只会续写的 base 模型"变成"会回答问题的 instruct 模型"。
核心：把对话序列化 + 只对 assistant 正文算 loss（masked cross entropy）。

用法：
    python -m train.sft --base-checkpoint runs/m55_16L512/checkpoint.pt \
        --out-dir runs/sft_m55 --n-samples 1500 --epochs 2
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer as HfTokenizer

from data.sft_dataset import build_sample, format_prompt_ids
from data.smoltalk import load_conversations
from inference.sample import load_pretrained, sample_next_token


EVAL_PROMPTS = [
    "What is a cat?",
    "What is 2 plus 2?",
    "Write a short story about a rabbit.",
    "Who is Lily?",
    "Tell me a joke about a banana.",
]


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """只对 mask=1 的位置算交叉熵，除以这些位置的个数。

    普通交叉熵会把 [B,T,V] 拉平成 [B*T,V] 求平均——它默认每个位置一样重要。
    这里先逐位置算出 loss，再乘 mask 把"不该学的"清零，最后除以有效个数。
    """
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",  # 不求和，保留每个位置的 loss
    ).reshape(targets.shape)
    n_valid = mask.sum().clamp(min=1)  # clamp 防止除 0
    return (ce * mask).sum() / n_valid


@torch.no_grad()
def generate_from_ids(model, ids: list[int], eos_id: int, max_new: int,
                      temperature: float, top_k: int, device: str) -> str:
    """从一串 prompt ids 出发生成回复（带 KV cache）。"""
    torch.manual_seed(0)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    caches = [() for _ in range(model.config.n_layer)]
    # 第一次喂整个 prompt，之后每次只喂最后 1 个 token
    logits, _, caches = model(idx, kv_caches=caches)
    for _ in range(max_new):
        logits, _, caches = model(idx[:, -1:], kv_caches=caches)
        nxt = sample_next_token(logits[:, -1, :], temperature, top_k)
        idx = torch.cat([idx, nxt], dim=1)
        if int(nxt) == eos_id:
            break
    return idx[0].tolist()


def run_evals(model, tokenizer, bos_id, eos_id, device) -> dict:
    answers = {}
    for q in EVAL_PROMPTS:
        ids = format_prompt_ids(q, tokenizer, bos_id, eos_id)
        out_ids = generate_from_ids(model, ids, eos_id, max_new=60,
                                    temperature=0.7, top_k=50, device=device)
        answers[q] = tokenizer.decode(out_ids)
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT a pretrained GPT")
    parser.add_argument("--base-checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--max-turns", type=int, default=4, help="每段对话取前几轮")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_pretrained(args.base_checkpoint, device)
    tokenizer = HfTokenizer.from_file("data/tokenizers/bpe_8k.json")
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    # ---- 建数据：每段对话取前几轮，序列化成 ids + mask ----
    conversations = load_conversations(args.n_samples)
    samples = []
    for conv in conversations:
        ids, mask = build_sample(
            conv[: args.max_turns], tokenizer, args.max_len, bos_id, eos_id, pad_id
        )
        if sum(mask) > 0:  # 丢掉没有任何 assistant 内容的空样本
            samples.append((ids, mask))
    print(f"usable samples: {len(samples)} / {len(conversations)}")

    split = int(len(samples) * 0.9)
    train_samples, val_samples = samples[:split], samples[split:]
    train_x = torch.tensor([s[0] for s in train_samples], dtype=torch.long)
    train_mask = torch.tensor([s[1] for s in train_samples], dtype=torch.float32)
    val_x = torch.tensor([s[0] for s in val_samples], dtype=torch.long)
    val_mask = torch.tensor([s[1] for s in val_samples], dtype=torch.float32)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ---- 训练前：先记录 base 模型对同样问题的"回答"（其实只会续写）----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    base_answers = run_evals(model, tokenizer, bos_id, eos_id, device)

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), args.batch_size):
            idx = perm[i : i + args.batch_size]
            x = train_x[idx].to(device)
            mask = train_mask[idx].to(device)

            optimizer.zero_grad()
            logits, _ = model(x)
            # 关键约定：位置 i 的 logits 预测的是"第 i+1 个 token"
            # （因果注意力 + 训练目标是输入的右移版本，和预训练一致）。
            # 所以 targets 取 x[:, 1:]，mask 也取 mask[:, 1:]——
            # 判断标准是"被预测的那个 token"是否属于 assistant 部分。
            loss = masked_cross_entropy(logits[:, :-1], x[:, 1:], mask[:, 1:])
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 20 == 0 or global_step == 1:
                print(f"epoch {epoch} step {global_step} | loss {loss.item():.4f}", flush=True)

        # ---- 每个 epoch 结束：验证 + 生成看行为变化 ----
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(val_x.to(device))
            val_loss = masked_cross_entropy(
                val_logits[:, :-1],
                val_x[:, 1:].to(device),
                val_mask[:, 1:].to(device),
            )
        sft_answers = run_evals(model, tokenizer, bos_id, eos_id, device)
        print(f"epoch {epoch} val_loss {val_loss.item():.4f}", flush=True)
        model.train()

    # ---- 保存最终模型（沿用 pretrain 的 checkpoint 结构，load_pretrained 可直接读）----
    torch.save(
        {
            "model": model.state_dict(),
            "base": args.base_checkpoint,
            "args": {
                "vocab_size": config.vocab_size,
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "block_size": config.block_size,
            },
        },
        out_dir / "checkpoint.pt",
    )

    # ---- 把 base vs SFT 的对照写进文件，也打一份到控制台 ----
    report = {"base": base_answers, "sft": sft_answers, "val_loss": float(val_loss)}
    with open(out_dir / "eval_output.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n================ BASE vs SFT ================")
    for q in EVAL_PROMPTS:
        print(f"\nQ: {q}")
        print(f"  base: {base_answers[q][:200]}")
        print(f"  sft : {sft_answers[q][:200]}")


if __name__ == "__main__":
    main()
