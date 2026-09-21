"""Stage 5：从零预训练。

把 Stage 3 的 token 流和 Stage 4 的 GPT 放进真正的训练循环：
AdamW + warmup/cosine 学习率 + 梯度裁剪 + checkpoint/resume + 指标记录。

用法：
    python -m train.pretrain --name m5 --data data/tinystories/train.bin \
        --val-data data/tinystories/val.bin --out-dir runs/m5 \
        --max-tokens 25000000 --batch-size 32
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from data.dataloader import shuffled_order
from model.config import GPTConfig
from model.gpt import GPT


def cosine_lr(step: int, base_lr: float, warmup_steps: int, total_steps: int, min_lr: float) -> float:
    """warmup 线性爬坡 + cosine 衰减到 min_lr。"""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def make_batch_iter(data: np.ndarray, seq_len: int, batch_size: int, buffer_size: int, seed: int):
    """无限产出 (x, y) 批次；每用完一轮就换新 seed 重洗，再从头来。"""
    n_chunks = (len(data) - 1) // seq_len
    epoch = 0
    while True:
        order = shuffled_order(n_chunks, buffer_size, seed + epoch)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            if len(idx) < batch_size:
                break
            x = np.stack([data[j * seq_len : (j + 1) * seq_len] for j in idx])
            y = np.stack([data[j * seq_len + 1 : (j + 1) * seq_len + 1] for j in idx])
            yield torch.from_numpy(x).long(), torch.from_numpy(y).long()
        epoch += 1


@torch.no_grad()
def evaluate(model, val_data, seq_len, batch_size, device, n_batches=10):
    """在验证集上取若干批，平均 loss（不更新参数）。"""
    model.eval()
    total = 0.0
    n = 0
    for x, y in make_batch_iter(val_data, seq_len, batch_size, buffer_size=512, seed=999):
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total += float(loss)
        n += 1
        if n >= n_batches:
            break
    model.train()
    return total / n


def save_checkpoint(path, model, optimizer, step, tokens, args):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "tokens": tokens,
            "args": vars(args),
        },
        path,
    )


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TF32：张量核加速，精度略低于 fp32，速度明显更快，业界预训练默认开
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_data = np.fromfile(args.data, dtype=np.uint16)
    val_data = np.fromfile(args.val_data, dtype=np.uint16)

    config = GPTConfig(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        dropout=0.0,
    )
    model = GPT(config).to(device)
    n_params = model.count_parameters()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    csv_path = out_dir / "log.csv"
    total_steps = math.ceil(args.max_tokens / (args.batch_size * args.block_size))
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    # ---- resume ----
    step, tokens_seen, ema_loss, t0 = 0, 0, 0.0, time.time()
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step, tokens_seen = ckpt["step"], ckpt["tokens"]
        print(f"[{args.name}] resume from step {step}, tokens {tokens_seen}")

    csv_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(
                ["step", "tokens", "loss", "lr", "val_loss", "tok_per_s", "gpu_mem_mb", "elapsed_s"]
            )

        iterator = make_batch_iter(
            train_data, args.block_size, args.batch_size, args.buffer_size, args.seed
        )
        last_ckpt = time.time()

        while step < total_steps:
            x, y = next(iterator)
            x, y = x.to(device), y.to(device)

            lr = cosine_lr(step, args.lr, warmup_steps, total_steps, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            # 梯度裁剪：防止某一步梯度爆炸把训练打飞
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            tokens_seen += int(x.numel())
            ema_loss = loss.item() if ema_loss == 0 else 0.99 * ema_loss + 0.01 * loss.item()
            elapsed = time.time() - t0

            if step % args.log_interval == 0 or step == 1:
                tok_per_s = tokens_seen / max(elapsed, 1e-6)
                mem = torch.cuda.max_memory_allocated() / 1e6
                val_loss = ""
                if step % args.val_interval == 0:
                    val_loss = f"{evaluate(model, val_data, args.block_size, args.batch_size, device):.4f}"
                writer.writerow([step, tokens_seen, f"{ema_loss:.4f}", f"{lr:.6f}",
                                 val_loss, f"{tok_per_s:.1f}", f"{mem:.0f}", f"{elapsed:.1f}"])
                f.flush()
                print(f"[{args.name}] step {step}/{total_steps} | loss {ema_loss:.4f} | "
                      f"lr {lr:.2e} | {tok_per_s:.0f} tok/s | {mem:.0f} MB | {elapsed:.0f}s",
                      flush=True)

            if step % args.ckpt_interval == 0 or time.time() - last_ckpt > 1800:
                save_checkpoint(ckpt_path, model, optimizer, step, tokens_seen, args)
                last_ckpt = time.time()

            # 时间上限保护：到点就优雅收尾，保证白天能看到结果
            if elapsed >= args.max_hours * 3600:
                print(f"[{args.name}] hit time budget {args.max_hours}h, stopping")
                break

    save_checkpoint(ckpt_path, model, optimizer, step, tokens_seen, args)
    val_loss = evaluate(model, val_data, args.block_size, args.batch_size, device)
    summary = {
        "name": args.name,
        "params": n_params,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "steps": step,
        "tokens": tokens_seen,
        "final_train_loss": round(ema_loss, 4),
        "final_val_loss": round(val_loss, 4),
        "tok_per_s": round(tokens_seen / max(time.time() - t0, 1e-6), 1),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[{args.name}] done: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a small GPT")
    parser.add_argument("--name", type=str, default="run")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=25_000_000)
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.02)
    parser.add_argument("--buffer-size", type=int, default=4096)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--val-interval", type=int, default=200)
    parser.add_argument("--ckpt-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
