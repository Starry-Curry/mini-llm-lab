"""一晚跑多组不同规模的预训练，最后汇总并画 scaling 曲线。

每个模型给同样的 token 预算（25M），模型大小从 ~1M 到 ~55M，
最终 loss 随参数量的变化就是一条迷你 scaling law。
"""

import json
import time
from argparse import Namespace
from pathlib import Path

from train.pretrain import run


SUITE = [
    dict(name="m1_2L128",  n_layer=2,  n_head=4, n_embd=128, batch_size=64, lr=1e-3, max_hours=0.5),
    dict(name="m5_4L256",  n_layer=4,  n_head=4, n_embd=256, batch_size=32, lr=1e-3, max_hours=0.75),
    dict(name="m14_6L384", n_layer=6,  n_head=6, n_embd=384, batch_size=24, lr=6e-4, max_hours=1.0),
    dict(name="m23_8L448", n_layer=8,  n_head=8, n_embd=448, batch_size=16, lr=3e-4, max_hours=1.25),
    dict(name="m55_16L512", n_layer=16, n_head=8, n_embd=512, batch_size=8,  lr=3e-4, max_hours=2.0),
]

COMMON = dict(
    data="data/tinystories_full/train.bin",
    val_data="data/tinystories_full/val.bin",
    vocab_size=8192,
    block_size=256,
    max_tokens=25_000_000,
    min_lr=6e-5,
    warmup_ratio=0.02,
    buffer_size=4096,
    log_interval=25,
    val_interval=200,
    ckpt_interval=1000,
    seed=0,
    resume=False,
)


def main() -> None:
    summaries = []
    for i, cfg in enumerate(SUITE):
        args = Namespace(**{**COMMON, **cfg, "out_dir": f"runs/{cfg['name']}"})
        print(f"\n===== 启动第 {i+1}/{len(SUITE)} 个：{cfg['name']} =====", flush=True)
        try:
            summaries.append(run(args))
        except Exception as e:  # 单个失败不拖垮整晚，记下来继续跑下一个
            print(f"[{cfg['name']}] FAILED: {e}", flush=True)
            summaries.append({"name": cfg["name"], "error": str(e)})

    out = Path("runs/suite_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\n===== 全部完成，汇总写入 {out} =====", flush=True)

    # 画 scaling 曲线（matplotlib 可用才画，失败不影响结果文件）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ok = [s for s in summaries if "params" in s]
        ok.sort(key=lambda s: s["params"])
        xs = [s["params"] / 1e6 for s in ok]
        ys = [s["final_val_loss"] for s in ok]
        plt.figure(figsize=(7, 5))
        plt.loglog(xs, ys, "o-")
        for x, y, s in zip(xs, ys, ok):
            plt.annotate(s["name"], (x, y), textcoords="offset points", xytext=(6, 2), fontsize=8)
        plt.xlabel("parameters (M)")
        plt.ylabel("final val loss")
        plt.title("Mini scaling law on TinyStories (25M tokens each)")
        plt.grid(True, which="both", alpha=0.3)
        plt.savefig("runs/scaling_curve.png", dpi=150, bbox_inches="tight")
        print("scaling_curve.png saved", flush=True)
    except Exception as e:
        print(f"绘图跳过: {e}", flush=True)


if __name__ == "__main__":
    main()
