"""Stage 1：最小训练骨架（框架版）。

学习目标：亲手补齐一个完整的 PyTorch 训练闭环。

凡是写着 TODO 的地方，需要你补上代码；注释里已经写清楚：
这一步在算什么、结果是什么形状、可以用哪个函数。

训练闭环全景（后面所有复杂训练都是它的变体）：

    for x, y in loader:          # 1. 取一批数据
        pred = model(x)          # 2. 前向：算出预测值
        loss = mse(pred, y)      # 3. 算 loss：错得多离谱
        loss.backward()          # 4. 反向：自动算出每个参数的梯度
        optimizer.step()         # 5. 更新：把参数往"错得更少"的方向挪
        optimizer.zero_grad()    # 6. 清梯度：防止和下一批的梯度累加

验收标准：
    1. python -m pytest tests/test_stage1.py -v  全绿
    2. python -m train.minimal_loop --epochs 200 能看到 loss 明显下降
    3. 中途 Ctrl+C 后加 --resume 能接着上次的 epoch 继续
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# 1. 数据集：负责"给我第 i 条样本"
# ============================================================================


class SyntheticDataset(Dataset):
    """人造的 (x, y) 回归数据：输入 x 是 8 维向量，输出 y 是一个标量。"""

    def __init__(self, n: int = 1000, seed: int = 0, noise: float = 0.05):
        # 保存两个配置参数，供生成数据时使用
        self.n = n
        self.noise = noise

        # 用 torch.Generator 固定随机数来源：
        # 只要 seed 相同，生成的数据就完全相同，这就是"可复现"。
        # 不要用全局的 torch.rand，否则不同数据集之间会互相影响。
        generator = torch.Generator()
        generator.manual_seed(seed)

        # 一次性把 n 条数据全部生成好，存成一张"表"。
        # 这样 ds[i] 每次都返回同一条数据，直观、也方便测试验证。
        self.samples = []
        for _ in range(n):  # _ 表示"这个循环变量我用不到"，只是要循环 n 次
            # 1) 生成 x：8 个 [0,1) 的随机数 -> 乘 2 得 [0,2) -> 减 1 得 [-1,1)
            #    * 和 - 会对张量里每一个数都做一次（逐元素运算）
            x = torch.rand(8, generator=generator) * 2 - 1

            # 2) 生成一个高斯噪声标量：() 表示 0 维（就是一个单独的数）
            eps = torch.randn((), generator=generator)

            # 3) 按公式算出目标值：
            #    x[0] 是取第 0 个元素（索引从 0 开始数）
            #    self.noise 是 __init__ 里保存的参数，用 self. 才能在这里拿到
            y = x[0] * x[1] + torch.sin(x[2]) + self.noise * eps

            # 4) 把这条 (x, y) 追加进列表末尾
            self.samples.append((x, y))

    def __len__(self) -> int:
        # PyTorch 用 len(dataset) 来知道一共有多少条数据。
        # （这部分已经填好了，不用动。）
        return self.n

    def __getitem__(self, index: int):
        """返回第 index 条样本 (x, y)。index 参数本课用不到，可以直接忽略。"""
        # 数据在 __init__ 里已经生成好了，这里只要按行号取出来
        return self.samples[index]


# ============================================================================
# 2. 模型：负责"从 x 算出预测值"
# ============================================================================


class TinyMLP(nn.Module):
    """两层小网络：x[8] -> Linear(8→32) -> ReLU -> Linear(32→1)。"""

    def __init__(self, in_dim: int = 8, hidden: int = 32, out_dim: int = 1):
        super().__init__()
        # nn.Linear(in, out) 内部自带可学习参数 W 和 b，公式是 y = x @ W^T + b。
        # 只要把层赋给 self.fc1、self.fc2，PyTorch 就会自动把它俩注册到
        # model.parameters() 里，优化器才能更新它们。
        # （这部分已经填好了，不用动。）
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 x: [B, 8]，输出预测值: [B, 1]。"""
        # 第一层：x [B,8] -> h [B,32]，然后 ReLU 把负数变成 0
        h = F.relu(self.fc1(x))
        # 第二层：h [B,32] -> [B,1]，直接返回作为预测值
        return self.fc2(h)


# ============================================================================
# 3. 训练 / 验证一个 epoch
# ============================================================================


def train_epoch(model, loader, optimizer, device) -> float:
    """把整个训练集过一遍，返回平均 loss。这是本课的核心函数。"""
    model.train()  # 训练模式。现在模型里没有 dropout/BN 所以没区别，但习惯要养成。
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        # 1) 把数据和标签搬到和模型同一个设备（GPU 或 CPU）
        x = x.to(device)
        y = y.to(device)

        # 2) 先清掉上一批数据留下的旧梯度，防止累加
        optimizer.zero_grad()

        # 3) 前向传播，得到预测值 [B, 1]
        pred = model(x)

        # 4) 算 loss。squeeze(-1) 把 [B,1] 压成 [B]，和 y 的形状对齐
        loss = F.mse_loss(pred.squeeze(-1), y)

        # 5) 反向传播：自动算出每个参数的梯度，存在各自 .grad 里
        loss.backward()

        # 6) 优化器用梯度更新参数
        optimizer.step()

        # 7) 统计：item() 把张量转成普通数字；+= 就是"自身加上右边的值"
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, loader, device) -> float:
    """验证集上只算 loss、不更新参数，返回平均 loss。"""
    model.eval()  # 验证模式
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        # no_grad 的意思是：下面这段代码不记录计算图。
        # 因为我们不 backward，不记录能省内存、也避免误更新。
        for x, y in loader:
            # 和 train_epoch 一样的四步：搬数据 -> 前向 -> 算 loss -> 统计。
            # 唯独没有 zero_grad / backward / step：验证阶段只看成绩、不改参数。
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = F.mse_loss(pred.squeeze(-1), y)

            total_loss += loss.item()
            n_batches += 1
    return total_loss / n_batches


# ============================================================================
# 4. 存盘 / 读档
# ============================================================================


def save_checkpoint(path, model, optimizer, epoch) -> None:
    """把模型、优化器、训练进度打包存成一个 .pt 文件。"""
    # 1) 先确保存放目录存在，否则 torch.save 会因为找不到目录而报错。
    #    Path(path)：把字符串路径变成好用的路径对象；
    #    .parent：取它的父目录（checkpoints/minimal.pt -> checkpoints）；
    #    mkdir(parents=True, exist_ok=True)：建目录，多层一起建，已存在也不报错。
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # 2) 把三样东西装进一个字典，一次性存盘：
    #    "model"      -> model.state_dict()      模型全部参数（权重 W、偏置 b）
    #    "optimizer"  -> optimizer.state_dict()  优化器内部状态（AdamW 的 m/v 等）
    #    "epoch"      -> epoch                   练到第几个 epoch
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(path, model, optimizer) -> int:
    """从 .pt 文件恢复模型和优化器，返回恢复到的 epoch。"""
    # 1) 把刚才存的那个字典读回来。
    #    map_location="cpu"：先把张量读回 CPU，之后由调用方决定放哪个设备，
    #    这样换机器、换设备都能读。
    ckpt = torch.load(path, map_location="cpu")

    # 2) 把权重按名字灌进模型，覆盖掉模型里当前的随机参数
    model.load_state_dict(ckpt["model"])

    # 3) 把优化器内部状态也灌进去（不存它的话，续训效果等于从头练）
    optimizer.load_state_dict(ckpt["optimizer"])

    # 4) 把上次练到的 epoch 交回给调用方
    return ckpt["epoch"]


# ============================================================================
# 5. 主程序：把上面所有零件串起来
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 minimal training loop")
    parser.add_argument("--epochs", type=int, default=100, help="一共训练多少个 epoch")
    parser.add_argument("--batch-size", type=int, default=64, help="每批多少条数据")
    parser.add_argument("--lr", type=float, default=3e-3, help="学习率")
    parser.add_argument("--hidden", type=int, default=32, help="隐藏层宽度")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（可复现）")
    parser.add_argument("--noise", type=float, default=0.05, help="数据噪声大小")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/minimal.pt")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 继续训练")
    parser.add_argument("--device", type=str, default="", help="留空则自动选 cuda/cpu")
    args = parser.parse_args()

    # 固定随机种子：同样的命令能复现同样的结果
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 自动选设备：有 GPU 用 GPU
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 训练集和验证集用不同 seed，保证验证数据是"没见过的"
    train_ds = SyntheticDataset(n=2000, seed=args.seed, noise=args.noise)
    val_ds = SyntheticDataset(n=400, seed=args.seed + 1, noise=args.noise)
    # shuffle=True 表示每轮打乱训练数据；num_workers=0 在 Windows 上最稳妥
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TinyMLP(in_dim=8, hidden=args.hidden, out_dim=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- 第一步：resume 逻辑，决定从第几个 epoch 开始 ----
    if args.resume and Path(args.checkpoint).exists():
        # 上次存的是"练到的 epoch"，这次要从它的下一个继续，所以要 +1
        start_epoch = load_checkpoint(args.checkpoint, model, optimizer) + 1
        print(f"已从 checkpoint 恢复，继续从 epoch {start_epoch} 开始")
    else:
        start_epoch = 0

    # ---- 第二步：主循环，一个 epoch = 训一遍 + 验一遍 + 存一次盘 ----
    history = []  # 收集每个 epoch 的 loss，最后一起写进 CSV
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}")

        # 每个 epoch 都存盘：随时 Ctrl+C 中断，--resume 都能接着练
        save_checkpoint(args.checkpoint, model, optimizer, epoch)
        history.append((epoch, train_loss, val_loss))

    # ---- 第三步：把 loss 曲线写成 CSV，方便以后画图 ----
    csv_path = Path(args.checkpoint).with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(history)
    print(f"训练完成，loss 曲线已保存到 {csv_path}")


if __name__ == "__main__":
    main()
