"""Stage 1：最小训练骨架的单元测试。

实现 train/minimal_loop.py 中的对应接口后，这三个测试应全部通过。
运行方式（仓库根目录、LLMlearning_cuda 环境）：

    python -m pytest tests/test_stage1.py -v
"""

import torch

from train.minimal_loop import (
    SyntheticDataset,
    TinyMLP,
    load_checkpoint,
    save_checkpoint,
)


def test_mlp_forward_shape_and_finite():
    torch.manual_seed(0)
    model = TinyMLP(in_dim=8, hidden=32, out_dim=1)
    x = torch.randn(4, 8)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, 1)
    assert torch.isfinite(y).all()


def test_dataset_deterministic():
    ds = SyntheticDataset(n=16, seed=7)
    assert len(ds) == 16

    x0, y0 = ds[0]
    x1, y1 = ds[1]
    assert x0.shape == (8,)
    assert y0.ndim == 0  # 目标是标量
    assert not torch.equal(x0, x1)

    # 相同 seed 重建数据集，样本必须完全一致（可复现）
    ds2 = SyntheticDataset(n=16, seed=7)
    assert torch.equal(ds[0][0], ds2[0][0])
    assert torch.equal(ds[0][1], ds2[0][1])


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 先跑一步，让 AdamW 产生内部状态（一阶矩 m / 二阶矩 v）
    x = torch.randn(4, 8)
    y = torch.randn(4, 1)
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()

    path = tmp_path / "minimal.pt"
    save_checkpoint(path, model, optimizer, epoch=3)

    fresh = TinyMLP()
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
    assert len(fresh_optimizer.state) == 0  # 新优化器初始无状态

    epoch = load_checkpoint(path, fresh, fresh_optimizer)

    assert epoch == 3
    assert len(fresh_optimizer.state) > 0  # 优化器状态也必须被恢复
    for (k, v), (k2, v2) in zip(
        model.state_dict().items(), fresh.state_dict().items()
    ):
        assert k == k2
        assert torch.equal(v, v2)
