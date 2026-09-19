# Mini-LLM → Post-Training → Agent Harness

从零手搓一个 Mini-LLM，一路做到 Agent Harness 的全链路学习项目。

核心目标不是训出一个"很强"的模型，而是亲手跑通现代 LLM 的完整生命周期：
从文本数据到 checkpoint、从 Base Model 到 Instruct Model，再到一个可运行、可评测的
Mini-Codex 风格 Agent Harness。

## 最终成果

- 一个从零实现并预训练的 Decoder-only Transformer（自写 tokenizer / 数据管线 / 模型 / 训练循环 / 采样 / 评测）
- 一个走过完整后训练的小模型（SFT → DPO → RLVR/GRPO，逐阶段对比）
- 一个具备工具调用能力的小模型（结构化 tool call，文件 / shell / python 工具）
- 一个自研 Agent Harness（agent loop、tool registry、context 管理、沙箱、trace、eval，后续扩展 MCP / skill / memory）

## 整条链路

```text
raw text → tokenizer → 数据管线 → 手写 Transformer → 预训练(base)
→ 推理(采样/KV cache) → SFT(instruct) → DPO(preference) → GRPO(RL)
→ 工具数据 + 工具 SFT → Agent loop + 工具 + 沙箱 → context/trace
→ benchmark → MCP → 消融实验
```

## 两条模型路线

- 路线 A（理解训练）：从零预训练 5M–50M 小模型（TinyStories → FineWeb-Edu 子集）
- 路线 B（理解后训练与 Agent）：SmolLM2-135M/360M 或 Qwen2.5-0.5B 做 SFT / DPO / GRPO / Tool Use

## 硬件与运行环境

- GPU：RTX 4060 Laptop 8GB
- Python 3.10 + PyTorch 2.11 (cu128)
- Conda 环境：`LLMlearning_cuda`，激活方式 `conda activate LLMlearning_cuda`

## 目录结构

```text
configs/        训练配置
tokenizer/      BPE tokenizer 训练与编解码
data/           数据集处理（不提交大文件）
model/          Transformer 各组件
train/          预训练 / SFT / DPO / GRPO 循环
inference/      采样生成 / KV cache / server
eval/           loss / generation / gsm8k / agent 评测
agent/          Message / Tool / Registry / Parser / Loop / Context / Trace
tools/          filesystem / shell / python 工具
agent_tasks/    Agent benchmark 任务
notebooks/      分析用 notebook
tests/          单元测试
notes/          每个模块的学习笔记
```

## 工作方式

每个模块固定节奏：

1. 先讲清原理、输入输出与 tensor shape 流转，给出接口设计和单元测试，代码自己动手写
2. 跑通单元测试后 review，再对照成熟实现（build-nanogpt / nanochat / TRL / smolagents / DeepAgents）找差异
3. 每个模块完成一份 `notes/*.md` 学习笔记

先手写最小版本，跑通后再读参考仓库，而不是一上来就调用框架 API。

## 数据与产物的存放约定

- 数据集放 `data/`，模型权重放 `checkpoints/`，外部库缓存放 `.cache/`
- 三者都保留在本机 D 盘，`.gitignore` 已排除，不上传 GitHub
