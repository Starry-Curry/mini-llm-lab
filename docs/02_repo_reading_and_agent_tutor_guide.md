# 开源仓库阅读与 Agent 辅助学习指南

> 目标：不要让 Agent 直接替你“看完一个仓库”，而是让它作为你的源码助教。

---

# 1. 推荐阅读顺序

## 第一层：Transformer 与 Pretraining

### build-nanogpt

适合：
- GPT 基础
- training loop
- GPT-2 reconstruction

你应该重点理解：
```text
model definition
forward
loss
optimizer
training loop
```

---

### LLMs-from-scratch

适合：
- 教材式学习
- Attention
- GPT
- Pretraining
- Finetuning

用途：
当你某个数学 / 代码细节不理解时查。

---

## 第二层：现代小 LLM

### nanochat

重点：
```text
tokenizer
data loader
modern GPT
base training
SFT
RL
inference
```

这是最适合做“完整 pipeline 对照答案”的仓库。

---

### SmolLM

重点：
```text
小模型 architecture
data recipe
training recipe
post-training
```

用途：
把你的教学模型升级成更现代的小模型。

---

### LitGPT

重点：
```text
成熟训练工程
config
checkpoint
LoRA
distributed
```

用途：
理解“教学实现”和“可维护训练工程”的差别。

---

# 2. Post-training 阅读顺序

## 先手写

### SFT
自己写：
```text
chat template
loss mask
```

### DPO
自己写：
```text
chosen/rejected logprob
reference model
loss
```

### GRPO
自己写简化版本：
```text
sampling
reward
advantage
policy loss
```

---

## 再看 TRL

分别定位：

```text
SFTTrainer
DPOTrainer
GRPOTrainer
```

不要一次读整个 TRL。

---

# 3. Agent 阅读顺序

## 第一版
完全自己写：

```text
while loop
tool registry
parser
executor
```

---

## 第二版
看 smolagents。

问题：

```text
它如何表示 Tool？
Agent loop 在哪里？
Tool result 如何加入 context？
什么时候停止？
错误如何处理？
```

---

## 第三版
看 DeepAgents。

问题：

```text
为什么需要 filesystem abstraction？
subagent 如何组织？
skills 如何组织？
context 如何管理？
persistent state 在哪里？
```

---

# 4. 最推荐的 Agent 提问模板

---

## 模板 A：寻找入口

```text
请不要总结整个仓库。

先帮我定位这个仓库的执行入口。

假设我运行：
python xxx.py

请告诉我：
1. 入口文件
2. 第一个被调用的核心函数
3. 接下来的调用链
4. 每一步的职责

暂时不要解释实现细节。
```

---

## 模板 B：追踪一个 batch

```text
请追踪一个训练 batch。

从 DataLoader 返回 batch 开始，
一直到 loss.backward()。

按实际调用顺序列出：
- 文件
- 函数
- 输入
- 输出
- Tensor shape

不要跳步骤。
```

---

## 模板 C：读 Attention

```text
请带我逐行阅读这个 attention forward。

要求：
1. 每一行说明输入 shape
2. 每一行说明输出 shape
3. reshape / transpose 为什么这样做
4. Q/K/V 的物理意义
5. causal mask 在哪里
6. 最后如何恢复到 [B,T,C]

不要直接给总结。
```

---

## 模板 D：训练参数

```text
请帮我定位这个仓库：

1. optimizer 在哪里初始化
2. 哪些参数 weight decay
3. learning rate 如何变化
4. gradient accumulation 在哪里
5. mixed precision 在哪里
6. gradient clipping 在哪里
7. checkpoint 在哪里保存
```

---

## 模板 E：让我自己写

```text
我正在自己实现这个模块。

不要给我完整代码。

请只告诉我：
1. 接口设计
2. 输入输出
3. 实现步骤
4. 最容易写错的地方
5. 3 个 unit test

我写完后再给你 review。
```

这是最推荐的模式。

---

## 模板 F：代码 Review

```text
这是我自己写的实现。

请不要直接重写。

先判断：
1. shape 是否正确
2. 数学逻辑是否正确
3. autograd 是否有问题
4. 是否可能产生 NaN
5. 是否存在性能问题
6. 与参考实现的差异

最后只给最小修改建议。
```

---

# 5. 推荐“自己写 / 看代码 / 调框架”比例

## Tokenizer
```text
自己写 70%
参考 30%
```

## Transformer
```text
自己写 80%
参考 20%
```

## Pretraining
```text
自己写 70%
参考 30%
```

## SFT
```text
自己写 60%
TRL 40%
```

## DPO
```text
自己写 50%
TRL 50%
```

## GRPO
```text
自己写教学版 40%
框架 60%
```

## Agent Loop
```text
自己写 80%
参考 20%
```

## Context / Memory / Skills
```text
自己写 50%
参考成熟 harness 50%
```

---

# 6. 每个仓库最适合学什么

| Repo | 最适合学 |
|---|---|
| build-nanogpt | GPT 与训练循环 |
| LLMs-from-scratch | 系统基础 |
| nanochat | 完整现代 LLM pipeline |
| SmolLM | 小模型 recipe |
| LitGPT | 训练工程 |
| TRL | 后训练算法 |
| PEFT | LoRA / QLoRA |
| smolagents | 极简 Agent |
| DeepAgents | 完整 Harness |
| lm-eval-harness | Benchmark |
| llm.c | C/CUDA 底层 |

---

# 7. 不建议的学习方式

避免：

```text
Agent 总结整个 repo
```

避免：

```text
直接把完整模块让 Agent 写完
```

避免：

```text
只看 README
```

避免：

```text
只会调 Trainer
```

避免：

```text
跑通 notebook 就认为理解了
```

---

# 8. 推荐源码学习循环

每个模块：

```text
1. 明确接口
2. 自己设计
3. 自己写
4. unit test
5. 跑通
6. 让 Agent review
7. 阅读参考实现
8. 对比差异
9. 重构
10. 写学习笔记
```

---

# 9. 每读完一个模块写一个 note

例如：

```text
notes/
├── tokenizer.md
├── attention.md
├── rope.md
├── pretraining.md
├── sft.md
├── dpo.md
├── grpo.md
├── kv_cache.md
├── tool_calling.md
└── agent_context.md
```

每一篇只回答：

```text
它解决什么问题？
输入是什么？
输出是什么？
核心公式是什么？
代码在哪里？
我自己是怎么实现的？
成熟实现做了哪些优化？
我踩了什么坑？
```

---

# 10. 最终目标

你不应该只是能够：

```text
运行 nanochat
```

而应该达到：

```text
看到 nanochat 的一段代码
↓
知道它为什么存在
↓
知道自己简化版怎么写
↓
知道成熟实现比你的版本多了什么
```

对于 Agent 也是一样：

```text
看到成熟 Harness
↓
知道最小 while-loop 在哪里
↓
知道额外 scaffolding 为什么存在
↓
知道哪些能力来自模型
↓
哪些能力来自 Harness
```

这才是整个项目最核心的学习价值。
