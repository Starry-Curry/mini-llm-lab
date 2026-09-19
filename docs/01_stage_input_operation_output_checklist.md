# Mini-LLM → Agent 阶段输入 / 操作 / 输出 Checklist

这份文档用于真正开工时快速查看每个阶段“从什么开始、自己要写什么、最后得到什么”。

---

# Stage 0 — 环境

## 输入
- Windows / Linux
- RTX 4060 8GB

## 自己做
- WSL2 / Ubuntu
- uv
- PyTorch CUDA
- Git
- pytest

## 输出
- 可运行 CUDA PyTorch 环境
- 空 Git repo

## 通过标准
```python
torch.cuda.is_available() == True
```

---

# Stage 1 — 最小训练循环

## 输入
- 随机数据

## 自己写
- Dataset
- DataLoader
- model
- loss
- backward
- optimizer
- checkpoint
- validation

## 输出
```text
train.py
checkpoint.pt
```

## 学习
PyTorch 完整训练过程。

---

# Stage 2 — Tokenizer

## 输入
```text
raw TinyStories text
```

## 自己写
- char tokenizer
- 简化 BPE
- encode
- decode
- special token

## 再对照
- Hugging Face tokenizers
- nanochat tokenizer

## 输出
```text
tokenizer.json
```

---

# Stage 3 — Data Pipeline

## 输入
```text
TinyStories / FineWeb-Edu
```

## 自己写
- cleaning
- tokenization
- packing
- chunking
- streaming
- train/val
- dataloader

## 输出
```text
train token stream
val token stream
```

---

# Stage 4 — Transformer

## 输入
```text
[B,T] token ids
```

## 自己写
- embedding
- causal mask
- QKV
- multi-head attention
- MLP
- residual
- norm
- LM head

## 输出
```text
[B,T,V] logits
```

## 升级
- RoPE
- RMSNorm
- SwiGLU
- GQA

---

# Stage 5 — Scratch Pretraining

## 输入
```text
model + token stream
```

## 自己写
- AdamW
- LR schedule
- warmup
- gradient accumulation
- mixed precision
- checkpoint
- resume
- logging

## 输出
```text
scratch-base-20m.pt
scratch-base-50m.pt
```

## 类型
**Base Model**

---

# Stage 6 — Generation

## 输入
```text
base model
```

## 自己写
- greedy
- temperature
- top-k
- top-p
- KV cache

## 输出
```text
sample.py
```

---

# Stage 7 — SFT

## 输入
```text
base model
+
SmolTalk subset
```

## 自己写
- chat template
- role serialization
- assistant loss mask
- packing
- SFT loop

## 输出
```text
sft-model
```

## 类型
**Instruction Model**

---

# Stage 8 — DPO

## 输入
```text
SFT model
+
prompt/chosen/rejected
```

## 自己写
- sequence logprob
- frozen reference model
- DPO loss

## 输出
```text
dpo-model
```

## 类型
**Preference-aligned Model**

---

# Stage 9 — RLVR / GRPO

## 输入
```text
SFT model
+
GSM8K
```

## 自己写
- sample K responses
- answer extraction
- correctness reward
- group reward normalization
- simplified policy update

## 输出
```text
grpo-model
```

---

# Stage 10 — Tool Data

## 输入
```text
filesystem/code tasks
```

## 自己写
- tool schemas
- synthetic task generator
- trajectory validator

## 输出
```text
tool_sft.jsonl
```

---

# Stage 11 — Tool-use SFT

## 输入
```text
Qwen 0.5B / SmolLM 360M
+
tool_sft.jsonl
```

## 自己写
- tool serialization
- tool-call masking
- JSON validity eval
- tool selection eval

## 输出
```text
tool-lora-adapter
```

---

# Stage 12 — Mini Agent

## 输入
```text
tool-capable model
```

## 自己写
```text
Message
Tool
ToolRegistry
Parser
AgentLoop
ModelBackend
```

## 输出
可以完成：

```text
read → edit → run tests → fix
```

---

# Stage 13 — Sandbox

## 自己写
- task working dir
- timeout
- output cap
- path restrictions

## 输出
安全的本地代码执行环境。

---

# Stage 14 — Context Management

## 自己写
- truncate
- sliding window
- summary
- structured state

## 输出
长任务不会立即爆 context。

---

# Stage 15 — Trace

## 自己写
- trajectory JSON
- token usage
- tool history
- error history

## 输出
```text
trace.json
```

---

# Stage 16 — Benchmark

## 自己写
- coding tasks
- hidden tests
- deterministic judge
- metric aggregator

## 输出
```text
success rate
avg steps
tool-call errors
token usage
```

---

# Stage 17 — MCP

## 输入
你自己的 Tool Registry。

## 操作
接入 MCP client/server。

## 输出
动态发现和调用外部工具。

---

# Stage 18 — 对照成熟实现

## LLM
- build-nanogpt
- LLMs-from-scratch
- nanochat
- SmolLM
- LitGPT

## Post-training
- TRL
- PEFT

## Agent
- smolagents
- DeepAgents

## Eval
- lm-evaluation-harness

---

# Stage 19 — Ablation

固定模型，改变：

```text
tools
retry
context management
memory
planning
```

固定 harness，改变：

```text
20M
50M
135M
360M
0.5B
```

最终研究问题：

> Small Model Capability vs Harness Capability
