# Mini-LLM → Post-Training → Agent Harness 全链路学习项目方案

> 目标设备：RTX 4060 Laptop 8GB  
> 可选扩展资源：Google Colab / Kaggle 约 16GB GPU  
> 核心目标：**不是训练一个“很强”的模型，而是亲手跑通现代 LLM 的完整生命周期，并在最后构建一个可运行、可评测的 Agent Harness。**

---

# 0. 项目最终目标

这个项目最终要得到四类成果：

1. **一个你从零实现并预训练的小型 Decoder-only Transformer**
   - 自己训练 tokenizer
   - 自己处理预训练数据
   - 自己实现 Transformer
   - 自己实现训练循环
   - 自己保存、加载、采样、评估

2. **一个经过现代后训练的小模型**
   - SFT
   - Preference Optimization / DPO
   - RLVR / GRPO
   - 对每一个阶段的效果进行定量和定性比较

3. **一个具备 Tool Use 能力的小模型或开源基座模型**
   - 能输出结构化 tool call
   - 能调用文件、shell、Python 等工具
   - 能基于工具返回结果继续推理

4. **一个你自己写的 Mini-Codex / Mini-Claude-Code 风格 Agent Harness**
   - Agent loop
   - tool registry
   - context management
   - retries / error handling
   - filesystem sandbox
   - trace
   - eval
   - 后续扩展 MCP / skill / memory / subagent

最终这个项目应该让你真正回答这些问题：

- 一个大模型从文本数据到 checkpoint 的全过程是什么？
- tokenizer 究竟在做什么？
- Transformer 每一层张量形状怎么变化？
- 预训练到底训练了什么？
- SFT 改变了什么？
- DPO / preference learning 改变了什么？
- RLVR / GRPO 为什么能提升可验证任务？
- Tool Use 是模型能力、数据能力还是 Harness 能力？
- 一个 Agent 和一个普通 Chat 模型之间到底差了什么？
- 成熟 Harness 为什么需要 context management、sandbox、trace、eval？
- 模型本身能力和 Harness scaffolding 的边界在哪里？

---

# 1. 项目原则

## 1.1 尽量“先手写，再调用框架”

推荐统一遵循：

> 手写最小版本 → 跑通 → 做实验 → 再阅读成熟实现 → 再用框架复现 → 对比差异

例如：

- 先自己写 causal attention
- 再看 `build-nanogpt`
- 再看 `nanochat`
- 最后再看 Hugging Face Transformers

而不是一上来：

```python
Trainer(...)
```

然后只知道“loss 在下降”。

---

## 1.2 两条模型路线并行

### 路线 A：从零预训练模型

建议规模：

- Debug：5M–10M
- 正式第一版：20M–50M
- 进阶：80M–150M

目的：

- 学懂 LLM 本体
- 学懂 pretraining
- 学懂数据 / tokenizer / optimization

---

### 路线 B：小型开源模型做后训练与 Agent

建议：

- SmolLM2-135M / 360M
- Qwen2.5-0.5B
- 16GB GPU 时可尝试 1.5B 左右 QLoRA

目的：

- SFT
- DPO
- GRPO
- tool use
- agent

原因：

你自己从零训练的 30M–100M 模型很适合研究训练过程，但通常不足以稳定承担复杂 Agent 任务。

因此：

> **从零模型负责“理解训练”，较强开源小模型负责“理解后训练与 Agent”。**

最后用同一套 Harness 比较不同模型。

---

# 2. 项目总流程

```text
环境配置
   ↓
最小 PyTorch 训练骨架
   ↓
Tokenizer
   ↓
预训练数据处理
   ↓
手写 Decoder-only Transformer
   ↓
5M Debug Model
   ↓
20M–50M Scratch Pretraining
   ↓
Inference / Sampling / KV Cache
   ↓
SFT
   ↓
DPO / Preference Optimization
   ↓
RLVR / GRPO
   ↓
Tool Calling Dataset
   ↓
Tool-use SFT
   ↓
Mini Agent Loop
   ↓
Filesystem / Shell / Python Tools
   ↓
Context Management / Retry / Trace
   ↓
MCP
   ↓
Agent Benchmark
   ↓
Ablation / Research-style Analysis
```

---

# 3. 推荐项目目录

```text
mini-llm-lab/

├── README.md
├── pyproject.toml
├── configs/
│   ├── debug_5m.yaml
│   ├── tiny_20m.yaml
│   ├── base_50m.yaml
│   └── sft_qwen05b.yaml
│
├── tokenizer/
│   ├── train_bpe.py
│   ├── tokenizer.py
│   ├── inspect_tokenizer.py
│   └── tests/
│
├── data/
│   ├── tinystories.py
│   ├── fineweb.py
│   ├── smoltalk.py
│   ├── preference.py
│   ├── gsm8k.py
│   ├── tool_data.py
│   ├── packing.py
│   └── dataloader.py
│
├── model/
│   ├── config.py
│   ├── embeddings.py
│   ├── rope.py
│   ├── rmsnorm.py
│   ├── attention.py
│   ├── mlp.py
│   ├── block.py
│   ├── gpt.py
│   └── generation.py
│
├── train/
│   ├── pretrain.py
│   ├── sft.py
│   ├── dpo.py
│   ├── grpo.py
│   ├── optimizer.py
│   └── checkpoint.py
│
├── inference/
│   ├── sample.py
│   ├── kv_cache.py
│   ├── chat_template.py
│   └── server.py
│
├── eval/
│   ├── lm_loss.py
│   ├── generation_eval.py
│   ├── gsm8k_eval.py
│   ├── tool_eval.py
│   └── agent_eval.py
│
├── agent/
│   ├── message.py
│   ├── tool.py
│   ├── registry.py
│   ├── parser.py
│   ├── loop.py
│   ├── context.py
│   ├── sandbox.py
│   ├── trace.py
│   ├── memory.py
│   ├── mcp_client.py
│   └── cli.py
│
├── tools/
│   ├── filesystem.py
│   ├── shell.py
│   ├── python_tool.py
│   └── grep.py
│
├── agent_tasks/
│   ├── task001/
│   ├── task002/
│   └── ...
│
├── notebooks/
│   ├── inspect_loss.ipynb
│   ├── tokenizer_analysis.ipynb
│   └── agent_trace_analysis.ipynb
│
└── tests/
```

---

# 4. Phase 0：环境与最小训练骨架

## 输入

你现在只有：

- RTX 4060 Laptop 8GB
- Python 基础
- PyTorch 可能有些生疏
- 一个空 Git repo

---

## 你要自己做什么

### 1. 配置开发环境

推荐：

```text
Windows
↓
WSL2
↓
Ubuntu
↓
NVIDIA Driver
↓
Python 3.11
↓
uv
↓
PyTorch CUDA
```

熟悉：

```bash
nvidia-smi
git
uv
python
pytest
```

---

### 2. 写一个最小训练脚本

先不要做 Transformer。

自己写：

```python
x -> Linear -> ReLU -> Linear -> loss
```

要求自己实现：

- Dataset
- DataLoader
- forward
- loss
- backward
- optimizer.step()
- zero_grad
- train loop
- validation loop
- checkpoint
- resume
- logging

---

## 输出

得到：

```text
train.py
checkpoint.pt
loss curve
```

---

## 你学到什么

- PyTorch training loop
- GPU tensor
- autograd
- optimizer
- checkpoint
- train / eval mode
- reproducibility

这一步非常重要，因为以后所有复杂训练，本质上都只是这个循环的扩展。

---

# 5. Phase 1：Tokenizer

## 起点

原始文本：

```text
Once upon a time...
```

---

## 最终结果

得到：

```text
tokenizer.model
tokenizer.json
vocab.json
```

以及：

```python
encode(text) -> token_ids
decode(token_ids) -> text
```

---

## 推荐数据

第一版：

- TinyStories 部分文本

---

## 你自己实现的内容

至少亲手做：

### 1. 字符 / byte tokenizer

先实现一个最简单版本：

```text
character -> id
```

理解 tokenizer 的基本概念。

---

### 2. 一个简化 BPE

自己写最小版：

```text
统计 pair
↓
找最高频 pair
↓
merge
↓
更新词表
↓
重复
```

不需要一开始追求速度。

---

### 3. 再用成熟 tokenizer 库重写

用：

```text
Hugging Face tokenizers
```

实现：

- vocab size 8K / 16K
- BOS
- EOS
- PAD
- special tokens

---

## 建议实验

比较：

```text
char tokenizer
BPE-2K
BPE-8K
BPE-16K
```

指标：

- 平均 token / sentence
- compression ratio
- vocab coverage
- 中文 / 英文差异
- sequence length

---

## 参考代码

重点阅读：

- nanochat tokenizer
- LLMs-from-scratch tokenizer 章节
- Hugging Face tokenizers

---

## 你学到什么

理解：

```text
text != model input
```

而是：

```text
raw text
↓
normalization
↓
tokenization
↓
token id
↓
embedding
```

以后看到 vocab size、embedding matrix、sequence length 才会真正有感觉。

---

# 6. Phase 2：预训练数据 Pipeline

## 起点

原始数据：

### 第一版

TinyStories

### 第二版

FineWeb-Edu streaming subset

---

## 你自己编写

```text
download / streaming
↓
clean
↓
filter
↓
tokenize
↓
concatenate
↓
chunk
↓
pack
↓
train/val split
↓
batch
```

---

## 要实现的代码

### 数据下载

```python
load_raw_data()
```

### 清洗

```python
normalize_text()
filter_document()
```

### tokenize

```python
tokenize_document()
```

### packing

把：

```text
doc1
doc2
doc3
```

拼成：

```text
[token token token ...]
```

再切：

```text
x = tokens[i:i+T]
y = tokens[i+1:i+T+1]
```

---

## 特别值得自己实现

### document packing

### streaming dataloader

### shuffle buffer

### train / validation split

### token count

---

## 输出

例如：

```text
data/tinystories/train.bin
data/tinystories/val.bin
```

或者 streaming iterator。

---

## 建议数据规模

### Debug

1M tokens

### 第一版

10M–30M tokens

### 正式 20M–50M 模型

50M–200M tokens

### 扩展实验

300M+ tokens

---

## 你学到什么

你会意识到：

> LLM training 很大一部分工作根本不是 Transformer，而是 Data Engineering。

---

# 7. Phase 3：手写 Decoder-only Transformer

这是整个项目最重要的一阶段。

---

# 7.1 起点

你只有：

```text
token ids
```

例如：

```python
[123, 832, 19, 42, ...]
```

---

# 7.2 最终结果

你要实现：

```python
logits = model(tokens)
```

输出：

```text
[B, T, vocab_size]
```

并能够：

```python
loss = cross_entropy(logits, target)
loss.backward()
```

---

# 7.3 第一版结构

建议先写 GPT-2 风格：

```text
Token Embedding
+
Position Embedding

↓

Transformer Block × N

↓

LayerNorm

↓

LM Head
```

每个 block：

```text
LayerNorm
↓
Causal Self Attention
↓
Residual

LayerNorm
↓
MLP
↓
Residual
```

---

# 7.4 你必须自己写的内容

### Embedding

### LayerNorm

可以先自己实现公式，再换 PyTorch LayerNorm。

### Causal Mask

自己构造：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

### QKV

理解：

```text
XWq
XWk
XWv
```

### Attention Score

```text
QK^T / sqrt(d)
```

### softmax

### multi-head reshape

### MLP

### residual

### LM head

### tied embedding（可选）

---

# 7.5 第二版升级现代结构

第一版跑通后再加入：

- RoPE
- RMSNorm
- SwiGLU
- GQA
- Flash Attention / SDPA

这时可以参考：

- nanochat
- SmolLM
- Qwen-style architecture

---

# 7.6 推荐模型规模

## Debug 版本

```text
layers = 4
hidden = 256
heads = 4
context = 256
vocab = 8K
```

约几百万参数。

---

## 第一正式版本

```text
layers = 6–8
hidden = 384–512
heads = 6–8
context = 512
vocab = 8K–16K
```

目标：

```text
20M–50M
```

---

# 7.7 你需要写的 Debug 工具

非常重要：

```python
print_shape()
count_parameters()
check_nan()
check_gradient_norm()
```

以及 unit tests：

```text
attention shape
causal mask
generation
checkpoint reload
```

---

# 7.8 推荐对照仓库

按照这个顺序看：

1. 自己写
2. build-nanogpt
3. LLMs-from-scratch
4. nanochat
5. SmolLM
6. LitGPT

---

# 7.9 你学到什么

这是你真正建立 Transformer 心智模型的阶段。

最后应该能够不查资料解释：

```text
[B,T]
↓
[B,T,C]
↓
Q/K/V
↓
[B,H,T,D]
↓
attention
↓
[B,T,C]
↓
MLP
↓
logits
```

---

# 8. Phase 4：从零预训练

## 输入

你已经有：

```text
Tokenizer
+
Tokenized dataset
+
GPT model
```

---

## 工作

实现完整：

```text
forward
↓
cross entropy
↓
backward
↓
AdamW
↓
gradient clipping
↓
learning rate schedule
↓
checkpoint
```

---

# 8.1 你必须手写

### training loop

### validation loop

### gradient accumulation

### checkpoint

### resume

### learning rate scheduler

推荐：

```text
warmup
+
cosine decay
```

### gradient clipping

### mixed precision

### logging

---

# 8.2 第二阶段再加入

- gradient checkpointing
- torch.compile
- fused AdamW
- Flash Attention
- sequence packing

---

# 8.3 第一轮训练

数据：

TinyStories

模型：

20M 左右

结果：

你应该能明显看到：

```text
random gibberish
↓
sentence-like text
↓
small coherent stories
```

---

# 8.4 第二轮训练

数据：

FineWeb-Edu subset

模型：

50M 左右

结果：

得到：

```text
base_model.pt
```

这个模型是：

> **预训练模型**

不是 Chat 模型。

你输入：

```text
The capital of France
```

它只是：

```text
continue text
```

而不是：

```text
assistant answering a user
```

这是理解 Base Model 与 Instruct Model 差异的关键。

---

# 8.5 实验

保存多个 checkpoint：

```text
step_1k
step_5k
step_10k
step_20k
```

比较：

- validation loss
- perplexity
- sample quality
- memorization
- repetition

---

# 9. Phase 5：Inference 与 Generation

不要立刻用：

```python
model.generate()
```

自己写。

---

# 9.1 输入

```text
base_model.pt
prompt
```

---

# 9.2 自己实现

### greedy decoding

### temperature

### top-k

### top-p

### repetition penalty（可选）

---

# 9.3 KV Cache

这是非常值得自己实现的一部分。

先写：

```text
每次重新计算整个 prefix
```

记录速度。

再写：

```text
KV Cache
```

比较：

```text
tokens / second
GPU memory
```

---

# 9.4 输出

得到：

```text
inference/sample.py
```

能够：

```bash
python sample.py \
  --checkpoint checkpoints/model.pt \
  --prompt "Once upon a time"
```

---

# 9.5 学到什么

理解：

```text
Training ≠ Inference
```

以及为什么现代 inference system 如 vLLM 会如此重要。

---

# 10. Phase 6：SFT

这是模型从：

```text
Text Completion Model
```

变成：

```text
Instruction-following Model
```

的关键步骤。

---

# 10.1 推荐两种实验

## 实验 A

你自己的 20M–50M scratch model

目的：

观察 SFT 的行为变化。

---

## 实验 B

SmolLM2-135M / 360M 或 Qwen2.5-0.5B

目的：

得到真正可用的小 Chat 模型。

---

# 10.2 数据

推荐：

- SmolTalk subset

第一版：

```text
5K
```

第二版：

```text
20K
```

第三版：

```text
50K+
```

---

# 10.3 你自己实现

### Chat Template

例如：

```text
<|system|>
...

<|user|>
...

<|assistant|>
...
```

---

### Loss Mask

只对 assistant token 计算 loss：

```text
system      mask=0
user        mask=0
assistant   mask=1
```

这一步必须亲手实现。

---

### SFT dataloader

### sequence packing

### train loop

---

# 10.4 第一版不要用 SFTTrainer

先自己写：

```python
loss = masked_cross_entropy(...)
```

之后再用：

```text
TRL SFTTrainer
```

重跑一次。

比较：

- 代码复杂度
- batch handling
- masking
- speed
- logging

---

# 10.5 输出

得到：

```text
base_model.pt
↓
sft_model.pt
```

或者：

```text
Qwen-0.5B
↓
LoRA
↓
tool_sft_adapter
```

---

# 10.6 学到什么

理解：

> 预训练决定“会不会语言”，SFT 很大程度上决定“怎么回答”。

---

# 11. Phase 7：DPO / Preference Learning

## 输入

一个 SFT 模型。

数据格式：

```text
prompt

chosen response

rejected response
```

---

## 数据

推荐：

UltraFeedback Binarized subset

第一版：

```text
5K–10K preference pairs
```

---

## 你自己实现

### logprob 计算

```python
logp_chosen
logp_rejected
```

### reference model

```text
π_ref
```

### DPO loss

手写公式。

---

## 第一版工作流

```text
SFT model
↓
copy frozen reference model
↓
chosen / rejected
↓
calculate token logprob
↓
sequence logprob
↓
DPO objective
↓
backprop
```

---

## 第二版

使用：

```text
TRL DPOTrainer
```

对照你的实现。

---

## 输出

```text
sft_model
↓
dpo_model
```

---

## 评估

准备约 100 个 prompt。

比较：

```text
Base
SFT
DPO
```

重点观察：

- response style
- instruction following
- verbosity
- refusal
- formatting
- preference win rate

---

# 12. Phase 8：RLVR / GRPO

不要第一步就研究大型 PPO stack。

先做一个：

> **有明确 correctness reward 的小型 RL 实验。**

---

# 12.1 数据

GSM8K

---

# 12.2 模型

建议：

```text
Qwen2.5-0.5B-Instruct
```

或者更小模型。

4060：

做小规模。

16GB GPU：

做更完整实验。

---

# 12.3 工作流

```text
question
↓
model generates K responses
↓
extract final answer
↓
compare ground truth
↓
reward = 0 / 1
↓
normalize reward within group
↓
policy update
```

---

# 12.4 你先自己写

不要完整复刻 TRL。

只做一个“教育版 GRPO”。

实现：

```text
group sampling
reward
advantage
logprob
policy loss
```

---

# 12.5 再看

- nanochat RL
- TRL GRPOTrainer

---

# 12.6 输出

```text
sft_model
↓
grpo_model
```

---

# 12.7 对比

看：

```text
GSM8K exact match
```

以及：

```text
response length
reasoning structure
reward hacking
format errors
```

---

# 12.8 学到什么

真正理解：

```text
SFT
vs
Preference Optimization
vs
RL
```

不是三种“Trainer API”，而是三种不同优化目标。

---

# 13. Phase 9：Tool Calling 数据

这一步正式进入 Agent。

---

# 13.1 首先定义 Tool Schema

例如：

```json
{
  "name": "read_file",
  "description": "Read a text file",
  "arguments": {
    "path": "string"
  }
}
```

---

# 13.2 第一批工具

只做：

```text
read_file
write_file
list_dir
grep
python
shell
```

---

# 13.3 数据格式

例如：

```text
User:
Find the bug in app.py and fix it.

Assistant:
<tool_call>
{"name":"read_file","arguments":{"path":"app.py"}}
</tool_call>

Tool:
...

Assistant:
...
```

---

# 13.4 数据来源

推荐混合：

### 手工构造

自己写 100–500 条。

这是最有价值的。

### 模板生成

自动生成：

```text
read
search
edit
run tests
```

任务。

### 大模型合成

让一个强模型生成 tool-use trajectory。

但你必须：

```text
验证 trajectory
运行工具
检查结果
```

不能直接相信。

---

# 13.5 输出

```text
tool_sft.jsonl
```

---

# 13.6 学到什么

你会真正理解：

> Tool use 往往不是“模型天然会调用工具”，而是通过 schema、prompt、training data、parser 和 Harness 一起构建出来的。

---

# 14. Phase 10：Tool-use SFT

## 推荐模型

优先：

```text
Qwen2.5-0.5B
```

或者：

```text
SmolLM2-360M
```

---

## 训练方式

4060 8GB：

```text
LoRA / QLoRA
```

---

## 你自己实现

### Tool schema serialization

### Tool call mask

### Assistant / tool role

### JSON parsing

### Tool call accuracy eval

---

## 输出

```text
tool_model_adapter/
```

---

## Eval

测试：

### Tool selection

应该用：

```text
read_file
```

还是：

```text
shell
```

### Argument accuracy

路径对不对？

### JSON validity

### multi-step tool use

---

# 15. Phase 11：手写 Mini Agent Harness

这是整个项目第二个最重要部分。

---

# 15.1 第一版 Agent 不要用 LangChain

自己写。

核心：

```python
while step < max_steps:

    prompt = build_context(messages, tools)

    output = model.generate(prompt)

    action = parse(output)

    if action.type == "tool":
        result = execute(action)
        messages.append(result)

    else:
        return output
```

---

# 15.2 自己设计核心类

```text
Message
Tool
ToolRegistry
Agent
ModelBackend
ContextManager
ToolExecutor
Trace
```

---

# 15.3 Tool Registry

例如：

```python
registry = {
    "read_file": read_file,
    "write_file": write_file,
}
```

然后逐渐变成类。

---

# 15.4 Parser

需要处理：

```text
valid JSON
invalid JSON
missing field
unknown tool
tool exception
```

---

# 15.5 第一版输出

你应该得到一个 CLI：

```bash
python -m agent.cli
```

可以输入：

```text
打开 test.py，找到错误，修复，并运行 pytest。
```

Agent 能：

```text
list_dir
↓
read_file
↓
edit
↓
pytest
↓
read error
↓
fix
↓
pytest
↓
answer
```

---

# 16. Phase 12：Sandbox

Agent 能执行 shell 后必须做 sandbox。

---

## 自己实现

限制：

```text
working directory
command timeout
max output
blocked paths
blocked commands
```

---

## 可以先做简单版

每个 task：

```text
/tmp/agent_task_xxx/
```

Agent 只能访问这个目录。

---

## 学到什么

理解为什么：

> Agent Harness 不只是 Prompt + Tool。

它同时也是：

> execution environment。

---

# 17. Phase 13：Context Management

长 Agent loop 很快会遇到：

```text
context overflow
```

---

## 第一版

最简单：

```text
truncate oldest messages
```

---

## 第二版

实现：

```text
keep system
keep recent N messages
summarize old trajectory
```

---

## 第三版

区分：

```text
conversation context
working memory
artifact state
tool result
```

---

## 自己做实验

固定任务：

比较：

```text
full context
truncate
summary
structured state
```

成功率变化。

---

# 18. Phase 14：Trace 与 Debugging

每一次 Agent run 保存：

```json
{
  "task": "...",
  "steps": [
    {
      "prompt": "...",
      "output": "...",
      "tool": "...",
      "result": "..."
    }
  ]
}
```

---

## 你自己写 Trace Viewer

最简单可以生成：

```text
HTML
```

或者 Markdown。

---

## 要统计

```text
steps
tokens
tool calls
invalid calls
errors
latency
success
```

---

# 19. Phase 15：Agent Benchmark

不要只凭感觉评价。

自己构造 benchmark。

---

## Task 例子

### Level 1

```text
读取文件并回答问题
```

### Level 2

```text
找到某个函数
```

### Level 3

```text
修改单文件 bug
```

### Level 4

```text
修改多个文件
```

### Level 5

```text
运行测试并迭代修复
```

---

## 每个任务目录

```text
task001/
├── repo/
├── instruction.md
├── hidden_tests/
└── metadata.json
```

---

## Judge

优先使用：

```text
deterministic test
```

例如：

```bash
pytest
```

而不是一开始就用 LLM Judge。

---

## 输出指标

```text
Success Rate
Average Steps
Average Tool Calls
Invalid Tool Call Rate
Token Usage
Wall Time
```

---

# 20. Phase 16：MCP

等你自己的 Tool Registry 完全理解后，再接 MCP。

---

## 第一版

写：

```text
MCP client
```

连接一个：

```text
filesystem MCP server
```

---

## 对比

原本：

```python
registry["read_file"]
```

现在：

```text
discover tools
↓
schema
↓
call MCP tool
```

---

## 学到什么

理解：

```text
tool implementation
vs
tool protocol
```

以及 MCP 在 Agent 生态里的位置。

---

# 21. Phase 17：阅读成熟 Agent Harness

这时再看：

## smolagents

重点看：

```text
Agent loop
Tool abstraction
CodeAgent
Model wrapper
```

---

## DeepAgents

重点看：

```text
filesystem
subagents
context
skills
memory
```

---

## 你的任务

不是照抄。

而是做：

```text
我的实现
vs
smolagents
vs
DeepAgents
```

对照表。

---

# 22. Phase 18：Ablation / 研究式实验

到这里项目就可以从学习工程升级成研究式项目。

---

# 22.1 模型规模

比较：

```text
20M
50M
135M
360M
0.5B
```

Agent 成功率。

---

# 22.2 SFT

比较：

```text
base
instruction SFT
tool SFT
```

---

# 22.3 Harness

同一个模型：

```text
plain chat
+ tools
+ retry
+ context management
+ memory
```

---

# 22.4 Tool Schema

比较：

```text
JSON
XML
special token
natural language
```

---

# 22.5 Error Recovery

故意让工具失败。

观察：

```text
模型是否重试
是否换工具
是否陷入循环
```

---

# 22.6 Context Compaction

比较不同策略。

---

# 22.7 这个阶段最值得研究的问题

最终可以形成一个很有意思的问题：

> **Small Model Capability vs Harness Capability**

即：

```text
模型本身不变
```

只改变：

```text
tools
memory
retrieval
context
planning
retry
```

Agent 能力能提升多少？

---

# 23. 建议模型路线

---

# Route A：Scratch Model

```text
TinyStories
↓
8K BPE
↓
20M GPT
↓
Pretrain
↓
Generation
```

再升级：

```text
FineWeb-Edu
↓
16K BPE
↓
50M Modern Transformer
↓
Pretrain
```

产物：

```text
scratch-base-20m
scratch-base-50m
```

---

# Route B：Scratch Post-training

```text
scratch-base-50m
↓
SmolTalk subset
↓
SFT
↓
UltraFeedback subset
↓
DPO
```

产物：

```text
scratch-sft-50m
scratch-dpo-50m
```

目的：

不是追求强能力。

而是观察：

```text
pretrain → instruction → preference
```

行为变化。

---

# Route C：Practical Small LLM

```text
Qwen2.5-0.5B / SmolLM2-360M
↓
LoRA SFT
↓
Tool SFT
↓
GRPO
↓
Agent Harness
```

产物：

```text
真正能做 tool use 的小模型
```

---

# 24. 显存与算力策略

## RTX 4060 8GB

最适合：

```text
开发
debug
5M–50M scratch
100M 级短实验
135M full SFT
360M LoRA
0.5B LoRA / QLoRA
Agent inference
```

---

## 16GB GPU

用来：

```text
更长 pretraining
100M–300M scratch
0.5B GRPO
1.5B QLoRA
更长 context
```

---

# 25. 训练时应该学会记录

每次训练保存：

```text
config
git commit
seed
dataset
token count
parameter count
batch size
gradient accumulation
learning rate
training steps
validation loss
GPU memory
tokens/sec
wall time
```

---

# 26. 每阶段推荐参考仓库

| 阶段 | 参考 |
|---|---|
| Transformer 基础 | build-nanogpt |
| 教材式实现 | LLMs-from-scratch |
| 完整现代最小 LLM | nanochat |
| 小模型训练 recipe | SmolLM |
| 成熟训练工程 | LitGPT |
| SFT / DPO / GRPO | TRL |
| PEFT | PEFT |
| Agent 最小实现 | smolagents |
| Agent Harness | DeepAgents |
| Eval | lm-evaluation-harness |
| CUDA 底层 | llm.c |

---

# 27. 如何让你的 Agent 带你读代码

不要让 Agent：

> “总结整个仓库。”

这样效果很差。

应该要求它按层次带你读。

---

## Step 1：入口

问：

```text
这个仓库从哪个文件启动？
执行 python xxx.py 之后调用链是什么？
```

---

## Step 2：核心数据结构

问：

```text
列出最重要的 5 个 class / function。
分别负责什么？
```

---

## Step 3：Tensor Shape

对于模型代码：

```text
请逐行标注 tensor shape。
```

---

## Step 4：训练路径

```text
一个 batch 从 dataloader 到 loss.backward 经过哪些函数？
```

---

## Step 5：参数更新

```text
optimizer 在哪里创建？
learning rate 在哪里变化？
gradient accumulation 在哪里实现？
```

---

## Step 6：自己重写

让 Agent：

```text
不要直接复制代码。
告诉我实现步骤，我自己写。
```

然后：

```text
我写完后帮我 review。
```

这是最推荐的学习方式。

---

# 28. 每一个阶段的“毕业标准”

## Tokenizer

你能解释：

```text
为什么 vocab 越大 sequence 越短？
```

---

## Data

你能自己写：

```text
streaming token packing
```

---

## Transformer

你能不看代码画出：

```text
QKV attention
```

---

## Pretraining

你能解释：

```text
一个 token 的 loss 从哪里来？
```

---

## SFT

你能解释：

```text
为什么 user token 通常不计算 loss？
```

---

## DPO

你能解释：

```text
reference model 为什么存在？
```

---

## RL

你能解释：

```text
reward 和 supervised target 的区别。
```

---

## Tool Use

你能解释：

```text
tool schema 如何进入模型上下文？
```

---

## Agent

你能自己写：

```text
while loop agent
```

---

## Harness

你能解释：

```text
为什么 context management 能提升 agent performance。
```

---

# 29. 最终毕业成果

项目最终至少保留：

## 模型

```text
scratch-base-20m
scratch-base-50m
scratch-sft
scratch-dpo
qwen-tool-lora
qwen-grpo
```

---

## 数据

```text
TinyStories processed
FineWeb subset
SFT subset
Preference subset
Tool-use dataset
Agent benchmark
```

---

## 代码

```text
Tokenizer
Transformer
Pretraining
SFT
DPO
GRPO
Inference
KV Cache
Agent Harness
Tools
MCP
Eval
```

---

## 实验报告

至少写一个：

```text
report.md
```

包括：

```text
模型规模
训练 loss
生成案例
SFT before/after
DPO before/after
GRPO before/after
Agent benchmark
Ablations
Failure cases
```

---

# 30. 最终项目应该形成的能力图

```text
                    ┌───────────────┐
                    │   Raw Text    │
                    └───────┬───────┘
                            ↓
                     Tokenizer
                            ↓
                      Data Pipeline
                            ↓
                Decoder-only Transformer
                            ↓
                       Pretraining
                            ↓
                        Base Model
                            ↓
                  ┌─────────┴──────────┐
                  ↓                    ↓
                 SFT             Inference Engine
                  ↓                    ↓
            Preference / DPO        KV Cache
                  ↓
              RLVR / GRPO
                  ↓
            Tool-use Training
                  ↓
             Tool-capable LLM
                  ↓
               Agent Loop
                  ↓
        ┌─────────┼──────────┐
        ↓         ↓          ↓
      Tools     Context     Trace
        ↓         ↓          ↓
      MCP       Memory      Eval
        └─────────┬──────────┘
                  ↓
             Agent Harness
                  ↓
             Benchmark
                  ↓
               Ablation
```

---

# 31. 最重要的学习原则

这个项目中最需要避免的是：

```text
Clone repo
↓
pip install
↓
python train.py
↓
成功
```

这只能算：

> “运行过一个 LLM 项目”。

你的目标应该是：

```text
先自己实现
↓
失败
↓
debug
↓
跑通
↓
阅读成熟实现
↓
理解差异
↓
重构
```

尤其是下面这些部分，建议尽量亲手写：

```text
Tokenizer
Data packing
Attention
Transformer block
Training loop
Generation
KV Cache
SFT mask
DPO loss
简化 GRPO
Tool parser
Agent loop
Tool registry
Context management
Sandbox
Trace
Eval
```

如果这些你都真正写过一次，那么以后无论读：

```text
Qwen
Llama
nanochat
TRL
vLLM
Claude Code
Codex
DeepAgents
```

你的理解速度都会完全不同。

---

# 32. 推荐最终命名

可以把整个项目命名为：

```text
MiniLLM-Lab
```

或者：

```text
MiniLLM-to-Agent
```

或者：

```text
FromScratchLLM
```

如果后续希望做得更研究化：

```text
BoundaryLab
```

也可以把：

```text
Small Model × Harness
```

作为主线实验主题。

最终目标不是“完成一个教程”。

而是形成一套以后可以不断加入：

```text
新模型
新数据
新 RL 方法
新 tool
新 memory
新 skill
新 benchmark
```

的个人 LLM / Agent 实验平台。
