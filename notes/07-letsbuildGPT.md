# Lecture 7 — Let's build GPT

## 课程仓库：[nanoGPT](https://github.com/karpathy/nanoGPT)

> 日期：2026-07-02

## 一、GPT 与 Transformer

这节课正式进入 GPT 的学习。ChatGPT 的核心原理是：根据已有上文，逐字预测下一个字符。底层架构来自 2017 年的划时代论文 *Attention Is All You Need* 提出的 Transformer。GPT 是 Generative Pre-trained Transformer 的缩写。

这节课不训练完整 ChatGPT——那工程量过大。我们训练的是一个基于 Transformer 架构的**字符级语言模型**，在 Karpathy 钟爱的 **Tiny Shakespeare** 数据集（莎士比亚作品，约 1MB、1115394 个字符）上训练。Karpathy 已将全部代码开源在 [nanoGPT](https://github.com/karpathy/nanoGPT) 仓库，核心仅两个约 300 行的文件。

数据集包含 65 个不同字符（大小写字母、标点、空格换行等）：

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)   # → 65
```

## 二、Tokenization 与数据准备

tokenization 是将原始文本按词汇表转换为数字序列。由于是字符级模型，最简单的做法是把每个独立字符映射为一个数字：

```python
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
# "hii there" → [46, 47, 47, 1, 58, 46, 43, 56, 43]
```

实际工业应用中，tokenizer 更复杂——Google 的 SentencePiece（sub-word）、OpenAI 的 tiktoken（GPT 使用，词汇表 50257 个）。Sub-word tokenizer 是主流，但这里用字符级，简单直观。

整篇文章转为 tensor 后 shape 为 `(1115394,)`，按 90/10 划分训练/验证集。关键设定：不会把全部文本一次性输入 Transformer——那样计算成本过高。实际在 chunks 上训练，通过 `block_size` 控制 chunk 最大长度，随机采样。

训练样本的构造方式：对于每个位置的输入 `x[:i+1]`，目标是 `y[i]`——即给定前面的字符，预测下一个。这种"滑动窗口"模式和之前 makemore 系列的构造逻辑完全一致：

```python
block_size = 8
x = train_data[:block_size]       # [18, 47, 56, 57, 58, 1, 15, 47]
y = train_data[1:block_size+1]    # [47, 56, 57, 58, 1, 15, 47, 58]
# when input is [18] the target: 47
# when input is [18, 47] the target: 56
# ...每个位置都在预测下一个字符
```

`get_batch(split)` 函数随机采样 `batch_size` 个连续的 chunk，x 和 y 的 shape 都是 `(B, T)`——输入是 `data[i:i+block_size]`，目标是 `data[i+1:i+block_size+1]`。

## 三、Bigram 基线：最简单的起点

在动手构建 Transformer 之前，先用一个最简单的 bigram 模型跑一遍，建立 baseline。这个模型只有一个 `nn.Embedding(vocab_size, vocab_size)` 的查找表——每个 token 直接从表中读取下一个 token 的 logits，完全不考虑历史上下文：

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T) → (B,T,C)
        # cross_entropy 要求 (B*T, C) 和 (B*T,) 的形式
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                  # (B,T,C)
            logits = logits[:, -1, :]              # 只取最后一个位置 (B,C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)
        return idx
```

这里有两个值得注意的细节。**cross_entropy 的形状转换**——logits 从 `(B, T, C)` 变为 `(B*T, C)`，targets 从 `(B, T)` 变为 `(B*T,)`——PyTorch 的交叉熵要求 `(N, C)` 对 `(N,)` 的形式。**生成时的迭代采样**——每次只取最后一个位置的 logits，softmax 后 multinomial 采样一个 token，拼接到序列末尾，循环 `max_new_tokens` 次。

用 AdamW（lr=1e-3）训练 10000 步后，loss 从 4.88 降到 2.38。生成结果勉强有点像英文文本了，但显然和莎士比亚的水平差距很大——因为它本质上还是 bigram，只依赖前一个字符预测下一个，token 之间没有任何"交流"。

## 四、自注意力的数学技巧

在进入真正的 self-attention 之前，Karpathy 先用一个例子引入核心的数学技巧。

给定 `x` 的 shape 为 `(B, T, C) = (4, 8, 2)`，想让每个 token 只能和它前面的 token 交流。最简单的方法是**对前面的 token 向量取平均**——信息只能从先前的文本流到当前位置，不能获得未来的任何信息：

```python
# 朴素版：对每个 token 逐位置计算前缀平均
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]          # (t, C)
        xbow[b, t] = torch.mean(xprev, 0)
```

这种双重循环极其低效。技巧是使用**下三角矩阵乘法**一次性完成：

```python
wei = torch.tril(torch.ones(T, T))         # (T,T) 下三角矩阵，因果掩码
wei = wei / wei.sum(1, keepdim=True)        # 归一化得到平均值
xbow2 = wei @ x                             # (T,T) @ (B,T,C)→(B,T,C)
```

`tril` 生成下三角全 1 矩阵，除以每行和之后，每一行就是一个平均权重的模板。`wei @ x` 利用 PyTorch 广播机制，自动把 `(T,T)` 扩展到 `(B,T,T)`，每个 batch 内独立进行 `(T,T) @ (T,C) = (T,C)`。

这里有一个浮点精度的细节：本机验证 `torch.allclose(xbow, xbow2)` 返回 **False**。通过 DW（Data Wrangler）审阅数据发现，两个 tensor 在一些小数位后面几位出现了不同。原因是计算路径不同导致的累积误差——虽然数学上 `(a+b+c)/3` 等于 `a*(1/3)+b*(1/3)+c*(1/3)`，但在二进制浮点数里，加法顺序和乘法顺序的改变会导致微小的舍入误差（1/3 在二进制里是无限循环小数）。不过 `xbow2` 和 `xbow3`（softmax 版本）之间 `allclose` 倒是 True。

Karpathy 还给出了第三个版本——用 **softmax** 实现：

```python
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))  # 上三角置 -inf
wei = F.softmax(wei, dim=-1)                       # softmax 归一化
xbow3 = wei @ x
```

`masked_fill` 将上三角位置设为 -inf，softmax 后这些位置的权重为 0。这个版本的意义在于：后续真正的 self-attention 使用的正是这种 "masked softmax" 模式，只不过权重不再是均匀的，而是由 Query 和 Key 动态计算出来的。

## 五、模型升级：Embedding + 位置编码

从 bigram 到真正的 Transformer，第一步改进引入了**词嵌入**和**位置编码**。具体改动（对比原始 bigram.py 与新版本 v2.py）：

- **Token Embedding**：原始用一个 `(vocab_size, vocab_size)` 的查找表直接出 logits。改进后用 `nn.Embedding(vocab_size, n_embd)` 将 token 映射为稠密向量（n_embd=32），模型能学习 token 间的语义关系。
- **Position Embedding**：原始完全不考虑 token 位置。改进后新增 `nn.Embedding(block_size, n_embd)`，为 0 到 block_size-1 的每个位置学习独立嵌入。将 token emb 和 pos emb 相加，使输入同时包含"是什么词"和"在什么位置"的信息。
- **嵌入与输出分离**：先得到 `x = tok_emb + pos_emb`（shape `(B, T, n_embd)`），再通过 `lm_head` 线性层映射到 vocab_size。这种"特征提取 + 分类"的结构为后续堆叠 Transformer 块留出了接口。

## 六、自注意力机制

### 核心思想

自注意力的核心：序列中每个 token 以**数据依赖**的方式获取其之前的信息。每个 token 会发射两个向量——**Query（Q，查询）**和 **Key（K，键）**。Query 表达"我在寻找什么"，Key 表达"我拥有什么"。通过当前 token 的 Q 去点积其他 token 的 K，得到关联度权重矩阵（affinities）。Key 与 Query 语义对齐时点积值高，权重更大，模型从该位置获取更多信息。

此外还有 **Value（V，值）**——通过 value 层把每个 token 的向量映射为实际要传递的信息内容。最终用权重矩阵对 V 做加权求和，得到每个 token 的上下文表示。

x 可以理解为每个 token 的 **private information**——信息存储在 x 里。通过 key/query 计算注意力权重，通过 value 提取信息内容，最后 `wei @ v` 把信息加权聚合。

### 单头实现

```python
head_size = 16
key   = nn.Linear(C, head_size, bias=False)    # (32→16)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)            # (B, T, 16) — "我有什么信息"
q = query(x)          # (B, T, 16) — "我想找什么信息"
v = value(x)          # (B, T, 16) — "实际信息内容"

wei = q @ k.transpose(-2, -1)                   # (B,T,16)@(B,16,T)→(B,T,T)
# 除以 sqrt(head_size) 防止点积过大 → softmax 饱和
wei = wei * head_size**-0.5
# 因果掩码：上三角置 -inf
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)                     # (B,T,T) 归一化权重

out = wei @ v     # (B, T, T) @ (B, T, 16) → (B, T, 16)
```

head_size=16 的设计原因：每个 token 的 32 维特征被压缩到 16 维子空间内计算注意力。压缩维度是因为后续引入多头注意力时，多个 head 拼接起来维度才不会爆炸，且每个 head 独立学习自己的投影矩阵，可捕捉不同特征模式。

观察 `wei[0]`——一个 8×8 的下三角矩阵，每行权重不再均匀，而是由 Q·K 动态决定。例如第 8 个 token 的注意力权重 `[0.021, 0.084, 0.056, 0.230, 0.057, 0.071, 0.242, 0.239]`——对不同位置的关注程度各不相同。

### 缩放（Scaled Attention）

一个重要的细节：如果不除以 `sqrt(head_size)`，wei 的方差会爆炸。验证：

```python
k = torch.randn(B, T, 16); q = torch.randn(B, T, 16)
wei = q @ k.transpose(-2, -1)         # wei.var() ≈ 17.5
wei = q @ k.transpose(-2, -1) * 16**-0.5  # wei.var() ≈ 1.09 ← 回到 1 附近
```

k 和 q 的方差都在 1 附近，但点积后 wei 方差膨胀到 ~17。softmax 在输入方差过大时输出会饱和（接近 0 或 1），梯度消失。除以 `sqrt(head_size)` 使 wei 方差保持 ~1，softmax 输出保持分散（diffuse），梯度传播稳定。

### Karpathy 的几点重要强调

- **注意力是一种通信机制**。可视作有向图，每个 token 是一个节点，通过数据依赖的权重对指向它的节点的信息加权求和。
- **注意力本身不含空间/位置信息**。它只是对一组向量操作。模型感知顺序必须通过位置编码显式注入。
- **不同 batch 的样本之间完全独立**，没有任何注意力交互。
- **Encoder vs Decoder**：删除 tril 掩码 → 所有 token 双向通信 = **Encoder**。保留掩码 → 只关注前面 = **Decoder**（自回归场景使用）。
- **Self-Attention vs Cross-Attention**：Self-Attention 中 K、V 与 Q 来自同一来源 x。Cross-Attention 中 Q 仍来自 x，但 K、V 来自外部来源（如 encoder 输出）。
- **Scaled Attention**：除以 `sqrt(head_size)` 防止 softmax 饱和。

## 七、多头注意力

从单头升级为多头（Multi-Head Attention），对应原论文的核心组件。让模型在多个不同的投影子空间中并行计算注意力：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)  # 输出投影

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # 拼接多头
        out = self.proj(out)                                    # 线性整合
        return out
```

每个 head 的 `head_size = n_embd // num_heads`（如 32/4=8），4 个头各自输出 `(B, T, 8)`，拼接后得到 `(B, T, 32)`。总参数量 `num_heads * 3 * (n_embd * head_size)` 等于 `3 * n_embd * n_embd`，与单头 head_size=n_embd 时完全相同——**参数量不变，表达能力增强**。不同头可以捕捉不同类型的依赖：一个关注邻近词、一个关注远距离语法结构等。

## 八、前馈网络

原论文中每个 Transformer 层还包含一个前馈网络（Feed-Forward Network, FFN）。它的作用：自注意力完成 token 间的"信息交流"后，FFN 对**每个位置的表示独立进行非线性变换**——相当于给 token 在"交流"之后一个"思考"和"消化"的过程：

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # 扩展 4 倍
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),   # 收缩回原始维度
        )
```

中间维度扩增 4 倍（Transformer 论文的经典设计），大幅增加非线性容量，然后投影回原始维度。注意 FFN 是**逐位置**的——每个 token 独立通过同一组权重，不像注意力那样涉及 token 间交互。

## 九、Block 封装与多层堆叠

把多头注意力和 FFN 打包成一个 Block：

```python
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        self.sa = MultiHeadAttention(n_head, n_embd//n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # 注意力 + 残差
        x = x + self.ffwd(self.ln2(x))   # FFN + 残差
        return x
```

这是 Transformer 中"通信（注意力）→ 计算（前馈）"的标准层结构。通过 `nn.Sequential`（在 nanoGPT 中用 `nn.ModuleList`）堆叠多个 Block，模型从"浅层单层"升级为"深层多层"。最终在模型输出前加一个额外的 `nn.LayerNorm` 做最终归一化，再送入 `lm_head`。

## 十、残差连接

深层网络直接训练面临梯度消失/爆炸的挑战。残差连接（Residual Connection）源自何凯明 2015 年的 ResNet 论文，是训练深层网络的关键技术。

实现非常简单：**每个子模块的输出与输入相加**——`x = x + self.sa(x)` 和 `x = x + self.ffwd(x)`。

数学直觉来自 micrograd（第一课）：**加法节点的梯度分配是等价的**——上游梯度会复制给每一个输入分支。残差主路始终获得完整梯度信号，不会因网络加深而衰减；各子层也能独立获得等量梯度，保证参数更新有效。从数据流角度看：数据沿主路径自上而下流动，每层从主路径上"分叉"出一路做变换，然后加回去——整个过程就是"一步一加，层层累加"。

## 十一、层归一化

Layer Normalization 与 Batch Normalization 思路高度相似，差别在于**归一化的维度**：

```python
class LayerNorm1d:
    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)    # 对特征维度求均值（BN 是对 batch 维度）
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out
```

BN 在 batch 维度归一化——每个神经元在 batch 内的所有样本上归一化。LN 在**特征维度**归一化——每个样本内部对所有特征归一化，样本之间完全独立。因此 LN 不需要 running buffer（动量更新），batch_size=1 时也能正常工作，特别适合 NLP 中变长序列和自回归任务。

采用 **Pre-LN** 结构（归一化放在子模块**之前**而非之后）——这已被多数现代 Transformer（包括 GPT）采用：

```python
x = x + self.sa(self.ln1(x))    # Pre-LN: 先归一化再注意力，再加残差
x = x + self.ffwd(self.ln2(x))
```

原始论文是 Post-LN（子模块之后归一化），但实践证明 Pre-LN 训练更稳定，尤其对深层网络。

## 十二、扩展规模与 Dropout

为测试完整模型能力，大幅提升超参数：

| 参数 | 旧值 | 新值 |
|------|------|------|
| block_size | 8 | 256 |
| n_embd | 32 | 384 |
| n_head | 4 (隐式) | 6 |
| n_layer | 3 | 6 |
| batch_size | 32 | 64 |
| learning_rate | 1e-3 | 3e-4 |

同时引入 **Dropout** 正则化（Srivastava et al. 2014）：每次前向/反向传播中随机丢弃部分神经元，相当于训练一堆不同的子网络。测试时所有神经元都参与，这些子网络融合成一个更好的模型。在注意力权重后、FFN 投影后添加 `nn.Dropout(dropout)`。

由于参数量过大（需要 4060 笔记本跑 300 步就花 5 分钟），本地没有完整训练。Karpathy 用 A100 跑了 15 分钟，最终 train loss 1.0763, val loss 1.4873，生成结果质量可观。

## 十三、Decoder-Only vs Encoder-Decoder

我们实现的 Transformer 与原论文存在显著差异：

- **我们的模型**：**decoder-only**，无 encoder、无 cross-attention。每个 Block 只有 self-attention + FFN。使用下三角掩码保证自回归性质（只看到前面的 token）。适用于**无条件文本生成**。
- **原论文模型**：完整的 **encoder-decoder** 架构。Encoder 对输入做完整双向理解（无掩码），Decoder 的 Q 来自自身输入、K 和 V 来自 encoder 输出（cross-attention）。面向**机器翻译**等有条件序列转换任务。

简言之，decoder-only 是原论文 decoder 部分的独立子集。这一认识为后续可能的扩展（如引入条件控制）奠定了基础。

## 十四、ChatGPT 的训练流程

Karpathy 介绍了训练 ChatGPT 的两阶段过程：

**第一阶段：Pre-training**。用海量互联网数据训练 decoder-only transformer。和这节课做的事情非常类似，只是规模巨大——GPT-3 有 175B 参数、96 层、12288 embedding、96 head、batch size 3.2M、在 300B tokens 上训练。OpenAI 用 subword tokenizer（非字符级），词汇表 ~50000。这一阶段得到的本质是一个 **document completer**，只能续写文本。

**第二阶段：Fine-tuning**。把 document completer 变成 assistant。分三步：
1. **收集演示数据并训练有监督策略**：收集大量 Q&A 格式数据集进行 fine-tuning，让模型学习"输入问题 → 给出答案"的对话格式。
2. **收集对比数据并训练奖励模型**：模型生成多个回答，人工打分，用分差训练一个奖励模型来预测"哪个回答更好"。
3. **PPO 强化学习**：让模型大量生成句子，奖励模型打分，根据分数反馈反向调整参数，以期生成更高分的回答。

## 十五、阶段对比

| | WaveNet 分层 (L6) | GPT / Transformer (L7) |
|---|---|---|
| 核心架构 | 分层融合（树形收缩） | 自注意力 + 前馈（Block 堆叠） |
| 序列处理 | FlattenConsecutive 逐步 | Q·K 动态权重，任意距离直接交互 |
| 位置信息 | 隐式（分层结构中自然保留） | Position Embedding 显式注入 |
| 归一化 | BatchNorm（沿 batch+seq） | LayerNorm（沿特征维度，Pre-LN） |
| 正则化 | 无 | Dropout |
| 残差连接 | 无 | 每个子模块都有 |
| 输入长度 | block_size=8 | block_size=256 |
| 参数量 | 22k~76k | 384 embd × 6 层（数量级远超） |
| 训练数据 | 32033 个人名 | Tiny Shakespeare（1MB，111 万字符） |
| val loss | 1.993（最佳） | 1.487（A100 完整训练） |
