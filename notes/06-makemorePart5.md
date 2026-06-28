# Lecture 6 — makemore Part 5

## 课程仓库：[makemore](https://github.com/karpathy/makemore)

> 日期：2026-06-28

## 一、从"一口吞"到"细嚼慢咽"

前面几节课的做法一直是：取前 3 个字符，embedding 后拼接成一个大向量，一股脑塞进一个单隐藏层 MLP，然后预测下一个字符。模型确实能学到东西，但 Karpathy 指出这种做法有一个结构性问题：**所有输入信息在一开始就被迅速"压扁"了**。

想象一下——8 个字符的上下文，每个 embed 到 10 维，flatten 成 80 维向量，然后一次性扔进一个 Linear 层。这相当于在模型的最早期就把所有信息的交互压缩到一起。后续不管加多少层、扩大多少神经元，本质上还是在处理已经被"暴力混合"过的信息。

这节课引入一种更合理的方式：**分层融合（hierarchical fusion）**。不是一口气把所有字符混合，而是像二叉树一样，先两两融合成 bigram 表示，再把这些 bigram 两两融合成 4-gram，再融合成 8-gram……信息沿着网络深度**逐步、缓慢地**被整合。这个架构来自 DeepMind 2016 年的 WaveNet 论文——原本是用于生成原始音频波形的自回归模型，但结构思想完全适用于字符级语言模型。

## 二、模块化重构

在动手改架构之前，先把之前散落的代码重新组织成类似 PyTorch API 的类。从 L4 的 `Linear`、`BatchNorm1d`、`Tanh` 基础上，新增三个模块：

```python
class Embedding:        # 封装 C[Xb] 查表操作，不再手动 emb = C[Xb]
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))
    def __call__(self, ix):
        self.out = self.weight[ix]
        return self.out

class Flatten:          # 封装 .view 操作，将 (B,T,C) → (B, T*C)
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

class Sequential:       # 串联多个层，类似 nn.Sequential
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
```

所有模块统一：`__call__` 把输出存在 `self.out` 里（方便后续 retain_grad 或查看），`parameters()` 返回可训练参数列表，`training` 属性控制 BN 行为。组装一个模型变得很简洁：

```python
model = Sequential([
    Embedding(vocab_size, n_embd),
    Flatten(),
    Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])
```

注意第一层 Linear 的 `bias=False`——BN 后面跟 Tanh，BN 有自己的 bnbias，前面 Linear 的 bias 是冗余的。

## 三、扩展输入上下文：3 → 8

先把 block_size 从 3 改成 8，看看单纯增加输入长度能带来多少提升。数据从 `(182625, 3)` 变成 `(182441, 8)`——由于名字最长 15 个字符，8 个字符的上下文窗口已经能覆盖大部分名字的全部前缀。

以 "elianys" 为例，训练数据变成：

```
........ --> e
.......e --> l
......el --> i
.....eli --> a
....elia --> n
...elian --> y
..eliany --> s
.elianys --> .
```

用原始扁平架构训练（n_embd=10, n_hidden=200, 22097 个参数），Karpathy 的结果是 train 1.918, val 2.027——相比于 3 个字符的 train 2.058, val 2.105 确实有提升，但这纯粹是因为增加了输入信息量，架构本身没有变化。

## 四、WaveNet 的分层架构

WaveNet 的核心思想是**逐步融合**。以 8 个字符输入为例，embedding 后 shape 为 `(B, 8, 10)`——8 个字符，每个 10 维向量。

传统做法：Flatten → `(B, 80)` → 一个 Linear 一次处理。WaveNet 的做法：把相邻字符两两分组 `(1,2) (3,4) (5,6) (7,8)`，形成 4 个 bigram，每个 bigram 是拼接后的 20 维向量 `(B, 4, 20)`，经过一层 Linear+BN+Tanh 处理。然后再两两分组 `(1+2, 3+4)` 变成 2 个 4-gram `(B, 2, 40)`，再处理。最后 2 组融合成 1 个 `(B, 80)`，送入输出层。整个过程是 8 → 4 → 2 → 1 的树形结构。

要实现这种操作，PyTorch 的 `@` 矩阵乘法有一个关键特性：**它可以对多维度 tensor 的最后一个维度进行乘法**，保留前面的维度结构：

```python
(torch.randn(4, 5, 80) @ torch.randn(80, 200)).shape    # → (4, 5, 200)
# 而不是 (4, 200)
```

这意味着一个 Linear 层可以并行处理多个"组"——对于 `(B, 4, 20)` 的输入，一个 `(20, 200)` 的 Linear 可以同时作用于 4 个 bigram 位置，输出 `(B, 4, 200)`。

## 五、FlattenConsecutive：展平但不完全展平

要实现 `(B, 8, 10) → (B, 4, 20)` 的分组拼接，原来的 `Flatten`（全部展平）不够用了。于是设计一个新层：

```python
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n    # 把每 n 个连续元素拼接在一起

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.n, C*self.n)  # (B, T/n, C*n)
        if x.shape[1] == 1:     # 如果只剩一组，去掉中间维度
            x = x.squeeze(1)    # (B, C*n)
        self.out = x
        return self.out
```

以 n=2 为例，`(B, 8, 10) → (B, 4, 20)`——每两个字符的 embedding 拼接到一起。这个操作和手动取偶数/奇数位再 cat 的效果完全等价：

```python
e[:,::2,:]    # 偶数位 (B, 4, 10)
e[:,1::2,:]   # 奇数位 (B, 4, 10)
explicit = torch.cat([e[:,::2,:], e[:,1::2,:]], dim=2)  # (B, 4, 20)

(e.view(4, 4, 20) == explicit).all()   # → True
```

`view` 在这里利用了 tensor 底层存储的连续性——不需要拷贝数据，只是重新解释存储布局。

## 六、分层网络的前向传播

最终的三层网络结构：

```python
model = Sequential([
    Embedding(vocab_size, n_embd),                              # (B,8) → (B,8,10)
    FlattenConsecutive(2), Linear(20, 68), BN(68), Tanh(),     # (B,8,10) → (B,4,20) → (B,4,68)
    FlattenConsecutive(2), Linear(136, 68), BN(68), Tanh(),    # (B,4,68) → (B,2,136) → (B,2,68)
    FlattenConsecutive(2), Linear(136, 68), BN(68), Tanh(),    # (B,2,68) → (B,136) → (B,68)
    Linear(68, vocab_size),                                     # (B,68) → (B,27)
])
```

以 batch=4 为例追踪各层 shape 变化：

```
Embedding            : (4, 8, 10)
FlattenConsecutive(2): (4, 4, 20)    ← 8个字符 → 4个bigram
Linear(20,68)        : (4, 4, 68)
BN + Tanh            : (4, 4, 68)
FlattenConsecutive(2): (4, 2, 136)   ← 4组 → 2组
Linear(136,68)       : (4, 2, 68)
BN + Tanh            : (4, 2, 68)
FlattenConsecutive(2): (4, 136)      ← 2组 → 1组，squeeze 掉中间维
Linear(136,68)       : (4, 68)
BN + Tanh            : (4, 68)
Linear(68,27)        : (4, 27)       ← 输出
```

每一步 `FlattenConsecutive(2)` 把序列长度减半、特征维度翻倍，然后 Linear 把翻倍的特征再压回 `n_hidden=68`。信息没有在一步内"爆炸式混合"，而是沿着三层逐步融合。n_hidden 从原来的 200 降到了 68——因为现在有三层，每层 68 个神经元，总表达能力不比 200 差。总参数 22397（n_embd=10, n_hidden=68）。

## 七、BatchNorm 的三维 bug

训练完分层网络，结果令人失望——train 1.941, val 2.029，和扁平架构的 train 1.918, val 2.027 基本打成平手。花了不少功夫搭了三层结构，参数也差不多（都 ~22k），效果却没有提升。

Karpathy 这时一脸平静地说："BatchNorm 里其实还有个 bug。"——当时我真没绷住

问题出在 `BatchNorm1d` 的均值计算。原本的实现：

```python
xmean = x.mean(0, keepdim=True)    # 对 batch 维度求均值
```

这在处理 2D tensor `(B, C)` 时没问题——对 B 求均值，得到 `(1, C)`，在 C 个 channel 上各自归一化。

但在分层架构中，BN 接收的是 3D tensor `(B, T, C)`，比如 `(32, 4, 68)`。此时 `mean(0)` 只对第 0 维求均值，得到 `(1, 4, 68)`——保留了 T=4 这个维度。这导致 BN 实际工作在 **4×68=272 个 channel** 上，而不是 68 个。每个 bigram 位置都被独立维护了独立的统计量。

而我们想要的行为是：**沿着 batch 和序列位置两个维度同时求均值**，得到 `(1, 1, 68)`，然后在 68 个 channel 上归一化。这样 4 个 bigram 位置才共享同一组统计量。

验证一下：

```python
e = torch.randn(32, 4, 68)
e.mean(0, keepdim=True).shape       # → (1, 4, 68)    ← bug：保留了 T 维度
e.mean((0,1), keepdim=True).shape   # → (1, 1, 68)    ← 正确：batch 和 seq 一起归一化
```

附带的影响是 running_mean/running_var 的 shape 也从 `(68,)` 被错误地膨胀到了 `(4, 68)`。

## 八、修复与训练

```python
class BatchNorm1d:
    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0              # (B, C) → 对 B 求均值
            elif x.ndim == 3:
                dim = (0, 1)         # (B, T, C) → 对 B 和 T 同时求均值
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        # ... 后续归一化和 running 统计量更新不变
```

这里额外提一句：我们自制的 BatchNorm1d 和 PyTorch 官方的接收维度不同。官方接收 `(N, C)` 或 `(N, C, L)`——第二种情况求和的维度是 0 和 2。我们接收的是 `(N, C)` 或 `(N, L, C)`——第二种情况求和的维度是 0 和 1。只是维度排列习惯不同，逻辑一致。

修复后的完整 benchmark 对比（引用 Karpathy 数据）：

```
original (3 chars + flat, 12k params):                  train 2.058   val 2.105
content 3→8 (flat, 22k params):                         train 1.918   val 2.027
flat → hierarchical (22k params, buggy BN):              train 1.941   val 2.029
fix bug in batchnorm:                                    train 1.910   val 2.022
```

修复 BN bug 后，分层架构终于带来了实质提升——train 从 1.941 降到 1.910，val 从 2.029 降到 2.022。虽然幅度不算巨大，但这是在参数量相同（都是 ~22k）的情况下，**架构本身带来的增益**。

## 九、扩展网络规模

把网络规模进一步扩大：n_embd=24, n_hidden=128，总参数 76k。Karpathy 的结果：

```
scaled up (76k params):              train 1.769   val 1.993
```

从最初的 2.058/2.105 到 1.769/1.993，val loss 下降了约 0.11。Karpathy 特别指出：他展示这些数字不是为了炫耀 loss 的下降——实际情况是我们在调参时"完全处于黑暗之中，没有 experimental harness，全在瞎猜"。而且这个模型也**没有完全使用 WaveNet 论文的架构**——我们没有 gate linear 单元、没有残差连接、没有 skip connection。真正的 WaveNet 还包含这些组件，效果会更好。

这节课的价值不在于最终的 loss 数字，而在于引入了**分层融合**这个架构思想。

## 十、与卷积的联系

Karpathy 在最后点了一下卷积。对于 8 个字符的输入，如果我们把同一组 Linear 权重依次应用到每个位置（滑动窗口），每个位置独立产生一个预测，那本质上就是在做卷积。卷积就是一个"for 循环"——允许我们在空间上高效地前向传播线性层，而不需要显式写出循环：

```python
# 逐个位置 forward（等价于卷积）
logits = torch.zeros(8, 27)
for i in range(8):
    logits[i] = model(Xtr[[7+i]])   # 每个位置独立通过同一组权重
```

卷积纯粹是为了效率——它并没有改变模型的数学本质。在后续课程中，当需要处理更长序列时，这个等价关系会变得很重要。

## 十一、阶段对比

| | MLP 手动反向传播 (L5) | WaveNet 分层架构 (L6) |
|---|---|---|
| 核心关注 | 梯度的底层实现 | 前向传播的架构设计 |
| 网络结构 | 1 层隐藏 (flat MLP) | 3 层隐藏 (hierarchical) |
| 输入长度 | block_size=3 | block_size=8 |
| 信息融合方式 | 一次性 flatten + Linear | FlattenConsecutive(2) 逐步融合 |
| shape 变化 | (B,30)→(B,200)→(B,27) | (B,8,10)→(B,4,68)→(B,2,68)→(B,68)→(B,27) |
| 参数量 | 12297 | 22397 → 76000 |
| 关键坑 | 双路梯度、C 散射累加 | BN 三维归一化 bug |
| 代码组织 | 手动 backward + no_grad | 模块化 Sequential/Embedding/FlattenConsecutive |
| 最终 val loss | 2.109 | 2.022 (基础) → 1.993 (扩展) |
