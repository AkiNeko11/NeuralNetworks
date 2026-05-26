# Lecture 3 — makemore Part 2

## 课程仓库：[makemore](https://github.com/karpathy/makemore)

> 日期：2026-05-25

## 一、为什么 Bigram 不够

02 课我们做了两个版本的 bigram 模型——计数概率表和神经网络 W 矩阵，最终效果一样，W 训练完趋近于计数概率。但 bigram 只看前一个字符，生成的名字质量很差。

要引入更多上下文（前 2 个、3 个字符...），如果用计数表的方式，表格尺寸会呈次方级爆炸——1 个字符是 27²，2 个字符是 (27×27)²，而且大量组合概率为 0，模型越来越稀疏。

所以这一课转向 **MLP（多层感知机）**，通过神经网络来学习字符之间的关系。

## 二、Bengio 2003 论文的核心思想

方法来源于论文 *A Neural Probabilistic Language Model* ([Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf))。原论文是 word-level 的（17000 个词映射到 30 或 60 维特征向量），我们做的是 character-level，但模型方法一致。

核心创新是 **Embedding（嵌入）**：把每个字符映射到一个低维向量空间。一开始随机初始化，训练过程中不断调整——语义相似的字符最终会聚在向量空间的相近位置。

论文里的例子：输入 "a dog was running in a ___"，尽管训练数据里没有一模一样的句子，但模型可以通过 embedding 知道 "a" 和 "the" 类似、"cat" 和 "dog" 都是动物——把学到 "the cat is walking in a room" 的知识迁移过来，从而给出合理的预测。

## 三、构建训练数据

用 block_size=3 的上下文窗口，滑动生成训练样本。以 "emma" 为例：

```
... ---> e
..e ---> m
.em ---> m
emm ---> a
mma ---> .
```

每个位置的输入是前 3 个字符的索引（初始用 `.` 填充），标签是下一个字符的索引。每个名字末尾加 `.` 作为结束符。

```python
context = [0] * block_size          # 初始化为 [...]
for ch in w + '.':
    ix = stoi[ch]
    X.append(context)               # 当前上下文
    Y.append(ix)                    # 要预测的下一个字符
    context = context[1:] + [ix]    # 窗口右移
```

## 四、Embedding 层

```python
C = torch.randn((27, 2))            # 27 个字符，每个映射到 2 维向量
emb = C[X]                          # (32, 3, 2)
```

`C[X]` 是查表操作——对于 X 中每个字符索引，取出 C 的对应行。结果形状 `(batch, block_size, embedding_dim)`。

这里有两种理解方式：
- **查表**：直接把 C 当成查找表，`C[5]` 就是索引 5 对应的嵌入向量，高效
- **线性层**：`F.one_hot(tensor(5), 27).float() @ C` 得到和 `C[5]` 一样的结果。独热编码 × C 本质上就是查表，只是多了一步矩阵乘法，开销大

实践中直接用查表。

## 五、拼接嵌入 → 隐藏层

三个字符的嵌入向量 `(32, 3, 2)` 需要展平成一个 `(32, 6)` 的矩阵才能送入全连接层。三种拼接方式：

- `torch.cat([emb[:,0,:], emb[:,1,:], emb[:,2,:]], 1)` — 手动按列拼接
- `torch.cat(torch.unbind(emb, 1), 1)` — 先拆成 3 个 `(32, 2)` 再拼接
- `emb.view(32, 6)` — **最高效**，不拷贝数据，只改变对底层存储的解释方式

Torch 底层存储是一维连续数组，不同 shape 的 tensor 只是对同一个存储的不同"视角"。`view` 直接重新解释 shape，没有数据拷贝，效率最高。

```python
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)   # (32, 100)
```

`tanh` 把输出限制在 [-1, 1]，引入非线性，之前 bigram 模型没有这一步。

## 六、输出层与交叉熵损失

```python
logits = h @ W2 + b2                            # (32, 27)
loss = F.cross_entropy(logits, Y)
```

`F.cross_entropy` 内部自动完成 softmax + log + nll，等价于之前手动写的：

```python
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
loss = -probs[torch.arange(N), Y].log().mean()
```

但 `F.cross_entropy` 有三个优势：
1. **不创建中间变量**（counts, probs），节省显存
2. **运行在 fused kernel 上**，表达式被化简，速度更快
3. **数值稳定**：内部通过减去最大值来避免 `exp` 溢出（当 logits 中某些值很大时，直接 `exp` 会爆 float）

这是使用 PyTorch 内置函数的重要理由——不仅是方便，更是数值稳定性和性能。

## 七、过拟合的直观感受

先用前 5 个名字（32 条数据）训练 1000 轮，3481 个参数。loss 很快降到 0.25，预测结果几乎完美——因为参数数量远超样本数，模型把训练数据"背"下来了。

但有一个有趣的细节：**loss 永远无法到 0**。因为每个名字的第一条数据都是 `... -> 首字母`，不同名字首字母不同（emma 首字母 e，olivia 首字母 o），模型不可能从 `...` 推断出正确的首字母。这是数据本身的固有不确定性。

## 八、Minibatch 梯度下降

全量数据有 228146 条，每次 forward/backward 都要算全部数据，速度太慢。实际中普遍使用 **minibatch**：

```python
ix = torch.randint(0, X.shape[0], (32,))    # 随机抽 32 条
emb = C[X[ix]]                                # 只对这些样本计算
loss = F.cross_entropy(logits, Y[ix])
```

用 minibatch 估计的梯度质量会下降（只是 32/228146 的数据），方向不精确。但核心洞察是：

**大约的梯度 + 更多次迭代 > 精确的梯度 + 更少的迭代**

随机梯度下降的噪声反而有助于跳出局部最优。

## 九、学习率搜索

如何选合适的学习率？用一个 trick：

```python
lre = torch.linspace(-3, 0, 1000)    # 学习率的指数从 -3 到 0
lrs = 10**lre                         # 实际学习率从 10⁻³ 到 10⁰
```

每步用一个略大的学习率，记录对应的 loss，画图。loss 下降最快、还没爆炸的区域就是好的学习率区间。从图中看到 10⁻¹ 附近比较合适。

实际训练中还会有学习率衰减：先以较大学习率训练，最后阶段逐步减小（比如除以 10），做精细收敛。

## 十、训练集 / 验证集 / 测试集

这课还引入了标准的数据集划分：

- **训练集 (training split)** 80%：训练模型参数
- **验证集 (dev/validation split)** 10%：调整超参数（学习率、网络大小等）
- **测试集 (test split)** 10%：最终评估模型表现，只在最后用一次

用 `random.seed(42)` 打乱数据后再划分，保证分布均匀。

## 十一、扩展模型

从最初的 3481 参数：

```python
C  = torch.randn((27, 2))        # 2 维嵌入
W1 = torch.randn((6, 100))       # 100 隐藏神经元
W2 = torch.randn((100, 27))
```

扩展到 11897 参数：

```python
C  = torch.randn((27, 10))       # 10 维嵌入
W1 = torch.randn((30, 200))      # 200 隐藏神经元
W2 = torch.randn((200, 27))
```

embedding 维度和隐藏层大小都是**超参数**，通过验证集上的表现来调整。

训练 50000 轮后，训练集 loss ≈ 2.17，验证集 loss ≈ 2.19，两者接近说明没有明显过拟合。

## 十二、字符嵌入的可视化

将 `C` 的前两个维度画成散点图，可以看到训练后字母在向量空间中的分布——元音字母（a, e, i, o, u）倾向于聚在一起，模式相似的字符合得近。这验证了 embedding 确实学到了字符之间的关系。

## 十三、从模型采样

```python
context = [0] * block_size               # 从 ... 开始
while True:
    emb = C[torch.tensor([context])]
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1, generator=g).item()
    context = context[1:] + [ix]         # 窗口右移
    if ix == 0: break
```

生成的样本质量明显优于 bigram（`carmah`, `jazonen`, `deliah`, `nellara`...），读起来更像真实名字了。

## 十四、阶段对比

| | Bigram (L2) | MLP (L3) |
|---|---|---|
| 输入 | 前 1 个字符 | 前 3 个字符 (block_size=3) |
| 字符表示 | One-hot 27 维 | Embedding 低维向量 |
| 隐藏层 | 无 | tanh 全连接层 |
| 非线性 | 无 | tanh |
| 训练方式 | 全批量梯度下降 | Minibatch 随机梯度下降 |
| 数据集 | 全部用于训练 | 训练/验证/测试 三分 |
| 学习率 | 固定 50 | 搜索 + 逐步衰减 |
| 损失函数 | 手动 softmax | F.cross_entropy |
| 生成质量 | 很差 | 明显改善 |
