# Lecture 3 — makemore Part 2

## 课程仓库：[makemore](https://github.com/karpathy/makemore)

> 日期：2026-05-25

## 一、为什么 Bigram 不够

02 课我们做了两个版本的 bigram 模型——计数概率表和神经网络 W 矩阵，最终效果一样，W 训练完趋近于计数概率。但 bigram 只看前一个字符，生成的名字质量很差。

要引入更多上下文（前 2 个、3 个字符…），如果用计数表的方式，表格尺寸会呈次方级爆炸——1 个字符是 27²，2 个字符就成了 (27×27)²，而且大量罕见组合概率为 0，表格越来越稀疏，这显然不是好模型。

所以这一课转向 **MLP（多层感知机）**，通过神经网络来学习字符之间的关系。

## 二、Bengio 2003 论文的核心思想

方法来源于论文 *A Neural Probabilistic Language Model* ([Bengio et al. 2003](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf))。先明确一点——我们做的是 **character-level** 的语言模型，在字母层面工作。原论文是 word-level 的：把 17000 个单词映射到 30（或 60）维的特征向量，每个单词都被 **embedded into（嵌入到）** 一个低维向量空间。

一开始这些单词向量是随机初始化的，然后在训练中通过反向传播不断微调，单词向量在空间中不断移动。你可以想象：含义相似的单词最后会聚在向量空间的相近位置，完全不同的则会远离。论文在训练中也最大化 log likelihood（我们是最小化负 log likelihood），两者的方法其他方面基本一致。

论文里一个直观的例子：输入 "a dog was running in a \_\_\_"，即使训练数据里从未出现过一模一样的句子，模型仍然可能给出合理答案。因为它见过 "The cat is walking in the bedroom"——通过 embedding，网络学会了 the 和 a 在很多地方可以互换，于是把它们放在向量空间相近位置；cat 和 dog 都是动物名词，也被放在相近位置。知识就通过 embedding 在相似表达之间传递了。

## 三、构建训练数据

用 block_size=3 的上下文窗口，滑动生成训练样本。取前 5 个名字为例：

```
emma:
  ... ---> e     ..e ---> m     .em ---> m     emm ---> a     mma ---> .
olivia:
  ... ---> o     ..o ---> l     .ol ---> i     oli ---> v     liv ---> i     ivi ---> a     via ---> .
ava:
  ... ---> a     ..a ---> v     .av ---> a     ava ---> .
isabella:
  ... ---> i     ..i ---> s     .is ---> a     isa ---> b     sab ---> e
  abe ---> l     bel ---> l     ell ---> a     lla ---> .
sophia:
  ... ---> s     ..s ---> o     .so ---> p     sop ---> h     oph ---> i     phi ---> a     hia ---> .
```

5 个名字共产生 32 条训练数据 `(32, 3)`。每个位置的输入是前 3 个字符的索引（初始用 `.` 填充），标签是下一个字符的索引：

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

`C[X]` 是查表操作——对于 X 中每个字符索引，取出 C 的对应行。可以验证查表的一致性：

```python
X[13, 2]          # → tensor(1)    ..a→v 中 'a' 的索引是 1
C[X][13, 2]       # → tensor([-0.8622, 1.3225])
C[1]              # → tensor([-0.8622, 1.3225])   一致
```

结果形状 `(batch=32, block_size=3, embedding_dim=2)`——32 行，每行 3 个字符，每个字符用 2 维向量表示。

这里有两种理解方式：
- **查表**：直接 `C[5]` 取第 5 行，高效。实践中用这个。
- **线性层**：`F.one_hot(tensor(5), 27).float() @ C` 得到和 `C[5]` 一样的结果，但多了一步矩阵乘法，开销大。

注意 `F.one_hot` 输出的 dtype 是 `torch.int64`，和 C 相乘前需要 `.float()`。这里 embedding 层没有非线性激活——它本质就是一个权重矩阵是 C 的纯线性层，只是在做查表。

## 五、拼接嵌入 → 隐藏层

三个字符的嵌入 `(32, 3, 2)` 需要展平成 `(32, 6)` 才能送入全连接层 `W1(6, 100)`。如果直接传 3D tensor：

```python
emb @ W1        # (32, 3, 2) @ (6, 100) → RuntimeError!
# mat1 and mat2 shapes cannot be multiplied (96x2 and 6x100)
```

PyTorch 把 `(32, 3, 2)` 当成 `(96, 2)` 和 `(6, 100)` 做乘法，维度不对。

三种拼接方法：

```python
torch.cat([emb[:,0,:], emb[:,1,:], emb[:,2,:]], 1)   # 手动逐列拼接
torch.cat(torch.unbind(emb, 1), 1)                    # 沿第1维拆成3个(32,2)再拼
emb.view(32, 6)                                        # 最高效：不拷贝数据
```

`view` 最高效的原因：Torch 底层存储是一维连续数组，不同 shape 的 tensor 只是对这个数组的不同"视角"——底层 storage 不变，只是 shape 和 stride 改变。所以 `view` 直接重新解释 shape，零拷贝。

```python
h = torch.tanh(emb.view(-1, 6) @ W1 + b1)   # (32, 100)
```

`tanh` 把输出限制在 [-1, 1]，引入非线性——这是 bigram 模型完全没有的。不过注意此时 h 中已经有大量 ±1.0 的极值（这是随机初始化导致 hpreact 分布太广的锅，但这一点要等到 04 课才会深入讨论）。

## 六、输出层与交叉熵损失

```python
logits = h @ W2 + b2                            # (32, 27)
counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)   # 32 行，每行 27 个概率
```

用随机初始化的权重跑一遍 32 条数据，模型给正确标签的概率极低：

```python
prob[torch.arange(32), Y]
# [7.47e-12, 2.15e-2, 3.17e-11, 1.62e-5, 6.83e-8, ...]
# 大量概率在 10^-8 到 10^-21 量级——基本等于 0
loss = -prob[torch.arange(32), Y].log().mean()   # → 20.1033
```

初始 loss 高达 20.1，因为随机 W 给出的概率分布几乎不给正确答案任何概率。

### F.cross_entropy 的三个优势

```python
loss = F.cross_entropy(logits, Y)    # → 17.7697（和手动计算一致）
```

等价于手动 `softmax → log → nll`，但更优（这是用户标注的重点）：

1. **不创建中间变量**：不用存 counts 和 probs 这两个 temp tensor，节省显存
2. **运行在 fused kernel 上**：表达式被化简，单次 kernel 调用完成，比多个独立操作更快
3. **数值稳定**：内部通过减去 logits 最大值来避免 softmax 的 exp 溢出。当 logits 中某些值很大时（比如 +36），直接 `exp(36)` 的结果远超 float 范围，而减去最大值后所有值 ≤ 0，exp 安全。手动计算在大量级数据上可能溢出或下溢

这就是为什么应该用 PyTorch 内置函数——不仅是方便，更是数值稳定性和性能。

## 七、过拟合的直观感受

先用前 5 个名字（32 条数据）训练 1000 轮，3481 个参数（超出样本数两个数量级）：

```python
for _ in range(1000):
    emb = C[X]; h = torch.tanh(emb.view(-1,6) @ W1 + b1)
    logits = h @ W2 + b2; loss = F.cross_entropy(logits, Y)
    for p in parameters: p.grad = None
    loss.backward()
    for p in parameters: p.data += -0.1 * p.grad
print(loss.item())   # → 0.2552
```

loss 从 20 降到 0.25，预测几乎完美：

```python
logits.max(1).indices   # [19, 13, 13, 1, 0, 19, 12, 9, 22, 9, 1, 0, ...]
Y                       # [ 5, 13, 13, 1, 0, 15, 12, 9, 22, 9, 1, 0, ...]
```

大部分预测都对了——3481 个参数"背"32 条数据绰绰有余。

但有一个有趣的细节：**loss 永远无法降到 0**。因为每个名字的第一条数据都是 `... → 首字母`，而不同名字首字母不同（emma→e, olivia→o, jack→j）。从 `...` 推断首字母本身就是伪命题。这是数据本身的固有不确定性。

## 八、全量数据与 Minibatch

扩展到全部 32033 个名字，共 **228146 条训练数据**。但全批量训练变得很慢——每次 forward 和 backward 都要算 228146 条：

```python
for _ in range(10):
    # 全批量 GD
    loss = F.cross_entropy(h @ W2 + b2, Y)
    loss.backward()
    # ↓ loss 从 8.33 → 7.14 → 7.97 → 7.80 → 7.63 → 7.48 → 7.33 → 7.18 → 7.05 → 6.92
```

10 轮全批量才从 8.33 降到 6.92，太慢了。

实际中的标准做法是 **minibatch**：每次只随机抽一小批数据计算，用它来估计梯度：

```python
ix = torch.randint(0, X.shape[0], (32,))    # 随机抽 32 条
emb = C[X[ix]]
loss = F.cross_entropy(logits, Y[ix])
```

minibatch 估计的梯度质量确实不如全批量精确（只是全部数据的 32/228146），方向会有偏差。但核心洞察是：

**大约的梯度 + 更多次迭代 > 精确的梯度 + 更少的迭代**

用廉价的近似梯度换更多的更新次数，总体收敛更快。而且随机梯度下降引入的噪声反而有助于跳出局部最优。所以实践中 minibatch 是标配。

## 九、学习率搜索

如何选合适的学习率？一个 trick：

```python
lre = torch.linspace(-3, 0, 1000)    # 指数从 -3 到 0
lrs = 10**lre                         # 实际 lr 从 10⁻³ 到 10⁰

for i in range(1000):
    # forward/backward with minibatch
    lr = lrs[i]                       # 每步用稍大的 lr
    for p in parameters: p.data += -lr * p.grad
    lossi.append(loss.item())

plt.plot(lri, lossi)                  # 找 loss 下降最快且未爆炸的区间
```

从图上看到 10⁻¹ 附近 loss 下降最快且稳定——所以选 **0.1**。实际训练时还会做**学习率衰减**：先用较大 lr 训练主体阶段，末尾阶段逐步减小（比如除以 10），做最后的精细收敛。

## 十、训练集 / 验证集 / 测试集

现有的模型只有 3481 个参数，自然想到可以通过增加参数来降低 loss。但参数一多，模型可能只是"背"下了训练集里的名字——用没见过的新名字去评估时 loss 会非常高。

所以引入标准的数据集划分：

- **训练集 (training split) 80%**：训练模型参数
- **验证集 (dev/validation split) 10%**：调整超参数（学习率、网络大小等）
- **测试集 (test split) 10%**：最终评估模型表现，只在最后用一次

```python
random.seed(42); random.shuffle(words)
n1 = int(0.8*len(words)); n2 = int(0.9*len(words))
Xtr, Ytr = build_dataset(words[:n1])     # 182625 条
Xdev, Ydev = build_dataset(words[n1:n2])  # 22655 条
Xte, Yte = build_dataset(words[n2:])      # 22866 条
```

`random.seed(42)` 保证打乱可复现，分布均匀。

## 十一、扩展模型

从 3481 参数（2 维 embedding、100 隐藏神经元）：

```python
C  = torch.randn((27, 2))        # 27×2 = 54
W1 = torch.randn((6, 100))       # 6×100 = 600
b1 = torch.randn(100)            # 100
W2 = torch.randn((100, 27))      # 100×27 = 2700
b2 = torch.randn(27)             # 27
# 总计: 54+600+100+2700+27 = 3481
```

扩展到 11897 参数（10 维 embedding、200 隐藏神经元）：

```python
C  = torch.randn((27, 10))       # 27×10 = 270
W1 = torch.randn((30, 200))      # 30×200 = 6000
b1 = torch.randn(200)            # 200
W2 = torch.randn((200, 27))      # 200×27 = 5400
b2 = torch.randn(27)             # 27
# 总计: 270+6000+200+5400+27 = 11897
```

注意 `W1` 的输入从 6 变成了 30：block_size=3 × embedding_dim=10。

embedding 维度和隐藏层大小都是**超参数**，通过验证集上的表现来调整。训练 50000 轮（minibatch=32, lr=0.01）：

```python
# 训练集:       loss ≈ 2.17
# 验证集:       loss ≈ 2.19
```

两者非常接近——说明没有明显过拟合，模型确实学到了泛化的规律。

## 十二、字符嵌入的可视化

将 `C` 的前两个维度画成 2D 散点图：

```python
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0], C[i,1], itos[i], ...)
```

可以看到训练后字母在向量空间中的分布——元音字母（a, e, i, o, u）倾向于聚在一起，模式相似的字母靠得近。这验证了 embedding 确实学到了字符之间的语义关系。

## 十三、从模型采样

```python
context = [0] * block_size               # 从 ... 开始
while True:
    emb = C[torch.tensor([context])]      # (1, 3, 10)
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    ix = torch.multinomial(probs, num_samples=1, generator=g).item()
    context = context[1:] + [ix]         # 窗口右移
    if ix == 0: break
```

生成的样本质量明显优于 bigram：

```
carmah. amilli. khi. mili. thilahnanden. jazonen. deliah.
jareei. nellara. chaiiv. kaleigh. ham. jorniquinn. ...
```

读起来更像真实名字了——一个 MLP + embedding 就已经有了质变。

## 十四、阶段对比

| | Bigram (L2) | MLP (L3) |
|---|---|---|
| 输入 | 前 1 个字符 | 前 3 个字符 (block_size=3) |
| 字符表示 | One-hot 27 维 | Embedding 低维向量 |
| 隐藏层 | 无（直接 W×x） | tanh 全连接层 100/200 神经元 |
| 非线性 | 无 | tanh 激活 |
| 参数数量 | 729 | 3481 → 11897 |
| 训练方式 | 全批量 GD（100 轮） | Minibatch SGD（50000 轮） |
| 数据集划分 | 全部用于训练 | 80/10/10 三分 |
| 学习率 | 固定 50 | 指数搜索 → 0.1，末尾衰减 |
| 损失函数 | 手动 softmax + NLL | F.cross_entropy |
| 初始 loss | ≈ 3.77 | ≈ 20（随机权重极度不自信） |
| 最终 loss | ≈ 2.45 | ≈ 2.17 (train) / 2.19 (val) |
| 生成质量 | 很差 | 明显改善 |
