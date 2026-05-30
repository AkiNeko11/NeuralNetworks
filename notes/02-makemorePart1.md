# Lecture 2 — makemore Part 1

## 课程仓库：[makemore](https://github.com/karpathy/makemore)

> 日期：2026-05-24

## 一、makemore 是什么

makemore 是一个**字符级语言模型 (character-level language model)**，在 32033 个人名的数据集上训练后，模型能学会生成类似风格的新名字。核心任务：给定前面的字符，预测下一个字符。

## 二、数据分析

先从数据入手。32033 个人名，最短 2 个字符，最长 15 个字符。

一开始简化假设：**只考虑前一个字符**对当前字符的影响（bigram 模型）。虽然实际上前面的整串字符都在决定当前字母的出现，但先从这里起步。

## 三、Bigram 计数模型

为了知道哪个字母最可能出现在特定字母后面，最简单的方法就是计数。

先从 Python 字典开始，遍历所有名字，用 `<S>` 和 `<E>` 包裹起止，统计每个 bigram 的出现次数。但字典太零散，改用 `torch.zeros((28, 28))` 的 2D 数组：行是第一个字符，列是第二个字符，`N[i, j]` 就是 j 跟在 i 后面的次数。

但 28 个符号（26 字母 + `<S>` + `<E>`）有冗余——`<E>` 那一行全空（结束符后面不会有字母），`<S>` 那一列也全空（开始符不可能在最后）。于是**统一为一个 `.` 符号**同时表示起止，索引设为 0，数组缩减为 27×27。用 `stoi`（string to index）做字符→索引的映射，`itos`（index to string）做反向映射。

`plt.imshow` 绘制 27×27 热力图，每个格子标注字母组合和次数。看一下起始符那行：

```python
N[0]   # [0, 4410, 1306, 1542, 1690, ...]
# .a: 4410 次, .b: 1306 次, .c: 1542 次...
# 名字以 a 开头的概率最大
```

### 从计数到采样

```python
p = N[0].float()          # 第一行计数
p = p / p.sum()           # 归一化为概率
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
itos[ix]                  # → 'c'
```

`torch.multinomial` 按概率分布采样——概率越高的字符越容易被抽到。`torch.Generator().manual_seed(2147483647)` 固定随机种子保证复现。

循环采样直到遇到 `.`（索引 0），就生成了一个名字。但生成的结果很差：

```
cexze.
momasurailezitynn.
konimittain.
llayn.
ka.
```

bigram model 就是这么糟糕，因为只考虑了两个字符的关系。

### 拉普拉斯平滑与广播

```python
P = (N+1).float()                              # 每个计数 +1
P = P / P.sum(1, keepdim=True)                # 按行归一化
```

`+1` 保证没有概率为零的 bigram——即使某个组合训练集从未出现，也有一个极小概率，避免采样时报错。

这里 `P.sum(1, keepdim=True)` 沿着列方向求每行总和，`keepdim=True` 保持维度得到 `(27, 1)`。除法时 `(27, 27) / (27, 1)`，广播规则把列向量横向复制 27 份，每行除以自己的行和，实现按行归一化。

### 模型评估：负对数似然 (NLL)

遍历全部 32k 个名字的所有 bigram，把每个概率取 log 再求和：

```python
log_likelihood += torch.log(P[ix1, ix2])   # 逐个 bigram 累加
n += 1
nll = -log_likelihood                      # 取负
avg_nll = nll / n                          # 2.4543
```

先从直观理解：一共 27 个字符，如果是均匀随机，每个搭配的概率是 `1/27 ≈ 0.037`。所以只要概率 > 0.037，就说明模型学到了一点东西——比如 `ma` 在这个矩阵里有将近 0.4 的概率，明显是学到了。

然后是理论：**似然 (Likelihood)** 是所有 bigram 概率的乘积，衡量整个数据集在模型下的概率。乘积越大模型越好。但概率都是 0.几，乘起来极小，不直观。所以用**对数似然 (Log Likelihood)**：`log(a*b*c) = log(a) + log(b) + log(c)`，范围 (-∞, 0]，越接近 0 越好。

再取负，变成 (0, +∞)，这就是**负对数似然 (NLL)**。除以总数 = 平均 NLL，可作为 loss 优化目标。

计数模型的平均 NLL ≈ **2.45**，明显优于随机基线 3.3。

## 四、用神经网络重新实现 Bigram

### 构造训练数据

以 "emma" 为例，滑动生成 bigram 对 (输入字符, 下一个字符)：

```
. → e    输入 0，标签 5
e → m    输入 5，标签 13
m → m    输入 13，标签 13
m → a    输入 13，标签 1
a → .    输入 1，标签 0
```

神经网络的任务：**输入一个字符索引，输出 27 个概率，预测下一个字符**。

### One-Hot 编码

`torch.tensor` 存的是 int64（注意：`torch.Tensor` 默认 float32，`torch.tensor` 默认 int64），不能直接和浮点权重做乘法。所以需要独热编码：

```python
xenc = F.one_hot(xs, num_classes=27).float()   # (N, 27)
```

本质是全 0 向量，只在对应索引位置置 1。这样向量和权重矩阵的每一列做点积才有意义。

### 权重矩阵与 logits

```python
W = torch.randn((27, 27))         # 27 个神经元，每个接收 27 维输入
logits = xenc @ W                  # (N, 27)
```

`W` 的每一列是一个神经元的权重向量。**重点**：`(xenc @ W)[3, 13]` 是第 3 个输入向量与第 13 列权重向量的点积：

```python
(xenc @ W)[3, 13]               # → 0.3901
# 等价于
xenc[3]                          # one-hot 向量，只有第 13 位是 1
W[:, 13]                        # 第 13 列权重
(xenc[3] * W[:, 13]).sum()      # 逐元素乘再求和 = 点积
```

因为 `xenc[3]` 是 one-hot（只有第 13 位是 1），这个点积本质上就是取 `W[13, 13]` 的值。One-hot 下的矩阵乘法等价于**查表**——直接把对应的权重行挑出来。这个值被称为第 13 个输出神经元对第 3 个输入的**激发率 (firing rate)**。

### logits → softmax → 概率

神经网络直接输出的是有正有负的任意实数，不能当概率（概率必须是正的且和为 1）。观察之前计数的那些概率：都是正数、和为 1——而神经网络输出有正有负，也不可能直接等同于整数 counts。换个角度：**把这些输出解释为 log-counts**，然后 exp 变正数，再归一化：

```python
logits = xenc @ W                              # log-counts（有正有负）
counts = logits.exp()                          # 等效计数（正数）
probs = counts / counts.sum(1, keepdim=True)   # 概率分布（和为 1）
```

后两步合起来叫 **softmax**。

以第一个输入 `.` 为例，神经网络给出的 27 条概率就是对下一个字母的预测分布——然后通过调 W 让这些概率更贴近真实的标签。

### 损失函数与逐条验证

```python
loss = -probs[torch.arange(N), ys].log().mean()
```

走一遍前 5 条数据，手动看看模型给正确答案的概率：

| bigram | 模型给正确标签的概率 | NLL |
|--------|-------------------|-----|
| . → e | 0.0123 | 4.40 |
| e → m | 0.0181 | 4.01 |
| m → m | 0.0267 | 3.62 |
| m → a | 0.0737 | 2.61 |
| a → . | 0.0150 | 4.20 |
| **平均** | | **3.77** |

初始的平均 NLL = 3.77——随机的 W 给正确答案的概率非常低。接下来就是通过梯度下降来降低这个 loss。

### 训练

```python
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
# 共 228146 个 bigram 训练样本

W = torch.randn((27, 27), requires_grad=True)

for k in range(100):
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()

    W.grad = None                               # 梯度归零
    loss.backward()                             # PyTorch 自动反向传播
    W.data += -50 * W.grad                      # 沿梯度反方向更新
```

几个要点：
- **228146 个训练样本**：全批量梯度下降，每轮算所有数据
- **L2 正则化** `0.01 * (W**2).mean()`：鼓励权重尽可能小、分布均匀，防止过拟合。`(W**2).mean()` 计算 W 所有元素的平方均值作为正则化项
- **学习率 50**：比 micrograd 的 0.05 大很多，因为是全批量、参数少
- **`W.grad = None`**：效果等价于 `zero_()`，但更高效
- 在 forward 时 PyTorch 内部构建了计算图（和 micrograd 的 `_prev` 一样），backward 时自动沿图链式求导

## 五、Bigram 模型的局限与未来

训练出来的 W 最终学到的和直接计数归一化几乎一样——因为这个模型本质上就是统计 bigram 频率。生成的名字仍然很差（`cexze`, `momasurailezityha`, `llayn`…），因为只看前一个字符，完全没有上下文。

如果用均匀分布替代学习到的概率去采样（`p = torch.ones(27)/27.0`），结果同样是"史"——这从反面说明尽管 bigram 模型不完美，但它仍然比均匀随机好得多。

但梯度下降方案的优势在于灵活性：可以扩展为输入多个字符、使用更深网络、加入非线性激活——这些纯计数做不到。后续课程正是沿着这个方向逐步改进。

## 六、阶段对比

| | micrograd (L1) | makemore P1 (L2) |
|---|---|---|
| 框架 | 纯 Python 手写 | PyTorch |
| 数据类型 | 标量 Value | Tensor 矩阵 |
| 输入编码 | 直接数值 | One-hot |
| 输出 | 任意值 | logits → softmax → 概率 |
| 损失 | MSE | 负对数似然 (交叉熵) |
| 正则化 | 无 | L2 (0.01) |
| 训练样本 | 4 | 228146 |
| 优化 | 手动 backward | loss.backward() 自动求导 |
