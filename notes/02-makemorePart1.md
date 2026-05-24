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

但 28 个符号（26 字母 + `<S>` + `<E>`）有冗余——`<E>` 那一行全空（结束符后面不会有字母），`<S>` 那一列也全空（开始符不可能在最后）。于是**统一为一个 `.` 符号**同时表示起止，索引设为 0，数组缩减为 27×27。

这里有个小细节：用 `stoi`（string to index）做字符→索引的映射，`itos`（index to string）做反向映射。

`plt.imshow` 绘制 27×27 热力图，每个格子标注对应的字母组合和次数，哪些字母组合更常见一目了然。

### 从计数到采样

```python
p = N[0].float()          # 第一行计数
p = p / p.sum()           # 归一化为概率
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
```

`torch.multinomial` 按概率分布采样——概率越高的字符越容易被抽到。循环采样直到遇到 `.`（索引 0），就生成了一个名字。

`torch.Generator().manual_seed(2147483647)` 固定随机种子，保证每次运行结果可复现。

### 拉普拉斯平滑 (Laplace Smoothing)

```python
P = (N+1).float()                              # 每个计数 +1
P = P / P.sum(1, keepdim=True)                # 按行归一化
```

`+1` 保证没有概率为零的 bigram——即使某个组合训练集从未出现，也有一个极小概率，避免采样时报错。

这里 `P.sum(1, keepdim=True)` 沿着列方向求每行总和，`keepdim=True` 保持维度得到 `(27, 1)` 的形状。除法时 `(27, 27) / (27, 1)`，广播规则把列向量横向复制 27 份，每行除以自己的行和，实现按行归一化。

### 模型评估：负对数似然 (NLL)

似然是所有 bigram 概率的乘积，衡量整个数据集在模型下的概率。乘积越大越好，但小数连乘会很小，所以取对数变成加法：`log(a*b*c) = log(a) + log(b) + log(c)`，范围 (-∞, 0]，越接近 0 越好。

再取负，变成 (0, +∞)，越小越好，这就是**负对数似然 (NLL)**。除以总数就是平均 NLL，可以作为 loss。

均匀概率的基线：`1/27 ≈ 0.037`，对应 log ≈ -3.3。只要模型给某个组合的概率 > 0.037，就说明学到了东西。

计数模型的平均 NLL ≈ **2.45**，明显优于 3.3 的随机基线。

## 四、用神经网络重新实现 Bigram

### One-Hot 编码

与计数模型不同，神经网络需要把输入变成可乘权重矩阵的形式。`torch.tensor` 存的是 int64（注意：`torch.Tensor` 默认 float32，`torch.tensor` 默认 int64），不能直接和浮点权重做乘法，所以需要独热编码：

```python
xenc = F.one_hot(xs, num_classes=27).float()   # (N, 27)
```

本质是创建一个全 0 向量，只在对应索引位置置 1。

### 权重矩阵与 logits

```python
W = torch.randn((27, 27))         # 27 个神经元，每个接收 27 维输入
logits = xenc @ W                  # (N, 27)
```

`W` 的每一列是一个神经元的权重向量。重点：`(xenc @ W)[3, 13]` 是**第 3 个输入向量与第 13 列权重向量的点积**：

```python
(xenc @ W)[3, 13]               # → 某个值
# 等价于
(xenc[3] * W[:, 13]).sum()      # 逐元素乘再求和
```

因为 `xenc[3]` 是 one-hot（只有第 13 位是 1），这个点积本质上就是取 `W` 第 13 行第 13 列的值，即**第 13 个输出神经元对第 3 个输入的激发率 (firing rate)**。One-hot 下的矩阵乘法等价于查表——直接把对应的权重行挑出来。

### logits → softmax → 概率

神经网络直接输出的是有正有负的任意实数，不能当概率（概率必须是正的且和为 1）。解决办法是把它理解为 **log-counts**，然后取 exp 变成"等效计数"，最后归一化：

```python
logits = xenc @ W                              # log-counts
counts = logits.exp()                          # 等效计数（正数）
probs = counts / counts.sum(1, keepdim=True)   # 概率分布（和为 1）
```

后两步合起来就叫 **softmax**，是机器学习里把 logits 转成概率的常用操作。

### 损失函数

```python
loss = -probs[torch.arange(N), ys].log().mean()
```

`probs[torch.arange(N), ys]` 取出每个样本正确标签对应的概率（模型对正确答案的置信度），对其取 log、取负、求平均。这就是交叉熵损失。

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
- **L2 正则化** `0.01 * (W**2).mean()`：鼓励权重尽可能小，防止过拟合
- **学习率 50**：比 micrograd 的 0.05 大很多，因为是全批量梯度下降，步长可以大
- **`W.grad = None`**：效果等价于 `zero_()`，但更高效
- 在 forward 时 PyTorch 内部构建了计算图（和 micrograd 的 `_prev` 一样），backward 时自动沿图链式求导

## 五、Bigram 模型的局限

训练出来的 W 最终学到的和直接计数归一化几乎一样——因为这个模型本质上就是统计 bigram 频率。生成的样本质量很差（`cexze`, `momasurailezityha`, `llayn`…），因为只看前一个字符，完全没有上下文。

但梯度下降方案的优势在于灵活性：可以扩展为输入多个字符、使用更深网络、加入非线性激活——这些纯计数做不到。后续课程正是沿着这个方向逐步改进。

## 六、阶段对比

| | micrograd (L1) | makemore P1 (L2) |
|---|---|---|
| 框架 | 纯 Python 手写 | PyTorch |
| 数据类型 | 标量 Value | Tensor 矩阵 |
| 输入编码 | 直接数值 | One-hot |
| 输出 | 任意值 | logits → softmax → 概率 |
| 损失 | MSE | 负对数似然 (交叉熵) |
| 正则化 | 无 | L2 |
| 优化 | 手动 backward | loss.backward() 自动求导 |
