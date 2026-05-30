# Lecture 4 — makemore Part 3

## 课程仓库：[makemore](https://github.com/karpathy/makemore)

> 日期：2026-05-30

## 一、为什么要在乎激活值和梯度

03 课我们照着 Bengio 2003 论文实现了一个 MLP，能生成不错的名字。但在迈向更深更大的网络（RNN、GRU、LSTM）之前，需要先深入理解：训练过程中，神经网络内部的**激活值 (activations)** 和**梯度 (gradients)** 到底在发生什么。

RNN 是通用万能逼近器，原则上可以执行任何算法——但我们发现，用一阶梯度优化来训练它们非常困难。而理解这个困难的钥匙，就在于观察激活值和梯度在训练过程中的具体表现。这节课程虽然仍在 MLP 框架内，但揭示的问题是普遍性的。

## 二、第一个问题：初始 loss 为何高达 27.88

建立和 03 一样的 MLP（block_size=3, embedding=10, hidden=200, 11897 个参数），训练第 0 步 loss = **27.88**。但均匀猜测 27 个字符的 baseline 应该是：

```python
-torch.tensor(1/27.0).log()    # → 3.2958
```

loss 比理论值大 8 倍。问题出在 `logits = h @ W2 + b2`——`W2` 和 `b2` 都是 `torch.randn` 生成的标准正态分布随机值，导致 logits 极度发散：

```python
logits[0]   # [-2.35, +36.43, -10.73, +5.71, +18.64, ...]
# 最大值 36.43，最小值 -24.42，跨度巨大
```

`F.cross_entropy` 内部的 softmax 会把最大那几项压到接近 1，其余接近 0——模型对完全错误的预测"极度自信"（confident mispredictions）。这就是为什么 loss 爆高。

### 修复

既然我们希望初始阶段每个字符被选中的概率均匀（都接近 1/27），那就需要 logits 都接近 0。`logits = h @ W2 + b2` 中，把 `W2` 缩小、`b2` 清零：

```python
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01  # 缩小 100 倍
b2 = torch.randn(vocab_size, generator=g) * 0                 # 直接归零
```

现在 `W2` 的值接近 0，`b2` 就是 0，所有 logits ≈ 0 → softmax 接近均匀分布。

再训练，初始 loss = **3.32**，非常接近理论值 3.29。

### 为什么不能干脆把 W2 设为 0？

有人会想直接把 0.01 换成 0，初始 logits 全为 0 岂不更好？实际上更复杂：

如果把初始 logits 设为 0，优化器就需要先把所有 logits"压扁"(squashing down)，然后再重新分配 (rearranging)。这等于把损失函数中"缩小权重"那部分简单的收益拿掉了——loss 曲线会失去最初的快速下降阶段，那一截形如**曲棍球杆 (hockey stick)** 的快速下降消失了。模型从一开头就面对更困难的优化目标。

另外，就算这样干了，还有更深层的问题没解决。

## 三、第二个问题：Dead Neuron

修正了 logits 后，看隐藏层的状态。隐藏层激活值 h（经过 tanh）和预激活值 hpreact（tanh 之前的输入）的分布：

```python
h = torch.tanh(hpreact)
# h 的直方图：大量 ±1 极值
# hpreact 的直方图：分布极广，-15 到 15
```

`tanh` 在输入绝对值很大时输出饱和在 ±1。tanh 的导数是 `1 - tanh²(x)`——当 tanh(x) 接近 ±1 时，导数接近 0。梯度传到这里，**由链式法则可知传递的梯度 = 这一层的梯度 (≈0) × 之前传过来的梯度**——梯度就此被摧毁。

更直观地，观察死神经元的分布图：

```python
plt.imshow(h.abs() > 0.99, cmap='gray')   # 32 行 × 200 列
```

这张图是 32×200 的——每一行代表一条输入，每一列代表一个神经元。如果 `h.abs() > 0.99`，显示为白色。图中出现大量白色像素，意味着很多神经元对几乎所有输入都输出 ±1。

**如果某一列全是白色的，那这个神经元就是 dead neuron**——对任何输入都只会输出 ±1，梯度恒为 0，什么也学不到。

### 追根溯源

`hpreact = embcat @ W1 + b1`，`embcat` 来自 `C[X]`（正态分布初始化），和 `W1`（正态分布）点积后，值远离了 0 附近，分布被扩展开。15 的跨度对 tanh 来说太大了。

### 修复

```python
W1 = torch.randn((block_size*n_embd, n_hidden), generator=g) * 0.1
b1 = torch.randn(n_hidden, generator=g) * 0.01    # 不设 0，引入一点点 entropy
```

现在 hpreact 聚集在 [-1.5, 1.5]，tanh 不再饱和。再看死神经元图——白色全部消失，每个神经元都有梯度、都能学习。

### 修正效果汇总

这里用的是 Karpathy 笔记里的 benchmark：

```python
# 原始（未修正）：
train 2.124   val 2.168

# 修正 softmax 高置信错误：
train 2.07    val 2.13

# 修正 tanh 层初始化饱和：
train 2.035   val 2.1026
```

两次修正不仅让 loss 在数值上下降了，更重要的是让训练过程变得**健康**——梯度能正常流动，每个神经元都在参与学习。对于一个简单的一层网络，这些问题可能不那么致命（有时候不修也能训练）。但一旦网络变深变大，初始化没处理好就直接没法训练了——**所以在简单网络上养成好的初始化习惯至关重要**。

## 四、魔法数字的困境

0.1、0.01 这些系数是凭感觉选的"魔法数字"。层数多了、每层的维度不一样，怎么选？

### 数学分析

用一个小实验说明问题：

```python
x = torch.randn(1000, 10)      # 输入：均值 0，标准差 1
w = torch.randn(10, 200)       # 权重：均值 0，标准差 1
y = x @ w                      # (1000, 200)
# x.std ≈ 1.0
# y.std ≈ 3.1  ← 标准差被放大！
```

正态分布点积后，输出标准差显著增大。这不是我们想要的——我们希望激活值在网络各层保持相似的分布（均值 0，标准差 1 左右）。

数学上，要让 `x @ w` 的输出标准差保持为 1，需要**除以 fan-in 的平方根**：

```python
w = torch.randn(10, 200) / 10**0.5    # 除以 sqrt(10)
y = x @ w
# x.std ≈ 1.0
# y.std ≈ 1.0  ← 保持一致了
```

处理后输出分布不再发散，正态的钟形曲线保持得很好。

### Kaiming 初始化 (He et al. 2015)

[He et al. 2015](https://arxiv.org/abs/1502.01852) 论文《Delving Deep into Rectifiers》系统研究了神经网络初始化的最优方式。论文原本针对 CNN + ReLU/PReLU，但分析与我们的 tanh MLP 一脉相承。

ReLU 把所有负值映射为 0——正态分布大约一半（负数部分）被丢弃。为了补偿这个信息损失，初始化标准差需要放大为 `sqrt(2/n)`，即标准正态分布除以根号 n 后再乘 √2。公式为：

```
std = gain / sqrt(fan_in)
```

不同激活函数的 gain（增益系数）：
- Linear / Identity：gain = **1**
- ReLU：gain = **√2**
- tanh：论文建议 gain = **5/3**（我们上面用 1，因为是简单网络所以影响不大）

论文还分析了反向传播的初始化——结论是如果前向做好了，反向通常也不会差，两者只差一个与前后层神经元数相关的常数因子，不太关键。

PyTorch 内置实现：
```python
torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
```
`mode='fan_in'` 考虑前向传播（反向用 `fan_out`），一般默认就行。

### 退一步看

深度网络对初始化高度敏感——层数越深，激活值发散或消失的风险越大。但现代技术大大缓解了这个问题：

- **残差连接** (Residual Connections)
- **归一化层** (Batch Norm, Layer Norm, Group Norm)
- **更先进的优化器** (RMSprop, Adam)

正是这些创新让初始化不再需要过分精细的调参。Karpathy 甚至坦言：他在实际中也就是简单把权重除以 fan-in 的平方根。

## 五、Batch Normalization

[Ioffe & Szegedy 2015](https://arxiv.org/abs/1502.03167) 提出的 BN 是第一种能让可靠训练非常深的网络成为可能的技术。它在 2015 年具有里程碑意义。

### 核心原理

既然我们希望 tanh 的输入大致是标准正态分布（均值 0、方差 1），那就**直接强制标准化**：

```python
hpreact = (hpreact - hpreact.mean(0, keepdim=True)) / hpreact.std(0, keepdim=True)
```

沿 batch 维度（第 0 维）对 `[32, 200]` 计算均值和标准差得到 `[1, 200]` 的统计量，每个样本用同一组统计量归一化。注意这时每个归一化后的样本都融合了当前 batch 中 32 个样本的统计信息，但每个样本仍然是独立的。

### Scale and Shift

但如果在每一轮训练中都强制归一化成标准正态，梯度反向传播时也会受到这个约束，大大削弱网络的表达灵活性。BN 论文因此引入**可学习的缩放和偏移**：

```python
hpreact = bngain * hpreact_norm + bnbias   # y = ax + b
```

`bngain` 初始全 1，`bnbias` 初始全 0——所以初始输出仍然接近标准正态。但在训练中它们可以被反向传播调整，网络可以自主学习每层最佳的分布形态，不再受强制归一化的限制。运算完这一步再送入 tanh。

参数从 11897 增加到 12297（多了 `bngain` 和 `bnbias` 各 200 个值）。

### Benchmark 对比

借用 Karpathy 笔记里的数据：

```python
# 原始（未修正）：                      train 2.124   val 2.168
# 修正 softmax：                        train 2.07    val 2.13
# 修正 tanh 饱和：                      train 2.036   val 2.103
# 使用半理论化的 Kaiming 初始化：         train 2.038   val 2.168
# 加入 BatchNorm 层：                   train 2.067   val 2.105
```

BN 在这个简单网络上提升有限——因为网络只有一层，Kaiming 初始化已经几乎把增益全拿到了。但扩展到更深更大的网络后，不可能为每层都算出精确 gain，**BN 的优势才会真正体现**。

### BN 的隐式正则化与开销

BN 引入了一个有趣的副作用：同一个样本在不同 batch 中得到的归一化结果**略有不同**——取决于它和谁分到一组。这相当于在激活值上添加了微小的噪声，效果类似 Dropout，可以减轻过拟合。这是 BN 的隐式正则化效果。

但这种**样本之间的耦合**有时并不受欢迎——比如 batch size 很小、或者序列长度不一时，结果过度依赖 batch 内的其他样本。这也催生了后续的 **Layer Normalization、Instance Normalization、Group Normalization** 等不耦合样本的方法。

### 推理时的处理

训练时依赖 batch 统计量，但推理时可能只有一条数据——没有 batch 可算均值方差。原始论文的做法是训练完后，用全量训练数据重跑一遍，重新计算全局均值和方差（calibrate the batch norm）：

```python
with torch.no_grad():
    emb = C[Xtr]
    hpreact = emb.view(-1) @ W1 + b1
    bnmean = hpreact.mean(0, keepdim=True)
    bnstd  = hpreact.std(0, keepdim=True)

# 推理时用 bnmean / bnstd 替代 batch 统计量
hpreact = bngain * (hpreact - bnmean) / bnstd + bnbias
```

但"训练完再重算一遍"显然太麻烦了。实际做法是**训练过程中同步维护 running mean/std**：

```python
bnmean_running = torch.zeros((1, n_hidden))     # 初始 0
bnstd_running  = torch.ones((1, n_hidden))      # 初始 1

# 每个 batch 训练时：
bnmeani = hpreact.mean(0, keepdim=True)
bnstdi  = hpreact.std(0, keepdim=True)

with torch.no_grad():   # 更新时不需要梯度
    bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
    bnstd_running  = 0.999 * bnstd_running  + 0.001 * bnstdi
```

用的是动量更新（momentum update）——旧值权重 0.999，新 batch 的值权重 0.001。训练中 running 统计量逐步逼近全局统计量，推理时直接用：

```python
hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
```

### 两个重要的补充细节

**第一，分母加 epsilon。** 原论文建议在 `bnstd` 上加一个小 epsilon（如 1e-5），防止某个 batch 的标准差恰好为 0 导致除零错误：

```python
hpreact = bngain * (hpreact - bnmeani) / (bnstdi + eps) + bnbias
```

**第二，BN 前面的 bias 是多余的。** `hpreact = embcat @ W1 + b1`，紧接着 BN 归一化就减掉了均值——`b1` 的贡献被完全抵消，它的梯度恒为 0，根本不起作用。BN 有自己的 `bnbias` 来做偏移，所以**使用 BN 时，前面的 Linear 层可以不加 bias**。

### BN 总结

BN 做的事情可以归纳为：
- 前向传播：计算每个 batch 的 mean 和 std → 归一化为 unit Gaussian → 用可学习的 gain 和 bias 修正
- 同时维护 running_mean 和 running_var（buffers，不通过梯度更新，只做简单动量更新）
- 推理时用 running 统计量替代 batch 统计量

通常在有乘法的层（Linear / Conv）后面加上 BN 层。

## 六、搭建深层网络 + 诊断工具

### 代码模块化

把之前的代码用类似 PyTorch 的 API 重新组织：

- **Linear**：`x @ weight / fan_in**0.5 + bias`（Kaiming 初始化内嵌），bias 可选
- **BatchNorm1d**：gamma、beta 为可训练参数；running_mean、running_var 为 buffer（动量更新，不参与梯度）。`training` 标志控制 behavior
- **Tanh**：单纯的 `torch.tanh`，无参数

最终网络：5 个 `Linear → BN → Tanh` 块 + 一个 `Linear → BN` 输出层。embedding = 10，hidden = 100，47551 个参数。

```python
layers = [
    Linear(30, 100),   BatchNorm1d(100), Tanh(),
    Linear(100, 100),  BatchNorm1d(100), Tanh(),
    Linear(100, 100),  BatchNorm1d(100), Tanh(),
    Linear(100, 100),  BatchNorm1d(100), Tanh(),
    Linear(100, 100),  BatchNorm1d(100), Tanh(),
    Linear(100, 27),   BatchNorm1d(27),
]
```

特别注意初始化时的微调：
- 最后一层 BN 的 `gamma *= 0.1`：压低调自信，让输出层初始概率更均匀
- 中间层 Linear 的 `weight *= 5/3`：tanh 对应的 gain

### 诊断工具

给深层网络训练时，需要多维度判断网络是否健康：

**1. 激活分布**
```python
# 对于每一层 Tanh 输出：
print(f'layer {i}: mean {t.mean():+.2f}, std {t.std():.2f}, '
      f'saturated: {(t.abs()>0.97).float().mean()*100:.2f}%')
# 画各层的直方图，观察是否有层饱和
```
目标是各层分布均匀，没有大量神经元饱和在 ±1。

**2. 梯度分布**
```python
# 对于每一层 Tanh 输出的梯度：
print(f'layer {i}: mean {t.mean():+f}, std {t.std():e}')
# 画各层梯度的直方图
```
目标是各层都有梯度流动，没有消失（全 0）或爆炸（巨大值）的层。

**3. 权重梯度分布**
```python
# 对于每个权重矩阵：
print(f'weight {tuple(p.shape)}: mean {t.mean():+f}, '
      f'std {t.std():e}, grad:data ratio {(t.std()/p.std()).item():e}')
# 画梯度分布直方图
```
观察 gradient 的标准差与 data 的标准差的比例。

**4. update / data ratio 曲线**
```python
ud = (lr * p.grad.std() / p.data.std()).log10()
```
追踪 `(lr * grad.std) / data.std`。理想比例大约在 **1e-3**（即 log10 约为 -3）。如果某个参数的这个比例远大于 -3，说明更新的步幅相对于参数本身太大，训练不稳定；远小于 -3，说明学得太慢。

画一张横向参考线在 -3 处的图，观察各参数的更新比例是否在合理范围。

这些都是工程实践中快速定位"网络为什么训不动"的必备工具。

## 七、阶段对比

| | MLP (L3) | MLP+诊断 (L4) |
|---|---|---|
| 核心关注 | 模型结构（Build） | 训练过程内部状态（Diagnose） |
| 初始化方式 | 直接 randn | W2 缩小 + b2 清零 + Kaiming / BN |
| 初始 loss | 未关注（实际约 27） | 理论推导 3.29 → 修正至 3.32 |
| 激活分布 | 未检查 | 直方图观察饱和 → 消除 dead neuron |
| 梯度流动 | 未检查 | 层梯度/权重梯度/更新比全程追踪 |
| 归一化 | 无 | Batch Normalization（含全校准 + running 统计量） |
| 网络深度 | 1 层隐藏 | 5 层隐藏（模块化 Linear/BN/Tanh 搭建） |
| 代码组织 | 脚本式 | 完全类封装，与 PyTorch API 同构 |
| 历史意义 | — | 理解了为何 2015 年 BN 让深层网络训练成为可能 |
