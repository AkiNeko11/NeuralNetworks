# Lecture 5 — makemore Part 4

## 课程仓库：[makemore](https://github.com/karpathy/makemore)

> 日期：2026-06-19

## 一、为什么还要再写一遍反向传播

前面的课程里我们已经搭了 MLP，能生成不错的名字了。但在继续往 RNN 等更复杂的架构走之前，Karpathy 认为需要多留一节课——这次不引入新模型，而是把 `loss.backward()` 这一行拆开，看看它里面到底发生了什么。

01 课我们在 micrograd 里写过反向传播，但那是在标量层面。Tensor 级别的 backprop 和标量有些不同——tensor 涉及矩阵乘法、广播、求和等操作，每个操作的梯度传播规则都不一样。所以尽管有了 micrograd 的经验，这节课还是从头把整个 MLP 的反向传播手写一遍。

Karpathy 提到，虽然现在几乎所有人都直接调用 `loss.backward()`，但在 2006 年和 2014 年的论文里，研究者们还在手写 backward。理解底层发生了什么，对于 debug 和设计新模型都是有意义的。

## 二、前向传播的展开

这节课的前向传播和 04 课有明显的不同——不再用 `F.cross_entropy` 和批归一化内置函数，而是把它们全部展开成基础运算，得到一个可管理的**中间变量链**。目的是在前向过程中产生足够多的中间 tensor，让后续反向传播时能逐个求导。

数据准备沿用了 04 的范式：32033 个名字，block_size=3，训练/验证/测试 80/10/10 三分，共 228146 条训练数据。

模型配置先从小版本开始（方便验证梯度）：

```python
n_embd = 10          # 字符嵌入维度
n_hidden = 64        # 隐藏层神经元（比之前的 200 小，方便调试）
# 总参数: 4137
```

参数初始化和 04 课保持一致——W1 用 Kaiming 初始化 `(5/3)/sqrt(fan_in)`，W2 和 b2 缩小到 0.1 倍，b1 也缩到 0.1，避免初始 logits 发散和 tanh 饱和。特别说明这些非标准初始化是为了让梯度验证时不会因全零掩盖错误。

前向传播把 `F.cross_entropy(logits, Yb)` 拆成了完整的几步：

```python
# 交叉熵的手动实现——把数值稳定版 softmax + log + nll 全展开
logit_maxes = logits.max(1, keepdim=True).values   # 每行最大值
norm_logits = logits - logit_maxes                  # 减去最大值防 exp 溢出
counts = norm_logits.exp()                          # 指数化 → "等效计数"
counts_sum = counts.sum(1, keepdims=True)           # 每行求和
counts_sum_inv = counts_sum**-1                     # 用 -1 次方而非除法
probs = counts * counts_sum_inv                     # 概率分布
logprobs = probs.log()                              # 对数概率
loss = -logprobs[range(n), Yb].mean()               # 负对数似然均值
```

这里有一个细节：`counts_sum**-1` 而不是 `1.0 / counts_sum`，是为了保证手动梯度能和 PyTorch 的结果精确到 bit 一致。这两种写法在数学上等价，但 PyTorch 内部的梯度实现路径不同，只有用 `**-1` 才能对上。

前向传播还对每个中间变量调用了 `t.retain_grad()`——这样 PyTorch 会保留非叶子节点的梯度，方便后续逐项比对。

## 三、手动反向传播：逐变量推导

这是整节课的核心——对前向传播中产生的每一个中间变量，手动写出它的局部梯度，然后按链式法则从 loss 往回传。

为了方便验证，先定义一个比对工具：

```python
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()       # 是否精确相等
    app = torch.allclose(dt, t.grad)           # 是否近似相等
    maxdiff = (dt - t.grad).abs().max().item() # 最大误差
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')
```

比较手动计算的 `dt` 和 PyTorch 自动求导得到的 `t.grad`。

以下按正向顺序反推，每条规则都附了推导逻辑：

### 3.1 loss → logprobs

```python
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
```

`loss = -logprobs[range(n), Yb].mean()`，所以只有正确标签位置有梯度 `-1/n`，其他位置为 0。类比 `loss = (a+b+c)/3`，`dloss/da = 1/3`（这里多了个负号）。

### 3.2 logprobs → probs

```python
dprobs = (1.0 / probs) * dlogprobs
```

`logprobs = log(probs)`，`d(log(x))/dx = 1/x`，所以局部梯度是 `1/probs`。

### 3.3 probs → counts_sum_inv 和 counts（双路梯度）

这里是第一个难点。`probs = counts * counts_sum_inv`，但 `counts` 和 `counts_sum_inv` 之间有关联——`counts_sum_inv = (sum(counts))**-1`。counts 通过**两条路**影响 probs：

- **路 1（分子）**：counts 直接作为因子，`probs = counts * k`（k = counts_sum_inv 视为常数），局部梯度 = `k = counts_sum_inv`
- **路 2（分母）**：counts 变大 → counts_sum 变大 → counts_sum_inv 变小 → probs 变小

```python
# 路 1：分子梯度
dcounts = counts_sum_inv * dprobs
# counts_sum_inv → counts_sum
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
# 路 2：分母梯度，counts_sum = sum(count_i)，对每个 count_i 导数为 1
dcounts += torch.ones_like(counts) * dcounts_sum
```

`-counts_sum**-2` 是 `d(1/S)/dS = -1/S²`。最后把两条路的梯度相加才是完整的 `dcounts`。

### 3.4 counts → norm_logits

```python
dnorm_logits = counts * dcounts
```

`counts = exp(norm_logits)`，`d(exp(x))/dx = exp(x) = counts`。

### 3.5 norm_logits → logits 和 logit_maxes（双路梯度）

```python
dlogits = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
```

`norm_logits = logits - logit_maxes`，对 logits 的局部梯度是 1（直接复制），对 logit_maxes 的局部梯度是 -1。

但 `dlogit_maxes` 只影响每行最大值所在的那个 logits 元素——因为 logit_maxes 只从那个位置取的值。用 `F.one_hot` 构建掩码，只在最大值位置添加 `dlogit_maxes` 的梯度。

### 3.6 logits → h, W2, b2

```python
dh = dlogits @ W2.T          # logits = h @ W2 + b2
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)         # b2 对每个样本的 logits 加同样值，梯度沿 batch 求和
```

`y = x @ W + b`，`dy/dx = W.T`，`dy/dW = x.T`，`dy/db = 1`。

### 3.7 h → hpreact（tanh 反向）

```python
dhpreact = (1.0 - h**2) * dh
```

`h = tanh(hpreact)`，`d(tanh(x))/dx = 1 - tanh²(x) = 1 - h²`。

### 3.8 hpreact → bngain, bnraw, bnbias

```python
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)    # hpreact = bngain * bnraw + bnbias
dbnraw = bngain * dhpreact
dbnbias = dhpreact.sum(0, keepdim=True)
```

`y = gain * x + bias`，`dy/dgain = x`，`dy/dx = gain`，`dy/dbias = 1`。

### 3.9 bnraw → bndiff 和 bnvar_inv（双路梯度）

```python
dbndiff = bnvar_inv * dbnraw                          # bnraw = bndiff * bnvar_inv
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
```

### 3.10 bnvar_inv → bnvar

```python
dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
```

`bnvar_inv = (bnvar + 1e-5)**-0.5`，`d(x**-0.5)/dx = -0.5 * x**-1.5`。

### 3.11 bnvar → bndiff2 → bndiff

```python
dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar    # bnvar = 1/(n-1) * sum(bndiff2)
dbndiff += (2*bndiff) * dbndiff2                             # bndiff2 = bndiff**2
```

注意这里用的是贝塞尔校正 `n-1`（无偏方差估计），不是 `n`。

### 3.12 bndiff → hprebn 和 bnmeani（双路梯度）

```python
dhprebn = dbndiff.clone()                    # bndiff = hprebn - bnmeani, 对 hprebn 梯度为 1
dbnmeani = (-dbndiff).sum(0)                 # 对 bnmeani 梯度为 -1
dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)
```

`bndiff = hprebn - bnmeani`，而 `bnmeani = 1/n * sum(hprebn)`。所以 bnmeani 变化时会影响所有 hprebn 元素——`d(bnmeani)/d(hprebn_i) = 1/n`。两部分梯度都要加上。

### 3.13 hprebn → embcat, W1, b1

```python
dembcat = dhprebn @ W1.T          # hprebn = embcat @ W1 + b1
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)
```

### 3.14 embcat → emb → C

```python
demb = dembcat.view(emb.shape)    # embcat 是 emb 的 reshape，一一对应
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
        ix = Xb[k, j]
        dC[ix] += demb[k, j]      # 把每个位置的梯度累加到对应的字符嵌入行
```

C 的梯度需要**散射累加**——同一个字符可能在 batch 中出现在多个位置，每个位置的梯度都要加到 C 的对应行上。

## 四、验证：与 PyTorch 逐项比对

用 cmp() 对所有 25 个中间变量逐一比对手动梯度和 PyTorch 自动梯度：

```
logprobs        | exact: True  | approximate: True  | maxdiff: 0.0
probs           | exact: True  | approximate: True  | maxdiff: 0.0
counts_sum_inv  | exact: True  | approximate: True  | maxdiff: 0.0
...
C               | exact: True  | approximate: True  | maxdiff: 0.0
```

全部 25 项都是 `exact: True`——手动计算的梯度和 PyTorch 的 autograd 结果**逐比特完全一致**。这证明了前面的链式法则推导每一步都是对的。

## 五、交叉熵的简化反向传播

逐变量求导验证了正确性之后，下一步是推导出一个**紧凑的公式**，一次性算出 logits 的梯度，而不是通过中间变量一步步传。

前向传播中交叉熵的展开过程：

```
logits → softmax → probs → log → logprobs → 取正确位置 → 取负 → 取均值
```

对这条链直接求导并化简后，得到一个极其简洁的公式：

```python
dlogits = F.softmax(logits, 1)        # 先取 softmax
dlogits[range(n), Yb] -= 1            # 正确标签位置减 1
dlogits /= n                           # 除以 batch 大小
```

推导逻辑：展开 loss 对 logits 的偏导——`∂loss/∂logits[i] = (probs[i] - 1_{i==target}) / n`。softmax 的结果就是 probs，正确标签位置减去 1 再除 n。

比对结果：

```
logits | exact: False | approximate: True | maxdiff: 6.3e-09
```

只能近似相等，差异在 10⁻⁹ 量级——这是因为 PyTorch 的 `F.cross_entropy` 内部用了不同的数值计算路径。但 6e-9 的误差在 float32 精度下完全可以忽略。

## 六、BatchNorm 的简化反向传播

同样对 BatchNorm 的前向传播链进行直接求导。前向过程：

```
hprebn → mean → diff → var → var_inv → bnraw → hpreact（× gain + bias）
```

对 `hprebn` 求偏导并化简，得到一个紧凑公式：

```python
dhprebn = bngain * bnvar_inv / n * (
    n * dhpreact
    - dhpreact.sum(0)
    - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0)
)
```

三个部分的含义：
- `n * dhpreact`：直接的梯度项
- `- dhpreact.sum(0)`：均值对每个元素的影响
- `- n/(n-1) * bnraw * (dhpreact*bnraw).sum(0)`：方差对每个元素的影响

比对结果：`exact: False | approximate: True | maxdiff: 9.3e-10`——同样是近似相等，误差在 10⁻¹⁰ 量级。

## 七、综合训练：用手动反向传播训练 MLP

把前面推导的简化公式全部整合，用自己写的 backward 替代 `loss.backward()`，完整训练一遍：

```python
n_embd = 10          # 字符嵌入维度
n_hidden = 200       # 隐藏层神经元
# 总参数: 12297

max_steps = 200000
batch_size = 32

with torch.no_grad():
    for i in range(200000):
        # minibatch
        ix = torch.randint(0, Xtr.shape[0], (32,))
        Xb, Yb = Xtr[ix], Ytr[ix]

        # forward pass（和之前一样）
        emb = C[Xb]; embcat = emb.view(-1, 30)
        hprebn = embcat @ W1 + b1
        # BatchNorm
        bnmean = hprebn.mean(0, keepdim=True)
        bnvar = hprebn.var(0, keepdim=True, unbiased=True)
        bnvar_inv = (bnvar + 1e-5)**-0.5
        bnraw = (hprebn - bnmean) * bnvar_inv
        hpreact = bngain * bnraw + bnbias
        h = torch.tanh(hpreact)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Yb)

        # ---- 手动反向传播 ----
        dlogits = F.softmax(logits, 1)
        dlogits[range(32), Yb] -= 1; dlogits /= 32

        dh = dlogits @ W2.T; dW2 = h.T @ dlogits; db2 = dlogits.sum(0)
        dhpreact = (1.0 - h**2) * dh
        dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
        dbnbias = dhpreact.sum(0, keepdim=True)
        dhprebn = bngain*bnvar_inv/32 * (32*dhpreact - dhpreact.sum(0)
                    - 32/31*bnraw*(dhpreact*bnraw).sum(0))

        dembcat = dhprebn @ W1.T; dW1 = embcat.T @ dhprebn; db1 = dhprebn.sum(0)
        demb = dembcat.view(emb.shape)
        dC = torch.zeros_like(C)
        for k in range(32):
            for j in range(3):
                dC[Xb[k,j]] += demb[k,j]

        # 更新参数
        lr = 0.1 if i < 100000 else 0.01   # 学习率衰减
        for p, grad in zip(parameters, [dC,dW1,db1,dW2,db2,dbngain,dbnbias]):
            p.data += -lr * grad
```

训练用了 `torch.no_grad()` 上下文管理器——因为手动反向传播不需要 PyTorch 构建计算图，关掉 autograd 可以大幅提升速度。

200000 步训练过程中的 loss 变化：

```
      0/ 200000: 3.8116
  10000/ 200000: 2.1846
  50000/ 200000: 2.3119
 100000/ 200000: 2.0201      ← 学习率从 0.1 衰减到 0.01
 150000/ 200000: 2.2273
 190000/ 200000: 1.8454
```

前 100k 步用 lr=0.1 大步快跑，后 100k 步用 lr=0.01 精细收敛。loss 整体呈下降趋势，虽然中间有波动（minibatch 的噪声），但最后降到了 1.85 附近。

## 八、评估与采样

训练结束后，先用全量训练集校准 BatchNorm 的全局统计量：

```python
with torch.no_grad():
    emb = C[Xtr]; embcat = emb.view(-1, 30)
    hpreact = embcat @ W1 + b1
    bnmean = hpreact.mean(0, keepdim=True)
    bnvar = hpreact.var(0, keepdim=True, unbiased=True)
```

推理时用 `bnmean` 和 `bnvar` 替代 batch 统计量，用 `bngain` 和 `bnbias` 做标准的归一化。

最终结果：
- 训练集 loss：**2.0702**
- 验证集 loss：**2.1094**

这和 Karpathy 笔记本里的结果（train 2.07, val 2.12）基本一致，说明手动反向传播的实现是正确的。

采样生成的名字：

```
carlah.    amori.    kitzimri.    reety.    salaysie.
mahnen.    delynn.   jareei.      ner.      kia.
chaiir.    kaleigh.  ham.         joce.     quinn.
saline.    liven.    coraelo.     dearyxia. kael.
```

质量不错——`carlah`、`delynn`、`quinn`、`saline` 这些都读起来像真名字，和 04 课用 PyTorch autograd 训练出来的水平相当。

## 九、阶段对比

| | MLP+诊断 (L4) | 手动反向传播 (L5) |
|---|---|---|
| 核心关注 | 训练过程的内部状态 | 梯度反向传播的底层实现 |
| 反向传播 | `loss.backward()` 自动 | 逐变量手写 + 简化公式 |
| 交叉熵梯度 | 不展开 | softmax - one_hot / n |
| BatchNorm 梯度 | 不展开 | 推导紧凑公式（三项） |
| 梯度验证 | 观察分布和比例 | 与 PyTorch autograd 逐比特比对 |
| 训练方式 | PyTorch autograd | 手动 backward + no_grad |
| 训练结果 | train 2.07, val 2.11 | train 2.07, val 2.11 |
| 最深收获 | 什么是健康的训练状态 | 链式法则在 tensor 层面如何执行 |

从 L4 到 L5 的转变，本质上是从"**知道怎么诊断**"进阶到"**知道怎么实现**"。L4 教你怎么看激活值和梯度的分布、怎么判断网络健不健康；L5 让你亲手把每一层的梯度规则实现一遍，理解 autograd 引擎到底在做什么。两课合在一起，构成了对神经网络训练过程的完整理解。
