# Lecture 1 — micrograd

> 日期：2026-05-21

## 一、从数值微分到自动微分

课程从最基础的导数定义出发：取极小的增量 h，计算 `(f(x+h) - f(x)) / h`，验证了 f(x) = 3x² - 4x + 5 在 x=3 处的导数为 14。

然后对一个多变量表达式 `d = a * b + c` 手动计算偏导数（对 a 求导得 b = -3），建立了直觉：每个变量对最终输出的"敏感度"就是它的梯度。

自动微分的核心思想：**构建计算图，前向计算输出，反向传播梯度**。不需要手动推公式，每一步运算都定义了对应的反向传播规则。

## 二、Value 类实现

### 数据结构

```python
class Value:
    data:   float      # 该节点的数值
    grad:   float      # 该节点对最终输出的梯度（反向传播时累积）
    _prev:  set        # 前驱节点集合（计算图的边）
    _op:    str        # 产生该节点的运算名称
    _backward: fn      # 该节点的反向传播函数（闭合 over 输入节点）
```

关键设计：
- `_prev` 用 **set** 存储，去重且查找高效
- `_backward` 默认是空函数 `lambda: None`，叶子节点不需要反向传播
- 每个运算方法在创建输出节点时，同时定义一个 closure 作为 `_backward`

### 支持的运算

| 运算 | 正向 | 反向（梯度传播） |
|------|------|-----------------|
| `+` | `a + b` | `a.grad += 1.0 * out.grad`, `b.grad += 1.0 * out.grad` |
| `*` | `a * b` | `a.grad += b.data * out.grad`, `b.grad += a.data * out.grad` |
| `**` | `a ** n` | `a.grad += n * a.data^(n-1) * out.grad` |
| `tanh` | `tanh(x)` | `x.grad += (1 - t²) * out.grad` |
| `exp` | `e^x` | `x.grad += e^x * out.grad` |

其他运算通过组合实现：
- `a / b` → `a * b**(-1)`
- `a - b` → `a + (-b)`
- `-a` → `a * (-1)`

### 反向传播流程

```python
def backward(self):
    # 1. 拓扑排序——从根节点 DFS，保证所有依赖节点排在前面
    topo = build_topo(self)
    # 2. 根节点对自己的梯度为 1
    self.grad = 1.0
    # 3. 逆序遍历拓扑序列，依次执行每个节点的 _backward
    for node in reversed(topo):
        node._backward()
```

拓扑排序保证了**链式法则的正确顺序**——每个节点在反向传播前，它的所有"下游"节点都已经计算完了梯度。

## 三、计算图可视化

使用 graphviz 绘制计算图，核心是 `trace` + `draw_dot`：

- **trace**：递归遍历 `_prev`，收集所有节点和边
- **draw_dot**：为每个数据节点画矩形（显示 label / data / grad），为每个运算画圆点，数据经过运算流向输出（LR 布局）

这个可视化工具对理解反向传播的梯度流动非常有帮助。

## 四、踩坑：梯度覆盖 vs 梯度累积

这是整个课程中最重要的 bug 发现。

### 问题场景

当同一个变量在计算图中被多次使用：

```python
a = Value(3.0)
b = a + a    # b = a + a，a 出现了两次
b.backward()
# 期望 a.grad = 2.0（因为 db/da = 1 + 1 = 2）
# 实际 a.grad = 1.0
```

### 原因

在 `__add__` 的 `_backward` 中，`self` 和 `other` 指向的是**同一个 a 对象**：

```python
self.grad = out.grad * 1.0     # a.grad = 1
other.grad = out.grad * 1.0    # a.grad = 1  ← 覆盖了！
```

两次赋值都是对同一个 `a.grad`，第二次覆盖了第一次。

### 更复杂的情况

```python
d = a * b    # a 和 b 各用一次
e = a + b    # a 和 b 又各用一次
f = d * e
```

反向传播时，先从 d 传播梯度给 a 和 b，再从 e 传播梯度给 a 和 b。但因为用的是 `=` 而非 `+=`，后面的梯度把前面的覆盖了。

### 修复

```python
# 从
self.grad = out.grad * 1.0
other.grad = out.grad * 1.0

# 改为
self.grad += out.grad * 1.0
other.grad += out.grad * 1.0
```

由于初始化时 `grad = 0.0`，第一次 `+=` 等价于 `=`。但当变量被多次使用时，`+=` 能正确累积来自不同路径的梯度。

**本质**：多元链式法则中，一个变量可能通过多条路径影响最终输出，每条路径贡献一部分偏导数，最终梯度是所有路径贡献的**总和**。`+=` 正是这个求和过程。

## 五、tanh 的两种实现

### 方法一：直接实现

```python
def tanh(self):
    t = (e^(2x) - 1) / (e^(2x) + 1)
    反向: self.grad += (1 - t²) * out.grad
```

### 方法二：用基础运算拆解

```python
e = (2*n).exp()          # e^(2x)
o = (e - 1) / (e + 1)    # tanh 公式
```

拆解后，tanh 的计算图中会出现 `exp`、`+`、`-`、`/` 等基础运算节点，反向传播由这些基础运算各自的 `_backward` 自动完成。这验证了**正确的原子运算可以组合出任意复杂函数的自动微分**。

## 六、PyTorch 对比

用 PyTorch 的 `torch.Tensor` + `requires_grad=True` 做了同样的计算，结果与手写 micrograd 一致：

- `o = torch.tanh(x1*w1 + x2*w2 + b)` → `o.backward()`
- 各参数的 `.grad` 值与 micrograd 版本吻合

PyTorch 的 `tensor.grad_fn` 展示了底层同样是一个计算图（`<TanhBackward0>`）。

## 七、构建神经网络

从底层到上层逐级搭建：

```
Value（标量 + 自动微分）
  → Neuron:   一组 weights + bias + tanh 激活
    → Layer:  多个 Neuron 并行，接受相同输入
      → MLP:  多层 Layer 串联，nin → nouts[0] → nouts[1] → ...
```

- **Neuron.parameters()**: 返回 `self.w + [self.b]`
- **Layer.parameters()**: 遍历所有 neuron 收集参数
- **MLP.parameters()**: 遍历所有 layer 收集参数

这种递归的参数收集方式让训练循环非常简洁。

## 八、完整训练流程

```python
# 数据：4 个样本的二分类任务
xs = [[2,3,-1], [3,-1,0.5], [0.5,1,1], [1,1,-1]]
ys = [1.0, -1.0, -1.0, 1.0]

# 模型：3 输入 → 4 隐藏 → 4 隐藏 → 1 输出
n = MLP(3, [4, 4, 1])

for k in range(20):
    # 1. 前向传播：计算预测值和损失
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # 2. 梯度归零（必须！否则梯度会一直累加）
    for p in n.parameters():
        p.grad = 0.0

    # 3. 反向传播
    loss.backward()

    # 4. 梯度下降更新参数
    for p in n.parameters():
        p.data += -0.05 * p.grad
```

训练 20 轮后 loss 从 7.94 降到 0.033，预测值接近目标值 [1, -1, -1, 1]。

关键细节：
- **每轮必须手动归零梯度**：PyTorch 用的是 `optimizer.zero_grad()`，这里原理一样
- **学习率 0.05**：沿梯度的**反方向**更新（梯度指向 loss 上升最快的方向，所以取负号）
- **MSE 损失**：`(yout - ygt)²`，适用于回归/二分类
