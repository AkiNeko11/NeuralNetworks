import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters 超参数
batch_size = 64     # 一次并行处理多少条独立的语句序列(也就是批次的大小)
block_size = 256      # 每条语句序列的最大长度(也就是上下文窗口的大小)
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
#--------

torch.manual_seed(1337)  # 设置随机种子，保证每次运行结果相同

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers     编码器，把字符串转化为整数列表
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string    解码器，把整数列表转化为字符串

# train and test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 前90%作为训练集，后10%作为验证集
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # 生成一个小批量数据
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """""one head of self-attention"""""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C) -> (B,T,head_size)
        q = self.query(x) # (B,T,C) -> (B,T,head_size)
        # complete attention scores ("affinities") 计算注意力分数（“亲和力”）
        wei = q @ k.transpose(-2, -1) * C**-0.5   # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)  这里是点积注意力，C**-0.5是缩放因子
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))   # (B,T,T)  这里是掩码操作，把上三角部分的值设为负无穷，这样softmax之后就会变成0
        wei = F.softmax(wei, dim=-1)   # (B,T,T)  这里是softmax操作，把注意力分数变成概率分布
        # perform the weighted aggregation of the values  执行值的加权聚合
        v = self.value(x)   # (B,T,C) -> (B,T,head_size)
        out = wei @ v   # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)  这里是加权求和，得到每个位置的输出
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 乘上 4 是因为在原始的 Transformer 论文中，前馈神经网络的隐藏层维度通常设置为输入维度的 4 倍，这样可以增加模型的表达能力
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # this is the projection layer that maps the output back to the original embedding dimension
            nn.Dropout(dropout),    # dropout是在residual connection返回到原始embedding维度之前，随机丢弃一些神经元的输出，以防止过拟合
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """Transformer block: coummunication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)     # 添加layer norm层

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table 每一个token直接从查找表中读取下一个token的logits
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)   # 这里从vsize*vsize的表变成了vsize*n_embd的表，n_embd是embedding的维度
        self.position_embedding_table = nn.Embedding(block_size, n_embd)   # 这里是一个位置embedding表，block_size是最大长度，n_embd是embedding的维度
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)   # 最后的layer norm层
        self.lm_head = nn.Linear(n_embd, vocab_size)   # 这里是一个线性层，把embedding的维度变成vocab_size的维度，也就是logits的维度
    
    def forward(self, idx, targets = None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)    # 因此这里也不再是logits，而是embedding，形状是(B,T,C(n_embd))
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))   # 这里是位置embedding，形状是(T,C(n_embd))
        x = tok_emb + pos_emb   # 这里是把token embedding和position embedding相加，形状是(B,T,C(n_embd))
        # 这样的话对于每一个x，它不仅包含了当前token的信息，还包含了它在序列中的位置信息，这样模型就可以区分同一个token在不同位置的含义
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)   # logits的形状是(B,T,vocab_size)，也就是每个token对应一个vocab_size维度的logits

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape          #这里需要注意，pytorch的cross_entropy函数要求输入的logits是(B,C,T)的形式，所以我们需要把logits(B,T,C)改变
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context idx是形如(B,T)的当前上下文中的索引数组
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens   将idx裁剪到最后的block_size个token
            idx_cond = idx[:, -block_size:]   # (B, block_size)  这里是为了保证输入的长度不超过block_size
            # get the predictions   得到预测
            logits, loss = self(idx_cond)
            # focus only on the last time step   只关注最后一个时间步的logits
            logits = logits[:, -1, :]   # becomes (B, C) logits是(B,T,C)，但是我们只需要关注最后一个维度，所以变成(B,C)
            # apply softmax to get probabilities   应用softmax得到概率
            probs = F.softmax(logits, dim=-1)   # (B, C)
            # sample from the distribution   从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)  从概率分布中采样，得到下一个token的索引
            # append sampled index to the running sequence   将采样的索引附加到运行序列中
            idx = torch.cat((idx, idx_next), dim=1)  #(B, T+1)  沿着第1维度，也就是时间维度，继续拼接idx

        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# 制作一个pytorch的optimizer，使用AdamW优化器
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 :
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))