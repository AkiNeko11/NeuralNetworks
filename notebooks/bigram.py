import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters 超参数
batch_size = 32     # 一次并行处理多少条独立的语句序列(也就是批次的大小)
block_size = 8      # 每条语句序列的最大长度(也就是上下文窗口的大小)
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table 每一个token直接从查找表中读取下一个token的logits
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets = None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)    #(B,T,C)  C是vocab_size  

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
            # get the predictions   得到预测
            logits, loss = self(idx)
            # focus only on the last time step   只关注最后一个时间步的logits
            logits = logits[:, -1, :]   # becomes (B, C) logits是(B,T,C)，但是我们只需要关注最后一个维度，所以变成(B,C)
            # apply softmax to get probabilities   应用softmax得到概率
            probs = F.softmax(logits, dim=-1)   # (B, C)
            # sample from the distribution   从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)  从概率分布中采样，得到下一个token的索引
            # append sampled index to the running sequence   将采样的索引附加到运行序列中
            idx = torch.cat((idx, idx_next), dim=1)  #(B, T+1)  沿着第1维度，也就是时间维度，继续拼接idx

        return idx
    
model = BigramLanguageModel(vocab_size)
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