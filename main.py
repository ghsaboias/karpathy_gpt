import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# --------------

torch.manual_seed(1337)

# GPT: Generatively Pre-trained Transformer
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # take string, output list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # take list of ints, output string

# train and validation splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# visualize some pairs of input and target
# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target is {target}")

xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)
# print('----')

# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()} the target is {target}")
# --------------------------------------------

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # [B, T, C]
        q = self.query(x) # [B, T, C]
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # [B, T, C] @ [B, C, T] --> [B, T, T]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # [B, T, T]
        wei = F.softmax(wei, dim=-1) # [B, T, T]
        # perform weighted aggregation of values
        v = self.value(x) # [B, T, C]
        out = wei @ v # [B, T, T] @ [B, T, C] --> [B, T, C]
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
    
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.net(x)
        
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # positional embeddings
        self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-dim self-attention
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # language modeling head

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_embd = self.token_embedding_table(idx) # (B, T, C): batch, time, channel
        pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_embd + pos_embd # (B, T, C); right-aligned, add dimension of 1 to the left, broadcast through batch dim
        x = self.sa_heads(x) # apply one head of self-attention; [B, T, C]
        x = self.ffwd(x) # apply feed-forward layer; [B, T, C]
        logits = self.lm_head(x) # (B, T, V): batch, time, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # to comply with pytorch
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T] array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens, due to positional embeddings size
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # focus only on last time step
            logits = logits[:, -1, :] # becomes [B, C]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # [B, C]
            # sample from distro
            idx_next = torch.multinomial(probs, num_samples=1) # [B, 1]
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # [B, T+1]
        return idx

model = BigramLanguageModel()
logits, loss = model(xb, yb)
idx = torch.zeros((1, 1), dtype=torch.long)

# create pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# training loop
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['eval']:.4f}")
    # sample batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(idx, max_new_tokens=200)[0].tolist()))







