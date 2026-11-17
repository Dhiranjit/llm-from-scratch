import torch
import torch.nn.functional as F
from torch import nn




class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head== 0
        self.head_size = n_embd // n_head
        self.n_head = n_head

        self.qkv = nn.Linear(n_embd, 3*n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)).reshape(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1) # (B, T, 3 * C) -> (B, T, C)

        q = q.reshape(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        k = k.reshape(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)

        wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5) # (B, nh, T, T)
        wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.mha   = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class CausalSelfAttentionModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layers, block_size, dropout):
        super().__init__()
        self.block_size         = block_size
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layers)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss


    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs     = F.softmax(logits[:, -1, :], dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx = torch.concat((idx, idx_next), dim=-1) 
        return idx