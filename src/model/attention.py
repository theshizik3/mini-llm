from torch import nn
from src.model.rope import RoPE

import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, max_len):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        self.rope = RoPE(self.head_dim, max_len=max_len)

    def forward(self, x):
        B, T, _ = x.shape

        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)

        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.out(y)
