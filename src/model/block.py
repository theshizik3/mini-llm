from torch import nn
from src.model.attention import Attention
from src.model.ffn import FFN


class Block(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, max_len):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, n_kv_heads, max_len)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = FFN(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
