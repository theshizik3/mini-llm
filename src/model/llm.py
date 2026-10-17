from torch import nn
from src.model.block import Block


class LLM(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, n_heads, n_kv_heads, n_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, n_kv_heads, seq_len) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, idx):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)
