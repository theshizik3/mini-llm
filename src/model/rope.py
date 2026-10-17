import torch

from torch import nn


class RoPE(nn.Module):
    def __init__(self, head_dim, max_len=32768, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(max_len).float()
        angles = torch.outer(pos, inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x):
        seq_len = x.size(-2)
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        return torch.stack([rot1, rot2], dim=-1).flatten(-2)
