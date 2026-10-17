from torch import nn

import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, d_model, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = int(8 * d_model / 3)
            hidden = 256 * ((hidden + 255) // 256)

        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
