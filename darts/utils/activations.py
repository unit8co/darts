import torch
from torch import nn

from darts.utils.feed_forward import FeedForward


# GLU Variants Improve Transformer https://arxiv.org/pdf/2002.05202.pdf
# GLU, Bliniear, and GELU can be found in the torch.nn module
class GEGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = FeedForward(
            d_model, d_ff, dropout, nn.GELU(), True, False, False, False
        )

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


class ReGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = FeedForward(
            d_model, d_ff, dropout, nn.ReLU(), True, False, False, False
        )

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = FeedForward(
            d_model, d_ff, dropout, nn.SiLU(), True, False, False, False
        )

    def forward(self, x: torch.Tensor):
        return self.ffn(x)
