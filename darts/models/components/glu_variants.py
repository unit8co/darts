import torch
from torch import nn

from darts.models.components.feed_forward import FeedForward

GLU_FFN = ["GLU", "Bilinear", "ReGLU", "GEGLU", "SwiGLU", "ReLU", "GELU"]


# GLU Variants Improve Transformer
# Shazeer, Noam, "GLU Variants Improve Transformer", 2020. arVix https://arxiv.org/abs/2002.05202
class GLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = FeedForward(
            d_model, d_ff, dropout, nn.Sigmoid(), True, False, False, False
        )

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


class Bilinear(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = FeedForward(
            d_model, d_ff, dropout, nn.Identity(), True, False, False, False
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


class GEGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = FeedForward(
            d_model, d_ff, dropout, nn.GELU(), True, False, False, False
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


class ReLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = FeedForward(d_model, d_ff, dropout, nn.ReLU())

    def forward(self, x: torch.Tensor):
        return self.ffn(x)


class GELU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = FeedForward(d_model, d_ff, dropout, nn.GELU())

    def forward(self, x: torch.Tensor):
        return self.ffn(x)
