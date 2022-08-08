import torch
import torch.nn as nn

from darts.utils.torch import MonteCarloDropout


class CustomFeedForwardEncoderLayer(nn.TransformerEncoderLayer):
    """Overwrites the PyTorch TransformerEncoderLayer to use Darts' Position-wise Feed-Forward variants."""

    def __init__(self, ffn: nn.Module, dropout: float, *args, **kwargs):
        """
        Parameters
        ----------
        ffn
            One of Darts' Position-wise Feed-Forward Network variants from darts.models.components.glu_variants
        dropout
            Fraction of neurons affected by Dropout (default=0.1).
        args
            positional arguments from torch.nn.TransformerEncoderLayer.
        kwargs
            keyword arguments from torch.nn.TransformerEncoderLayer. `activation` will have no effect.
        """
        super().__init__(*args, **kwargs)
        self.ffn = ffn
        self.dropout = MonteCarloDropout(dropout)

    # overwrite the feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn(x)
        return self.dropout(x)


class CustomFeedForwardDecoderLayer(nn.TransformerDecoderLayer):
    """Overwrites the PyTorch TransformerDecoderLayer to use Darts' custom Position Wise Feed Forward Layers."""

    def __init__(self, ffn: nn.Module, dropout: float, *args, **kwargs):
        """
        Parameters
        ----------
        ffn
            One of Darts' Position-wise Feed-Forward Network variants from darts.models.components.glu_variants
        dropout
            Fraction of neurons affected by Dropout (default=0.1).
        args
            positional arguments from torch.nn.TransformerEncoderLayer.
        kwargs
            keyword arguments from torch.nn.TransformerEncoderLayer. `activation` will have no effect.
        """
        super().__init__(*args, **kwargs)
        self.ffn = ffn
        self.dropout = MonteCarloDropout(dropout)

    # overwrite the feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn(x)
        return self.dropout(x)
