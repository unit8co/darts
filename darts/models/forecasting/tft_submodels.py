"""
Implementation of ``nn.Modules`` for Temporal Fusion Transformer from PyTorch-Forecasting:
https://github.com/jdb78/pytorch-forecasting

PyTorch Forecasting v0.9.1 License from https://github.com/jdb78/pytorch-forecasting/blob/master/LICENSE, accessed
on Wed, November 3, 2021:
'THE MIT License

Copyright 2020 Jan Beitner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
'
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.logging import get_logger
from darts.utils.torch import MonteCarloDropout

logger = get_logger(__name__)

HiddenState = Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]


def get_embedding_size(n: int, max_size: int = 100) -> int:
    """
    Determine empirically good embedding sizes (formula taken from fastai).
    Args:
        n (int): number of classes
        max_size (int, optional): maximum embedding size. Defaults to 100.
    Returns:
        int: embedding size
    """
    if n > 2:
        return min(round(1.6 * n**0.56), max_size)
    else:
        return 1


class _TimeDistributedEmbeddingBag(nn.EmbeddingBag):
    def __init__(self, *args, batch_first: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return super().forward(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)

        y = super().forward(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.size(0), -1, y.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class _MultiEmbedding(nn.Module):
    def __init__(
        self,
        embedding_sizes: dict[str, tuple[int, int]],
        variable_names: list[str],
    ):
        """Embedding layer for categorical variables including groups of categorical variables.
        Enabled for static and dynamic categories (i.e. 3 dimensions for batch x time x categories).

        Parameters
        ----------
        embedding_sizes
            dictionary of embedding sizes, e.g. ``{'cat1': (10, 3)}``
            indicates that the first categorical variable has 10 unique values which are mapped to 3 embedding
            dimensions. Use :py:func:`~pytorch_forecasting.utils.get_embedding_size` to automatically obtain
            reasonable embedding sizes depending on the number of categories.
        variable_names
            list of categorical variable names to ensure ordered iterations.
        """
        super().__init__()
        self.embedding_sizes = embedding_sizes
        self.variable_names = variable_names

        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(*embedding_sizes[name]) for name in variable_names
        })

    @property
    def input_size(self) -> int:
        return len(self.variable_names)

    @property
    def output_size(self) -> Union[dict[str, int], int]:
        return {name: sizes[1] for name, sizes in self.embedding_sizes.items()}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x
            input tensor of shape batch x (optional) time x categoricals in the order of ``variable_names``.

        Returns
        -------
        dict
            dictionary of category names to embeddings of shape batch x (optional) time x embedding_size if
            ``embedding_size`` is given as dictionary.
        """
        return {
            name: self.embeddings[name](x[..., i])
            for i, name in enumerate(self.variable_names)
        }


class _TimeDistributedInterpolation(nn.Module):
    def __init__(
        self, output_size: int, batch_first: bool = False, trainable: bool = False
    ):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(
            x.unsqueeze(1), self.output_size, mode="linear", align_corners=True
        ).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)

        y = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.size(0), -1, y.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class _GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None):
        super().__init__()

        if dropout is not None:
            self.dropout = MonteCarloDropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x


class _ResampleNorm(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = None,
        trainable_add: bool = True,
        norm=nn.LayerNorm,
    ):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size

        if self.input_size != self.output_size:
            self.resample = _TimeDistributedInterpolation(
                self.output_size, batch_first=True, trainable=False
            )

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = norm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0

        output = self.norm(x)
        return output


class _AddNorm(nn.Module):
    def __init__(
        self,
        input_size: int,
        skip_size: int = None,
        trainable_add: bool = True,
        norm=nn.LayerNorm,
    ):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size

        if self.input_size != self.skip_size:
            self.resample = _TimeDistributedInterpolation(
                self.input_size, batch_first=True, trainable=False
            )

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = norm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.input_size != self.skip_size:
            skip = self.resample(skip)

        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0

        output = self.norm(x + skip)
        return output


class _GateAddNorm(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        skip_size: int = None,
        trainable_add: bool = False,
        dropout: float = None,
        layer_norm: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout

        self.glu = _GatedLinearUnit(
            self.input_size, hidden_size=self.hidden_size, dropout=self.dropout
        )
        self.add_norm = _AddNorm(
            self.hidden_size,
            skip_size=self.skip_size,
            trainable_add=trainable_add,
            norm=layer_norm,
        )

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class _GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = False,
        layer_norm: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resample_norm = _ResampleNorm(
                residual_size, self.output_size, norm=layer_norm
            )

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()

        self.gate_norm = _GateAddNorm(
            input_size=self.hidden_size,
            skip_size=self.output_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(
                    p, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x


class _VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: dict[str, int],
        hidden_size: int,
        input_embedding_flags: Optional[dict[str, bool]] = None,
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Optional[dict[str, _GatedResidualNetwork]] = None,
        prescalers: Optional[dict[str, nn.Linear]] = None,
        layer_norm: nn.Module = nn.LayerNorm,
    ):
        """
        Calculate weights for ``num_inputs`` variables  which are each of size ``input_size``
        """
        super().__init__()

        input_embedding_flags = (
            input_embedding_flags if input_embedding_flags is not None else {}
        )
        single_variable_grns = (
            single_variable_grns if single_variable_grns is not None else {}
        )
        prescalers = prescalers if prescalers is not None else {}

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.input_embedding_flags = input_embedding_flags
        self.dropout = dropout
        self.context_size = context_size

        if self.num_inputs > 1:
            if self.context_size is not None:
                self.flattened_grn = _GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    self.context_size,
                    residual=False,
                )
            else:
                self.flattened_grn = _GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    residual=False,
                )

        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            elif self.input_embedding_flags.get(name, False):
                self.single_variable_grns[name] = _ResampleNorm(
                    input_size,
                    self.hidden_size,
                    norm=layer_norm,
                )
            else:
                self.single_variable_grns[name] = _GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    output_size=self.hidden_size,
                    dropout=self.dropout,
                )
            if name in prescalers:  # reals need to be first scaled up
                self.prescalers[name] = prescalers[name]
            elif not self.input_embedding_flags.get(name, False):
                self.prescalers[name] = nn.Linear(1, input_size)

        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(
            size if name in self.input_embedding_flags else size
            for name, size in self.input_sizes.items()
        )

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, x: dict[str, torch.Tensor], context: torch.Tensor = None):
        if self.num_inputs > 1:
            # transform single variables
            var_outputs = []
            weight_inputs = []
            for name in self.input_sizes.keys():
                # select embedding belonging to a single input
                variable_embedding = x[name]
                if name in self.prescalers:
                    variable_embedding = self.prescalers[name](variable_embedding)
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[name](variable_embedding))
            var_outputs = torch.stack(var_outputs, dim=-1)

            # calculate variable weights
            flat_embedding = torch.cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(flat_embedding, context)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

            outputs = var_outputs * sparse_weights
            outputs = outputs.sum(dim=-1)
        else:  # for one input, do not perform variable selection but just encoding
            name = next(iter(self.single_variable_grns.keys()))
            variable_embedding = x[name]
            if name in self.prescalers:
                variable_embedding = self.prescalers[name](variable_embedding)
            outputs = self.single_variable_grns[name](
                variable_embedding
            )  # fast forward if only one variable
            if outputs.ndim == 3:  # -> batch size, time, hidden size, n_variables
                sparse_weights = torch.ones(
                    outputs.size(0), outputs.size(1), 1, 1, device=outputs.device
                )  #
            else:  # ndim == 2 -> batch size, hidden size, n_variables
                sparse_weights = torch.ones(
                    outputs.size(0), 1, 1, device=outputs.device
                )
        return outputs, sparse_weights


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super().__init__()
        if dropout is not None:
            self.dropout = MonteCarloDropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class _InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = MonteCarloDropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList([
            nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)
        ])
        self.k_layers = nn.ModuleList([
            nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)
        ])
        self.attention = _ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []
        vs = self.v_layer(v)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn
