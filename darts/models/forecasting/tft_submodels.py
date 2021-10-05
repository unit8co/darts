"""
Implementation of ``nn.Modules`` for temporal fusion transformer.
"""
from abc import ABC, abstractmethod
import math
from typing import Dict, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn

# # # 

"""
Implementations of flexible GRU and LSTM that can handle sequences of length 0.
"""

HiddenState = Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class RNN(ABC, nn.RNNBase):
    """
    Base class flexible RNNs.

    Forward function can handle sequences of length 0.
    """

    @abstractmethod
    def handle_no_encoding(
        self, hidden_state: HiddenState, no_encoding: torch.BoolTensor, initial_hidden_state: HiddenState
    ) -> HiddenState:
        """
        Mask the hidden_state where there is no encoding.

        Args:
            hidden_state (HiddenState): hidden state where some entries need replacement
            no_encoding (torch.BoolTensor): positions that need replacement
            initial_hidden_state (HiddenState): hidden state to use for replacement

        Returns:
            HiddenState: hidden state with propagated initial hidden state where appropriate
        """
        pass

    @abstractmethod
    def init_hidden_state(self, x: torch.Tensor) -> HiddenState:
        """
        Initialise a hidden_state.

        Args:
            x (torch.Tensor): network input

        Returns:
            HiddenState: default (zero-like) hidden state
        """
        pass

    @abstractmethod
    def repeat_interleave(self, hidden_state: HiddenState, n_samples: int) -> HiddenState:
        """
        Duplicate the hidden_state n_samples times.

        Args:
            hidden_state (HiddenState): hidden state to repeat
            n_samples (int): number of repetitions

        Returns:
            HiddenState: repeated hidden state
        """
        pass

    def forward(
        self,
        x: Union[rnn.PackedSequence, torch.Tensor],
        hx: HiddenState = None,
        lengths: torch.LongTensor = None,
        enforce_sorted: bool = True,
    ) -> Tuple[Union[rnn.PackedSequence, torch.Tensor], HiddenState]:
        """
        Forward function of rnn that allows zero-length sequences.

        Functions as normal for RNN. Only changes output if lengths are defined.

        Args:
            x (Union[rnn.PackedSequence, torch.Tensor]): input to RNN. either packed sequence or tensor of
                padded sequences
            hx (HiddenState, optional): hidden state. Defaults to None.
            lengths (torch.LongTensor, optional): lengths of sequences. If not None, used to determine correct returned
                hidden state. Can contain zeros. Defaults to None.
            enforce_sorted (bool, optional): if lengths are passed, determines if RNN expects them to be sorted.
                Defaults to True.

        Returns:
            Tuple[Union[rnn.PackedSequence, torch.Tensor], HiddenState]: output and hidden state.
                Output is packed sequence if input has been a packed sequence.
        """
        if isinstance(x, rnn.PackedSequence) or lengths is None:
            assert lengths is None, "cannot combine x of type PackedSequence with lengths argument"
            return super().forward(x, hx=hx)
        else:
            min_length = lengths.min()
            max_length = lengths.max()
            assert min_length >= 0, "sequence lengths must be great equals 0"

            if max_length == 0:
                hidden_state = self.init_hidden_state(x)
                if self.batch_first:
                    out = torch.zeros(lengths.size(0), x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
                else:
                    out = torch.zeros(x.size(0), lengths.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
                return out, hidden_state
            else:
                pack_lengths = lengths.where(lengths > 0, torch.ones_like(lengths))
                packed_out, hidden_state = super().forward(
                    rnn.pack_padded_sequence(
                        x, pack_lengths.cpu(), enforce_sorted=enforce_sorted, batch_first=self.batch_first
                    ),
                    hx=hx,
                )
                # replace hidden cell with initial input if encoder_length is zero to determine correct initial state
                if min_length == 0:
                    no_encoding = (lengths == 0)[
                        None, :, None
                    ]  # shape: n_layers * n_directions x batch_size x hidden_size
                    if hx is None:
                        initial_hidden_state = self.init_hidden_state(x)
                    else:
                        initial_hidden_state = hx
                    # propagate initial hidden state when sequence length was 0
                    hidden_state = self.handle_no_encoding(hidden_state, no_encoding, initial_hidden_state)

                # return unpacked sequence
                out, _ = rnn.pad_packed_sequence(packed_out, batch_first=self.batch_first)
                return out, hidden_state


class LSTM(RNN, nn.LSTM):
    """LSTM that can handle zero-length sequences"""

    def handle_no_encoding(
        self, hidden_state: HiddenState, no_encoding: torch.BoolTensor, initial_hidden_state: HiddenState
    ) -> HiddenState:
        hidden, cell = hidden_state
        hidden = hidden.masked_scatter(no_encoding, initial_hidden_state[0])
        cell = cell.masked_scatter(no_encoding, initial_hidden_state[0])
        return hidden, cell

    def init_hidden_state(self, x: torch.Tensor) -> HiddenState:
        num_directions = 2 if self.bidirectional else 1
        if self.batch_first:
            batch_size = x.size(0)
        else:
            batch_size = x.size(1)
        hidden = torch.zeros(
            (self.num_layers * num_directions, batch_size, self.hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
        cell = torch.zeros(
            (self.num_layers * num_directions, batch_size, self.hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
        return hidden, cell

    def repeat_interleave(self, hidden_state: HiddenState, n_samples: int) -> HiddenState:
        hidden, cell = hidden_state
        hidden = hidden.repeat_interleave(n_samples, 1)
        cell = cell.repeat_interleave(n_samples, 1)
        return hidden, cell


class MultiEmbedding(nn.Module):
    def __init__(
        self,
        embedding_sizes: Dict[str, Tuple[int, int]],
        categorical_groups: Dict[str, List[str]],
        embedding_paddings: List[str],
        x_categoricals: List[str],
        max_embedding_size: int = None,
    ):
        super().__init__()
        self.embedding_sizes = embedding_sizes
        self.categorical_groups = categorical_groups
        self.embedding_paddings = embedding_paddings
        self.max_embedding_size = max_embedding_size
        self.x_categoricals = x_categoricals

        self.init_embeddings()

    def init_embeddings(self):
        self.embeddings = nn.ModuleDict()
        for name in self.embedding_sizes.keys():
            embedding_size = self.embedding_sizes[name][1]
            if self.max_embedding_size is not None:
                embedding_size = min(embedding_size, self.max_embedding_size)
            # convert to list to become mutable
            self.embedding_sizes[name] = list(self.embedding_sizes[name])
            self.embedding_sizes[name][1] = embedding_size
            if name in self.categorical_groups:  # embedding bag if related embeddings
                self.embeddings[name] = TimeDistributedEmbeddingBag(
                    self.embedding_sizes[name][0], embedding_size, mode="sum", batch_first=True
                )
            else:
                if name in self.embedding_paddings:
                    padding_idx = 0
                else:
                    padding_idx = None
                self.embeddings[name] = nn.Embedding(
                    self.embedding_sizes[name][0],
                    embedding_size,
                    padding_idx=padding_idx,
                )

    def names(self):
        return list(self.keys())

    def items(self):
        return self.embeddings.items()

    def keys(self):
        return self.embeddings.keys()

    def values(self):
        return self.embeddings.values()

    def __getitem__(self, name: str):
        return self.embeddings[name]

    def forward(self, x):
        input_vectors = {}
        for name, emb in self.embeddings.items():
            if name in self.categorical_groups:
                input_vectors[name] = emb(
                    x[
                        ...,
                        [self.x_categoricals.index(cat_name) for cat_name in self.categorical_groups[name]],
                    ]
                )
            else:
                input_vectors[name] = emb(x[..., self.x_categoricals.index(name)])
        return input_vectors


class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool = False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class TimeDistributedInterpolation(nn.Module):
    """interpolates input size to output size
    TODO check this
    """
    def __init__(self, output_size: int, batch_first: bool = False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.output_size, mode="linear", align_corners=True).squeeze(1)
        return upsampled

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self,
                 input_size: int,
                 hidden_size: int = None,
                 dropout: float = None):

        super().__init__()

        self.dropout = nn.Dropout(dropout) if dropout is not None else dropout
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


class ResampleNorm(nn.Module):
    """Resamples an input to an output size
    TODO: why??
    I think this was added by pytorch-forecasting -> read model description
    """

    def __init__(self,
                 input_size: int,
                 output_size: int = None):

        super().__init__()

        self.input_size = input_size
        self.output_size = output_size or input_size

        if input_size != output_size:
            self.resample = TimeDistributedInterpolation(output_size, batch_first=True)

        self.mask = nn.Parameter(torch.zeros(output_size, dtype=torch.float))
        self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)

        x = x * self.gate(self.mask) * 2.0

        output = self.norm(x)
        return output


class AddNorm(nn.Module):
    def __init__(self, input_size: int, skip_size: int = None):
        super().__init__()

        self.input_size = input_size
        self.skip_size = skip_size or input_size

        if self.input_size != self.skip_size:
            self.resample = TimeDistributedInterpolation(self.input_size, batch_first=True)

        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.input_size != self.skip_size:
            skip = self.resample(skip)

        output = self.norm(x + skip)
        return output


class GateAddNorm(nn.Module):
    """Equation (2)"""
    def __init__(self,
                 input_size: int,
                 hidden_size: int = None,
                 skip_size: int = None,
                 dropout: float = None):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout

        self.glu = GatedLinearUnit(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = AddNorm(self.hidden_size, skip_size=self.skip_size)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class GatedResidualNetwork(nn.Module):
    """Top right graph in Figure 2 and formulas (2) -- (5)

    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 dropout: float = 0.1,
                 context_size: int = None):

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        # convert raw input into output size for residuals
        if input_size != output_size:
            self.resample_norm = ResampleNorm(input_size, output_size)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()

        if context_size is not None:
            # according to equation (4), no context bias
            self.context = nn.Linear(context_size, hidden_size, bias=False)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # TODO: what does this do?
        self.init_weights()

        self.gate_add_norm = GateAddNorm(
            input_size=hidden_size,
            skip_size=output_size,
            hidden_size=output_size,
            dropout=dropout,
        )

    def init_weights(self):
        # TODO: what is this?
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size:
            residual = self.resample_norm(residual)

        # TODO: why not both a and c into the same fc (dense layer)?
        # -> paper equation (4) supports this
        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_add_norm(x, residual)
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        input_embedding_flags: Dict[str, bool] = {},
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, GatedResidualNetwork] = {},
        prescalers: torch.nn.ModuleDict = {},
    ):
        """
        Calcualte weights for ``num_inputs`` variables  which are each of size ``input_size``
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.input_embedding_flags = input_embedding_flags
        self.dropout = dropout
        self.context_size = context_size

        if self.num_inputs > 1:
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(
                    input_size=self.input_size_total,
                    hidden_size=min(self.hidden_size, self.num_inputs),
                    output_size=self.num_inputs,
                    dropout=self.dropout,
                    context_size=self.context_size
                )
            else:
                self.flattened_grn = GatedResidualNetwork(
                    input_size=self.input_size_total,
                    hidden_size=min(self.hidden_size, self.num_inputs),
                    output_size=self.num_inputs,
                    dropout=self.dropout,
                )

        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            elif self.input_embedding_flags.get(name, False):
                self.single_variable_grns[name] = ResampleNorm(input_size, self.hidden_size)
            else:
                self.single_variable_grns[name] = GatedResidualNetwork(
                    input_size=input_size,
                    hidden_size=min(input_size, self.hidden_size),
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
        return sum(size if name in self.input_embedding_flags else size for name, size in self.input_sizes.items())

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, x: Dict[str, torch.Tensor], context: torch.Tensor = None):
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
            outputs = self.single_variable_grns[name](variable_embedding)  # fast forward if only one variable
            if outputs.ndim == 3:  # -> batch size, time, hidden size, n_variables
                sparse_weights = torch.ones(outputs.size(0), outputs.size(1), 1, 1, device=outputs.device)  #
            else:  # ndim == 2 -> batch size, hidden size, n_variables
                sparse_weights = torch.ones(outputs.size(0), 1, 1, device=outputs.device)
        return outputs, sparse_weights


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        assert d_model % 2 == 0, "model dimension has to be multiple of 2 (encode sin(pos) and cos(pos))"
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(0)
            pe = self.pe[:, :seq_len].view(seq_len, 1, self.d_model)
            x = x + pe
            return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
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


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
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