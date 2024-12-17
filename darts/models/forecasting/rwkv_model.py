########################################################################################################
# Inspired by the RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from darts.models.forecasting.pl_forecasting_module import PLPastCovariatesModule
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel

logger = logging.getLogger(__name__)


class ChannelMixModule(nn.Module):
    def __init__(self, input_dim, layer_id, n_layers):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.receptance = nn.Linear(input_dim, input_dim)

        self.epsilon_key = _channel_mixing_epsilon(layer_id, n_layers)
        self.epsilon_receptance = _channel_mixing_epsilon(layer_id, n_layers)

        self.X_prev = None  # shape (N, 1, C)

    def forward(self, x):
        N, _, C = x.shape
        out = (
            self._recurrent_forward(x)
            if self._is_recurrent()
            else self._parallel_forward(x)
        )
        self.X_prev = x[:, -1:, :]  # token shift, shape (N, 1, C)
        return out

    def _is_recurrent(self):
        return self.X_prev is not None

    def _parallel_forward(self, x):
        R = self.receptance(_time_interpolation(x, self.epsilon_receptance))
        K = self.key(_time_interpolation(x, self.epsilon_key))
        V = self.value(torch.square(F.relu(K)))
        return F.sigmoid(R) * V

    def _recurrent_forward(self, x):
        x = torch.cat(
            [self.X_prev, x[:, -1:, :]], dim=1
        )  # token shift for interpolation
        return self._parallel_forward(x)[:, -1:, :]

    def clear_state(self):
        self.X_prev = None


class TimeMixModule(nn.Module):
    def __init__(
        self,
        input_dim,
        attention_dim,
        n_head,
        layer_id,
        n_layers,
        device,
    ):
        super().__init__()
        assert attention_dim % n_head == 0

        self.n_head = n_head
        self.head_size = attention_dim // n_head

        self.receptance = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.output = nn.Linear(attention_dim, input_dim)

        self.epsilon_receptance = _time_mix_epsilon(layer_id, n_layers, False) / 2
        self.epsilon_key = _time_mix_epsilon(layer_id, n_layers, True)
        self.epsilon_value = _time_mix_epsilon(layer_id, n_layers, False)

        self.time_decay = nn.Parameter(
            torch.full([attention_dim], 0.1).to(device)
        )  # TODO add better initialization
        self.U = nn.Parameter(
            torch.full([attention_dim], 0.1).to(device)
        )  # TODO add better initialization

        self.X_prev = None
        self.A = None
        self.B = None

    def forward(self, x):
        if self._is_recurrent():
            return self._recurrent_forward(x)
        return self._parallel_forward(x)

    def _is_recurrent(self):
        return self.X_prev is not None

    def _recurrent_forward(self, x):
        N, _, C = x.shape
        attention_shape = N, -1, self.n_head, self.head_size

        x = torch.cat([self.X_prev, x[:, -1:, :]], dim=1)

        R = self.receptance(_time_interpolation(x, self.epsilon_receptance))[
            :,
            1:,
        ].view(attention_shape)
        K = self.key(_time_interpolation(x, self.epsilon_key))[
            :,
            1:,
        ].view(attention_shape)
        V = self.value(_time_interpolation(x, self.epsilon_value))[
            :,
            1:,
        ].view(attention_shape)

        U = self.U.reshape(1, 1, self.n_head, self.head_size).repeat(N, 1, 1, 1)

        A = self.A * torch.exp(-self.time_decay) + torch.exp(U + K) * V
        B = self.B * torch.exp(-self.time_decay) + torch.exp(U + K)

        WKV = A / B

        self.X_prev = x[:, -1:, :]
        self.A = self.A * torch.exp(-self.time_decay) + torch.exp(K) * V
        self.B = self.B * torch.exp(-self.time_decay) + torch.exp(K)

        return self.output(F.sigmoid(R) * WKV).view(N, 1, -1)

    def _parallel_forward(self, x):
        N, T, C = x.shape
        attention_shape = N, -1, self.n_head, self.head_size
        device = x.get_device()

        R = self.receptance(_time_interpolation(x, self.epsilon_receptance)).view(
            attention_shape
        )
        K = self.key(_time_interpolation(x, self.epsilon_key)).view(attention_shape)
        V = self.value(_time_interpolation(x, self.epsilon_value)).view(attention_shape)

        # [T - 1; 0]  across T dim
        time_arrange = T - torch.arange(1, T + 1, device=device).reshape(
            1, T, 1, 1
        ).repeat(N, 1, self.n_head, self.head_size)
        W = -torch.relu(self.time_decay) * time_arrange.to("cuda")

        clamped_exp = torch.clamp(
            W + K, max=10, min=-20
        )  # TODO check switching to only clamping K

        past_attention = torch.cumsum(torch.exp(clamped_exp) * V, dim=1)
        past_energy = torch.cumsum(torch.exp(clamped_exp), dim=1)

        time_shift_padding = nn.ZeroPad2d((0, 0, -1, 1))
        U = self.U.reshape(1, 1, self.n_head, self.head_size).repeat(N, T, 1, 1)

        clamped_u = torch.clamp(
            U + K, max=10, min=-20
        )  # TODO check switching to only clamping K

        A = time_shift_padding(past_attention) + torch.exp(clamped_u) * V
        B = time_shift_padding(past_energy) + torch.exp(clamped_u)

        WKV = A / B
        self.X_prev = x[:, -1:, :]
        self.A = past_attention[:, -1:, :]
        self.B = past_energy[:, -1:, :]

        return self.output(F.sigmoid(R) * WKV).view(N, -1, C)

    def clear_state(self):
        self.X_prev = None
        self.A = None
        self.B = None


class Block(nn.Module):
    def __init__(self, input_dim, n_attn, n_head, layer_id, n_layers, device):
        super().__init__()
        self.n_embed = input_dim

        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)

        self.attn = TimeMixModule(input_dim, n_attn, n_head, layer_id, n_layers, device)
        self.mlp = ChannelMixModule(input_dim, layer_id, n_layers)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x

    def clear_state(self):
        self.attn.clear_state()
        self.mlp.clear_state()


def _time_interpolation(x, epsilon):
    """
    :param x: shape(N, T, C)
    :param epsilon:
    :return:
    """
    curr_pad = nn.ZeroPad2d((0, 0, 1, -1))
    prev_pad = nn.ZeroPad2d((0, 0, -1, 1))

    return epsilon * curr_pad(x) + (1 - epsilon) * prev_pad(x)


def _channel_mixing_epsilon(layer_id, n_layers):
    return math.pow(math.e, -layer_id / n_layers)


def _time_mix_epsilon(layer_id, n_layers, is_key):
    eps = _channel_mixing_epsilon(layer_id, n_layers)
    if is_key:
        return eps + (0.3 * layer_id / (n_layers - 1)) if n_layers > 1 else 0.3 * eps
    return eps


class _RWKVModule(PLPastCovariatesModule):
    def __init__(self, input_dim, output_dim, num_layers, n_attn, n_head, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layer = num_layers
        self.n_attn = n_attn
        self.n_head = n_head

        self.sequential = nn.Sequential(
            *[
                Block(input_dim, n_attn, n_head, layer_id, num_layers, self.device)
                for layer_id in range(num_layers)
            ],
            nn.Linear(input_dim, output_dim, bias=False),
        )

    def forward(self, x_in):
        x, _ = x_in

        for m in self.sequential.children():
            if isinstance(m, Block):
                m.clear_state()

        y = [self.sequential(x)[:, -1:, :]]
        for i in range(self.output_chunk_length - 1):
            pred_step = self.sequential(y[-1])
            y.append(pred_step)
        y = torch.cat(y, dim=1)

        y = y.view(y.shape[0], self.output_chunk_length, self.input_dim, 1)[
            :, :, : self.output_dim, :
        ]
        return y


class RWKVModel(PastCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        num_layers: int,
        n_attn: int,
        n_head: int,
        **kwargs
    ):
        """RWKV Model
        RWKV model.

        Parameters
        ----------
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))
        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        self.num_layers = num_layers
        self.n_attn = n_attn
        self.n_head = n_head

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (
            train_sample[1].shape[1] if train_sample[1] is not None else 0
        )
        output_dim = train_sample[-1].shape[1]

        # TODO add likelihood support in the future
        return _RWKVModule(
            input_dim,
            output_dim,
            self.num_layers,
            self.n_attn,
            self.n_head,
            **self.pl_module_params,
        )

    @property
    def supports_multivariate(self) -> bool:
        return True
