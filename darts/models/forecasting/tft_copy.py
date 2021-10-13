"""
N-BEATS
-------
"""

from copy import copy
from typing import NewType, Union, List, Optional, Tuple, Dict, Sequence
from enum import Enum

from matplotlib import pyplot as plt
import numpy as np
from numpy.random import RandomState

import torch
from torch import nn

from darts import TimeSeries
from darts.utils.data import DualCovariatesShiftedDataset, TrainingDataset, MixedCovariatesShiftedDataset, MixedCovariatesSequentialDataset
from darts.utils.likelihood_models import Likelihood
from darts.utils.torch import random_method
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel, TorchParametricProbabilisticForecastingModel, DualCovariatesTorchModel
from darts.logging import get_logger, raise_log, raise_if_not

from darts.models.forecasting.tft_submodels_darts import (
    AddNorm,
    GateAddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
    LSTM,
    MultiEmbedding,
    QuantileLoss
)

logger = get_logger(__name__)

DARTS_IMPLEMENTATION = True

class _TFTModule(nn.Module):

    def __init__(self,
                 # input_dim: int,
                 # output_dim: int,
                 # input_chunk_length: int,
                 # output_chunk_length: int,
                 # generic_architecture: bool,
                 # num_stacks: int,
                 # num_blocks: int,
                 # num_layers: int,
                 # layer_widths: List[int],
                 # expansion_coefficient_dim: int,
                 # trend_polynomial_degree: int
                 input_dim: int,
                 output_dim: int,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 variables: Dict[str, Dict[str, List[str]]],
                 hidden_size: Union[int, List[int]] = 16,
                 lstm_layers: int = 1,
                 dropout: float = 0.1,
                 output_size: Union[int, List[int]] = 7,
                 loss_fn: nn.Module = QuantileLoss(),
                 attention_head_size: int = 4,
                 max_encoder_length: int = 10,
                 categorical_groups: Dict[str, List[str]] = {},
                 x_reals: List[str] = [],
                 x_categoricals: List[str] = [],
                 hidden_continuous_size: int = 8,
                 hidden_continuous_sizes: Dict[str, int] = {},
                 embedding_sizes: Dict[str, Tuple[int, int]] = {},
                 embedding_paddings: List[str] = [],
                 embedding_labels: Dict[str, np.ndarray] = {},
                 learning_rate: float = 1e-3,
                 log_interval: Union[int, float] = -1,
                 log_val_interval: Union[int, float] = None,
                 log_gradient_flow: bool = False,
                 reduce_on_plateau_patience: int = 1000,
                 monotone_constaints: Dict[str, int] = {},
                 share_single_variable_networks: bool = False,
                 logging_metrics: nn.ModuleList = None,
                 **kwargs
                 ):
        """ PyTorch module implementing the TFT architecture.

        """
        super(_TFTModule, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.variables = variables

        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.output_size = output_size
        self.loss_fn = loss_fn
        self.attention_head_size = attention_head_size
        self.max_encoder_length = max_encoder_length
        self.categorical_groups = categorical_groups
        self.x_reals = x_reals
        self.x_categoricals = x_categoricals
        self.hidden_continuous_size = hidden_continuous_size
        self.hidden_continuous_sizes = hidden_continuous_sizes
        self.embedding_sizes = embedding_sizes
        self.embedding_paddings = embedding_paddings
        self.embedding_labels = embedding_labels
        self.learning_rate = learning_rate
        self.log_interval = log_interval
        self.log_val_interval = log_val_interval
        self.log_gradient_flow = log_gradient_flow
        self.reduce_on_plateau_patience = reduce_on_plateau_patience
        self.monotone_constaints = monotone_constaints
        self.share_single_variable_networks = share_single_variable_networks
        self.logging_metrics = logging_metrics
        self.kwargs = kwargs
        self.n_targets = output_dim

        # # processing inputs
        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, self.hidden_continuous_sizes.get(name, self.hidden_continuous_size))
                for name in self.reals
            }
        )

        static_input_sizes = {
            name: self.hidden_continuous_sizes.get(name, self.hidden_continuous_size)
            for name in self.static_variables
        }

        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},  # this would be required for categorical inputs
            dropout=self.dropout,
            prescalers=self.prescalers,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.hidden_continuous_sizes.get(name, self.hidden_continuous_size)
            for name in self.encoder_variables
        }

        decoder_input_sizes = {
            name: self.hidden_continuous_sizes.get(name, self.hidden_continuous_size)
            for name in self.decoder_variables
        }

        # create single variable grns that are shared across decoder and encoder
        if self.share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    self.hidden_size,
                    self.dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, self.hidden_size),
                        self.hidden_size,
                        self.dropout,
                    )

        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},  # this would be required for categorical inputs
            dropout=self.dropout,
            context_size=self.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={} if not self.share_single_variable_networks else self.shared_single_variable_grns
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},  # this would be required for categorical inputs
            dropout=self.dropout,
            context_size=self.hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not self.share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # for hidden state of the lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            self.hidden_size, self.hidden_size, self.hidden_size, self.dropout
        )

        # lstm encoder (history) and decoder (future) for local processing
        self.lstm_encoder = LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # skip connection for lstm
        self.post_lstm_gate_encoder = GatedLinearUnit(self.hidden_size, dropout=self.dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        self.post_lstm_add_norm_encoder = AddNorm(self.hidden_size)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past LSTM
        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            context_size=self.hidden_size,
        )

        # attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.hidden_size, n_head=self.attention_head_size, dropout=self.dropout
        )
        self.post_attn_gate_norm = GateAddNorm(self.hidden_size, dropout=self.dropout)
        self.pos_wise_ff = GatedResidualNetwork(
            self.hidden_size, self.hidden_size, self.hidden_size, dropout=self.dropout
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(self.hidden_size, dropout=None)

        if self.n_targets > 1:  # if to run with multiple targets
            self.output_layer = nn.ModuleList(
                [nn.Linear(self.hidden_size, output_size) for output_size in self.output_size]
            )
        else:
            self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    @property
    def reals(self) -> List[str]:
        """List of all continuous variables in model"""
        return self.variables['model_config']['reals_input']

    @property
    def categoricals(self) -> List[str]:
        """List of all categorical variables in model"""
        # return list(
        #     dict.fromkeys(
        #         self.static_categoricals
        #         + self.time_varying_categoricals_encoder
        #         + self.time_varying_categoricals_decoder
        #     )
        # )
        raise NotImplementedError('TFT does not yet support categorical variables')

    @property
    def static_variables(self) -> List[str]:
        """List of all static variables in model"""
        # return self.static_categoricals + self.static_reals
        return self.variables['model_config']['static_input']

    @property
    def encoder_variables(self) -> List[str]:
        """List of all encoder variables in model (excluding static variables)"""
        # return self.time_varying_categoricals_encoder + self.time_varying_reals_encoder
        return self.variables['model_config']['time_varying_encoder_input']

    @property
    def decoder_variables(self) -> List[str]:
        """List of all decoder variables in model (excluding static variables)"""
        # return self.time_varying_categoricals_decoder + self.time_varying_reals_decoder
        return self.variables['model_config']['time_varying_decoder_input']

    def expand_static_context(self, context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(self,
                           encoder_lengths: torch.Tensor,
                           decoder_length: int,
                           device: str):
        """
        Returns causal mask to apply for self-attention layer.

        Args:
            self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=device)[:, None]
        # do not attend to steps to self or after prediction
        # todo: there is potential value in attending to future forecasts if they are made with knowledge currently
        #   available
        #   one possibility is here to use a second attention layer for future attention (assuming different effects
        #   matter in the future than the past)
        #   or alternatively using the same layer but allowing forward attention - i.e. only masking out non-available
        #   data and self
        decoder_mask = attend_step >= predict_step
        # do not attend to steps where data is padded
        encoder_mask = self.create_mask(encoder_lengths.max(), encoder_lengths)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(encoder_lengths.size(0), -1, -1),
            ),
            dim=2,
        )
        return mask

    @staticmethod
    def create_mask(size: int, lengths: torch.LongTensor, inverse: bool = False) -> torch.BoolTensor:
        """
        Create boolean masks of shape len(lenghts) x size.

        An entry at (i, j) is True if lengths[i] > j.

        Args:
            size (int): size of second dimension
            lengths (torch.LongTensor): tensor of lengths
            inverse (bool, optional): If true, boolean mask is inverted. Defaults to False.

        Returns:
            torch.BoolTensor: mask
        """

        if inverse:  # return where values are
            return torch.arange(size, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(-1)
        else:  # return where no values are
            return torch.arange(size, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(-1)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        past_target, past_covariates, historic_future_covariates, future_covariates = x
        # TODO: impelement static covariates
        static_covariates = None
        # data is of size (batch_size, input_length, input_size)
        x_cont_past = torch.cat(
            [tensor for tensor in [past_target,
                                   past_covariates,
                                   historic_future_covariates,
                                   static_covariates] if tensor is not None], dim=2
        )
        x_cont_future = torch.cat(
            [tensor for tensor in [future_covariates,
                                   static_covariates] if tensor is not None], dim=2
        )
        # encoder_lengths = x["encoder_lengths"]
        # decoder_lengths = x["decoder_lengths"]
        # x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
        # x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension

        # timesteps = x_cont.size(1)  # encode + decode length
        # max_encoder_length = int(encoder_lengths.max())

        # batch_size = x.shape[0]
        encoder_lengths = torch.tensor([self.input_chunk_length] * x_cont_past.shape[0], dtype=past_target.dtype)
        decoder_lengths = torch.tensor([self.output_chunk_length] * x_cont_future.shape[0], dtype=past_target.dtype)

        # TODO: if we ever have categoricals
        # x_cat = torch.empty(x.shape, dtype=x.dtype)
        # x_cont_known = torch.cat([historic_future_covariates, future_covariates])
        # x_cont
        # x_cont = torch.cat([t for t in x if t is not None], dim=1)

        timesteps = self.input_chunk_length + self.output_chunk_length
        max_encoder_length = self.input_chunk_length
        input_vectors_past = {
            name: x_cont_past[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.encoder_variables)
        }
        input_vectors_future = {
            name: x_cont_future[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.decoder_variables)
        }

        # Embedding and variable selection

        # print('CHECK IF INPUT VECTORS IS FOR POST/FUTURE OR IF STATIC ALWAYS THE SAME THIS HERE')
        if static_covariates is not None:
            # # static embeddings will be constant over entire batch
            # static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            # static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
            pass
        else:
            static_embedding = torch.zeros(
                (past_target.size(0), self.hidden_size), dtype=past_target.dtype, device=past_target.device
            )
            static_variable_selection = torch.zeros((past_target.size(0), 0), dtype=past_target.dtype, device=past_target.device)

        static_context_variable_selection = self.expand_static_context(
            context=self.static_context_variable_selection(static_embedding),
            timesteps=timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors_past[name] for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            x=embeddings_varying_encoder,
            context=static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors_future[name] for name in self.decoder_variables
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            x=embeddings_varying_decoder,
            context=static_context_variable_selection[:, max_encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.lstm_layers, -1, -1)

        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            x=embeddings_varying_encoder,
            hx=(input_hidden, input_cell),
            lengths=encoder_lengths,
            enforce_sorted=False
        )

        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            embeddings_varying_decoder,
            (hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # skip connection over lstm
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)

        # static enrichment
        static_context_enrichment = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            lstm_output, self.expand_static_context(
                context=static_context_enrichment,
                timesteps=timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths,
                decoder_length=timesteps - max_encoder_length,
                device=past_target.device
            ),
        )

        # skip connection over attention
        attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length:])

        output = self.pos_wise_ff(attn_output)

        # skip connection over temporal fusion decoder (not LSTM decoder despite the LSTM output contains
        # a skip from the variable selection network)
        output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length:])
        if self.n_targets > 1:  # if to use multi-target architecture
            output = [output_layer(output) for output_layer in self.output_layer]
        else:
            output = self.output_layer(output)
        # x_scale = torch.tensor([[0.5477, 0.2469]], dtype=past_target.dtype)
        # output = self.loss_fn.rescale_parameters(output, target_scale=x_scale, encoder=self.output_transformer)
        # return self.to_network_output(
        #     prediction=self.transform_output(output, target_scale=x["target_scale"]),
        #     attention=attn_output_weights,
        #     static_variables=static_variable_selection,
        #     encoder_variables=encoder_sparse_weights,
        #     decoder_variables=decoder_sparse_weights,
        #     decoder_lengths=decoder_lengths,
        #     encoder_lengths=encoder_lengths,
        # )
        return output

    def on_fit_end(self):
        if self.log_interval > 0:
            self.log_embeddings()

    def create_log(self, x, y, out, batch_idx, **kwargs):
        log = super().create_log(x, y, out, batch_idx, **kwargs)
        if self.log_interval > 0:
            log["interpretation"] = self._log_interpretation(out)
        return log

    def _log_interpretation(self, out):
        # calculate interpretations etc for latter logging
        interpretation = self.interpret_output(
            detach(out),
            reduction="sum",
            attention_prediction_horizon=0,  # attention only for first prediction horizon
        )
        return interpretation

    def epoch_end(self, outputs):
        """
        run at epoch end for training or validation
        """
        if self.log_interval > 0:
            self.log_interpretation(outputs)

    def interpret_output(self,
                         out: Dict[str, torch.Tensor],
                         reduction: str = "none",
                         attention_prediction_horizon: int = 0,
                         attention_as_autocorrelation: bool = False) -> Dict[str, torch.Tensor]:
        """
        interpret output of model

        Args:
            out: output as produced by ``forward()``
            reduction: "none" for no averaging over batches, "sum" for summing attentions, "mean" for
                normalizing by encode lengths
            attention_prediction_horizon: which prediction horizon to use for attention
            attention_as_autocorrelation: if to record attention as autocorrelation - this should be set to true in
                case of ``reduction != "none"`` and differing prediction times of the samples. Defaults to False

        Returns:
            interpretations that can be plotted with ``plot_interpretation()``
        """

        # histogram of decode and encode lengths
        encoder_length_histogram = integer_histogram(out["encoder_lengths"], min=0, max=self.max_encoder_length)
        decoder_length_histogram = integer_histogram(
            out["decoder_lengths"], min=1, max=out["decoder_variables"].size(1)
        )

        # mask where decoder and encoder where not applied when averaging variable selection weights
        encoder_variables = out["encoder_variables"].squeeze(-2)
        encode_mask = create_mask(encoder_variables.size(1), out["encoder_lengths"])
        encoder_variables = encoder_variables.masked_fill(encode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        encoder_variables /= (
            out["encoder_lengths"]
            .where(out["encoder_lengths"] > 0, torch.ones_like(out["encoder_lengths"]))
            .unsqueeze(-1)
        )

        decoder_variables = out["decoder_variables"].squeeze(-2)
        decode_mask = create_mask(decoder_variables.size(1), out["decoder_lengths"])
        decoder_variables = decoder_variables.masked_fill(decode_mask.unsqueeze(-1), 0.0).sum(dim=1)
        decoder_variables /= out["decoder_lengths"].unsqueeze(-1)

        # static variables need no masking
        static_variables = out["static_variables"].squeeze(1)
        # attention is batch x time x heads x time_to_attend
        # average over heads + only keep prediction attention and attention on observed timesteps
        attention = out["attention"][
            :, attention_prediction_horizon, :, : out["encoder_lengths"].max() + attention_prediction_horizon
        ].mean(1)

        if reduction != "none":  # if to average over batches
            static_variables = static_variables.sum(dim=0)
            encoder_variables = encoder_variables.sum(dim=0)
            decoder_variables = decoder_variables.sum(dim=0)

            # reorder attention or averaging
            for i in range(len(attention)):  # very inefficient but does the trick
                if 0 < out["encoder_lengths"][i] < attention.size(1) - attention_prediction_horizon - 1:
                    relevant_attention = attention[
                        i, : out["encoder_lengths"][i] + attention_prediction_horizon
                    ].clone()
                    if attention_as_autocorrelation:
                        relevant_attention = autocorrelation(relevant_attention)
                    attention[i, -out["encoder_lengths"][i] - attention_prediction_horizon :] = relevant_attention
                    attention[i, : attention.size(1) - out["encoder_lengths"][i] - attention_prediction_horizon] = 0.0
                elif attention_as_autocorrelation:
                    attention[i] = autocorrelation(attention[i])

            attention = attention.sum(dim=0)
            if reduction == "mean":
                attention = attention / encoder_length_histogram[1:].flip(0).cumsum(0).clamp(1)
                attention = attention / attention.sum(-1).unsqueeze(-1)  # renormalize
            elif reduction == "sum":
                pass
            else:
                raise ValueError(f"Unknown reduction {reduction}")

            attention = torch.zeros(
                self.max_encoder_length + attention_prediction_horizon, device=self.device
            ).scatter(
                dim=0,
                index=torch.arange(
                    self.max_encoder_length + attention_prediction_horizon - attention.size(-1),
                    self.max_encoder_length + attention_prediction_horizon,
                    device=self.device,
                ),
                src=attention,
            )
        else:
            attention = attention / attention.sum(-1).unsqueeze(-1)  # renormalize

        interpretation = dict(
            attention=attention,
            static_variables=static_variables,
            encoder_variables=encoder_variables,
            decoder_variables=decoder_variables,
            encoder_length_histogram=encoder_length_histogram,
            decoder_length_histogram=decoder_length_histogram,
        )
        return interpretation

    def plot_prediction(self,
                        x: Dict[str, torch.Tensor],
                        out: Dict[str, torch.Tensor],
                        idx: int,plot_attention: bool = True,
                        add_loss_to_title: bool = False,
                        show_future_observed: bool = True,
                        ax=None,
                        **kwargs) -> plt.Figure:
        """
        Plot actuals vs prediction and attention

        Args:
            x (Dict[str, torch.Tensor]): network input
            out (Dict[str, torch.Tensor]): network output
            idx (int): sample index
            plot_attention: if to plot attention on secondary axis
            add_loss_to_title: if to add loss to title. Default to False.
            show_future_observed: if to show actuals for future. Defaults to True.
            ax: matplotlib axes to plot on

        Returns:
            plt.Figure: matplotlib figure
        """

        # plot prediction as normal
        fig = super().plot_prediction(
            x,
            out,
            idx=idx,
            add_loss_to_title=add_loss_to_title,
            show_future_observed=show_future_observed,
            ax=ax,
            **kwargs,
        )

        # add attention on secondary axis
        if plot_attention:
            interpretation = self.interpret_output(out)
            for f in to_list(fig):
                ax = f.axes[0]
                ax2 = ax.twinx()
                ax2.set_ylabel("Attention")
                encoder_length = x["encoder_lengths"][idx]
                ax2.plot(
                    torch.arange(-encoder_length, 0),
                    interpretation["attention"][idx, :encoder_length].detach().cpu(),
                    alpha=0.2,
                    color="k",
                )
                f.tight_layout()
        return fig

    def plot_interpretation(self,
                            interpretation: Dict[str, torch.Tensor]) -> Dict[str, plt.Figure]:
        """
        Make figures that interpret model.

        * Attention
        * Variable selection weights / importances

        Args:
            interpretation: as obtained from ``interpret_output()``

        Returns:
            dictionary of matplotlib figures
        """
        figs = {}

        # attention
        fig, ax = plt.subplots()
        attention = interpretation["attention"].detach().cpu()
        attention = attention / attention.sum(-1).unsqueeze(-1)
        ax.plot(
            np.arange(-self.max_encoder_length, attention.size(0) - self.max_encoder_length), attention
        )
        ax.set_xlabel("Time index")
        ax.set_ylabel("Attention")
        ax.set_title("Attention")
        figs["attention"] = fig

        # variable selection
        def make_selection_plot(title, values, labels):
            fig, ax = plt.subplots(figsize=(7, len(values) * 0.25 + 2))
            order = np.argsort(values)
            values = values / values.sum(-1).unsqueeze(-1)
            ax.barh(np.arange(len(values)), values[order] * 100, tick_label=np.asarray(labels)[order])
            ax.set_title(title)
            ax.set_xlabel("Importance in %")
            plt.tight_layout()
            return fig

        figs["static_variables"] = make_selection_plot(
            "Static variables importance", interpretation["static_variables"].detach().cpu(), self.static_variables
        )
        figs["encoder_variables"] = make_selection_plot(
            "Encoder variables importance", interpretation["encoder_variables"].detach().cpu(), self.encoder_variables
        )
        figs["decoder_variables"] = make_selection_plot(
            "Decoder variables importance", interpretation["decoder_variables"].detach().cpu(), self.decoder_variables
        )

        return figs

    def log_interpretation(self, outputs):
        """
        Log interpretation metrics to tensorboard.
        """
        # extract interpretations
        interpretation = {
            # use padded_stack because decoder length histogram can be of different length
            name: padded_stack([x["interpretation"][name].detach() for x in outputs], side="right", value=0).sum(0)
            for name in outputs[0]["interpretation"].keys()
        }
        # normalize attention with length histogram squared to account for: 1. zeros in attention and
        # 2. higher attention due to less values
        attention_occurances = interpretation["encoder_length_histogram"][1:].flip(0).cumsum(0).float()
        attention_occurances = attention_occurances / attention_occurances.max()
        attention_occurances = torch.cat(
            [
                attention_occurances,
                torch.ones(
                    interpretation["attention"].size(0) - attention_occurances.size(0),
                    dtype=attention_occurances.dtype,
                    device=attention_occurances.device,
                ),
            ],
            dim=0,
        )
        interpretation["attention"] = interpretation["attention"] / attention_occurances.pow(2).clamp(1.0)
        interpretation["attention"] = interpretation["attention"] / interpretation["attention"].sum()

        figs = self.plot_interpretation(interpretation)  # make interpretation figures
        label = ["val", "train"][self.training]
        # log to tensorboard
        for name, fig in figs.items():
            self.logger.experiment.add_figure(
                f"{label.capitalize()} {name} importance", fig, global_step=self.global_step
            )

        # log lengths of encoder/decoder
        for type in ["encoder", "decoder"]:
            fig, ax = plt.subplots()
            lengths = (
                padded_stack([out["interpretation"][f"{type}_length_histogram"] for out in outputs])
                .sum(0)
                .detach()
                .cpu()
            )
            if type == "decoder":
                start = 1
            else:
                start = 0
            ax.plot(torch.arange(start, start + len(lengths)), lengths)
            ax.set_xlabel(f"{type.capitalize()} length")
            ax.set_ylabel("Number of samples")
            ax.set_title(f"{type.capitalize()} length distribution in {label} epoch")

            self.logger.experiment.add_figure(
                f"{label.capitalize()} {type} length distribution", fig, global_step=self.global_step
            )

    def log_embeddings(self):
        """
        Log embeddings to tensorboard
        """
        for name, emb in self.input_embeddings.items():
            labels = self.embedding_labels[name]
            self.logger.experiment.add_embedding(
                emb.weight.data.detach().cpu(), metadata=labels, tag=name, global_step=self.global_step
            )


class TFTModel(TorchParametricProbabilisticForecastingModel, MixedCovariatesTorchModel):
    @random_method
    def __init__(self,
                 # training_length: int = 24,
                 likelihood: Optional[Likelihood] = None,
                 random_state: Optional[Union[int, RandomState]] = None,

                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1,
                 output_size: Union[int, List[int]] = 7,
                 hidden_size: Union[int, List[int]] = 16,
                 lstm_layers: int = 1,
                 dropout: float = 0.1,
                 loss_fn: Optional[nn.Module] = QuantileLoss(),
                 attention_head_size: int = 4,
                 max_encoder_length: int = 10,
                 categorical_groups: Dict[str, List[str]] = {},
                 x_reals: List[str] = [],
                 x_categoricals: List[str] = [],
                 hidden_continuous_size: int = 8,
                 hidden_continuous_sizes: Dict[str, int] = {},
                 embedding_sizes: Dict[str, Tuple[int, int]] = {},
                 embedding_paddings: List[str] = [],
                 embedding_labels: Dict[str, np.ndarray] = {},
                 learning_rate: float = 1e-3,
                 log_interval: Union[int, float] = -1,
                 log_val_interval: Union[int, float] = None,
                 log_gradient_flow: bool = False,
                 reduce_on_plateau_patience: int = 1000,
                 monotone_constaints: Dict[str, int] = {},
                 share_single_variable_networks: bool = False,
                 logging_metrics: nn.ModuleList = None,
                 **kwargs
                 ):
        """Temporal Fusion Transformers (TFT) for Interpretable Multi-horizon Time Series Forecasting.

        This is an implementation of the TFT architecture, as outlined in this paper:
        https://arxiv.org/pdf/1912.09363.pdf

        This model supports mixed covariates (includes static covariates; past covariates known for `input_chunk_length`
        points before prediction time; future covariates known for `input_chunk_length` points before prediction time
        and `input_chunk_length` after prediction time).

        Parameters
        ----------
        input_chunk_length
            The length of the input sequence fed to the model.
        output_chunk_length
            The length of the forecast of the model.
        generic_architecture
            Boolean value indicating whether the generic architecture of N-BEATS is used.
            If not, the interpretable architecture outlined in the paper (consisting of one trend
            and one seasonality stack with appropriate waveform generator functions).
        num_stacks
            The number of stacks that make up the whole model. Only used if `generic_architecture` is set to `True`.
            The interpretable architecture always uses two stacks - one for trend and one for seasonality.
        num_blocks
            The number of blocks making up every stack.
        num_layers
            The number of fully connected layers preceding the final forking layers in each block of every stack.
            Only used if `generic_architecture` is set to `True`.
        layer_widths
            Determines the number of neurons that make up each fully connected layer in each block of every stack.
            If a list is passed, it must have a length equal to `num_stacks` and every entry in that list corresponds
            to the layer width of the corresponding stack. If an integer is passed, every stack will have blocks
            with FC layers of the same width.
        expansion_coefficient_dim
            The dimensionality of the waveform generator parameters, also known as expansion coefficients.
            Only used if `generic_architecture` is set to `True`.
        trend_polynomial_degree
            The degree of the polynomial used as waveform generator in trend stacks. Only used if
            `generic_architecture` is set to `False`.
        random_state
            Control the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random-state>`_ for more details.
        batch_size
            Number of time series (input and output sequences) used in each training pass.
        n_epochs
            Number of epochs over which to train the model.
        optimizer_cls
            The PyTorch optimizer class to be used (default: `torch.optim.Adam`).
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., `{'lr': 1e-3}`
            for specifying a learning rate). Otherwise the default values of the selected `optimizer_cls`
            will be used.
        lr_scheduler_cls
            Optionally, the PyTorch learning rate scheduler class to be used. Specifying `None` corresponds
            to using a constant learning rate.
        lr_scheduler_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer.
        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the `likelihood` parameter is specified.
            Default: `torch.nn.MSELoss()`.
        model_name
            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,
            defaults to the following string "YYYY-mm-dd_HH:MM:SS_torch_model_run_PID", where the initial part of the
            name is formatted with the local date and time, while PID is the processed ID (preventing models spawned at
            the same time by different processes to share the same model_name). E.g.,
            2021-06-14_09:53:32_torch_model_run_44607.
        work_dir
            Path of the working directory, where to save checkpoints and Tensorboard summaries.
            (default: current working directory).
        log_tensorboard
            If set, use Tensorboard to log the different parameters. The logs will be located in:
            `[work_dir]/.darts/runs/`.
        nr_epochs_val_period
            Number of epochs to wait before evaluating the validation loss (if a validation
            `TimeSeries` is passed to the `fit()` method).
        torch_device_str
            Optionally, a string indicating the torch device to use. (default: "cuda:0" if a GPU
            is available, otherwise "cpu")
        force_reset
            If set to `True`, any previously-existing model with the same name will be reset (all checkpoints will
            be discarded).
        """
        kwargs['loss_fn'] = loss_fn
        kwargs['input_chunk_length'] = input_chunk_length
        kwargs['output_chunk_length'] = output_chunk_length
        super().__init__(likelihood=likelihood, **kwargs)

        self.dropout = dropout
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        # self.training_length = [input_chunk_length, output_chunk_length]
        # self.training_length = 10

        self.hidden_size = 16
        self.lstm_layers = 2
        self.dropout = 0.1
        self.output_size = 7

        self.loss_fn = QuantileLoss()
        self.attention_head_size = 4
        self.max_encoder_length = 24*7
        self.categorical_groups = categorical_groups
        self.x_reals = x_reals
        self.x_categoricals = x_categoricals
        self.hidden_continuous_size = hidden_continuous_size
        self.hidden_continuous_sizes = hidden_continuous_sizes
        self.embedding_sizes = embedding_sizes
        self.embedding_paddings = embedding_paddings
        self.embedding_labels = embedding_labels
        self.learning_rate = 0.03
        self.log_interval = 5
        self.log_val_interval = log_val_interval
        self.log_gradient_flow = log_gradient_flow
        self.reduce_on_plateau_patience = 4
        self.monotone_constaints = monotone_constaints
        self.share_single_variable_networks = share_single_variable_networks
        self.logging_metrics = logging_metrics

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> nn.Module:
        """
        `train_samples` contains tensors: 
            (past_target, past_covariates, historic_future_covariates, future_covariates, future_target)
            
            each tensor has shape (n_timesteps, n_variables)
            - past/historic tensors (input_chunk_length, n_variables)
            - future tensors (output_chun_length, n_variables)
        
        From Darts POV
        past_targets -> time_varying_unknown (known in past but not in future)
        past_covariates -> time_varying_unknown
        historic_future_covariates -> time_varying_known (in past of prediction point)
        future_covariates -> time_varying_known (in future of prediction point)
        
        From pytorch-forecasting POV
        time_varying_knowns : future_covariates (including historic_future_covariates)
        time_varying_unknowns : past_targets, past_covariates
        
        time_varying_encoders : [past_targets, past_covariates, historic_future_covariates, future_covariates]
        time_varying_decoders : [historic_future_covariates, future_covariates]
        
        x_reals : all variables from (past_targets, past_covariates, historic_future_covariates)
        
        for categoricals (if we want to use it in the future) we would need embeddings
        """
        past_target, past_covariate, historic_future_covariate, future_covariate, future_target = train_sample
        static_covariates = None

        input_dim = sum([
            t.shape[1] for t in [past_target, past_covariate, historic_future_covariate] if t is not None
        ])
        output_dim = future_target.shape[1]

        tensors = [
            past_target, past_covariate, historic_future_covariate,  # for time varying encoders
            future_covariate, future_target,  # for time varying decoders
            static_covariates  # for static encoder
        ]
        type_names = [
            'past_target', 'past_covariate', 'historic_future_covariate',
            'future_covariate', 'future_target',
            'static_covariate'
        ]
        variable_names = [
            'target', 'past_covariate', 'future_covariate',
            'future_covariate', 'target',
            'static_covariate',
        ]

        variables = {
            'input': {
                type_name: [f'{var_name}_{i}' for i in range(tensor.shape[1])]
                for type_name, var_name, tensor in zip(type_names, variable_names, tensors) if tensor is not None
            },
            'model_config': {}
        }

        reals_input = []
        time_varying_encoder_input = []
        time_varying_decoder_input = []
        static_input = []
        for input_var in type_names:
            if input_var in variables['input']:
                vars = variables['input'][input_var]
                reals_input += vars
                if input_var in ['past_target', 'past_covariate', 'historic_future_covariate']:
                    time_varying_encoder_input += vars
                elif input_var in ['future_covariate']:
                    time_varying_decoder_input += vars
                elif input_var in ['static_covariate']:
                    static_input += vars

        variables['model_config']['reals_input'] = list(dict.fromkeys(reals_input))
        variables['model_config']['time_varying_encoder_input'] = list(dict.fromkeys(time_varying_encoder_input))
        variables['model_config']['time_varying_decoder_input'] = list(dict.fromkeys(time_varying_decoder_input))
        variables['model_config']['static_input'] = list(dict.fromkeys(static_input))

        return _TFTModule(
            input_dim=input_dim,
            output_dim=output_dim,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            variables=variables,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            output_size=self.output_size,
            loss_fn=self.loss_fn,
            attention_head_size=self.attention_head_size,
            max_encoder_length=self.max_encoder_length,
            categorical_groups=self.categorical_groups,
            x_reals=self.x_reals,
            x_categoricals=self.x_categoricals,
            hidden_continuous_size=self.hidden_continuous_size,
            hidden_continuous_sizes=self.hidden_continuous_sizes,
            embedding_sizes=self.embedding_sizes,
            embedding_paddings=self.embedding_paddings,
            embedding_labels=self.embedding_labels,
            learning_rate=self.learning_rate,
            log_interval=self.log_interval,
            log_val_interval=self.log_val_interval,
            log_gradient_flow=self.log_gradient_flow,
            reduce_on_plateau_patience=self.reduce_on_plateau_patience,
            monotone_constaints=self.monotone_constaints,
            share_single_variable_networks=self.share_single_variable_networks,
            logging_metrics=self.logging_metrics,
        )

    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             past_covariates: Optional[Sequence[TimeSeries]],
                             future_covariates: Optional[Sequence[TimeSeries]]) -> MixedCovariatesSequentialDataset:

        return MixedCovariatesSequentialDataset(target_series=target,
                                                past_covariates=past_covariates,
                                                future_covariates=future_covariates,
                                                input_chunk_length=self.input_chunk_length,
                                                output_chunk_length=self.output_chunk_length,
                                                max_samples_per_ts=None)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        raise_if_not(isinstance(train_dataset, MixedCovariatesSequentialDataset),
                     'TFTModel requires a training dataset of type MixedCovariatesSequentialDataset.')

    def _produce_train_output(self, input_batch: Tuple):
        # for MixedCovariatesSequentialDataset
        # past_target, past_covariates, historic_future_covariates, future_covariates = input_batch


        # for DualCovariatesShiftedDataset
        # past_target, historic_future_covariates, future_covariates = input_batch

        # For the RNN we concatenate the past_target with the future_covariates
        # (they have the same length because we enforce a Shift dataset for RNNs)

        # model_input = torch.cat
        # model_input = (past_target, past_covariates, historic_future_covariates, future_covariates)
        # model_input = torch.cat([past_target, future_covariates],
        #                         dim=2) if future_covariates is not None else past_target
        return self.model(input_batch)[0]

    @random_method
    def _produce_predict_output(self, x):
        output = self.model(x).median(dim=2)[0].unsqueeze(dim=2)
        return output if not self.likelihood else self.likelihood.sample(output)

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        """
        This model is a MixedCovariate model

        Parameters:
        ----------
        input_batch
            (past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates)
        """

        series_names = [
            'past_target', 'past_covariates', 'historic_future_covariates', 'future_covariates',
            'future_past_covariates'
        ]
        roll_series = {
            name: torch.clone(series) for name, series in zip(series_names, input_batch) if not series is None
        }

        # determine split point for future series
        output_chunk_start = \
            roll_series['past_target'].shape[1] if n >= self.output_chunk_length else -(self.output_chunk_length - n)



        input_series = tuple(roll_series.get(name, None) for name in series_names[:-1])
        batch_prediction = []
        out = self._produce_predict_output(input_series)
        batch_prediction.append(out[:, :roll_size, :])
        prediction_length = roll_size

        while prediction_length < n:



            # create new input to model from last prediction and current covariates, if available
            new_input = (
                torch.cat([out[:, -1:, :], cov_future[:, prediction_length - 1:prediction_length, :]], dim=2)
                if cov_future is not None else out[:, -1:, :]
            )

            # feed new input to model, including the last hidden state from the previous iteration
            out, last_hidden_state = self._produce_predict_output(new_input, last_hidden_state)

            # append prediction to batch prediction array, increase counter
            batch_prediction.append(out[:, -1:, :])
            prediction_length += 1

        # bring predictions into desired format and drop unnecessary values
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]

        return batch_prediction
