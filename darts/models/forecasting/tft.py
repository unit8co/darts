"""
N-BEATS
-------
"""

from typing import NewType, Union, List, Optional, Tuple, Dict, Sequence
from enum import Enum
import numpy as np
from numpy.random import RandomState
import torch
import torch.nn as nn

from darts.logging import get_logger, raise_log, raise_if_not
from darts.utils.torch import random_method
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel, TorchParametricProbabilisticForecastingModel
from darts.utils.data import DualCovariatesShiftedDataset, TrainingDataset

from darts.metrics import mae, mape, mase, rmse, smape
from darts.models import RNNModel  # for LSTM
from darts import TimeSeries

from darts.models.forecasting.tft_submodels import (
    AddNorm,
    GateAddNorm,
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
    MultiEmbedding,
    LSTM,
)

logger = get_logger(__name__)


class _GType(Enum):
    GENERIC = 1
    TREND = 2
    SEASONALITY = 3


GTypes = NewType('GTypes', _GType)


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
                 hidden_size: Union[int, List[int]] = 16,
                 lstm_layers: int = 1,
                 dropout: float = 0.1,
                 output_size: Union[int, List[int]] = 7,
                 loss_fn: Optional[nn.Module] = torch.nn.MSELoss,
                 attention_head_size: int = 4,
                 max_encoder_length: int = 10,
                 static_categoricals: List[str] = [],
                 static_reals: List[str] = [],
                 time_varying_categoricals_encoder: List[str] = [],
                 time_varying_categoricals_decoder: List[str] = [],
                 categorical_groups: Dict[str, List[str]] = {},
                 time_varying_reals_encoder: List[str] = [],
                 time_varying_reals_decoder: List[str] = [],
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
        self.input_chunk_length_multi = input_chunk_length*input_dim
        self.output_chunk_length = output_chunk_length
        self.target_length = output_chunk_length*input_dim

        # if logging_metrics is None:
        #     logging_metrics = nn.ModuleList([smape(), mae(), rmse(), mape()])
        # # TODO: implement QuantileLoss()?
        # if loss_fn is None:
        #     # loss_fn = QuantileLoss()
        #     raise NotImplementedError('QuantileLoss() not yet implemented')

        # processing inputs
        # embeddings
        self.input_embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            categorical_groups=categorical_groups,
            embedding_paddings=embedding_paddings,
            x_categoricals=x_categoricals,
            max_embedding_size=hidden_size,
        )

        # continuous variable processing
        self.prescalers = nn.ModuleDict(
            {
                name: nn.Linear(1, hidden_continuous_sizes.get(name, hidden_continuous_size))
                for name in self.reals
            }
        )

        # variable selection
        # variable selection for static variables
        static_input_sizes = {name: embedding_sizes[name][1] for name in static_categoricals}
        static_input_sizes.update(
            {
                name: hidden_continuous_sizes.get(name, hidden_continuous_size)
                for name in static_reals
            }
        )
        self.static_variable_selection = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=hidden_size,
            input_embedding_flags={name: True for name in static_categoricals},
            dropout=dropout,
            prescalers=self.prescalers,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: embedding_sizes[name][1] for name in time_varying_categoricals_encoder
        }
        encoder_input_sizes.update(
            {
                name: hidden_continuous_sizes.get(name, hidden_continuous_size)
                for name in time_varying_reals_encoder
            }
        )

        decoder_input_sizes = {
            name: embedding_sizes[name][1] for name in time_varying_categoricals_decoder
        }
        decoder_input_sizes.update(
            {
                name: hidden_continuous_sizes.get(name, hidden_continuous_size)
                for name in time_varying_reals_decoder
            }
        )

        # create single variable grns that are shared across decoder and encoder
        if share_single_variable_networks:
            self.shared_single_variable_grns = nn.ModuleDict()
            for name, input_size in encoder_input_sizes.items():
                self.shared_single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, hidden_size),
                    hidden_size,
                    dropout,
                )
            for name, input_size in decoder_input_sizes.items():
                if name not in self.shared_single_variable_grns:
                    self.shared_single_variable_grns[name] = GatedResidualNetwork(
                        input_size,
                        min(input_size, hidden_size),
                        hidden_size,
                        dropout,
                    )

        self.encoder_variable_selection = VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=hidden_size,
            input_embedding_flags={name: True for name in time_varying_categoricals_encoder},
            dropout=dropout,
            context_size=hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=hidden_size,
            input_embedding_flags={name: True for name in time_varying_categoricals_decoder},
            dropout=dropout,
            context_size=hidden_size,
            prescalers=self.prescalers,
            single_variable_grns={}
            if not share_single_variable_networks
            else self.shared_single_variable_grns,
        )

        # static encoders
        # for variable selection
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # for hidden state of the lstm
        self.static_context_initial_hidden_lstm = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout
        )

        # TODO: Check LSTMs
        # lstm encoder (history) and decoder (future) for local processing
        self.lstm_encoder = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True,
        )

        # self.lstm_encoder = RNNModel(
        #     model='LSTM',
        #     input_chunk_length=hidden_size,
        #     hidden_dim=hidden_size,
        #     n_rnn_layers=lstm_layers,
        #     dropout=dropout if lstm_layers > 1 else 0,
        #     # batch_first=True,
        # )
        #
        # self.lstm_decoder = RNNModel(
        #     model='LSTM',
        #     input_chunk_length=hidden_size,
        #     hidden_dim=hidden_size,
        #     n_rnn_layers=lstm_layers,
        #     dropout=dropout if lstm_layers > 1 else 0,
        #     # batch_first=True,
        # )

        # skip connection for lstm
        self.post_lstm_gate_encoder = GatedLinearUnit(hidden_size, dropout=dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder
        # self.post_lstm_gate_decoder = GatedLinearUnit(hidden_size, dropout=dropout)
        self.post_lstm_add_norm_encoder = AddNorm(hidden_size, trainable_add=False)
        # self.post_lstm_add_norm_decoder = AddNorm(hidden_size, trainable_add=True)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder

        # static enrichment and processing past LSTM
        self.static_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size,
        )

        # attention for long-range processing
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=hidden_size, n_head=attention_head_size, dropout=dropout
        )
        self.post_attn_gate_norm = GateAddNorm(
            hidden_size, dropout=dropout, trainable_add=False
        )
        self.pos_wise_ff = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, dropout=dropout
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gate_norm = GateAddNorm(hidden_size, dropout=None, trainable_add=False)

        if self.n_targets > 1:  # if to run with multiple targets
            self.output_layer = nn.ModuleList(
                [nn.Linear(hidden_size, output_size) for output_size in output_size]
            )
        else:
            self.output_layer = nn.Linear(hidden_size, output_size)

    @staticmethod
    def expand_static_context(context, timesteps):
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(self, encoder_lengths: torch.LongTensor, decoder_length: int):
        """
        Returns causal mask to apply for self-attention layer.

        Args:
            self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=self.device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
        # do not attend to steps to self or after prediction
        # todo: there is potential value in attending to future forecasts if they are made with knowledge currently
        #   available
        #   one possibility is here to use a second attention layer for future attention (assuming different effects
        #   matter in the future than the past)
        #   or alternatively using the same layer but allowing forward attention - i.e. only masking out non-available
        #   data and self
        decoder_mask = attend_step >= predict_step
        # do not attend to steps where data is padded
        encoder_mask = create_mask(encoder_lengths.max(), encoder_lengths)
        # combine masks along attended time - first encoder and then decoder
        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(encoder_lengths.size(0), -1, -1),
            ),
            dim=2,
        )
        return mask

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        input dimensions: n_samples x time x variables
        """
        encoder_lengths = x["encoder_lengths"]
        decoder_lengths = x["decoder_lengths"]
        x_cat = torch.cat([x["encoder_cat"], x["decoder_cat"]], dim=1)  # concatenate in time dimension
        x_cont = torch.cat([x["encoder_cont"], x["decoder_cont"]], dim=1)  # concatenate in time dimension
        timesteps = x_cont.size(1)  # encode + decode length
        max_encoder_length = int(encoder_lengths.max())
        input_vectors = self.input_embeddings(x_cat)
        input_vectors.update(
            {
                name: x_cont[..., idx].unsqueeze(-1)
                for idx, name in enumerate(self.hparams.x_reals)
                if name in self.reals
            }
        )

        # Embedding and variable selection
        if len(self.static_variables) > 0:
            # static embeddings will be constant over entire batch
            static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            static_embedding, static_variable_selection = self.static_variable_selection(static_embedding)
        else:
            static_embedding = torch.zeros(
                (x_cont.size(0), self.hparams.hidden_size), dtype=self.dtype, device=self.device
            )
            static_variable_selection = torch.zeros((x_cont.size(0), 0), dtype=self.dtype, device=self.device)

        static_context_variable_selection = self.expand_static_context(
            self.static_context_variable_selection(static_embedding), timesteps
        )

        embeddings_varying_encoder = {
            name: input_vectors[name][:, :max_encoder_length] for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_varying_encoder,
            static_context_variable_selection[:, :max_encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors[name][:, max_encoder_length:] for name in self.decoder_variables  # select decoder
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_varying_decoder,
            static_context_variable_selection[:, max_encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.hparams.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(self.hparams.lstm_layers, -1, -1)

        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            embeddings_varying_encoder, (input_hidden, input_cell), lengths=encoder_lengths, enforce_sorted=False
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
            lstm_output, self.expand_static_context(static_context_enrichment, timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, max_encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths, decoder_length=timesteps - max_encoder_length
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

        return self.to_network_output(
            prediction=self.transform_output(output, target_scale=x["target_scale"]),
            attention=attn_output_weights,
            static_variables=static_variable_selection,
            encoder_variables=encoder_sparse_weights,
            decoder_variables=decoder_sparse_weights,
            decoder_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
        )

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


class TFTModel(TorchParametricProbabilisticForecastingModel, MixedCovariatesTorchModel):
    @random_method
    def __init__(self,
                 # input_chunk_length: int,
                 # output_chunk_length: int,
                 # generic_architecture: bool = True,
                 # num_stacks: int = 30,
                 # num_blocks: int = 1,
                 # num_layers: int = 4,
                 # layer_widths: Union[int, List[int]] = 256,
                 # expansion_coefficient_dim: int = 5,
                 # trend_polynomial_degree: int = 2,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 hidden_size: Union[int, List[int]] = 16,
                 lstm_layers: int = 1,
                 dropout: float = 0.1,
                 output_size: Union[int, List[int]] = 7,
                 loss_fn: Optional[nn.Module] = torch.nn.MSELoss,
                 attention_head_size: int = 4,
                 max_encoder_length: int = 10,
                 static_categoricals: List[str] = [],
                 static_reals: List[str] = [],
                 time_varying_categoricals_encoder: List[str] = [],
                 time_varying_categoricals_decoder: List[str] = [],
                 categorical_groups: Dict[str, List[str]] = {},
                 time_varying_reals_encoder: List[str] = [],
                 time_varying_reals_decoder: List[str] = [],
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
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs):
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

        kwargs['input_chunk_length'] = input_chunk_length
        kwargs['output_chunk_length'] = output_chunk_length
        super().__init__(**kwargs)

        raise_if_not(isinstance(hidden_size, int) or len(hidden_size) == lstm_layers,
                     "Please pass an integer or a list of integers with length `lstm_layers`"
                     "as value for the `hidden_size` argument.", logger)

        self.random_state = random_state
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.output_size = output_size
        self.loss_fn = loss_fn
        self.attention_head_size = attention_head_size
        self.max_encoder_length = max_encoder_length
        self.static_categoricals = static_categoricals
        self.static_reals = static_reals
        self.time_varying_categoricals_encoder = time_varying_categoricals_encoder
        self.time_varying_categoricals_decoder = time_varying_categoricals_decoder
        self.categorical_groups = categorical_groups
        self.time_varying_reals_encoder = time_varying_reals_encoder
        self.time_varying_reals_decoder = time_varying_reals_decoder
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

        if isinstance(hidden_size, int):
            self.hidden_size = [hidden_size] * lstm_layers

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (train_sample[1].shape[1] if train_sample[1] is not None else 0)
        output_dim = train_sample[-1].shape[1]

        return _TFTModule(
            input_dim=input_dim,
            output_dim=output_dim,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            random_state=self.random_state,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            output_size=self.output_size,
            loss_fn=self.loss_fn,
            attention_head_size=self.attention_head_size,
            max_encoder_length=self.max_encoder_length,
            static_categoricals=self.static_categoricals,
            static_reals=self.static_reals,
            time_varying_categoricals_encoder=self.time_varying_categoricals_encoder,
            time_varying_categoricals_decoder=self.time_varying_categoricals_decoder,
            categorical_groups=self.categorical_groups,
            time_varying_reals_encoder=self.time_varying_reals_encoder,
            time_varying_reals_decoder=self.time_varying_reals_decoder,
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
            logging_metrics=self.logging_metrics
        )

    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             past_covariates: Optional[Sequence[TimeSeries]],
                             future_covariates: Optional[Sequence[TimeSeries]]) -> DualCovariatesShiftedDataset:

        return MixedCovariatesTorchModel(target_series=target,
                                            covariates=future_covariates,
                                            length=self.training_length,
                                            shift=1)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        raise_if_not(isinstance(train_dataset, DualCovariatesShiftedDataset),
                     'RNNModel requires a training dataset of type DualCovariatesShiftedDataset.')
        raise_if_not(train_dataset.ds_past.shift == 1, 'RNNModel requires a shifted training dataset with shift=1.')

    def _produce_train_output(self, input_batch: Tuple):
        past_target, historic_future_covariates, future_covariates = input_batch
        # For the RNN we concatenate the past_target with the future_covariates
        # (they have the same length because we enforce a Shift dataset for RNNs)
        model_input = torch.cat([past_target, future_covariates],
                                dim=2) if future_covariates is not None else past_target
        return self.model(model_input)[0]

    @random_method
    def _produce_predict_output(self, x, last_hidden_state=None):
        if self.likelihood:
            output, hidden = self.model(x, last_hidden_state)
            return self.likelihood.sample(output), hidden
        else:
            return self.model(x, last_hidden_state)

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        """
        This model is recurrent, so we have to write a specific way to obtain the time series forecasts of length n.
        """
        past_target, historic_future_covariates, future_covariates = input_batch

        if historic_future_covariates is not None:
            # RNNs need as inputs (target[t] and covariates[t+1]) so here we shift the covariates
            all_covariates = torch.cat([historic_future_covariates[:, 1:, :], future_covariates], dim=1)
            cov_past, cov_future = all_covariates[:, :past_target.shape[1], :], all_covariates[:, past_target.shape[1]:, :]
            input_series = torch.cat([past_target, cov_past], dim=2)
        else:
            input_series = past_target
            cov_future = None

        batch_prediction = []
        out, last_hidden_state = self._produce_predict_output(input_series)
        batch_prediction.append(out[:, -1:, :])
        prediction_length = 1

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
