"""
Temporal Fusion Transformer (TFT)
-------
"""

from typing import Union, List, Optional, Tuple, Dict, Sequence

import numpy as np
from numpy.random import RandomState

import torch
from torch import nn

from darts import TimeSeries
from darts.utils.torch import random_method
from darts.logging import get_logger, raise_if_not, raise_if
from darts.utils.likelihood_models import QuantileRegression, Likelihood
from darts.utils.timeseries_generation import datetime_attribute_timeseries, _generate_index
from darts.utils.data import (
    TrainingDataset,
    MixedCovariatesSequentialDataset,
    MixedCovariatesTrainingDataset,
    MixedCovariatesInferenceDataset
)
from darts.models.forecasting.torch_forecasting_model import (
    MixedCovariatesTorchModel,
    TorchParametricProbabilisticForecastingModel
)
from darts.models.forecasting.tft_submodels import (
    _GateAddNorm,
    _GatedResidualNetwork,
    _InterpretableMultiHeadAttention,
    _VariableSelectionNetwork,
)
from torch.nn import LSTM as _LSTM

logger = get_logger(__name__)

MixedCovariatesTrainTensorType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class _TFTModule(nn.Module):

    def __init__(self,
                 output_dim: Tuple[int, int],
                 input_chunk_length: int,
                 output_chunk_length: int,
                 variables_meta: Dict[str, Dict[str, List[str]]],
                 hidden_size: Union[int, List[int]] = 16,
                 lstm_layers: int = 1,
                 num_attention_heads: int = 4,
                 full_attention: bool = False,
                 hidden_continuous_size: int = 8,
                 dropout: float = 0.1,
                 add_relative_index: bool = False,
                 likelihood: Optional[Likelihood] = None):

        """ PyTorch module implementing the TFT architecture from `this paper <https://arxiv.org/pdf/1912.09363.pdf>`_
        The implementation is built upon `pytorch-forecasting's TemporalFusionTransformer
        <https://pytorch-forecasting.readthedocs.io/en/latest/models.html>`_.

        Parameters
        ----------
        output_dim : Tuple[int, int]
            shape of output given by (n_targets, loss_size). (loss_size corresponds to nr_params in other models).
        input_chunk_length : int
            encoder length; number of past time steps that are fed to the forecasting module at prediction time.
        output_chunk_length : int
            decoder length; number of future time steps that are fed to the forecasting module at prediction time.
        variables_meta : Dict[str, Dict[str, List[str]]]
            dict containing variable enocder, decoder variable names for mapping tensors in `_TFTModule.forward()`
        hidden_size : int
            hidden state size of the TFT. It is the main hyper-parameter and common across the internal TFT architecture.
        lstm_layers : int
            number of layers for the Long Short Term Memory (LSTM) Encoder and Decoder (1 is a good default).
        num_attention_heads : int
            number of attention heads (4 is a good default)
        full_attention : bool
            If `True`, applies multi-head attention query on past (encoder) and future (decoder) parts. Otherwise,
            only queries on future part. Defaults to `False`.
        hidden_continuous_size : int
            default for hidden size for processing continuous variables
        dropout : float
            Fraction of neurons afected by Dropout.
        add_relative_index : bool
            Whether to add positional values to future covariates. Defaults to `False`.
            This allows to use the TFTModel without having to pass future_covariates to `fit()` and `train()`.
            It gives a value to the position of each step from input and output chunk relative to the prediction
            point. The values are normalized with `input_chunk_length`.
        likelihood
            The likelihood model to be used for probabilistic forecasts. By default the TFT uses
            a ``QuantileRegression`` likelihood.
        """

        super(_TFTModule, self).__init__()

        self.n_targets, self.loss_size = output_dim
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.variables_meta = variables_meta
        self.hidden_size = hidden_size
        self.hidden_continuous_size = hidden_continuous_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.full_attention = full_attention
        self.dropout = dropout
        self.likelihood = likelihood
        self.add_relative_index = add_relative_index

        # initialize last batch size to check if new mask needs to be generated
        self.batch_size_last = -1
        self.attention_mask = None
        self.relative_index = None

        # general information on variable name endings:
        # _vsn: VariableSelectionNetwork
        # _grn: GatedResidualNetwork
        # _glu: GatedLinearUnit
        # _gan: GateAddNorm
        # _attn: Attention

        # # processing inputs
        # continuous variable processing
        self.prescalers_linear = {name: nn.Linear(1, self.hidden_continuous_size) for name in self.reals}

        static_input_sizes = {name: self.hidden_continuous_size for name in self.static_variables}

        self.static_covariates_vsn = _VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},  # this would be required for categorical inputs
            dropout=self.dropout,
            prescalers=self.prescalers_linear,
            single_variable_grns={},
            context_size=None  # no context for static variables
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {name: self.hidden_continuous_size for name in self.encoder_variables}

        decoder_input_sizes = {name: self.hidden_continuous_size for name in self.decoder_variables}

        self.encoder_vsn = _VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},  # this would be required for categorical inputs
            dropout=self.dropout,
            context_size=self.hidden_size,
            prescalers=self.prescalers_linear,
            single_variable_grns={}
        )

        self.decoder_vsn = _VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},  # this would be required for categorical inputs
            dropout=self.dropout,
            context_size=self.hidden_size,
            prescalers=self.prescalers_linear,
            single_variable_grns={}
        )

        # static encoders
        # for variable selection
        self.static_context_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # for hidden state of the lstm
        self.static_context_hidden_encoder_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # for cell state of the lstm
        self.static_context_cell_encoder_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout
        )

        # lstm encoder (history) and decoder (future) for local processing
        self.lstm_encoder = _LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = _LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        # post lstm GateAddNorm
        self.post_lstm_gan = _GateAddNorm(
            input_size=self.hidden_size,
            dropout=dropout
        )

        # static enrichment and processing past LSTM
        self.static_enrichment_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            context_size=self.hidden_size,
        )

        # attention for long-range processing
        self.multihead_attn = _InterpretableMultiHeadAttention(
            d_model=self.hidden_size, n_head=self.num_attention_heads, dropout=self.dropout
        )
        self.post_attn_gan = _GateAddNorm(self.hidden_size, dropout=self.dropout)
        self.positionwise_feedforward_grn = _GatedResidualNetwork(
            self.hidden_size, self.hidden_size, self.hidden_size, dropout=self.dropout
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gan = _GateAddNorm(self.hidden_size, dropout=None)

        self.output_layer = nn.Linear(self.hidden_size, self.n_targets * self.loss_size)

    @property
    def reals(self) -> List[str]:
        """
        List of all continuous variables in model
        """
        return self.variables_meta['model_config']['reals_input']

    @property
    def static_variables(self) -> List[str]:
        """
        List of all static variables in model
        """
        # TODO: (Darts: dbader) we might want to include static variables in the future?
        return self.variables_meta['model_config']['static_input']

    @property
    def encoder_variables(self) -> List[str]:
        """
        List of all encoder variables in model (excluding static variables)
        """
        return self.variables_meta['model_config']['time_varying_encoder_input']

    @property
    def decoder_variables(self) -> List[str]:
        """
        List of all decoder variables in model (excluding static variables)
        """
        return self.variables_meta['model_config']['time_varying_decoder_input']

    @staticmethod
    def expand_static_context(context: torch.Tensor, time_steps: int) -> torch.Tensor:
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, time_steps, -1)

    @staticmethod
    def get_relative_index(encoder_length: int,
                                decoder_length: int,
                                batch_size: int,
                                dtype: torch.dtype,
                                device: torch.device) -> torch.Tensor:
        """
        Returns scaled time index relative to prediction point.
        """
        index = torch.arange(encoder_length + decoder_length, dtype=dtype, device=device)
        prediction_index = encoder_length - 1
        index[:encoder_length] = index[:encoder_length] / prediction_index
        index[encoder_length:] = index[encoder_length:] / prediction_index
        return index.resize(1, len(index), 1).repeat(batch_size, 1, 1)

    @staticmethod
    def get_attention_mask_full(time_steps: int,
                                batch_size: int,
                                dtype: torch.dtype,
                                device: torch.device) -> torch.Tensor:
        """
        Returns causal mask to apply for self-attention layer.
        """
        eye = torch.eye(time_steps, dtype=dtype, device=device)
        mask = torch.cumsum(eye.unsqueeze(0).repeat(batch_size, 1, 1), dim=1)
        return mask < 1

    @staticmethod
    def get_attention_mask_future(encoder_length: int,
                                  decoder_length: int,
                                  batch_size: int,
                                  device: str) -> torch.Tensor:
        """
        Returns causal mask to apply for self-attention layer that acts on future input only.
        """
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=device)[:, None]
        # do not attend to steps to self or after prediction
        decoder_mask = attend_step >= predict_step
        # do not attend to past input
        encoder_mask = torch.zeros(batch_size, encoder_length, dtype=torch.bool, device=device)
        # combine masks along attended time - first encoder and then decoder

        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(batch_size, -1, -1),
            ),
            dim=2
        )
        return mask

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        input dimensions: (n_samples, n_time_steps, n_variables)
        """

        dim_samples, dim_time, dim_variable, dim_loss = 0, 1, 2, 3
        past_target, past_covariates, historic_future_covariates, future_covariates = x

        batch_size = past_target.shape[dim_samples]
        encoder_length = self.input_chunk_length
        decoder_length = self.output_chunk_length
        time_steps = encoder_length + decoder_length

        # avoid unnecessary regeneration of attention mask
        if batch_size != self.batch_size_last:
            if self.full_attention:
                self.attention_mask = self.get_attention_mask_full(time_steps=time_steps,
                                                                   batch_size=batch_size,
                                                                   dtype=past_target.dtype,
                                                                   device=past_target.device)
            else:
                self.attention_mask = self.get_attention_mask_future(encoder_length=encoder_length,
                                                                     decoder_length=decoder_length,
                                                                     batch_size=batch_size,
                                                                     device=past_target.device)
            if self.add_relative_index:
                self.relative_index = self.get_relative_index(encoder_length=encoder_length,
                                                              decoder_length=decoder_length,
                                                              batch_size=batch_size,
                                                              device=past_target.device,
                                                              dtype=past_target.dtype)

            self.batch_size_last = batch_size

        if self.add_relative_index:
            historic_future_covariates = torch.cat(
                [ts[:, :encoder_length, :] for ts in [historic_future_covariates, self.relative_index] if
                 ts is not None],
                dim=dim_variable
            )
            future_covariates = torch.cat(
                [ts[:, -decoder_length:, :] for ts in [future_covariates, self.relative_index] if
                 ts is not None],
                dim=dim_variable
            )
        # TODO: impelement static covariates
        static_covariates = None

        # data is of size (batch_size, input_length, input_size)
        x_cont_past = torch.cat(
            [tensor for tensor in [past_target,
                                   past_covariates,
                                   historic_future_covariates,
                                   static_covariates] if tensor is not None], dim=dim_variable
        )

        x_cont_future = torch.cat(
            [tensor for tensor in [future_covariates,
                                   static_covariates] if tensor is not None], dim=dim_variable
        )

        input_vectors_past = {
            name: x_cont_past[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.encoder_variables)
        }
        input_vectors_future = {
            name: x_cont_future[..., idx].unsqueeze(-1)
            for idx, name in enumerate(self.decoder_variables)
        }

        # Embedding and variable selection
        if static_covariates is not None:
            # TODO: impelement static covariates
            # # static embeddings will be constant over entire batch
            # static_embedding = {name: input_vectors[name][:, 0] for name in self.static_variables}
            # static_embedding, static_covariate_var = self.static_covariates_vsn(static_embedding)
            raise NotImplementedError('Static covariates have yet to be defined')
        else:
            static_embedding = torch.zeros((past_target.shape[0], self.hidden_size),
                                           dtype=past_target.dtype,
                                           device=past_target.device)

            # this is only to interpret the output
            static_covariate_var = torch.zeros((past_target.shape[0], 0),
                                               dtype=past_target.dtype,
                                               device=past_target.device)

        if future_covariates is None and static_covariates is None:
            raise NotImplementedError('make zero tensor if future covariates is None')

        static_context_expanded = self.expand_static_context(
            context=self.static_context_grn(static_embedding),
            time_steps=time_steps
        )

        embeddings_varying_encoder = {name: input_vectors_past[name] for name in self.encoder_variables}
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_vsn(
            x=embeddings_varying_encoder,
            context=static_context_expanded[:, :encoder_length],
        )

        embeddings_varying_decoder = {name: input_vectors_future[name] for name in self.decoder_variables}
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_vsn(
            x=embeddings_varying_decoder,
            context=static_context_expanded[:, encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = self.static_context_hidden_encoder_grn(static_embedding).expand(
            self.lstm_layers, -1, -1
        )
        input_cell = self.static_context_cell_encoder_grn(static_embedding).expand(
            self.lstm_layers, -1, -1
        )

        # run local lstm encoder
        encoder_out, (hidden, cell) = self.lstm_encoder(input=embeddings_varying_encoder, hx=(input_hidden, input_cell))

        # run local lstm decoder
        decoder_out, _ = self.lstm_decoder(input=embeddings_varying_decoder, hx=(hidden, cell))

        lstm_layer = torch.cat([encoder_out, decoder_out], dim=dim_time)
        input_embeddings = torch.cat([embeddings_varying_encoder, embeddings_varying_decoder], dim=dim_time)

        # post lstm GateAddNorm
        lstm_out = self.post_lstm_gan(x=lstm_layer, skip=input_embeddings)

        # static enrichment
        static_context_enriched = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment_grn(
            x=lstm_out,
            context=self.expand_static_context(context=static_context_enriched, time_steps=time_steps)
        )

        # multi-head attention
        attn_out, attn_out_weights = self.multihead_attn(
            q=attn_input if self.full_attention else attn_input[:, encoder_length:],
            k=attn_input,
            v=attn_input,
            mask=self.attention_mask
        )

        # skip connection over attention
        attn_out = self.post_attn_gan(x=attn_out,
                                      skip=attn_input if self.full_attention else attn_input[:, encoder_length:])

        # position-wise feed-forward
        out = self.positionwise_feedforward_grn(x=attn_out, context=None)

        # skip connection over temporal fusion decoder from LSTM post _GateAddNorm
        out = self.pre_output_gan(x=out,
                                  skip=lstm_out if self.full_attention else lstm_out[:, encoder_length:])

        # generate output for n_targets and loss_size elements for loss evaluation
        
        out = self.output_layer(out[:, encoder_length:] if self.full_attention else out)
        out = out.view(batch_size, self.output_chunk_length, self.n_targets, self.loss_size)

        # TODO: (Darts) remember this in case we want to output interpretation
        # return self.to_network_output(
        #     prediction=self.transform_output(out, target_scale=x["target_scale"]),
        #     attention=attn_out_weights,
        #     static_variables=static_covariate_var,
        #     encoder_variables=encoder_sparse_weights,
        #     decoder_variables=decoder_sparse_weights,
        #     decoder_lengths=decoder_lengths,
        #     encoder_lengths=encoder_lengths,
        # )

        return out


class TFTModel(TorchParametricProbabilisticForecastingModel, MixedCovariatesTorchModel):
    @random_method
    def __init__(self,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1,
                 hidden_size: Union[int, List[int]] = 16,
                 lstm_layers: int = 1,
                 num_attention_heads: int = 4,
                 full_attention: bool = False,
                 dropout: float = 0.1,
                 hidden_continuous_size: int = 8,
                 add_cyclic_encoder: Optional[str] = None,
                 add_relative_index: bool = False,
                 loss_fn: Optional[nn.Module] = None,
                 likelihood: Optional[Likelihood] = None,
                 max_samples_per_ts: Optional[int] = None,
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs
                 ):
        """Temporal Fusion Transformers (TFT) for Interpretable Time Series Forecasting.

        This is an implementation of the TFT architecture, as outlined in this paper:
        https://arxiv.org/pdf/1912.09363.pdf.

        The internal sub models are adopted from `pytorch-forecasting's TemporalFusionTransformer
        <https://pytorch-forecasting.readthedocs.io/en/latest/models.html>`_ implementation.

        This model supports mixed covariates (includes past covariates known for `input_chunk_length`
        points before prediction time and future covariates known for `output_chunk_length` after prediction time).

        The TFT applies multi-head attention queries on future inputs from mandatory `future_covariates`.
        Specifying `add_cyclic_encoder` (read below) adds cyclic temporal encoding to the model and allows to use
        the model without having to specify additional `future_covariates` for training and prediction.

        By default, this model uses the ``QuantileRegression`` likelihood, which means that its forecasts are
        probabilistic; it is recommended to call ``predict()`` with ``num_samples >> 1`` to get meaningful results.

        Parameters
        ----------
        input_chunk_length : int
            encoder length; number of past time steps that are fed to the forecasting module at prediction time.
        output_chunk_length : int
            decoder length; number of future time steps that are fed to the forecasting module at prediction time.
        hidden_size : int
            hidden state size of the TFT. It is the main hyper-parameter and common across the internal TFT
            architecture.
        lstm_layers : int
            number of layers for the Long Short Term Memory (LSTM) Encoder and Decoder (1 is a good default).
        num_attention_heads : int
            number of attention heads (4 is a good default)
        full_attention : bool
            If `True`, applies multi-head attention query on past (encoder) and future (decoder) parts. Otherwise,
            only queries on future part. Defaults to `False`.
        dropout : float
            Fraction of neurons afected by Dropout.
        hidden_continuous_size : int
            default for hidden size for processing continuous variables
        add_cyclic_encoder : optional str
            If other than None, apply cycling encoding to an attribute of the time index and add it to the
            `future_covariates`.
            This allows to use the TFTModel without having to pass future_covariates to `fit()` and `train()`
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        add_relative_index : bool
            Whether to add positional values to future covariates. Defaults to `False`.
            This allows to use the TFTModel without having to pass future_covariates to `fit()` and `train()`.
            It gives a value to the position of each step from input and output chunk relative to the prediction
            point. The values are normalized with `input_chunk_length`.
        loss_fn : nn.Module
            PyTorch loss function used for training. By default the TFT model is probabilistic and uses a ``likelihood``
            instead (``QuantileRegression``). To make the model deterministic, you can set the ``likelihood`` to None
            and give a ``loss_fn`` argument.
        likelihood
            The likelihood model to be used for probabilistic forecasts. By default the TFT uses
            a ``QuantileRegression`` likelihood.
        max_samples_per_ts
            Optionally, a maximum number of training sample to generate per time series.
        random_state
            Control the randomness of the weights initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random-state>`_ for more details.
        **kwargs
            Optional arguments to initialize the torch.Module

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
        save_checkpoints
            Whether or not to automatically save the untrained model and checkpoints from training.
            If set to `False`, the model can still be manually saved using :meth:`save_model()
            <TorchForeCastingModel.save_model()>` and loaded using :meth:`load_model()
            <TorchForeCastingModel.load_model()>`.
        """
        if likelihood is None and loss_fn is None:
            # This is the default if no loss information is provided
            likelihood = QuantileRegression()

        kwargs['loss_fn'] = loss_fn
        kwargs['input_chunk_length'] = input_chunk_length
        kwargs['output_chunk_length'] = output_chunk_length
        super().__init__(likelihood=likelihood, **kwargs)

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.full_attention = full_attention
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.add_cyclic_encoder = add_cyclic_encoder
        self.add_relative_index = add_relative_index
        self.loss_fn = loss_fn
        self.likelihood = likelihood
        self.max_sample_per_ts = max_samples_per_ts
        self.output_dim: Optional[Tuple[int, int]] = None

    def _create_model(self,
                      train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """
        `train_sample` contains the following tensors:
            (past_target, past_covariates, historic_future_covariates, future_covariates, future_target)

            each tensor has shape (n_timesteps, n_variables)
            - past/historic tensors have shape (input_chunk_length, n_variables)
            - future tensors have shape (output_chunk_length, n_variables)

        Darts Interpretation of pytorch-forecasting's TimeSeriesDataSet:
            time_varying_knowns : future_covariates (including historic_future_covariates)
            time_varying_unknowns : past_targets, past_covariates

            time_varying_encoders : [past_targets, past_covariates, historic_future_covariates, future_covariates]
            time_varying_decoders : [historic_future_covariates, future_covariates]

        `variable_meta` is used in TFT to access specific variables
        """
        past_target, past_covariate, historic_future_covariate, future_covariate, future_target = train_sample

        # add a covariate placeholder so that relative index will be included
        if self.add_relative_index:
            time_steps = self.input_chunk_length + self.output_chunk_length

            expand_future_covariate = np.arange(time_steps).reshape((time_steps, 1))

            historic_future_covariate = np.concatenate(
                [ts[:self.input_chunk_length] for ts in [historic_future_covariate, expand_future_covariate] if
                 ts is not None],
                axis=1
            )
            future_covariate = np.concatenate(
                [ts[-self.output_chunk_length:] for ts in [future_covariate, expand_future_covariate] if
                 ts is not None],
                axis=1
            )

        static_covariates = None  # placeholder for future

        self.output_dim = (future_target.shape[1], 1) if self.likelihood is None else \
            (future_target.shape[1], self.likelihood.num_parameters)

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

        variables_meta = {
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
            if input_var in variables_meta['input']:
                vars = variables_meta['input'][input_var]
                reals_input += vars
                if input_var in ['past_target', 'past_covariate', 'historic_future_covariate']:
                    time_varying_encoder_input += vars
                elif input_var in ['future_covariate']:
                    time_varying_decoder_input += vars
                elif input_var in ['static_covariate']:
                    static_input += vars

        variables_meta['model_config']['reals_input'] = list(dict.fromkeys(reals_input))
        variables_meta['model_config']['time_varying_encoder_input'] = list(dict.fromkeys(time_varying_encoder_input))
        variables_meta['model_config']['time_varying_decoder_input'] = list(dict.fromkeys(time_varying_decoder_input))
        variables_meta['model_config']['static_input'] = list(dict.fromkeys(static_input))

        return _TFTModule(
            variables_meta=variables_meta,
            output_dim=self.output_dim,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            num_attention_heads=self.num_attention_heads,
            full_attention=self.full_attention,
            hidden_continuous_size=self.hidden_continuous_size,
            likelihood=self.likelihood,
            add_relative_index=self.add_relative_index
        )

    def _build_train_dataset(self,
                             target: Sequence[TimeSeries],
                             past_covariates: Optional[Sequence[TimeSeries]],
                             future_covariates: Optional[Sequence[TimeSeries]]) -> MixedCovariatesSequentialDataset:

        raise_if(future_covariates is None and self.add_cyclic_encoder is None and not self.add_relative_index,
                 'TFTModel requires future covariates. The model applies multi-head attention queries on future '
                 'inputs. Consider specifying `add_cyclic_encoder` or setting `add_relative_index` to `True` '
                 'at model creation (read TFT model docs for more information). These will automatically generate '
                 '`future_covariates` from indexes.',
                 logger)

        if self.add_cyclic_encoder is not None:
            future_covariates = self._add_cyclic_encoder(target, future_covariates=future_covariates, n=None)

        return MixedCovariatesSequentialDataset(target_series=target,
                                                past_covariates=past_covariates,
                                                future_covariates=future_covariates,
                                                input_chunk_length=self.input_chunk_length,
                                                output_chunk_length=self.output_chunk_length,
                                                max_samples_per_ts=self.max_sample_per_ts)

    def _add_cyclic_encoder(self,
                            target: Sequence[TimeSeries],
                            future_covariates: Optional[Sequence[TimeSeries]] = None,
                            n: Optional[int] = None) -> Sequence[TimeSeries]:
        """adds cyclic encoding of time index to future covariates.
        For training (when `n` is `None`) we can simply use the future covariates (if available) or target as
        reference to extract the time index.
        For prediction (`n` is given) we have to distinguish between two cases:
            1)
                if future covariates are given, we can use them as reference
            2)
                if future covariates are missing, we need to generate a time index that starts `input_chunk_length`
                before the end of `target` and ends `max(n, output_chunk_length)` after the end of `target`

        Parameters
        ----------
        target
            past target TimeSeries
        future_covariates
            future covariates TimeSeries
        n
            prediciton length (only given for predictions)

        Returns
        -------
        Sequence[TimeSeries]
            future covariates including cyclic encoded time index
        """

        if n is None:  # training
            encode_ts = future_covariates if future_covariates is not None else target
        else:  # prediction
            if future_covariates is not None:
                encode_ts = future_covariates
            else:
                encode_ts = [_generate_index(start=ts.end_time() - ts.freq * (self.input_chunk_length - 1),
                                             length=self.input_chunk_length + max(n, self.output_chunk_length),
                                             freq=ts.freq) for ts in target]

        encoded_times = [
            datetime_attribute_timeseries(ts, 
                                          attribute=self.add_cyclic_encoder, 
                                          cyclic=True, 
                                          dtype=target[0].dtype) 
            for ts in encode_ts
        ]

        if future_covariates is None:
            future_covariates = encoded_times
        else:
            future_covariates = [fc.stack(et) for fc, et in zip(future_covariates, encoded_times)]

        return future_covariates

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        raise_if_not(isinstance(train_dataset, MixedCovariatesTrainingDataset),
                     'TFTModel requires a training dataset of type MixedCovariatesTrainingDataset.')

    def _build_inference_dataset(self,
                                 target: Sequence[TimeSeries],
                                 n: int,
                                 past_covariates: Optional[Sequence[TimeSeries]],
                                 future_covariates: Optional[Sequence[TimeSeries]]) -> MixedCovariatesInferenceDataset:

        if self.add_cyclic_encoder is not None:
            future_covariates = self._add_cyclic_encoder(target, future_covariates=future_covariates, n=n)

        return MixedCovariatesInferenceDataset(target_series=target,
                                               past_covariates=past_covariates,
                                               future_covariates=future_covariates,
                                               n=n,
                                               input_chunk_length=self.input_chunk_length,
                                               output_chunk_length=self.output_chunk_length)

    def _produce_train_output(self, input_batch: Tuple):
        return self.model(input_batch)

    def predict(self, n, *args, **kwargs):
        """
        since we have future covariates, the inference dataset for future input must be at least of length
        `output_chunk_length`. If not, we would have to step back which causes past input to be shorter than
        `input_chunk_length`.
        """
        if n >= self.output_chunk_length:
            return super().predict(n, *args, **kwargs)
        else:
            return super().predict(self.output_chunk_length, *args, **kwargs)[:n]

    @random_method
    def _produce_predict_output(self, x):
        if self.likelihood:
            output = self.model(x)
            return self.likelihood.sample(output)
        else:
            return self.model(x).squeeze(dim=-1)

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        """
        Feeds MixedCovariatesModel with input and output chunks of a MixedCovariatesSequentialDataset to farecast
        the next `n` target values per target variable.

        Parameters:
        ----------
        n
            prediction length
        input_batch
            (past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates)
        roll_size
            roll input arrays after every sequence by `roll_size`. Initially, `roll_size` is equivalent to
            `self.output_chunk_length`
        """
        dim_component = 2
        past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates \
            = input_batch

        n_targets = past_target.shape[dim_component]
        n_past_covs = past_covariates.shape[dim_component] if not past_covariates is None else 0
        n_future_covs = future_covariates.shape[dim_component] if not future_covariates is None else 0

        input_past = torch.cat(
            [ds for ds in [past_target, past_covariates, historic_future_covariates] if ds is not None],
            dim=dim_component
        )

        input_future = future_covariates[:, :roll_size, :] if future_covariates is not None else None

        out = self._produce_predict_output(
            x=(past_target, past_covariates, historic_future_covariates, input_future)
        )[:, self.first_prediction_index:, :]

        batch_prediction = [out[:, :roll_size, :]]
        prediction_length = roll_size

        while prediction_length < n:
            # we want the last prediction to end exactly at `n` into the future.
            # this means we may have to truncate the previous prediction and step
            # back the roll size for the last chunk
            if prediction_length + self.output_chunk_length > n:
                spillover_prediction_length = prediction_length + self.output_chunk_length - n
                roll_size -= spillover_prediction_length
                prediction_length -= spillover_prediction_length
                batch_prediction[-1] = batch_prediction[-1][:, :roll_size, :]

            # ==========> PAST INPUT <==========
            # roll over input series to contain latest target and covariate
            input_past = torch.roll(input_past, -roll_size, 1)

            # update target input to include next `roll_size` predictions
            if self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, :n_targets] = out[:, :roll_size, :]
            else:
                input_past[:, :, :n_targets] = out[:, -self.input_chunk_length:, :]

            # set left and right boundaries for extracting future elements
            if self.input_chunk_length >= roll_size:
                left_past, right_past = prediction_length - roll_size, prediction_length
            else:
                left_past, right_past = prediction_length - self.input_chunk_length, prediction_length

            # update past covariates to include next `roll_size` future past covariates elements
            if n_past_covs and self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, n_targets:n_targets + n_past_covs] = (
                    future_past_covariates[:, left_past:right_past, :]
                )
            elif n_past_covs:
                input_past[:, :, n_targets:n_targets + n_past_covs] = (
                    future_past_covariates[:, left_past:right_past, :]
                )

            # update historic future covariates to include next `roll_size` future covariates elements
            if n_future_covs and self.input_chunk_length >= roll_size:
                input_past[:, -roll_size:, n_targets + n_past_covs:] = (
                    future_covariates[:, left_past:right_past, :]
                )
            elif n_future_covs:
                input_past[:, :, n_targets + n_past_covs:] = (
                    future_covariates[:, left_past:right_past, :]
                )

            # ==========> FUTURE INPUT <==========
            left_future, right_future = right_past, right_past + self.output_chunk_length
            # update future covariates to include next `roll_size` future covariates elements
            if n_future_covs:
                input_future = future_covariates[:, left_future:right_future, :]

            # convert back into separate datasets
            input_past_target = input_past[:, :, :n_targets]
            input_past_covs = input_past[:, :, n_targets:n_targets + n_past_covs] if n_past_covs else None
            input_historic_future_covs = input_past[:, :, n_targets + n_past_covs:] if n_future_covs else None
            input_future_covs = input_future if n_future_covs else None

            # take only last part of the output sequence where needed
            out = self._produce_predict_output(
                x=(input_past_target, input_past_covs, input_historic_future_covs, input_future_covs)
            )[:, self.first_prediction_index:, :]

            batch_prediction.append(out)
            prediction_length += self.output_chunk_length

        # bring predictions into desired format and drop unnecessary values
        batch_prediction = torch.cat(batch_prediction, dim=1)
        batch_prediction = batch_prediction[:, :n, :]
        return batch_prediction
