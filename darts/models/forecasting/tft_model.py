"""
Temporal Fusion Transformer (TFT)
-------
"""

from typing import Union, List, Optional, Tuple, Dict, Sequence

from numpy.random import RandomState

import torch
from torch import nn

from darts import TimeSeries
from darts.utils.torch import random_method
from darts.logging import get_logger, raise_if_not
from darts.utils.likelihood_models import Likelihood
from darts.utils.data import TrainingDataset, MixedCovariatesSequentialDataset

from darts.models.forecasting.torch_forecasting_model import (
    MixedCovariatesTorchModel,
    TorchParametricProbabilisticForecastingModel
)

from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    AddNorm as _AddNorm,
    GateAddNorm as _GateAddNorm,
    GatedLinearUnit as _GatedLinearUnit,
    GatedResidualNetwork as _GatedResidualNetwork,
    InterpretableMultiHeadAttention as _InterpretableMultiHeadAttention,
    VariableSelectionNetwork as _VariableSelectionNetwork,
)

from pytorch_forecasting.models.nn.rnn import (
    LSTM as _LSTM,
)

from pytorch_forecasting.models.nn.embeddings import (
    MultiEmbedding as _MultiEmbedding
)

from darts.models.forecasting.tft_submodels import (
    # _AddNorm,
    # _GateAddNorm,
    # _GatedLinearUnit,
    # _GatedResidualNetwork,
    # _InterpretableMultiHeadAttention,
    # _VariableSelectionNetwork,
    # _LSTM,
    # _MultiEmbedding,
    QuantileLoss
)

logger = get_logger(__name__)

USE_POST_LSTM_GLU = False

MixedCovariatesTrainTensorType = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class _TFTModule(nn.Module):

    def __init__(self,
                 output_dim: Tuple[int, int],
                 input_chunk_length: int,
                 output_chunk_length: int,
                 variables_meta: Dict[str, Dict[str, List[str]]],
                 hidden_size: Union[int, List[int]] = 16,
                 lstm_layers: int = 2,
                 attention_head_size: int = 4,
                 hidden_continuous_size: int = 8,
                 dropout: float = 0.1,
                 loss_fn: nn.Module = QuantileLoss(),
                 likelihood: Optional[Likelihood] = None):

        """ PyTorch module implementing the TFT architecture from `this paper <https://arxiv.org/pdf/1912.09363.pdf>`_
        The implementation is built upon `pytorch-forecasting's TemporalFusionTransformer
        <https://pytorch-forecasting.readthedocs.io/en/latest/models.html>`_.

        Parameters
        ----------
        output_dim : Tuple[int, int]
            shape of output given by (n_targets, loss_size).
        input_chunk_length : int
            encoder length; number of past time steps that are fed to the forecasting module at prediction time.
        output_chunk_length : int
            decoder length; number of future time steps that are fed to the forecasting module at prediction time.
        variables_meta : Dict[str, Dict[str, List[str]]]
            dict containing variable enocder, decoder variable names for mapping tensors in `_TFTModule.forward()`
        hidden_size : int
            hidden state size of the TFT. It is the main hyper-parameter and common across the internal TFT architecture.
        lstm_layers : int
            number of layers for the Long Short Term Memory (LSTM) Encoder and Decoder (2 is a good default).
        attention_head_size : int
            number of attention heads (4 is a good default)
        hidden_continuous_size : int
            default for hidden size for processing continuous variables (similar to categorical embedding size)
        dropout : float
            Fraction of neurons afected by Dropout.
        loss_fn : nn.Module
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the `likelihood` parameter is specified.
            Per default the TFT uses quantile loss as defined in the original paper.
            Default: `darts.models.forecasting.tft_submodels.QuantileLoss()`.
        """

        super(_TFTModule, self).__init__()

        self.n_targets, self.loss_size = output_dim
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.variables_meta = variables_meta
        self.hidden_size = hidden_size
        self.hidden_continuous_size = hidden_continuous_size
        self.lstm_layers = lstm_layers
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.loss_fn = loss_fn
        self.likelihood = likelihood

        # general information on variable name endings:
        # _vsn: VariableSelectionNetwork
        # _grn: GatedResidualNetwork
        # _glu: GatedLinearUnit
        # _an: AddNorm
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
        self.static_context_initial_hidden_lstm = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
        )

        # for cell state of the lstm
        self.static_context_initial_cell_lstm = _GatedResidualNetwork(
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

        if not USE_POST_LSTM_GLU:
            # skip connection for lstm
            self.post_lstm_encoder_glu = _GatedLinearUnit(self.hidden_size, dropout=self.dropout)
            self.post_lstm_decoder_glu = self.post_lstm_encoder_glu
            self.post_lstm_encoder_an = _AddNorm(self.hidden_size)
            self.post_lstm_decoder_an = self.post_lstm_encoder_an
        else:
            self.post_lstm_gan = _GateAddNorm(
                input_size=self.hidden_size,
                dropout=self.dropout
            )

        # static enrichment and processing past LSTM
        self.static_enrichment = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            context_size=self.hidden_size,
        )

        # attention for long-range processing
        self.multihead_attn = _InterpretableMultiHeadAttention(
            d_model=self.hidden_size, n_head=self.attention_head_size, dropout=self.dropout
        )
        self.post_attn_gan = _GateAddNorm(self.hidden_size, dropout=self.dropout)
        self.positionwise_feedforward_grn = _GatedResidualNetwork(
            self.hidden_size, self.hidden_size, self.hidden_size, dropout=self.dropout
        )

        # output processing -> no dropout at this late stage
        self.pre_output_gan = _GateAddNorm(self.hidden_size, dropout=None)

        self.output_layer = \
            nn.ModuleList([nn.Linear(self.hidden_size, self.loss_size) for _ in range(self.n_targets)])

    @property
    def reals(self) -> List[str]:
        """
        List of all continuous variables in model
        """
        return self.variables_meta['model_config']['reals_input']

    @property
    def categoricals(self) -> List[str]:
        """
        List of all categorical variables in model
        """
        # TODO: (Darts) we might want to include categorical variables in the future?
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
        """
        List of all static variables in model
        """
        # TODO: (Darts: dbader) we might want to include categorical variables in the future?
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
    def expand_static_context(context: torch.Tensor, timesteps: int) -> torch.Tensor:
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, timesteps, -1)

    def get_attention_mask(self,
                           encoder_lengths: torch.Tensor,
                           decoder_length: int,
                           device: str):
        """
        Returns causal mask to apply for self-attention layer that acts on future input only.
        """
        # indices to which is attended
        attend_step = torch.arange(decoder_length, device=device)
        # indices for which is predicted
        predict_step = torch.arange(0, decoder_length, device=device)[:, None]
        # do not attend to steps to self or after prediction
        decoder_mask = attend_step >= predict_step
        # do not attend to steps where data is padded
        encoder_mask = self.create_mask(encoder_lengths.max(), encoder_lengths)
        # combine masks along attended time - first encoder and then decoder

        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(encoder_lengths.shape[0], -1, -1),
            ),
            dim=2,
        )
        return mask

    @staticmethod
    def create_mask(size: int, lengths: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """
        Create boolean masks of shape len(lenghts) x size.

        An entry at (i, j) is True if lengths[i] > j.

        Args:
            size (int): size of second dimension
            lengths (torch.LongTensor): tensor of lengths
            inverse (bool, optional): If true, boolean mask is inverted. Defaults to False.

        Returns:
            torch.Tensor: mask
        """
        steps = torch.arange(size, device=lengths.device).unsqueeze(0)
        geq_steps = torch.ge(steps, lengths.unsqueeze(-1))
        return geq_steps if not inverse else ~geq_steps

    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        input dimensions: (n_samples, n_time_steps, n_variables)
        """

        dim_samples, dim_time, dim_variable, dim_loss = (0, 1, 2, 3)
        past_target, past_covariates, historic_future_covariates, future_covariates = x

        # TODO: impelement static covariates
        static_covariates = None

        # when there are no future_covariates we will need to create a zero tensor of
        # shape (n_samples, output_chunk_length, 1) in _TFTModule.forward()
        if future_covariates is None:
            historic_future_covariates = torch.zeros((past_target.shape[dim_samples], self.input_chunk_length, 1),
                                                     dtype=past_target.dtype,
                                                     device=past_target.device)
            future_covariates = torch.zeros((past_target.shape[dim_samples], self.output_chunk_length, 1),
                                            dtype=past_target.dtype,
                                            device=past_target.device)

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

        # batch_size = x.shape[0]
        encoder_lengths = torch.tensor(
            [self.input_chunk_length] * x_cont_past.shape[dim_samples], 
            dtype=past_target.dtype,
            device=past_target.device
        )
        decoder_lengths = torch.tensor(
            [self.output_chunk_length] * x_cont_future.shape[dim_samples], 
            dtype=past_target.dtype,
            device=past_target.device
        )

        timesteps = self.input_chunk_length + self.output_chunk_length
        encoder_length = self.input_chunk_length
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
            timesteps=timesteps
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
        input_hidden = self.static_context_initial_hidden_lstm(static_embedding).expand(
            self.lstm_layers, -1, -1
        )
        input_cell = self.static_context_initial_cell_lstm(static_embedding).expand(
            self.lstm_layers, -1, -1
        )

        # run local encoder
        encoder_output, (hidden, cell) = self.lstm_encoder(
            x=embeddings_varying_encoder,
            hx=(input_hidden, input_cell),
            lengths=encoder_lengths,
            enforce_sorted=False
        )

        # run local decoder
        decoder_output, _ = self.lstm_decoder(
            x=embeddings_varying_decoder,
            hx=(hidden, cell),
            lengths=decoder_lengths,
            enforce_sorted=False,
        )

        # TODO: (Darts: dbader) why not simply _GateAddNorm?
        if not USE_POST_LSTM_GLU:
            # skip connection over lstm
            lstm_output_encoder = self.post_lstm_encoder_glu(encoder_output)
            lstm_output_encoder = self.post_lstm_encoder_an(
                x=lstm_output_encoder,
                skip=embeddings_varying_encoder
            )

            lstm_output_decoder = self.post_lstm_decoder_glu(decoder_output)
            lstm_output_decoder = self.post_lstm_decoder_an(
                x=lstm_output_decoder,
                skip=embeddings_varying_decoder
            )
            lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)
        else:
            lstm_out = torch.cat([encoder_output, decoder_output], dim=1)
            input_embeddings = torch.cat([embeddings_varying_encoder, embeddings_varying_decoder], dim=1)
            lstm_output = self.post_lstm_gan(
                x=lstm_out,
                skip=input_embeddings
            )

        # static enrichment
        static_context_enriched = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment(
            x=lstm_output,
            context=self.expand_static_context(context=static_context_enriched, timesteps=timesteps)
        )

        # Attention
        attn_output, attn_output_weights = self.multihead_attn(
            q=attn_input[:, encoder_length:],  # query only for predictions
            k=attn_input,
            v=attn_input,
            mask=self.get_attention_mask(
                encoder_lengths=encoder_lengths,
                decoder_length=timesteps - encoder_length,
                device=past_target.device
            ),
        )

        # skip connection over attention
        attn_output = self.post_attn_gan(
            x=attn_output,
            skip=attn_input[:, encoder_length:]
        )

        output = self.positionwise_feedforward_grn(
            x=attn_output,
            context=None
        )

        # skip connection over temporal fusion decoder from LSTM post _GateAddNorm
        output = self.pre_output_gan(
            x=output,
            skip=lstm_output[:, encoder_length:]
        )

        # generate output for n_targets and loss_size elements for loss evaluation
        output = [output_layer(output) for output_layer in self.output_layer]

        # stack output
        if self.likelihood is not None or self.loss_size == 1 and self.n_targets > 1:
            # returns shape (n_samples, n_timesteps, n_likelihood_params/n_targets)
            output = torch.cat(output, dim=dim_variable)
        elif self.loss_size == 1 and self.n_targets == 1:
            # returns shape (n_samples, n_timesteps, 1) for univariate
            output = output[0]
        else:
            # loss_size > 1 for losses such as QuantileLoss
            # returns shape (n_samples, n_timesteps, n_targets, n_losses)
            output = torch.cat([out_i.unsqueeze(dim_variable) for out_i in output], dim=dim_variable)

        # TODO: (Darts) remember this in case we want to output interpretation
        # return self.to_network_output(
        #     prediction=self.transform_output(output, target_scale=x["target_scale"]),
        #     attention=attn_output_weights,
        #     static_variables=static_covariate_var,
        #     encoder_variables=encoder_sparse_weights,
        #     decoder_variables=decoder_sparse_weights,
        #     decoder_lengths=decoder_lengths,
        #     encoder_lengths=encoder_lengths,
        # )

        return output


class TFTModel(TorchParametricProbabilisticForecastingModel, MixedCovariatesTorchModel):
    @random_method
    def __init__(self,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1,
                 hidden_size: Union[int, List[int]] = 16,
                 lstm_layers: int = 1,
                 attention_head_size: int = 4,
                 dropout: float = 0.1,
                 loss_fn: Optional[nn.Module] = QuantileLoss(),
                 hidden_continuous_size: int = 8,
                 likelihood: Optional[Likelihood] = None,
                 max_samples_per_ts: Optional[int] = None,
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs
                 ):
        """Temporal Fusion Transformers (TFT) for Interpretable Time Series Forecasting.

        This is an implementation of the TFT architecture, as outlined in this paper:
        https://arxiv.org/pdf/1912.09363.pdf.

        The internal TFT architecture uses a majority of `pytorch-forecasting's TemporalFusionTransformer
        <https://pytorch-forecasting.readthedocs.io/en/latest/models.html>`_ implementation.

        This model supports mixed covariates (includes past covariates known for `input_chunk_length`
        points before prediction time and future covariates known for `output_chunk_length` after prediction time).

        The TFT applies multi-head attention queries on future inputs. Without future covariates, the model performs
        much worse. Consider supplying a cyclic encoding of the time index as future_covariates to the `fit()` and
        `predict()` methods. See :meth:`darts.utils.timeseries_generation.datetime_attribute_timeseries()
        <darts.utils.timeseries_generation.datetime_attribute_timeseries>`.

        Parameters
        ----------
        input_chunk_length : int
            encoder length; number of past time steps that are fed to the forecasting module at prediction time.
        output_chunk_length : int
            decoder length; number of future time steps that are fed to the forecasting module at prediction time.
        hidden_size : int
            hidden state size of the TFT. It is the main hyper-parameter and common across the internal TFT architecture.
        lstm_layers : int
            number of layers for the Long Short Term Memory (LSTM) Encoder and Decoder (2 is a good default).
        attention_head_size : int
            number of attention heads (4 is a good default)
        dropout : float
            Fraction of neurons afected by Dropout.
        loss_fn : nn.Module
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the `likelihood` parameter is specified.
            Per default the TFT uses quantile loss as defined in the original paper.
            Default: :meth:`darts.models.forecasting.tft_submodels.QuantileLoss()
            <darts.models.forecasting.tft_submodels.QuantileLoss>`
        hidden_continuous_size : int
            default for hidden size for processing continuous variables (similar to categorical embedding size)
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
        likelihood
            Optionally, the likelihood model to be used for probabilistic forecasts.
            If no likelihood model is provided, forecasts will be deterministic.
        """
        kwargs['loss_fn'] = loss_fn
        kwargs['input_chunk_length'] = input_chunk_length
        kwargs['output_chunk_length'] = output_chunk_length
        super().__init__(likelihood=likelihood, **kwargs)

        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.loss_fn = loss_fn
        # QuantileLoss requires one output per quantile, MSELoss requires 1
        self.loss_size = 1 if not isinstance(loss_fn, QuantileLoss) else len(loss_fn.quantiles)
        self.attention_head_size = attention_head_size
        self.hidden_continuous_size = hidden_continuous_size
        self.likelihood = likelihood
        self.max_sample_per_ts = max_samples_per_ts
        self.output_dim: Tuple[int, int] = None

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
        
        for categoricals (if we want to use it in the future) we would need embeddings

        `variable_meta` is used in TFT to access specific variables
        """
        past_target, past_covariate, historic_future_covariate, future_covariate, future_target = train_sample

        static_covariates = None  # placeholder for future

        self.output_dim = (future_target.shape[1], self.loss_size) if self.likelihood is None else \
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

        # TODO: we might want to include cyclic encoding variable here (which will need to be added to
        #  train and predict datasets)?
        # when there are no future_covariates we will need to create a zero tensor of
        # shape (n_samples, output_chunk_length, 1) in _TFTModule.forward()
        if future_covariate is None:
            dummy_name = 'future_covariate_0'
            variables_meta['input']['historic_future_covariate'] = [dummy_name]
            variables_meta['input']['future_covariate'] = [dummy_name]

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
            attention_head_size=self.attention_head_size,
            hidden_continuous_size=self.hidden_continuous_size,
            likelihood=self.likelihood
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
                                                max_samples_per_ts=self.max_sample_per_ts)

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        raise_if_not(isinstance(train_dataset, MixedCovariatesSequentialDataset),
                     'TFTModel requires a training dataset of type MixedCovariatesSequentialDataset.')

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
        output = self.model(x)
        if self.likelihood is not None:
            return self.likelihood.sample(output)
        elif isinstance(self.loss_fn, QuantileLoss):
            p50_index = self.loss_fn.quantiles.index(0.5)
            return output[..., p50_index]
        else:
            return output

    def _get_batch_prediction(self, n: int, input_batch: Tuple, roll_size: int) -> torch.Tensor:
        """
        Feeds MixedCovariatesModel with input and output chunks of a MixedCovariatesSequentialDataset to farecast
        the next `n` target values per target variable.

        Parameters:
        ----------
        input_batch
            (past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates)
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

        input_future = torch.clone(future_covariates[:, :roll_size, :]) if future_covariates is not None else None

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
