"""
Temporal Fusion Transformer (TFT)
-------
"""

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import LSTM as _LSTM

from darts import TimeSeries
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.models.components import glu_variants, layer_norm_variants
from darts.models.components.glu_variants import GLU_FFN
from darts.models.forecasting.pl_forecasting_module import (
    PLMixedCovariatesModule,
    io_processor,
)
from darts.models.forecasting.tft_submodels import (
    _GateAddNorm,
    _GatedResidualNetwork,
    _InterpretableMultiHeadAttention,
    _MultiEmbedding,
    _VariableSelectionNetwork,
    get_embedding_size,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data import (
    MixedCovariatesSequentialDataset,
    MixedCovariatesTrainingDataset,
    TrainingDataset,
)
from darts.utils.likelihood_models import Likelihood, QuantileRegression

logger = get_logger(__name__)

MixedCovariatesTrainTensorType = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class _TFTModule(PLMixedCovariatesModule):
    def __init__(
        self,
        output_dim: tuple[int, int],
        variables_meta: dict[str, dict[str, list[str]]],
        num_static_components: int,
        hidden_size: Union[int, list[int]],
        lstm_layers: int,
        num_attention_heads: int,
        full_attention: bool,
        feed_forward: str,
        hidden_continuous_size: int,
        categorical_embedding_sizes: dict[str, tuple[int, int]],
        dropout: float,
        add_relative_index: bool,
        norm_type: Union[str, nn.Module],
        **kwargs,
    ):
        """PyTorch module implementing the TFT architecture from `this paper <https://arxiv.org/pdf/1912.09363.pdf>`_
        The implementation is built upon `pytorch-forecasting's TemporalFusionTransformer
        <https://pytorch-forecasting.readthedocs.io/en/latest/models.html>`_.

        Parameters
        ----------
        output_dim : Tuple[int, int]
            shape of output given by (n_targets, loss_size). (loss_size corresponds to nr_params in other models).
        variables_meta : Dict[str, Dict[str, List[str]]]
            dict containing variable encoder, decoder variable names for mapping tensors in `_TFTModule.forward()`
        num_static_components
            the number of static components (not variables) of the input target series. This is either equal to the
            number of target components or 1.
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
        feed_forward
            Set the feedforward network block. default `GatedResidualNetwork` or one of the  glu variant.
            Defaults to `GatedResidualNetwork`.
        hidden_continuous_size : int
            default for hidden size for processing continuous variables.
        categorical_embedding_sizes : dict
            A dictionary containing embedding sizes for categorical static covariates. The keys are the column names
            of the categorical static covariates. The values are tuples of integers with
            `(number of unique categories, embedding size)`. For example `{"some_column": (64, 8)}`.
            Note that `TorchForecastingModels` can only handle numeric data. Consider transforming/encoding your data
            with `darts.dataprocessing.transformers.static_covariates_transformer.StaticCovariatesTransformer`.
        dropout : float
            Fraction of neurons affected by Dropout.
        add_relative_index : bool
            Whether to add positional values to future covariates. Defaults to `False`.
            This allows to use the TFTModel without having to pass future_covariates to `fit()` and `train()`.
            It gives a value to the position of each step from input and output chunk relative to the prediction
            point. The values are normalized with `input_chunk_length`.
        likelihood
            The likelihood model to be used for probabilistic forecasts. By default, the TFT uses
            a ``QuantileRegression`` likelihood.
        norm_type: str | nn.Module
            The type of LayerNorm variant to use.
        **kwargs
            all parameters required for :class:`darts.models.forecasting.pl_forecasting_module.PLForecastingModule`
            base class.
        """

        super().__init__(**kwargs)

        self.n_targets, self.loss_size = output_dim
        self.variables_meta = variables_meta
        self.num_static_components = num_static_components
        self.hidden_size = hidden_size
        self.hidden_continuous_size = hidden_continuous_size
        self.categorical_embedding_sizes = categorical_embedding_sizes
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.full_attention = full_attention
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.add_relative_index = add_relative_index

        if isinstance(norm_type, str):
            try:
                self.layer_norm = getattr(layer_norm_variants, norm_type)
            except AttributeError:
                raise_log(
                    AttributeError("please provide a valid layer norm type"),
                )
        else:
            self.layer_norm = norm_type

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
        # embeddings
        self.input_embeddings = _MultiEmbedding(
            embedding_sizes=categorical_embedding_sizes,
            variable_names=self.categorical_static_variables,
        )

        # continuous variable processing
        self.prescalers_linear = {
            name: nn.Linear(
                (
                    1
                    if name not in self.numeric_static_variables
                    else self.num_static_components
                ),
                self.hidden_continuous_size,
            )
            for name in self.reals
        }

        # static (categorical and numerical) variables
        static_input_sizes = {
            name: self.input_embeddings.output_size[name]
            for name in self.categorical_static_variables
        }
        static_input_sizes.update({
            name: self.hidden_continuous_size for name in self.numeric_static_variables
        })

        self.static_covariates_vsn = _VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={
                name: True for name in self.categorical_static_variables
            },
            dropout=self.dropout,
            prescalers=self.prescalers_linear,
            single_variable_grns={},
            context_size=None,  # no context for static variables
            layer_norm=self.layer_norm,
        )

        # variable selection for encoder and decoder
        encoder_input_sizes = {
            name: self.hidden_continuous_size for name in self.encoder_variables
        }

        decoder_input_sizes = {
            name: self.hidden_continuous_size for name in self.decoder_variables
        }

        self.encoder_vsn = _VariableSelectionNetwork(
            input_sizes=encoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},  # this would be required for non-static categorical inputs
            dropout=self.dropout,
            context_size=self.hidden_size,
            prescalers=self.prescalers_linear,
            single_variable_grns={},
            layer_norm=self.layer_norm,
        )

        self.decoder_vsn = _VariableSelectionNetwork(
            input_sizes=decoder_input_sizes,
            hidden_size=self.hidden_size,
            input_embedding_flags={},  # this would be required for non-static categorical inputs
            dropout=self.dropout,
            context_size=self.hidden_size,
            prescalers=self.prescalers_linear,
            single_variable_grns={},
            layer_norm=self.layer_norm,
        )

        # static encoders
        # for variable selection
        self.static_context_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )

        # for hidden state of the lstm
        self.static_context_hidden_encoder_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )

        # for cell state of the lstm
        self.static_context_cell_encoder_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
        )

        # for post lstm static enrichment
        self.static_context_enrichment = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
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
            input_size=self.hidden_size, dropout=dropout, layer_norm=self.layer_norm
        )

        # static enrichment and processing past LSTM
        self.static_enrichment_grn = _GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            context_size=self.hidden_size,
            layer_norm=self.layer_norm,
        )

        # attention for long-range processing
        self.multihead_attn = _InterpretableMultiHeadAttention(
            d_model=self.hidden_size,
            n_head=self.num_attention_heads,
            dropout=self.dropout,
        )
        self.post_attn_gan = _GateAddNorm(
            self.hidden_size, dropout=self.dropout, layer_norm=self.layer_norm
        )

        if self.feed_forward == "GatedResidualNetwork":
            self.feed_forward_block = _GatedResidualNetwork(
                self.hidden_size,
                self.hidden_size,
                self.hidden_size,
                dropout=self.dropout,
                layer_norm=self.layer_norm,
            )
        else:
            raise_if_not(
                self.feed_forward in GLU_FFN,
                f"'{self.feed_forward}' is not in {GLU_FFN + ['GatedResidualNetwork']}",
            )
            # use glu variant feedforward layers
            # 4 is a commonly used feedforward multiplier
            self.feed_forward_block = getattr(glu_variants, self.feed_forward)(
                d_model=self.hidden_size, d_ff=self.hidden_size * 4, dropout=dropout
            )

        # output processing -> no dropout at this late stage
        self.pre_output_gan = _GateAddNorm(
            self.hidden_size, dropout=None, layer_norm=self.layer_norm
        )

        self.output_layer = nn.Linear(self.hidden_size, self.n_targets * self.loss_size)

        self._attn_out_weights = None
        self._static_covariate_var = None
        self._encoder_sparse_weights = None
        self._decoder_sparse_weights = None

    @property
    def reals(self) -> list[str]:
        """
        List of all continuous variables in model
        """
        return self.variables_meta["model_config"]["reals_input"]

    @property
    def static_variables(self) -> list[str]:
        """
        List of all static variables in model
        """
        return self.variables_meta["model_config"]["static_input"]

    @property
    def numeric_static_variables(self) -> list[str]:
        """
        List of numeric static variables in model
        """
        return self.variables_meta["model_config"]["static_input_numeric"]

    @property
    def categorical_static_variables(self) -> list[str]:
        """
        List of categorical static variables in model
        """
        return self.variables_meta["model_config"]["static_input_categorical"]

    @property
    def encoder_variables(self) -> list[str]:
        """
        List of all encoder variables in model (excluding static variables)
        """
        return self.variables_meta["model_config"]["time_varying_encoder_input"]

    @property
    def decoder_variables(self) -> list[str]:
        """
        List of all decoder variables in model (excluding static variables)
        """
        return self.variables_meta["model_config"]["time_varying_decoder_input"]

    @staticmethod
    def expand_static_context(context: torch.Tensor, time_steps: int) -> torch.Tensor:
        """
        add time dimension to static context
        """
        return context[:, None].expand(-1, time_steps, -1)

    @staticmethod
    def get_relative_index(
        encoder_length: int,
        decoder_length: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns scaled time index relative to prediction point.
        """
        index = torch.arange(
            encoder_length + decoder_length, dtype=dtype, device=device
        )
        prediction_index = encoder_length - 1
        index[:encoder_length] = index[:encoder_length] / prediction_index
        index[encoder_length:] = index[encoder_length:] / prediction_index
        return index.reshape(1, len(index), 1).repeat(batch_size, 1, 1)

    @staticmethod
    def get_attention_mask_full(
        time_steps: int, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """
        Returns causal mask to apply for self-attention layer.
        """
        eye = torch.eye(time_steps, dtype=dtype, device=device)
        mask = torch.cumsum(eye.unsqueeze(0).repeat(batch_size, 1, 1), dim=1)
        return mask < 1

    @staticmethod
    def get_attention_mask_future(
        encoder_length: int,
        decoder_length: int,
        batch_size: int,
        device: str,
        full_attention: bool,
    ) -> torch.Tensor:
        """
        Returns causal mask to apply for self-attention layer that acts on future input only.
        The model will attend to all `False` values.
        """
        if full_attention:
            # attend to entire past and future input
            decoder_mask = torch.zeros(
                (decoder_length, decoder_length), dtype=torch.bool, device=device
            )
        else:
            # attend only to past steps relative to forecasting step in the future
            # indices to which is attended
            attend_step = torch.arange(decoder_length, device=device)
            # indices for which is predicted
            predict_step = torch.arange(0, decoder_length, device=device)[:, None]
            # do not attend to steps to self or after prediction
            decoder_mask = attend_step >= predict_step
        # attend to all past input
        encoder_mask = torch.zeros(
            batch_size, encoder_length, dtype=torch.bool, device=device
        )
        # combine masks along attended time - first encoder and then decoder

        mask = torch.cat(
            (
                encoder_mask.unsqueeze(1).expand(-1, decoder_length, -1),
                decoder_mask.unsqueeze(0).expand(batch_size, -1, -1),
            ),
            dim=2,
        )
        return mask

    @io_processor
    def forward(
        self, x_in: tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """TFT model forward pass.

        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
            is the output/future chunk. Input dimensions are `(n_samples, n_time_steps, n_variables)`

        Returns
        -------
        torch.Tensor
            the output tensor
        """
        x_cont_past, x_cont_future, x_static = x_in
        dim_samples, dim_time, dim_variable = 0, 1, 2
        device = x_in[0].device

        batch_size = x_cont_past.shape[dim_samples]
        encoder_length = self.input_chunk_length
        decoder_length = self.output_chunk_length
        time_steps = encoder_length + decoder_length

        # avoid unnecessary regeneration of attention mask
        if batch_size != self.batch_size_last:
            self.attention_mask = self.get_attention_mask_future(
                encoder_length=encoder_length,
                decoder_length=decoder_length,
                batch_size=batch_size,
                device=device,
                full_attention=self.full_attention,
            )
            if self.add_relative_index:
                self.relative_index = self.get_relative_index(
                    encoder_length=encoder_length,
                    decoder_length=decoder_length,
                    batch_size=batch_size,
                    device=device,
                    dtype=x_cont_past.dtype,
                )

            self.batch_size_last = batch_size

        if self.add_relative_index:
            x_cont_past = torch.cat(
                [
                    ts[:, :encoder_length, :]
                    for ts in [x_cont_past, self.relative_index]
                    if ts is not None
                ],
                dim=dim_variable,
            )
            x_cont_future = torch.cat(
                [
                    ts[:, -decoder_length:, :]
                    for ts in [x_cont_future, self.relative_index]
                    if ts is not None
                ],
                dim=dim_variable,
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
        if self.static_variables:
            # categorical static covariate embeddings
            if self.categorical_static_variables:
                static_embedding = self.input_embeddings(
                    torch.cat(
                        [
                            x_static[:, :, idx]
                            for idx, name in enumerate(self.static_variables)
                            if name in self.categorical_static_variables
                        ],
                        dim=1,
                    ).int()
                )
            else:
                static_embedding = {}
            # add numerical static covariates
            static_embedding.update({
                name: x_static[:, :, idx]
                for idx, name in enumerate(self.static_variables)
                if name in self.numeric_static_variables
            })
            static_embedding, static_covariate_var = self.static_covariates_vsn(
                static_embedding
            )
        else:
            static_embedding = torch.zeros(
                (x_cont_past.shape[0], self.hidden_size),
                dtype=x_cont_past.dtype,
                device=device,
            )
            static_covariate_var = None

        static_context_expanded = self.expand_static_context(
            context=self.static_context_grn(static_embedding), time_steps=time_steps
        )

        embeddings_varying_encoder = {
            name: input_vectors_past[name] for name in self.encoder_variables
        }
        embeddings_varying_encoder, encoder_sparse_weights = self.encoder_vsn(
            x=embeddings_varying_encoder,
            context=static_context_expanded[:, :encoder_length],
        )

        embeddings_varying_decoder = {
            name: input_vectors_future[name] for name in self.decoder_variables
        }
        embeddings_varying_decoder, decoder_sparse_weights = self.decoder_vsn(
            x=embeddings_varying_decoder,
            context=static_context_expanded[:, encoder_length:],
        )

        # LSTM
        # calculate initial state
        input_hidden = (
            self.static_context_hidden_encoder_grn(static_embedding)
            .expand(self.lstm_layers, -1, -1)
            .contiguous()
        )
        input_cell = (
            self.static_context_cell_encoder_grn(static_embedding)
            .expand(self.lstm_layers, -1, -1)
            .contiguous()
        )

        # run local lstm encoder
        encoder_out, (hidden, cell) = self.lstm_encoder(
            input=embeddings_varying_encoder, hx=(input_hidden, input_cell)
        )

        # run local lstm decoder
        decoder_out, _ = self.lstm_decoder(
            input=embeddings_varying_decoder, hx=(hidden, cell)
        )

        lstm_layer = torch.cat([encoder_out, decoder_out], dim=dim_time)
        input_embeddings = torch.cat(
            [embeddings_varying_encoder, embeddings_varying_decoder], dim=dim_time
        )

        # post lstm GateAddNorm
        lstm_out = self.post_lstm_gan(x=lstm_layer, skip=input_embeddings)

        # static enrichment
        static_context_enriched = self.static_context_enrichment(static_embedding)
        attn_input = self.static_enrichment_grn(
            x=lstm_out,
            context=self.expand_static_context(
                context=static_context_enriched, time_steps=time_steps
            ),
        )

        # multi-head attention
        attn_out, attn_out_weights = self.multihead_attn(
            q=attn_input[:, encoder_length:],
            k=attn_input,
            v=attn_input,
            mask=self.attention_mask,
        )

        # skip connection over attention
        attn_out = self.post_attn_gan(
            x=attn_out,
            skip=attn_input[:, encoder_length:],
        )

        # feed-forward
        out = self.feed_forward_block(x=attn_out)

        # skip connection over temporal fusion decoder from LSTM post _GateAddNorm
        out = self.pre_output_gan(
            x=out,
            skip=lstm_out[:, encoder_length:],
        )

        # generate output for n_targets and loss_size elements for loss evaluation
        out = self.output_layer(out)
        out = out.view(
            batch_size, self.output_chunk_length, self.n_targets, self.loss_size
        )
        self._attn_out_weights = attn_out_weights
        self._static_covariate_var = static_covariate_var
        self._encoder_sparse_weights = encoder_sparse_weights
        self._decoder_sparse_weights = decoder_sparse_weights
        return out


class TFTModel(MixedCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        hidden_size: Union[int, list[int]] = 16,
        lstm_layers: int = 1,
        num_attention_heads: int = 4,
        full_attention: bool = False,
        feed_forward: str = "GatedResidualNetwork",
        dropout: float = 0.1,
        hidden_continuous_size: int = 8,
        categorical_embedding_sizes: Optional[
            dict[str, Union[int, tuple[int, int]]]
        ] = None,
        add_relative_index: bool = False,
        loss_fn: Optional[nn.Module] = None,
        likelihood: Optional[Likelihood] = None,
        norm_type: Union[str, nn.Module] = "LayerNorm",
        use_static_covariates: bool = True,
        **kwargs,
    ):
        """Temporal Fusion Transformers (TFT) for Interpretable Time Series Forecasting.

        This is an implementation of the TFT architecture, as outlined in [1]_.

        The internal sub models are adopted from `pytorch-forecasting's TemporalFusionTransformer
        <https://pytorch-forecasting.readthedocs.io/en/latest/models.html>`_ implementation.

        This model supports past covariates (known for `input_chunk_length` points before prediction time),
        future covariates (known for `output_chunk_length` points after prediction time), static covariates,
        as well as probabilistic forecasting.

        The TFT applies multi-head attention queries on future inputs from mandatory ``future_covariates``.
        Specifying future encoders with ``add_encoders`` (read below) can automatically generate future covariates
        and allows to use the model without having to pass any ``future_covariates`` to :func:`fit()` and
        :func:`predict()`.

        By default, this model uses the ``QuantileRegression`` likelihood, which means that its forecasts are
        probabilistic; it is recommended to call :func`predict()` with ``num_samples >> 1`` to get meaningful results.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it).
            Also called: Encoder length
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
            from future covariates to use as a model input (if the model supports future covariates). It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
            using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents
            auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
            the model from using future values of past and / or future covariates for prediction (depending on the
            model's covariate support).
            Also called: Decoder length
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`).
        hidden_size
            Hidden state size of the TFT. It is the main hyper-parameter and common across the internal TFT
            architecture.
        lstm_layers
            Number of layers for the Long Short Term Memory (LSTM) Encoder and Decoder (1 is a good default).
        num_attention_heads
            Number of attention heads (4 is a good default)
        full_attention
            If ``False``, only attends to previous time steps in the decoder. If ``True`` attends to previous,
            current, and future time steps. Defaults to ``False``.
        feed_forward
            A feedforward network is a fully-connected layer with an activation. Can be one of the glu variant's
            FeedForward Network (FFN)[2]. The glu variant's FeedForward Network are a series of FFNs designed to work
            better with Transformer based models. Defaults to ``"GatedResidualNetwork"``. ["GLU", "Bilinear", "ReGLU",
            "GEGLU", "SwiGLU", "ReLU", "GELU"] or the TFT original FeedForward Network ["GatedResidualNetwork"].
        dropout
            Fraction of neurons affected by dropout. This is compatible with Monte Carlo dropout
            at inference time for model uncertainty estimation (enabled with ``mc_dropout=True`` at
            prediction time).
        hidden_continuous_size
            Default for hidden size for processing continuous variables
        categorical_embedding_sizes
            A dictionary used to construct embeddings for categorical static covariates. The keys are the column names
            of the categorical static covariates. Each value is either a single integer or a tuple of integers.
            For a single integer give the number of unique categories (n) of the corresponding variable. For example
            ``{"some_column": 64}``. The embedding size will be automatically determined by
            ``min(round(1.6 * n**0.56), 100)``.
            For a tuple of integers, give (number of unique categories, embedding size). For example
            ``{"some_column": (64, 8)}``.
            Note that ``TorchForecastingModels`` only support numeric data. Consider transforming/encoding your data
            with `darts.dataprocessing.transformers.static_covariates_transformer.StaticCovariatesTransformer`.
        add_relative_index
            Whether to add positional values to future covariates. Defaults to ``False``.
            This allows to use the TFTModel without having to pass future_covariates to :func:`fit()` and
            :func:`train()`. It gives a value to the position of each step from input and output chunk relative
            to the prediction point. The values are normalized with ``input_chunk_length``.
        loss_fn: nn.Module
            PyTorch loss function used for training. By default, the TFT model is probabilistic and uses a
            ``likelihood`` instead (``QuantileRegression``). To make the model deterministic, you can set the `
            `likelihood`` to None and give a ``loss_fn`` argument.
        likelihood
            The likelihood model to be used for probabilistic forecasts. By default, the TFT uses
            a ``QuantileRegression`` likelihood.
        norm_type: str | nn.Module
            The type of LayerNorm variant to use.  Default: ``LayerNorm``. Available options are
            ["LayerNorm", "RMSNorm", "LayerNormNoBias"], or provide a custom nn.Module.
        use_static_covariates
            Whether the model should use static covariate information in case the input `series` passed to ``fit()``
            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce
            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        torch_metrics
            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found
            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.
        optimizer_cls
            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{'lr': 1e-3}``
            for specifying a learning rate). Otherwise, the default values of the selected ``optimizer_cls``
            will be used. Default: ``None``.
        lr_scheduler_cls
            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds
            to using a constant learning rate. Default: ``None``.
        lr_scheduler_kwargs
            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.
        use_reversible_instance_norm
            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [3]_.
            It is only applied to the features of the target series and not the covariates.
        batch_size
            Number of time series (input and output sequences) used in each training pass. Default: ``32``.
        n_epochs
            Number of epochs over which to train the model. Default: ``100``.
        model_name
            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,
            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part
            of the name is formatted with the local date and time, while PID is the processed ID (preventing models
            spawned at the same time by different processes to share the same model_name). E.g.,
            ``"2021-06-14_09_53_32_torch_model_run_44607"``.
        work_dir
            Path of the working directory, where to save checkpoints and Tensorboard summaries.
            Default: current working directory.
        log_tensorboard
            If set, use Tensorboard to log the different parameters. The logs will be located in:
            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.
        nr_epochs_val_period
            Number of epochs to wait before evaluating the validation loss (if a validation
            ``TimeSeries`` is passed to the :func:`fit()` method). Default: ``1``.
        force_reset
            If set to ``True``, any previously-existing model with the same name will be reset (all checkpoints will
            be discarded). Default: ``False``.
        save_checkpoints
            Whether to automatically save the untrained model and checkpoints from training.
            To load the model from checkpoint, call :func:`MyModelClass.load_from_checkpoint()`, where
            :class:`MyModelClass` is the :class:`TorchForecastingModel` class that was used (such as :class:`TFTModel`,
            :class:`NBEATSModel`, etc.). If set to ``False``, the model can still be manually saved using
            :func:`save()` and loaded using :func:`load()`. Default: ``False``.
        add_encoders
            A large number of past and future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions that
            will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler` can be added to
            transform the generated covariates. This happens all under one hood and only needs to be specified at
            model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more about
            ``add_encoders``. Default: ``None``. An example showing some of ``add_encoders`` features:

            .. highlight:: python
            .. code-block:: python

                def encode_year(idx):
                    return (idx.year - 1950) / 50

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [encode_year]},
                    'transformer': Scaler(),
                    'tz': 'CET'
                }
            ..
        random_state
            Control the randomness of the weight's initialization. Check this
            `link <https://scikit-learn.org/stable/glossary.html#term-random_state>`_ for more details.
            Default: ``None``.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
            that performs the training, validation and prediction processes. These presets include automatic
            checkpointing, tensorboard logging, setting the torch device and more.
            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer
            object. Check the `PL Trainer documentation
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`_ for more information about the
            supported kwargs. Default: ``None``.
            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",
            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``
            dict:

            - ``{"accelerator": "cpu"}`` for CPU,
            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.

            For more info, see here:
            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and
            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus

            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts'
            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.
            The model will stop training early if the validation loss `val_loss` does not improve beyond
            specifications. For more information on callbacks, visit:
            `PyTorch Lightning Callbacks
            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`_

            .. highlight:: python
            .. code-block:: python

                from pytorch_lightning.callbacks.early_stopping import EarlyStopping

                # stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
                # a period of 5 epochs (`patience`)
                my_stopper = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    min_delta=0.05,
                    mode='min',
                )

                pl_trainer_kwargs={"callbacks": [my_stopper]}
            ..

            Note that you can also use a custom PyTorch Lightning Trainer for training and prediction with optional
            parameter ``trainer`` in :func:`fit()` and :func:`predict()`.
        show_warnings
            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of
            your forecasting use case. Default: ``False``.

        References
        ----------
        .. [1] https://arxiv.org/pdf/1912.09363.pdf
        .. [2] Shazeer, Noam, "GLU Variants Improve Transformer", 2020. arVix https://arxiv.org/abs/2002.05202.
        .. [3] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against
                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p

        Examples
        --------
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import TFTModel
        >>> series = WeatherDataset().load()
        >>> # predicting atmospheric pressure
        >>> target = series['p (mbar)'][:100]
        >>> # optionally, past observed rainfall (pretending to be unknown beyond index 100)
        >>> past_cov = series['rain (mm)'][:100]
        >>> # future temperatures (pretending this component is a forecast)
        >>> future_cov = series['T (degC)'][:106]
        >>> # by default, TFTModel is trained using a `QuantileRegression` making it a probabilistic forecasting model
        >>> model = TFTModel(
        >>>     input_chunk_length=6,
        >>>     output_chunk_length=6,
        >>>     n_epochs=5,
        >>> )
        >>> # future_covariates are mandatory for `TFTModel`
        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
        >>> # TFTModel is probabilistic by definition; using `num_samples >> 1` to generate probabilistic forecasts
        >>> pred = model.predict(6, num_samples=100)
        >>> # shape : (forecast horizon, components, num_samples)
        >>> pred.all_values().shape
        (6, 1, 100)
        >>> # showing the first 3 samples for each timestamp
        >>> pred.all_values()[:,:,:3]
        array([[[-0.06414202, -0.7188093 ,  0.52541292]],
               [[ 0.02928407, -0.40867163,  1.19650033]],
               [[ 0.77252372, -0.50859694,  0.360166  ]],
               [[ 0.9586113 ,  1.24147138, -0.01625545]],
               [[ 1.06863863,  0.2987822 , -0.69213369]],
               [[-0.83076568, -0.25780816, -0.28318784]]])

        .. note::
            `TFT example notebook <https://unit8co.github.io/darts/examples/13-TFT-examples.html>`_ presents
            techniques that can be used to improve the forecasts quality compared to this simple usage example.
        """
        model_kwargs = {key: val for key, val in self.model_params.items()}
        if likelihood is None and loss_fn is None:
            # This is the default if no loss information is provided
            model_kwargs["loss_fn"] = None
            model_kwargs["likelihood"] = QuantileRegression()

        super().__init__(**self._extract_torch_model_params(**model_kwargs))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)

        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.full_attention = full_attention
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.hidden_continuous_size = hidden_continuous_size
        self.categorical_embedding_sizes = (
            categorical_embedding_sizes
            if categorical_embedding_sizes is not None
            else {}
        )
        self.add_relative_index = add_relative_index
        self.output_dim: Optional[tuple[int, int]] = None
        self.norm_type = norm_type
        self._considers_static_covariates = use_static_covariates

    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """
        `train_sample` contains the following tensors:
            (past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates,
            future_target)

            each tensor has shape (n_timesteps, n_variables)
            - past/historic tensors have shape (input_chunk_length, n_variables)
            - future tensors have shape (output_chunk_length, n_variables)
            - static covariates have shape (component, static variable)

        Darts Interpretation of pytorch-forecasting's TimeSeriesDataSet:
            time_varying_knowns : future_covariates (including historic_future_covariates)
            time_varying_unknowns : past_targets, past_covariates

            time_varying_encoders : [past_targets, past_covariates, historic_future_covariates, future_covariates]
            time_varying_decoders : [historic_future_covariates, future_covariates]

        `variable_meta` is used in TFT to access specific variables
        """
        (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariates,
            future_target,
        ) = train_sample

        # add a covariate placeholder so that relative index will be included
        if self.add_relative_index:
            time_steps = self.input_chunk_length + self.output_chunk_length

            expand_future_covariate = np.arange(time_steps).reshape((time_steps, 1))

            historic_future_covariate = np.concatenate(
                [
                    ts[: self.input_chunk_length]
                    for ts in [historic_future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )
            future_covariate = np.concatenate(
                [
                    ts[-self.output_chunk_length :]
                    for ts in [future_covariate, expand_future_covariate]
                    if ts is not None
                ],
                axis=1,
            )

        self.output_dim = (
            (future_target.shape[1], 1)
            if self.likelihood is None
            else (future_target.shape[1], self.likelihood.num_parameters)
        )

        tensors = [
            past_target,
            past_covariate,
            historic_future_covariate,  # for time varying encoders
            future_covariate,
            future_target,  # for time varying decoders
            static_covariates,  # for static encoder
        ]
        type_names = [
            "past_target",
            "past_covariate",
            "historic_future_covariate",
            "future_covariate",
            "future_target",
            "static_covariate",
        ]
        variable_names = [
            "target",
            "past_covariate",
            "future_covariate",
            "future_covariate",
            "target",
            "static_covariate",
        ]

        variables_meta = {
            "input": {
                type_name: [f"{var_name}_{i}" for i in range(tensor.shape[1])]
                for type_name, var_name, tensor in zip(
                    type_names, variable_names, tensors
                )
                if tensor is not None
            },
            "model_config": {},
        }

        reals_input = []
        categorical_input = []
        time_varying_encoder_input = []
        time_varying_decoder_input = []
        static_input = []
        static_input_numeric = []
        static_input_categorical = []
        categorical_embedding_sizes = {}
        for input_var in type_names:
            if input_var in variables_meta["input"]:
                vars_meta = variables_meta["input"][input_var]
                if input_var in [
                    "past_target",
                    "past_covariate",
                    "historic_future_covariate",
                ]:
                    time_varying_encoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["future_covariate"]:
                    time_varying_decoder_input += vars_meta
                    reals_input += vars_meta
                elif input_var in ["static_covariate"]:
                    if (
                        self.static_covariates is None
                    ):  # when training with fit_from_dataset
                        static_cols = pd.Index([
                            i for i in range(static_covariates.shape[1])
                        ])
                    else:
                        static_cols = self.static_covariates.columns
                    numeric_mask = ~static_cols.isin(self.categorical_embedding_sizes)
                    for idx, (static_var, col_name, is_numeric) in enumerate(
                        zip(vars_meta, static_cols, numeric_mask)
                    ):
                        static_input.append(static_var)
                        if is_numeric:
                            static_input_numeric.append(static_var)
                            reals_input.append(static_var)
                        else:
                            # get embedding sizes for each categorical variable
                            embedding = self.categorical_embedding_sizes[col_name]
                            raise_if_not(
                                isinstance(embedding, (int, tuple)),
                                "Dict values of `categorical_embedding_sizes` must either be integers or tuples. Read "
                                "the TFTModel documentation for more information.",
                                logger,
                            )
                            if isinstance(embedding, int):
                                embedding = (embedding, get_embedding_size(n=embedding))
                            categorical_embedding_sizes[vars_meta[idx]] = embedding

                            static_input_categorical.append(static_var)
                            categorical_input.append(static_var)

        variables_meta["model_config"]["reals_input"] = list(dict.fromkeys(reals_input))
        variables_meta["model_config"]["categorical_input"] = list(
            dict.fromkeys(categorical_input)
        )
        variables_meta["model_config"]["time_varying_encoder_input"] = list(
            dict.fromkeys(time_varying_encoder_input)
        )
        variables_meta["model_config"]["time_varying_decoder_input"] = list(
            dict.fromkeys(time_varying_decoder_input)
        )
        variables_meta["model_config"]["static_input"] = list(
            dict.fromkeys(static_input)
        )
        variables_meta["model_config"]["static_input_numeric"] = list(
            dict.fromkeys(static_input_numeric)
        )
        variables_meta["model_config"]["static_input_categorical"] = list(
            dict.fromkeys(static_input_categorical)
        )

        n_static_components = (
            len(static_covariates) if static_covariates is not None else 0
        )

        self.categorical_embedding_sizes = categorical_embedding_sizes

        return _TFTModule(
            output_dim=self.output_dim,
            variables_meta=variables_meta,
            num_static_components=n_static_components,
            hidden_size=self.hidden_size,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
            num_attention_heads=self.num_attention_heads,
            full_attention=self.full_attention,
            feed_forward=self.feed_forward,
            hidden_continuous_size=self.hidden_continuous_size,
            categorical_embedding_sizes=self.categorical_embedding_sizes,
            add_relative_index=self.add_relative_index,
            norm_type=self.norm_type,
            **self.pl_module_params,
        )

    def _build_train_dataset(
        self,
        target: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        sample_weight: Optional[Sequence[TimeSeries]],
        max_samples_per_ts: Optional[int],
    ) -> MixedCovariatesSequentialDataset:
        raise_if(
            future_covariates is None and not self.add_relative_index,
            "TFTModel requires future covariates. The model applies multi-head attention queries on future "
            "inputs. Consider specifying a future encoder with `add_encoders` or setting `add_relative_index` "
            "to `True` at model creation (read TFT model docs for more information). "
            "These will automatically generate `future_covariates` from indexes.",
            logger,
        )

        return MixedCovariatesSequentialDataset(
            target_series=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            output_chunk_shift=self.output_chunk_shift,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=self.uses_static_covariates,
            sample_weight=sample_weight,
        )

    def _verify_train_dataset_type(self, train_dataset: TrainingDataset):
        raise_if_not(
            isinstance(train_dataset, MixedCovariatesTrainingDataset),
            "TFTModel requires a training dataset of type MixedCovariatesTrainingDataset.",
        )

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def supports_static_covariates(self) -> bool:
        return True
