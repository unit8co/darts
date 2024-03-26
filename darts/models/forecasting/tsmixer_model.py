"""
Time-Series Mixer (TSMixer)
-------
The inner layers (``nn.Modules``) and the ``TimeBatchNorm2d`` were provided by a PyTorch implementation
of TSMixer: https://github.com/ditschuk/pytorch-tsmixer

The License of pytorch-tsmixer v0.2.0 from https://github.com/ditschuk/pytorch-tsmixer/blob/main/LICENSE,
accessed Thursday, March 21st, 2024:
'The MIT License

Copyright 2023 Konstantin Ditschuneit

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the “Software”), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
'
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from functools import reduce
from typing import Optional, Tuple, Union

import torch
from torch import nn

from darts.logging import get_logger, raise_if_not, raise_log
from darts.models.components import layer_norm_variants
from darts.models.forecasting.pl_forecasting_module import (
    PLMixedCovariatesModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.torch import MonteCarloDropout

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

logger = get_logger(__name__)

ACTIVATIONS = [
    "ReLU",
    "RReLU",
    "PReLU",
    "ELU",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "Sigmoid",
    "GELU",
]


def time_to_feature(x: torch.Tensor) -> torch.Tensor:
    """Converts a time series Tensor to a feature Tensor."""
    return x.permute(0, 2, 1)


class TimeBatchNorm2d(nn.BatchNorm1d):
    def __init__(self, normalized_shape: Tuple[int, int]):
        """A batch normalization layer that normalizes over the last two dimensions of a
        sequence in PyTorch, mimicking Keras behavior based on the
        `PyTorch implementation of TSMixer <https://github.com/ditschuk/pytorch-tsmixer>`_.

        This class extends nn.BatchNorm1d to apply batch normalization across time and
        feature dimensions.

        Parameters
        ----------
        normalized_shape: Tuple
            A Tuple (num_time_steps, num_channels) representing the shape of the time and feature
            dimensions to normalize.
        """
        self.num_time_steps, self.num_channels = normalized_shape
        super().__init__(reduce(operator.mul, normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input Tensor, but got {x.ndim}D Tensor" " instead."
            )

        x = x.reshape(x.shape[0], -1, 1)
        x = super().forward(x)
        x = x.reshape(x.shape[0], self.num_time_steps, self.num_channels)
        return x


class FeatureMixing(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        output_dim: int,
        ff_size: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        dropout: float,
        normalize_before: bool,
        norm_type: type[nn.Module],
    ) -> None:
        """A module for feature mixing with flexibility in normalization and activation based on the
        `PyTorch implementation of TSMixer <https://github.com/ditschuk/pytorch-tsmixer>`_.

        This module provides options for batch normalization before or after mixing
        features, uses dropout for regularization, and allows for different activation
        functions.

        Parameters
        ----------
        sequence_length: int
            The length of the input sequences.
        input_dim: int
            The number of input channels to the module.
        output_dim: int
            The number of output channels from the module.
        ff_size: int
            The dimension of the feed-forward network internal to the module.
        activation: Callable[[torch.Tensor], torch.Tensor]
            The activation function used within the feed-forward network.
        dropout: float
            The dropout probability used for regularization.
        normalize_before: bool
            A boolean indicating whether to apply normalization before
            the rest of the operations.
        norm_type: type[nn.Module]
            The type of normalization to use.
        """
        super().__init__()

        self.projection = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )
        self.norm_before = (
            norm_type((sequence_length, input_dim))
            if normalize_before
            else nn.Identity()
        )
        self.fc1 = nn.Linear(input_dim, ff_size)
        self.activation = activation
        self.dropout = MonteCarloDropout(dropout)
        self.fc2 = nn.Linear(ff_size, output_dim)
        self.norm_after = (
            norm_type((sequence_length, output_dim))
            if not normalize_before
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.projection(x)
        x = self.norm_before(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x_proj + x
        x = self.norm_after(x)
        return x


class ConditionalFeatureMixing(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        output_dim: int,
        static_dim: int,
        ff_size: int,
        activation: Callable,
        dropout: float,
        normalize_before: bool,
        norm_type: type[nn.Module],
    ) -> None:
        """Conditional feature mixing module that incorporates static features based on the
        `PyTorch implementation of TSMixer <https://github.com/ditschuk/pytorch-tsmixer>`_.

        This module extends the feature mixing process by including static features. It uses
        a linear transformation to integrate static features into the dynamic feature space,
        then applies the feature mixing on the concatenated features.

        Parameters
        ----------
        sequence_length: int
            The length of the sequences to be transformed.
        input_dim: int
            The number of input channels of the dynamic features.
        output_dim: int
            The number of output channels after feature mixing.
        static_dim: int
            The number of channels in the static feature input.
        ff_size: int
            The inner dimension of the feedforward network used in feature mixing.
        activation: Callable
            The activation function used in feature mixing.
        dropout: float
            The dropout probability used in the feature mixing operation.
        normalize_before: bool
            Whether to apply normalization before or after feature mixing.
        norm_type: type[nn.Module]
            The type of normalization to use.
        """
        super().__init__()
        self.static_dim = static_dim
        if self.static_dim != 0:
            self.fr_static: Optional[nn.Linear] = nn.Linear(static_dim, output_dim)
            feature_mixing_input = input_dim + output_dim
        else:
            self.fr_static = None
            feature_mixing_input = input_dim

        self.fm = FeatureMixing(
            sequence_length=sequence_length,
            input_dim=feature_mixing_input,
            output_dim=output_dim,
            ff_size=ff_size,
            activation=activation,
            dropout=dropout,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    def forward(
        self, x: torch.Tensor, x_static: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.fr_static is None:
            return self.fm(x)
        v = self.fr_static(x_static)
        v = v.repeat(1, x.size(1) // v.size(1), 1)
        return self.fm(torch.cat([x, v], dim=-1))


class TimeMixing(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        activation: Callable,
        dropout: float,
        norm_type: type[nn.Module],
    ) -> None:
        """Applies a transformation over the time dimension of a sequence based on the
        `PyTorch implementation of TSMixer <https://github.com/ditschuk/pytorch-tsmixer>`_.

        This module applies a linear transformation followed by an activation function
        and dropout over the sequence length of the input feature torch.Tensor after converting
        feature maps to the time dimension and then back.

        Parameters
        ----------
        sequence_length: int
            The length of the sequences to be transformed.
        input_dim: int
            The number of input channels to the module.
        activation: Callable
            The activation function to be used after the linear
            transformation.
        dropout: float
            The dropout probability to be used after the activation function.
        norm_type: type[nn.Module]
            The type of normalization to use.
        """
        super().__init__()
        self.norm = norm_type((sequence_length, input_dim))
        self.activation = activation
        self.dropout = MonteCarloDropout(dropout)
        self.fc1 = nn.Linear(sequence_length, sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # permute the feature dim with the time dim
        x_temp = time_to_feature(x)
        x_temp = self.activation(self.fc1(x_temp))
        x_temp = self.dropout(x_temp)
        # permute back the time dim with the feature dim
        x_res = time_to_feature(x_temp)
        x_temp = self.norm(x + x_res)
        return x_temp


class ConditionalMixerLayer(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        output_dim: int,
        static_dim: int,
        ff_size: int,
        activation: Callable,
        dropout: float,
        normalize_before: bool,
        norm_type: type[nn.Module],
    ) -> None:
        """Conditional mix layer combining time and feature mixing with static context based on the
        `PyTorch implementation of TSMixer <https://github.com/ditschuk/pytorch-tsmixer>`_.

        This module combines time mixing and conditional feature mixing, where the latter
        is influenced by static features. This allows the module to learn representations
        that are influenced by both dynamic and static features.

        Parameters
        ----------
        sequence_length: int
            The length of the input sequences.
        input_dim: int
            The number of input channels of the dynamic features.
        output_dim: int
            The number of output channels after feature mixing.
        static_dim: int
            The number of channels in the static feature input.
        ff_size: int
            The inner dimension of the feedforward network used in feature mixing.
        activation: Callable
            The activation function used in both mixing operations.
        dropout: float
            The dropout probability used in both mixing operations.
        normalize_before: bool
            Whether to apply normalization before or after mixing.
        norm_type: type[nn.Module]
            The type of normalization to use.
        """
        super().__init__()

        self.time_mixing = TimeMixing(
            sequence_length=sequence_length,
            input_dim=input_dim,
            activation=activation,
            dropout=dropout,
            norm_type=norm_type,
        )
        self.feature_mixing = ConditionalFeatureMixing(
            sequence_length=sequence_length,
            input_dim=input_dim,
            output_dim=output_dim,
            static_dim=static_dim,
            ff_size=ff_size,
            activation=activation,
            dropout=dropout,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    def forward(
        self, x: torch.Tensor, x_static: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = self.time_mixing(x)
        x = self.feature_mixing(x, x_static)
        return x


class _TSMixerModule(PLMixedCovariatesModule):
    def __init__(
        self,
        activation: str,
        blocks: int,
        dropout: float,
        hidden_size: int,
        static_cov_dim: int,
        past_cov_dim: int,
        future_cov_dim: int,
        input_dim: int,
        output_dim: int,
        ff_size: int,
        nr_params: int,
        normalize_before: bool,
        norm_type: Union[str, nn.Module],
        **kwargs,
    ) -> None:
        """
        Initializes the TSMixer module for use within a Darts forecasting model.

        Parameters
        ----------
        input_dim: int
            Number of input channels (features).
        output_dim: int
            Number of output channels (forecast horizon).
        ff_size: int
            Dimension of the feedforward network internal to the module.
        dropout: float
            Dropout rate for regularization.
        blocks: int
            Number of mixer blocks.
        activation: str
            Activation function to use.
        normalize_before: bool
            Whether to apply normalization before or after mixing.
        norm_type: str
            Type of normalization to use.
        """
        super().__init__(**kwargs)

        self.ff_size = ff_size
        self.hidden_size = hidden_size
        self.normalize_before = normalize_before

        self.norm_type = norm_type
        self.dropout = dropout
        self.activation = activation

        self.static_cov_dim = static_cov_dim
        self.past_cov_dim = past_cov_dim
        self.future_cov_dim = future_cov_dim
        self.nr_params = nr_params
        self.input_dim = input_dim

        self.output_dim = output_dim
        self.blocks = blocks

        raise_if_not(
            activation in ACTIVATIONS, f"'{activation}' is not in {ACTIVATIONS}"
        )
        self.activation_func = getattr(nn, activation)()

        if isinstance(norm_type, str):
            if norm_type == "TimeBatchNorm2d":
                self.layer_norm = TimeBatchNorm2d
            else:
                try:
                    self.layer_norm = getattr(layer_norm_variants, norm_type)
                except AttributeError:
                    raise_log(
                        AttributeError("Please provide a valid layer norm type"),
                    )
        else:
            self.layer_norm = norm_type

        self.fc_hist = nn.Linear(self.input_chunk_length, self.output_chunk_length)

        mixer_params = {
            "ff_size": self.ff_size,
            "activation": self.activation_func,
            "dropout": self.dropout,
            "static_dim": self.static_cov_dim,
            "norm_type": self.layer_norm,
            "normalize_before": self.normalize_before,
        }

        hidden_output_size = self.hidden_size * self.output_dim * self.nr_params

        self.feature_mixing_hist = ConditionalFeatureMixing(
            sequence_length=self.output_chunk_length,
            input_dim=self.input_dim + self.past_cov_dim + self.future_cov_dim,
            output_dim=hidden_output_size,
            **mixer_params,
        )

        self.feature_mixing_future = ConditionalFeatureMixing(
            sequence_length=self.output_chunk_length,
            input_dim=self.future_cov_dim,
            output_dim=hidden_output_size,
            **mixer_params,
        )

        self.conditional_mixer = self._build_mixer(
            prediction_length=self.output_chunk_length,
            blocks=self.blocks,
            hidden_size=hidden_output_size,
            **mixer_params,
        )

        self.fc_out = nn.Linear(
            hidden_output_size,
            self.output_dim * self.nr_params,
        )

    def _build_mixer(
        self, blocks: int, hidden_size: int, prediction_length: int, **kwargs
    ):
        """Build the mixer blocks for the model."""
        # in case there are no future covariates, the feature_mixing_future layer
        # does not provide any output, so the input to the mixer layers is
        # the same as the output of the feature_mixing_hist layer
        input_dim = 2 * hidden_size if self.future_cov_dim > 0 else hidden_size

        mixer_layers = nn.ModuleList()
        for _ in range(blocks):
            layer = ConditionalMixerLayer(
                input_dim=input_dim,
                output_dim=hidden_size,
                sequence_length=prediction_length,
                **kwargs,
            )
            mixer_layers.append(layer)
            # After the first layer, input_dim should match the output_dim of the layers
            input_dim = hidden_size

        return mixer_layers

    @io_processor
    def forward(
        self,
        x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """TSMixer model forward pass.
        Parameters
        ----------
        x_in
            comes as Tuple `(x_past, x_future, x_static)` where
            `x_past` is the input/past chunk and
            `x_future` is the output/future chunk.
            Input dimensions are `(batch_size, time_steps, components)`
        Returns
        -------
        torch.torch.Tensor
            The output  Tensorof shape
            `(batch_size, output_chunk_length, output_dim, nr_params)`
        """
        x, x_future_covariates, x_static_covariates = x_in

        # x_hist contains the historical time series data and the historical
        # past and future covariates
        # (batch_size, input_chunk_length, input_dim + past_cov_dim + future_cov_dim)
        x_hist = x[:, :, : self.input_dim + self.past_cov_dim + self.future_cov_dim]

        x_static = x_static_covariates

        # Feature space to time space transformations and linear transformations
        # permute the feature dim with the time dim
        x_hist_temp = time_to_feature(x_hist)
        x_hist_temp = self.fc_hist(x_hist_temp)
        # permute back the time dim with the feature dim
        x_hist = time_to_feature(x_hist_temp)

        # Conditional feature mixing for historical and future data
        x_hist = self.feature_mixing_hist(x_hist, x_static=x_static)
        if x_future_covariates is not None:
            x_future = self.feature_mixing_future(
                x_future_covariates, x_static=x_static
            )
            x = torch.cat([x_hist, x_future], dim=-1)
        else:
            x = x_hist

        # Process through mixer layers
        for mixing_layer in self.conditional_mixer:
            x = mixing_layer(x, x_static=x_static)

        # Final linear transformation to produce the forecast
        x = self.fc_out(x)
        x = x.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)

        return x


class TSMixerModel(MixedCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        hidden_size: int = 256,
        ff_size: int = 64,
        dropout: float = 0.2,
        blocks: int = 6,
        activation: str = "ReLU",
        normalize_before: bool = False,
        norm_type: Union[str, nn.Module] = "LayerNorm",
        use_static_covariates: bool = True,
        **kwargs,
    ) -> None:
        """Time-Series Mixer (TSMixer): An All-MLP Architecture for Time Series.

        The internal layers are adopted from the `PyTorch implementation
        <https://github.com/ditschuk/pytorch-tsmixer>`_.
        This is an implementation of the TSMixer architecture, as outlined in [1]_.

        TSMixer forecasts time series data by integrating historical time series
        data, future known inputs, and static contextual information. It uses a
        combination of conditional feature mixing and mixer layers to process and
        combine these different types of data for effective forecasting.

        This model supports past covariates (known for `input_chunk_length` points
        before prediction time), future covariates (known for `output_chunk_length`
        points after prediction time), static covariates, as well as probabilistic
        forecasting.

        Parameters
        ----------
        input_chunk_length: int
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it).
            Also called: Encoder length
        output_chunk_length: int
            Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
            from future covariates to use as a model input (if the model supports future covariates). It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
            using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents
            auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
            the model from using future values of past and / or future covariates for prediction (depending on the
            model's covariate support).
            Also called: Decoder length
        output_chunk_shift: int
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`).
        hidden_size: int
            Hidden state size of the TSMixer.
        activation: str
            The activation function of the mixer layers (default='ReLU').
            Supported activations: ["ReLU", "RReLU", "PReLU", "ELU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid",
            "GELU"]
        blocks: int
            The number of mixer blocks in the model.
        dropout: float
            Fraction of neurons affected by dropout. This is compatible with Monte Carlo dropout
            at inference time for model uncertainty estimation (enabled with ``mc_dropout=True`` at
            prediction time).
        ff_size: int
            The inner size of the feedforward network in the mixer layers.
        normalize_before: bool
            Whether to apply layer normalization before or after mixer layer.
        loss_fn: nn.Module
            PyTorch loss function used for training. By default, the TFT
            model is probabilistic and uses a ``likelihood`` instead
            (``QuantileRegression``). To make the model deterministic, you
            can set the ``likelihood`` to None and give a ``loss_fn``
            argument.
        likelihood
            The likelihood model to be used for probabilistic forecasts.
        norm_type: str | nn.Module
            The type of LayerNorm variant to use.  Default: ``LayerNorm``. Available options are
            ["LayerNormNoBias", "LayerNorm", "RINorm"], or provide a custom nn.Module.
        use_static_covariates: bool
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
            Name of the model. Used for creating checkpoints and saving torch.Tensorboard data. If not specified,
            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial part
            of the name is formatted with the local date and time, while PID is the processed ID (preventing models
            spawned at the same time by different processes to share the same model_name). E.g.,
            ``"2021-06-14_09_53_32_torch_model_run_44607"``.
        work_dir
            Path of the working directory, where to save checkpoints and torch.Tensorboard summaries.
            Default: current working directory.
        log_torch.Tensorboard
            If set, use torch.Tensorboard to log the different parameters. The logs will be located in:
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
            checkpointing, torch.Tensorboard logging, setting the torch device and more.
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
        .. [1] https://arxiv.org/abs/2303.06053

        Examples
        --------
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import TSMixerModel
        >>> series = WeatherDataset().load()
        >>> # predicting atmospheric pressure
        >>> target = series['p (mbar)'][:100]
        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)
        >>> past_cov = series['rain (mm)'][:100]
        >>> # optionally, use future temperatures (pretending this component is a forecast)
        >>> future_cov = series['T (degC)'][:106]
        >>> model = TSMixerModel(
        >>>     input_chunk_length=6,
        >>>     output_chunk_length=6,
        >>>     n_epochs=20
        >>> )
        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[26.75691625],
            [27.3836554 ],
            [26.02049222],
            [26.9186771 ],
            [27.88810697],
            [25.86081595]])
        """
        model_kwargs = {key: val for key, val in self.model_params.items()}
        super().__init__(**self._extract_torch_model_params(**model_kwargs))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)

        # Model specific parameters
        self.ff_size = ff_size
        self.dropout = dropout
        self.blocks = blocks
        self.activation = activation
        self.normalize_before = normalize_before
        self.norm_type = norm_type
        self.hidden_size = hidden_size
        self._considers_static_covariates = use_static_covariates

    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """
        `train_sample` contains the following torch.Tensors:
            (past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates,
            future_target)

            each torch.Tensor has shape (n_timesteps, n_variables)
            - past/historic torch.Tensors have shape (input_chunk_length, n_variables)
            - future torch.Tensors have shape (output_chunk_length, n_variables)
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
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = train_sample

        self.input_dim = past_target.shape[1]
        self.output_dim = future_target.shape[1]

        self.static_cov_dim = (
            static_covariates.shape[-1] if static_covariates is not None else 0
        )
        self.future_cov_dim = (
            future_covariates.shape[1] if future_covariates is not None else 0
        )
        self.past_cov_dim = (
            past_covariates.shape[1] if past_covariates is not None else 0
        )
        self.nr_params = (
            1 if self.likelihood is None else self.likelihood.num_parameters
        )

        self.model = _TSMixerModule(
            hidden_size=self.hidden_size,
            static_cov_dim=self.static_cov_dim,
            past_cov_dim=self.past_cov_dim,
            future_cov_dim=self.future_cov_dim,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            ff_size=self.ff_size,
            dropout=self.dropout,
            blocks=self.blocks,
            activation=self.activation,
            normalize_before=self.normalize_before,
            norm_type=self.norm_type,
            nr_params=self.nr_params,
            **self.pl_module_params,
        )

        return self.model

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def supports_static_covariates(self) -> bool:
        return True

    @property
    def supports_future_covariates(self) -> bool:
        return True

    @property
    def supports_past_covariates(self) -> bool:
        return True
