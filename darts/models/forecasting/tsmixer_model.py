"""
Time-Series Mixer (TSMixer)
---------------------------
"""

# The inner layers (``nn.Modules``) and the ``TimeBatchNorm2d`` were provided by a PyTorch implementation
# of TSMixer: https://github.com/ditschuk/pytorch-tsmixer
#
# The License of pytorch-tsmixer v0.2.0 from https://github.com/ditschuk/pytorch-tsmixer/blob/main/LICENSE,
# accessed Thursday, March 21st, 2024:
# 'The MIT License
#
# Copyright 2023 Konstantin Ditschuneit
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the “Software”), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
# '

from typing import Callable, Optional, Union

import torch
from torch import nn

from darts.logging import get_logger, raise_log
from darts.models.components import layer_norm_variants
from darts.models.forecasting.pl_forecasting_module import (
    PLMixedCovariatesModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.torch import MonteCarloDropout

MixedCovariatesTrainTensorType = tuple[
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

NORMS = [
    "LayerNorm",
    "LayerNormNoBias",
    "TimeBatchNorm2d",
]


def _time_to_feature(x: torch.Tensor) -> torch.Tensor:
    """Converts a time series Tensor to a feature Tensor."""
    return x.permute(0, 2, 1)


class TimeBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        """A batch normalization layer that normalizes over the last two dimensions of a Tensor."""
        super().__init__(num_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `x` has shape (batch_size, time, features)
        if x.ndim != 3:
            raise_log(
                ValueError(
                    f"Expected 3D input Tensor, but got {x.ndim}D Tensor instead."
                ),
                logger=logger,
            )
        # apply 2D batch norm over reshape input_data `(batch_size, 1, timepoints, features)`
        output = super().forward(x.unsqueeze(1))
        # reshape back to (batch_size, timepoints, features)
        return output.squeeze(1)


class _FeatureMixing(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        output_dim: int,
        ff_size: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        dropout: float,
        normalize_before: bool,
        norm_type: nn.Module,
    ) -> None:
        """A module for feature mixing with flexibility in normalization and activation based on the
        `PyTorch implementation of TSMixer <https://github.com/ditschuk/pytorch-tsmixer>`_.

        This module provides options for batch normalization before or after mixing
        features, uses dropout for regularization, and allows for different activation
        functions.

        Parameters
        ----------
        sequence_length
            The length of the input sequences.
        input_dim
            The number of input channels to the module.
        output_dim
            The number of output channels from the module.
        ff_size
            The dimension of the feed-forward network internal to the module.
        activation
            The activation function used within the feed-forward network.
        dropout
            The dropout probability used for regularization.
        normalize_before
            A boolean indicating whether to apply normalization before
            the rest of the operations.
        norm_type
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
        self.dropout1 = MonteCarloDropout(dropout)
        self.fc2 = nn.Linear(ff_size, output_dim)
        self.dropout2 = MonteCarloDropout(dropout)
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
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = x_proj + x
        x = self.norm_after(x)
        return x


class _TimeMixing(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        activation: Callable,
        dropout: float,
        normalize_before: bool,
        norm_type: nn.Module,
    ) -> None:
        """Applies a transformation over the time dimension of a sequence based on the
        `PyTorch implementation of TSMixer <https://github.com/ditschuk/pytorch-tsmixer>`_.

        This module applies a linear transformation followed by an activation function
        and dropout over the sequence length of the input feature torch.Tensor after converting
        feature maps to the time dimension and then back.

        Parameters
        ----------
        sequence_length
            The length of the sequences to be transformed.
        input_dim
            The number of input channels to the module.
        activation
            The activation function to be used after the linear
            transformation.
        dropout
            The dropout probability to be used after the activation function.
        normalize_before
            Whether to apply normalization before or after feature mixing.
        norm_type
            The type of normalization to use.
        """
        super().__init__()
        self.normalize_before = normalize_before
        self.norm_before = (
            norm_type((sequence_length, input_dim))
            if normalize_before
            else nn.Identity()
        )
        self.activation = activation
        self.dropout = MonteCarloDropout(dropout)
        self.fc1 = nn.Linear(sequence_length, sequence_length)
        self.norm_after = (
            norm_type((sequence_length, input_dim))
            if not normalize_before
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # permute the feature dim with the time dim
        x_temp = self.norm_before(x)
        x_temp = _time_to_feature(x_temp)
        x_temp = self.activation(self.fc1(x_temp))
        x_temp = self.dropout(x_temp)
        # permute back the time dim with the feature dim
        x_temp = x + _time_to_feature(x_temp)
        x_temp = self.norm_after(x_temp)
        return x_temp


class _ConditionalMixerLayer(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        input_dim: int,
        output_dim: int,
        static_cov_dim: int,
        ff_size: int,
        activation: Callable,
        dropout: float,
        normalize_before: bool,
        norm_type: nn.Module,
    ) -> None:
        """Conditional mix layer combining time and feature mixing with static context based on the
        `PyTorch implementation of TSMixer <https://github.com/ditschuk/pytorch-tsmixer>`_.

        This module combines time mixing and conditional feature mixing, where the latter
        is influenced by static features. This allows the module to learn representations
        that are influenced by both dynamic and static features.

        Parameters
        ----------
        sequence_length
            The length of the input sequences.
        input_dim
            The number of input channels of the dynamic features.
        output_dim
            The number of output channels after feature mixing.
        static_cov_dim
            The number of channels in the static feature input.
        ff_size
            The inner dimension of the feedforward network used in feature mixing.
        activation
            The activation function used in both mixing operations.
        dropout
            The dropout probability used in both mixing operations.
        normalize_before
            Whether to apply normalization before or after mixing.
        norm_type
            The type of normalization to use.
        """
        super().__init__()

        mixing_input = input_dim
        if static_cov_dim != 0:
            self.feature_mixing_static = _FeatureMixing(
                sequence_length=sequence_length,
                input_dim=static_cov_dim,
                output_dim=output_dim,
                ff_size=ff_size,
                activation=activation,
                dropout=dropout,
                normalize_before=normalize_before,
                norm_type=norm_type,
            )
            mixing_input += output_dim
        else:
            self.feature_mixing_static = None

        self.time_mixing = _TimeMixing(
            sequence_length=sequence_length,
            input_dim=mixing_input,
            activation=activation,
            dropout=dropout,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )
        self.feature_mixing = _FeatureMixing(
            sequence_length=sequence_length,
            input_dim=mixing_input,
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
        if self.feature_mixing_static is not None:
            x_static_mixed = self.feature_mixing_static(x_static)
            x = torch.cat([x, x_static_mixed], dim=-1)
        x = self.time_mixing(x)
        x = self.feature_mixing(x)
        return x


class _TSMixerModule(PLMixedCovariatesModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        past_cov_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        nr_params: int,
        hidden_size: int,
        ff_size: int,
        num_blocks: int,
        activation: str,
        dropout: float,
        norm_type: Union[str, nn.Module],
        normalize_before: bool,
        **kwargs,
    ) -> None:
        """
        Initializes the TSMixer module for use within a Darts forecasting model.

        Parameters
        ----------
        input_dim
            Number of input target features.
        output_dim
            Number of output target features.
        past_cov_dim
            Number of past covariate features.
        future_cov_dim
            Number of future covariate features.
        static_cov_dim
            Number of static covariate features (number of target features
            (or 1 if global static covariates) * number of static covariate features).
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        hidden_size
           Hidden state size of the TSMixer.
        ff_size
            Dimension of the feedforward network internal to the module.
        num_blocks
            Number of mixer blocks.
        activation
            Activation function to use.
        dropout
            Dropout rate for regularization.
        norm_type
            Type of normalization to use.
        normalize_before
            Whether to apply normalization before or after mixing.
        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params

        if activation not in ACTIVATIONS:
            raise_log(
                ValueError(
                    f"Invalid `activation={activation}`. Must be on of {ACTIVATIONS}."
                ),
                logger=logger,
            )
        activation = getattr(nn, activation)()

        if isinstance(norm_type, str):
            if norm_type not in NORMS:
                raise_log(
                    ValueError(
                        f"Invalid `norm_type={norm_type}`. Must be on of {NORMS}."
                    ),
                    logger=logger,
                )
            if norm_type == "TimeBatchNorm2d":
                norm_type = TimeBatchNorm2d
            else:
                norm_type = getattr(layer_norm_variants, norm_type)
        else:
            norm_type = norm_type

        mixer_params = {
            "ff_size": ff_size,
            "activation": activation,
            "dropout": dropout,
            "norm_type": norm_type,
            "normalize_before": normalize_before,
        }

        self.fc_hist = nn.Linear(self.input_chunk_length, self.output_chunk_length)
        self.feature_mixing_hist = _FeatureMixing(
            sequence_length=self.output_chunk_length,
            input_dim=input_dim + past_cov_dim + future_cov_dim,
            output_dim=hidden_size,
            **mixer_params,
        )
        if future_cov_dim:
            self.feature_mixing_future = _FeatureMixing(
                sequence_length=self.output_chunk_length,
                input_dim=future_cov_dim,
                output_dim=hidden_size,
                **mixer_params,
            )
        else:
            self.feature_mixing_future = None
        self.conditional_mixer = self._build_mixer(
            prediction_length=self.output_chunk_length,
            num_blocks=num_blocks,
            hidden_size=hidden_size,
            future_cov_dim=future_cov_dim,
            static_cov_dim=static_cov_dim,
            **mixer_params,
        )
        self.fc_out = nn.Linear(hidden_size, output_dim * nr_params)

    @staticmethod
    def _build_mixer(
        prediction_length: int,
        num_blocks: int,
        hidden_size: int,
        future_cov_dim: int,
        static_cov_dim: int,
        **kwargs,
    ) -> nn.ModuleList:
        """Build the mixer blocks for the model."""
        # the first block takes `x` consisting of concatenated features with size `hidden_size`:
        # - historic features
        # - optional future features
        input_dim_block = hidden_size * (1 + int(future_cov_dim > 0))

        mixer_layers = nn.ModuleList()
        for _ in range(num_blocks):
            layer = _ConditionalMixerLayer(
                input_dim=input_dim_block,
                output_dim=hidden_size,
                sequence_length=prediction_length,
                static_cov_dim=static_cov_dim,
                **kwargs,
            )
            mixer_layers.append(layer)
            # after the first block, `x` consists of previous block output with size `hidden_size`
            input_dim_block = hidden_size
        return mixer_layers

    @io_processor
    def forward(
        self,
        x_in: tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
    ) -> torch.Tensor:
        # x_hist contains the historical time series data and the historical
        """TSMixer model forward pass.

        Parameters
        ----------
        x_in
            comes as Tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and
            `x_future` is the output/future chunk. Input dimensions are `(batch_size, time_steps,
            components)`.

        Returns
        -------
        torch.torch.Tensor
            The output  Tensorof shape `(batch_size, output_chunk_length, output_dim, nr_params)`.
        """
        # B: batch size
        # L: input chunk length
        # T: output chunk length
        # C: target components
        # P: past cov features
        # F: future cov features
        # S: static cov features
        # H = C + P + F: historic features
        # H_S: hidden Size
        # N_P: likelihood parameters

        # `x`: (B, L, H), `x_future`: (B, T, F), `x_static`: (B, C or 1, S)
        x, x_future, x_static = x_in

        # swap feature and time dimensions (B, L, H) -> (B, H, L)
        x = _time_to_feature(x)
        # linear transformations to horizon (B, H, L) -> (B, H, T)
        x = self.fc_hist(x)
        # (B, H, T) -> (B, T, H)
        x = _time_to_feature(x)

        # feature mixing for historical features (B, T, H) -> (B, T, H_S)
        x = self.feature_mixing_hist(x)
        if self.future_cov_dim:
            # feature mixing for future features (B, T, F) -> (B, T, H_S)
            x_future = self.feature_mixing_future(x_future)
            # (B, T, H_S) + (B, T, H_S) -> (B, T, 2*H_S)
            x = torch.cat([x, x_future], dim=-1)

        if self.static_cov_dim:
            # (B, C, S) -> (B, 1, C * S)
            x_static = x_static.reshape(x_static.shape[0], 1, -1)
            # repeat to match horizon (B, 1, C * S) -> (B, T, C * S)
            x_static = x_static.repeat(1, self.output_chunk_length, 1)

        for mixing_layer in self.conditional_mixer:
            # conditional mixer layers with static covariates (B, T, 2 * H_S), (B, T, C * S) -> (B, T, H_S)
            x = mixing_layer(x, x_static=x_static)

        # linear transformation to generate the forecast (B, T, H_S) -> (B, T, C * N_P)
        x = self.fc_out(x)
        # (B, T, C * N_P) -> (B, T, C, N_P)
        x = x.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)
        return x


class TSMixerModel(MixedCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        hidden_size: int = 64,
        ff_size: int = 64,
        num_blocks: int = 2,
        activation: str = "ReLU",
        dropout: float = 0.1,
        norm_type: Union[str, nn.Module] = "LayerNorm",
        normalize_before: bool = False,
        use_static_covariates: bool = True,
        **kwargs,
    ) -> None:
        """Time-Series Mixer (TSMixer): An All-MLP Architecture for Time Series.

        This is an implementation of the TSMixer architecture, as outlined in [1]_. A major part of the architecture
        was adopted from `this PyTorch implementation <https://github.com/ditschuk/pytorch-tsmixer>`_. Additional
        changes were applied to increase model performance and efficiency.

        TSMixer forecasts time series data by integrating historical time series data, future known inputs, and static
        contextual information. It uses a combination of conditional feature mixing and mixer layers to process and
        combine these different types of data for effective forecasting.

        This model supports past covariates (known for `input_chunk_length` points before prediction time), future
        covariates (known for `output_chunk_length` points after prediction time), static covariates, as well as
        probabilistic forecasting.

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
            The hidden state size / size of the second feed-forward layer in the feature mixing MLP.
        ff_size
            The size of the first feed-forward layer in the feature mixing MLP.
        num_blocks
            The number of mixer blocks in the model. The number includes the first block and all subsequent blocks.
        activation
            The name of the activation function to use in the mixer layers. Default: `"ReLU"`. Must be one of
            `"ReLU", "RReLU", "PReLU", "ELU", "Softplus", "Tanh", "SELU", "LeakyReLU", "Sigmoid", "GELU"`.
        dropout
            Fraction of neurons affected by dropout. This is compatible with Monte Carlo dropout at inference time
            for model uncertainty estimation (enabled with ``mc_dropout=True`` at prediction time).
        norm_type
            The type of `LayerNorm` variant to use.  Default: `"LayerNorm"`. If a string, must be one of
            `"LayerNormNoBias", "LayerNorm", "TimeBatchNorm2d"`. Otherwise, must be a custom `nn.Module`.
        normalize_before
            Whether to apply layer normalization before or after mixer layer.
        use_static_covariates
            Whether the model should use static covariate information in case the input `series` passed to ``fit()``
            contain static covariates. If ``True``, and static covariates are available at fitting time, will enforce
            that all target `series` have the same static covariate dimensionality in ``fit()`` and ``predict()``.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.
            Default: ``torch.nn.MSELoss()``.
        likelihood
            One of Darts' :meth:`Likelihood <darts.utils.likelihood_models.Likelihood>` models to be used for
            probabilistic forecasts. Default: ``None``.
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
        >>> # predicting temperatures
        >>> target = series['T (degC)'][:100]
        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)
        >>> past_cov = series['rain (mm)'][:100]
        >>> # optionally, use future atmospheric pressure (pretending this component is a forecast)
        >>> future_cov = series['p (mbar)'][:106]
        >>> model = TSMixerModel(
        >>>     input_chunk_length=6,
        >>>     output_chunk_length=6,
        >>>     use_reversible_instance_norm=True,
        >>>     n_epochs=20
        >>> )
        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[3.92519848],
            [4.05650312],
            [4.21781987],
            [4.29394973],
            [4.4122863 ],
            [4.42762751]])
        """
        model_kwargs = {key: val for key, val in self.model_params.items()}
        super().__init__(**self._extract_torch_model_params(**model_kwargs))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**model_kwargs)

        # Model specific parameters
        self.ff_size = ff_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.activation = activation
        self.normalize_before = normalize_before
        self.norm_type = norm_type
        self.hidden_size = hidden_size
        self._considers_static_covariates = use_static_covariates

    def _create_model(self, train_sample: MixedCovariatesTrainTensorType) -> nn.Module:
        """
        Parameters
        ----------
        train_sample
            contains the following torch.Tensors: `(past_target, past_covariates, historic_future_covariates,
            future_covariates, static_covariates, future_target)`:

            - past/historic torch.Tensors have shape (input_chunk_length, n_variables)
            - future torch.Tensors have shape (output_chunk_length, n_variables)
            - static covariates have shape (component, static variable)
        """
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = train_sample

        input_dim = past_target.shape[1]
        output_dim = future_target.shape[1]

        static_cov_dim = (
            static_covariates.shape[0] * static_covariates.shape[1]
            if static_covariates is not None
            else 0
        )
        future_cov_dim = (
            future_covariates.shape[1] if future_covariates is not None else 0
        )
        past_cov_dim = past_covariates.shape[1] if past_covariates is not None else 0
        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _TSMixerModule(
            input_dim=input_dim,
            output_dim=output_dim,
            future_cov_dim=future_cov_dim,
            past_cov_dim=past_cov_dim,
            static_cov_dim=static_cov_dim,
            nr_params=nr_params,
            hidden_size=self.hidden_size,
            ff_size=self.ff_size,
            num_blocks=self.num_blocks,
            activation=self.activation,
            dropout=self.dropout,
            norm_type=self.norm_type,
            normalize_before=self.normalize_before,
            **self.pl_module_params,
        )

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
