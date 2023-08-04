"""
Time-series Dense Encoder (TiDE)
------
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from darts.logging import get_logger
from darts.models.components.layer_norm_variants import RINorm
from darts.models.forecasting.pl_forecasting_module import PLMixedCovariatesModule
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel

MixedCovariatesTrainTensorType = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


logger = get_logger(__name__)


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        dropout: float,
        use_layer_norm: bool,
    ):
        """Pytorch module implementing the Residual Block from the TiDE paper."""
        super().__init__()

        # dense layer with ReLU activation with dropout
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Dropout(dropout),
        )

        # linear skip connection from input to output of self.dense
        self.skip = nn.Linear(input_dim, output_dim)

        # layer normalization as output
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # residual connection
        x = self.dense(x) + self.skip(x)

        # layer normalization
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x


class _TideModule(PLMixedCovariatesModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        nr_params: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        decoder_output_dim: int,
        hidden_size: int,
        temporal_decoder_hidden: int,
        temporal_width: int,
        use_layer_norm: bool,
        use_reversible_instance_norm: bool,
        dropout: float,
        **kwargs,
    ):
        """Pytorch module implementing the TiDE architecture.

        Parameters
        ----------
        input_dim
            The number of input components (target + optional past covariates + optional future covariates).
        output_dim
            Number of output components in the target.
        future_cov_dim
            Number of future covariates.
        static_cov_dim
            Number of static covariates.
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        num_encoder_layers
            Number of stacked Residual Blocks in the encoder.
        num_decoder_layers
            Number of stacked Residual Blocks in the decoder.
        decoder_output_dim
            The number of output components of the decoder.
        hidden_size
            The width of the hidden layers in the encoder/decoder Residual Blocks.
        temporal_decoder_hidden
            The width of the hidden layers in the temporal decoder.
        temporal_width
            The width of the future covariate embedding space.
        use_layer_norm
            Whether to use layer normalization in the Residual Blocks.
        use_reversible_instance_norm
            Whether to use reversible instance normalization.
        dropout
            Dropout probability
        **kwargs
            all parameters required for :class:`darts.model.forecasting_models.PLForecastingModule` base class.

        Inputs
        ------
        x
            Tuple of Tensors `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and
            `x_future`is the output/future chunk. Input dimensions are `(batch_size, time_steps, components)`
        Outputs
        -------
        y
            Tensor of shape `(batch_size, output_chunk_length, output_dim, nr_params)`

        """

        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_decoder_hidden = temporal_decoder_hidden
        self.use_layer_norm = use_layer_norm
        self.use_reversible_instance_norm = use_reversible_instance_norm
        self.dropout = dropout
        self.temporal_width = temporal_width

        # residual block for input feature projection
        # this is only needed when covariates are used
        if future_cov_dim:
            self.feature_projection = _ResidualBlock(
                input_dim=future_cov_dim,
                output_dim=temporal_width,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
        else:
            self.feature_projection = None

        # original paper doesn't specify how to use past covariates
        # we assume that they pass them raw to the encoder
        historical_future_covariates_flat_dim = (
            self.input_chunk_length + self.output_chunk_length
        ) * (self.temporal_width if future_cov_dim > 0 else 0)
        encoder_dim = (
            self.input_chunk_length * (input_dim - future_cov_dim)
            + historical_future_covariates_flat_dim
            + static_cov_dim
        )

        self.encoders = nn.Sequential(
            _ResidualBlock(
                input_dim=encoder_dim,
                output_dim=hidden_size,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers - 1)
            ],
        )

        self.decoders = nn.Sequential(
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers - 1)
            ],
            # add decoder output layer
            _ResidualBlock(
                input_dim=hidden_size,
                output_dim=decoder_output_dim
                * self.output_chunk_length
                * self.nr_params,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
        )

        self.temporal_decoder = _ResidualBlock(
            input_dim=decoder_output_dim * self.nr_params
            + (temporal_width if future_cov_dim > 0 else 0),
            output_dim=output_dim * self.nr_params,
            hidden_size=temporal_decoder_hidden,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        self.lookback_skip = nn.Linear(
            self.input_chunk_length, self.output_chunk_length * self.nr_params
        )

        if self.use_reversible_instance_norm:
            self.rin = RINorm(input_dim=output_dim)
        else:
            self.rin = None

    def forward(
        self, x_in: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """TiDE model forward pass.
        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
            is the output/future chunk. Input dimensions are `(batch_size, time_steps, components)`
        Returns
        -------
        torch.Tensor
            The output Tensor of shape `(batch_size, output_chunk_length, output_dim, nr_params)`
        """

        # x has shape (batch_size, input_chunk_length, input_dim)
        # x_future_covariates has shape (batch_size, input_chunk_length, future_cov_dim)
        # x_static_covariates has shape (batch_size, static_cov_dim)
        x, x_future_covariates, x_static_covariates = x_in

        if self.use_reversible_instance_norm:
            x[:, :, : self.output_dim] = self.rin(x[:, :, : self.output_dim])

        x_lookback = x[:, :, : self.output_dim]

        # future covariates need to be extracted from x and stacked with historical future covariates
        if self.future_cov_dim > 0:
            x_dynamic_covariates = torch.cat(
                [
                    x_future_covariates,
                    x[
                        :,
                        :,
                        None if self.future_cov_dim == 0 else -self.future_cov_dim :,
                    ],
                ],
                dim=1,
            )

            # project input features across all input time steps
            x_dynamic_covariates_proj = self.feature_projection(x_dynamic_covariates)

        else:
            x_dynamic_covariates = None
            x_dynamic_covariates_proj = None

        # extract past covariates, if they exist
        if self.input_dim - self.output_dim - self.future_cov_dim > 0:
            x_past_covariates = x[
                :,
                :,
                self.output_dim : None
                if self.future_cov_dim == 0
                else -self.future_cov_dim :,
            ]
        else:
            x_past_covariates = None

        # setup input to encoder
        encoded = [
            x_lookback,
            x_past_covariates,
            x_dynamic_covariates_proj,
            x_static_covariates,
        ]
        encoded = [t.flatten(start_dim=1) for t in encoded if t is not None]
        encoded = torch.cat(encoded, dim=1)

        # encoder, decode, reshape
        encoded = self.encoders(encoded)
        decoded = self.decoders(encoded)

        # get view that is batch size x output chunk length x self.decoder_output_dim x nr params
        decoded = decoded.view(x.shape[0], self.output_chunk_length, -1)

        # stack and temporally decode with future covariate last output steps
        temporal_decoder_input = [
            decoded,
            x_dynamic_covariates_proj[:, -self.output_chunk_length :, :]
            if self.future_cov_dim > 0
            else None,
        ]
        temporal_decoder_input = [t for t in temporal_decoder_input if t is not None]

        temporal_decoder_input = torch.cat(temporal_decoder_input, dim=2)
        temporal_decoded = self.temporal_decoder(temporal_decoder_input)

        # pass x_lookback through self.lookback_skip but swap the last two dimensions
        # this is needed because the skip connection is applied across the input time steps
        # and not across the output time steps
        skip = self.lookback_skip(x_lookback.transpose(1, 2)).transpose(1, 2)

        # add skip connection
        y = temporal_decoded + skip.reshape_as(
            temporal_decoded
        )  # skip.view(temporal_decoded.shape)

        y = y.view(-1, self.output_chunk_length, self.output_dim, self.nr_params)

        if self.use_reversible_instance_norm:
            y = self.rin.inverse(y)

        return y


class TiDEModel(MixedCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        decoder_output_dim: int = 16,
        hidden_size: int = 128,
        temporal_width: int = 4,
        temporal_decoder_hidden: int = 32,
        use_layer_norm: bool = False,
        use_reversible_instance_norm: bool = False,
        dropout: float = 0.1,
        use_static_covariates: bool = True,
        **kwargs,
    ):
        """An implementation of the TiDE model, as presented in [1]_.

        TiDE is similar to Transformers (implemented in :class:`TransformerModel`),
        but attempts to provide better performance at lower computational cost by introducing
        multilayer perceptron (MLP)-based encoder-decoders without attention.

        The model is implemented as a :class:`MixedCovariatesTorchModel`, which means that it supports
        both past and future covariates, as well as static covariates. Probabilistic forecasting is supported through
        the use of a `likelihood` instead of a `loss_fn`.
        The original paper does not describe how past covariates are treated in detail, so we assume that they are
        passed to the encoder as-is.

        The encoder and decoder are implemented as a series of residual blocks. The number of residual blocks in
        the encoder and decoder can be controlled via ``num_encoder_layers`` and ``num_decoder_layers`` respectively.
        The width of the layers in the residual blocks can be controlled via ``hidden_size``. Similarly, the width
        of the layers in the temporal decoder can be controlled via ``temporal_decoder_hidden``.

        Parameters
        ----------
        input_chunk_length
            The length of the input sequence fed to the model.
        output_chunk_length
            The length of the forecast of the model.
        num_encoder_layers
            The number of residual blocks in the encoder.
        num_decoder_layers
            The number of residual blocks in the decoder.
        decoder_output_dim
            The dimensionality of the output of the decoder.
        hidden_size
            The width of the layers in the residual blocks of the encoder and decoder.
        temporal_width
            The width of the layers in the future covariate projection residual block.
        temporal_decoder_hidden
            The width of the layers in the temporal decoder.
        use_layer_norm
            Whether to use layer normalization in the residual blocks.
        use_reversible_instance_norm
            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [2]_.
            It is only applied to the features of the target series and not the covariates.
        dropout
            The dropout probability to be used in fully connected layers. This is compatible with Monte Carlo dropout
            at inference time for model uncertainty estimation (enabled with ``mc_dropout=True`` at
            prediction time).
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
            Whether or not to automatically save the untrained model and checkpoints from training.
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

                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'future': ['hour', 'dayofweek']},
                    'position': {'past': ['relative'], 'future': ['relative']},
                    'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
                }
            ..
        random_state
            Control the randomness of the weights initialization. Check this
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
        .. [1] A. Das et al. "Long-term Forecasting with TiDE: Time-series Dense Encoder",
                http://arxiv.org/abs/2304.08424
        .. [2] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against
                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim
        self.hidden_size = hidden_size
        self.temporal_width = temporal_width
        self.temporal_decoder_hidden = temporal_decoder_hidden

        self._considers_static_covariates = use_static_covariates

        self.use_layer_norm = use_layer_norm
        self.use_reversible_instance_norm = use_reversible_instance_norm
        self.dropout = dropout

    def _create_model(
        self, train_sample: MixedCovariatesTrainTensorType
    ) -> torch.nn.Module:
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = train_sample

        # target, past covariates, historic future covariates
        input_dim = (
            past_target.shape[1]
            + (past_covariates.shape[1] if past_covariates is not None else 0)
            + (
                historic_future_covariates.shape[1]
                if historic_future_covariates is not None
                else 0
            )
        )

        output_dim = future_target.shape[1]

        future_cov_dim = (
            future_covariates.shape[1] if future_covariates is not None else 0
        )
        static_cov_dim = (
            static_covariates.shape[0] * static_covariates.shape[1]
            if static_covariates is not None
            else 0
        )

        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _TideModule(
            input_dim=input_dim,
            output_dim=output_dim,
            future_cov_dim=future_cov_dim,
            static_cov_dim=static_cov_dim,
            nr_params=nr_params,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_output_dim=self.decoder_output_dim,
            hidden_size=self.hidden_size,
            temporal_width=self.temporal_width,
            temporal_decoder_hidden=self.temporal_decoder_hidden,
            use_layer_norm=self.use_layer_norm,
            use_reversible_instance_norm=self.use_reversible_instance_norm,
            dropout=self.dropout,
            **self.pl_module_params,
        )

    @property
    def supports_static_covariates(self) -> bool:
        return True

    @property
    def supports_multivariate(self) -> bool:
        return True

    def predict(self, n, *args, **kwargs):
        # since we have future covariates, the inference dataset for future input must be at least of length
        # `output_chunk_length`. If not, we would have to step back which causes past input to be shorter than
        # `input_chunk_length`.

        if n >= self.output_chunk_length:
            return super().predict(n, *args, **kwargs)
        else:
            return super().predict(self.output_chunk_length, *args, **kwargs)[:n]
