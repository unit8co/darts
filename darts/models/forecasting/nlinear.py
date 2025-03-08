"""
N-Linear
------
"""

from typing import Optional

import torch
import torch.nn as nn

from darts.logging import raise_if
from darts.models.forecasting.pl_forecasting_module import (
    PLMixedCovariatesModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel


class _NLinearModule(PLMixedCovariatesModule):
    """
    NLinear module
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        future_cov_dim: int,
        static_cov_dim: int,
        nr_params: int,
        shared_weights: bool,
        const_init: bool,
        normalize: bool,
        **kwargs,
    ):
        """PyTorch module implementing the N-HiTS architecture.

        Parameters
        ----------
        input_dim
            The number of input components (target + optional covariate)
        output_dim
            Number of output components in the target
        future_cov_dim
            Number of components in the future covariates
        static_cov_dim
            Dimensionality of the static covariates (either component-specific or shared)
        nr_params
            The number of parameters of the likelihood (or 1 if no likelihood is used).
        shared_weights
            Whether to use shared weights for the components of the series.
            ** Ignores covariates when True. **
        const_init
            Whether to initialize the weights to 1/in_len
        normalize
            Whether to apply the "normalization" described in the paper.

        **kwargs
            all parameters required for :class:`darts.models.forecasting.pl_forecasting_module.PLForecastingModule`
            base class.

        Inputs
        ------
        x of shape `(batch_size, input_chunk_length)`
            Tensor containing the input sequence.

        Outputs
        -------
        y of shape `(batch_size, output_chunk_length, target_size/output_dim, nr_params)`
            Tensor containing the output of the NBEATS module.

        """
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.future_cov_dim = future_cov_dim
        self.static_cov_dim = static_cov_dim
        self.nr_params = nr_params
        self.shared_weights = shared_weights
        self.const_init = const_init
        self.normalize = normalize

        def _create_linear_layer(in_dim, out_dim):
            layer = nn.Linear(in_dim, out_dim)
            if self.const_init:
                layer.weight = nn.Parameter(
                    (1.0 / in_dim) * torch.ones(layer.weight.shape)
                )
            return layer

        if self.shared_weights:
            layer_in_dim = self.input_chunk_length
            layer_out_dim = self.output_chunk_length * self.nr_params
        else:
            layer_in_dim = self.input_chunk_length * self.input_dim
            layer_out_dim = self.output_chunk_length * self.output_dim * self.nr_params

        self.layer = _create_linear_layer(layer_in_dim, layer_out_dim)

        if self.future_cov_dim != 0:
            # future covariates layer acts on time steps independently
            self.linear_fut_cov = _create_linear_layer(
                self.future_cov_dim, self.output_dim * self.nr_params
            )
        if self.static_cov_dim != 0:
            self.linear_static_cov = _create_linear_layer(
                self.static_cov_dim, layer_out_dim
            )

    @io_processor
    def forward(
        self, x_in: tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ):
        """
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
            is the output/future chunk. Input dimensions are `(n_samples, n_time_steps, n_variables)`
        """
        x, x_future, x_static = x_in  # x: (batch, in_len, in_dim)
        batch, _, _ = x.shape

        if self.shared_weights:
            # discard covariates, to ensure that in_dim == out_dim
            x = x[:, :, : self.output_dim]
            x = x.permute(0, 2, 1)  # (batch, out_dim, in_len)

            if self.normalize:
                seq_last = x[:, :, -1:].detach()  # (batch, out_dim, 1)
                x = x - seq_last

            x = self.layer(x)  # (batch, out_dim, out_len * nr_params)

            if self.normalize:
                x = x + seq_last

            # extract nr_params
            x = x.view(batch, self.output_dim, self.output_chunk_length, self.nr_params)

            # permute back to (batch, out_len, out_dim, nr_params)
            x = x.permute(0, 2, 1, 3)
        else:
            if self.normalize:
                # get last values only for target features
                seq_last = x[:, -1:, : self.output_dim].detach().clone()
                # normalize the target features only (ignore the covariates)
                x[:, :, : self.output_dim] = x[:, :, : self.output_dim] - seq_last

            x = self.layer(x.view(batch, -1))  # (batch, out_len * out_dim * nr_params)
            x = x.view(
                batch, self.output_chunk_length, self.output_dim * self.nr_params
            )

            if self.future_cov_dim != 0:
                # x_future might be shorter than output_chunk_length when n < output_chunk_length
                # so we need to pad it with zeros at the end to match the output_chunk_length
                x_future = torch.nn.functional.pad(
                    input=x_future,
                    pad=(0, 0, 0, self.output_chunk_length - x_future.shape[1]),
                    mode="constant",
                    value=0,
                )

                fut_cov_output = self.linear_fut_cov(x_future)
                x = x + fut_cov_output.view(
                    batch, self.output_chunk_length, self.output_dim * self.nr_params
                )

            if self.static_cov_dim != 0:
                static_cov_output = self.linear_static_cov(x_static.reshape(batch, -1))
                x = x + static_cov_output.view(
                    batch, self.output_chunk_length, self.output_dim * self.nr_params
                )

            x = x.view(batch, self.output_chunk_length, self.output_dim, self.nr_params)
            if self.normalize:
                # model only forecasts target components, no need to slice
                x = x + seq_last.view(seq_last.shape + (1,))
        return x


class NLinearModel(MixedCovariatesTorchModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        shared_weights: bool = False,
        const_init: bool = True,
        normalize: bool = False,
        use_static_covariates: bool = True,
        **kwargs,
    ):
        """An implementation of the NLinear model, as presented in [1]_.

        This implementation is improved by allowing the optional use of past covariates (known for
        `input_chunk_length` points before prediction time), future covariates (known for `output_chunk_length`
        points after prediction time) and static covariates, as well as supporting probabilistic forecasting.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it).
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
            from future covariates to use as a model input (if the model supports future covariates). It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
            using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents
            auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
            the model from using future values of past and / or future covariates for prediction (depending on the
            model's covariate support).
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`).
        shared_weights
            Whether to use shared weights for all components of multivariate series.

            .. warning::
                When set to True, covariates will be ignored as a 1-to-1 mapping is
                required between input dimensions and output dimensions.
            ..

            Default: False.

        const_init
            Whether to initialize the weights to 1/in_len. If False, the default PyTorch
            initialization is used (default='True').
        normalize
            Whether to apply the simple "normalization" proposed in the paper, which consists
            in subtracting the last value of the input sequence from the input sequence. Default: False.

            .. note::
                This cannot be applied to probabilistic models.
            ..
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
            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [2]_.
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
        .. [1] Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2022).
               Are Transformers Effective for Time Series Forecasting?. arXiv preprint arXiv:2205.13504.
        .. [2] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against
                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p

        Examples
        --------
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import NLinearModel
        >>> series = WeatherDataset().load()
        >>> # predicting atmospheric pressure
        >>> target = series['p (mbar)'][:100]
        >>> # optionally, use past observed rainfall (pretending to be unknown beyond index 100)
        >>> past_cov = series['rain (mm)'][:100]
        >>> # optionally, use future temperatures (pretending this component is a forecast)
        >>> future_cov = series['T (degC)'][:106]
        >>> # predict 6 pressure values using the 12 past values of pressure and rainfall, as well as the 6 temperature
        >>> # values corresponding to the forecasted period
        >>> model = NLinearModel(
        >>>     input_chunk_length=6,
        >>>     output_chunk_length=6,
        >>>     n_epochs=20,
        >>> )
        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[429.56117169],
               [428.93264096],
               [428.35210616],
               [428.13154426],
               [427.98781641],
               [428.00325481]])
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

        self.shared_weights = shared_weights
        self.const_init = const_init
        self.normalize = normalize
        self._considers_static_covariates = use_static_covariates

        raise_if(
            "likelihood" in self.model_params
            and self.model_params["likelihood"] is not None
            and self.normalize,
            "normalize = True cannot be used with probabilistic NLinearModel",
        )

    def _create_model(self, train_sample: tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of
        # (past_target, past_covariates, historic_future_covariates,
        #  future_covariates, static_covariates, future_target)

        raise_if(
            self.shared_weights
            and (train_sample[1] is not None or train_sample[2] is not None),
            "Covariates have been provided, but the model has been built with shared_weights=True."
            + "Please set shared_weights=False to use covariates.",
        )

        input_dim = train_sample[0].shape[1] + sum(
            # add past covariates dim and historic future covariates dim, if present
            train_sample[i].shape[1] if train_sample[i] is not None else 0
            for i in (1, 2)
        )
        future_cov_dim = train_sample[3].shape[1] if train_sample[3] is not None else 0

        if train_sample[4] is None:
            static_cov_dim = 0
        else:
            # account for component-specific or shared static covariates representation
            static_cov_dim = train_sample[4].shape[0] * train_sample[4].shape[1]

        output_dim = train_sample[-1].shape[1]

        nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters

        return _NLinearModule(
            input_dim=input_dim,
            output_dim=output_dim,
            future_cov_dim=future_cov_dim,
            static_cov_dim=static_cov_dim,
            nr_params=nr_params,
            shared_weights=self.shared_weights,
            const_init=self.const_init,
            normalize=self.normalize,
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
        return not self.shared_weights

    @property
    def supports_past_covariates(self) -> bool:
        return not self.shared_weights
