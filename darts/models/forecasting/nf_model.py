"""
NeuralForecastModel
-------------------
"""

"""
Throughout this file, we use the following notation for tensor shapes:

    SYMBOL: Darts / NeuralForecast definition
    ------------------------------------------------
    B: batch size / number of windows
    L: input chunk length / input window length
    H: output chunk length / horizon
    C: target components / number of series
    X: past covariate components / historical exogenous variables
    F: future covariate components / future exogenous variables
    S: static covariate components / static exogenous variables (per target component)
    N: likelihood parameters

In NeuralForecast, `BaseModel.forward()` takes a single argument which is a dictionary
containing all inputs. See `BaseModel._parse_windows()` and `BaseModel.training_step()`
to see how these inputs are being built and used.

We thus define the expected keys and their types below:
"""
from typing import Optional, TypedDict

import torch
from neuralforecast.common._base_model import BaseModel
from neuralforecast.losses.pytorch import BasePointLoss

from darts.logging import get_logger, raise_log
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import MixedCovariatesTorchModel
from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample
from darts.utils.likelihood_models.torch import TorchLikelihood

logger = get_logger(__name__)


class _WindowBatch(TypedDict):
    insample_y: torch.Tensor
    insample_mask: torch.Tensor
    # outsample_y: Optional[torch.Tensor]
    # outsample_mask: Optional[torch.Tensor]
    hist_exog: Optional[torch.Tensor]
    futr_exog: Optional[torch.Tensor]
    stat_exog: Optional[torch.Tensor]


IGNORED_NF_MODEL_PARAM_NAMES = {
    "loss",
    "valid_loss",
    "learning_rate",
    "max_steps",
    "val_check_steps",
    "batch_size",
    "valid_batch_size",
    "windows_batch_size",
    "inference_windows_batch_size",
    "start_padding_enabled",
    "training_data_availability_threshold",
    "n_series",
    "n_samples",
    "h_train",
    "inference_input_size",
    "step_size",
    "num_lr_decays",
    "early_stop_patience_steps",
    "scaler_type",
    "futr_exog_list",  # prepared by Darts
    "hist_exog_list",  # prepared by Darts
    "stat_exog_list",  # prepared by Darts
    "exclude_insample_y",  # TODO: check if this should be ignored
    "drop_last_loader",
    "random_seed",  # TODO: check if this should be ignored
    "alias",  # TODO: check if this should be ignored
    "optimizer",
    "optimizer_kwargs",
    "lr_scheduler",
    "lr_scheduler_kwargs",
    "dataloader_kwargs",
}


class _PseudoLoss(BasePointLoss):
    def __init__(self, likelihood: Optional[TorchLikelihood]):
        n_likelihood_params = likelihood.num_parameters if likelihood is not None else 1
        super().__init__(outputsize_multiplier=n_likelihood_params)


class _NFModel(BaseModel):
    """This serves as a protocol for expected NeuralForecast BaseModel API."""

    def forward(self, window_batch: _WindowBatch) -> torch.Tensor: ...


class _PLForecastingModule(PLForecastingModule):
    def __init__(
        self,
        nf_model: _NFModel,
        n_past_covs: int,
        n_future_covs: int,
        is_multivariate: bool,
        **kwargs,
    ):
        """PyTorch Lightning module that wraps around the NeuralForecast model and
        implements the :func:`forward()` API for Darts' ``PLForecastingModule``.

        Parameters
        ----------
        nf_model
            An instance of a NeuralForecast base model.
        n_past_covs
            Number of past covariate components (X).
        n_future_covs
            Number of future covariate components (F).
        is_multivariate
            Whether the NeuralForecast base model is multivariate (i.e., supports C >= 1 target components).
        **kwargs
            all parameters required for :class:`darts.models.forecasting.pl_forecasting_module.PLForecastingModule`
            base class.
        """
        super().__init__(**kwargs)
        self.nf = nf_model
        self.is_multivariate = is_multivariate
        self.past_slice = (
            slice(self.n_targets, self.n_targets + n_past_covs)
            if n_past_covs > 0
            else None
        )
        self.future_slice = (
            slice(
                self.n_targets + n_past_covs,
                self.n_targets + n_past_covs + n_future_covs,
            )
            if n_future_covs > 0
            else None
        )

    @io_processor
    def forward(self, x_in: PLModuleInput):
        """PyTorch-native forward pass.

        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk, `x_future`
            is the output/future chunk, and `x_static` is the static covariates.
            Input dimensions are `(n_samples, n_time_steps, n_variables)` for `x_past` and `x_future`,
            and `(n_samples, n_targets, n_static_covariates)` for `x_static`.

        Returns
        -------
        torch.Tensor
            the output tensor in the shape of `(n_samples, n_time_steps, n_targets, n_likelihood_params)`,
            where `n_likelihood_params` is the number of parameters required by the likelihood model
            (e.g., 2 for Gaussian likelihood with mean and variance) or 1 if no likelihood is specified.
        """
        # unpack inputs
        # `x_past`: (B, L, C + X + F)
        # `x_future`: (B, H, F)
        # `x_static`: (B, C, S)
        x_past, x_future, x_static = x_in

        # build window_batch dict expected by `nf.forward()`
        # Expected shapes in the univariate case (C=1):
        # - `insample_y`: (B, L, C)
        # - `insample_mask`: (B, L)
        # - `hist_exog`: (B, L, X) or None
        # - `futr_exog`: (B, L + H, F) or None
        # - `stat_exog`: (B, C * S) or None
        # Expected shapes in the multivariate case (C >= 1):
        # - `insample_y`: (B, L, C)
        # - `insample_mask`: (B, L)
        # - `hist_exog`: (B, X, L, C) or None
        # - `futr_exog`: (B, F, L + H, C) or None
        # - `stat_exog`: (C, S) or None

        insample_y = x_past[:, :, : self.n_targets]
        insample_mask = torch.ones_like(x_past[:, :, 0])
        hist_exog, futr_exog, stat_exog = None, None, None

        # process past covariates if supported and provided
        if self.past_slice is not None:
            # `hist_exog`: (B, L, X)
            hist_exog = x_past[:, :, self.past_slice]
            if self.is_multivariate:
                # -> (B, X, L, 1)
                hist_exog = hist_exog.transpose(1, 2).unsqueeze(-1)
                # -> (B, X, L, C)
                hist_exog = hist_exog.repeat(1, 1, 1, self.n_targets)

        # process future covariates if supported and provided
        if x_future is not None:
            # `futr_exog`: (B, L + H, F)
            futr_exog = torch.cat([x_past[:, :, self.future_slice], x_future], dim=1)
            if self.is_multivariate:
                # -> (B, F, L + H, 1)
                futr_exog = futr_exog.transpose(1, 2).unsqueeze(-1)
                # -> (B, F, L + H, C)
                futr_exog = futr_exog.repeat(1, 1, 1, self.n_targets)

        # process static covariates if supported and provided
        if x_static is not None:
            if self.is_multivariate:
                # `stat_exog`: (B, C, S) -> (C, S)
                # For multivariate models, NeuralForecast expects `stat_exog` to be of
                # shape (C, S) and shared across the batch dimension,
                # but Darts provides them in shape (B, C, S).
                # Here, we assume that static covariates are the same across each sample
                # in the batch and simply take the first sample's static covariates.
                stat_exog = x_static[0]
            else:
                # `stat_exog`: (B, C * S) [C=1]
                stat_exog = x_static.squeeze(1)

        window_batch = _WindowBatch(
            insample_y=insample_y,
            insample_mask=insample_mask,
            hist_exog=hist_exog,
            futr_exog=futr_exog,
            stat_exog=stat_exog,
        )

        # forward pass through NeuralForecast model
        # `y_pred`: (B, H, C * N)
        y_pred: torch.Tensor = self.nf(window_batch)
        # -> (B, H, C, N)
        y_pred = y_pred.unflatten(-1, (self.n_targets, -1))

        return y_pred


class NeuralForecastModel(MixedCovariatesTorchModel):
    def __init__(
        self,
        model: BaseModel,
        output_chunk_shift: int = 0,
        use_static_covariates: bool = False,
        **kwargs,
    ):
        """NeuralForecast Model.

        Can be used to fit any `NeuralForecast` univariate or multivariate base model.
        For a list of available base models,
        see `NeuralForecast package <https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html>`__.

        This converts the `NeuralForecast` base model into a ``TorchForecastingModel`` and enable full Darts
        functionality, such as covariate support, probabilistic forecasting, backtesting, etc.
        See `Torch Forecasting Models User Guide <https://unit8co.github.io/darts/userguide/torch_forecasting_models.html>`__
        for details and usage examples.

        Our ``NeuralForecastModel`` has the following support, depending on the provided `NeuralForecast` base model:

        - **Univariate forecasting**: Supported for any base model, univariate or multivariate.

          - Simply set ``model`` to a base model instance when initializing ``NeuralForecastModel``,
            e.g., ``model=KAN(input_size=..., h=...)``.

          - For multivariate base models, you should set ``n_series`` in the base model to an arbitrary
            positive integer as it is required by `NeuralForecast`,
            e.g., ``model=TSMixerx(input_size=..., h=..., n_series=1)``.
            However, it will be overridden internally by Darts when fitting to match the number of target components.

        - **Multivariate forecasting**: Supported only if the base model is multivariate.

        - **Past/future covariates**: Supported only if the base model supports exogenous historical/future variables,
          respectively.

        - **Static covariates**: Supported only if the base model supports exogenous static variables:

          - Simply set ``use_static_covariates=True``.

          - For multivariate base models, `NeuralForecast` requires static covariates to be the same across time
            series, but may be different across target components. See the warning below for recommendations.

          - For univariate base models, static covariates can be different across time series.

        - **Multiple time series**: Supported for any base model, univariate or multivariate.

          - Simply pass a sequence of time series as ``series`` to :func:`fit()` and :func:`predict()`.

        - **Loss function**: Supported for any base model, univariate or multivariate.

          - Simply set ``loss_fn`` to a PyTorch loss function (default is ``torch.nn.MSELoss()``).

        - **Probabilistic forecasting**: Supported for any base model, univariate or multivariate.

          - Simply set ``likelihood`` to a :meth:`TorchLikelihood <darts.utils.likelihood_models.torch.TorchLikelihood>`
            instance to be used for probabilistic forecasting.

        - **Output chunk shift**: Supported for any base model, univariate or multivariate.

          - Simply set ``output_chunk_shift`` to create a time gap between the input and output.

        Parameters
        ----------
        model
            An instance of a `NeuralForecast` base model, e.g., ``KAN(input_size=..., h=...)``. Input and output chunk
            lengths are set to match ``input_size`` and ``h`` of the base model, respectively.
        output_chunk_shift
            The number of steps to shift the start of the output chunk into the future (relative to the input
            chunk end). This will create a gap between the input and output. Default: ``0``.
        use_static_covariates
            Whether to consider static covariates if supported by the base model. Default: `False`.
            See **Static covariates** section above for details and caveats.
        **kwargs
            Optional arguments to initialize the ``pytorch_lightning.Module``, ``pytorch_lightning.Trainer``, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for training.
            This parameter will be ignored for probabilistic models if the ``likelihood`` parameter is specified.
            Default: ``torch.nn.MSELoss()``.
        torch_metrics
            A ``torchmetric.Metric`` or a ``MetricCollection`` used for evaluation. A full list of available metrics
            can be found `here <https://torchmetrics.readthedocs.io/en/latest/>`__. Default: ``None``.
        likelihood
            One of Darts' :meth:`TorchLikelihood <darts.utils.likelihood_models.torch.TorchLikelihood>` models to be
            used for probabilistic forecasts. Default: ``None``.
        optimizer_cls
            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{'lr': 1e-3}``
            for specifying a learning rate). Otherwise the default values of the selected ``optimizer_cls``
            will be used. Default: ``None``.
        lr_scheduler_cls
            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None`` corresponds
            to using a constant learning rate. Default: ``None``.
        lr_scheduler_kwargs
            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.
        use_reversible_instance_norm
            Whether to use reversible instance normalization `RINorm` against distribution shift as shown in [1]_.
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
            Controls the randomness of the weights initialization and reproducible forecasting.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful presets
            that performs the training, validation and prediction processes. These presets include automatic
            checkpointing, tensorboard logging, setting the torch device and more.
            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning trainer
            object. Check the `PL Trainer documentation
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for more information about the
            supported kwargs. Default: ``None``.
            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying keys ``"accelerator",
            "devices", and "auto_select_gpus"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``
            dict:

            - ``{"accelerator": "cpu"}`` for CPU,
            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUS.

            For more info, see here:
            `trainer flags
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags>`__,
            and `training on multiple gpus
            <https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus>`__.

            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts'
            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.
            The model will stop training early if the validation loss `val_loss` does not improve beyond
            specifications. For more information on callbacks, visit:
            `PyTorch Lightning Callbacks
            <https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html>`__

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
        .. [1] T. Kim et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against
                Distribution Shift", https://openreview.net/forum?id=cGDAkQo1C0p

        Examples
        --------
        >>> from neuralforecast.models import KAN
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import NeuralForecastModel
        >>> # load the dataset
        >>> series = WeatherDataset().load().astype("float32")
        >>> # predicting temperatures
        >>> target = series['T (degC)'][:100]
        >>> # optionally, use future atmospheric pressure (pretending this component is a forecast)
        >>> future_cov = series['p (mbar)'][:106]
        >>> # create a NeuralForecast base model with `input_size` and `h` (forecast horizon)
        >>> nf_model = KAN(input_size=7, h=6)
        >>> # wrap it in `NeuralForecastModel`
        >>> model = NeuralForecastModel(model=nf_model, n_epochs=20)
        >>> # fit and predict
        >>> model.fit(target, future_covariates=future_cov)
        >>> pred = model.predict(6)
        >>> print(pred.values())
        [[ 1.6961709 ]
        [-2.4282002 ]
        [-0.01969378]
        [ 3.3592758 ]
        [-0.8043982 ]
        [-2.2625582 ]]

        .. note::
            HINT is not supported as it is not a `NeuralForecast` base model.
        .. note::
            Recurrent `NeuralForecast` base models like ``GRU`` and ``LSTM`` are not supported. Many are, however,
            natively implemented as :class:`RNNModel <darts.models.forecasting.rnn_model.RNNModel>` in Darts.
        .. note::
            Training-specific parameters of ``model`` such as ``loss``, ``learning_rate``, and ``hist_exog_list``
            will be ignored as Darts manages them via ``TorchForecastingModel`` APIs. Only architectural
            parameters, ``input_size``, and ``h`` in ``model`` are relevant and used.
        .. note::
            Under the hood, a new base model instance will be created with the relevant parameters from ``model``.
            That means that ``model`` itself will not be trained or updated.
            Use :func:`nf_model` to access the new base model instance.
        .. warning::
            For compatibility, when static covariates are enabled for a multivariate base model, Darts will use the
            static covariates of the first sample in each batch as the static covariates for the entire batch.
            This may cause issues if you have multiple time series with different static covariates.
            Please consider setting ``use_static_covariates=False`` to disable support or
            setting ``batch_size=1`` to ensure that each batch only contains one time series.
        """
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)
        # assign input/output chunk lengths
        self.pl_module_params["input_chunk_length"] = model.input_size
        self.pl_module_params["output_chunk_length"] = model.h
        # NeuralForecast models do not use `output_chunk_shift`, but we still allow users
        # to set it to create a time gap. From the base model's perspective, it is trained
        # to predict the shifted output chunk as if it immediately follows the input chunk.
        if output_chunk_shift > 0:
            logger.warning(
                f"NeuralForecast base models natively do not use `output_chunk_shift`. "
                f"Setting `output_chunk_shift={output_chunk_shift}` will create a time gap "
                f"that is imperceptible to the base model. The model is trained to predict "
                f"the shifted output chunk as if it immediately follows the input chunk."
            )

        self.nf_model_class = model.__class__
        self.nf_model_params = dict(model.hparams)
        self._validate_nf_model_params(
            self.pl_module_params.get("use_reversible_instance_norm", False)
        )

        if self.nf_model_class.RECURRENT:
            raise_log(
                NotImplementedError(
                    "Recurrent NeuralForecast models are currently not supported."
                ),
                logger,
            )
        if self.supports_multivariate and use_static_covariates:
            logger.warning(
                "Multivariate NeuralForecast models require static covariates to be the same "
                "across time series, but may be different across target components. "
                "If you have multiple time series, setting `use_static_covariates=True` "
                "will use the static covariates of the first sample in each batch, instead of "
                "providing different static covariates per time series."
            )

        # consider static covariates if supported by `nf_model_class`
        self._considers_static_covariates = use_static_covariates

    def _validate_nf_model_params(self, use_reversible_instance_norm: bool) -> None:
        ignored_params_in_use = IGNORED_NF_MODEL_PARAM_NAMES.intersection(
            self.nf_model_params.keys()
        )
        # remove ignored params
        if len(ignored_params_in_use) > 0:
            logger.info(
                f"The following NeuralForecast model parameters will be ignored "
                f"as they are either managed by Darts or not relevant: {ignored_params_in_use}"
            )
            for param in ignored_params_in_use:
                self.nf_model_params.pop(param)
        if self.nf_model_params.get("use_norm", False) and use_reversible_instance_norm:
            logger.warning(
                "NeuralForecast model's `use_norm=True` is incompatible with "
                "PLForecastingModule's `use_reversible_instance_norm=True` since they"
                "both apply normalization to the target series. Disabling `use_norm` "
                "to avoid potential issues."
            )
            self.nf_model_params["use_norm"] = False

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        # unpack train sample
        # `past_target`: (L, C)
        # `past_covariates`: (L, X)
        # `historic_future_covariates`: (L, F)
        # `future_covariates`: (H, F)
        # `static_covariates`: (C, S)
        (
            past_target,
            past_covariates,
            historic_future_covariates,
            future_covariates,
            static_covariates,
            future_target,
        ) = train_sample

        # TODO: sanity checks on covariate support of the nf_model_class

        # validate number of target components
        n_targets = future_target.shape[1]
        if n_targets != 1 and not self.supports_multivariate:
            raise_log(
                ValueError(
                    f"The provided {self.nf_model_class.__name__} is a univariate model "
                    f"but the target has {n_targets} component(s)."
                ),
                logger,
            )

        # create pseudo *_exog_list inputs expected by NeuralForecast
        def build_exog_list(prefix: str, n_components: int) -> list[str]:
            return [f"{prefix}_{i}" for i in range(n_components)]

        futr_exog_list, hist_exog_list, stat_exog_list = None, None, None
        n_past_covs, n_future_covs, n_stat_covs = 0, 0, 0
        if future_covariates is not None:
            n_future_covs = future_covariates.shape[1]
            futr_exog_list = build_exog_list("futr_exog", n_future_covs)
        if past_covariates is not None:
            n_past_covs = past_covariates.shape[1]
            hist_exog_list = build_exog_list("hist_exog", n_past_covs)
        if static_covariates is not None:
            n_stat_covs = static_covariates.shape[1]
            stat_exog_list = build_exog_list("stat_exog", n_stat_covs)

        # set loss to pseudo loss with correct number of likelihood parameters
        loss = _PseudoLoss(self.likelihood)

        # initialize nf_model instance
        nf_model = self.nf_model_class(
            **self.nf_model_params,
            loss=loss,
            n_series=n_targets,
            futr_exog_list=futr_exog_list,
            hist_exog_list=hist_exog_list,
            stat_exog_list=stat_exog_list,
        )

        pl_module_params = self.pl_module_params or {}
        return _PLForecastingModule(
            nf_model=nf_model,  # pyright: ignore[reportArgumentType]
            n_past_covs=n_past_covs,
            n_future_covs=n_future_covs,
            is_multivariate=self.supports_multivariate,
            **pl_module_params,
        )

    @property
    def nf_model(self) -> BaseModel:
        if not isinstance(self.model, _PLForecastingModule):
            raise_log(
                ValueError(
                    "The underlying NeuralForecast model has not been created yet."
                )
            )
        return self.model.nf

    @property
    def supports_multivariate(self) -> bool:
        return self.nf_model_class.MULTIVARIATE

    @property
    def supports_past_covariates(self) -> bool:
        return self.nf_model_class.EXOGENOUS_HIST

    @property
    def supports_future_covariates(self) -> bool:
        return self.nf_model_class.EXOGENOUS_FUTR

    @property
    def supports_static_covariates(self) -> bool:
        return self.nf_model_class.EXOGENOUS_STAT
