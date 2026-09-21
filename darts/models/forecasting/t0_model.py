"""
T0: Zero-Shot Forecasting
-------------------------

T0 can be used the same way as other foundation models (e.g. Chronos2). In addition to univariate and
multivariate series, it supports past and future covariates.

The weights are published on the Hugging Face Hub at `theforecastingcompany/t0-alpha
<https://huggingface.co/theforecastingcompany/t0-alpha>`__ and `theforecastingcompany/t0-beta
<https://huggingface.co/theforecastingcompany/t0-beta>`__ (more checkpoints might be available in the future).
The repository is gated: request access on the model page and authenticate (``hf auth login`` via CLI or set
``HF_TOKEN``) before first use.

For detailed examples and tutorials, see:

* `Foundation Model Examples
  <https://unit8co.github.io/darts/examples/25-FoundationModel-examples.html>`__
* `Fine-Tuning Examples
  <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__
"""

import dataclasses
import os

import torch
from t0 import T0Config, T0Forecaster
from t0 import TimeSeries as T0TimeSeries
from t0.data import VariateType

from darts.logging import get_logger, raise_log
from darts.models.components.huggingface_connector import HuggingFaceConnector
from darts.models.forecasting.foundation_model import FoundationModel
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.utils.data import ModuleStage, PLModuleOutput, TorchTrainingSample
from darts.utils.data.torch_datasets.utils import PLModuleInput
from darts.utils.likelihood_models.torch import QuantileRegression

logger = get_logger(__name__)


class _T0Module(PLForecastingModule):
    def __init__(
        self,
        hub_model_name: str,
        hub_model_revision: str | None,
        local_dir: str | os.PathLike | None,
        **kwargs,
    ):
        """PyTorch Lightning module wrapping a pre-loaded T0 forecaster.

        Targets and past covariates are forecast jointly, and past-covariate predictions are dropped.
        Future covariates are mapped to T0's ``[batch, n_covariates, context + horizon]`` format.

        Fine-tuning runs T0's forward pass over all pre-trained quantiles; prediction returns only the
        user-specified ones.

        Parameters
        ----------
        hub_model_name
            The HuggingFace repository name where the model is stored.
        hub_model_revision
            The revision of the model in the HuggingFace repository. Must be a branch name, tag name, or commit hash.
            If not provided, the default branch and the latest commit will be used.
        local_dir
            Optional local directory to load the pre-downloaded model. If specified and the directory is empty, the
            model will be downloaded from HuggingFace Hub and saved to this directory. Default is ``None``, which will
            use a cache directory managed by ``huggingface_hub`` instead.
        **kwargs
            Additional arguments passed to `PLForecastingModule`.
        """
        enable_finetuning = kwargs.pop("enable_finetuning", False)
        super().__init__(**kwargs)

        # rebuild T0 from the Hub config, then load the weights into it
        connector = HuggingFaceConnector(
            model_name=hub_model_name,
            model_revision=hub_model_revision,
            local_dir=local_dir,
        )
        config = connector.load_config()
        config_kwargs = {
            field.name: config[field.name]
            for field in dataclasses.fields(T0Config)
            if field.name in config
        }
        if "quantile_levels" in config_kwargs:
            config_kwargs["quantile_levels"] = tuple(config_kwargs["quantile_levels"])
        if enable_finetuning:
            # set dropout to 0, which would otherwise inject noise into a frozen backbone
            config_kwargs["dropout"] = 0.0
        config = T0Config(**config_kwargs)

        self.t0: T0Forecaster = T0Forecaster.from_config(config)
        self.future_len = (self.output_chunk_length or 0) + self.output_chunk_shift
        if enable_finetuning and self.future_len > self.t0.max_horizon:
            raise_log(
                ValueError(
                    f"T0 fine-tuning only supports up to {self.t0.max_horizon} steps got {self.future_len}"
                ),
            )

        connector.load_model_weights(self.t0)

        # gather user-specified quantiles used at prediction time
        self.user_quantiles: list[float] = (
            self.likelihood.quantiles
            if isinstance(self.likelihood, QuantileRegression)
            else [0.5]
        )

        if enable_finetuning:
            # loss is computed over all pre-trained quantiles to preserve the distribution;
            # user-specified quantiles are selected at prediction time
            self._finetuning_likelihood = QuantileRegression(
                list(config.quantile_levels)
            )
        else:
            self._finetuning_likelihood = None

    def forward(self, x_in: PLModuleInput) -> PLModuleOutput:
        # Dimension notation in comments below:
        #   B: batch size
        #   L: input chunk length
        #   T: output chunk length
        #   S: output chunk shift
        #   H: future length = T + S
        #   C: target components
        #   P: past covariate components
        #   F: future covariate components
        #   V: context variates = C + P (target + past covariates, jointly forecast)
        #   Qp: pre-trained quantiles (returned during fine-tuning)
        #   N: likelihood quantiles (user-specified, 1 if deterministic)

        # Past covariates are forecast jointly with the target (T0 is variate-agnostic) and dropped from the output.
        # context: (B, V, L)
        context = torch.cat(
            [el for el in [x_in.past_target, x_in.past_covariates] if el is not None],
            dim=2,
        ).transpose(1, 2)
        batch_size, n_context, _ = context.shape

        # Future covariates and their historic part are passed to T0's covariate branch.
        # T0 expects covariates over context (historic) + horizon (future);
        # the `output_chunk_shift` gap is left NaN (T0 treats NaN as missing).
        historic_future_covariates = x_in.historic_future_covariates
        future_covariates = x_in.future_covariates
        if historic_future_covariates is not None and future_covariates is not None:
            if self.output_chunk_shift == 0:
                future_covariates = x_in.concatenate_future_along_time()
            else:
                n_future_covs = future_covariates.shape[-1]
                gap = torch.full(
                    (batch_size, self.output_chunk_shift, n_future_covs),
                    torch.nan,
                    device=context.device,
                    dtype=context.dtype,
                )
                future_covariates = torch.cat(
                    [historic_future_covariates, gap, future_covariates], dim=1
                )
            future_covariates = future_covariates.transpose(1, 2)

        if x_in.stage is ModuleStage.TRAIN:
            # scale the input, run one forward pass, and rescale the predictions
            patch_size = self.t0.patch_size
            context_length = context.shape[-1]
            model_input = self.t0.patcher.pad(
                T0TimeSeries.from_array(
                    context, future_covariates, horizon=self.future_len
                )
            )
            scaled, loc_scale = self.t0.scaler.scale_input(model_input)
            # (rows, patches, patch_size, Qp) in data space
            patches = self.t0.scaler.rescale_predictions(
                self.t0(scaled), loc_scale, patch_size
            )
            # each position predicts the patch that follows it, so the forecast starts at the last
            # context patch; ceil-divide to count whole patches
            first = -(-context_length // patch_size) - 1
            n_patches = -(-self.future_len // patch_size)
            is_target = model_input.variate_type[:, -1] == VariateType.TARGET
            # -> (B * V, H, Qp) -> (B, V, H, Qp), matching `predict()`'s multivariate layout
            quantiles = (
                patches[is_target][:, first : first + n_patches]
                .flatten(1, 2)[:, : self.future_len]
                .unflatten(0, (batch_size, n_context))
            )
        else:
            # quantiles: (B, V, H, N)
            quantiles = self.t0.predict(
                context,
                horizon=self.future_len,
                quantile_levels=self.user_quantiles,
                future_covariates=future_covariates,
            ).quantiles
        # drop the past-covariate variates, keep targets: (B, V, H, N) -> (B, C, H, N)
        quantiles = quantiles[:, : self.n_targets]
        # (B, C, H, N) -> (B, H, C, N) -> slice output shift -> (B, T, C, N)
        quantiles = quantiles.permute(0, 2, 1, 3)[:, self.output_chunk_shift :, :, :]
        return PLModuleOutput(prediction=quantiles)

    def _compute_loss(self, output, target, criterion, sample_weight):
        if self.training:
            # compute loss on pre-trained quantiles
            return self._finetuning_likelihood.compute_loss(
                output.prediction, target, sample_weight
            )
        else:
            return super()._compute_loss(output, target, criterion, sample_weight)


class T0Model(FoundationModel):
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        likelihood: QuantileRegression | None = None,
        hub_model_name: str = "theforecastingcompany/t0-beta",
        hub_model_revision: str | None = None,
        local_dir: str | os.PathLike | None = None,
        **kwargs,
    ):
        """
        T0 foundation model for zero-shot time series forecasting.

        This is a Darts wrapper around The Forecasting Company's open-weights T0 model [1]_, [2]_. The implementation
        delegates all forecasting logic to the optional `tfc-t0 <https://pypi.org/project/tfc-t0>`_ package while
        exposing a standard :class:`TorchForecastingModel` interface.

        T0 is a pre-trained patch-transformer foundation model designed for zero-shot forecasting across both short and
        long horizons. Multiple checkpoints are available via ``hub_model_name``:

        - `theforecastingcompany/t0-alpha <https://huggingface.co/theforecastingcompany/t0-alpha>`_: The original
          ~100M-parameter T0 model.
        - `theforecastingcompany/t0-beta <https://huggingface.co/theforecastingcompany/t0-beta>`_ (default): The second
          256M-parameter model iteration

        This model supports univariate and multivariate time series, as well as past and future covariates.
        Because T0 is variate-agnostic, past covariates are forecast jointly with the target series (and dropped
        from the output); future covariates are conditioned on but not forecast.

        Using this model will automatically download and cache the pre-trained ``t0-beta`` model from HuggingFace Hub.
        Alternatively, you can specify a local directory containing the model config and weights using the ``local_dir``
        parameter.

        By default, the model is deterministic (median forecast only). To enable probabilistic forecasts, pass a
        :class:`~darts.utils.likelihood_models.torch.QuantileRegression` instance to the ``likelihood`` parameter.
        It is recommended to call :func:`predict()` with ``predict_likelihood_parameters=True`` or ``num_samples >> 1``
        to get meaningful results. T0 was trained on quantile levels [0.1, 0.25, 0.5, 0.75, 0.9]; other levels are
        interpolated, so any quantiles in the open interval (0, 1) may be requested.

        .. tip::
            You can perform full or partial fine-tuning of the model by setting the ``enable_finetuning`` parameter.
            Read more in the parameter description below and in the `Fine-Tuning Examples
            <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__.
            Fine-tuning runs a single forward pass, so ``output_chunk_length`` plus ``output_chunk_shift`` must not
            exceed T0's ``max_horizon`` (1024). Dropout is disabled so a frozen backbone stays deterministic.
        .. note::
            The T0 weights are open but the Hub repository is gated, so request access on the model card and
            authenticate before first use.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it).
            Can be either an ``int`` for a fixed input window, or a ``(min_length, max_length)`` tuple to enable
            variable-length inputs for inference and fine-tuning.
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
            cannot generate autoregressive predictions (`n > output_chunk_length`). The covariate values inside the
            gap are unknown to the model and are masked out.
        likelihood
            The likelihood model to be used for probabilistic forecasts. Must be ``None`` or an instance of
            :class:`~darts.utils.likelihood_models.torch.QuantileRegression`. For zero-shot predicitons, any quantiles
            in the open interval (0, 1) are supported (T0 interpolates levels it was not trained on).
            Default: ``None``, which will make the model deterministic (median quantile only).
            When fine-tuning is enabled, the training loss is always computed on all pre-trained quantiles to
            preserve the full distribution, regardless of the ``likelihood`` setting. The ``likelihood`` parameter
            only affects prediction output.
        hub_model_name
            The model ID on HuggingFace Hub. Default: ``"theforecastingcompany/t0-beta"``.
        hub_model_revision
            The model version to use. This can be a branch name, tag name, or commit hash. Default is ``None``, which
            will use the default branch from ``hub_model_name``.
        local_dir
            Optional local directory to load the pre-downloaded model. If specified and the directory is empty, the
            model will be downloaded from HuggingFace Hub and saved to this directory. Default is ``None``, which will
            use a cache directory managed by ``huggingface_hub`` instead. Note that this is different from the
            ``work_dir`` parameter used for saving model checkpoints during fine-tuning.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for fine-tuning a deterministic model. Ignored for probabilistic models when
            ``likelihood`` is specified. Default: ``nn.MSELoss()``.
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
            of the name is formatted with the local date and time, while PID is the process ID (preventing models
            spawned at the same time by different processes to share the same model_name). E.g.,
            ``"2021-06-14_09_53_32_torch_model_run_44607"``.
        work_dir
            Path of the working directory, where to save checkpoints and Tensorboard summaries.
            Default: current working directory.
        log_tensorboard
            If set, use Tensorboard to log the different parameters. The logs will be located in
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
            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUs.

            For more info, see here:
            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and
            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#accelerators-gpu-basic

            With parameter ``"callbacks"`` you can add custom or PyTorch-Lightning built-in callbacks to Darts'
            :class:`TorchForecastingModel`. Below is an example for adding EarlyStopping to the training process.
            The model will stop training early if the validation loss `val_loss` does not improve beyond
            specifications.

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
        enable_finetuning
            Enables model fine-tuning. Only effective if not ``None``.
            If a bool, specifies whether to perform full fine-tuning / training (all parameters are updated) or keep
            all parameters frozen. If a dict, specifies which parameters to fine-tune. Must only contain one key-value
            record. Can be used to:

            - Unfreeze specific parameters, while keeping everything else frozen:
              ``{"unfreeze": ["param.name.patterns.*"]}``
            - Freeze specific parameters, while keeping everything else unfrozen:
              ``{"freeze": ["param.name.patterns.*"]}``

            Default: ``None``.

        References
        ----------
        .. [1] The Forecasting Company. "T0". https://huggingface.co/theforecastingcompany/t0-alpha.
        .. [2] The Forecasting Company. "t0: A Time-Series Foundation Model for Forecasting with Context".
               https://www.theforecastingcompany.com/papers/files/t0-technical-report.pdf.

        Examples
        --------
        Point forecasting:

        >>> from darts.models import T0Model
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load().astype("float32")
        >>> model = T0Model(input_chunk_length=12, output_chunk_length=6)
        >>> model.fit(series)
        >>> pred = model.predict(n=6)
                    #Passengers
        Month
        1961-01-01   449.567627
        1961-02-01   452.335510
        1961-03-01   452.222931
        1961-04-01   455.815308
        1961-05-01   456.107849
        1961-06-01   454.145569

        Probabilistic forecasting:

        >>> from darts.utils.likelihood_models import QuantileRegression
        >>> model = T0Model(
        >>>     input_chunk_length=12,
        >>>     output_chunk_length=6,
        >>>     likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        >>> )
        >>> model.fit(series)
        >>> pred = model.predict(n=6, predict_likelihood_parameters=True)
                    #Passengers_q0.100  #Passengers_q0.500  #Passengers_q0.900
        Month
        1961-01-01          374.172150          449.567627          544.139648
        1961-02-01          342.856903          452.335510          600.989746
        1961-03-01          324.758545          452.222931          636.924072
        1961-04-01          307.714386          455.815308          658.392700
        1961-05-01          295.556030          456.107849          690.342773
        1961-06-01          286.393066          454.145569          702.022705
        """
        if likelihood is not None and not isinstance(likelihood, QuantileRegression):
            raise_log(
                ValueError(
                    f"Only QuantileRegression likelihood is supported for T0 in Darts. "
                    f"Got {type(likelihood)}."
                ),
            )

        super().__init__(**kwargs)

        self._hub_model_name = hub_model_name
        self._hub_model_revision = hub_model_revision
        self._local_dir = local_dir

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        # enable_finetuning is injected into pl_module_params by the base class;
        # _T0Module accepts it as an explicit parameter and converts dict form to bool
        return _T0Module(
            hub_model_name=self._hub_model_name,
            hub_model_revision=self._hub_model_revision,
            local_dir=self._local_dir,
            **(self.pl_module_params or {}),
        )
