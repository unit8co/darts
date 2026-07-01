"""
T0: Zero-Shot Forecasting
-------------------------

T0 can be used the same way as other foundation models (e.g. Chronos2). In addition to univariate and
multivariate series, it supports past and future covariates.

For detailed examples and tutorials, see:

* `Foundation Model Examples
  <https://unit8co.github.io/darts/examples/25-FoundationModel-examples.html>`__
* `Fine-Tuning Examples
  <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__
"""

from typing import Any

import torch
from t0 import T0Forecaster

from darts.logging import get_logger, raise_log
from darts.models.forecasting.foundation_model import FoundationModel
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.utils.data.torch_datasets.utils import PLModuleInput
from darts.utils.likelihood_models.torch import QuantileRegression

logger = get_logger(__name__)


class _T0Module(PLForecastingModule):
    """PyTorch Lightning module wrapping a pre-loaded T0 forecaster.

    Adapts T0's ``predict`` interface to Darts' ``PLForecastingModule`` API. Targets and past covariates are
    forecast jointly (past-covariate predictions are dropped). Future covariates are mapped to T0's
    ``[batch, n_covariates, context + horizon]`` format.
    """

    def __init__(
        self,
        t0_kwargs: dict[str, Any],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.t0: T0Forecaster = T0Forecaster.from_pretrained(**t0_kwargs).eval()
        self.future_len = (self.output_chunk_length or 0) + self.output_chunk_shift

    def forward(self, x_in: PLModuleInput, *args, **kwargs):
        """Forward pass returning quantile predictions shaped ``(batch, time, n_targets, n_quantiles)``."""
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
        #   N: likelihood quantiles (user-specified, 1 if deterministic)

        # `x_past`: (B, L, C + P + F) stack of [past_target, past_covariates, historic_future_covariates];
        # `x_future`: (B, T, F) future covariates, or None.
        x_past, x_future, _, _ = x_in
        batch_size = x_past.shape[0]

        # Past covariates are forecast jointly with the target (T0 is variate-agnostic) and dropped from the
        # output. Future covariates are the trailing columns of `x_past` (their historic part) and are passed to
        # T0's covariate branch instead. `x_future` width gives the number of future covariates.
        n_future_covs = x_future.shape[-1] if x_future is not None else 0
        n_context = x_past.shape[-1] - n_future_covs

        # context: (B, V, L)
        context = x_past[:, :, :n_context].transpose(1, 2)

        # T0 expects covariates over context + horizon. Re-assemble them from the historic part (in `x_past`)
        # and the future chunk (`x_future`); the `output_chunk_shift` gap is left NaN (T0 treats NaN as missing).
        future_covariates = None
        if n_future_covs > 0:
            historic = x_past[:, :, n_context:]  # (B, L, F)
            future = torch.full(
                (batch_size, self.future_len, n_future_covs),
                torch.nan,
                device=x_past.device,
                dtype=x_past.dtype,
            )
            if x_future is not None:
                future[:, -(self.output_chunk_length or 0) :, :] = x_future
            # (B, L + H, F) -> (B, F, L + H)
            future_covariates = torch.cat([historic, future], dim=1).transpose(1, 2)

        user_q: list[float] = (
            self.likelihood.quantiles
            if isinstance(self.likelihood, QuantileRegression)
            else [0.5]
        )
        # quantiles: (B, V, H, N)
        quantiles = self.t0.predict(
            context,
            horizon=self.future_len,
            quantiles=user_q,
            future_covariates=future_covariates,
        ).quantiles
        # drop the past-covariate variates, keep targets: (B, V, H, N) -> (B, C, H, N)
        quantiles = quantiles[:, : self.n_targets]
        # (B, C, H, N) -> (B, H, C, N) -> slice output shift -> (B, T, C, N)
        return quantiles.permute(0, 2, 1, 3)[:, self.output_chunk_shift :, :, :]


class T0Model(FoundationModel):
    # Quantile levels T0 was trained on. Other levels are interpolated, so any quantiles in (0, 1) are accepted.
    _PRETRAINED_QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        likelihood: QuantileRegression | None = None,
        hub_model_name: str = "theforecastingcompany/t0-alpha",
        hub_model_revision: str | None = None,
        **kwargs,
    ):
        """
        T0 foundation model for zero-shot time series forecasting.

        This is a Darts wrapper around The Forecasting Company's open-weights T0 model. The implementation delegates
        all forecasting logic and weight loading to the optional `tfc-t0 <https://pypi.org/project/tfc-t0>`_ package
        while exposing a standard :class:`TorchForecastingModel` interface.

        T0 is a ~100M-parameter pre-trained patch-transformer foundation model designed for zero-shot forecasting
        across both short and long horizons.

        This model supports univariate and multivariate time series, as well as past and future covariates.
        Because T0 is variate-agnostic, past covariates are forecast jointly with the target series (and dropped
        from the output); future covariates are conditioned on but not forecast.

        By default, the model is deterministic (median forecast only). To enable probabilistic forecasts, pass a
        :class:`~darts.utils.likelihood_models.torch.QuantileRegression` instance to the ``likelihood`` parameter.
        It is recommended to call :func:`predict()` with ``predict_likelihood_parameters=True`` or ``num_samples >> 1``
        to get meaningful results. T0 was trained on quantile levels [0.1, 0.25, 0.5, 0.75, 0.9]; other levels are
        interpolated, so any quantiles in the open interval (0, 1) may be requested.

        For more details on the T0 model, see the `model card <https://huggingface.co/theforecastingcompany/t0-alpha>`_
        and the `tfc-t0 repository <https://github.com/theforecastingcompany/tfc-t0>`_.

        .. note::
            Fine-tuning is not supported for ``T0Model``; the model is used for zero-shot inference only.

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
        likelihood
            The likelihood model to be used for probabilistic forecasts. Must be ``None`` or an instance of
            :class:`~darts.utils.likelihood_models.torch.QuantileRegression`. Any quantiles in the open interval
            (0, 1) are supported (T0 interpolates levels it was not trained on). Default: ``None``, which will make
            the model deterministic (median quantile only).
        hub_model_name
            The model ID on HuggingFace Hub. Default: ``"theforecastingcompany/t0-alpha"``.
        hub_model_revision
            The model version to use. This can be a branch name, tag name, or commit hash. Default: ``None``, which
            will use the default branch from ``hub_model_name``.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        torch_metrics
            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics can be found
            at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.
        batch_size
            Number of time series (input and output sequences) used in each prediction pass. Default: ``32``.
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
            If set, use Tensorboard to log the different parameters. The logs will be located in:
            ``"{work_dir}/darts_logs/{model_name}/logs/"``. Default: ``False``.
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
            Controls the randomness of reproducible forecasting.
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
            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus
        show_warnings
            whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of
            your forecasting use case. Default: ``False``.

        References
        ----------
        .. [1] The Forecasting Company, "T0", https://huggingface.co/theforecastingcompany/t0-alpha.

        Examples
        --------
        Point forecasting:

        >>> from darts.models import T0Model
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load().astype("float32")
        >>> model = T0Model(input_chunk_length=24, output_chunk_length=12)
        >>> model.fit(series)
        >>> pred = model.predict(n=12)

        Probabilistic forecasting:

        >>> from darts.utils.likelihood_models import QuantileRegression
        >>> model = T0Model(
        ...     input_chunk_length=24,
        ...     output_chunk_length=12,
        ...     likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        ... )
        >>> model.fit(series)
        >>> pred = model.predict(n=12, predict_likelihood_parameters=True)
        """
        if likelihood is not None and not isinstance(likelihood, QuantileRegression):
            raise_log(
                ValueError(
                    f"Only QuantileRegression likelihood is supported for T0 in Darts. "
                    f"Got {type(likelihood)}."
                ),
            )

        if kwargs.get("enable_finetuning"):
            raise_log(
                ValueError(
                    "Fine-tuning is not supported for `T0Model`; it is a zero-shot inference model. "
                    "Leave `enable_finetuning` unset (or `False`)."
                ),
            )

        super().__init__(**kwargs)

        self.t0_kwargs = {
            "pretrained_model_name_or_path": hub_model_name,
            **(
                {"revision": hub_model_revision}
                if hub_model_revision is not None
                else {}
            ),
        }

    @property
    def supports_past_covariates(self) -> bool:
        return True

    @property
    def supports_future_covariates(self) -> bool:
        return True

    def _create_model(self, train_sample) -> PLForecastingModule:
        return _T0Module(t0_kwargs=self.t0_kwargs, **(self.pl_module_params or {}))
