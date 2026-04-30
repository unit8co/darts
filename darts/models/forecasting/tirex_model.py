"""
TiRex: Zero-Shot Forecasting
----------------------------

TiRex can be used the same way as other foundation models (e.g. Chronos2), with the exception
that it does not support any type of covariates.

For detailed examples and tutorials, see:

* `Foundation Model Examples
  <https://unit8co.github.io/darts/examples/25-FoundationModel-examples.html>`__
* `Fine-Tuning Examples
  <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__
"""

import os
from typing import Any

import torch
from tirex import load_model
from tirex.models.tirex import TiRexZero

from darts.logging import get_logger, raise_log
from darts.models.forecasting.foundation_model import FoundationModel
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.utils.likelihood_models.torch import QuantileRegression

logger = get_logger(__name__)


class _TiRexModule(PLForecastingModule):
    """PyTorch Lightning module wrapping a pre-loaded TiRex pipeline.

    Adapts TiRex's forecasting interface to Darts' PLForecastingModule API. Multivariate inputs are supported by
    folding components into the batch dimension. Covariates are not supported.

    During fine-tuning, the gradient-enabled ``_forecast_tensor`` path is used, bypassing the
    ``@torch.inference_mode()`` decorator on ``_forecast_quantiles``. Loss is computed over all 9 pre-trained
    quantiles; only user-specified quantiles are returned at prediction time.
    """

    _user_quantile_indices: torch.Tensor

    def __init__(
        self,
        tirex_kwargs: dict[str, Any],
        all_quantiles: tuple[float, ...],
        enable_finetuning: bool | dict = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        tirex_kwargs["device"] = str(self.device)
        self.tirex: TiRexZero = load_model(**tirex_kwargs)

        self.future_len = (self.output_chunk_length or 0) + self.output_chunk_shift
        # bool(dict) is True for a non-empty dict; _setup_finetuning() handles the actual
        # parameter freeze/unfreeze pattern — here we only need a flag for the forward path
        self._enable_finetuning = bool(enable_finetuning)

        user_q: list[float] = (
            self.likelihood.quantiles
            if isinstance(self.likelihood, QuantileRegression)
            else [0.5]
        )
        self.register_buffer(
            "_user_quantile_indices",
            torch.tensor(
                [all_quantiles.index(q) for q in user_q],
                dtype=torch.long,
                device=self.device,
            ),
        )

        if enable_finetuning:
            # loss is computed over all pre-trained quantiles to preserve the distribution;
            # user-specified quantiles are selected at prediction time
            self._finetuning_likelihood = QuantileRegression(list(all_quantiles))
        else:
            self._finetuning_likelihood = None

    def forward(self, x_in, *args, **kwargs):
        """Forward pass returning quantile predictions shaped ``(batch, time, n_targets, n_quantiles)``.

        During training with fine-tuning enabled, all 9 pre-trained quantiles are returned
        for the loss. At prediction time, only user-specified quantiles are returned.
        """
        # Dimension notation in comments below:
        #   B: batch size
        #   L: input chunk length
        #   T: output chunk length
        #   S: output chunk shift
        #   H: forecast horizon = T + S
        #   C: target components
        #   Q = 9: likelihood quantiles returned by TiRex
        #   N: likelihood quantiles (user-specified, 1 if deterministic)

        # `x_past`: (B, L, C)
        x_past, _, _ = x_in
        # fold target components into batch dim for multivariate support: (B, L, C) -> (B*C, L)
        x_past = x_past.transpose(1, 2).flatten(start_dim=0, end_dim=1)

        if self.training and self._enable_finetuning:
            # call _forecast_tensor directly to keep gradients flowing — _forecast_quantiles
            # is decorated with @torch.inference_mode() which would block backprop
            # output: (B*C, Q, H) -> swapaxes -> (B*C, H, Q) -> slice output shift -> (B*C, T, Q)
            q_sel = self.tirex._forecast_tensor(
                context=x_past,
                prediction_length=self.future_len,
            ).swapaxes(1, 2)[:, self.output_chunk_shift :, :]
        else:
            quantiles, _ = self.tirex._forecast_quantiles(
                context=x_past,
                prediction_length=self.future_len,
                output_device=str(x_past.device),
            )
            # (B*C, H, Q) -> slice output shift -> select user quantiles -> (B*C, T, N)
            q_sel = quantiles[:, self.output_chunk_shift :, :].index_select(
                dim=-1, index=self._user_quantile_indices
            )

        # unfold batch dim and permute to Darts' output shape: (B, T, C, N or Q)
        return q_sel.unflatten(dim=0, sizes=(-1, self.n_targets)).permute(0, 2, 1, 3)

    def _compute_loss(self, output, target, criterion, sample_weight):
        if self.training and self._enable_finetuning:
            # compute loss on pre-trained quantiles
            return self._finetuning_likelihood.compute_loss(
                output, target, sample_weight
            )
        return super()._compute_loss(output, target, criterion, sample_weight)


class TiRexModel(FoundationModel):
    _DEFAULT_QUANTILES: tuple[float, ...] = (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    )
    _MAX_PREDICTION_LENGTH = 2048

    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        accept_license: bool = False,
        likelihood: QuantileRegression | None = None,
        hub_model_name: str = "NX-AI/TiRex",
        hub_model_revision: str | None = None,
        local_dir: str | os.PathLike | None = None,
        tirex_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        TiRex foundation model for zero-shot time series forecasting.

        This is a Darts wrapper around the TiRex model introduced in Auer et al. (2025) [1]_. The implementation
        delegates all forecasting logic and weight loading to the optional `tirex-ts
        <https://pypi.org/project/tirex-ts>`_ package while exposing a standard :class:`TorchForecastingModel`
        interface.

        TiRex is a pre-trained foundation model designed for zero-shot forecasting across both short and long horizons.

        This model supports either univariate or multivariate time series, but does not support covariates.
        For multivariate time series, the model is applied independently to each component.

        By default, the model is deterministic (median forecast only). To enable probabilistic forecasts, pass a
        :class:`~darts.utils.likelihood_models.torch.QuantileRegression` instance to the ``likelihood`` parameter.
        It is recommended to call :func:`predict()` with ``predict_likelihood_parameters=True`` or ``num_samples >> 1``
        to get meaningful results.

        For more details on the TiRex model, see the original paper [1]_, the `TiRex base repo
        <https://github.com/NX-AI/tirex>`_, `model card <https://huggingface.co/NX-AI/TiRex>`_, and `docs
        <https://nx-ai.github.io/tirex/>`_.

        .. note::
            TiRex is distributed under the `NXAI Community License <https://github.com/NX-AI/tirex/blob/main/LICENSE>`_.
            You must explicitly acknowledge this license by passing ``accept_license=True`` when constructing the model.

        .. note::
            Partial fine-tuning is supported via
            ``enable_finetuning={"unfreeze": ["tirex.output_patch_embedding*", ...]}``. Fine-tuning requires
            ``tirex_kwargs={"backend": "torch"}``; only the last sLSTM blocks and the output head are gradient-safe
            (see notebook for recommended configurations). Full fine-tuning (``enable_finetuning=True``) is
            **not supported** — backpropagation through the early sLSTM blocks produces NaN gradients.

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
            For TiRex, `output_chunk_length + output_chunk_shift` must be less than or equal to ``2048``.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`).
        likelihood
            The likelihood model to be used for probabilistic forecasts. Must be ``None`` or an instance of
            :class:`~darts.utils.likelihood_models.torch.QuantileRegression`. If using ``QuantileRegression``,
            the quantiles must be a subset of those used during Chronos-2 pre-training:
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
            Default: ``None``, which will make the model deterministic (median quantile only).
            When fine-tuning is enabled, the training loss is always computed on all pre-trained quantiles to
            preserve the full distribution, regardless of the ``likelihood`` setting. The ``likelihood`` parameter
            only affects prediction output.
        accept_license
            Must be set to ``True`` to confirm acceptance of the NXAI Community License. Default: ``False``.
        hub_model_name
            The model ID on HuggingFace Hub. Default: ``"NX-AI/TiRex"``.
        hub_model_revision
            The model version to use. This can be a branch name, tag name, or commit hash. Default: ``None``, which
            will use the default branch from ``hub_model_name``.
        local_dir
            Optional local directory to load the pre-downloaded model. If specified and the directory is empty, the
            model will be downloaded from HuggingFace Hub and saved to this directory. Default is ``None``, which will
            use a cache directory managed by ``huggingface_hub`` instead. Note that this is different from the
            ``work_dir`` parameter used for saving model checkpoints during fine-tuning.
        tirex_kwargs
            Additional keyword arguments forwarded to ``tirex.load_model()``.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for fine-tuning a deterministic Chronos-2 model. Ignored for probabilistic
            Chronos-2 when ``likelihood`` is specified. Default: ``nn.MSELoss()``.
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
            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags , and
            https://pytorch-lightning.readthedocs.io/en/stable/accelerators/gpu_basic.html#train-on-multiple-gpus

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
        .. [1] A. Auer et al., "TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context
                Learning", NeurIPS 2025. https://arxiv.org/abs/2505.23719.

        Examples
        --------
        >>> from darts.models import TiRexModel
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load().astype("float32")
        >>> # you must explicitly set `accept_license=True` to use the model
        >>> model = TiRexModel(
        ...     input_chunk_length=12,
        ...     output_chunk_length=6,
        ...     accept_license=False,
        ... )
        >>> model.fit(series)
        >>> pred = model.predict(n=6)
        >>> print(pred.all_values())
        [[[440.2505 ]]
        [[444.44373]]
        [[447.36362]]
        [[451.50375]]
        [[458.05853]]
        [[461.98694]]]

        Probabilistic forecasting:

        >>> from darts.utils.likelihood_models import QuantileRegression
        >>> # you must explicitly set `accept_license=True` to use the model
        >>> model = TiRexModel(
        ...     input_chunk_length=12,
        ...     output_chunk_length=6,
        ...     likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        ...     accept_license=False,
        ... )
        >>> model.fit(series)
        >>> pred = model.predict(n=6, predict_likelihood_parameters=True)
        """
        if not accept_license:
            raise_log(
                ValueError(
                    "TiRex is distributed under the NXAI Community License. "
                    "Set `accept_license=True` to confirm you have reviewed and accept the terms: "
                    "https://github.com/NX-AI/tirex/blob/main/LICENSE"
                ),
                logger,
            )

        if likelihood is not None:
            if not isinstance(likelihood, QuantileRegression):
                raise_log(
                    ValueError(
                        f"Only QuantileRegression likelihood is supported for TiRex in Darts. "
                        f"Got {type(likelihood)}."
                    ),
                    logger,
                )
            user_quantiles: list[float] = likelihood.quantiles
            if not set(user_quantiles).issubset(self._DEFAULT_QUANTILES):
                raise_log(
                    ValueError(
                        f"The quantiles for QuantileRegression likelihood {user_quantiles} "
                        f"must be a subset of TiRex quantiles {self._DEFAULT_QUANTILES}."
                    ),
                    logger,
                )

        if output_chunk_length + output_chunk_shift > self._MAX_PREDICTION_LENGTH:
            raise_log(
                ValueError(
                    f"`output_chunk_length` {output_chunk_length} plus `output_chunk_shift` {output_chunk_shift} "
                    f"cannot be greater than model's maximum prediction length {self._MAX_PREDICTION_LENGTH}"
                ),
                logger,
            )

        tirex_kwargs = tirex_kwargs or {}
        if "path" in tirex_kwargs:
            raise_log(
                ValueError(
                    "The `path` argument for loading the TiRex model should be passed via `hub_model_name`,"
                    "not `tirex_kwargs`."
                ),
                logger,
            )

        if kwargs.get("enable_finetuning", None):
            logger.info(
                "TiRex is licensed under the NXAI Community License — "
                "https://github.com/NX-AI/tirex/blob/main/LICENSE\n"
                "Fine-tuned weights are derivative works subject to the same terms."
            )

        super().__init__(**kwargs)

        hf_kwargs = {
            **(
                {"revision": hub_model_revision}
                if hub_model_revision is not None
                else {}
            ),
            **({"local_dir": local_dir} if local_dir is not None else {}),
            **(tirex_kwargs.get("hf_kwargs", {})),
        }
        self.tirex_kwargs = {
            "path": hub_model_name,
            **({"hf_kwargs": hf_kwargs} if hf_kwargs else {}),
            **tirex_kwargs,
        }

    @property
    def supports_past_covariates(self) -> bool:
        return False

    @property
    def supports_future_covariates(self) -> bool:
        return False

    def _create_model(self, train_sample) -> PLForecastingModule:
        pl_module_params = self.pl_module_params or {}
        # enable_finetuning is injected into pl_module_params by the base class;
        # _TiRexModule accepts it as an explicit parameter and converts dict form to bool
        return _TiRexModule(
            tirex_kwargs=self.tirex_kwargs,
            all_quantiles=self._DEFAULT_QUANTILES,
            **pl_module_params,
        )
