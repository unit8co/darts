"""
TiRex-2: Zero-Shot Multivariate Forecasting
-----------------------------------------

TiRex-2 supports joint multivariate forecasting with past and future covariates.

For detailed examples and tutorials, see:

* `Foundation Model Examples
  <https://unit8co.github.io/darts/examples/25-FoundationModel-examples.html>`__
* `Fine-Tuning Examples
  <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__
"""

import os
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
from tirex2 import TimeseriesType, load_model

from darts.logging import raise_log
from darts.models.forecasting.foundation_model import FoundationModel
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.utils.data.torch_datasets.utils import (
    InputChunkLength,
    ModuleStage,
    PLModuleInput,
    PLModuleOutput,
    TorchTrainingSample,
)
from darts.utils.likelihood_models.torch import QuantileRegression

if TYPE_CHECKING:
    from tirex2.model import TiRex2


class _TiRex2Module(PLForecastingModule):
    def __init__(
        self,
        setup_kwargs: dict[str, Any],
        inference_kwargs: dict[str, Any],
        **kwargs,
    ):
        """PyTorch Lightning module wrapping a pre-loaded TiRex-2 forecaster.

        Parameters
        ----------
        setup_kwargs
            Additional keyword arguments forwarded to ``tirex2.load_model()``.
        inference_kwargs
            Optional keyword arguments forwarded to ``tirex2.TiRex2._predict()``, such as ``tta_diff``.
        **kwargs
            Additional arguments passed to `PLForecastingModule`.
        """
        # for fine-tuning, model should be trained on pre-trained quantiles
        self._enable_finetuning = kwargs.pop("enable_finetuning", False)
        super().__init__(**kwargs)

        self._setup_kwargs = setup_kwargs
        self._inference_kwargs = inference_kwargs

        # Always load the model on CPU first, then move to the execution device during
        # configure_model(). This is because Trainer is not yet set up during __init__(),
        # so we cannot know the execution device.
        self.tirex2: TiRex2 = self._load_tirex2(load_device=torch.device("cpu"))

        # Validate against the checkpoint's maximum prediction length
        self._future_len = (self.output_chunk_length or 0) + self.output_chunk_shift

        # Validate against the checkpoint's pretrained quantiles
        all_quantiles = [round(float(q), 6) for q in self.tirex2.quantiles]
        user_quantiles = self.likelihood.quantiles if self.likelihood else [0.5]

        # Register the indices of the requested quantiles
        self.register_buffer(
            "_user_quantile_indices",
            torch.tensor([all_quantiles.index(q) for q in user_quantiles]),
        )

        # during fine-tuning, train on ALL pre-trained quantiles to preserve the
        # full distribution; prediction uses only user-specified quantiles
        if self._enable_finetuning:
            self._finetuning_likelihood = QuantileRegression(all_quantiles)
        else:
            self._finetuning_likelihood = None

    def _load_tirex2(self, load_device: torch.device) -> "TiRex2":
        """Load TiRex-2 on ``load_device``."""
        device_type = load_device.type
        context = (
            torch.cuda.device(load_device) if device_type == "cuda" else nullcontext()
        )
        with context:
            model: TiRex2 = load_model(**self._setup_kwargs, device=device_type).model
        return model

    def configure_model(self) -> None:
        # Lightning also calls this hook during checkpoint loading without a
        # Trainer. Keep the CPU model until a Trainer supplies the execution device.
        if self._trainer is None:
            return

        # TiRex-2 chooses its recurrent kernels at construction; moving its
        # tensors later does not switch backends. Reload when the execution device
        # differs (e.g. weights restored on CPU, predict on GPU).
        device = self.trainer.strategy.root_device
        if device.type != self.tirex2.device:
            previous = self.tirex2
            self.tirex2 = self._load_tirex2(device)
            self.tirex2.load_state_dict(previous.state_dict())
            self.tirex2.train(previous.training)
            for name, parameter in self.tirex2.named_parameters():
                parameter.requires_grad_(previous.get_parameter(name).requires_grad)

        if self._enable_finetuning:
            self.tirex2.train(self.training)

    def forward(self, x_in: PLModuleInput) -> PLModuleOutput:
        # Dimension notation in comments below:
        #   B: batch size
        #   L: input chunk length
        #   T: output chunk length
        #   S: output chunk shift
        #   C: target components
        #   X: past covariate components
        #   F: future covariate components
        #   Q: likelihood quantiles returned by TiRex-2 (9 pre-trained)
        #   N: likelihood quantiles (user-specified, 1 if deterministic)

        # target: (B, L, C); past_covariates: (B, L, X);
        # historic_future_covariates: (B, L, F); future_covariates: (B, T, F)
        target = x_in.past_target
        past_covariates = x_in.past_covariates
        historic_future_covariates = x_in.historic_future_covariates
        future_covariates = x_in.future_covariates
        batch_size = target.shape[0]
        output_chunk_shift = self.output_chunk_shift

        if historic_future_covariates is not None and future_covariates is not None:
            n_future_covs = future_covariates.shape[-1]
            # Darts does not supply future covariates inside the output shift.
            # Mark that gap as missing, preserving alignment with target history.
            gap = torch.full(
                (batch_size, output_chunk_shift, n_future_covs),
                float("nan"),
                device=target.device,
                dtype=target.dtype,
            )
            # (B, L + S + T, F)
            future_covariates = torch.cat(
                (historic_future_covariates, gap, future_covariates), dim=1
            )

        # TiRex-2 expects a list of multivariate samples:
        # target: (C, L), past_covariates: (X, L), future_covariates: (F, L + S + T)
        timeseries = [
            TimeseriesType(
                target=target[i].T,
                past_covariates=past_covariates[i].T
                if past_covariates is not None
                else None,
                future_covariates=future_covariates[i].T
                if future_covariates is not None
                else None,
            )
            for i in range(batch_size)
        ]

        # Each sample is an independent joint multivariate task. Native output:
        # B tensors of shape (C, Q, S + T), with all pre-trained quantiles.
        inference_kwargs = self._inference_kwargs
        if x_in.stage is ModuleStage.TRAIN:
            inference_kwargs = {**inference_kwargs, "preserve_grad": True}
        forecasts = self.tirex2._predict(
            timeseries, prediction_length=self._future_len, **inference_kwargs
        )
        # Stack and permute the forecasts: (B, S + T, C, Q)
        output = torch.stack(forecasts).permute(0, 3, 1, 2)
        # Select the requested horizon: (B, T, C, Q).
        prediction = output[:, output_chunk_shift:]

        # during training (fine-tuning), output all pre-trained quantiles for loss;
        # during prediction, output only user-specified quantiles
        if not (x_in.stage is ModuleStage.TRAIN and self._enable_finetuning):
            prediction = prediction.index_select(-1, self._user_quantile_indices)

        return PLModuleOutput(prediction=prediction)

    def _compute_loss(self, output: PLModuleOutput, target, criterion, sample_weight):
        if self.training:
            # compute loss on pre-trained quantiles
            return self._finetuning_likelihood.compute_loss(
                output.prediction, target, sample_weight
            )
        else:
            return super()._compute_loss(output, target, criterion, sample_weight)


class TiRex2Model(FoundationModel):
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
    _MAX_PREDICTION_LENGTH = 320

    def __init__(
        self,
        input_chunk_length: InputChunkLength,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        likelihood: QuantileRegression | None = None,
        hub_model_name: str = "NX-AI/TiRex-2",
        hub_model_revision: str | None = None,
        local_dir: str | os.PathLike | None = None,
        setup_kwargs: dict[str, Any] | None = None,
        inference_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """TiRex-2 foundation model for zero-shot multivariate forecasting.

        Compared to TiRex-1, TiRex-2 offers several advantages:

        - Native support for multivariate forecasting with covariates.
        - Stronger results on the public benchmarks.
        - Efficient memory usage and inference speed.
        - A permissive Apache 2.0 license, which allows commercial use.

        This a Darts wrapper around the TiRex2 model introduced in Podest et al. (2026) [1]_. The implementation
        delegates all forecasting logic and weight loading to the optional
        `tirex-2 <https://pypi.org/project/tirex-2/>`_ package while exposing a standard :class:`TorchForecastingModel`
        interface.

        TiRex-2 jointly forecasts all target components and supports past and future covariates. By default, the model
        is deterministic (median forecast only). To enable probabilistic forecasts, pass a
        :class:`~darts.utils.likelihood_models.torch.QuantileRegression` instance to the ``likelihood`` parameter.
        For probabilistic forecasts, it is recommended to call :func:`predict()` with
        ``predict_likelihood_parameters=True`` or ``num_samples >> 1`` to get meaningful results.

        For more details on the TiRex-2 model, see the original paper [1]_ and `docs
        <https://nx-ai.github.io/tirex-2/>`_.

        .. note::
            TiRex-2 is licensed under the `Apache-2.0 License <https://github.com/NX-AI/tirex-2/blob/main/LICENSE>`_,
            copyright NXAI GmbH or its affiliates. By using this model, you agree to the terms and conditions of
            the license.

        .. note::
            TiRex-2 supports full or partial fine-tuning. For small datasets, fine-tune only the final decoder layer
            and output head with ``enable_finetuning={"unfreeze": ["*.stack.11.*", "*.output_patch_embedding.*"]}``.
            See the `Fine-Tuning Examples
            <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__
            notebook for more details.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it).
            Can be either an ``int`` for a fixed input window, or a ``(min_length, max_length)`` tuple to enable
            variable-length inputs for inference.
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
            from future covariates to use as a model input (if the model supports future covariates). It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
            using either a one-shot or autoregressive forecast. Setting `n <= output_chunk_length` prevents
            auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
            the model from using future values of past and / or future covariates for prediction (depending on the
            model's covariate support).
            For TiRex-2, `output_chunk_length + output_chunk_shift` must not exceed the checkpoint
            maximum prediction length, which is validated when loading the model.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`).
        likelihood
            The likelihood model to be used for probabilistic forecasts. Must be ``None`` or an instance of
            :class:`~darts.utils.likelihood_models.torch.QuantileRegression`. If using ``QuantileRegression``,
            the quantiles must be a subset of those used during TiRex-2 pre-training:
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
            Default: ``None``, which will make the model deterministic (median quantile only).
            When fine-tuning is enabled, the training loss is always computed on all pre-trained quantiles to
            preserve the full distribution, regardless of the ``likelihood`` setting. The ``likelihood`` parameter
            only affects prediction output.
        hub_model_name
            The model ID on Hugging Face Hub, or a local checkpoint directory containing ``model-config.yaml`` and
            ``model.ckpt``. Default: ``"NX-AI/TiRex-2"``.
        hub_model_revision
            The model version to use. This can be a branch name, tag name, or commit hash. Default: ``None``, which
            will use the default branch from ``hub_model_name``.
        local_dir
            Optional download directory forwarded to Hugging Face. Default: ``None``, which uses the cache
            managed by ``huggingface_hub``. To load an existing checkpoint without downloading, pass its
            directory as ``hub_model_name`` instead.
        setup_kwargs
            Additional keyword arguments forwarded to ``tirex2.load_model()`` when the backend is initialized. Not
            passed to Darts :func:`predict`.
        inference_kwargs
            Optional keyword arguments forwarded to ``tirex2.TiRex2._predict()`` on each internal forecast call
            (e.g. ``tta_diff``). Not passed to Darts :func:`predict`.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`. Training-related options below are inherited from the
            base class and have no effect while fine-tuning is disabled.

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
            Running on GPU(s) is also possible using ``pl_trainer_kwargs`` by specifying ``"accelerator"`` and
            ``"devices"``. Some examples for setting the devices inside the ``pl_trainer_kwargs``
            dict:

            - ``{"accelerator": "cpu"}`` for CPU,
            - ``{"accelerator": "gpu", "devices": [i]}`` to use only GPU ``i`` (``i`` must be an integer),
            - ``{"accelerator": "gpu", "devices": -1}`` to use all available GPUs.

            For more info, see here:
            https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags, and
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
        .. [1] P. Podest, M. Pichler et al. "TiRex-2: Generalizing TiRex to Multivariate Data and
           Streaming", 2026. https://arxiv.org/abs/2607.01204

        Examples
        --------
        Point forecasting:

        >>> from darts.models import TiRex2Model
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load().astype("float32")
        >>> model = TiRex2Model(
        ...     input_chunk_length=12,
        ...     output_chunk_length=6,
        ... )
        >>> model.fit(series)
        >>> pred = model.predict(n=6)
        >>> pred
                    #Passengers
        Month
        1961-01-01   446.937927
        1961-02-01   454.835083
        1961-03-01   459.679291
        1961-04-01   461.265442
        1961-05-01   463.061127
        1961-06-01   462.221680

        Probabilistic forecasting:

        >>> from darts.utils.likelihood_models import QuantileRegression
        >>> model = TiRex2Model(
        ...     input_chunk_length=12,
        ...     output_chunk_length=6,
        ...     likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        ... )
        >>> model.fit(series)
        >>> pred = model.predict(n=6, predict_likelihood_parameters=True)
        >>> pred
                    #Passengers_q0.100  #Passengers_q0.500  #Passengers_q0.900
        Month
        1961-01-01          383.065399          446.937927          521.927185
        1961-02-01          365.019501          454.835083          562.312622
        1961-03-01          352.370239          459.679291          598.056763
        1961-04-01          344.941345          461.265442          613.258972
        1961-05-01          338.307678          463.061127          626.979980
        1961-06-01          333.445587          462.221680          630.935303
        """
        # Validate likelihood argument
        if likelihood is not None:
            if not isinstance(likelihood, QuantileRegression):
                raise_log(
                    ValueError(
                        "Only QuantileRegression likelihood is supported for TiRex-2. "
                        f"Got {type(likelihood)}."
                    )
                )
            user_quantiles: list[float] = likelihood.quantiles
            if not set(user_quantiles).issubset(self._DEFAULT_QUANTILES):
                raise_log(
                    ValueError(
                        f"The quantiles for QuantileRegression likelihood {user_quantiles} "
                        f"must be a subset of TiRex-2 quantiles {self._DEFAULT_QUANTILES}."
                    ),
                )

        if output_chunk_length + output_chunk_shift > self._MAX_PREDICTION_LENGTH:
            raise_log(
                ValueError(
                    f"`output_chunk_length` {output_chunk_length} plus `output_chunk_shift` {output_chunk_shift} "
                    f"cannot be greater than model's maximum prediction length {self._MAX_PREDICTION_LENGTH}"
                ),
            )

        # Validate setup_kwargs argument
        load_kwargs = dict(setup_kwargs or {})
        if "ckpt_path" in load_kwargs:
            raise_log(
                ValueError("Pass `ckpt_path` via `hub_model_name`, not `setup_kwargs`.")
            )
        if "device" in load_kwargs:
            raise_log(
                ValueError(
                    'Pass `device` via `pl_trainer_kwargs["accelerator"]`, not `setup_kwargs`.'
                )
            )
        hf_kwargs = dict(load_kwargs.pop("hf_kwargs", {}) or {})
        if hub_model_revision is not None:
            hf_kwargs["revision"] = hub_model_revision
        if local_dir is not None:
            hf_kwargs["local_dir"] = local_dir
        self._setup_kwargs = {
            "ckpt_path": hub_model_name,
            **load_kwargs,
        }
        if hf_kwargs:
            self._setup_kwargs["hf_kwargs"] = hf_kwargs

        self._inference_kwargs = inference_kwargs or {}

        super().__init__(**kwargs)

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        return _TiRex2Module(
            setup_kwargs=self._setup_kwargs,
            inference_kwargs=self._inference_kwargs,
            **self.pl_module_params,
        )
