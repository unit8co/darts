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

"""
Throughout this file, we use the following notation for tensor shapes:

    SYMBOL: Description
    ------------------------------------------------
    B: batch size
    L: input chunk length
    H: output chunk length
    S: output chunk shift
    C: target components
    X: past covariate components
    F: future covariate components
    N: likelihood parameters
    Q: number of pretrained quantiles (9 for TiRex-2)

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
    """Adapt TiRex-2's native multivariate inputs and quantiles to Darts."""

    def __init__(
        self, tirex2_kwargs: dict[str, Any], predict_kwargs: dict[str, Any], **kwargs
    ):
        # for fine-tuning, model should be trained on pre-trained quantiles
        self._enable_finetuning = kwargs.pop("enable_finetuning", False)
        super().__init__(**kwargs)

        self._tirex2_kwargs = tirex2_kwargs
        self._predict_kwargs = predict_kwargs

        # Always load the model on CPU first, then move to the actual device during
        # configure_model(). This allows
        self.tirex2: TiRex2 = load_model(**self._tirex2_kwargs, device="cpu").model
        self.tirex2.to(dtype=self.dtype)

        # Validate against the checkpoint's maximum prediction length
        self._future_len = self.output_chunk_length + self.output_chunk_shift
        if self._future_len > self.tirex2.future_len:
            raise_log(
                ValueError(
                    f"`output_chunk_length` {self.output_chunk_length} plus "
                    f"`output_chunk_shift` {self.output_chunk_shift} cannot exceed "
                    f"the checkpoint's maximum prediction length {self.tirex2.future_len}."
                )
            )

        # Validate against the checkpoint's pretrained quantiles
        all_quantiles = [round(float(q), 6) for q in self.tirex2.quantiles]
        user_quantiles = self.likelihood.quantiles if self.likelihood else [0.5]
        if not set(user_quantiles).issubset(all_quantiles):
            raise_log(
                ValueError(
                    f"The checkpoint does not support the requested quantiles: {user_quantiles}. "
                    f"Supported quantiles are: {all_quantiles}."
                )
            )

        # Register the indices of the requested quantiles
        self.register_buffer(
            "_user_quantile_indices",
            torch.tensor([all_quantiles.index(q) for q in user_quantiles]),
        )
        self.register_buffer(
            "_finetuning_quantile_indices",
            torch.tensor(list(range(len(all_quantiles)))),
        )

        # during fine-tuning, train on ALL pre-trained quantiles to preserve the
        # full distribution; prediction uses only user-specified quantiles
        if self._enable_finetuning:
            self._finetuning_likelihood = QuantileRegression(all_quantiles)
        else:
            self._finetuning_likelihood = None

    def _load_tirex2_model(self, device: torch.device) -> TiRex2:
        # The loader accepts "cuda", not "cuda:N". Select the correct GPU for
        # its allocations, including when running one process per GPU.
        context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        with context:
            model = load_model(**self._tirex2_kwargs, device=device.type).model
        return model.to(dtype=self.dtype)

    def configure_model(self) -> None:
        # TiRex-2 chooses its recurrent kernels at construction; moving its
        # tensors later does not switch backends. Lightning knows the actual
        # execution device here, before moving the model or restoring weights.
        device = self.trainer.strategy.root_device
        if device.type != self.tirex2.device:
            # The loader accepts "cuda", not "cuda:N". Select the correct GPU for
            # its allocations, including when running one process per GPU.
            context = (
                torch.cuda.device(device) if device.type == "cuda" else nullcontext()
            )
            with context:
                model: TiRex2 = load_model(
                    **self._tirex2_kwargs, device=device.type
                ).model
            model.to(dtype=self.dtype)
            # restore the weights and training mode from the original model, which was loaded on CPU
            model.load_state_dict(self.tirex2.state_dict())
            model.train(self.tirex2.training)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(self.tirex2.get_parameter(name).requires_grad)
            self.tirex2 = model

        if self._enable_finetuning:
            self.tirex2.train(self.training)

    def forward(self, x_in: PLModuleInput, *args, **kwargs) -> PLModuleOutput:
        # target: (B, L, C); past_covariates: (B, L, X);
        # historic_future_covariates: (B, L, F); future_covariates: (B, H, F)
        target = x_in.past_target
        past_covariates = x_in.past_covariates
        historic_future_covariates = x_in.historic_future_covariates
        future_covariates = x_in.future_covariates
        B = target.shape[0]
        S = self.output_chunk_shift
        F = future_covariates.shape[-1] if future_covariates is not None else 0

        # Prepare future covariates (if any)
        future = None
        if F:
            # Darts does not supply future covariates inside the output shift.
            # Mark that gap as missing, preserving alignment with target history.
            # gap: (B, S, F) filled with NaN
            gap = target.new_full((B, S, F), float("nan"))
            # future: (B, L + S + H, F)
            future = torch.cat(
                (historic_future_covariates, gap, future_covariates), dim=1
            )

        # Prepare TiRex-2's native multivariate inputs: a list of TimeseriesType objects.
        # Each object contains a single multivariate sample with optional past and future covariates:
        # target: (C, L), past_covariates: (X, L), future_covariates: (F, L + S + H)
        timeseries = [
            TimeseriesType(
                target=past.T,
                past_covariates=past_covariates[i].T
                if past_covariates is not None
                else None,
                future_covariates=future[i].T if future is not None else None,
            )
            for i, past in enumerate(target)
        ]

        # Each sample remains a joint multivariate task, independent of other
        # samples in the batch. Native output: B tensors of shape
        # (C, Q, H + S), with all pretrained quantiles.
        predict_kwargs = self._predict_kwargs
        if x_in.stage is ModuleStage.TRAIN:
            predict_kwargs = {**predict_kwargs, "preserve_grad": True}
        forecasts = self.tirex2._predict_once(
            timeseries, prediction_length=self._future_len, **predict_kwargs
        )
        # Stack and permute the forecasts: (B, S + H, C, Q)
        output = torch.stack(forecasts).permute(0, 3, 1, 2)
        # Select the requested horizon: (B, H, C, Q).
        prediction = output[:, S:]

        # during training (fine-tuning), output all pre-trained quantiles for loss;
        # during prediction, output only user-specified quantiles
        if x_in.stage is ModuleStage.TRAIN:
            prediction = prediction.index_select(-1, self._finetuning_quantile_indices)
        else:
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
    def __init__(
        self,
        input_chunk_length: InputChunkLength,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        likelihood: QuantileRegression | None = None,
        hub_model_name: str = "NX-AI/TiRex-2",
        hub_model_revision: str | None = None,
        local_dir: str | os.PathLike | None = None,
        tirex2_kwargs: dict[str, Any] | None = None,
        predict_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """TiRex-2 foundation model for zero-shot multivariate forecasting.

        Compared to TiRex-1, TiRex-2 offers several advantages:

        - Native support for multivariate forecasting with covariates.
        - Stronger results on the public benchmarks.
        - Efficient memory usage and inference speed.
        - A permissive Apache 2.0 license, which allows commercial use.

        This wraps the pretrained xLSTM model from Podest et al. (2026) [1]_ using the `tirex-2
        <https://pypi.org/project/tirex-2/>`_ package (Python 3.11 or newer). Please install it alongside
        ``darts[torch]``.

        TiRex-2 jointly forecasts all target components and supports past and future covariates. The default
        forecast is the median; pass :class:`~darts.utils.likelihood_models.torch.QuantileRegression` to
        ``likelihood`` and call :func:`predict()` with ``predict_likelihood_parameters=True`` or ``num_samples >>
        1`` to obtain quantiles or probabilistic samples.

        For more details on the TiRex-2 model, see the original paper [1]_ and `docs
        <https://nx-ai.github.io/tirex-2/>`_.

        .. note::
            TiRex-2 is licensed under the `Apache-2.0 License <https://github.com/NX-AI/tirex-2/blob/main/LICENSE>`_,
            copyright NXAI GmbH or its affiliates. By using this model, you agree to the terms and conditions of
            the license.

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
        hub_model_name
            The model ID on Hugging Face Hub, or a local checkpoint directory containing
            ``model-config.yaml`` and ``model.ckpt``. Default: ``"NX-AI/TiRex-2"``.
        hub_model_revision
            The model version to use. This can be a branch name, tag name, or commit hash. Default: ``None``, which
            will use the default branch from ``hub_model_name``.
        local_dir
            Optional download directory forwarded to Hugging Face. Default: ``None``, which uses the cache
            managed by ``huggingface_hub``. To load an existing checkpoint without downloading, pass its
            directory as ``hub_model_name`` instead.
        tirex2_kwargs
            Additional arguments to ``tirex2.load_model()``, such as ``use_flex_attention`` and ``hf_kwargs``.
        predict_kwargs
            Optional arguments to ``tirex2.TiRex2._predict_once()``, such as ``tta_diff``.
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

        # Validate tirex2_kwargs argument
        load_kwargs = dict(tirex2_kwargs or {})
        if "ckpt_path" in load_kwargs:
            raise_log(
                ValueError(
                    "Pass `ckpt_path` via `hub_model_name`, not `tirex2_kwargs`."
                )
            )
        if "device" in load_kwargs:
            raise_log(
                ValueError(
                    'Pass `device` via `pl_trainer_kwargs["accelerator"]`, not `tirex2_kwargs`.'
                )
            )
        hf_kwargs = dict(load_kwargs.pop("hf_kwargs", {}) or {})
        if hub_model_revision is not None:
            hf_kwargs["revision"] = hub_model_revision
        if local_dir is not None:
            hf_kwargs["local_dir"] = local_dir
        self._tirex2_kwargs = {
            "ckpt_path": hub_model_name,
            **load_kwargs,
        }
        if hf_kwargs:
            self._tirex2_kwargs["hf_kwargs"] = hf_kwargs

        self._predict_kwargs = predict_kwargs or {}

        super().__init__(**kwargs)

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        return _TiRex2Module(
            tirex2_kwargs=self._tirex2_kwargs,
            predict_kwargs=self._predict_kwargs,
            **self.pl_module_params,
        )
