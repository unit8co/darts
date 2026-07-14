"""
PatchTST-FM
-----------

PatchTST-FM can be used the same way as other foundation models (e.g. Chronos2), with the exception
that it does not support covariates.

For detailed examples and tutorials, see:

* `Foundation Model Examples
  <https://unit8co.github.io/darts/examples/25-FoundationModel-examples.html>`__
* `Fine-Tuning Examples
  <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__
"""

import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.logging import raise_log
from darts.models.components.huggingface_connector import HuggingFaceConnector
from darts.models.components.patchtst_fm_submodels import (
    _LearnedPositionalEmbedding,
    _make_attn_mask,
    _ResidualBlock,
    _RevIN,
    _TransformerBlock,
)
from darts.models.forecasting.foundation_model import FoundationModel
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
    io_processor,
)
from darts.utils.data.torch_datasets.utils import (
    InputChunkLength,
    PLModuleInput,
    TorchTrainingSample,
    _parse_input_chunk_length,
)
from darts.utils.likelihood_models.torch import QuantileRegression


class _PatchTSTFMBackbone(nn.Module):
    """The PatchTST-FM backbone: patch embedding, transformer encoder, quantile head.

    Faithful port of ``PatchTSTFMModel`` from ``ibm-granite/granite-tsfm``
    (branch ``patchtst-fm``).  Parameter names match the original so that
    safetensors weights can be loaded directly.
    """

    def __init__(
        self,
        context_length: int = 8192,
        d_patch: int = 16,
        d_model: int = 1024,
        n_head: int = 16,
        n_layer: int = 20,
        num_quantile: int = 99,
        **kwargs,
    ):
        super().__init__()
        self.context_length = context_length
        self.d_patch = d_patch
        self.n_patch = context_length // d_patch
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.num_quantile = num_quantile

        self.pos_embed = _LearnedPositionalEmbedding(
            d_model=d_model, max_len=self.n_patch, kind="add"
        )
        self.blocks = nn.ModuleList([
            _TransformerBlock(
                d_model, n_head, mlp_ratio=4.0, norm_first=True, dropout=0.1
            )
            for _ in range(n_layer)
        ])
        self.in_layer = _ResidualBlock(d_patch * 2, d_model, d_model)
        self.out_layer = _ResidualBlock(d_model, d_patch * (num_quantile + 1), d_model)
        self.norm_fn = _RevIN(dim=-1, std_min=1e-5, use_sinh=True)

    def forward(
        self,
        inputs: torch.Tensor,
        pred_mask: torch.Tensor,
        miss_mask: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the backbone forward pass (matches ``PatchTSTFMModel.forward``).

        Returns
        -------
        quantile_predictions
            Raw (normalised-space) quantile predictions, shape
            ``(B, context_length, num_quantile)``.
        loss_mask
            Float mask for loss computation, shape ``(B, context_length)``.
        normed_target
            Instance-normalised target, shape ``(B, context_length)``.
        """
        x = inputs
        pad_mask = pad_mask.bool()
        pred_mask = pred_mask.bool()
        miss_mask = miss_mask.bool()

        B, T = x.shape
        ts_mask = pred_mask | pad_mask | miss_mask

        x_target = self.norm_fn.fit_transform(x, mask=ts_mask)
        x_input = torch.where(ts_mask, torch.zeros_like(x_target), x_target)

        x_patch = x_input.reshape(B, self.n_patch, self.d_patch)
        mask_patch = ts_mask.reshape(B, self.n_patch, self.d_patch)
        pad_patch_mask = (
            pad_mask.reshape(B, self.n_patch, self.d_patch).float().mean(dim=-1).gt(0.9)
        )

        q_pred = self.decode(
            x=x_patch, mask=mask_patch.float(), t_pad_mask=pad_patch_mask
        )

        # q_pred: (B, num_quantile, n_patch, d_patch) -> (B, context_length, num_quantile)
        q_pred = q_pred.permute(0, 2, 3, 1)
        B, N, D, Q = q_pred.shape
        q_pred = q_pred.reshape(B, N * D, Q)
        return q_pred

    def decode(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        t_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode patches through transformer and quantile head."""
        B, N, D = x.shape
        x = self.in_layer(torch.cat([x, 1 - mask], dim=-1))
        pad_attn_mask = _make_attn_mask(t_pad_mask, t_pad_mask).unsqueeze(1)

        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x, pad_attn_mask)
        x = self.out_layer(x)

        q_raw = x.reshape(B, N, self.num_quantile + 1, self.d_patch).permute(0, 2, 1, 3)
        q = q_raw[:, 0, :, :].unsqueeze(1) + torch.cumsum(
            F.softplus(q_raw[:, 1:, :, :]) / self.num_quantile, dim=1
        )
        return q


class _PatchTSTFMModule(PLForecastingModule):
    def __init__(
        self,
        context_length: int = 8192,
        d_patch: int = 16,
        d_model: int = 1024,
        n_head: int = 16,
        n_layer: int = 20,
        num_quantile: int = 99,
        quantile_levels: list[float] | None = None,
        **kwargs,
    ):
        """PyTorch module implementing PatchTST-FM, ported from
        `ibm-granite/granite-tsfm <https://github.com/ibm-granite/granite-tsfm>`_
        and adapted for Darts :class:`PLForecastingModule` interface.

        Parameters
        ----------
        context_length
            Maximum context length of the model (input + forecast).
        d_patch
            Patch size for splitting the time series.
        d_model
            Dimension of the transformer model.
        n_head
            Number of attention heads.
        n_layer
            Number of transformer encoder layers.
        num_quantile
            Number of quantiles produced by the model.
        quantile_levels
            List of quantile levels produced by the model.
        **kwargs
            All parameters required for :class:`PLForecastingModule` base class.
        """
        enable_finetuning = kwargs.pop("enable_finetuning", False)
        super().__init__(**kwargs)

        self.context_length = context_length
        self.d_patch = d_patch
        self.d_model = d_model
        self.num_quantile = num_quantile
        self.quantile_levels = quantile_levels or [
            i / (num_quantile + 1) for i in range(1, num_quantile + 1)
        ]

        self.backbone = _PatchTSTFMBackbone(
            context_length=context_length,
            d_patch=d_patch,
            d_model=d_model,
            n_head=n_head,
            n_layer=n_layer,
            num_quantile=num_quantile,
        )

        # gather indices of user-specified quantiles (used at prediction time)
        user_quantiles: list[float] = (
            self.likelihood.quantiles
            if isinstance(self.likelihood, QuantileRegression)
            else [0.5]
        )
        self.user_quantile_indices = [
            self.quantile_levels.index(q) for q in user_quantiles
        ]

        # during fine-tuning, train on ALL pre-trained quantiles
        if enable_finetuning:
            self._finetuning_likelihood = QuantileRegression(self.quantile_levels)
            self._finetuning_quantile_indices = list(range(num_quantile))
        else:
            self._finetuning_likelihood = None
            self._finetuning_quantile_indices = None

    @io_processor
    def forward(self, x_in: PLModuleInput, *args, **kwargs) -> Any:
        """PatchTST-FM model forward pass adapted for Darts interface.

        Parameters
        ----------
        x_in
            Comes as tuple `(x_past, x_future, x_static, future_target)` where `x_past` is the input/past chunk
            and `x_future` is the output/future chunk. Input dimensions are
            `(n_samples, n_time_steps, n_variables)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(n_samples, n_time_steps, n_targets, n_quantiles)` for
            probabilistic forecasts, or `(n_samples, n_time_steps, n_targets, 1)` for
            deterministic forecasts.
        """
        # B: batch size
        # L: input chunk length
        # T: output chunk length
        # CONT = 8192: context length
        # W = 99: quantiles
        # C: target components
        # N: likelihood quantiles (user-specified)

        # `x_past`: (B, L, C)
        x_past, _, _, _ = x_in
        batch_size, past_length, n_variables = x_past.shape
        output_chunk_length = self.output_chunk_length or 0
        output_chunk_shift = self.output_chunk_shift
        forecast_length = output_chunk_shift + output_chunk_length

        # PatchTST-FM is a univariate model and its inputs do not have a variable dimension,
        # so here we reshape `x_past` to (B * C, L)
        context = x_past.permute(0, 2, 1).reshape(-1, past_length)
        effective_batch = context.shape[0]

        # compute the mean for padding: (B * C, 1)
        context_mean = context.nanmean(dim=1, keepdim=True)
        nan_mask = torch.isnan(context)
        context = torch.where(nan_mask, context_mean.expand_as(context), context)

        # Build the full context window
        left_pad = self.context_length - (past_length + forecast_length)
        # pad context to the left with mean values: (B * C, CONT - L - T)
        pad_values = context_mean.expand(effective_batch, left_pad)
        # `full_input`: (B * C, CONT)
        full_input = torch.cat(
            [
                pad_values,
                context,
                context_mean.expand(effective_batch, forecast_length),
            ],
            dim=1,
        )
        # `pad_mask`: (B * C, CONT)
        # only treat leading NaNs (from variable input chunk length padding) and not
        # actual NaNs as padding for attention masking
        has_nan = nan_mask.any()
        if has_nan:
            leading_nan_mask = nan_mask & ((~nan_mask).cumsum(dim=1) == 0)
        else:
            leading_nan_mask = nan_mask

        pad_mask = torch.cat(
            [
                torch.ones(effective_batch, left_pad, device=context.device),
                leading_nan_mask.float(),
                torch.zeros(
                    effective_batch,
                    forecast_length,
                    device=context.device,
                ),
            ],
            dim=1,
        )
        # `pred_mask`: (B * C, CONT)
        pred_mask = torch.cat(
            [
                torch.zeros(
                    effective_batch, left_pad + past_length, device=context.device
                ),
                torch.ones(effective_batch, forecast_length, device=context.device),
            ],
            dim=1,
        )
        # `miss_mask`: (B * C, CONT)
        miss_mask = torch.cat(
            [
                torch.zeros(effective_batch, left_pad, device=context.device),
                nan_mask.float(),
                torch.zeros(effective_batch, forecast_length, device=context.device),
            ],
            dim=1,
        )

        # forward pass through backbone
        # `q_pred`: (B * C, CONT, W)  -- raw normalised-space quantile predictions
        q_pred = self.backbone(full_input, pred_mask, miss_mask, pad_mask)

        # inverse normalization: (B * C, CONT, W) -> (B * C, W, CONT)
        q_out = q_pred.permute(0, 2, 1)
        q_out = self.backbone.norm_fn.inverse_transform(q_out)

        # extract forecast region
        # `q_forecast`: (B * C, W, T)
        forecast_start = left_pad + past_length + output_chunk_shift
        forecast_end = forecast_start + forecast_length
        q_forecast = q_out[:, :, forecast_start:forecast_end]

        # -> (B, C, W, T)
        q_forecast = q_forecast.reshape(
            batch_size, n_variables, self.num_quantile, output_chunk_length
        )
        # -> (B, T, C, W)
        q_forecast = q_forecast.permute(0, 3, 1, 2)

        # during training, output all pre-trained quantiles for loss
        # during prediction, output only user-specified quantiles
        # -> (B, T, C, N)
        if self.training:
            q_forecast = q_forecast[:, :, :, self._finetuning_quantile_indices]
        else:
            q_forecast = q_forecast[:, :, :, self.user_quantile_indices]

        return q_forecast

    def _compute_loss(self, output, target, criterion, sample_weight):
        if self.training:
            return self._finetuning_likelihood.compute_loss(
                output, target, sample_weight
            )
        else:
            return super()._compute_loss(output, target, criterion, sample_weight)


class PatchTSTFMModel(FoundationModel):
    def __init__(
        self,
        input_chunk_length: InputChunkLength,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        likelihood: QuantileRegression | None = None,
        hub_model_name: str = "ibm-granite/granite-timeseries-patchtst-fm-r1",
        hub_model_revision: str | None = "151f9c6d576281b95c2ff784d0863bd3f12c80f1",
        local_dir: str | os.PathLike | None = None,
        **kwargs,
    ):
        """PatchTST-FM Model for zero-shot forecasting.

        This is an implementation of IBM's PatchTST-FM model [1]_, ported from
        `ibm-granite/granite-tsfm <https://github.com/ibm-granite/granite-tsfm>`_
        with adaptations to use the Darts API. PatchTST-FM is a ~260M-parameter, pretrained time series foundation
        model for probabilistic forecasting. It uses a patch-based transformer encoder architecture with a quantile
        head producing 99 quantiles (0.01 to 0.99).

        This model supports either univariate or multivariate time series, but does not support covariates.
        For multivariate time series, the model is applied independently to each component.

        Using this model will automatically download and cache the pre-trained model from HuggingFace Hub
        (`ibm-granite/granite-timeseries-patchtst-fm-r1
        <https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r1>`_).
        Alternatively, you can specify a local directory containing the model config and weights using the ``local_dir``
        parameter.

        By default, this model is deterministic and outputs only the median (0.5 quantile). To enable probabilistic
        forecasts, pass a :class:`~darts.utils.likelihood_models.torch.QuantileRegression` instance to the
        ``likelihood`` parameter. The quantiles used must be a subset of those used during PatchTST-FM pre-training,
        see below for details. It is recommended to call :func:`predict()` with ``predict_likelihood_parameters=True``
        or ``num_samples >> 1`` to get meaningful results.

        .. tip::
            You can perform full or partial fine-tuning of the model by setting the ``enable_finetuning`` parameter.
            Read more in the parameter description below and in the `Fine-Tuning Examples
            <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__.
        .. note::
            PatchTST-FM weights from ``ibm-granite/granite-timeseries-patchtst-fm-r1`` are licensed under
            the `Apache-2.0 License <https://github.com/ibm-granite/granite-tsfm/blob/main/LICENSE>`_,
            copyright IBM. By using this model, you agree to the terms and conditions of the license.
        .. note::
            You may use non-commercial, research version of PatchTST-FM from
            `ibm-research/patchtst-fm-r1 <https://huggingface.co/ibm-research/patchtst-fm-r1>`_ licensed under
            the `Creative Commons Attribution Non Commercial Share Alike 4.0 <https://spdx.org/licenses/CC-BY-NC-SA-4.0>`_.
            Note that this version may not be used for commercial purposes.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it).
            Can be either an ``int`` for a fixed input window, or a ``(min_length, max_length)`` tuple to enable
            variable-length inputs for inference and fine-tuning.
            For PatchTST-FM, ``max_length + output_chunk_length + output_chunk_shift`` must be ``<=8192``.
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
            from future covariates to use as a model input (if the model supports future covariates). It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
            using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents
            auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
            the model from using future values of past and / or future covariates for prediction (depending on the
            model's covariate support).
            For PatchTST-FM, `input_chunk_length + output_chunk_length + output_chunk_shift` must be `<=8192`.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`).
            For PatchTST-FM, `input_chunk_length + output_chunk_length + output_chunk_shift` must be `<=8192`.
        likelihood
            The likelihood model to be used for probabilistic forecasts. Must be ``None`` or an instance of
            :class:`~darts.utils.likelihood_models.torch.QuantileRegression`. If using ``QuantileRegression``,
            the quantiles must be a subset of those used during PatchTST-FM pre-training:
            [0.01, 0.02, ..., 0.99].
            Default: ``None``, which will make the model deterministic (median quantile only).
            When fine-tuning is enabled, the training loss is always computed on all pre-trained quantiles to
            preserve the full distribution, regardless of the ``likelihood`` setting. The ``likelihood`` parameter
            only affects prediction output.
        hub_model_name
            The model ID on HuggingFace Hub.
            Default: ``"ibm-granite/granite-timeseries-patchtst-fm-r1"`` (Apache-2.0).
        hub_model_revision
            The model version to use. This can be a branch name, tag name, or commit hash. Default is
            ``151f9c6d576281b95c2ff784d0863bd3f12c80f1``, which will use the March 25, 2026 release of PatchTST-FM.
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
            - ``{"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}`` to use all available GPUs.

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
        .. [1] Y. Wen, W. M. Gifford, C. Reddy, L. M. Nguyen, J. Kalagnanam, and A. A. Julius,
               "Revisiting the Generic Transformer: Deconstructing a Strong Baseline for Time Series
               Foundation Models," arXiv:2602.06909, 2026.

        Examples
        --------
        Point forecasting:

        >>> from darts.models import PatchTSTFMModel
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load().astype("float32")
        >>> model = PatchTSTFMModel(
        ...     input_chunk_length=12,
        ...     output_chunk_length=6,
        ... )
        >>> model.fit(series)
        >>> pred = model.predict(n=6)
        >>> pred
                    #Passengers
        Month
        1961-01-01   507.465973
        1961-02-01   517.345459
        1961-03-01   519.231140
        1961-04-01   506.727661
        1961-05-01   504.759125
        1961-06-01   496.883820

        Probabilistic forecasting:

        >>> from darts.utils.likelihood_models import QuantileRegression
        >>> model = PatchTSTFMModel(
        ...     input_chunk_length=12,
        ...     output_chunk_length=6,
        ...     likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        ... )
        >>> model.fit(series)
        >>> pred = model.predict(n=6, predict_likelihood_parameters=True)
        >>> pred
                    #Passengers_q0.100  #Passengers_q0.500  #Passengers_q0.900
        Month
        1961-01-01          395.053131          507.465973          602.820312
        1961-02-01          402.696472          517.345459          612.596741
        1961-03-01          394.399231          519.231140          625.937439
        1961-04-01          381.966797          506.727661          619.151367
        1961-05-01          388.510803          504.759125          635.277893
        1961-06-01          375.241638          496.883820          635.320679
        """
        hf_connector = HuggingFaceConnector(
            model_name=hub_model_name,
            model_revision=hub_model_revision,
            local_dir=local_dir,
        )

        config = hf_connector.load_config()

        # validate input_chunk_length + output_chunk_length + output_chunk_shift <= context_length
        context_length = config["context_length"]
        _, max_icl = _parse_input_chunk_length(input_chunk_length)
        if max_icl + output_chunk_length + output_chunk_shift > context_length:
            raise_log(
                ValueError(
                    f"`input_chunk_length` {max_icl} plus `output_chunk_length` {output_chunk_length} "
                    f"plus `output_chunk_shift` {output_chunk_shift} cannot be greater than model's maximum "
                    f"context_length {context_length}"
                ),
            )

        quantile_levels = config["quantile_levels"]
        # by default (`likelihood=None`), model is deterministic
        # otherwise, only QuantileRegression likelihood is supported and quantiles must be
        # a subset of the pre-trained quantiles
        if likelihood is not None:
            if not isinstance(likelihood, QuantileRegression):
                raise_log(
                    ValueError(
                        f"Only QuantileRegression likelihood is supported for PatchTST-FM in Darts. "
                        f"Got {type(likelihood)}."
                    ),
                )
            user_quantiles: list[float] = likelihood.quantiles
            if not set(user_quantiles).issubset(quantile_levels):
                raise_log(
                    ValueError(
                        f"The quantiles for QuantileRegression likelihood {user_quantiles} "
                        f"must be a subset of PatchTST-FM quantiles {quantile_levels}."
                    ),
                )

        self.hf_connector = hf_connector
        super().__init__(**kwargs)

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        pl_module_params = self.pl_module_params or {}
        return self.hf_connector.load_model(
            module_class=_PatchTSTFMModule,
            pl_module_params=pl_module_params,
        )

    @property
    def supports_past_covariates(self) -> bool:
        return False

    @property
    def supports_future_covariates(self) -> bool:
        return False
