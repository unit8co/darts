"""
PatchTST-FM
-----------

PatchTST-FM can be used the same way as other foundation models (e.g. Chronos2), with the exception
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
import torch.nn as nn
import torch.nn.functional as F

from darts.logging import get_logger, raise_log
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
from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample
from darts.utils.likelihood_models.torch import QuantileRegression

logger = get_logger(__name__)


class _PatchTSTFMBackbone(nn.Module):
    """The PatchTST-FM backbone: patch embedding, transformer encoder, quantile head.

    This matches the parameter naming of the original ``PatchTSTFMModel`` class from
    ``granite-tsfm`` so that safetensors weights can be loaded directly.
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
            d_model=d_model, max_len=self.n_patch
        )
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_head, mlp_ratio=4.0, dropout=0.1)
            for _ in range(n_layer)
        ])
        self.in_layer = _ResidualBlock(d_patch * 2, d_model, d_model)
        self.out_layer = _ResidualBlock(d_model, d_patch * (num_quantile + 1), d_model)
        self.norm_fn = _RevIN(dim=-1, std_min=1e-5, use_sinh=True)

    def forward(
        self,
        inputs: torch.Tensor,
        pred_mask: torch.Tensor,
        pad_mask: torch.Tensor,
        miss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the PatchTST-FM backbone forward pass.

        Returns the quantile predictions (unnormalized) and the combined
        pred_mask & pad_mask & miss_mask for loss computation.
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

        q_pred = self._decode(
            x=x_patch, mask=mask_patch.float(), t_pad_mask=pad_patch_mask
        )

        # q_pred shape: (B, num_quantile, n_patch, d_patch) -> (B, T, num_quantile)
        q_pred = q_pred.permute(0, 2, 3, 1)
        B, N, D, Q = q_pred.shape
        q_pred = q_pred.reshape(B, N * D, Q)

        # inverse normalization: shape (B, Q, T)
        q_out = q_pred.permute(0, 2, 1)
        q_out = self.norm_fn.inverse_transform(q_out)

        loss_mask = (pred_mask & ~pad_mask & ~miss_mask).float()
        return q_out, loss_mask, x_target

    def _decode(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        t_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
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

    def _forward(
        self,
        inputs: torch.Tensor,
        pred_mask: torch.Tensor,
        pad_mask: torch.Tensor,
        miss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Original PatchTST-FM forward pass.

        Parameters
        ----------
        inputs
            Tensor of shape (batch, context_length) containing the padded time series.
        pred_mask
            Boolean mask indicating forecast positions.
        pad_mask
            Boolean mask indicating left-padding positions.
        miss_mask
            Boolean mask indicating missing/NaN positions.

        Returns
        -------
        torch.Tensor
            Quantile predictions of shape (batch, num_quantile, context_length).
        """
        q_out, _, _ = self.backbone(inputs, pred_mask, pad_mask, miss_mask)
        return q_out

    @io_processor
    def forward(self, x_in: PLModuleInput, *args, **kwargs) -> Any:
        """PatchTST-FM model forward pass adapted for Darts interface.

        Parameters
        ----------
        x_in
            Comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk
            and `x_future` is the output/future chunk. Input dimensions are
            `(n_samples, n_time_steps, n_variables)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(n_samples, n_time_steps, n_targets, n_quantiles)` for
            probabilistic forecasts, or `(n_samples, n_time_steps, n_targets, 1)` for
            deterministic forecasts.
        """
        x_past, _, _ = x_in
        batch_size, past_length, n_variables = x_past.shape
        output_chunk_length = self.output_chunk_length or 0
        output_chunk_shift = self.output_chunk_shift
        forecast_length = output_chunk_shift + output_chunk_length

        # PatchTST-FM is univariate (channel-independent): reshape to (batch * vars, time)
        context = x_past.permute(0, 2, 1).reshape(-1, past_length)
        effective_batch = context.shape[0]

        # compute the mean for padding
        context_mean = context.nanmean(dim=1, keepdim=True)
        nan_mask = torch.isnan(context)
        context = torch.where(nan_mask, context_mean.expand_as(context), context)

        # Build the full window: context + forecast region
        total_length = past_length + forecast_length
        if total_length <= self.context_length:
            left_pad = self.context_length - total_length
            # pad context to the left with mean values
            pad_values = context_mean.expand(effective_batch, left_pad)
            full_input = torch.cat(
                [
                    pad_values,
                    context,
                    context_mean.expand(effective_batch, forecast_length),
                ],
                dim=1,
            )
            # masks
            pad_mask = torch.cat(
                [
                    torch.ones(effective_batch, left_pad, device=context.device),
                    torch.zeros(
                        effective_batch,
                        past_length + forecast_length,
                        device=context.device,
                    ),
                ],
                dim=1,
            )
            pred_mask = torch.cat(
                [
                    torch.zeros(
                        effective_batch, left_pad + past_length, device=context.device
                    ),
                    torch.ones(effective_batch, forecast_length, device=context.device),
                ],
                dim=1,
            )
        else:
            raise_log(
                ValueError(
                    f"input_chunk_length ({past_length}) + output_chunk_length ({output_chunk_length}) "
                    f"+ output_chunk_shift ({output_chunk_shift}) exceeds "
                    f"model context_length ({self.context_length}). "
                    f"Please reduce input_chunk_length or output_chunk_length."
                ),
                logger,
            )

        miss_mask = torch.cat(
            [
                torch.zeros(effective_batch, left_pad, device=context.device),
                nan_mask.float(),
                torch.zeros(effective_batch, forecast_length, device=context.device),
            ],
            dim=1,
        )

        # forward pass through backbone
        q_out, _, _ = self.backbone(full_input, pred_mask, pad_mask, miss_mask)

        # q_out shape: (effective_batch, num_quantile, context_length)
        # extract forecast region
        forecast_start = left_pad + past_length
        forecast_end = forecast_start + forecast_length
        q_forecast = q_out[:, :, forecast_start:forecast_end]

        # select output_chunk_length after shift
        q_forecast = q_forecast[:, :, output_chunk_shift:]

        # reshape: (batch * vars, num_quantile, output_chunk_length)
        #       -> (batch, vars, num_quantile, output_chunk_length)
        #       -> (batch, output_chunk_length, vars, num_quantile)
        q_forecast = q_forecast.reshape(
            batch_size, n_variables, self.num_quantile, output_chunk_length
        )
        q_forecast = q_forecast.permute(0, 3, 1, 2)

        # select only target variables
        q_forecast = q_forecast[:, :, : self.n_targets, :]

        # during training, output all pre-trained quantiles for loss
        # during prediction, output only user-specified quantiles
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
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        likelihood: QuantileRegression | None = None,
        hub_model_name: str = "ibm-granite/granite-timeseries-patchtst-fm-r1",
        hub_model_revision: str | None = None,
        local_dir: str | os.PathLike | None = None,
        **kwargs,
    ):
        """PatchTST-FM Model for zero-shot forecasting.

        This is an implementation of IBM's PatchTST-FM model [1]_, ported from
        `ibm-granite/granite-tsfm <https://github.com/ibm-granite/granite-tsfm>`_
        with adaptations to use the Darts API. PatchTST-FM is a ~260M-parameter, univariate,
        pretrained time series foundation model for probabilistic forecasting. It uses a
        patch-based transformer encoder architecture with a quantile head producing 99 quantiles
        (0.01 to 0.99).

        PatchTST-FM is univariate-only and does not support any covariates. Multi-component target
        series are handled independently per component (channel-independent processing).

        By default, using this model will automatically download and cache the pre-trained model from
        HuggingFace Hub (ibm-granite/granite-timeseries-patchtst-fm-r1). Alternatively, you can specify
        a local directory containing the model config and weights using the ``local_dir`` parameter.

        By default, this model is deterministic and outputs only the median (0.5 quantile). To enable
        probabilistic forecasts, pass a :class:`~darts.utils.likelihood_models.torch.QuantileRegression`
        instance to the ``likelihood`` parameter. The quantiles used must be a subset of those used during
        PatchTST-FM pre-training (0.01 to 0.99 in steps of 0.01). It is recommended to call
        :func:`predict()` with ``predict_likelihood_parameters=True`` or ``num_samples >> 1`` to get
        meaningful results.

        .. tip::
            You can perform full or partial fine-tuning of the model by setting the ``enable_finetuning``
            parameter. Read more in the parameter description below and in the `Fine-Tuning Examples
            <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series. Maximum value depends on the model's context length minus the output chunk length:
            ``input_chunk_length + output_chunk_length + output_chunk_shift <= 8192``.
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points
            generated using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length`
            prevents auto-regression.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative
            to the input chunk end). This will create a gap between the input and output. Predictions will
            start `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift`
            is set, the model cannot generate autoregressive predictions
            (`n > output_chunk_length`).
        likelihood
            The likelihood model to be used for probabilistic forecasts. Must be ``None`` or an instance of
            :class:`~darts.utils.likelihood_models.torch.QuantileRegression`. If using ``QuantileRegression``,
            the quantiles must be a subset of those used during PatchTST-FM pre-training:
            [0.01, 0.02, ..., 0.99].
            Default: ``None``, which will make the model deterministic (median quantile only).
            When fine-tuning is enabled, the training loss is always computed on all pre-trained quantiles to
            preserve the full distribution, regardless of the ``likelihood`` setting. The ``likelihood``
            parameter only affects prediction output.
        hub_model_name
            The model ID on HuggingFace Hub.
            Default: ``"ibm-granite/granite-timeseries-patchtst-fm-r1"`` (Apache-2.0).
        hub_model_revision
            The model version to use. This can be a branch name, tag name, or commit hash. Default is
            ``None``, which will use the default branch from ``hub_model_name``.
        local_dir
            Optional local directory to load the pre-downloaded model. If specified and the directory is
            empty, the model will be downloaded from HuggingFace Hub and saved to this directory. Default
            is ``None``, which will use a cache directory managed by ``huggingface_hub`` instead.
        **kwargs
            Optional arguments to initialize the pytorch_lightning.Module, pytorch_lightning.Trainer, and
            Darts' :class:`TorchForecastingModel`.

        loss_fn
            PyTorch loss function used for fine-tuning a deterministic PatchTST-FM model. Ignored for
            probabilistic PatchTST-FM when ``likelihood`` is specified. Default: ``nn.MSELoss()``.
        torch_metrics
            A torch metric or a ``MetricCollection`` used for evaluation. A full list of available metrics
            can be found at https://torchmetrics.readthedocs.io/en/latest/. Default: ``None``.
        optimizer_cls
            The PyTorch optimizer class to be used. Default: ``torch.optim.Adam``.
        optimizer_kwargs
            Optionally, some keyword arguments for the PyTorch optimizer (e.g., ``{'lr': 1e-3}``
            for specifying a learning rate). Otherwise, the default values of the selected
            ``optimizer_cls`` will be used. Default: ``None``.
        lr_scheduler_cls
            Optionally, the PyTorch learning rate scheduler class to be used. Specifying ``None``
            corresponds to using a constant learning rate. Default: ``None``.
        lr_scheduler_kwargs
            Optionally, some keyword arguments for the PyTorch learning rate scheduler. Default: ``None``.
        batch_size
            Number of time series (input and output sequences) used in each training pass. Default: ``32``.
        n_epochs
            Number of epochs over which to train the model. Default: ``100``.
        model_name
            Name of the model. Used for creating checkpoints and saving tensorboard data. If not specified,
            defaults to the following string ``"YYYY-mm-dd_HH_MM_SS_torch_model_run_PID"``, where the initial
            part of the name is formatted with the local date and time, while PID is the processed ID
            (preventing models spawned at the same time by different processes to share the same model_name).
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
            If set to ``True``, any previously-existing model with the same name will be reset (all
            checkpoints will be discarded). Default: ``False``.
        save_checkpoints
            Whether to automatically save the untrained model and checkpoints from training. Default: ``False``.
        add_encoders
            A large number of past and future covariates can be automatically generated with `add_encoders`.
            This can be done by adding multiple pre-defined index encoders and/or custom user-made functions
            that will be used as index encoders. Additionally, a transformer such as Darts' :class:`Scaler`
            can be added to transform the generated covariates. This happens all under one hood and only needs
            to be specified at model creation.
            Read :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>` to find out more
            about ``add_encoders``. Default: ``None``.
        random_state
            Controls the randomness of the weights initialization and reproducible forecasting.
        pl_trainer_kwargs
            By default :class:`TorchForecastingModel` creates a PyTorch Lightning Trainer with several useful
            presets that performs the training, validation and prediction processes. These presets include
            automatic checkpointing, tensorboard logging, setting the torch device and more.
            With ``pl_trainer_kwargs`` you can add additional kwargs to instantiate the PyTorch Lightning
            trainer object. Check the `PL Trainer documentation
            <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html>`__ for more information
            about the supported kwargs. Default: ``None``.
        show_warnings
            Whether to show warnings raised from PyTorch Lightning. Useful to detect potential issues of
            your forecasting use case. Default: ``False``.
        enable_finetuning
            Enables model fine-tuning. Only effective if not ``None``.
            If a bool, specifies whether to perform full fine-tuning / training (all parameters are updated)
            or keep all parameters frozen. If a dict, specifies which parameters to fine-tune. Must only
            contain one key-value record. Can be used to:

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
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import PatchTSTFMModel
        >>> series = WeatherDataset().load().astype("float32")
        >>> target = series['p (mbar)'][:100]
        >>> model = PatchTSTFMModel(
        >>>     input_chunk_length=64,
        >>>     output_chunk_length=16,
        >>> )
        >>> model.fit(target)
        >>> pred = model.predict(16)

        .. note::
            PatchTST-FM weights from ``ibm-granite/granite-timeseries-patchtst-fm-r1`` are licensed under
            the `Apache-2.0 License <https://github.com/ibm-granite/granite-tsfm/blob/main/LICENSE>`_,
            copyright IBM. By using this model, you agree to the terms and conditions of the license.
        """
        hf_connector = HuggingFaceConnector(
            model_name=hub_model_name,
            model_revision=hub_model_revision,
            local_dir=local_dir,
        )

        config = hf_connector.load_config()

        # validate input_chunk_length + output_chunk_length <= context_length
        context_length = config["context_length"]
        total_length = input_chunk_length + output_chunk_length + output_chunk_shift
        if total_length > context_length:
            raise_log(
                ValueError(
                    f"`input_chunk_length` ({input_chunk_length}) + `output_chunk_length` "
                    f"({output_chunk_length}) + `output_chunk_shift` ({output_chunk_shift}) = "
                    f"{total_length} cannot exceed model's context_length ({context_length})."
                ),
                logger,
            )

        quantile_levels = config["quantile_levels"]
        # by default (`likelihood=None`), model is deterministic
        # otherwise, only QuantileRegression likelihood is supported
        if likelihood is not None:
            if not isinstance(likelihood, QuantileRegression):
                raise_log(
                    ValueError(
                        f"Only QuantileRegression likelihood is supported for PatchTSTFMModel in Darts. "
                        f"Got {type(likelihood)}."
                    ),
                    logger,
                )
            user_quantiles: list[float] = likelihood.quantiles
            if not set(user_quantiles).issubset(quantile_levels):
                raise_log(
                    ValueError(
                        f"The quantiles for QuantileRegression likelihood {user_quantiles} "
                        f"must be a subset of PatchTST-FM quantiles {quantile_levels}."
                    ),
                    logger,
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
