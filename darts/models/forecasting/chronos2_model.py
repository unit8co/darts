"""
Chronos-2
---------
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union, cast

import torch
from torch import nn

from darts.logging import get_logger, raise_log
from darts.models.components.chronos2_submodels import (
    _Chronos2Encoder,
    _InstanceNorm,
    _Patch,
    _ResidualBlock,
)
from darts.models.forecasting.foundation_model import (
    FoundationModel,
    HuggingFaceModelMixin,
)
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
)
from darts.utils.data.torch_datasets.utils import PLModuleInput, TorchTrainingSample
from darts.utils.likelihood_models.torch import QuantileRegression

logger = get_logger(__name__)


@dataclass
class _Chronos2ForecastingConfig:
    context_length: int
    output_patch_size: int
    input_patch_size: int
    input_patch_stride: int
    quantiles: list[float]
    use_reg_token: bool = False
    use_arcsinh: bool = False
    max_output_patches: int = 1
    time_encoding_scale: int | None = None


class _Chronos2Module(PLForecastingModule):
    def __init__(
        self,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        feed_forward_proj: str = "relu",
        rope_theta: float = 10000.0,
        attn_implementation: Literal["eager", "sdpa"] | None = None,
        chronos_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """PyTorch module implementing the Chronos-2 model, ported from
        `amazon-science/chronos-forecasting <https://github.com/amazon-science/chronos-forecasting>`_ and
        adapted for Darts :class:`PLForecastingModule` interface.

        Parameters
        ----------
        d_model
            Dimension of the model embeddings, also called "model size" in Transformer.
        d_kv
            Dimension of the key and value projections in multi-head attention.
        d_ff
            Dimension of the feed-forward network hidden layer.
        num_layers
            Number of Chronos-2 encoder layers.
        num_heads
            Number of attention heads in each encoder block.
        dropout_rate
            Dropout rate of the model.
        layer_norm_epsilon
            Epsilon value for layer normalization layers.
        feed_forward_proj
            Acctivation of feed-forward network.
        rope_theta
            Base period for Rotary Position Embeddings (RoPE).
        attn_implementation
            Attention implementation to use. If None, defaults to "sdpa".
        chronos_config
            Configuration parameters for Chronos-2 model. See :class:`_Chronos2ForecastingConfig` for details.
        **kwargs
            all parameters required for :class:`darts.models.forecasting.pl_forecasting_module.PLForecastingModule`
            base class.
        """

        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.feed_forward_proj = feed_forward_proj
        self.rope_theta = rope_theta

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if self.is_gated_act:
            raise_log(
                ValueError("gated activation is not supported"),
                logger,
            )

        # Attention implementation - default to "sdpa" if not specified
        self.attn_implementation = attn_implementation or "sdpa"
        if self.attn_implementation not in ["eager", "sdpa"]:
            raise_log(
                ValueError(
                    f"attn_implementation {self.attn_implementation} is not supported"
                ),
                logger,
            )

        # Chronos-2 forecasting specific config
        chronos_config = chronos_config or {}
        self.chronos_config = _Chronos2ForecastingConfig(**chronos_config)

        # Only decoder_start_id (and optionally REG token)
        if self.chronos_config.use_reg_token:
            self.reg_token_id = 1

        if (
            self.chronos_config.input_patch_size
            != self.chronos_config.output_patch_size
        ):
            raise_log(
                ValueError(
                    f"input_patch_size and output_patch_size sizes must be equal, "
                    f"but found {self.chronos_config.input_patch_size} and {self.chronos_config.output_patch_size}"
                ),
                logger,
            )

        self.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(self.vocab_size, self.d_model)

        # Input patch embedding layer
        self.input_patch_embedding = _ResidualBlock(
            # x3 for [time_embedding, patch, patch_mask]
            in_dim=self.chronos_config.input_patch_size * 3,
            h_dim=self.d_ff,
            out_dim=self.d_model,
            act_fn_name=self.dense_act_fn,
            dropout_p=self.dropout_rate,
        )

        # patching layer
        self.patch = _Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )

        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = _InstanceNorm(use_arcsinh=self.chronos_config.use_arcsinh)

        self.encoder = _Chronos2Encoder(
            d_model=self.d_model,
            d_kv=self.d_kv,
            d_ff=self.d_ff,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            rope_theta=self.rope_theta,
            attn_implementation=self.attn_implementation,
            dense_act_fn=self.dense_act_fn,
            layer_norm_epsilon=self.layer_norm_epsilon,
            is_gated_act=self.is_gated_act,
            num_layers=self.num_layers,
        )

        quantiles = self.chronos_config.quantiles
        self.num_quantiles = len(quantiles)
        quantiles_tensor = torch.tensor(quantiles)
        self.register_buffer("quantiles", quantiles_tensor, persistent=False)

        # gather indices of user-specified quantiles
        user_quantiles: list[float] = (
            self.likelihood.quantiles
            if isinstance(self.likelihood, QuantileRegression)
            else [0.5]
        )
        self.user_quantile_indices = [quantiles.index(q) for q in user_quantiles]

        self.output_patch_embedding = _ResidualBlock(
            in_dim=self.d_model,
            h_dim=self.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.output_patch_size,
            act_fn_name=self.dense_act_fn,
            dropout_p=self.dropout_rate,
        )

    def _prepare_patched_context(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        context_mask = torch.isnan(context).logical_not().to(context.dtype)

        batch_size, _ = context.shape

        # scaling
        context, loc_scale = self.instance_norm(context)

        # scaling is done in 32-bit precision, then the context is moved to model's dtype
        context = context.to(self.dtype)
        context_mask = context_mask.to(self.dtype)

        # patching
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(context_mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)

        # attention_mask = 1 if at least one item in the patch is observed
        attention_mask = patched_mask.sum(dim=-1) > 0  # (batch_size, num_patches)
        num_context_patches = attention_mask.shape[-1]

        # context time encoding: every observation is assigned a sequential time index,
        # scaled by model's context length = [-C, -(C-1), ..., -1] / context_length
        final_context_length = (
            num_context_patches * self.chronos_config.input_patch_size
        )
        context_time_enc = torch.arange(
            start=-final_context_length, end=0, device=self.device, dtype=torch.float32
        )
        context_time_enc = context_time_enc.div(
            cast(int, self.chronos_config.time_encoding_scale)
        ).to(self.dtype)
        context_time_enc = context_time_enc.view(
            1, num_context_patches, self.chronos_config.input_patch_size
        ).expand(
            batch_size,
            num_context_patches,
            self.chronos_config.input_patch_size,
        )

        # concat time encoding, context and mask along the last (feature) dim
        patched_context = torch.cat(
            [context_time_enc, patched_context, patched_mask], dim=-1
        )

        return patched_context, attention_mask, loc_scale

    def _prepare_patched_future(
        self,
        future_covariates: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_patch_size = self.chronos_config.output_patch_size
        future_covariates, _ = self.instance_norm(future_covariates, loc_scale)
        future_covariates = cast(torch.Tensor, future_covariates)
        future_covariates = future_covariates.to(self.dtype)

        future_covariates_mask = (
            torch.isnan(future_covariates).logical_not().to(future_covariates.dtype)
        )

        future_covariates = torch.where(
            future_covariates_mask > 0.0, future_covariates, 0.0
        )

        # add padding if the length of future_covariates is not an integer multiple of output_patch_size
        if num_output_patches * output_patch_size > future_covariates.shape[-1]:
            padding_shape = (
                *future_covariates.shape[:-1],
                num_output_patches * output_patch_size - future_covariates.shape[-1],
            )
            future_covariates = torch.cat(
                [
                    future_covariates,
                    torch.zeros(padding_shape).to(future_covariates),
                ],
                dim=-1,
            )
            future_covariates_mask = torch.cat(
                [
                    future_covariates_mask,
                    torch.zeros(padding_shape).to(future_covariates_mask),
                ],
                dim=-1,
            )

        patched_future_covariates = future_covariates.view(
            batch_size, num_output_patches, output_patch_size
        )
        patched_future_covariates_mask = future_covariates_mask.view(
            batch_size, num_output_patches, output_patch_size
        )

        # future time encoding: every future timestep is assigned a sequential time index,
        # scaled by model's context length = [0, 1, ..., h-1] / context_length
        final_future_length = num_output_patches * output_patch_size
        future_time_enc = torch.arange(
            start=0, end=final_future_length, device=self.device, dtype=torch.float32
        )
        future_time_enc = future_time_enc.div(
            cast(int, self.chronos_config.time_encoding_scale)
        ).to(self.dtype)
        future_time_enc = future_time_enc.view(
            1, num_output_patches, output_patch_size
        ).expand(
            batch_size,
            num_output_patches,
            output_patch_size,
        )

        patched_future = torch.cat(
            [
                future_time_enc,
                patched_future_covariates,
                patched_future_covariates_mask,
            ],
            dim=-1,
        )

        return patched_future, patched_future_covariates_mask

    def _forward(
        self,
        context: torch.Tensor,
        group_ids: torch.Tensor,
        future_covariates: torch.Tensor,
        num_output_patches: int = 1,
    ) -> torch.Tensor:
        """Original forward pass of the Chronos-2 model.

        Parameters
        ----------
        context
            Input tensor of shape (batch_size, context_length) containing the historical values
        group_ids : torch.Tensor | None, optional
            Group IDs of shape (batch_size,) indicating which times series in the batch form a group.
            A group indicates a task, for example, for a batch of size 6:
            - if groups_ids = [0, 1, 2, 3, 4, 5], each time series is treated independently.
            - if groups_ids = [0, 0, 1, 1, 1, 2], information is mixed across the first two time series (id=0),
                the next three time series (id=1) and the last time series is treated separately. Information is
                NOT shared among time series from different groups.
            The ordering and specific values of group_ids are not important, all time series with the same group
            ID form a group.
        future_covariates
            Tensor of shape (batch_size, future_length) containing future covariates. Note that the size of
            tensor along the first axis is equal to the batch_size. This means that future values (which may be NaNs)
            must be provided for each time series in the batch. For any time series that need to be forecasted, the
            future_covariates can be set to NaNs, if ``future_covariates_mask`` is omitted or to an arbitrary dummy
            value when ``future_covariates_mask`` is provided. ``future_covariates`` can be used with ``group_ids``
            to construct heterogenous forecasting tasks in a single batch. For example:
            - future_covariates = [[nan, ...], [nan, ...], [v1, ...], [v2, ...], [nan, ...], [nan, ...]]
            - groups_ids = [0, 0, 1, 1, 1, 2]
            - future_covariates_mask = None
            contains 3 types of forecasting tasks:
            - [0, 0]: The first task, both future_covariates are missing, which implies that the two time series need to
                be forecasted jointly, i.e., multivariate forecasting.
            - [1, 1, 1]: In the next task, the first two future_covariates are available and the last one is missing
                ([v1, ...], [v2, ...], [nan, ...]), where [v1, ...] and [v1, ...] denote an arbitrary sequence of
                values. This indicates that the first two time series are known covariates and the third one needs to be
                forecasted by the model.
            - [2]: The last task has a single time series in the group which needs to be forecasted independently.
            There is no theoretical limit on the number of time series in a group, i.e., the number of targets and known
            covariates in a task. The above setup subsumes tasks with past-only covariates as the model's prediction for
            those time series can simply be ignored downstream.
        num_output_patches
            Number of output patches to generate predictions for, by default 1
            When ``future_covariates`` and/or ``future_target`` are provided, num_output_patches should be large enough
            to accommodate their lengths, i.e., num_output_patches * output_patch_size >= future_length

        Returns
        -------
        torch.Tensor
            Quantile predictions of shape `(batch_size, n_variables * n_output_patches * n_quantiles * patch_size)`.
            quantile_preds will contain an entry for every time series in the context batch regardless of whether it
            was a known future covariate.
        """

        batch_size = context.shape[0]
        patched_context, attention_mask, loc_scale = self._prepare_patched_context(
            context=context
        )
        num_context_patches = attention_mask.shape[-1]

        # get input embeddings of shape (batch, num_context_patches, d_model)
        input_embeds: torch.Tensor = self.input_patch_embedding(patched_context)
        # append [REG] special token embedding, if needed
        if self.chronos_config.use_reg_token:
            reg_input_ids = torch.full(
                (batch_size, 1), self.reg_token_id, device=input_embeds.device
            )
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [
                    attention_mask.to(self.dtype),
                    torch.ones_like(reg_input_ids).to(self.dtype),
                ],
                dim=-1,
            )

        patched_future, _ = self._prepare_patched_future(
            future_covariates=future_covariates,
            loc_scale=loc_scale,
            num_output_patches=num_output_patches,
            batch_size=batch_size,
        )
        future_attention_mask = torch.ones(
            batch_size,
            num_output_patches,
            dtype=attention_mask.dtype,
            device=self.device,
        )

        # get future embeddings of shape (batch, num_output_patches, d_model)
        future_embeds: torch.Tensor = self.input_patch_embedding(patched_future)

        # concatenate context and future embeddings and masks
        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask, future_attention_mask], dim=-1)

        hidden_states: torch.Tensor = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            group_ids=group_ids,
        )

        assert hidden_states.shape == (
            batch_size,
            num_context_patches + 1 + num_output_patches,
            self.d_model,
        )

        # slice the last num_output_patches hidden states to be input into the output_patch_embedding
        forecast_embeds = hidden_states[:, -num_output_patches:]
        quantile_preds: torch.Tensor = self.output_patch_embedding(forecast_embeds)

        quantile_preds = quantile_preds.view(batch_size, -1)
        quantile_preds = self.instance_norm.inverse(quantile_preds, loc_scale)

        return quantile_preds

    # TODO: fine-tuning support w/ normalized loss
    # Currently, Darts own `RINorm` is not used as Chronos-2 has its own implementation. Major differences
    # 1. Chronos-2 `RINorm` normalizes both target and covariates, while Darts normalizes target only.
    # 2. Chronos-2 `RINorm` additionally applies `arcsinh` transformation after standardization
    # 3. Chronos-2 uses normalized values for loss computation, while Darts uses denormalized values.
    # We need to think about how best to implement Chronos-2 `RINorm` in `io_processor()` without
    # breaking existing behavior, while also allowing fine-tuning with normalized loss.
    def forward(self, x_in: PLModuleInput, *args, **kwargs) -> Any:
        """Chronos-2 model forward pass.

        Parameters
        ----------
        x_in
            comes as tuple `(x_past, x_future, x_static)` where `x_past` is the input/past chunk and `x_future`
            is the output/future chunk. Input dimensions are `(n_samples, n_time_steps, n_variables)`

        Returns
        -------
        torch.Tensor
            the output tensor in the shape of `(n_samples, n_time_steps, n_targets, n_quantiles)` for
            probabilistic forecasts, or `(n_samples, n_time_steps, n_targets, 1)` for
            deterministic forecasts (median only).
        """
        x_past, x_future, _ = x_in
        # x_past is a stack of [past_target, past_covariates, historic_future_covariates],
        # x_future is just future_covariates.
        # So here we need to create `future_covariates` in Chronos2's format that is
        # a stack of [past_target (NaNs), past_covariates (NaNs), future_covariates].
        batch_size, past_length, n_variables = x_past.shape
        output_chunk_length = self.output_chunk_length or 0
        output_chunk_shift = self.output_chunk_shift
        future_length = output_chunk_shift + output_chunk_length
        future_covariates = torch.full(
            (batch_size, future_length, n_variables),
            torch.nan,
            device=x_past.device,
        )
        if x_future is not None:
            n_future_covs = x_future.shape[-1]
            future_covariates[:, -output_chunk_length:, -n_future_covs:] = x_future

        # reshape x_past and future_covariates to (batch * vars, time)
        context = x_past.permute(0, 2, 1).reshape(-1, past_length)
        future_covariates = future_covariates.permute(0, 2, 1).reshape(
            -1, future_length
        )

        # create group_ids according to sample index within the batch
        group_ids = torch.arange(batch_size, device=context.device).repeat_interleave(
            n_variables
        )

        # determine minimum number of patches to cover future_length
        num_output_patches = math.ceil(
            future_length / self.chronos_config.output_patch_size
        )

        # call original Chronos-2 forward pass
        # Unlike the original, we remove `context_mask`, `future_covariates_mask`, `future_target`,
        # `future_target_mask`, and `output_attentions` parameters. They are not needed for Darts'
        # implementation.
        # We also remove `einops` rearrange operation at the end so the raw output tensor is returned,
        # in shape of `(batch, vars * patches * quantiles * patch_size)`
        quantile_preds = self._forward(
            context=context,
            group_ids=group_ids,
            future_covariates=future_covariates,
            num_output_patches=num_output_patches,
        )

        # The permutation and reshaping operations below replace the `einops` rearrange
        # operations in the original Chronos-2 code to return the output tensor in Darts'
        # expected shape.
        # reshape quantile_preds to (batch, vars, patches, quantiles, patch_size)
        quantile_preds = quantile_preds.view(
            batch_size,
            n_variables,
            num_output_patches,
            self.num_quantiles,
            self.chronos_config.output_patch_size,
        )
        # permute and reshape to (batch, time, vars, quantiles)
        quantile_preds = quantile_preds.permute(0, 2, 4, 1, 3).reshape(
            batch_size,
            num_output_patches * self.chronos_config.output_patch_size,
            n_variables,
            self.num_quantiles,
        )

        # truncate to output_chunk_length
        quantile_preds = quantile_preds[:, output_chunk_shift:future_length, :, :]

        # select only target variables
        quantile_preds = quantile_preds[:, :, : self.n_targets, :]

        # select only user-specified quantiles or median if deterministic
        quantile_preds = quantile_preds[:, :, :, self.user_quantile_indices]

        return quantile_preds


class Chronos2Model(FoundationModel, HuggingFaceModelMixin):
    _repo_id = "amazon/chronos-2"
    _repo_commit = "18128c7b4f3fd286f06d6d4efe1d252f1d2a9a7c"

    # Fine-tuning is turned off for now pending proper fine-tuning support
    # and configuration.
    _allows_finetuning = False

    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        likelihood: Optional[QuantileRegression] = None,
        local_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        """Chronos-2 Model for zero-shot forecasting.

        This is an implementation of Amazon's Chronos-2 model [1]_, [2]_, ported from
        `amazon-science/chronos-forecasting <https://github.com/amazon-science/chronos-forecasting>`_
        with adaptations to use the Darts API. From the original authors:

        "Chronos-2 is a 120M-parameter, encoder-only time series foundation model for zero-shot forecasting. It supports
        univariate, multivariate, and covariate-informed tasks within a single architecture. Inspired by the T5 encoder,
        Chronos-2 produces multi-step-ahead quantile forecasts and uses a group attention mechanism for efficient
        in-context learning across related series and covariates. Trained on a combination of real-world and large-scale
        synthetic datasets, it achieves state-of-the-art zero-shot accuracy among public models on fev-bench, GIFT-Eval,
        and Chronos Benchmark II. Chronos-2 is also highly efficient, delivering over 300 time series forecasts per
        second on a single A10G GPU and supporting both GPU and CPU inference."

        This model supports past covariates (known for `input_chunk_length` points before prediction time),
        and future covariates (known for `output_chunk_length` points after prediction time).

        By default, using this model will automatically download the pre-trained model from HuggingFace Hub
        (amazon/chronos-2). Alternatively, you can specify a local directory containing the model config and weights
        using the ``local_dir`` parameter.

        By default, this model is deterministic and outputs only the median (0.5 quantile). To enable probabilistic
        forecasts, pass a :class:`~darts.utils.likelihood_models.torch.QuantileRegression` instance to the
        ``likelihood`` parameter. The quantiles used must be a subset of those used during Chronos-2 pre-training, see
        below for details. It is recommended to call :func:`predict()` with ``predict_likelihood_parameters=True``
        or ``num_samples >> 1`` to get meaningful results.

        Fine-tuning of Chronos-2 is not supported at the moment.

        Chronos-2 is licensed under the Apache-2.0 License, copyright Amazon.com, Inc. or its affiliates. By using
        this model, you agree to the terms and conditions of the license.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it).
            Maximum is 8192 for Chronos-2.
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
            from future covariates to use as a model input (if the model supports future covariates). It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
            using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents
            auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
            the model from using future values of past and / or future covariates for prediction (depending on the
            model's covariate support).
            For Chronos-2, `output_chunk_length + output_chunk_shift` must be less than or equal to 1024.
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
            [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
            0.95, 0.99].
            Default is ``None``, which will make Chronos-2 deterministic (median quantile only).
        local_dir
            Optional local directory to load the model config and weights from instead of downloading them from
            HuggingFace Hub.
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
        use_reversible_instance_norm
            Whether to use reversible instance normalization `RINorm` against distribution shift. Ignored by
            Chronos-2 as it has its own `RINorm` implementation.
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

        References
        ----------
        .. [1] A. Ansari, O. Shchur, J. Küken et al. "Chronos-2: From Univariate to Universal Forecasting", 2025.
                arXiv https://arxiv.org/abs/2510.15821.
        .. [2] "Introducing Chronos-2: From univariate to universal forecasting", 2025. Amazon Science Blog.
                https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting

        Examples
        --------
        >>> from darts.datasets import WeatherDataset
        >>> from darts.models import Chronos2Model
        >>> # load data in float32 format (macOS issues with float64 and PyTorch)
        >>> series = WeatherDataset().load().astype("float32")
        >>> # predicting atmospheric pressure
        >>> target = series['p (mbar)'][:100]
        >>> # optionally, past observed rainfall (pretending to be unknown beyond index 100)
        >>> past_cov = series['rain (mm)'][:100]
        >>> # optionally, use future temperatures (pretending this component is a forecast)
        >>> future_cov = series['T (degC)'][:106]
        >>> # by default, Chronos2Model is deterministic; to enable probabilistic forecasts,
        >>> # set likelihood to QuantileRegression and use a subset of the pre-trained quantiles
        >>> model = Chronos2Model(
        >>>     input_chunk_length=6,
        >>>     output_chunk_length=6,
        >>> )
        >>> # calling fit is still mandatory to ensure consistent number of components; however,
        >>> # Chronos2Model is training-free and the model weights are not updated
        >>> model.fit(target, past_covariates=past_cov, future_covariates=future_cov)
        >>> # when Chronos2Model is probabilistic, set ``predict_likelihood_parameters=True``
        >>> # or ``num_samples>>1`` to get meaningful results
        >>> pred = model.predict(6)
        >>> print(pred.all_values())
            [[[1005.7576 ]]
            [[1005.7418 ]]
            [[1005.7186 ]]
            [[1005.7074 ]]
            [[1005.6928 ]]
            [[1005.69617]]]

        .. note::
            Due to differences in probabilistic sampling methods, zero-shot forecasts obtained here would differ from
            those obtained using the original implementation when prediction horizon `n` is larger than 1024.
        """
        self.local_dir = local_dir

        # load model config for validation
        config = self._load_config()
        chronos_config = config["chronos_config"]

        # validate `input_chunk_length` against model's context_length
        context_length = chronos_config["context_length"]
        if input_chunk_length > context_length:
            raise_log(
                ValueError(
                    f"`input_chunk_length` {input_chunk_length} cannot be greater than "
                    f"model's context_length {context_length}"
                ),
                logger,
            )

        # validate `output_chunk_length` and `output_chunk_shift` against model's prediction length
        prediction_length = (
            chronos_config["output_patch_size"] * chronos_config["max_output_patches"]
        )
        if output_chunk_length + output_chunk_shift > prediction_length:
            raise_log(
                ValueError(
                    f"`output_chunk_length` {output_chunk_length} plus `output_chunk_shift` {output_chunk_shift} "
                    f"cannot be greater than model's maximum prediction length {prediction_length}"
                ),
                logger,
            )

        quantiles = chronos_config["quantiles"]
        # by default (`likelihood=None`), model is deterministic
        # otherwise, only QuantileRegression likelihood is supported and quantiles must be
        # a subset of Chronos-2 quantiles
        if likelihood is not None:
            if not isinstance(likelihood, QuantileRegression):
                raise_log(
                    ValueError(
                        f"Only QuantileRegression likelihood is supported for Chronos2Model in Darts. "
                        f"Got {type(likelihood)}."
                    ),
                    logger,
                )
            user_quantiles: list[float] = likelihood.quantiles
            if not set(user_quantiles).issubset(quantiles):
                raise_log(
                    ValueError(
                        f"The quantiles for QuantileRegression likelihood {user_quantiles} "
                        f"must be a subset of Chronos-2 quantiles {quantiles}."
                    ),
                    logger,
                )

        super().__init__(enable_finetuning=False, **kwargs)

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        pl_module_params = self.pl_module_params or {}

        module = self._load_model(
            module_class=_Chronos2Module,
            pl_module_params=pl_module_params,
        )
        return module
