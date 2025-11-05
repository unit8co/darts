"""
Amazon Chronos 2 Pre-trained Model for Time Series Forecasting
--------------------

Apache-2.0 License from https://github.com/amazon-science/chronos-forecasting/blob/main/LICENSE,
accessed on 4 November 2025:
'
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0

Authors: Abdul Fatir Ansari <ansarnd@amazon.com>
'

Adapted for Darts with custom `PLForecastingModule` and `FoundationModel` integration:
- Remove dependencies on `transformers` and `einops` libraries.
- Load model config and weights from HuggingFace Hub using `HuggingFaceModelMixin`.
- Remove `output_attentions` option from forward pass.
- Integrate likelihood model and loss computation with Darts `QuantileRegression`, and
    remove original loss computation in forward pass.
-
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union, cast

import torch
from torch import nn

from darts.logging import get_logger, raise_if, raise_if_not
from darts.models.forecasting.chronos2_submodels import (
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

    @classmethod
    def editable_fields(cls) -> list[str]:
        """
        Fields that maybe modified during the fine-tuning stage.
        """
        return ["context_length", "max_output_patches"]


class _Chronos2Module(PLForecastingModule):
    """
    Chronos2 module
    """

    _supports_long_horizon: bool = True
    _supports_future_covariates: bool = True
    _supports_sdpa: bool = True

    quantiles: torch.Tensor

    def __init__(
        self,
        d_model: int = 512,
        d_kv: int = 64,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        layer_norm_epsilon: float = 1e-6,
        initializer_factor: float = 0.05,
        feed_forward_proj: str = "relu",
        vocab_size: int = 2,
        rope_theta: float = 10000.0,
        attn_implementation: Literal["eager", "sdpa"] | None = None,
        chronos_config: dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.rope_theta = rope_theta

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        raise_if(
            self.is_gated_act,
            "gated activation is not supported",
            logger,
        )

        # Attention implementation - default to "sdpa" if not specified
        self.attn_implementation = attn_implementation or "sdpa"
        raise_if_not(
            self.attn_implementation in ["eager", "sdpa"],
            f"attn_implementation {self.attn_implementation} not supported",
            logger,
        )

        self.chronos_config = _Chronos2ForecastingConfig(**chronos_config)

        # Only decoder_start_id (and optionally REG token)
        if self.chronos_config.use_reg_token:
            self.reg_token_id = 1

        raise_if_not(
            self.chronos_config.input_patch_size
            == self.chronos_config.output_patch_size,
            f"input_patch_size and output_patch_size sizes must be equal, "
            f"but found {self.chronos_config.input_patch_size} and {self.chronos_config.output_patch_size}",
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

        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles)
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = _ResidualBlock(
            in_dim=self.d_model,
            h_dim=self.d_ff,
            out_dim=self.num_quantiles * self.chronos_config.output_patch_size,
            act_fn_name=self.dense_act_fn,
            dropout_p=self.dropout_rate,
        )

    def _prepare_patched_context(
        self, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        context_mask = (
            context_mask.to(context.dtype)
            if context_mask is not None
            else torch.isnan(context).logical_not().to(context.dtype)
        )

        batch_size, context_length = context.shape
        # truncate context if it's longer than model's context length
        if context_length > self.chronos_config.context_length:
            context = context[..., -self.chronos_config.context_length :]
            context_mask = context_mask[..., -self.chronos_config.context_length :]

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
        future_covariates: torch.Tensor | None,
        future_covariates_mask: torch.Tensor | None,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_patch_size = self.chronos_config.output_patch_size
        if future_covariates is not None:
            future_covariates, _ = self.instance_norm(future_covariates, loc_scale)
            future_covariates = cast(torch.Tensor, future_covariates)
            future_covariates = future_covariates.to(self.dtype)

            if future_covariates_mask is None:
                future_covariates_mask = (
                    torch.isnan(future_covariates)
                    .logical_not()
                    .to(future_covariates.dtype)
                )

            future_covariates = torch.where(
                future_covariates_mask > 0.0, future_covariates, 0.0
            )

            if torch.isnan(future_covariates).any():
                # TODO: replace this
                raise ValueError(
                    "future_covariates contains NaN values at indices not masked by future_covariates_mask. "
                    "Input the correct future_covariates_mask or omit it to automatically infer the mask based on NaN "
                    "values."
                )

            # add padding if the length of future_covariates is not an integer multiple of output_patch_size
            if num_output_patches * output_patch_size > future_covariates.shape[-1]:
                padding_shape = (
                    *future_covariates.shape[:-1],
                    num_output_patches * output_patch_size
                    - future_covariates.shape[-1],
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
        else:
            patched_future_covariates = torch.zeros(
                batch_size,
                num_output_patches,
                output_patch_size,
                device=self.device,
                dtype=self.dtype,
            )
            patched_future_covariates_mask = torch.zeros(
                batch_size,
                num_output_patches,
                output_patch_size,
                device=self.device,
                dtype=self.dtype,
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

    # TODO: fine-tuning support w/ normalised loss
    def _forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
    ) -> torch.Tensor:
        """Forward pass of the Chronos2 model.

        Parameters
        ----------
        context
            Input tensor of shape (batch_size, context_length) containing the historical values
        context_mask
            Binary mask tensor of same shape as context indicating which values are valid (1) vs missing (0)
            If missing, the context_mask will be automatically constructed based on the NaN values in context.
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
        future_covariates_mask
            Binary mask tensor of same shape as future_covariates indicating which future values are known
            If omitted, future_covariates_mask is automatically constructed based on future_covariates with
            all non-NaN values treated as known future values.
        num_output_patches
            Number of output patches to generate predictions for, by default 1
            When ``future_covariates`` and/or ``future_target`` are provided, num_output_patches should be large enough
            to accommodate their lengths, i.e., num_output_patches * output_patch_size >= future_length

        Returns
        -------
        Chronos2Output containing:
        - loss: Training loss, if `future_target` is provided
        - quantile_preds: Quantile predictions of shape
            (batch_size, num_quantiles, num_output_patches * output_patch_size).
            quantile_preds will contain an entry for every time series in the context batch regardless of whether it
            was a known future covariate.
        - enc_time_self_attn_weights: Time self attention weights, if output_attentions=True
        - enc_group_self_attn_weights: Group self attention weights, if output_attentions=True
        """

        batch_size = context.shape[0]
        patched_context, attention_mask, loc_scale = self._prepare_patched_context(
            context=context, context_mask=context_mask
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

        patched_future, patched_future_covariates_mask = self._prepare_patched_future(
            future_covariates=future_covariates,
            future_covariates_mask=future_covariates_mask,
            loc_scale=loc_scale,
            num_output_patches=num_output_patches,
            batch_size=batch_size,
        )
        future_attention_mask = torch.ones(
            batch_size, num_output_patches, dtype=self.dtype, device=self.device
        )

        # get future embeddings of shape (batch, num_output_patches, d_model)
        future_embeds: torch.Tensor = self.input_patch_embedding(patched_future)

        # concatenate context and future embeddings and masks
        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask, future_attention_mask], dim=-1)

        if group_ids is None:
            # by default, each time series is treated independently, i.e., no mixing across the batch
            group_ids = torch.arange(batch_size, dtype=torch.long, device=self.device)

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

    def forward(self, x_in: PLModuleInput, *args, **kwargs) -> Any:
        x_past, x_future, _ = x_in
        # According to `self._process_input_batch()` in `PLForecastingModule`,
        # x_past is a stack of [past_target, past_covariates, historic_future_covariates],
        # while x_future is just future_covariates.
        # So here we need to create future_covariates in Chronos2's format that is
        # a stack of [past_target, past_covariates, future_covariates].
        batch_size, past_length, n_variables = x_past.shape
        future_length = self.output_chunk_length or 0
        future_covariates = torch.full(
            (batch_size, future_length, n_variables),
            torch.nan,
            device=x_past.device,
        )
        if x_future is not None:
            n_future_covs = x_future.shape[-1]
            future_covariates[:, :, -n_future_covs:] = x_future

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

        quantile_preds = self._forward(
            context=context,
            group_ids=group_ids,
            future_covariates=future_covariates,
            num_output_patches=num_output_patches,
        )

        # reshape quantile_preds to (batch, time, vars, quantiles)
        quantile_preds = quantile_preds.view(
            batch_size,
            n_variables,
            num_output_patches,
            self.num_quantiles,
            self.chronos_config.output_patch_size,
        )
        quantile_preds = quantile_preds.permute(0, 2, 4, 1, 3).reshape(
            batch_size,
            num_output_patches * self.chronos_config.output_patch_size,
            n_variables,
            self.num_quantiles,
        )

        # truncate to future_length
        quantile_preds = quantile_preds[:, :future_length]

        # truncate to only target variables
        quantile_preds = quantile_preds[:, :, : self.n_targets, :]

        return quantile_preds


class Chronos2Model(FoundationModel, HuggingFaceModelMixin):
    _repo_id = "amazon/chronos-2"
    _repo_commit = "18128c7b4f3fd286f06d6d4efe1d252f1d2a9a7c"

    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        local_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        self.local_dir = local_dir

        # validate output_chunk_shift
        raise_if_not(
            output_chunk_shift == 0,
            f"`output_chunk_shift` {output_chunk_shift} other than 0 is not supported for now",
            logger,
        )

        # load model config for validation
        config = self._load_config()
        chronos_config = config["chronos_config"]

        # validate `input_chunk_length` against model's context_length
        context_length = chronos_config["context_length"]
        raise_if(
            input_chunk_length > context_length,
            f"`input_chunk_length` {input_chunk_length} cannot be greater than "
            f"model's context_length {context_length}",
            logger,
        )

        # validate `output_chunk_length` against model's prediction length
        prediction_length = (
            chronos_config["output_patch_size"] * chronos_config["max_output_patches"]
        )
        raise_if(
            output_chunk_length > prediction_length,
            f"`output_chunk_length` {output_chunk_length} cannot be greater than "
            f"model's maximum prediction length {prediction_length}",
            logger,
        )

        super().__init__(**kwargs)

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        pl_module_params = self.pl_module_params or {}

        # convert Chronos 2's quantiles into Darts' QuantileRegression likelihood model
        pl_module_params["likelihood"] = QuantileRegression(
            quantiles=self._load_config()["chronos_config"]["quantiles"]
        )

        module = self._load_model(
            module_class=_Chronos2Module,
            pl_module_params=pl_module_params,
        )
        return module
