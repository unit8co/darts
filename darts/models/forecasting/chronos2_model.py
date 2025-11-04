# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

import copy
import os
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union, cast

import torch
from chronos.chronos_bolt import InstanceNorm, Patch
from einops import rearrange, repeat
from torch import nn

from darts.logging import get_logger, raise_if, raise_if_not
from darts.models.forecasting.foundation_model import (
    FoundationModel,
    HuggingFaceModelMixin,
)
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
)
from darts.utils.data.torch_datasets.utils import TorchTrainingSample

from .layers import (
    ResidualBlock,
)
from .unimported import (
    Chronos2Encoder,
    Chronos2EncoderOutput,
    Chronos2Output,
)

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
        )

        # Attention implementation - default to "sdpa" if not specified
        self.attn_implementation = attn_implementation or "sdpa"
        raise_if_not(
            self.attn_implementation in ["eager", "sdpa"],
            f"attn_implementation {self.attn_implementation} not supported",
        )

        self.chronos_config = _Chronos2ForecastingConfig(**chronos_config)

        raise_if_not(
            self.chronos_config.input_patch_size
            == self.chronos_config.output_patch_size,
            f"input_patch_size and output_patch_size sizes must be equal, "
            f"but found {self.chronos_config.input_patch_size} and {self.chronos_config.output_patch_size}",
        )

        self.vocab_size = 2 if self.chronos_config.use_reg_token else 1
        self.shared = nn.Embedding(self.vocab_size, self.d_model)

        # Input patch embedding layer
        self.input_patch_embedding = ResidualBlock(
            # x3 for [time_embedding, patch, patch_mask]
            in_dim=self.chronos_config.input_patch_size * 3,
            h_dim=self.d_ff,
            out_dim=self.d_model,
            act_fn_name=self.dense_act_fn,
            dropout_p=self.dropout_rate,
        )

        # patching layer
        self.patch = Patch(
            patch_size=self.chronos_config.input_patch_size,
            patch_stride=self.chronos_config.input_patch_stride,
        )

        # instance normalization, also referred to as "scaling" in Chronos and GluonTS
        self.instance_norm = InstanceNorm(use_arcsinh=self.chronos_config.use_arcsinh)

        encoder_config = copy.deepcopy()
        encoder_config.is_decoder = False
        self.encoder = Chronos2Encoder(encoder_config)

        self.num_quantiles = len(self.chronos_config.quantiles)
        quantiles = torch.tensor(self.chronos_config.quantiles)
        self.quantiles: torch.Tensor
        self.register_buffer("quantiles", quantiles, persistent=False)

        self.output_patch_embedding = ResidualBlock(
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
        context_time_enc = (
            repeat(
                context_time_enc,
                "(n p) -> b n p",
                b=batch_size,
                n=num_context_patches,
                p=self.chronos_config.input_patch_size,
            )
            .div(cast(int, self.chronos_config.time_encoding_scale))
            .to(self.dtype)
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

            patched_future_covariates = rearrange(
                future_covariates,
                "b (n p) -> b n p",
                n=num_output_patches,
                p=output_patch_size,
            )
            patched_future_covariates_mask = rearrange(
                future_covariates_mask,
                "b (n p) -> b n p",
                n=num_output_patches,
                p=output_patch_size,
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
        future_time_enc = (
            repeat(
                future_time_enc,
                "(n p) -> b n p",
                b=batch_size,
                n=num_output_patches,
                p=output_patch_size,
            )
            .div(cast(int, self.chronos_config.time_encoding_scale))
            .to(self.dtype)
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

    def _compute_loss(
        self,
        quantile_preds: torch.Tensor,
        future_target: torch.Tensor,
        future_target_mask: torch.Tensor | None,
        patched_future_covariates_mask: torch.Tensor,
        loc_scale: tuple[torch.Tensor, torch.Tensor],
        num_output_patches: int,
    ) -> torch.Tensor:
        batch_size = future_target.shape[0]
        output_patch_size = self.chronos_config.output_patch_size
        assert (
            quantile_preds.shape[0] == batch_size
            and quantile_preds.shape[-1] >= future_target.shape[-1]
        )

        # normalize target and mask
        future_target, _ = self.instance_norm(future_target, loc_scale)
        future_target = future_target.unsqueeze(1)
        future_target = future_target.to(self.device)
        future_target_mask = (
            future_target_mask.unsqueeze(1).to(self.device)
            if future_target_mask is not None
            else ~torch.isnan(future_target)
        )
        future_target = torch.where(future_target_mask > 0.0, future_target, 0.0)

        # pad target and target_mask if they are shorter than model's prediction
        if quantile_preds.shape[-1] > future_target.shape[-1]:
            padding_shape = (
                *future_target.shape[:-1],
                quantile_preds.shape[-1] - future_target.shape[-1],
            )
            future_target = torch.cat(
                [future_target, torch.zeros(padding_shape).to(future_target)], dim=-1
            )
            future_target_mask = torch.cat(
                [future_target_mask, torch.zeros(padding_shape).to(future_target_mask)],
                dim=-1,
            )

        quantiles = rearrange(self.quantiles, "num_quantiles -> 1 num_quantiles 1")
        quantile_loss = 2 * torch.abs(
            (future_target - quantile_preds)
            * ((future_target <= quantile_preds).float() - quantiles)
        )
        inv_future_covariate_mask = 1 - rearrange(
            patched_future_covariates_mask,
            "b n p -> b 1 (n p)",
            b=batch_size,
            n=num_output_patches,
            p=output_patch_size,
        )
        # the first components masks any missing targets and the second component masks known future values
        loss_mask = future_target_mask.float() * inv_future_covariate_mask
        loss = quantile_loss * loss_mask
        # mean over prediction horizon, sum over quantile levels and mean over batch
        loss = loss.mean(dim=-1).sum(dim=-1).mean()

        return loss

    def forward(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
        future_covariates: torch.Tensor | None = None,
        future_covariates_mask: torch.Tensor | None = None,
        num_output_patches: int = 1,
        future_target: torch.Tensor | None = None,
        future_target_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> Chronos2Output:
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
        future_target
            Target tensor of shape (batch_size, future_length) used during training. If ``future_covariates`` are
            provided, both target and future_covariates must have the same shape.
        future_target_mask
            Binary mask tensor of same shape as `future_target` indicating which values are valid (1) vs missing (0)
            If missing, the `future_target_mask` will be automatically constructed based on the NaN values in
            `future_target`.
        output_attentions
            Whether to return attention weights, by default False

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
                (batch_size, 1), self.config.reg_token_id, device=input_embeds.device
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

        encoder_outputs: Chronos2EncoderOutput = self.encoder(
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            group_ids=group_ids,
            output_attentions=output_attentions,
        )
        hidden_states: torch.Tensor = encoder_outputs[0]

        assert hidden_states.shape == (
            batch_size,
            num_context_patches + 1 + num_output_patches,
            self.d_model,
        )

        # slice the last num_output_patches hidden states to be input into the output_patch_embedding
        forecast_embeds = hidden_states[:, -num_output_patches:]
        quantile_preds: torch.Tensor = self.output_patch_embedding(forecast_embeds)
        quantile_preds = rearrange(
            quantile_preds,
            "b n (q p) -> b q (n p)",
            n=num_output_patches,
            q=self.num_quantiles,
            p=self.chronos_config.output_patch_size,
        )

        loss = (
            self._compute_loss(
                quantile_preds=quantile_preds,
                future_target=future_target,
                future_target_mask=future_target_mask,
                patched_future_covariates_mask=patched_future_covariates_mask,
                loc_scale=loc_scale,
                num_output_patches=num_output_patches,
            )
            if future_target is not None
            else None
        )

        # Unscale predictions
        quantile_preds = rearrange(
            quantile_preds,
            "b q h -> b (q h)",
            b=batch_size,
            q=self.num_quantiles,
            h=num_output_patches * self.chronos_config.output_patch_size,
        )
        quantile_preds = self.instance_norm.inverse(quantile_preds, loc_scale)
        quantile_preds = rearrange(
            quantile_preds,
            "b (q h) -> b q h",
            q=self.num_quantiles,
            h=num_output_patches * self.chronos_config.output_patch_size,
        )

        return Chronos2Output(
            loss=loss,
            quantile_preds=quantile_preds,
            enc_time_self_attn_weights=encoder_outputs.all_time_self_attn_weights,
            enc_group_self_attn_weights=encoder_outputs.all_group_self_attn_weights,
        )


class Chronos2Model(FoundationModel, HuggingFaceModelMixin):
    _repo_id = "amazon/chronos-2"
    _repo_commit = "18128c7b4f3fd286f06d6d4efe1d252f1d2a9a7c"

    def __init__(
        self,
        local_dir: Optional[Union[str, os.PathLike]] = None,
        **kwargs,
    ):
        super().__init__(**self._extract_torch_model_params(**self.model_params))

        self.local_dir = local_dir

        # extract pytorch lightning module kwargs
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        module = self._load_model(
            _Chronos2Module,
        )
        return module
