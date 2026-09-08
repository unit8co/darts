"""
TimesFM 3.0
-----------

TimesFM 3.0 can be used the same way as other foundation models (e.g. Chronos2), with the
exception that it natively supports multivariate series as well as past and future covariates.

For detailed examples and tutorials, see:

* `Foundation Model Examples
  <https://unit8co.github.io/darts/examples/25-FoundationModel-examples.html>`__
* `Fine-Tuning Examples
  <https://unit8co.github.io/darts/examples/27-Torch-and-Foundation-Model-Fine-Tuning-examples.html>`__
"""

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from darts.logging import raise_log
from darts.models.components.huggingface_connector import HuggingFaceConnector
from darts.models.components.timesfm3_submodels import (
    _cpm_iterative_revin_refine,
    _get_output_patch_via_roll,
    _get_running_stats,
    _ResidualBlock,
    _ResidualBlockConfig,
    _revin,
    _StackedMixingTransformer,
    _StackedTransformersConfig,
    _stitch_patches,
    _TransformerConfig,
    build_residual_block_config,
    build_stacked_transformers_config,
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


class _TimesFM3Module(PLForecastingModule):
    """PyTorch module implementing the TimesFM 3.0 model, ported from
    `google-research/timesfm <https://github.com/google-research/timesfm/>`_ and
    adapted for Darts :class:`PLForecastingModule` interface.

    Multivariate inputs are supported natively: all target components form a single
    multivariate context processed with variate attention. Past covariates are passed as
    past-only channels, and future covariates as past-and-future channels of the model.

    The total number of channels (targets + past covariates + future covariates) must not
    exceed the ``max_variates`` of the checkpoint configuration (32 for the released
    checkpoint).
    """

    def __init__(
        self,
        # architecture parameters, extracted from the HuggingFace `config.json`
        input_patch_len: int = 32,
        output_patch_len: int = 64,
        quantiles: list[float] | None = None,
        residual_block_config: dict | _ResidualBlockConfig | None = None,
        transformer_config: dict | _StackedTransformersConfig | None = None,
        use_variate_attention: bool = True,
        value_clip: float = 1e20,
        use_stitching: bool = True,
        use_linear_detrending: bool = True,
        linear_detrending_threshold: float = 0.5,
        use_iterative_cpm_revin: bool = True,
        use_frozen_running_stats: bool = False,
        input_transform: str = "identity",
        **kwargs,
    ):
        # fine-tuning is not supported yet: the PL module would require a
        # gradient-enabled (non `@torch.no_grad()`) forecast path
        enable_finetuning = kwargs.pop("enable_finetuning", False)
        if enable_finetuning:
            raise_log(
                ValueError(
                    "Fine-tuning is not yet supported for TimesFM 3.0 in Darts."
                ),
            )
        super().__init__(**kwargs)

        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if residual_block_config is None:
            residual_block_config = _ResidualBlockConfig(
                hidden_dims=1280,
                output_dims=1280,
                use_bias=False,
                activation="relu",
            )
        else:
            residual_block_config = build_residual_block_config(residual_block_config)
        if transformer_config is None:
            transformer_config = _StackedTransformersConfig(
                num_layers=20,
                transformer=_TransformerConfig(
                    model_dims=1280,
                    hidden_dims=1280,
                    num_heads=16,
                    attention_norm="rms",
                    feedforward_norm="rms",
                    qk_norm="rms",
                    use_rope_seq=True,
                    use_rope_var=False,
                    use_bias=False,
                    ff_activation="relu",
                    deterministic=True,
                ),
            )
        else:
            transformer_config = build_stacked_transformers_config(transformer_config)

        if output_patch_len % input_patch_len != 0:
            raise_log(
                ValueError(
                    f"Output patch len {output_patch_len} must be a multiple of"
                    f" input patch len {input_patch_len}."
                ),
            )
        if (
            residual_block_config.output_dims
            != transformer_config.transformer.model_dims
        ):
            raise_log(
                ValueError(
                    "ResidualBlock output_dims must match Transformer model_dims."
                ),
            )
        if use_stitching and output_patch_len <= input_patch_len:
            raise_log(
                ValueError("use_stitching requires output_patch_len > input_patch_len"),
            )
        if input_transform != "identity":
            raise_log(
                ValueError(
                    f"Only the 'identity' input transformation is supported. "
                    f"Got {input_transform}."
                ),
            )

        self.input_patch_len = input_patch_len
        self.output_patch_len = output_patch_len
        self.model_quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.rolls = output_patch_len // input_patch_len
        self.max_variates = transformer_config.transformer.max_variates
        self.value_clip = value_clip
        self.use_stitching = use_stitching
        self.use_linear_detrending = use_linear_detrending
        self.linear_detrending_threshold = linear_detrending_threshold
        self.use_iterative_cpm_revin = use_iterative_cpm_revin
        self.use_frozen_running_stats = use_frozen_running_stats
        if use_stitching:
            self._stitching_extract_len = min(2 * input_patch_len, output_patch_len)

        # define model submodules; attribute names match the HuggingFace checkpoint
        # weights so they can be loaded with `load_state_dict()` without remapping
        self.pre_transformer_resblock = _ResidualBlock(config=residual_block_config)
        self.pre_transformer_resblock.set_input_dims(
            2 * (input_patch_len + output_patch_len)
        )
        self.transformer_stack = _StackedMixingTransformer(
            config=transformer_config,
            use_variate_attention=use_variate_attention,
        )
        self.output_head = nn.Linear(
            transformer_config.transformer.model_dims,
            output_patch_len * self.num_quantiles,
            bias=True,
        )

        self.future_len = (self.output_chunk_length or 0) + self.output_chunk_shift

        # gather indices of user-specified quantiles (used at prediction time);
        # by default (`likelihood=None`), the model is deterministic (median, index 4)
        user_quantiles: list[float] = (
            self.likelihood.quantiles
            if isinstance(self.likelihood, QuantileRegression)
            else [0.5]
        )
        self.user_quantile_indices = [
            self.model_quantiles.index(q) for q in user_quantiles
        ]

    def _preprocess(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
        patch_is_target: torch.Tensor,
        freeze_after: int | None = None,
        patch_cpm_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """Applies preprocessing: RevIN, masking, future covariates, ResBlock.

        Parameters
        ----------
        values
            Patched values of shape ``(batch, variates, n_patches, patch_len)``.
        masks
            Boolean mask of the same shape, where True = masked.
        patch_is_target
            Boolean mask of shape ``(batch, variates, n_patches)``.
        freeze_after
            Optional patch index after which running stats freeze.
        patch_cpm_mask
            Boolean mask of shape ``(batch, n_patches)`` or ``None``. If given, target
            variates at True positions are additionally masked (used for horizon CPM
            masking).

        Returns
        -------
        tuple
            ``(resblock_input, resblock_output, patch_mask, (running_mean, running_std),
            running_n)``.
        """
        running_n, running_mean, running_std = _get_running_stats(values, masks)
        if freeze_after is not None:
            _, _, n, _ = values.shape
            if 0 <= freeze_after < n - 1:
                running_mean[:, :, freeze_after + 1 :] = running_mean[
                    :, :, freeze_after : freeze_after + 1
                ]
                running_std[:, :, freeze_after + 1 :] = running_std[
                    :, :, freeze_after : freeze_after + 1
                ]

        # Apply CPM mask: mask target variates at CPM positions.
        if patch_cpm_mask is not None:
            cpm_bvnp = patch_cpm_mask[:, None, :, None]  # (b, 1, n, 1)
            cpm_target_only = cpm_bvnp & patch_is_target.unsqueeze(-1)
            masks = masks | cpm_target_only

        values_bvnp = _revin(values, running_mean, running_std, reverse=False)
        values_bvnp = torch.where(masks, 0.0, values_bvnp)

        # Roll values to get future covariate patches
        values_fcov, wrap_mask = _get_output_patch_via_roll(values, self.rolls)
        values_fcov = _revin(values_fcov, running_mean, running_std, reverse=False)

        # Roll the (CPM-modified) masks for future covariate masking.
        masks_fcov_raw, _ = _get_output_patch_via_roll(masks, self.rolls)
        masks_fcov = masks_fcov_raw | patch_is_target.unsqueeze(-1) | wrap_mask
        values_fcov = torch.where(masks_fcov, 0.0, values_fcov)

        values_cat = torch.cat([values_bvnp, values_fcov], dim=-1)
        masks_cat = torch.cat([masks, masks_fcov], dim=-1)

        resblock_input = torch.cat([values_cat, masks_cat.float()], dim=-1)
        resblock_output = self.pre_transformer_resblock(resblock_input)

        # Patch mask: a patch is fully masked if ALL points are masked
        patch_mask_bvn = masks_cat.all(dim=3)

        return (
            resblock_input,
            resblock_output,
            patch_mask_bvn,
            (running_mean, running_std),
            running_n,
        )

    def _forward_core(
        self,
        inputs: dict[str, torch.Tensor],
        freeze_after: int | None = None,
        patch_cpm_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full-sequence forward pass (equivalent to the original ``forward()``).

        Parameters
        ----------
        inputs
            Dictionary with keys ``"values"``: (batch, variates, n_patches, patch_len),
            ``"masks"``: same shape booleans, and ``"patch_is_target"``:
            (batch, variates, n_patches) booleans.
        freeze_after
            Optional patch index after which running stats freeze.
        patch_cpm_mask
            Boolean mask of shape ``(batch, n_patches)`` or ``None``. Horizon CPM mask.

        Returns
        -------
            Logits of shape (batch, variates, n_patches, output_patch_len,
            num_quantiles).
        """
        values = inputs["values"]
        values = torch.nan_to_num(values, nan=0.0)
        values = torch.clamp(values, -self.value_clip, self.value_clip)
        masks = inputs["masks"].bool()
        patch_is_target = inputs["patch_is_target"]

        _, _, _, p = values.shape
        if p != self.input_patch_len:
            raise_log(
                ValueError(
                    f"Input patch_len {p} != model input_patch_len {self.input_patch_len}"
                ),
            )

        # Preprocessing & ResBlock
        (
            _,
            transformer_input,
            transformer_patch_mask,
            revin_stats,
            running_n,
        ) = self._preprocess(
            values,
            masks,
            patch_is_target,
            freeze_after=freeze_after,
            patch_cpm_mask=patch_cpm_mask,
        )

        # Transformer
        # Only mask *leading* fully-masked patches (left-padding). This keeps horizon
        # patches (which are fully masked but come after valid context) visible to
        # attention.
        effective_patch_mask = torch.cumprod(transformer_patch_mask.int(), dim=2).bool()
        transformer_output = self.transformer_stack(
            transformer_input,
            effective_patch_mask,
        )

        # Output head
        raw_logits = self.output_head(transformer_output)
        revin_mean, revin_std = revin_stats

        if self.use_iterative_cpm_revin and patch_cpm_mask is not None:
            refined_mu, refined_sigma = _cpm_iterative_revin_refine(
                raw_logits,
                revin_n=running_n,
                revin_mu=revin_mean,
                revin_sigma=revin_std,
                patch_cpm_mask=patch_cpm_mask,
                median_q_idx=self.num_quantiles // 2,
                rolls=self.rolls,
                patch_len=self.input_patch_len,
                num_quantiles=self.num_quantiles,
                value_clip=self.value_clip,
            )
            cpm_bvn = patch_cpm_mask.unsqueeze(1)  # (b, 1, n)
            revin_mean = torch.where(cpm_bvn, refined_mu, revin_mean)
            revin_std = torch.where(cpm_bvn, refined_sigma, revin_std)

        revin_logits = _revin(raw_logits, revin_mean, revin_std, reverse=True)
        clipped_logits = torch.clamp(revin_logits, -self.value_clip, self.value_clip)

        # Reshape: (b, v, n, o*q) -> (b, v, n, o, q)
        b, v, n_patches = clipped_logits.shape[:3]
        return clipped_logits.view(
            b, v, n_patches, self.output_patch_len, self.num_quantiles
        )

    @torch.no_grad()
    def _decode(
        self,
        target: torch.Tensor,
        horizon: int = 0,
        past_only_covariates: torch.Tensor | None = None,
        past_future_covariates: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        past_only_mask: torch.Tensor | None = None,
        past_future_mask: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Non-autoregressive single-pass decoding for TimesFM 3.0.

        Parameters
        ----------
        target
            Shape ``(batch, n_targets, context_len)``.
        horizon
            Forecast horizon. Inferred if ``past_future_covariates`` given.
        past_only_covariates
            Shape ``(batch, n_past_only, context_len)`` or ``None``.
        past_future_covariates
            Shape ``(batch, n_past_future, context_len + horizon)`` or ``None``.
        target_mask
            Shape ``(batch, n_targets, context_len)`` booleans or ``None``.
        past_only_mask
            Shape ``(batch, n_past_only, context_len)`` booleans or ``None``.
        past_future_mask
            Shape ``(batch, n_past_future, context_len + horizon)`` booleans or ``None``.
        mask
            Shape ``(batch, context_len)`` global boolean mask or ``None``.

        Returns
        -------
            Logits of shape ``(batch, num_variates, horizon, num_quantiles)``.
        """
        device = target.device
        batch_size, num_target, context = target.shape

        if past_future_covariates is not None:
            horizon = past_future_covariates.shape[-1] - context
        if horizon <= 0:
            raise_log(ValueError("Decode function requires horizon > 0."))

        # 1. Pad context to multiple of input_patch_len
        ctx_padding = (
            self.input_patch_len - (context % self.input_patch_len)
        ) % self.input_patch_len
        if ctx_padding > 0:
            target = F.pad(target, (ctx_padding, 0))
            if mask is not None:
                mask = F.pad(mask, (ctx_padding, 0), value=True)
            if past_only_covariates is not None:
                past_only_covariates = F.pad(past_only_covariates, (ctx_padding, 0))
            if past_future_covariates is not None:
                past_future_covariates = F.pad(past_future_covariates, (ctx_padding, 0))
            if target_mask is not None:
                target_mask = F.pad(target_mask, (ctx_padding, 0), value=True)
            if past_only_mask is not None:
                past_only_mask = F.pad(past_only_mask, (ctx_padding, 0), value=True)
            if past_future_mask is not None:
                past_future_mask = F.pad(past_future_mask, (ctx_padding, 0), value=True)
            context = context + ctx_padding

        if mask is None:
            mask = torch.zeros(batch_size, context, dtype=torch.bool, device=device)
            if ctx_padding > 0:
                mask[:, :ctx_padding] = True

        # 2. Pad horizon
        if self.use_stitching:
            extract_len = self._stitching_extract_len
            overlap = extract_len - self.input_patch_len
            num_forecast_patches = max(
                math.ceil((horizon - overlap) / self.input_patch_len), 1
            )
            num_horizon_patches = num_forecast_patches + self.rolls - 1
            padded_horizon = num_horizon_patches * self.input_patch_len
            hor_padding = padded_horizon - horizon
        else:
            hor_padding = (-horizon) % self.output_patch_len
            padded_horizon = horizon + hor_padding
            num_horizon_patches = padded_horizon // self.input_patch_len
        num_context_patches = context // self.input_patch_len

        # 3. Build context & horizon inputs
        if target_mask is None:
            target_mask = torch.zeros_like(target, dtype=torch.bool)
        target_mask = target_mask | mask.unsqueeze(1)

        all_ctx_vals = [target]
        all_ctx_masks = [target_mask]
        num_past_only = 0
        if past_only_covariates is not None:
            num_past_only = past_only_covariates.shape[1]
            if past_only_mask is None:
                past_only_mask = torch.zeros_like(
                    past_only_covariates, dtype=torch.bool
                )
            all_ctx_vals.append(past_only_covariates)
            all_ctx_masks.append(past_only_mask | mask.unsqueeze(1))
        if past_future_covariates is not None:
            if past_future_mask is None:
                past_future_mask = torch.zeros_like(
                    past_future_covariates, dtype=torch.bool
                )
            all_ctx_vals.append(past_future_covariates[..., :context])
            all_ctx_masks.append(past_future_mask[..., :context] | mask.unsqueeze(1))

        ctx_vals = torch.cat(all_ctx_vals, dim=1)
        ctx_masks = torch.cat(all_ctx_masks, dim=1)

        if self.use_linear_detrending:
            t_ctx = torch.arange(-(context - 1), 1, dtype=torch.float32, device=device)
            t_ctx_bvc = t_ctx[None, None, :]
            t_ctx_bvc_normalized = t_ctx_bvc / context

            valid = ~ctx_masks
            n_v = valid.float().sum(dim=-1, keepdim=True)
            sum_t = torch.where(valid, t_ctx_bvc_normalized, 0.0).sum(
                dim=-1, keepdim=True
            )
            sum_t2 = torch.where(valid, t_ctx_bvc_normalized**2, 0.0).sum(
                dim=-1, keepdim=True
            )
            sum_y = torch.where(valid, ctx_vals, 0.0).sum(dim=-1, keepdim=True)
            sum_ty = torch.where(valid, t_ctx_bvc_normalized * ctx_vals, 0.0).sum(
                dim=-1, keepdim=True
            )

            det = n_v * sum_t2 - sum_t**2
            safe_det = torch.where(det == 0.0, 1.0, det)
            m_trend = torch.where(
                det == 0.0, 0.0, (n_v * sum_ty - sum_t * sum_y) / safe_det
            )
            c_trend = torch.where(
                det == 0.0,
                torch.where(n_v > 0, sum_y / torch.clamp_min(n_v, 1.0), 0.0),
                (sum_y - m_trend * sum_t) / torch.clamp_min(n_v, 1.0),
            )

            ctx_vals_detrended = ctx_vals - (m_trend * t_ctx_bvc_normalized + c_trend)

            mean_y = sum_y / torch.clamp_min(n_v, 1.0)
            sum_y2 = torch.where(valid, ctx_vals**2, 0.0).sum(dim=-1, keepdim=True)
            var_orig = torch.clamp_min(
                sum_y2 / torch.clamp_min(n_v, 1.0) - mean_y**2, 0.0
            )
            std_orig = torch.sqrt(var_orig)

            sum_yd = torch.where(valid, ctx_vals_detrended, 0.0).sum(
                dim=-1, keepdim=True
            )
            mean_yd = sum_yd / torch.clamp_min(n_v, 1.0)
            sum_yd2 = torch.where(valid, ctx_vals_detrended**2, 0.0).sum(
                dim=-1, keepdim=True
            )
            var_det = torch.clamp_min(
                sum_yd2 / torch.clamp_min(n_v, 1.0) - mean_yd**2, 0.0
            )
            std_det = torch.sqrt(var_det)

            apply_detrend = std_det < self.linear_detrending_threshold * std_orig
            ctx_vals = torch.where(apply_detrend, ctx_vals_detrended, ctx_vals)
        else:
            num_variates = ctx_vals.shape[1]
            m_trend = torch.zeros(
                (batch_size, num_variates, 1), dtype=torch.float32, device=device
            )
            c_trend = torch.zeros(
                (batch_size, num_variates, 1), dtype=torch.float32, device=device
            )
            apply_detrend = torch.zeros(
                (batch_size, num_variates, 1), dtype=torch.bool, device=device
            )

        ctx_vals = torch.where(ctx_masks, 0.0, ctx_vals)

        all_hor_vals = [
            torch.zeros(batch_size, num_target, padded_horizon, device=device),
            torch.zeros(batch_size, num_past_only, padded_horizon, device=device),
        ]
        all_hor_masks = [
            torch.ones(
                batch_size,
                num_target,
                padded_horizon,
                dtype=torch.bool,
                device=device,
            ),
            torch.ones(
                batch_size,
                num_past_only,
                padded_horizon,
                dtype=torch.bool,
                device=device,
            ),
        ]

        if past_future_covariates is not None:
            if past_future_mask is None:
                past_future_mask = torch.zeros_like(
                    past_future_covariates, dtype=torch.bool
                )
            pf_future_vals = past_future_covariates[..., context : context + horizon]
            pf_future_masks = past_future_mask[..., context : context + horizon]
            if self.use_linear_detrending:
                m_pf = m_trend[:, num_target + num_past_only :, :]
                c_pf = c_trend[:, num_target + num_past_only :, :]
                apply_detrend_pf = apply_detrend[:, num_target + num_past_only :, :]
                t_hor_pf = torch.arange(
                    1, horizon + 1, dtype=torch.float32, device=device
                )[None, None, :]
                t_hor_pf_normalized = t_hor_pf / context
                pf_trend_hor = m_pf * t_hor_pf_normalized + c_pf
                pf_future_vals = torch.where(
                    apply_detrend_pf, pf_future_vals - pf_trend_hor, pf_future_vals
                )
            pf_future_vals = torch.where(pf_future_masks, 0.0, pf_future_vals)
            if hor_padding > 0:
                pf_future_vals = F.pad(pf_future_vals, (0, hor_padding))
                pf_future_masks = F.pad(pf_future_masks, (0, hor_padding), value=True)
            all_hor_vals.append(pf_future_vals)
            all_hor_masks.append(pf_future_masks)

        hor_vals = torch.cat(all_hor_vals, dim=1)
        hor_masks = torch.cat(all_hor_masks, dim=1)

        all_vals = torch.cat([ctx_vals, hor_vals], dim=-1)
        all_masks = torch.cat([ctx_masks, hor_masks], dim=-1)

        num_variates = all_vals.shape[1]
        patch_is_target = torch.zeros(
            (batch_size, num_variates, num_context_patches + num_horizon_patches),
            dtype=torch.bool,
            device=device,
        )
        patch_is_target[:, : num_target + num_past_only, :] = True

        # Reshape values & masks to patched shape (b, v, n, p)
        values_bvnp = all_vals.reshape(
            batch_size, num_variates, -1, self.input_patch_len
        )
        masks_bvnp = all_masks.reshape(
            batch_size, num_variates, -1, self.input_patch_len
        )

        inputs = {
            "values": values_bvnp,
            "masks": masks_bvnp,
            "patch_is_target": patch_is_target,
        }

        # Build horizon CPM mask: context=False, horizon=True.
        num_total_patches = num_context_patches + num_horizon_patches
        horizon_cpm_mask = torch.zeros(
            batch_size, num_total_patches, dtype=torch.bool, device=device
        )
        horizon_cpm_mask[:, num_context_patches:] = True

        freeze_after = (
            num_context_patches - 1 if self.use_frozen_running_stats else None
        )
        logits = self._forward_core(
            inputs,
            freeze_after=freeze_after,
            patch_cpm_mask=horizon_cpm_mask,
        )  # (b, v, n, output_patch_len, num_quantiles)

        if self.use_stitching:
            extract_len = self._stitching_extract_len
            overlap = extract_len - self.input_patch_len
            num_forecast_patches = max(
                math.ceil((horizon - overlap) / self.input_patch_len), 1
            )
            forecast_indices = torch.arange(num_forecast_patches, device=device) + (
                num_context_patches - 1
            )
            patch_preds = logits[:, :, forecast_indices, :extract_len, :]
            horizon_logits = _stitch_patches(
                patch_preds,
                self.input_patch_len,
            )[:, :, :horizon, :]
        else:
            num_forecast_chunks = padded_horizon // self.output_patch_len
            forecast_indices = torch.arange(
                num_forecast_chunks, device=device
            ) * self.rolls + (num_context_patches - 1)
            forecast_logits = logits[:, :, forecast_indices, :, :]
            horizon_logits = forecast_logits.reshape(
                batch_size, num_variates, -1, self.num_quantiles
            )[:, :, :horizon, :]

        if self.use_linear_detrending:
            t_forecast = torch.arange(
                1, horizon + 1, dtype=torch.float32, device=device
            )
            t_forecast_normalized = t_forecast / context
            trend_forecast = (
                m_trend[:, :, 0, None] * t_forecast_normalized[None, None, :]
                + c_trend[:, :, 0, None]
            )
            trend_forecast = torch.where(
                apply_detrend[:, :, 0, None], trend_forecast, 0.0
            )
            horizon_logits = horizon_logits + trend_forecast[:, :, :, None]

        return horizon_logits

    @io_processor
    def forward(self, x_in: PLModuleInput, *args, **kwargs) -> torch.Tensor:
        """TimesFM 3.0 model forward pass.

        Parameters
        ----------
        x_in
            comes as a tuple `(x_past, x_future, x_static, future_target)` where `x_past` is the input/past chunk and
            `x_future` is the output/future chunk. Input dimensions are `(n_samples, n_time_steps, n_variables)`

        Returns
        -------
        torch.Tensor
            the output tensor in the shape `(n_samples, n_time_steps, n_targets, n_quantiles)` for
            probabilistic forecasts, or `(n_samples, n_time_steps, n_targets, 1)` for
            deterministic forecasts (median only).
        """
        # Dimension notation in comments below:
        #   B: batch size
        #   L: input chunk length
        #   T: output chunk length
        #   S: output chunk shift
        #   H = T + S: forecast horizon delegated to the model per chunk
        #   C: target components
        #   P: past covariates (past-only channels)
        #   W: future covariates (past-and-future channels)
        #   V = C + P + W: total variates (must be <= max_variates of the config)
        #   Q = 9: pre-trained quantiles returned by the model
        #   N: likelihood quantiles (user-specified, 1 if deterministic)

        # `x_past` is a stack of [past_target (C), past_covariates (P),
        # historic_future_covariates (W)], `x_future` is just future_covariates.
        x_past, x_future, _, _ = x_in
        batch_size, past_length, n_variables = x_past.shape
        n_targets = self.n_targets
        n_future_covs = x_future.shape[-1] if x_future is not None else 0
        n_past_covs = n_variables - n_targets - n_future_covs

        if n_variables > self.max_variates:
            raise_log(
                ValueError(
                    f"The total number of target components and covariates {n_variables} "
                    f"exceeds the maximum number of variates {self.max_variates} "
                    f"supported by the TimesFM 3.0 checkpoint."
                ),
            )

        # targets: (B, L, C) -> (B, C, L); missing values are handled with masks
        target = x_past[:, :, :n_targets].transpose(1, 2)
        target_mask = torch.isnan(target)
        target = torch.nan_to_num(target, nan=0.0)

        # past covariates: (B, L, P) -> (B, P, L)
        past_only_covariates = None
        past_only_mask = None
        if n_past_covs > 0:
            past_only_covariates = x_past[:, :, n_targets : n_targets + n_past_covs]
            past_only_covariates = past_only_covariates.transpose(1, 2)
            past_only_mask = torch.isnan(past_only_covariates)
            past_only_covariates = torch.nan_to_num(past_only_covariates, nan=0.0)

        # future covariates: (B, L + H, W): the historic part comes from `x_past`, the
        # future part from `x_future`. The values in the gap created by
        # `output_chunk_shift` are unknown and are masked out.
        past_future_covariates = None
        past_future_mask = None
        if n_future_covs > 0:
            historic_fc = x_past[:, :, n_targets + n_past_covs :].transpose(1, 2)
            future_fc = x_future.transpose(1, 2)
            if self.output_chunk_shift > 0:
                gap = torch.full(
                    (batch_size, n_future_covs, self.output_chunk_shift),
                    float("nan"),
                    device=x_past.device,
                    dtype=x_past.dtype,
                )
                past_future_covariates = torch.cat(
                    [historic_fc, gap, future_fc], dim=-1
                )
            else:
                past_future_covariates = torch.cat([historic_fc, future_fc], dim=-1)
            past_future_mask = torch.isnan(past_future_covariates)
            past_future_covariates = torch.nan_to_num(past_future_covariates, nan=0.0)

        # single-pass decode over the whole horizon H
        # -> (B, C + P + W, H, Q)
        horizon_logits = self._decode(
            target=target,
            horizon=self.future_len,
            past_only_covariates=past_only_covariates,
            past_future_covariates=past_future_covariates,
            target_mask=target_mask,
            past_only_mask=past_only_mask,
            past_future_mask=past_future_mask,
        )

        # slice the output shift: (B, V, S + T, Q) -> (B, V, T, Q)
        horizon_logits = horizon_logits[:, :, self.output_chunk_shift :, :]
        # select the target variates: (B, C, T, Q)
        horizon_logits = horizon_logits[:, :n_targets, :, :]
        # (B, C, T, Q) -> (B, T, C, Q)
        horizon_logits = horizon_logits.permute(0, 2, 1, 3)
        # select the user-specified quantiles: (B, T, C, N)
        return horizon_logits[..., self.user_quantile_indices]


class TimesFM3Model(FoundationModel):
    # Structural limits of the original implementation. These are constants of the
    # upstream `TimesFM3Forecaster` and are not part of the HuggingFace `config.json`
    # (which only carries the architecture parameters read by `_create_model()`).
    _MAX_CONTEXT_LENGTH = 15360
    # no upstream limit exists (stitching supports arbitrarily long horizons); this
    # is a conservative cap aligned with the 1024-step long head of TimesFM 2.5
    _MAX_PREDICTION_LENGTH = 1024

    def __init__(
        self,
        input_chunk_length: InputChunkLength,
        output_chunk_length: int,
        output_chunk_shift: int = 0,
        accept_license: bool = False,
        likelihood: QuantileRegression | None = None,
        hub_model_name: str = "google/timesfm-3.0-pytorch",
        hub_model_revision: str | None = "43046b85ec22d584a13f8098c2ed39c889e129c2",
        local_dir: str | os.PathLike | None = None,
        **kwargs,
    ):
        """
        TimesFM 3.0 foundation model for zero-shot time series forecasting.

        This is an implementation of Google's TimesFM 3.0 model, ported from
        `google-research/timesfm <https://github.com/google-research/timesfm>`_ with
        adaptations to use the Darts API.

        TimesFM 3.0 is a pre-trained foundation model designed for zero-shot forecasting
        across both short and long horizons. Unlike previous versions, it natively supports
        multivariate time series (with variate attention), past covariates, and future
        covariates.

        Using this model will automatically download and cache the pre-trained model from
        HuggingFace Hub
        (`google/timesfm-3.0-pytorch <https://huggingface.co/google/timesfm-3.0-pytorch>`_).
        Alternatively, you can specify a local directory containing the model config and
        weights using the ``local_dir`` parameter.

        By default, the model is deterministic (median forecast only). To enable
        probabilistic forecasts, pass a
        :class:`~darts.utils.likelihood_models.torch.QuantileRegression` instance to the
        ``likelihood`` parameter. It is recommended to call :func:`predict()` with
        ``predict_likelihood_parameters=True`` or ``num_samples >> 1`` to get meaningful
        results.

        .. note::
            The TimesFM 3.0 pre-trained weights are distributed under the
            `TimesFM Non-Commercial License v1.0
            <https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE>`_
            and are restricted to non-commercial, non-production use. You must explicitly
            acknowledge this license by passing ``accept_license=True`` when constructing
            the model. The model code is licensed under the Apache-2.0 License.
        .. note::
            Fine-tuning is not supported yet.
        .. note::
            The total number of target components and covariates must not exceed 32 (the
            maximum number of variates supported by the checkpoint).
        .. note::
            Zero-shot forecasts match the original implementation when the forecast
            horizon ``n`` is smaller than or equal to ``output_chunk_length``. For longer
            horizons, Darts auto-regressively applies the model on
            ``output_chunk_length``-sized chunks, which may produce slightly different
            results than the original single-pass decoding. Set
            ``output_chunk_length >= n`` to obtain a single-pass forecast.

        Parameters
        ----------
        input_chunk_length
            Number of time steps in the past to take as a model input (per chunk). Applies to the target
            series, and past and/or future covariates (if the model supports it).
            Can be either an ``int`` for a fixed input window, or a ``(min_length, max_length)`` tuple to enable
            variable-length inputs for inference and fine-tuning.
            For TimesFM 3.0, ``max_length`` must be less than or equal to 15,360.
        output_chunk_length
            Number of time steps predicted at once (per chunk) by the internal model. Also, the number of future values
            from future covariates to use as a model input (if the model supports future covariates). It is not the same
            as forecast horizon `n` used in `predict()`, which is the desired number of prediction points generated
            using either a one-shot- or autoregressive forecast. Setting `n <= output_chunk_length` prevents
            auto-regression. This is useful when the covariates don't extend far enough into the future, or to prohibit
            the model from using future values of past and / or future covariates for prediction (depending on the
            model's covariate support).
            For TimesFM 3.0, `output_chunk_length + output_chunk_shift` must be less than or equal to 1024.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future (relative to the
            input chunk end). This will create a gap between the input and output. If the model supports
            `future_covariates`, the future values are extracted from the shifted output chunk. Predictions will start
            `output_chunk_shift` steps after the end of the target `series`. If `output_chunk_shift` is set, the model
            cannot generate autoregressive predictions (`n > output_chunk_length`). The covariate values inside the
            gap are unknown to the model and are masked out.
        likelihood
            The likelihood model to be used for probabilistic forecasts. Must be ``None`` or an instance of
            :class:`~darts.utils.likelihood_models.torch.QuantileRegression`. If using ``QuantileRegression``,
            the quantiles must be a subset of those used during TimesFM 3.0 pre-training:
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
            Default: ``None``, which will make the model deterministic (median quantile only).
        accept_license
            Must be set to ``True`` to confirm acceptance of the TimesFM Non-Commercial
            License v1.0 for the pre-trained weights. Default: ``False``.
        hub_model_name
            The model ID on HuggingFace Hub. Default: ``"google/timesfm-3.0-pytorch"``.
        hub_model_revision
            The model version to use. This can be a branch name, tag name, or commit hash. Default is
            ``43046b85ec22d584a13f8098c2ed39c889e129c2``, which pins the current version of the
            ``google/timesfm-3.0-pytorch`` repository.
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
            Not supported for TimesFM 3.0 in Darts. Setting it to anything other than ``None``/``False``
            will raise an exception.

        References
        ----------
        .. [1] A. Das, W. Kong, R. Sen, Y. Zhou. "A decoder-only foundation model for time-series forecasting", 2025.
                arXiv https://arxiv.org/abs/2310.10688.
        .. [2] "TimesFM 3.0", 2026. Google Research.
                https://huggingface.co/google/timesfm-3.0-pytorch

        Examples
        --------
        Point forecasting:

        >>> from darts.models import TimesFM3Model
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load().astype("float32")
        >>> # you must explicitly set `accept_license=True` to use the model
        >>> model = TimesFM3Model(
        ...     input_chunk_length=12,
        ...     output_chunk_length=6,
        ...     accept_license=True,
        ... )
        >>> model.fit(series)
        >>> pred = model.predict(n=6)
        >>> pred
                    #Passengers
        Month
        1961-01-01   436.861908
        1961-02-01   440.754639
        1961-03-01   449.636871
        1961-04-01   458.311768
        1961-05-01   459.939941
        1961-06-01   466.235779

        Probabilistic forecasting:

        >>> from darts.utils.likelihood_models import QuantileRegression
        >>> # you must explicitly set `accept_license=True` to use the model
        >>> model = TimesFM3Model(
        ...     input_chunk_length=12,
        ...     output_chunk_length=6,
        ...     likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
        ...     accept_license=True,
        ... )
        >>> model.fit(series)
        >>> pred = model.predict(n=6, predict_likelihood_parameters=True)
        >>> pred
                    #Passengers_q0.100  #Passengers_q0.500  #Passengers_q0.900
        Month
        1961-01-01          380.856689          436.861908          517.565186
        1961-02-01          365.766022          440.754639          550.635132
        1961-03-01          361.614166          449.636871          581.230164
        1961-04-01          356.575531          458.311768          607.764648
        1961-05-01          347.237854          459.939941          618.507874
        1961-06-01          345.421265          466.235779          631.610840
        """
        if not accept_license:
            raise_log(
                ValueError(
                    "TimesFM 3.0 pre-trained weights are distributed under the TimesFM "
                    "Non-Commercial License v1.0 (non-commercial, non-production use). "
                    "Set `accept_license=True` to confirm you have reviewed and accept "
                    "the terms: "
                    "https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE"
                ),
            )

        # validate `input_chunk_length` against the model's maximum context length
        _, max_icl = _parse_input_chunk_length(input_chunk_length)
        if max_icl > self._MAX_CONTEXT_LENGTH:
            raise_log(
                ValueError(
                    f"`input_chunk_length` {max_icl} cannot be greater than model's "
                    f"maximum context length {self._MAX_CONTEXT_LENGTH}"
                ),
            )

        # validate `output_chunk_length` and `output_chunk_shift` against the model's
        # maximum prediction length
        if output_chunk_length + output_chunk_shift > self._MAX_PREDICTION_LENGTH:
            raise_log(
                ValueError(
                    f"`output_chunk_length` {output_chunk_length} plus `output_chunk_shift` "
                    f"{output_chunk_shift} cannot be greater than model's maximum prediction "
                    f"length {self._MAX_PREDICTION_LENGTH}"
                ),
            )

        # load the model configuration for validation, as done by `Chronos2Model`
        self.hf_connector = HuggingFaceConnector(
            model_name=hub_model_name,
            model_revision=hub_model_revision,
            local_dir=local_dir,
        )
        config = self.hf_connector.load_config()
        self._model_quantiles = tuple(config["quantiles"])
        self._max_variates = config["transformer_config"]["transformer"]["max_variates"]

        # by default (`likelihood=None`), model is deterministic; otherwise, only
        # QuantileRegression likelihood is supported and quantiles must be a subset of
        # the quantiles used during pre-training (read from the model configuration)
        if likelihood is not None:
            if not isinstance(likelihood, QuantileRegression):
                raise_log(
                    ValueError(
                        f"Only QuantileRegression likelihood is supported for TimesFM 3.0 "
                        f"in Darts. Got {type(likelihood)}."
                    ),
                )
            user_quantiles: list[float] = likelihood.quantiles
            if not set(user_quantiles).issubset(self._model_quantiles):
                raise_log(
                    ValueError(
                        f"The quantiles for QuantileRegression likelihood {user_quantiles} "
                        f"must be a subset of TimesFM 3.0 quantiles "
                        f"{self._model_quantiles}."
                    ),
                )

        # fine-tuning is not supported yet
        if kwargs.get("enable_finetuning", None):
            raise_log(
                ValueError(
                    "Fine-tuning is not yet supported for TimesFM 3.0 in Darts."
                ),
            )

        super().__init__(**kwargs)

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        # validate the number of variates early, at fit time; `_TimesFM3Module`
        # additionally validates at prediction time in `forward()`. The train sample
        # is `(past_target, past_cov, historic_future_cov, future_cov, static_cov,
        # future_target)`; `historic_future_cov` and `future_cov` stem from the same
        # future covariates series, so its width must only be counted once
        past_target, past_cov, _, future_cov = train_sample[:4]
        n_variates = past_target.shape[1]
        for variate in (past_cov, future_cov):
            if variate is not None:
                n_variates += variate.shape[1]
        if n_variates > self._max_variates:
            raise_log(
                ValueError(
                    f"The total number of target components and covariates {n_variates} "
                    f"exceeds the maximum number of variates {self._max_variates} "
                    f"supported by the TimesFM 3.0 checkpoint."
                ),
            )
        # the architecture parameters are extracted from the HuggingFace `config.json`
        pl_module_params = self.pl_module_params or {}
        return self.hf_connector.load_model(
            module_class=_TimesFM3Module,
            pl_module_params=pl_module_params,
        )
