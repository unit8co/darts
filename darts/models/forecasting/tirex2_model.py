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
from typing import Any

import torch
from tirex2 import TimeseriesType, load_model

from darts.logging import raise_log
from darts.models.forecasting.foundation_model import FoundationModel
from darts.models.forecasting.pl_forecasting_module import (
    PLForecastingModule,
    io_processor,
)
from darts.utils.data.torch_datasets.utils import (
    InputChunkLength,
    PLModuleInput,
    TorchTrainingSample,
)
from darts.utils.likelihood_models.torch import QuantileRegression


class _TiRex2Module(PLForecastingModule):
    """Adapt TiRex-2's native multivariate inputs and quantiles to Darts."""

    def __init__(self, tirex2_kwargs: dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        # ForecastModel is not an nn.Module. Register its backbone directly so
        # Lightning can move, freeze, and serialize all pretrained parameters.
        self.tirex2 = load_model(**tirex2_kwargs).model
        self.future_len = self.output_chunk_length + self.output_chunk_shift
        if self.future_len > self.tirex2.future_len:
            raise_log(
                ValueError(
                    "`output_chunk_length` plus `output_chunk_shift` cannot exceed "
                    f"the checkpoint's maximum prediction length {self.tirex2.future_len}."
                )
            )
        all_quantiles = [round(float(q), 6) for q in self.tirex2.quantiles]
        user_quantiles = self.likelihood.quantiles if self.likelihood else [0.5]
        if not set(user_quantiles).issubset(all_quantiles):
            raise_log(
                ValueError("The checkpoint does not support the requested quantiles.")
            )
        self.register_buffer(
            "_user_quantile_indices",
            torch.tensor([all_quantiles.index(q) for q in user_quantiles]),
        )

    @io_processor
    def forward(self, x_in: PLModuleInput, *args, **kwargs):
        x_past, x_future, _, _ = x_in
        # Darts concatenates target, past covariates, then historic future covariates.
        n_future = x_future.shape[-1] if x_future is not None else 0
        past_end = x_past.shape[-1] - n_future
        future = None
        if n_future:
            # Darts does not supply future covariates inside the output shift.
            # Mark that gap as missing, preserving alignment with target history.
            gap = x_past.new_full(
                (len(x_past), self.output_chunk_shift, n_future), float("nan")
            )
            future = torch.cat((x_past[:, :, past_end:], gap, x_future), dim=1)
        timeseries = [
            TimeseriesType(
                target=past[:, : self.n_targets].T,
                past_covariates=(
                    past[:, self.n_targets : past_end].T
                    if past_end > self.n_targets
                    else None
                ),
                future_covariates=future[i].T if future is not None else None,
            )
            for i, past in enumerate(x_past)
        ]
        # Each sample remains a joint multivariate task, independent of other
        # samples in the batch. Native output: list[(targets, quantiles, horizon)].
        forecasts = self.tirex2.predict(timeseries, prediction_length=self.future_len)
        output = torch.stack(forecasts).permute(0, 3, 1, 2)
        return (
            output[:, self.output_chunk_shift :]
            .index_select(-1, self._user_quantile_indices)
            .to(x_past)
        )


class TiRex2Model(FoundationModel):
    _DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

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
        **kwargs,
    ):
        """TiRex-2 foundation model for zero-shot time series forecasting.

        Wraps the pretrained recurrent model from Podest et al. (2026) [1]_ using
        the optional `tirex-2 <https://github.com/NX-AI/tirex-2>`_ package
        (Python 3.11 or newer). Install it alongside ``darts[torch]``.
        TiRex-2 jointly forecasts all target components and supports past and
        future covariates. The default forecast is the median; pass
        ``QuantileRegression`` to obtain quantiles or probabilistic samples.

        Calling ``fit()`` loads the checkpoint and records the series and
        covariates without training. Fine-tuning is not supported by this wrapper.
        The upstream model and weights are distributed under Apache-2.0.

        Parameters
        ----------
        input_chunk_length
            Number of past time steps used for targets and covariates. A
            ``(min_length, max_length)`` tuple enables variable-length context.
        output_chunk_length
            Number of time steps forecast per model call. Together with
            ``output_chunk_shift``, must not exceed the checkpoint's maximum
            horizon (validated when loading the model). ``predict(n)`` can use
            autoregression when ``n > output_chunk_length`` and the shift is zero.
        output_chunk_shift
            Number of steps between the input and forecast. Future covariates
            within this gap are treated as missing. Autoregression is unavailable
            when this is nonzero.
        likelihood
            ``None`` for deterministic median forecasts, or ``QuantileRegression``
            with a subset of the pretrained quantiles 0.1, 0.2, ..., 0.9.
            Use ``predict_likelihood_parameters=True`` to return the quantiles
            directly, or ``num_samples > 1`` to sample from their distribution.
        hub_model_name
            Hugging Face model ID, or a local checkpoint directory containing
            ``model-config.yaml`` and ``model.ckpt``.
        hub_model_revision
            Optional Hugging Face branch, tag, or commit hash.
        local_dir
            Optional download directory forwarded to Hugging Face. To load an
            existing checkpoint without downloading, pass its directory as
            ``hub_model_name`` instead.
        tirex2_kwargs
            Additional arguments to ``tirex2.load_model()``, such as
            ``use_flex_attention`` and ``hf_kwargs``. The loading ``device``
            defaults to ``"cpu"``, selecting portable native kernels. For fused
            CUDA kernels, set ``device="cuda"`` here and configure the Lightning
            trainer for GPU execution. CUDA kernels require a compatible CUDA
            toolkit. Lightning controls the execution device through
            ``pl_trainer_kwargs``.
        **kwargs
            Additional :class:`FoundationModel` and :class:`TorchForecastingModel`
            options, including ``batch_size``, ``add_encoders``, ``random_state``,
            and ``pl_trainer_kwargs``. ``enable_finetuning`` must be false or None.

        Examples
        --------
        >>> from darts.models import TiRex2Model
        >>> from darts.utils.timeseries_generation import sine_timeseries
        >>> series = sine_timeseries(length=128).astype("float32")
        >>> model = TiRex2Model(64, 16)
        >>> model.fit(series)
        >>> forecast = model.predict(16)

        References
        ----------
        .. [1] Podest et al. "TiRex-2: Generalizing TiRex to Multivariate Data and
           Streaming", 2026. https://arxiv.org/abs/2607.01204
        """
        if kwargs.get("enable_finetuning"):
            raise_log(ValueError("TiRex2Model does not support fine-tuning."))
        if likelihood is not None:
            if not isinstance(likelihood, QuantileRegression):
                raise_log(
                    ValueError(
                        "Only QuantileRegression likelihood is supported for TiRex-2."
                    )
                )
            if not set(likelihood.quantiles).issubset(self._DEFAULT_QUANTILES):
                raise_log(
                    ValueError(
                        "The quantiles must be a subset of TiRex-2 quantiles "
                        f"{self._DEFAULT_QUANTILES}."
                    )
                )
        load_kwargs = dict(tirex2_kwargs or {})
        if "ckpt_path" in load_kwargs:
            raise_log(
                ValueError(
                    "Pass `ckpt_path` via `hub_model_name`, not `tirex2_kwargs`."
                )
            )
        hf_kwargs = dict(load_kwargs.pop("hf_kwargs", {}) or {})
        if hub_model_revision is not None:
            hf_kwargs["revision"] = hub_model_revision
        if local_dir is not None:
            hf_kwargs["local_dir"] = local_dir
        self.tirex2_kwargs = {
            "ckpt_path": hub_model_name,
            "device": "cpu",
            **load_kwargs,
        }
        if hf_kwargs:
            self.tirex2_kwargs["hf_kwargs"] = hf_kwargs
        super().__init__(**kwargs)

    def _create_model(self, train_sample: TorchTrainingSample) -> PLForecastingModule:
        return _TiRex2Module(
            tirex2_kwargs=self.tirex2_kwargs,
            **self.pl_module_params,
        )
