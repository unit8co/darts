"""
TiRex: Zero-Shot Forecasting across Long and Short Horizons
-----------------------------------------------------------

Darts wrapper for the pre-trained forecasting model TiRex introduced in [1].
The implementation is built around `tirex-ts <https://pypi.org/project/tirex-ts/>`.
The TiRex base repo <https://github.com/NX-AI/tirex>,
model card <https://huggingface.co/NX-AI/TiRex> and
docs <https://nx-ai.github.io/tirex/> provide more details.

Note: TiRex is released under the NXAI Community License. See
<https://github.com/NX-AI/tirex-internal/blob/main/LICENSE> for details.
Users must explicitly acknowledge the license by passing `accept_license=True` when
constructing `TiRexModel`.

References
----------
.. [1] https://arxiv.org/abs/2505.23719
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from darts.logging import get_logger, raise_log
from darts.models.forecasting.foundation_model import FoundationModel
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule
from darts.utils.likelihood_models.torch import QuantileRegression

logger = get_logger(__name__)


def _require_tirex():
    """Import and return the TiRex loader from the optional `tirex-ts` dependency."""
    try:
        # `tirex-ts` exposes a `load_model` entry point.
        from tirex import load_model  # type: ignore

    except Exception as e:  # pragma: no cover
        raise_log(
            ImportError(
                "Optional dependency `tirex-ts` is required to use TiRexModel. "
                "Install it with `pip install tirex-ts` (it provides the `tirex` Python package; "
                "extras include `tirex-ts[hfdataset]`, `tirex-ts[gluonts]`)."
            ),
            logger,
        )
        raise e

    return load_model


@dataclass(frozen=True)
class _TiRexQuantiles:
    # TiRex returns 9 quantiles by default (0.1..0.9).
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class _TiRexModule(PLForecastingModule):
    def __init__(
        self,
        tirex_pipeline,
        all_quantiles: tuple[float, ...],
        **kwargs,
    ):
        # `kwargs` must include PLForecastingModule args (incl. output_chunk_length,
        # output_chunk_shift, likelihood, etc.)
        super().__init__(**kwargs)

        self.tirex_pipeline = tirex_pipeline
        self.register_buffer(
            "_all_quantiles", torch.tensor(all_quantiles, dtype=torch.float32)
        )

        if self.likelihood is None:
            median_idx = (self._all_quantiles == 0.5).nonzero(as_tuple=True)[0]
            if len(median_idx) != 1:
                raise ValueError(
                    "Expected exactly one median quantile (0.5) in TiRex quantiles."
                )
            self.register_buffer(
                "_user_quantile_indices", median_idx.to(dtype=torch.long)
            )
        else:
            user_q = torch.tensor(self.likelihood.quantiles, dtype=torch.float32)
            indices: list[int] = []
            for q in user_q.tolist():
                matches = (
                    self._all_quantiles == torch.tensor(q, dtype=torch.float32)
                ).nonzero(as_tuple=True)[0]
                if len(matches) != 1:
                    raise ValueError(
                        f"Requested quantile {q} is not available in TiRex quantiles {all_quantiles}."
                    )
                indices.append(int(matches.item()))
            self.register_buffer(
                "_user_quantile_indices", torch.tensor(indices, dtype=torch.long)
            )

    def forward(self, x_in, *args, **kwargs):
        # PLModuleInput is typically a tuple: (x_past, x_future, x_static)
        x_past, x_future, _ = x_in

        # TiRex initial integration: no covariates
        if x_future is not None:
            # some datasets may provide an empty tensor; tolerate that
            if not (torch.is_tensor(x_future) and x_future.numel() == 0):
                raise ValueError("TiRexModel does not support future covariates.")

        # x_past: (B, T, C). Enforce univariate.
        if x_past.shape[-1] != 1:
            raise ValueError("TiRexModel currently supports univariate targets only.")

        context = x_past[..., 0]  # (B, T)

        # TiRex should forecast output_chunk_shift + output_chunk_length steps,
        # then we slice away the shift to match Darts' output_chunk_length.
        future_len = self.output_chunk_shift + self.output_chunk_length

        quantiles, mean = self.tirex_pipeline.forecast(
            context=context,
            prediction_length=future_len,
        )

        # Support both numpy arrays and torch tensors (tirex-ts supports both).
        if not torch.is_tensor(quantiles):
            quantiles = torch.as_tensor(quantiles)
        if not torch.is_tensor(mean):
            mean = torch.as_tensor(mean)

        quantiles = quantiles.to(device=context.device, dtype=torch.float32)
        mean = mean.to(device=context.device, dtype=torch.float32)

        # Expect TiRex outputs: quantiles (B, H, Q), mean (B, H)
        if quantiles.ndim != 3:
            raise ValueError(
                f"Unexpected TiRex quantiles shape: {tuple(quantiles.shape)}"
            )
        if mean.ndim != 2:
            raise ValueError(f"Unexpected TiRex mean shape: {tuple(mean.shape)}")

        # slice away output_chunk_shift
        quantiles = quantiles[:, self.output_chunk_shift : future_len, :]

        # select user-requested quantiles (or median)
        idx = self._user_quantile_indices
        if idx.device != quantiles.device:
            idx = idx.to(device=quantiles.device)
        idx = idx.to(dtype=torch.long).contiguous().clone()
        q_sel = quantiles.index_select(dim=-1, index=idx)

        # Darts expects output shape: (B, H, n_targets, n_quantiles)
        return q_sel.unsqueeze(2)


class TiRexModel(FoundationModel):
    """TiRex zero-shot forecasting model wrapper for Darts.

    This integration follows Darts' `FoundationModel` interface, and delegates
    forecasting logic and weight management to the optional `tirex-ts` package.

    Constraints (initial integration):
    - univariate target series only
    - no covariates
    - probabilistic forecasts via quantile outputs (supports sampling through Darts)

    Users must explicitly acknowledge the NXAI Community License by passing
    `accept_license=True`.
    """

    # TiRex quantiles returned by default (0.1..0.9)
    _DEFAULT_QUANTILES = _TiRexQuantiles().quantiles

    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        model_name: str = "NX-AI/TiRex",
        output_chunk_shift: int = 0,
        likelihood: QuantileRegression | None = None,
        accept_license: bool = False,
        device: str | None = None,
        backend: str | None = None,
        compile: bool | None = None,
        add_encoders: dict | None = None,
        **tirex_kwargs,
    ):
        if not accept_license:
            raise_log(
                ValueError(
                    "TiRex is distributed under the NXAI Community License. "
                    "Set `accept_license=True` to confirm you have reviewed and accept the terms: "
                    "https://github.com/NX-AI/tirex-internal/blob/main/LICENSE"
                ),
                logger,
            )

        if likelihood is not None and not isinstance(likelihood, QuantileRegression):
            raise_log(
                ValueError(
                    "Only QuantileRegression likelihood is supported for TiRexModel."
                ),
                logger,
            )

        # validate that requested quantiles are a subset of TiRex quantiles
        if likelihood is not None:
            req = tuple(float(q) for q in likelihood.quantiles)
            if not set(req).issubset(set(self._DEFAULT_QUANTILES)):
                raise_log(
                    ValueError(
                        "Requested quantiles must be a subset of TiRex quantiles "
                        f"{self._DEFAULT_QUANTILES}."
                    ),
                    logger,
                )

        super().__init__(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
            likelihood=likelihood,
            add_encoders=add_encoders,
        )

        self.model_name = model_name
        self.device = device
        self.backend = backend
        self.compile = compile
        self.tirex_kwargs = tirex_kwargs

    @property
    def supports_multivariate(self) -> bool:
        return False

    def _create_model(self, train_sample) -> PLForecastingModule:
        pl_module_params = self.pl_module_params or {}

        load_model = _require_tirex()

        kwargs = dict(self.tirex_kwargs)
        if self.device is not None:
            kwargs["device"] = self.device
        if self.backend is not None:
            kwargs["backend"] = self.backend
        if self.compile is not None:
            kwargs["compile"] = self.compile

        tirex_pipeline = load_model(self.model_name, **kwargs)

        return _TiRexModule(
            tirex_pipeline=tirex_pipeline,
            all_quantiles=self._DEFAULT_QUANTILES,
            **pl_module_params,
        )

    def fit(self, series, past_covariates=None, future_covariates=None, verbose=None):
        # enforce initial integration constraints early
        if past_covariates is not None or future_covariates is not None:
            raise_log(ValueError("TiRexModel does not support covariates."), logger)

        # univariate-only
        series_list = [series] if not isinstance(series, Sequence) else list(series)
        if any(s.n_components != 1 for s in series_list):
            raise_log(
                ValueError("TiRexModel currently supports univariate series only."),
                logger,
            )

        return super().fit(
            series=series,
            past_covariates=None,
            future_covariates=None,
            verbose=verbose,
        )
