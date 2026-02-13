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

References
----------
.. [1] https://arxiv.org/abs/2505.23719
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.utils.timeseries_generation import _build_forecast_series

logger = get_logger(__name__)


def _require_tirex():
    try:
        from tirex import load_model  # tirex-ts package provides `tirex` module
    except ImportError as e:
        raise ImportError(
            "TiRexModel requires the optional dependency 'tirex-ts'. "
            "Install it with `pip install tirex-ts`."
        ) from e
    return load_model


@dataclass(frozen=True)
class _QuantileConfig:
    # TiRex returns 9 quantiles by default (0.1..0.9) per docs.
    # Keep configurable in case TiRex changes or user wants different set later.
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class TiRexModel(GlobalForecastingModel):
    """
    TiRex zero-shot forecasting model wrapper for Darts.

    Constraints (initial integration):
    - univariate only
    - no covariates
    - probabilistic via sampling from TiRex quantile outputs
    """

    def __init__(
        self,
        model_name: str = "NX-AI/TiRex",
        device: str | None = None,
        backend: str | None = None,
        compile: bool | None = None,
        context_length: int | None = None,
        add_encoders: dict | None = None,
        quantile_config: _QuantileConfig = _QuantileConfig(),
        **tirex_kwargs,
    ):
        super().__init__(add_encoders=add_encoders)
        self.model_name = model_name
        self.device = device
        self.backend = backend
        self.compile = compile
        self.context_length = context_length
        self.quantile_config = quantile_config
        self.tirex_kwargs = tirex_kwargs

        self._model = None
        self._default_series: TimeSeries | None = None

    @property
    def supports_multivariate(self) -> bool:
        return False

    @property
    def supports_transferable_series_prediction(self) -> bool:
        return True

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return True

    @property
    def _model_encoder_settings(self):
        """Encoder settings required by Darts' base classes.

        TiRexModel does not use Darts encoders in the initial integration.
        """
        return None

    @property
    def _target_window_lengths(self) -> tuple[int, int]:
        """Return (min_input_length, min_output_length) for the target series.

        For zero-shot inference we only require a (potentially truncated) context
        window and can always produce at least 1-step forecasts.
        """
        in_len = int(self.context_length) if self.context_length is not None else 1
        return in_len, 1

    @property
    def extreme_lags(self):
        """Return extreme lags tuple expected by Darts.

        Format: (min_target_lag, max_target_lag, min_past_cov_lag, max_past_cov_lag,
        min_future_cov_lag, max_future_cov_lag, output_shift).
        """
        in_len = int(self.context_length) if self.context_length is not None else 1
        return (-in_len, -1, None, None, None, None, 0)

    @property
    def min_train_samples(self) -> int:
        return 1

    def fit(
        self,
        series: TimeSeries | Sequence[TimeSeries],
        past_covariates=None,
        future_covariates=None,
        verbose: bool | None = None,
    ) -> TiRexModel:
        if past_covariates is not None or future_covariates is not None:
            raise_log(
                ValueError("TiRexModel currently does not support covariates."), logger
            )

        # enforce univariate inputs for initial integration
        series_list = [series] if isinstance(series, TimeSeries) else list(series)
        if any(s.n_components != 1 for s in series_list):
            raise_log(
                ValueError("TiRexModel currently supports univariate series only."),
                logger,
            )

        # keep a reference series for building forecast time index when `predict()` is called without `series`
        self._default_series = series_list[0]

        super().fit(
            series=series, past_covariates=None, future_covariates=None, verbose=verbose
        )

        load_model = _require_tirex()
        # TiRex docs: model = load_model("NX-AI/TiRex", backend="cuda") etc.
        # Device/backends are described in TiRex docs.
        kwargs = dict(self.tirex_kwargs)
        if self.device is not None:
            kwargs["device"] = self.device
        if self.backend is not None:
            kwargs["backend"] = self.backend
        if self.compile is not None:
            kwargs["compile"] = self.compile

        self._model = load_model(self.model_name, **kwargs)
        return self

    def predict(
        self,
        n: int,
        series: TimeSeries | Sequence[TimeSeries] | None = None,
        past_covariates=None,
        future_covariates=None,
        num_samples: int = 1,
        verbose: bool | None = None,
        predict_likelihood_parameters: bool = False,
        **kwargs,
    ):
        if past_covariates is not None or future_covariates is not None:
            raise_log(ValueError("TiRexModel does not support covariates."), logger)
        if self._model is None:
            raise_log(ValueError("Call fit() before predict()."), logger)

        if series is None:
            # Global models can store training series in various internal forms; keep a
            # stable reference TimeSeries for index generation.
            if self._default_series is None:
                raise_log(
                    ValueError(
                        "No series provided and no default series available. Call fit() first."
                    ),
                    logger,
                )
            series = self._default_series

        if (
            isinstance(series, (list, tuple))
            and len(series) == 1
            and isinstance(series[0], TimeSeries)
        ):
            series = series[0]

        # validate input types early to avoid confusing downstream errors
        if isinstance(series, TimeSeries):
            pass
        elif isinstance(series, (list, tuple)) and all(
            isinstance(s, TimeSeries) for s in series
        ):
            pass
        else:
            raise_log(
                TypeError(
                    "`series` must be a TimeSeries or a sequence of TimeSeries. "
                    f"Got type={type(series)}"
                ),
                logger,
            )

        if isinstance(series, TimeSeries):
            return self._predict_one(
                series,
                n=n,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
            )
        return [
            self._predict_one(
                s,
                n=n,
                num_samples=num_samples,
                predict_likelihood_parameters=predict_likelihood_parameters,
            )
            for s in series
        ]

    def _predict_one(
        self,
        series: TimeSeries,
        n: int,
        num_samples: int,
        predict_likelihood_parameters: bool,
    ) -> TimeSeries:
        if not isinstance(series, TimeSeries):
            raise_log(
                TypeError(f"Expected `series` to be a TimeSeries, got {type(series)}"),
                logger,
            )

        if series.n_components != 1:
            raise_log(
                ValueError("TiRexModel currently supports univariate series only."),
                logger,
            )

        if predict_likelihood_parameters and num_samples > 1:
            raise_log(
                ValueError(
                    "Use either `predict_likelihood_parameters=True` (returns quantiles) "
                    "or `num_samples>1` (returns samples), but not both."
                ),
                logger,
            )

        x = series.values(copy=False).astype(np.float32)[:, 0]  # (T,)

        if self.context_length is not None and len(x) > self.context_length:
            x = x[-self.context_length :]

        context = np.expand_dims(x, axis=0)  # (1, T)

        # TiRex returns (quantiles, mean)
        quantiles, mean = self._model.forecast(context=context, prediction_length=n)

        q = np.asarray(quantiles)
        m = np.asarray(mean)

        # squeeze batch dimension if present
        if q.ndim >= 1 and q.shape[0] == 1:
            q = np.squeeze(q, axis=0)
        if m.ndim >= 1 and m.shape[0] == 1:
            m = np.squeeze(m, axis=0)

        # normalize quantiles to (n, Q)
        # common candidates: (n, Q) or (Q, n)
        if q.ndim != 2:
            raise_log(
                ValueError(f"Unexpected quantile output shape from TiRex: {q.shape}"),
                logger,
            )

        if q.shape[0] == n:
            q_nq = q
        elif q.shape[1] == n:
            q_nq = q.T
        else:
            raise_log(
                ValueError(f"Unexpected quantile output shape from TiRex: {q.shape}"),
                logger,
            )

        # ---- Mode 1: return quantiles as components (Darts style) ----
        if predict_likelihood_parameters:
            # Darts expects (time, components). For univariate quantiles: components = Q.
            y = q_nq.astype(np.float32)  # (n, Q)

            # build forecast time index
            idx = series.time_index
            if isinstance(idx, pd.DatetimeIndex):
                start = series.end_time() + series.freq
                time_index = pd.date_range(start=start, periods=n, freq=series.freq)
            elif isinstance(idx, pd.RangeIndex):
                step = idx.step
                start = idx[-1] + step
                time_index = pd.RangeIndex(
                    start=start, stop=start + step * n, step=step
                )
            else:
                # fallback: integer index
                time_index = pd.RangeIndex(start=0, stop=n, step=1)

            # component names for quantiles
            cols = [f"q{q:g}" for q in self.quantile_config.quantiles]

            return TimeSeries.from_times_and_values(
                times=time_index,
                values=y,
                columns=cols,
                static_covariates=series.static_covariates,
                hierarchy=series.hierarchy,
            )

        # ---- Mode 2: sampling (Darts style) ----
        if num_samples > 1:
            samples = self._sample_from_quantiles(
                q_nq=q_nq,
                quantile_levels=self.quantile_config.quantiles,
                num_samples=num_samples,
            )  # (n, S)
            y = samples.reshape(n, 1, num_samples).astype(np.float32)  # (n, 1, S)
            return _build_forecast_series(y, series)

        # ---- Mode 3: deterministic mean ----
        m = np.asarray(m, dtype=np.float32).reshape(-1)
        if m.shape[0] != n:
            raise_log(
                ValueError(f"Unexpected mean output shape from TiRex: {m.shape}"),
                logger,
            )

        y = m.reshape(n, 1)
        return _build_forecast_series(y, series)

    @staticmethod
    def _sample_from_quantiles(
        q_nq: np.ndarray, quantile_levels: tuple[float, ...], num_samples: int
    ) -> np.ndarray:
        # q_nq: (n, Q)
        q = q_nq
        levels = np.asarray(quantile_levels, dtype=np.float32)  # (Q,)

        if q.shape[1] != len(levels):
            raise ValueError(
                f"Quantile count mismatch: got {q.shape[1]}, expected {len(levels)}"
            )

        # Inverse CDF sampling with piecewise linear interpolation between quantile points
        u = np.random.rand(q.shape[0], num_samples).astype(np.float32)  # (n, S)

        # For each horizon step, interpolate u across quantile levels
        out = np.empty((q.shape[0], num_samples), dtype=np.float32)
        for t in range(q.shape[0]):
            out[t, :] = np.interp(u[t, :], levels, q[t, :])

        return out
