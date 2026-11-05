from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest

from darts.tests.conftest import T0_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

if not T0_AVAILABLE:
    pytest.skip(
        f"tfc-t0 not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import torch

from darts import TimeSeries, concatenate
from darts.models import T0Model
from darts.tests.models.forecasting.foundation_test_utils import tiny_t0_dir
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)

# `T0Model` rebuilds T0 from the Hub config and loads the weights through the connector; point it at
# a tiny local checkpoint (no gated download) and swap the rebuilt model for a stub.
_LOCAL = {"local_dir": tiny_t0_dir()}
_PATCH_T0_FROM_CONFIG = "darts.models.forecasting.t0_model.T0Forecaster.from_config"
_PATCH_LOAD_WEIGHTS = "darts.models.components.huggingface_connector.HuggingFaceConnector.load_model_weights"


@contextmanager
def _stub_t0(stub: "_StubT0Forecaster | None" = None):
    """Build `stub` instead of the real T0 forecaster; weight loading becomes a no-op."""
    stub = _StubT0Forecaster() if stub is None else stub
    with patch(_PATCH_T0_FROM_CONFIG, return_value=stub), patch(_PATCH_LOAD_WEIGHTS):
        yield stub


class _StubForecast:
    def __init__(self, quantiles: torch.Tensor):
        self.quantiles = quantiles


class _StubT0Forecaster(torch.nn.Module):
    """Stub emulating the `tfc-t0` ``T0Forecaster`` API used by the wrapper.

    ``predict(context, horizon, quantiles, future_covariates)`` returns a ``Forecast``-like object whose
    ``quantiles`` is shaped ``(B, V, horizon, Q)`` — matching ``T0Forecaster`` for ndim-3 (multivariate) context.
    """

    def __init__(self):
        super().__init__()
        # a parameter so `next(self.parameters()).device` works like the real model
        self._p = torch.nn.Parameter(torch.zeros(1))

    def predict(self, context, horizon, quantile_levels, future_covariates=None):
        assert torch.is_tensor(context) and context.ndim == 3  # (B, V, T)
        batch, n_variates, _ = context.shape
        n_q = len(quantile_levels)
        if future_covariates is not None:
            # covariates must span context + horizon
            assert future_covariates.shape[0] == batch
            assert future_covariates.shape[2] == context.shape[-1] + horizon
        base = torch.arange(1, horizon + 1, dtype=torch.float32, device=context.device)
        quantile_offsets = torch.tensor(
            [float(q) - 0.5 for q in quantile_levels], device=context.device
        )
        # (B, V, horizon, Q)
        out = base.view(1, 1, horizon, 1) + quantile_offsets.view(1, 1, 1, n_q)
        return _StubForecast(out.expand(batch, n_variates, horizon, n_q).contiguous())


def series_with_nans(series: TimeSeries, start: int, end: int | None) -> TimeSeries:
    """Returns a copy of `series` with the values in [start, end) set to NaN."""
    values = series.values(copy=True).astype(np.float32)
    values[start:end, :] = np.nan
    return TimeSeries.from_times_and_values(series.time_index, values)


class TestT0Model:
    np.random.seed(42)

    series = linear_timeseries(length=200, dtype=np.float32, column_name="A")
    series_multi = concatenate(
        [
            linear_timeseries(length=200, dtype=np.float32, column_name="A"),
            sine_timeseries(length=200, dtype=np.float32, column_name="B"),
            gaussian_timeseries(length=200, dtype=np.float32, column_name="C"),
        ],
        axis=1,
    )
    cov = sine_timeseries(length=400, dtype=np.float32, column_name="cov")

    def test_creation(self):
        # only QuantileRegression likelihood is supported
        with pytest.raises(ValueError, match="Only QuantileRegression likelihood is"):
            T0Model(
                input_chunk_length=12,
                output_chunk_length=6,
                likelihood=GaussianLikelihood(),
                **tfm_kwargs,
            )

        # fine-tuning is supported
        model = T0Model(
            input_chunk_length=12,
            output_chunk_length=6,
            enable_finetuning=True,
            **tfm_kwargs,
        )
        assert model.enable_finetuning is True

    def test_default(self):
        model = T0Model(
            input_chunk_length=24, output_chunk_length=12, **_LOCAL, **tfm_kwargs
        )
        with _stub_t0():
            model.fit(self.series)
        assert model.model_created
        assert not model.supports_probabilistic_prediction

        # deterministic, single component
        pred = model.predict(n=10, series=self.series)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == 1

        # autoregressive prediction (n > output_chunk_length)
        pred_ar = model.predict(n=20, series=self.series)
        assert isinstance(pred_ar, TimeSeries)
        assert len(pred_ar) == 20
        assert pred_ar.n_components == 1

    def test_probabilistic(self):
        model = T0Model(
            input_chunk_length=24,
            output_chunk_length=12,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            **_LOCAL,
            **tfm_kwargs,
        )
        with _stub_t0():
            model.fit(self.series)
        assert model.model_created
        assert model.supports_probabilistic_prediction

        pred = model.predict(
            n=6, series=self.series, predict_likelihood_parameters=True
        )
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 6
        assert pred.n_components == 3  # 3 quantiles

        # probabilistic model allows autoregressive predictions (8 > 6)
        pred_ar = model.predict(
            n=14,
            series=self.series,
            num_samples=10,
        )
        assert isinstance(pred_ar, TimeSeries)
        assert len(pred_ar) == 14
        assert pred_ar.n_components == 1  # sampling yields single component
        assert pred_ar.n_samples == 10

    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_multivariate(self, probabilistic: bool):
        model = T0Model(
            input_chunk_length=24,
            output_chunk_length=8,
            likelihood=(
                QuantileRegression(quantiles=[0.1, 0.5, 0.9]) if probabilistic else None
            ),
            **_LOCAL,
            **tfm_kwargs,
        )
        with _stub_t0():
            model.fit(series=self.series_multi)
        pred = model.predict(n=7, predict_likelihood_parameters=probabilistic)
        assert len(pred) == 7
        if probabilistic:
            assert pred.n_components == 9  # 3 variables x 3 quantiles
        else:
            assert pred.n_components == 3

    @pytest.mark.parametrize("which", ["future", "past", "both"])
    def test_covariates(self, which: str):
        # past covariates are forecast jointly with the target and dropped from the output;
        # future covariates are passed to T0's covariate branch ([B, F, context+horizon], asserted by the stub).
        model = T0Model(
            input_chunk_length=24, output_chunk_length=12, **_LOCAL, **tfm_kwargs
        )
        past_cov = self.cov if which in ("past", "both") else None
        future_cov = self.cov if which in ("future", "both") else None

        with _stub_t0():
            model.fit(
                series=self.series,
                past_covariates=past_cov,
                future_covariates=future_cov,
            )
        pred = model.predict(
            n=12,
            series=self.series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 12
        # only the single target component is returned, never the past covariate
        assert pred.n_components == 1

    def test_missing_values(self):
        """NaNs in target and covariates are handled via the masking logic of the
        ported `decode()`, instead of the linear interpolation applied by the
        upstream `TimesFM3Forecaster.predict_batch()`. Predictions must not
        contain NaNs for any of the missing value locations.
        """

        def make_model() -> T0Model:
            return T0Model(
                input_chunk_length=8,
                output_chunk_length=4,
                **_LOCAL,
                **tfm_kwargs,
            )

        # NaNs inside the target series, incl. autoregressive prediction
        series_nan = series_with_nans(self.series, 20, 26)
        model = make_model()
        model.fit(series=series_nan)
        pred = model.predict(n=6, series=series_nan)
        assert isinstance(pred, TimeSeries)
        assert not np.isnan(pred.all_values(copy=False)).any()

        # NaNs at the end of the target series (trailing missing values)
        series_trailing_nan = series_with_nans(self.series, len(self.series) - 3, None)
        model = make_model()
        model.fit(series=series_trailing_nan)
        pred = model.predict(n=4, series=series_trailing_nan)
        assert isinstance(pred, TimeSeries)
        assert not np.isnan(pred.all_values(copy=False)).any()

        # NaNs in the past covariates
        past_cov_nan = series_with_nans(self.cov, 5, 10)
        model = make_model()
        model.fit(series=self.series, past_covariates=past_cov_nan)
        pred = model.predict(n=4, series=self.series, past_covariates=past_cov_nan)
        assert isinstance(pred, TimeSeries)
        assert not np.isnan(pred.all_values(copy=False)).any()

        # NaNs in the future covariates
        future_cov_nan = series_with_nans(self.cov, 204, 208)
        model = make_model()
        model.fit(series=self.series, future_covariates=future_cov_nan)
        pred = model.predict(n=4, series=self.series, future_covariates=future_cov_nan)
        assert isinstance(pred, TimeSeries)
        assert not np.isnan(pred.all_values(copy=False)).any()

    def test_output_chunk_shift(self):
        """With `output_chunk_shift`, predictions start after the shifted gap and
        auto-regressive prediction (`n > output_chunk_length`) is not allowed."""
        model = T0Model(
            input_chunk_length=8,
            output_chunk_length=4,
            output_chunk_shift=2,
            **_LOCAL,
            **tfm_kwargs,
        )
        model.fit(series=self.series, future_covariates=self.cov)
        pred = model.predict(n=4, series=self.series, future_covariates=self.cov)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 4
        # predictions start `output_chunk_shift + 1` steps after the end of the series
        assert pred.start_time() == self.series.end_time() + self.series.freq * 3
        assert not np.isnan(pred.all_values(copy=False)).any()

        # auto-regression is not allowed with an output chunk shift
        with pytest.raises(ValueError, match="output_chunk_shift > 0"):
            model.predict(n=5, series=self.series, future_covariates=self.cov)

    def test_multiple_series(self):
        model = T0Model(
            input_chunk_length=24, output_chunk_length=8, **_LOCAL, **tfm_kwargs
        )
        series_multi_2 = concatenate(
            [
                linear_timeseries(length=150, dtype=np.float32, column_name="A"),
                sine_timeseries(length=150, dtype=np.float32, column_name="B"),
                gaussian_timeseries(length=150, dtype=np.float32, column_name="C"),
            ],
            axis=1,
        )
        with _stub_t0():
            model.fit(series=[self.series_multi, series_multi_2])
        pred = model.predict(n=5, series=[self.series_multi, series_multi_2])
        assert isinstance(pred, list) and len(pred) == 2
        assert all(len(p) == 5 for p in pred)
        assert all(p.n_components == 3 for p in pred)
