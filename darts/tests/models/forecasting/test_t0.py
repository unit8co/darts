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
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)

# `T0Model` uses `from t0 import T0Forecaster`; mock `from_pretrained` in darts' module.
_PATCH_T0_FROM_PRETRAINED = (
    "darts.models.forecasting.t0_model.T0Forecaster.from_pretrained"
)


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

    def predict(self, context, horizon, quantiles, future_covariates=None):
        assert torch.is_tensor(context) and context.ndim == 3  # (B, V, T)
        batch, n_variates, _ = context.shape
        n_q = len(quantiles)
        if future_covariates is not None:
            # covariates must span context + horizon
            assert future_covariates.shape[0] == batch
            assert future_covariates.shape[2] == context.shape[-1] + horizon
        base = torch.arange(1, horizon + 1, dtype=torch.float32, device=context.device)
        quantile_offsets = torch.tensor(
            [float(q) - 0.5 for q in quantiles], device=context.device
        )
        # (B, V, horizon, Q)
        out = base.view(1, 1, horizon, 1) + quantile_offsets.view(1, 1, 1, n_q)
        return _StubForecast(out.expand(batch, n_variates, horizon, n_q).contiguous())


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

        # fine-tuning is not supported
        with pytest.raises(ValueError, match="Fine-tuning is not supported"):
            T0Model(
                input_chunk_length=12,
                output_chunk_length=6,
                enable_finetuning=True,
                **tfm_kwargs,
            )

    def test_default(self):
        model = T0Model(input_chunk_length=24, output_chunk_length=12, **tfm_kwargs)
        with patch(_PATCH_T0_FROM_PRETRAINED, return_value=_StubT0Forecaster()):
            model.fit(self.series)

        # deterministic, single component
        pred = model.predict(n=10, series=self.series)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == 1

        # autoregressive prediction (n > output_chunk_length)
        pred_ar = model.predict(n=20, series=self.series)
        assert len(pred_ar) == 20

    def test_probabilistic(self):
        model = T0Model(
            input_chunk_length=24,
            output_chunk_length=12,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            **tfm_kwargs,
        )
        with patch(_PATCH_T0_FROM_PRETRAINED, return_value=_StubT0Forecaster()):
            model.fit(self.series)
        assert model.model_created
        assert model.supports_probabilistic_prediction

        pred = model.predict(
            n=6, series=self.series, predict_likelihood_parameters=True
        )
        assert pred.n_components == 3  # 3 quantiles

    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_multivariate(self, probabilistic: bool):
        model = T0Model(
            input_chunk_length=24,
            output_chunk_length=8,
            likelihood=(
                QuantileRegression(quantiles=[0.1, 0.5, 0.9]) if probabilistic else None
            ),
            **tfm_kwargs,
        )
        with patch(_PATCH_T0_FROM_PRETRAINED, return_value=_StubT0Forecaster()):
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
        model = T0Model(input_chunk_length=24, output_chunk_length=12, **tfm_kwargs)
        past_cov = self.cov if which in ("past", "both") else None
        future_cov = self.cov if which in ("future", "both") else None

        with patch(_PATCH_T0_FROM_PRETRAINED, return_value=_StubT0Forecaster()):
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

    def test_multiple_series(self):
        model = T0Model(input_chunk_length=24, output_chunk_length=8, **tfm_kwargs)
        series_multi_2 = concatenate(
            [
                linear_timeseries(length=150, dtype=np.float32, column_name="A"),
                sine_timeseries(length=150, dtype=np.float32, column_name="B"),
                gaussian_timeseries(length=150, dtype=np.float32, column_name="C"),
            ],
            axis=1,
        )
        with patch(_PATCH_T0_FROM_PRETRAINED, return_value=_StubT0Forecaster()):
            model.fit(series=[self.series_multi, series_multi_2])
        pred = model.predict(n=5, series=[self.series_multi, series_multi_2])
        assert isinstance(pred, list) and len(pred) == 2
        assert all(len(p) == 5 for p in pred)
        assert all(p.n_components == 3 for p in pred)
