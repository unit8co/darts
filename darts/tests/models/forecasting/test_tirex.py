from unittest.mock import patch

import numpy as np
import pytest

from darts.models.forecasting import tirex_model
from darts.tests.conftest import TORCH_AVAILABLE
from darts.utils.timeseries_generation import linear_timeseries

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )


# TiRex default quantiles (same as TimesFM 2.5 pretraining list)
ALL_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class _StubTiRex:
    """Stub model emulating tirex-ts forecast API."""

    def forecast(self, context, prediction_length: int):
        n = prediction_length
        Q = len(ALL_QUANTILES)

        # mean: (batch=1, n)
        mean = np.arange(1, n + 1, dtype=np.float32)[None, :]

        # quantiles: choose layout (1, n, Q) to keep wrapper simple
        quantiles = np.zeros((1, n, Q), dtype=np.float32)
        for qi, q in enumerate(ALL_QUANTILES):
            quantiles[0, :, qi] = mean[0, :] + (q - 0.5)

        return quantiles, mean


class TestTiRexModel:
    np.random.seed(42)

    series = linear_timeseries(length=200, dtype=np.float32, column_name="A")
    series_multi = linear_timeseries(
        length=200, dtype=np.float32, column_name="A"
    ).stack(linear_timeseries(length=200, dtype=np.float32, column_name="B"))
    cov = linear_timeseries(length=200, dtype=np.float32, column_name="C")

    def test_default_deterministic(self):
        model = tirex_model.TiRexModel(context_length=64)

        with patch.object(
            tirex_model,
            "_require_tirex",
            return_value=lambda *_a, **_k: _StubTiRex(),
        ):
            model.fit(self.series)

        pred = model.predict(n=10, series=self.series)
        assert len(pred) == 10
        assert pred.n_components == 1  # univariate
        assert pred.n_samples == 1

    def test_probabilistic_quantiles(self):
        model = tirex_model.TiRexModel(context_length=64)

        with patch.object(
            tirex_model,
            "_require_tirex",
            return_value=lambda *_a, **_k: _StubTiRex(),
        ):
            model.fit(self.series)

        # In Darts: predict_likelihood_parameters=True should return quantiles as components
        pred_q = model.predict(
            n=12, series=self.series, predict_likelihood_parameters=True
        )
        assert len(pred_q) == 12
        assert pred_q.n_components == len(ALL_QUANTILES)
        assert pred_q.n_samples == 1

    def test_probabilistic_sampling(self):
        model = tirex_model.TiRexModel(context_length=64)

        with patch.object(
            tirex_model,
            "_require_tirex",
            return_value=lambda *_a, **_k: _StubTiRex(),
        ):
            model.fit(self.series)

        pred_s = model.predict(n=12, series=self.series, num_samples=25)
        assert len(pred_s) == 12
        assert pred_s.n_components == 1
        assert pred_s.n_samples == 25

    def test_rejects_quantiles_and_sampling_together(self):
        model = tirex_model.TiRexModel(context_length=64)

        with patch.object(
            tirex_model,
            "_require_tirex",
            return_value=lambda *_a, **_k: _StubTiRex(),
        ):
            model.fit(self.series)

        with pytest.raises(ValueError, match="but not both"):
            model.predict(
                n=12,
                series=self.series,
                predict_likelihood_parameters=True,
                num_samples=10,
            )

    def test_rejects_multivariate(self):
        model = tirex_model.TiRexModel()
        with patch.object(
            tirex_model,
            "_require_tirex",
            return_value=lambda *_a, **_k: _StubTiRex(),
        ):
            with pytest.raises(ValueError, match="univariate"):
                model.fit(self.series_multi)

    def test_rejects_covariates(self):
        model = tirex_model.TiRexModel()
        with pytest.raises(ValueError, match="covariates"):
            model.fit(self.series, past_covariates=self.cov)

        with patch.object(
            tirex_model,
            "_require_tirex",
            return_value=lambda *_a, **_k: _StubTiRex(),
        ):
            model.fit(self.series)

        with pytest.raises(ValueError, match="covariates"):
            model.predict(n=5, series=self.series, future_covariates=self.cov)
