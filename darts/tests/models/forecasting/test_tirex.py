from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from darts.models.forecasting import tirex_model
from darts.tests.conftest import TORCH_AVAILABLE
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.timeseries_generation import linear_timeseries

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

# TiRex default quantiles used by the wrapper
ALL_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class _StubTiRexPipeline:
    """Stub pipeline emulating `tirex-ts` API used by the wrapper.

    Must provide `forecast(context, prediction_length)`.
    The wrapper calls this inside a torch `forward()`.
    """

    def forecast(self, context, prediction_length: int):
        # context: torch.Tensor of shape (B, T)
        assert torch.is_tensor(context)
        B = int(context.shape[0])
        H = int(prediction_length)
        Q = len(ALL_QUANTILES)

        # mean: (B, H)
        mean = torch.arange(
            1, H + 1, dtype=torch.float32, device=context.device
        ).repeat(B, 1)

        # quantiles: (B, H, Q)
        quantiles = torch.zeros((B, H, Q), dtype=torch.float32, device=context.device)
        for qi, q in enumerate(ALL_QUANTILES):
            quantiles[:, :, qi] = mean + (float(q) - 0.5)

        return quantiles, mean


def _stub_loader(*_a, **_k):
    return _StubTiRexPipeline()


class TestTiRexModel:
    # set random seed
    np.random.seed(42)

    series = linear_timeseries(length=200, dtype=np.float32, column_name="A")
    series_multi = linear_timeseries(
        length=200, dtype=np.float32, column_name="A"
    ).stack(linear_timeseries(length=200, dtype=np.float32, column_name="B"))
    cov = linear_timeseries(length=200, dtype=np.float32, column_name="C")

    def test_requires_license_acceptance(self):
        with pytest.raises(ValueError, match="accept_license=True"):
            tirex_model.TiRexModel()

    def test_default_deterministic(self):
        model = tirex_model.TiRexModel(
            input_chunk_length=64,
            output_chunk_length=12,
            accept_license=True,
        )

        with patch.object(tirex_model, "_require_tirex", return_value=_stub_loader):
            model.fit(self.series)

        pred = model.predict(n=10, series=self.series)
        assert len(pred) == 10
        assert pred.n_components == 1
        assert pred.n_samples == 1
        assert pred.all_values().shape == (10, 1, 1)

    def test_probabilistic_quantiles(self):
        model = tirex_model.TiRexModel(
            input_chunk_length=64,
            output_chunk_length=12,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            accept_license=True,
        )

        with patch.object(tirex_model, "_require_tirex", return_value=_stub_loader):
            model.fit(self.series)

        pred_q = model.predict(
            n=10, series=self.series, predict_likelihood_parameters=True
        )
        assert len(pred_q) == 10
        assert pred_q.n_components == 3
        assert pred_q.n_samples == 1
        assert pred_q.all_values().shape == (10, 3, 1)

    def test_probabilistic_sampling(self):
        model = tirex_model.TiRexModel(
            input_chunk_length=64,
            output_chunk_length=12,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            accept_license=True,
        )

        with patch.object(tirex_model, "_require_tirex", return_value=_stub_loader):
            model.fit(self.series)

        pred_s = model.predict(n=10, series=self.series, num_samples=25)
        assert len(pred_s) == 10
        assert pred_s.n_components == 1
        assert pred_s.n_samples == 25
        assert pred_s.all_values().shape == (10, 1, 25)

    def test_rejects_covariates(self):
        model = tirex_model.TiRexModel(
            input_chunk_length=64,
            output_chunk_length=12,
            accept_license=True,
        )

        with pytest.raises(ValueError, match="covariates"):
            model.fit(self.series, past_covariates=self.cov)

        with patch.object(tirex_model, "_require_tirex", return_value=_stub_loader):
            model.fit(self.series)

        with pytest.raises(ValueError, match="covariates"):
            model.predict(n=5, series=self.series, future_covariates=self.cov)

    def test_rejects_multivariate(self):
        model = tirex_model.TiRexModel(
            input_chunk_length=64,
            output_chunk_length=12,
            accept_license=True,
        )

        with patch.object(tirex_model, "_require_tirex", return_value=_stub_loader):
            with pytest.raises(ValueError, match="univariate"):
                model.fit(self.series_multi)

    def test_rejects_too_long_horizon(self):
        # max horizon in TiRex is 2048; wrapper should enforce output_chunk_length + shift
        with pytest.raises(ValueError, match="2048"):
            tirex_model.TiRexModel(
                input_chunk_length=64,
                output_chunk_length=2049,
                accept_license=True,
            )

        with pytest.raises(ValueError, match="2048"):
            tirex_model.TiRexModel(
                input_chunk_length=64,
                output_chunk_length=2048,
                output_chunk_shift=1,
                accept_license=True,
            )
