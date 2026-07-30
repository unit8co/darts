import logging
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

from darts import TimeSeries, concatenate
from darts.models import T0Model
from darts.tests.models.forecasting.foundation_test_utils import tiny_t0, tiny_t0_dir
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)

# Load a small real model from a local dir through the shared HuggingFace connector,
# exactly like the other foundation models — no gated t0-alpha download.
_LOCAL = {"local_dir": tiny_t0_dir()}
_PATCH_T0_FROM_CONFIG = "darts.models.forecasting.t0_model.T0Forecaster.from_config"


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

        # fine-tuning is supported: construction with enable_finetuning must not raise
        T0Model(
            input_chunk_length=12,
            output_chunk_length=6,
            enable_finetuning=True,
            **tfm_kwargs,
        )

    def test_default(self):
        model = T0Model(
            input_chunk_length=24, output_chunk_length=12, **_LOCAL, **tfm_kwargs
        )
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
            **_LOCAL,
            **tfm_kwargs,
        )
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
            **_LOCAL,
            **tfm_kwargs,
        )
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
        # future covariates are passed to T0's covariate branch ([B, F, context+horizon]).
        model = T0Model(
            input_chunk_length=24, output_chunk_length=12, **_LOCAL, **tfm_kwargs
        )
        past_cov = self.cov if which in ("past", "both") else None
        future_cov = self.cov if which in ("future", "both") else None

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

    def test_finetuning_caps_horizon_with_warning(self, caplog):
        # fine-tuning is a single parallel-patch pass: a horizon beyond max_horizon is not supported,
        # so the loss is truncated to the first max_horizon steps with a warning (no error).
        # (The fine-tuning contract itself — requires_grad, fit with a val series, predict — is
        # covered by test_foundation.py::test_finetuning_all_models.)
        model = T0Model(
            input_chunk_length=24,
            output_chunk_length=16,
            enable_finetuning=True,
            n_epochs=1,
            **_LOCAL,
            **tfm_kwargs,
        )
        tiny = tiny_t0()
        tiny.max_horizon = 8  # multiple of patch_size; horizon (16) now exceeds it
        with caplog.at_level(logging.WARNING):  # noqa: PT012
            with patch(_PATCH_T0_FROM_CONFIG, return_value=tiny):
                model.fit(self.series)
        assert "not supported for training" in caplog.text

        # fine-tuning still completes and the model forecasts through the inference path
        pred = model.predict(n=6, series=self.series)
        assert len(pred) == 6

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
        model.fit(series=[self.series_multi, series_multi_2])
        pred = model.predict(n=5, series=[self.series_multi, series_multi_2])
        assert isinstance(pred, list) and len(pred) == 2
        assert all(len(p) == 5 for p in pred)
        assert all(p.n_components == 3 for p in pred)
