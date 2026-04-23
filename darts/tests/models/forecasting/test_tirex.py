import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from darts.tests.conftest import TIREX_AVAILABLE, TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

if not TIREX_AVAILABLE:
    pytest.skip(
        f"TiRex not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

# TiRex default quantiles used by the wrapper
ALL_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

import torch

from darts import TimeSeries, concatenate
from darts.datasets import ElectricityConsumptionZurichDataset
from darts.models import TiRexModel
from darts.tests.conftest import tfm_kwargs
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)

# `TiRexModel` uses `from tirex import load_model`; mock the name in darts' module.
_PATCH_TIREX_LOAD_MODEL = "darts.models.forecasting.tirex_model.load_model"


def load_validation_inputs():
    """Load validation inputs for TiRexModel fidelity tests."""
    # convert to float32 due to MPS not supporting float64
    ts_energy = ElectricityConsumptionZurichDataset().load().astype(np.float32)
    ts_energy = ts_energy[["Value_NE5", "Value_NE7"]]
    validation_cutoff = pd.Timestamp("2022-01-01")
    ts_energy_train, ts_energy_val = ts_energy.split_after(validation_cutoff)
    return ts_energy_train, ts_energy_val


class _StubTiRexPipeline:
    """Stub pipeline emulating `tirex-ts` API used by the wrapper.

    Must provide `_forecast_quantiles(context, prediction_length)`.
    The wrapper calls this inside a torch `forward()`.
    """

    def _forecast_quantiles(self, context, prediction_length: int, **_kwargs):
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


class TestTiRexModel:
    # set random seed
    np.random.seed(42)

    # ---- Fidelity Tests ---- #
    # load validation inputs once for fidelity tests
    ts_energy_train, ts_energy_val = load_validation_inputs()
    # prediction length for fidelity test
    prediction_length = 512
    # maximum prediction length w/o triggering auto-regression where the results
    # would diverge from the original implementation due to different sampling methods
    max_prediction_length = 2048

    # ---- Dummy Tests ---- #
    series = linear_timeseries(length=200, dtype=np.float32, column_name="A")
    series_multi = concatenate(
        [
            linear_timeseries(length=200, dtype=np.float32, column_name="A"),
            sine_timeseries(length=200, dtype=np.float32, column_name="B"),
            gaussian_timeseries(length=200, dtype=np.float32, column_name="C"),
        ],
        axis=1,
    )
    series_multi_2 = concatenate(
        [
            linear_timeseries(length=150, dtype=np.float32, column_name="A"),
            sine_timeseries(length=150, dtype=np.float32, column_name="B"),
            gaussian_timeseries(length=150, dtype=np.float32, column_name="C"),
        ],
        axis=1,
    )
    cov = linear_timeseries(length=200, dtype=np.float32, column_name="C")

    @pytest.mark.slow
    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_fidelity(self, probabilistic: bool):
        """Test TiRexModel predictions against the original tirex-ts implementation.
        The test passes if the predictions match up to a certain numerical tolerance.
        Original predictions were generated with the following code:

        ```python
        import numpy as np
        import pandas as pd
        import torch
        from darts.datasets import ElectricityConsumptionZurichDataset
        from tirex import load_model

        ts_energy = ElectricityConsumptionZurichDataset().load().astype(np.float32)
        ts_energy = ts_energy[["Value_NE5", "Value_NE7"]]
        ts_energy_train, _ = ts_energy.split_after(pd.Timestamp("2022-01-01"))

        pipeline = load_model("NX-AI/TiRex")

        context_length = 2048
        prediction_length = 512

        # context: each variable as its own batch item -> (n_variables, T)
        context = torch.tensor(ts_energy_train.values().T, dtype=torch.float32)
        context = context[:, -context_length:]

        # quantiles: (B, H, Q) where B=2, H=512, Q=9
        quantiles, _ = pipeline._forecast_quantiles(
            context=context,
            prediction_length=prediction_length,
            output_device=context.device,
        )
        # (B, H, Q) -> (H, B, Q) = (time, variables, quantiles)
        pred_np = quantiles.cpu().numpy().transpose(1, 0, 2)

        np.savez_compressed("tirex.npz", pred=pred_np)
        ```

        Code accessed from https://github.com/NX-AI/tirex commit used on 26 March 2026.

        """
        model = TiRexModel(
            input_chunk_length=2048,  # use generous context
            output_chunk_length=self.prediction_length,  # no auto-regression
            likelihood=(
                QuantileRegression(quantiles=list(ALL_QUANTILES))
                if probabilistic
                else None
            ),
            accept_license=True,
            **tfm_kwargs,
        )
        # fit w/o fine-tuning
        model.fit(series=self.ts_energy_train)

        pred = model.predict(
            n=self.prediction_length,
            predict_likelihood_parameters=probabilistic,
        )
        assert isinstance(pred, TimeSeries)
        # reshape to (time, variables, quantiles)
        pred_np = pred.values().reshape(
            self.prediction_length, self.ts_energy_train.n_components, -1
        )

        # load reference predictions
        path = (
            Path(__file__).parent
            / "artefacts"
            / "tirex"
            / "tirex_prediction"
            / "tirex.npz"
        )
        original = np.load(path)["pred"]

        if not probabilistic:
            original = original[:, :, [4]]  # median quantile (index 4 = 0.5)

        # increase tolerance due to platform differences
        # reference: https://github.com/NX-AI/tirex/blob/30702459b2454660242d63e4ef8f57906e6be65b/tests/test_forecast.py
        np.testing.assert_allclose(pred_np, original, rtol=1.6e-2, atol=1e-5)

    @pytest.mark.slow
    def test_creation(self, caplog):
        # requires accepting the license
        with pytest.raises(ValueError, match="accept_license=True"):
            TiRexModel(36, 12, **tfm_kwargs)

        # can use shorter input/output chunk length than max
        kwargs = {"accept_license": True, **tfm_kwargs}
        # cannot create longer output chunk length than max
        with pytest.raises(ValueError, match=r"`output_chunk_length` \d+ plus"):
            TiRexModel(
                input_chunk_length=19,
                output_chunk_length=self.max_prediction_length + 1,
                **kwargs,
            )

        # cannot create longer output chunk length + output chunk shift than max
        with pytest.raises(ValueError, match=r"`output_chunk_length` \d+ plus"):
            TiRexModel(
                input_chunk_length=23,
                output_chunk_length=self.max_prediction_length - 1,
                output_chunk_shift=3,
                **kwargs,
            )

        # cannot use likelihood others than QuantileRegression
        with pytest.raises(ValueError, match="Only QuantileRegression likelihood is"):
            TiRexModel(
                input_chunk_length=29,
                output_chunk_length=12,
                likelihood=GaussianLikelihood(),
                **kwargs,
            )

        # cannot use quantiles other than those used in pre-training
        with pytest.raises(ValueError, match="must be a subset of TiRex quantiles"):
            TiRexModel(
                input_chunk_length=7,
                output_chunk_length=6,
                likelihood=QuantileRegression(quantiles=[0.23, 0.5, 0.77]),
                **kwargs,
            )

        # cannot use quantiles other than those used in pre-training
        with pytest.raises(
            ValueError,
            match="The `path` argument for loading the TiRex model should be passed via `hub_model_name`",
        ):
            TiRexModel(
                input_chunk_length=7,
                output_chunk_length=6,
                tirex_kwargs={"path": "dummy_path"},
                **kwargs,
            )

        # should give info on fine-tuned models conforming to same license
        with caplog.at_level(logging.INFO):
            _ = TiRexModel(
                input_chunk_length=2,
                output_chunk_length=3,
                enable_finetuning=True,
                **kwargs,
            )
            assert (
                "Fine-tuned weights are derivative works subject to the same terms"
                in caplog.text
            )
        caplog.clear()

    def test_default(self):
        model = TiRexModel(
            input_chunk_length=3,
            output_chunk_length=4,
            accept_license=True,
            **tfm_kwargs,
        )

        with patch(_PATCH_TIREX_LOAD_MODEL, return_value=_StubTiRexPipeline()):
            model.fit(self.series)

        # predictions should not be probabilistic
        pred = model.predict(n=10, series=self.series)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == 1

        # default model allows autoregressive predictions (6 > 4)
        pred_ar = model.predict(n=6, series=self.series)
        assert isinstance(pred_ar, TimeSeries)
        assert len(pred_ar) == 6
        assert pred_ar.n_components == 1

    @pytest.mark.slow
    def test_probabilistic(self):
        # probabilistic model
        model = TiRexModel(
            input_chunk_length=5,
            output_chunk_length=6,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            accept_license=True,
            **tfm_kwargs,
        )

        # calling `fit()` should not use `trainer.fit()`
        with patch(_PATCH_TIREX_LOAD_MODEL, return_value=_StubTiRexPipeline()):
            model.fit(self.series)
        assert model.model_created
        assert model.supports_probabilistic_prediction

        # predictions should be probabilistic
        pred = model.predict(
            n=5, series=self.series, predict_likelihood_parameters=True
        )
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 5
        assert pred.n_components == 3  # 3 quantiles

        # probabilistic model allows autoregressive predictions (8 > 6)
        pred_ar = model.predict(
            n=8,
            series=self.series,
            num_samples=10,
        )
        assert isinstance(pred_ar, TimeSeries)
        assert len(pred_ar) == 8
        assert pred_ar.n_components == 1  # sampling yields single component
        assert pred_ar.n_samples == 10

    @pytest.mark.slow
    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_multivariate(self, probabilistic: bool):
        # create model
        model = TiRexModel(
            input_chunk_length=3,
            output_chunk_length=8,
            likelihood=(
                QuantileRegression(quantiles=[0.1, 0.5, 0.9]) if probabilistic else None
            ),
            accept_license=True,
            **tfm_kwargs,
        )
        with patch(_PATCH_TIREX_LOAD_MODEL, return_value=_StubTiRexPipeline()):
            model.fit(series=self.series_multi)
        pred = model.predict(n=7, predict_likelihood_parameters=probabilistic)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 7
        if probabilistic:
            assert pred.n_components == 9  # 3 variables x 3 quantiles
        else:
            assert pred.n_components == 3

    def test_covariates(self):
        model = TiRexModel(
            input_chunk_length=21,
            output_chunk_length=23,
            accept_license=True,
            **tfm_kwargs,
        )

        # past covariates are not supported
        with pytest.raises(ValueError, match="does not support `past_covariates`"):
            model.fit(series=self.series, past_covariates=self.cov)

        # future covariates are not supported
        with pytest.raises(ValueError, match="does not support `future_covariates`"):
            model.fit(series=self.series, future_covariates=self.cov)

        # past and future covariates are not supported
        with pytest.raises(
            ValueError, match="does not support `past_covariates`, `future_covariates`"
        ):
            model.fit(
                series=self.series,
                past_covariates=self.cov,
                future_covariates=self.cov,
            )

    @pytest.mark.slow
    def test_multiple_series(self):
        # create model
        model = TiRexModel(
            input_chunk_length=2,
            output_chunk_length=3,
            accept_license=True,
            **tfm_kwargs,
        )
        with patch(_PATCH_TIREX_LOAD_MODEL, return_value=_StubTiRexPipeline()):
            model.fit(series=[self.series_multi, self.series_multi_2])
        pred = model.predict(n=5, series=[self.series_multi, self.series_multi_2])

        # check that we get a list of predictions
        assert isinstance(pred, list) and len(pred) == 2
        assert all(isinstance(p, TimeSeries) for p in pred)

        # check that each prediction has correct length
        assert all(len(p) == 5 for p in pred)
        # check that each prediction is deterministic with 3 components
        assert all(p.n_components == 3 for p in pred)
