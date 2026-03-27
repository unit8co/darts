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

from darts import TimeSeries
from darts.datasets import ElectricityConsumptionZurichDataset
from darts.models import TiRexModel
from darts.models.forecasting import tirex_model
from darts.tests.conftest import tfm_kwargs
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.timeseries_generation import linear_timeseries


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


def _stub_loader(*_a, **_k):
    return _StubTiRexPipeline()


class TestTiRexModel:
    # set random seed
    np.random.seed(42)

    # ---- Fidelity Tests ---- #
    # load validation inputs once for fidelity tests
    ts_energy_train, ts_energy_val = load_validation_inputs()
    # prediction length for fidelity test
    prediction_length = 2048

    # ---- Dummy Tests ---- #
    series = linear_timeseries(length=200, dtype=np.float32, column_name="A")
    series_multi = linear_timeseries(
        length=200, dtype=np.float32, column_name="A"
    ).stack(linear_timeseries(length=200, dtype=np.float32, column_name="B"))
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

        np.testing.assert_allclose(pred_np, original, rtol=1e-5, atol=1e-5)

    def test_requires_license_acceptance(self):
        with pytest.raises(ValueError, match="accept_license=True"):
            TiRexModel(36, 12)

    def test_default_deterministic(self):
        model = TiRexModel(
            input_chunk_length=64,
            output_chunk_length=12,
            accept_license=True,
            **tfm_kwargs,
        )

        with patch.object(tirex_model, "_require_tirex", return_value=_stub_loader):
            model.fit(self.series)

        pred = model.predict(n=10, series=self.series)
        assert len(pred) == 10
        assert pred.n_components == 1
        assert pred.n_samples == 1
        assert pred.all_values().shape == (10, 1, 1)

    def test_fit_accepts_standard_foundation_kwargs(self):
        model = TiRexModel(
            input_chunk_length=64,
            output_chunk_length=12,
            accept_license=True,
            **tfm_kwargs,
        )

        with patch.object(tirex_model, "_require_tirex", return_value=_stub_loader):
            model.fit(self.series, val_series=self.series, load_best=False)

    def test_probabilistic_quantiles(self):
        model = TiRexModel(
            input_chunk_length=64,
            output_chunk_length=12,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            accept_license=True,
            **tfm_kwargs,
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
        model = TiRexModel(
            input_chunk_length=64,
            output_chunk_length=12,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            accept_license=True,
            **tfm_kwargs,
        )

        with patch.object(tirex_model, "_require_tirex", return_value=_stub_loader):
            model.fit(self.series)

        pred_s = model.predict(n=10, series=self.series, num_samples=25)
        assert len(pred_s) == 10
        assert pred_s.n_components == 1
        assert pred_s.n_samples == 25
        assert pred_s.all_values().shape == (10, 1, 25)

    def test_rejects_covariates(self):
        model = TiRexModel(
            input_chunk_length=64,
            output_chunk_length=12,
            accept_license=True,
            **tfm_kwargs,
        )

        with pytest.raises(ValueError, match="does not support any covariates"):
            model.fit(self.series, past_covariates=self.cov)

        with patch.object(tirex_model, "_require_tirex", return_value=_stub_loader):
            model.fit(self.series)

        with pytest.raises(ValueError, match="does not support any covariates"):
            model.predict(n=5, series=self.series, future_covariates=self.cov)

    def test_covariate_support_flags(self):
        model = TiRexModel(
            input_chunk_length=64,
            output_chunk_length=12,
            accept_license=True,
            **tfm_kwargs,
        )
        assert not model.supports_past_covariates
        assert not model.supports_future_covariates

    def test_rejects_too_long_horizon(self):
        # max horizon in TiRex is 2048; wrapper should enforce output_chunk_length + shift
        with pytest.raises(ValueError, match="2048"):
            TiRexModel(
                input_chunk_length=64,
                output_chunk_length=2049,
                accept_license=True,
                **tfm_kwargs,
            )

        with pytest.raises(ValueError, match="2048"):
            TiRexModel(
                input_chunk_length=64,
                output_chunk_length=2048,
                output_chunk_shift=1,
                accept_license=True,
                **tfm_kwargs,
            )
