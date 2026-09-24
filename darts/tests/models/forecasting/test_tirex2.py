import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from darts.tests.conftest import TIREX2_AVAILABLE, TORCH_AVAILABLE, tfm_kwargs

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

if not TIREX2_AVAILABLE:
    pytest.skip(
        f"TiRex2 not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import torch

from darts import TimeSeries, concatenate
from darts.datasets import ElectricityConsumptionZurichDataset
from darts.models import TiRex2Model
from darts.tests.models.forecasting.foundation_test_utils import (
    TIREX2_LOAD_MODEL_PATCH_TARGET,
    TIREX2_MAX_PREDICTION_LENGTH,
    TIREX2_QUANTILES,
    TiRex2Stub,
)
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)


def load_validation_inputs():
    """Load validation inputs for TiRex2Model fidelity tests. The data imports
    here are adapted from the `20-SKLearnModel-examples` notebook.
    """
    # convert to float32 due to MPS not supporting float64
    ts_energy = ElectricityConsumptionZurichDataset().load().astype(np.float32)

    # extract temperature, solar irradiation and rain duration
    ts_weather = ts_energy[["T [°C]", "StrGlo [W/m2]", "RainDur [min]"]]
    # extract other weather features as past covariates for the sake of example
    # including humidity, wind direction, wind speed and air pressure
    ts_other = ts_energy[["Hr [%Hr]", "WD [°]", "WVs [m/s]", "WVv [m/s]", "p [hPa]"]]

    # extract households energy consumption
    ts_energy = ts_energy[["Value_NE5", "Value_NE7"]]

    # create train and validation splits
    validation_cutoff = pd.Timestamp("2022-01-01")
    ts_energy_train, ts_energy_val = ts_energy.split_after(validation_cutoff)
    return ts_energy_train, ts_energy_val, ts_weather, ts_other


@pytest.fixture
def pipeline():
    with patch(
        TIREX2_LOAD_MODEL_PATCH_TARGET,
        side_effect=lambda **kwargs: SimpleNamespace(model=TiRex2Stub()),
    ) as loader:
        yield loader


@pytest.fixture
def pl_trainer_fit():
    with patch("pytorch_lightning.Trainer.fit") as fit:
        yield fit


class TestTiRex2Model:
    # set random seed
    np.random.seed(42)

    # ---- Fidelity Tests ---- #
    # load validation inputs once for fidelity tests
    ts_energy_train, ts_energy_val, ts_weather, ts_other = load_validation_inputs()
    # prediction length for fidelity test
    prediction_length = 320

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
    past_cov = linear_timeseries(length=200, dtype=np.float32, column_name="B")
    future_cov = linear_timeseries(length=300, dtype=np.float32, column_name="C")

    @pytest.mark.slow
    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_fidelity(self, probabilistic: bool):
        """Test TiRex2Model predictions against the original tirex-ts implementation.
        The test passes if the predictions match up to a certain numerical tolerance.
        Original predictions were generated with the following code:

        ```python
        import numpy as np
        import pandas as pd
        import torch
        from darts.datasets import ElectricityConsumptionZurichDataset
        from tirex2 import TimeseriesType, load_model

        context_length = 2048
        prediction_length = 320

        # adapted from `20-SKLearnModel-examples` notebook
        ts_energy = ElectricityConsumptionZurichDataset().load().astype(np.float32)

        # future covariates: extract temperature, solar irradiation and rain duration
        ts_weather = ts_energy[["T [°C]", "StrGlo [W/m2]", "RainDur [min]"]]
        # past covariates: extract other weather features for the sake of example
        # including humidity, wind direction, wind speed and air pressure
        ts_other = ts_energy[["Hr [%Hr]", "WD [°]", "WVs [m/s]", "WVv [m/s]", "p [hPa]"]]

        # target: extract households energy consumption
        ts_energy = ts_energy[["Value_NE5", "Value_NE7"]]

        # create train and validation splits
        validation_cutoff = pd.Timestamp("2022-01-01")
        ts_energy_train, ts_energy_val = ts_energy.split_after(validation_cutoff)
        ts_weather_train, ts_weather_val = ts_weather.split_after(validation_cutoff)
        ts_other_train, ts_other_val = ts_other.split_after(validation_cutoff)

        pipeline = load_model("NX-AI/TiRex-2", device="cpu")

        # context: a TimeseriesType object that contains the target, past covariates and future covariates
        to_tensor = lambda df: torch.tensor(df.values().T, dtype=torch.float32)
        context = TimeseriesType(
            target=to_tensor(ts_energy_train)[:, -context_length:],
            past_covariates=to_tensor(ts_other_train)[:, -context_length:],
            future_covariates=torch.cat([
                to_tensor(ts_weather_train)[:, -context_length:],
                to_tensor(ts_weather_val)[:, :prediction_length],
            ], dim=1)
        )

        # forecast: (C, Q, P) = (variables, quantiles, prediction_length)
        forecast = pipeline.forecast(
            timeseries=[context],
            prediction_length=prediction_length,
        )[0]
        # (C, Q, P) -> (P, C, Q) for saving to npz
        pred_np = forecast.cpu().numpy().transpose(2, 0, 1)

        np.savez_compressed("tirex2.npz", pred=pred_np)
        ```

        Code accessed from https://github.com/NX-AI/tirex-2 commit used on 10 Sep 2026.

        """
        model = TiRex2Model(
            input_chunk_length=2048,  # use generous context
            output_chunk_length=self.prediction_length,  # no auto-regression
            likelihood=(
                QuantileRegression(quantiles=list(TIREX2_QUANTILES))
                if probabilistic
                else None
            ),
            **tfm_kwargs,
        )
        # fit w/o fine-tuning
        model.fit(
            series=self.ts_energy_train,
            past_covariates=self.ts_other,
            future_covariates=self.ts_weather,
        )

        pred = model.predict(
            n=self.prediction_length,
            past_covariates=self.ts_other,
            future_covariates=self.ts_weather,
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
            / "tirex2"
            / "tirex2_prediction"
            / "tirex2.npz"
        )
        original = np.load(path)["pred"]

        if not probabilistic:
            original = original[:, :, [4]]  # median quantile (index 4 = 0.5)

        # increase tolerance due to platform differences
        # reference: https://github.com/NX-AI/tirex/blob/30702459b2454660242d63e4ef8f57906e6be65b/tests/test_forecast.py
        np.testing.assert_allclose(pred_np, original, rtol=1.6e-2, atol=1e-5)

    @pytest.mark.slow
    def test_creation(self, pl_trainer_fit):
        kwargs = tfm_kwargs

        # ----- Input/output chunk length checks ----- #
        # can use shorter input/output chunk length than max
        model = TiRex2Model(
            input_chunk_length=7,
            output_chunk_length=TIREX2_MAX_PREDICTION_LENGTH - 1,
            **kwargs,
        )
        model.fit(self.series)
        pl_trainer_fit.assert_not_called()

        # loading-time check: cannot use a longer output chunk length than max
        with pytest.raises(ValueError, match=r"`output_chunk_length` \d+ plus"):
            model = TiRex2Model(
                input_chunk_length=19,
                output_chunk_length=TIREX2_MAX_PREDICTION_LENGTH + 1,
                **kwargs,
            )
            model.fit(self.series)
            _ = model.predict(n=5, series=self.series)

        # loading-time check: output chunk length plus shift cannot exceed max
        with pytest.raises(ValueError, match=r"`output_chunk_length` \d+ plus"):
            model = TiRex2Model(
                input_chunk_length=23,
                output_chunk_length=TIREX2_MAX_PREDICTION_LENGTH - 1,
                output_chunk_shift=3,
                **kwargs,
            )
            model.fit(self.series)
            _ = model.predict(n=5, series=self.series)

        # ----- Likelihood checks ----- #
        # can use likelihood QuantileRegression with supported quantiles
        TiRex2Model(
            input_chunk_length=11,
            output_chunk_length=34,
            likelihood=QuantileRegression([0.1, 0.5, 0.9]),
            **kwargs,
        )

        # cannot use likelihood others than QuantileRegression
        with pytest.raises(ValueError, match="Only QuantileRegression likelihood is"):
            TiRex2Model(
                input_chunk_length=29,
                output_chunk_length=12,
                likelihood=GaussianLikelihood(),
                **kwargs,
            )

        # loading-time check: quantiles must match those used in pre-training
        with pytest.raises(
            ValueError, match="does not support the requested quantiles"
        ):
            model = TiRex2Model(
                input_chunk_length=7,
                output_chunk_length=6,
                likelihood=QuantileRegression(quantiles=[0.23, 0.5, 0.77]),
                **kwargs,
            )
            model.fit(self.series)
            _ = model.predict(n=5, series=self.series)

        # ----- Checkpoint path checks ----- #
        # can use `hub_model_name` and `hub_model_revision` to specify checkpoint path
        # no model download should occur since model is not created yet
        TiRex2Model(
            input_chunk_length=5,
            output_chunk_length=3,
            hub_model_name="NX-AI/TiRex-2",
            hub_model_revision="05e5b26db52bfb256f1ae1bdf785589850482de3",
            local_dir="/tmp/weights",
            **kwargs,
        )

        # cannot use `ckpt_path` in `tirex2_kwargs`
        with pytest.raises(
            ValueError,
            match="Pass `ckpt_path` via `hub_model_name`, not `tirex2_kwargs`.",
        ):
            TiRex2Model(
                input_chunk_length=7,
                output_chunk_length=6,
                tirex2_kwargs={"ckpt_path": "/tmp/weights"},
                **kwargs,
            )

    def test_default(self, pipeline):
        model = TiRex2Model(
            input_chunk_length=3,
            output_chunk_length=4,
            **tfm_kwargs,
        )

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

    def test_probabilistic(self, pipeline):
        # probabilistic model
        model = TiRex2Model(
            input_chunk_length=5,
            output_chunk_length=6,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            **tfm_kwargs,
        )

        # calling `fit()` should not use `trainer.fit()`
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

    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_multivariate(self, pipeline, probabilistic: bool):
        # create model
        model = TiRex2Model(
            input_chunk_length=3,
            output_chunk_length=8,
            likelihood=(
                QuantileRegression(quantiles=[0.1, 0.5, 0.9]) if probabilistic else None
            ),
            **tfm_kwargs,
        )
        model.fit(series=self.series_multi)
        pred = model.predict(n=7, predict_likelihood_parameters=probabilistic)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 7
        if probabilistic:
            assert pred.n_components == 9  # 3 variables x 3 quantiles
        else:
            assert pred.n_components == 3

    def test_past_covariates(self):
        model = TiRex2Model(
            input_chunk_length=11,
            output_chunk_length=13,
            **tfm_kwargs,
        )
        model.fit(series=self.series, past_covariates=self.past_cov)
        pred = model.predict(n=10, series=self.series, past_covariates=self.past_cov)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == 1

    def test_future_covariates(self):
        model = TiRex2Model(
            input_chunk_length=11,
            output_chunk_length=13,
            **tfm_kwargs,
        )
        model.fit(series=self.series, future_covariates=self.future_cov)
        pred = model.predict(
            n=10, series=self.series, future_covariates=self.future_cov
        )
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == 1

    def test_past_and_future_covariates(self):
        model = TiRex2Model(
            input_chunk_length=11,
            output_chunk_length=13,
            **tfm_kwargs,
        )
        model.fit(
            series=self.series,
            past_covariates=self.past_cov,
            future_covariates=self.future_cov,
        )
        pred = model.predict(
            n=10,
            series=self.series,
            past_covariates=self.past_cov,
            future_covariates=self.future_cov,
        )
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == 1

    def test_multiple_series(self):
        # create model
        model = TiRex2Model(
            input_chunk_length=2,
            output_chunk_length=3,
            **tfm_kwargs,
        )
        model.fit(series=[self.series_multi, self.series_multi_2])
        pred = model.predict(n=5, series=[self.series_multi, self.series_multi_2])

        # check that we get a list of predictions
        assert isinstance(pred, list) and len(pred) == 2
        assert all(isinstance(p, TimeSeries) for p in pred)

        # check that each prediction has correct length
        assert all(len(p) == 5 for p in pred)
        # check that each prediction is deterministic with 3 components
        assert all(p.n_components == 3 for p in pred)

    @pytest.mark.parametrize("accelerator", ["cpu", "cuda", "mps"])
    def test_accelerator_selection(self, accelerator, pipeline):
        if accelerator == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available.")
        if accelerator == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS is not available.")

        load_kwargs = {
            "ckpt_path": "test-checkpoint",
            "hf_kwargs": {"revision": "test"},
        }

        kwargs = copy.deepcopy(tfm_kwargs)
        kwargs["pl_trainer_kwargs"]["accelerator"] = accelerator
        # create model: loader should not be called yet
        model = TiRex2Model(
            hub_model_name=load_kwargs["ckpt_path"],
            hub_model_revision=load_kwargs["hf_kwargs"]["revision"],
            input_chunk_length=3,
            output_chunk_length=4,
            **kwargs,
        )
        pipeline.assert_not_called()

        # fit() skips the Lightning stage when there is no fine-tuning.
        model.fit(self.series)
        pipeline.assert_not_called()

        # predict() loads the model on the execution device in configure_model().
        model.predict(n=5, series=self.series)
        pipeline.assert_called_once()
        pipeline.assert_called_with(**load_kwargs, device=accelerator)
