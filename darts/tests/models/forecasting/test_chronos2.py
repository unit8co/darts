from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries, concatenate
from darts.datasets import ElectricityConsumptionZurichDataset
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils.likelihood_models import GaussianLikelihood
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

from darts.models import Chronos2Model
from darts.utils.likelihood_models import QuantileRegression

# quantiles used during Chronos-2 pre-training
all_quantiles = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]


def load_validation_inputs():
    """Load validation inputs for Chronos2Model fidelity tests. The data imports
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


class TestChronos2Model:
    # ---- Fidelity Tests ---- #
    # load validation inputs once for fidelity tests
    ts_energy_train, ts_energy_val, ts_weather, ts_other = load_validation_inputs()
    # maximum prediction length w/o triggering auto-regression where the results
    # would diverge from the original implementation due to different sampling methods
    max_prediction_length = 1024

    # ---- Dummy Tests ---- #
    dummy_local_dir = (Path(__file__).parent / "dummy" / "chronos2").absolute()
    dummy_max_context_length = 21
    dummy_max_prediction_length = 77
    series = linear_timeseries(length=200, dtype=np.float32, column_name="A")
    past_cov = linear_timeseries(length=200, dtype=np.float32, column_name="B")
    future_cov = linear_timeseries(length=300, dtype=np.float32, column_name="C")

    @pytest.mark.slow
    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_fidelity(self, probabilistic: bool):
        """Test Chronos2Model predictions against original implementation.
        The test passes if the predictions match up to a certain numerical tolerance.
        Original predictions were generated with the following code:

        ```python
        import numpy as np
        import pandas as pd
        from chronos import BaseChronosPipeline, Chronos2Pipeline
        from darts.datasets import ElectricityConsumptionZurichDataset

        # adapted from `20-SKLearnModel-examples` notebook
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
        ts_weather_train, ts_weather_val = ts_weather.split_after(validation_cutoff)
        ts_other_train, ts_other_val = ts_other.split_after(validation_cutoff)

        prediction_length = 1024

        # load the Chronos-2 pipeline
        pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2")

        # make predictions
        quantiles, mean = pipeline.predict_quantiles(
            inputs=[
                {
                    "target": ts_energy_train.values().T,
                    "past_covariates": {
                        c: series[c].values().flatten()
                        for series in [ts_weather_train, ts_other_train]
                        for c in series.components
                    },
                    "future_covariates": {
                        c: ts_weather_val[c].values().flatten()[:prediction_length]
                        for c in ts_weather.components
                    }
                }
            ],
            prediction_length=prediction_length,
            quantile_levels=pipeline.quantiles,
        )

        # convert to numpy array with shape (time, quantile, variables)
        quantiles_np = quantiles[0].cpu().numpy()
        quantiles_np = quantiles_np.transpose(1, 0, 2)

        # save quantiles to a npz file
        np.savez_compressed("chronos2.npz", pred=quantiles_np)
        ```

        Code accessed from https://github.com/amazon-science/chronos-forecasting/commit/93419cfe9fd678b06503b3ce22113f4482c44b6f
        on 5 November 2025.

        """
        # load model
        model = Chronos2Model(
            input_chunk_length=8192,  # maximum context length
            output_chunk_length=self.max_prediction_length,  # maximum prediction length w/o AR
            likelihood=(
                QuantileRegression(quantiles=all_quantiles) if probabilistic else None
            ),
            **tfm_kwargs,
        )
        # fit model w/o fine-tuning
        model.fit(
            series=self.ts_energy_train,
            past_covariates=self.ts_other,
            future_covariates=self.ts_weather,
        )

        # predict on the validation inputs w/ covariates
        pred = model.predict(
            n=self.max_prediction_length,
            past_covariates=self.ts_other,
            future_covariates=self.ts_weather,
            predict_likelihood_parameters=probabilistic,
        )
        assert isinstance(pred, TimeSeries)
        # reshape to (time, variables, quantiles)
        pred_np = pred.values().reshape(
            self.max_prediction_length, self.ts_energy_train.n_components, -1
        )

        # load the original predictions
        path = Path(__file__).parent / "fidelity" / "chronos2.npz"
        original = np.load(path)["pred"]

        if not probabilistic:
            original = original[:, :, [10]]  # median quantile

        # compare predictions to original
        np.testing.assert_allclose(pred_np, original, rtol=1e-5, atol=1e-5)

    def test_creation(self):
        # can use shorter input/output chunk length than max
        model = Chronos2Model(
            input_chunk_length=self.dummy_max_context_length - 5,
            output_chunk_length=self.dummy_max_prediction_length - 5,
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )
        model.fit(series=self.series)
        pred = model.predict(n=10, series=self.series)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10

        # cannot create longer input chunk length than max
        with pytest.raises(
            ValueError, match=r"`input_chunk_length` \d+ cannot be greater"
        ):
            Chronos2Model(
                input_chunk_length=self.dummy_max_context_length + 1,
                output_chunk_length=self.dummy_max_prediction_length,
                local_dir=self.dummy_local_dir,
                **tfm_kwargs,
            )

        # cannot create longer output chunk length than max
        with pytest.raises(ValueError, match=r"`output_chunk_length` \d+ plus"):
            Chronos2Model(
                input_chunk_length=self.dummy_max_context_length,
                output_chunk_length=self.dummy_max_prediction_length + 1,
                local_dir=self.dummy_local_dir,
                **tfm_kwargs,
            )

        # cannot create longer output chunk length + output chunk shift than max
        with pytest.raises(ValueError, match=r"`output_chunk_length` \d+ plus"):
            Chronos2Model(
                input_chunk_length=self.dummy_max_context_length,
                output_chunk_length=self.dummy_max_prediction_length - 1,
                output_chunk_shift=3,
                local_dir=self.dummy_local_dir,
                **tfm_kwargs,
            )

        # cannot use likelihood others than QuantileRegression
        with pytest.raises(ValueError, match="Only QuantileRegression likelihood is"):
            Chronos2Model(
                input_chunk_length=self.dummy_max_context_length,
                output_chunk_length=self.dummy_max_prediction_length,
                likelihood=GaussianLikelihood(),
                local_dir=self.dummy_local_dir,
                **tfm_kwargs,
            )

        # cannot use quantiles other than those used in pre-training
        with pytest.raises(ValueError, match="must be a subset of Chronos-2 quantiles"):
            Chronos2Model(
                input_chunk_length=self.dummy_max_context_length,
                output_chunk_length=self.dummy_max_prediction_length,
                likelihood=QuantileRegression(quantiles=[0.23, 0.5, 0.77]),
                local_dir=self.dummy_local_dir,
                **tfm_kwargs,
            )

    def test_default(self):
        # default model is deterministic
        model = Chronos2Model(
            input_chunk_length=self.dummy_max_context_length,
            output_chunk_length=self.dummy_max_prediction_length,
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )

        # calling `fit()` should not use `trainer.fit()`
        with patch("pytorch_lightning.Trainer.fit") as mock_fit:
            model.fit(series=self.series)
            mock_fit.assert_not_called()
        assert model.model_created
        assert not model.supports_probabilistic_prediction

        # predictions should not be probabilistic
        pred = model.predict(n=10, series=self.series)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == 1

        # default model allows autoregressive predictions
        pred_ar = model.predict(
            n=self.dummy_max_prediction_length + 10, series=self.series
        )
        assert isinstance(pred_ar, TimeSeries)
        assert len(pred_ar) == self.dummy_max_prediction_length + 10
        assert pred_ar.n_components == 1

    def test_probabilistic(self):
        # probabilistic model
        model = Chronos2Model(
            input_chunk_length=self.dummy_max_context_length,
            output_chunk_length=self.dummy_max_prediction_length,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )

        # calling `fit()` should not use `trainer.fit()`
        with patch("pytorch_lightning.Trainer.fit") as mock_fit:
            model.fit(series=self.series)
            mock_fit.assert_not_called()
        assert model.model_created
        assert model.supports_probabilistic_prediction

        # predictions should be probabilistic
        pred = model.predict(
            n=10, series=self.series, predict_likelihood_parameters=True
        )
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == 3  # 3 quantiles

        # probabilistic model allows autoregressive predictions
        pred_ar = model.predict(
            n=self.dummy_max_prediction_length + 10,
            series=self.series,
            num_samples=10,
        )
        assert isinstance(pred_ar, TimeSeries)
        assert len(pred_ar) == self.dummy_max_prediction_length + 10
        assert pred_ar.n_components == 1  # sampling yields single component
        assert pred_ar.n_samples == 10

    def test_multivariate(self):
        # create multivariate time series
        series_multi = concatenate(
            [
                linear_timeseries(length=200, dtype=np.float32, column_name="A"),
                sine_timeseries(length=200, dtype=np.float32, column_name="B"),
                gaussian_timeseries(length=200, dtype=np.float32, column_name="C"),
            ],
            axis=1,
        )
        # create model
        model = Chronos2Model(
            input_chunk_length=self.dummy_max_context_length,
            output_chunk_length=self.dummy_max_prediction_length,
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )
        model.fit(series=series_multi)
        pred = model.predict(n=15, series=series_multi)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 15
        assert pred.n_components == 3

        # create probabilistic model
        model_prob = Chronos2Model(
            input_chunk_length=self.dummy_max_context_length,
            output_chunk_length=self.dummy_max_prediction_length,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )
        model_prob.fit(series=series_multi)
        pred_prob = model_prob.predict(
            n=15, series=series_multi, predict_likelihood_parameters=True
        )
        assert isinstance(pred_prob, TimeSeries)
        assert len(pred_prob) == 15
        assert pred_prob.n_components == 9  # 3 variables x 3 quantiles

    def test_past_covariates(self):
        model = Chronos2Model(
            input_chunk_length=self.dummy_max_context_length,
            output_chunk_length=self.dummy_max_prediction_length,
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )
        model.fit(series=self.series, past_covariates=self.past_cov)
        pred = model.predict(n=10, series=self.series, past_covariates=self.past_cov)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10
        assert pred.n_components == 1

    def test_future_covariates(self):
        model = Chronos2Model(
            input_chunk_length=self.dummy_max_context_length,
            output_chunk_length=self.dummy_max_prediction_length,
            local_dir=self.dummy_local_dir,
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
        model = Chronos2Model(
            input_chunk_length=self.dummy_max_context_length,
            output_chunk_length=self.dummy_max_prediction_length,
            local_dir=self.dummy_local_dir,
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
