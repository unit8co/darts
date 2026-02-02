from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from darts.tests.conftest import TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

from darts import TimeSeries, concatenate
from darts.datasets import ElectricityConsumptionZurichDataset
from darts.models import TimesFM2p5Model
from darts.tests.conftest import tfm_kwargs
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)

# quantiles used during TimesFM 2.5 pre-training
all_quantiles = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]


def load_validation_inputs():
    """Load validation inputs for TimesFM2p5Model fidelity tests. The data imports
    here are adapted from the `20-SKLearnModel-examples` notebook.
    """
    # convert to float32 due to MPS not supporting float64
    ts_energy = ElectricityConsumptionZurichDataset().load().astype(np.float32)

    # extract households energy consumption
    ts_energy = ts_energy[["Value_NE5", "Value_NE7"]]

    # create train and validation splits
    validation_cutoff = pd.Timestamp("2022-01-01")
    ts_energy_train, ts_energy_val = ts_energy.split_after(validation_cutoff)
    return ts_energy_train, ts_energy_val


class TestTimesFM2p5Model:
    # set random seed
    np.random.seed(42)

    # ---- Fidelity Tests ---- #
    # load validation inputs once for fidelity tests
    ts_energy_train, ts_energy_val = load_validation_inputs()
    # maximum context (input_chunk_length + output_chunk_length + output_chunk_shift)
    context_limit = 16384
    # maximum prediction length of the chosen output head
    max_prediction_length = 1024

    # ---- Dummy Tests ---- #
    # univariate time series
    series = linear_timeseries(length=200, dtype=np.float32, column_name="A")
    past_cov = linear_timeseries(length=200, dtype=np.float32, column_name="B")
    future_cov = linear_timeseries(length=300, dtype=np.float32, column_name="C")
    # multivariate time series
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

    @pytest.mark.slow
    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_fidelity(self, probabilistic: bool):
        """Test TimesFM2p5Model predictions against original implementation.
        The test passes if the predictions match up to a certain numerical tolerance.
        Original predictions were generated with the following code:

        ```python
        import numpy as np
        import pandas as pd
        import timesfm
        import torch

        from darts.datasets import ElectricityConsumptionZurichDataset

        # adapted from `20-SKLearnModel-examples` notebook
        ts_energy = ElectricityConsumptionZurichDataset().load().astype(np.float32)

        # extract households energy consumption
        ts_energy = ts_energy[["Value_NE5", "Value_NE7"]]

        # create train and validation splits
        validation_cutoff = pd.Timestamp("2022-01-01")
        ts_energy_train, ts_energy_val = ts_energy.split_after(validation_cutoff)

        # set torch precision
        torch.set_float32_matmul_precision("high")

        # load TimesFM 2.5 model
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

        # compile the model with forecast configuration
        model.compile(
            timesfm.ForecastConfig(
                max_context=1024,
                max_horizon=1024,
                normalize_inputs=False,
                use_continuous_quantile_head=False,
                force_flip_invariance=False,
                infer_is_positive=False,
                fix_quantile_crossing=False,
            )
        )
        # setting horizon to 1024 allows us to get maximum prediction length of the larger head
        # but will trigger autoregressive predictions internally
        point_forecast, quantile_forecast = model.forecast(
            horizon=1024, # use output_quantile_len here to maximize prediction length
            inputs=[
                series
                for series in ts_energy_train.values().T
            ], # Two energy consumption series
        )

        # convert to numpy array with shape (time, variables, quantiles)
        quantile_forecast = quantile_forecast.transpose(1, 0, 2)
        # exclude the mean forecast (first quantile)
        quantile_forecast = quantile_forecast[:, :, 1:]

        # save quantiles to a npz file
        np.savez_compressed("timesfm2p5.npz", pred=quantile_forecast)
        ```

        Code accessed from https://github.com/google-research/timesfm/commit/6bd8044275f8b76cdc9554f2fecccac5f31a156c
        on 26th December 2025.

        """
        # load model
        model = TimesFM2p5Model(
            input_chunk_length=1024,  # maximum context length
            output_chunk_length=self.max_prediction_length,  # maximum prediction length w/o AR
            likelihood=(
                QuantileRegression(quantiles=all_quantiles) if probabilistic else None
            ),
            **tfm_kwargs,
        )
        # fit model w/o fine-tuning
        model.fit(series=self.ts_energy_train)

        # predict on the validation inputs w/ covariates
        pred = model.predict(
            n=self.max_prediction_length,
            predict_likelihood_parameters=probabilistic,
        )
        assert isinstance(pred, TimeSeries)
        # reshape to (time, variables, quantiles)
        pred_np = pred.values().reshape(
            self.max_prediction_length, self.ts_energy_train.n_components, -1
        )

        # load the original predictions
        path = (
            Path(__file__).parent
            / "artefacts"
            / "timesfm2p5"
            / "timesfm2p5_prediction"
            / "timesfm2p5.npz"
        )
        original: np.ndarray = np.load(path)["pred"]

        if not probabilistic:
            original = original[:, :, [4]]  # median quantile

        original_mean = original.mean(axis=0, keepdims=True)
        normalized_deviation = np.abs(pred_np - original) / original_mean

        # check that normalized deviation is close to zero
        np.testing.assert_allclose(
            normalized_deviation,
            np.zeros_like(normalized_deviation),
            rtol=1e-3,
        )

    @pytest.mark.slow
    def test_creation(self):
        # can use shorter input/output chunk length than max
        model = TimesFM2p5Model(
            input_chunk_length=11,
            output_chunk_length=13,
            **tfm_kwargs,
        )
        model.fit(series=self.series)
        pred = model.predict(n=10, series=self.series)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10

        # cannot create longer input chunk length than max
        with pytest.raises(ValueError, match=r"`input_chunk_length` \d+ plus"):
            TimesFM2p5Model(
                input_chunk_length=self.context_limit,
                output_chunk_length=11,
                **tfm_kwargs,
            )

        # cannot create longer output chunk length than max
        with pytest.raises(ValueError, match=r"`output_chunk_length` \d+ plus"):
            TimesFM2p5Model(
                input_chunk_length=19,
                output_chunk_length=self.max_prediction_length + 1,
                **tfm_kwargs,
            )

        # cannot create longer output chunk length + output chunk shift than max
        with pytest.raises(ValueError, match=r"`output_chunk_length` \d+ plus"):
            TimesFM2p5Model(
                input_chunk_length=23,
                output_chunk_length=self.max_prediction_length - 1,
                output_chunk_shift=3,
                **tfm_kwargs,
            )

        # cannot use likelihood others than QuantileRegression
        with pytest.raises(ValueError, match="Only QuantileRegression likelihood is"):
            TimesFM2p5Model(
                input_chunk_length=29,
                output_chunk_length=12,
                likelihood=GaussianLikelihood(),
                **tfm_kwargs,
            )

        # cannot use quantiles other than those used in pre-training
        with pytest.raises(
            ValueError, match="must be a subset of TimesFM 2.5 quantiles"
        ):
            TimesFM2p5Model(
                input_chunk_length=7,
                output_chunk_length=6,
                likelihood=QuantileRegression(quantiles=[0.23, 0.5, 0.77]),
                **tfm_kwargs,
            )

    @pytest.mark.slow
    def test_default(self):
        # default model is deterministic
        model = TimesFM2p5Model(
            input_chunk_length=3,
            output_chunk_length=4,
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

        # default model allows autoregressive predictions (6 > 4)
        pred_ar = model.predict(n=6, series=self.series)
        assert isinstance(pred_ar, TimeSeries)
        assert len(pred_ar) == 6
        assert pred_ar.n_components == 1

    @pytest.mark.slow
    def test_probabilistic(self):
        # probabilistic model
        model = TimesFM2p5Model(
            input_chunk_length=5,
            output_chunk_length=6,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
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
        model = TimesFM2p5Model(
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

    def test_covariates(self):
        model = TimesFM2p5Model(
            input_chunk_length=21,
            output_chunk_length=23,
            **tfm_kwargs,
        )

        # past covariates are not supported
        with pytest.raises(ValueError, match="does not support `past_covariates`"):
            model.fit(series=self.series, past_covariates=self.past_cov)

        # future covariates are not supported
        with pytest.raises(ValueError, match="does not support `future_covariates`"):
            model.fit(series=self.series, future_covariates=self.future_cov)

        # past and future covariates are not supported
        with pytest.raises(
            ValueError, match="does not support `past_covariates`, `future_covariates`"
        ):
            model.fit(
                series=self.series,
                past_covariates=self.past_cov,
                future_covariates=self.future_cov,
            )

    @pytest.mark.slow
    def test_multiple_series(self):
        # create model
        model = TimesFM2p5Model(
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
