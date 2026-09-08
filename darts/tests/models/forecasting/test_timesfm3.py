from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

from darts import TimeSeries, concatenate
from darts.datasets import ElectricityConsumptionZurichDataset
from darts.models import TimesFM3Model
from darts.tests.models.forecasting.foundation_test_utils import TIMESFM3_TINY_DIR
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)

# quantiles used during TimesFM 3.0 pre-training
all_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_validation_inputs():
    """Load validation inputs for TimesFM3Model fidelity tests. The data imports
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


def generate_series(n_variables: int, length: int, prefix: str):
    return concatenate(
        [
            linear_timeseries(
                length=length, dtype=np.float32, column_name=f"{prefix}_{i}"
            )
            for i in range(n_variables)
        ],
        axis=1,
    )


def series_with_nans(series: TimeSeries, start: int, end: int | None) -> TimeSeries:
    """Returns a copy of `series` with the values in [start, end) set to NaN."""
    values = series.values(copy=True).astype(np.float32)
    values[start:end, :] = np.nan
    return TimeSeries.from_times_and_values(series.time_index, values)


class TestTimesFM3Model:
    # set random seed
    np.random.seed(42)

    # ---- Fidelity Tests ---- #
    # load validation inputs once for fidelity tests
    ts_energy_train, ts_energy_val = load_validation_inputs()
    # maximum context length of the model
    max_context_length = 15360
    # maximum prediction length supported in Darts (upstream has no hard limit,
    # but Darts caps `output_chunk_length + output_chunk_shift` at 1024)
    max_prediction_length = 1024

    # ---- Dummy Tests ---- #
    # univariate time series
    series = linear_timeseries(length=200, dtype=np.float32, column_name="A")
    # past covariates are longer than the series: with autoregressive predictions
    # (n > output_chunk_length) they must extend beyond the end of the series
    past_cov = linear_timeseries(length=210, dtype=np.float32, column_name="B")
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
        """Test TimesFM3Model predictions against original implementation.
        The test passes if the predictions match up to a certain numerical tolerance.
        Original predictions were generated with the following code:

        ```python
        import numpy as np
        import pandas as pd

        from darts.datasets import ElectricityConsumptionZurichDataset
        from timesfm3 import ModelConfig, TimesFM3Forecaster

        # adapted from `20-SKLearnModel-examples` notebook
        ts_energy = ElectricityConsumptionZurichDataset().load().astype(np.float32)

        # extract households energy consumption
        ts_energy = ts_energy[["Value_NE5", "Value_NE7"]]

        # create train and validation splits
        validation_cutoff = pd.Timestamp("2022-01-01")
        ts_energy_train, ts_energy_val = ts_energy.split_after(validation_cutoff)

        # load TimesFM 3.0 forecaster with the original implementation
        forecaster = TimesFM3Forecaster(
            config=ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", device="cpu")
        )

        # forecast with a multivariate context of the two energy consumption series,
        # truncated to the last 1024 points to match the Darts `input_chunk_length`
        context = ts_energy_train.values().T[:, -1024:].astype(np.float32)
        out = list(
            forecaster.predict_batch(
                contexts=[context],
                horizon=128,
                return_quantiles=True,
                sort_quantiles=False,
            )
        )[0]

        # convert to numpy array with shape (time, variables, quantiles)
        quantile_forecast = out.quantiles.transpose(1, 0, 2)

        # save quantiles to a npz file
        np.savez_compressed("timesfm3.npz", pred=quantile_forecast)
        ```

        Code accessed from https://github.com/google-research/timesfm/commit/9de33f6f487baf8adc26eb757f29b6a7a557b823
        on 8th September 2026.

        """
        # load model
        model = TimesFM3Model(
            input_chunk_length=1024,  # context length used to generate the reference
            output_chunk_length=128,  # prevent auto-regressive forecasting
            accept_license=True,
            likelihood=(
                QuantileRegression(quantiles=all_quantiles) if probabilistic else None
            ),
            **tfm_kwargs,
        )
        # fit model w/o fine-tuning
        model.fit(series=self.ts_energy_train)

        # predict on the validation inputs
        pred = model.predict(
            n=128,
            predict_likelihood_parameters=probabilistic,
        )
        assert isinstance(pred, TimeSeries)
        # reshape to (time, variables, quantiles)
        pred_np = pred.values().reshape(128, self.ts_energy_train.n_components, -1)

        # load the original predictions
        path = (
            Path(__file__).parent
            / "artefacts"
            / "timesfm3"
            / "timesfm3_prediction"
            / "timesfm3.npz"
        )
        original = np.load(path)["pred"]

        if not probabilistic:
            original = original[:, :, [4]]  # median quantile

        # compare predictions to original
        np.testing.assert_allclose(pred_np, original, rtol=1e-5, atol=1e-5)

    def test_creation(self):
        # cannot create model without accepting the non-commercial license
        with pytest.raises(ValueError, match="accept_license"):
            TimesFM3Model(
                input_chunk_length=8,
                output_chunk_length=4,
            )

        # can create a valid model with the tiny artefact
        model = TimesFM3Model(
            input_chunk_length=8,
            output_chunk_length=4,
            accept_license=True,
            local_dir=TIMESFM3_TINY_DIR,
            **tfm_kwargs,
        )
        model.fit(series=self.series)
        pred = model.predict(n=4, series=self.series)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 4

        # cannot create longer input chunk length than the context limit
        with pytest.raises(ValueError, match="cannot be greater than model's"):
            TimesFM3Model(
                input_chunk_length=self.max_context_length + 1,
                output_chunk_length=4,
                accept_license=True,
            )

        # cannot create longer output chunk length than max
        with pytest.raises(ValueError, match=r"`output_chunk_length` \d+ plus"):
            TimesFM3Model(
                input_chunk_length=8,
                output_chunk_length=self.max_prediction_length + 1,
                accept_license=True,
                local_dir=TIMESFM3_TINY_DIR,
            )

        # cannot create longer output chunk length + output chunk shift than max
        with pytest.raises(ValueError, match=r"`output_chunk_length` \d+ plus"):
            TimesFM3Model(
                input_chunk_length=8,
                output_chunk_length=self.max_prediction_length - 1,
                output_chunk_shift=3,
                accept_license=True,
                local_dir=TIMESFM3_TINY_DIR,
            )

        # cannot use likelihood others than QuantileRegression
        with pytest.raises(ValueError, match="Only QuantileRegression likelihood is"):
            TimesFM3Model(
                input_chunk_length=8,
                output_chunk_length=4,
                likelihood=GaussianLikelihood(),
                accept_license=True,
                local_dir=TIMESFM3_TINY_DIR,
            )

        # cannot use quantiles other than those used in pre-training
        with pytest.raises(
            ValueError, match="must be a subset of TimesFM 3.0 quantiles"
        ):
            TimesFM3Model(
                input_chunk_length=8,
                output_chunk_length=4,
                likelihood=QuantileRegression(quantiles=[0.23, 0.5, 0.77]),
                accept_license=True,
                local_dir=TIMESFM3_TINY_DIR,
            )

        # cannot enable fine-tuning (not supported yet)
        with pytest.raises(ValueError, match="Fine-tuning is not yet supported"):
            TimesFM3Model(
                input_chunk_length=8,
                output_chunk_length=4,
                accept_license=True,
                enable_finetuning=True,
                local_dir=TIMESFM3_TINY_DIR,
            )

    def test_default(self):
        # default model is deterministic
        model = TimesFM3Model(
            input_chunk_length=8,
            output_chunk_length=4,
            accept_license=True,
            local_dir=TIMESFM3_TINY_DIR,
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

    def test_probabilistic(self):
        # probabilistic model
        model = TimesFM3Model(
            input_chunk_length=8,
            output_chunk_length=6,
            likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
            accept_license=True,
            local_dir=TIMESFM3_TINY_DIR,
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

    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_multivariate(self, probabilistic: bool):
        model = TimesFM3Model(
            input_chunk_length=8,
            output_chunk_length=8,
            accept_license=True,
            local_dir=TIMESFM3_TINY_DIR,
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
        model = TimesFM3Model(
            input_chunk_length=8,
            output_chunk_length=4,
            accept_license=True,
            local_dir=TIMESFM3_TINY_DIR,
            **tfm_kwargs,
        )
        model.fit(series=self.series, past_covariates=self.past_cov)
        pred = model.predict(n=6, series=self.series, past_covariates=self.past_cov)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 6
        assert pred.n_components == 1
        assert not np.isnan(pred.all_values(copy=False)).any()

    def test_future_covariates(self):
        model = TimesFM3Model(
            input_chunk_length=8,
            output_chunk_length=4,
            accept_license=True,
            local_dir=TIMESFM3_TINY_DIR,
            **tfm_kwargs,
        )
        model.fit(series=self.series, future_covariates=self.future_cov)
        pred = model.predict(n=6, series=self.series, future_covariates=self.future_cov)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 6
        assert pred.n_components == 1
        assert not np.isnan(pred.all_values(copy=False)).any()

    def test_past_and_future_covariates(self):
        model = TimesFM3Model(
            input_chunk_length=8,
            output_chunk_length=4,
            accept_license=True,
            local_dir=TIMESFM3_TINY_DIR,
            **tfm_kwargs,
        )
        model.fit(
            series=self.series,
            past_covariates=self.past_cov,
            future_covariates=self.future_cov,
        )
        pred = model.predict(
            n=6,
            series=self.series,
            past_covariates=self.past_cov,
            future_covariates=self.future_cov,
        )
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 6
        assert pred.n_components == 1
        assert not np.isnan(pred.all_values(copy=False)).any()

    def test_missing_values(self):
        """NaNs in target and covariates are handled via the masking logic of the
        ported `decode()`, instead of the linear interpolation applied by the
        upstream `TimesFM3Forecaster.predict_batch()`. Predictions must not
        contain NaNs for any of the missing value locations.
        """

        def make_model() -> TimesFM3Model:
            return TimesFM3Model(
                input_chunk_length=8,
                output_chunk_length=4,
                accept_license=True,
                local_dir=TIMESFM3_TINY_DIR,
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
        past_cov_nan = series_with_nans(self.past_cov, 5, 10)
        model = make_model()
        model.fit(series=self.series, past_covariates=past_cov_nan)
        pred = model.predict(n=4, series=self.series, past_covariates=past_cov_nan)
        assert isinstance(pred, TimeSeries)
        assert not np.isnan(pred.all_values(copy=False)).any()

        # NaNs in the future covariates
        future_cov_nan = series_with_nans(self.future_cov, 204, 208)
        model = make_model()
        model.fit(series=self.series, future_covariates=future_cov_nan)
        pred = model.predict(n=4, series=self.series, future_covariates=future_cov_nan)
        assert isinstance(pred, TimeSeries)
        assert not np.isnan(pred.all_values(copy=False)).any()

    def test_output_chunk_shift(self):
        """With `output_chunk_shift`, predictions start after the shifted gap and
        auto-regressive prediction (`n > output_chunk_length`) is not allowed."""
        model = TimesFM3Model(
            input_chunk_length=8,
            output_chunk_length=4,
            output_chunk_shift=2,
            accept_license=True,
            local_dir=TIMESFM3_TINY_DIR,
            **tfm_kwargs,
        )
        model.fit(series=self.series, future_covariates=self.future_cov)
        pred = model.predict(n=4, series=self.series, future_covariates=self.future_cov)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 4
        # predictions start `output_chunk_shift + 1` steps after the end of the series
        assert pred.start_time() == self.series.end_time() + self.series.freq * 3
        assert not np.isnan(pred.all_values(copy=False)).any()

        # auto-regression is not allowed with an output chunk shift
        with pytest.raises(ValueError, match="output_chunk_shift > 0"):
            model.predict(n=5, series=self.series, future_covariates=self.future_cov)

    def test_too_many_variates(self):
        def make_model() -> TimesFM3Model:
            return TimesFM3Model(
                input_chunk_length=8,
                output_chunk_length=4,
                accept_license=True,
                local_dir=TIMESFM3_TINY_DIR,
                **tfm_kwargs,
            )

        # 32 target components (the maximum supported) work
        series_max = generate_series(n_variables=32, length=64, prefix="V")
        model = make_model()
        model.fit(series=series_max)
        pred = model.predict(n=4, series=series_max)
        assert pred.n_components == 32

        # 33 target components exceed the 32 variates supported by the checkpoint;
        # the number of variates is validated at fit time
        series = generate_series(n_variables=33, length=64, prefix="V")
        model = make_model()
        with pytest.raises(ValueError, match="maximum number of variates"):
            model.fit(series=series)

    def test_multiple_series(self):
        model = TimesFM3Model(
            input_chunk_length=8,
            output_chunk_length=4,
            accept_license=True,
            local_dir=TIMESFM3_TINY_DIR,
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
