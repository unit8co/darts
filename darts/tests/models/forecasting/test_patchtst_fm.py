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
from darts.models import PatchTSTFMModel
from darts.tests.models.forecasting.foundation_test_utils import (
    PATCHTST_FM_TINY_CONTEXT_LENGTH,
    PATCHTST_FM_TINY_DIR,
)
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)


def load_validation_inputs():
    """Load validation inputs for PatchTSTFMModel fidelity tests. The data imports
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


class TestPatchTSTFMModel:
    # set random seed
    np.random.seed(42)

    # ---- Fidelity Tests ---- #
    # load validation inputs once for fidelity tests
    ts_energy_train, ts_energy_val = load_validation_inputs()

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

    # ---- Tiny Model ---- #
    dummy_local_dir = PATCHTST_FM_TINY_DIR
    dummy_context_length = PATCHTST_FM_TINY_CONTEXT_LENGTH

    @pytest.mark.slow
    @pytest.mark.parametrize("probabilistic", [True, False])
    def test_fidelity(self, probabilistic: bool):
        """Test PatchTSTFMModel predictions against reference output.
        The reference was generated using the official granite-tsfm package.

        ```bash
        pip install granite-tsfm==0.3.6
        ```

        Reference generation code:

        ```python
        import numpy as np
        import pandas as pd

        from darts.datasets import ElectricityConsumptionZurichDataset

        from tsfm_public import PatchTSTFMForPrediction, TimeSeriesForecastingPipeline

        timestamp_column = "Timestamp"
        target_columns = ["Value_NE5", "Value_NE7"]
        prediction_length = 1024
        context_length = 4096

        ts_energy = ElectricityConsumptionZurichDataset().load().astype(np.float32)

        # extract households energy consumption
        validation_cutoff = pd.Timestamp("2022-01-01")
        ts_energy = ts_energy[target_columns]
        ts_energy_train, ts_energy_val = ts_energy.split_after(validation_cutoff)
        ts_energy_train_df = ts_energy_train[-context_length:].to_dataframe(time_as_index=False)

        # load the PatchTSTFM pipeline
        model = PatchTSTFMForPrediction.from_pretrained("ibm-granite/granite-timeseries-patchtst-fm-r1")
        pipeline = TimeSeriesForecastingPipeline(
            model=model,
            id_columns=[],
            timestamp_column=timestamp_column,
            target_columns=target_columns,
            max_context_length=context_length,
            context_length=context_length,
            prediction_length=prediction_length,
            device="cpu",
            quantile_levels=[0.1, 0.5, 0.9],
        )

        # make predictions
        forecast = pipeline(ts_energy_train_df)

        # convert to numpy array with shape (time, variables, quantiles)
        forecast_cols = []
        for col in target_columns:
            forecast_col = forecast.loc[:, forecast.columns.str.startswith(f"{col}_prediction_q")].iloc[0]
            forecast_col = np.vstack(forecast_col.tolist()).T[:, np.newaxis, :]
            forecast_cols.append(forecast_col)
        forecast = np.concatenate(forecast_cols, axis=1)

        # save quantiles to a npz file
        np.savez_compressed("patchtstfm.npz", pred=forecast)
        ```
        """
        input_chunk_length = 4096
        output_chunk_length = 1024
        quantiles = [0.1, 0.5, 0.9]

        # load model
        model = PatchTSTFMModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            likelihood=(
                QuantileRegression(quantiles=quantiles) if probabilistic else None
            ),
            **tfm_kwargs,
        )
        # fit model w/o fine-tuning
        model.fit(series=self.ts_energy_train)

        # predict on the validation inputs w/ covariates
        pred = model.predict(
            n=output_chunk_length,
            predict_likelihood_parameters=probabilistic,
        )
        assert isinstance(pred, TimeSeries)
        # reshape to (time, variables, quantiles)
        pred_np = pred.values().reshape(
            output_chunk_length, self.ts_energy_train.n_components, -1
        )

        # load the original predictions
        path = (
            Path(__file__).parent
            / "artefacts"
            / "patchtstfm"
            / "patchtstfm_prediction"
            / "patchtstfm.npz"
        )
        original = np.load(path)["pred"]

        if not probabilistic:
            original = original[:, :, [1]]  # median quantile

        # compare predictions to original
        np.testing.assert_allclose(pred_np, original, rtol=1e-5, atol=1e-5)

    @pytest.mark.slow
    def test_creation(self):
        # can use shorter input/output chunk length than max
        model = PatchTSTFMModel(
            input_chunk_length=11,
            output_chunk_length=13,
            local_dir=self.dummy_local_dir,
            **tfm_kwargs,
        )
        model.fit(series=self.series)
        pred = model.predict(n=10, series=self.series)
        assert isinstance(pred, TimeSeries)
        assert len(pred) == 10

        # (icl + ocl + ocs) must be <= context length
        with pytest.raises(ValueError, match=r"`input_chunk_length` \d+ plus"):
            PatchTSTFMModel(
                input_chunk_length=self.dummy_context_length,
                output_chunk_length=1,
                local_dir=self.dummy_local_dir,
                **tfm_kwargs,
            )

        # (icl + ocl + ocs) must be <= context length
        with pytest.raises(ValueError, match=r"`input_chunk_length` \d+ plus"):
            PatchTSTFMModel(
                input_chunk_length=self.dummy_context_length - 1,
                output_chunk_length=1,
                output_chunk_shift=1,
                local_dir=self.dummy_local_dir,
                **tfm_kwargs,
            )

        # cannot use likelihood others than QuantileRegression
        with pytest.raises(ValueError, match="Only QuantileRegression likelihood is"):
            PatchTSTFMModel(
                input_chunk_length=29,
                output_chunk_length=12,
                likelihood=GaussianLikelihood(),
                local_dir=self.dummy_local_dir,
                **tfm_kwargs,
            )

        # cannot use quantiles other than those used in pre-training
        with pytest.raises(
            ValueError, match="must be a subset of PatchTST-FM quantiles"
        ):
            PatchTSTFMModel(
                input_chunk_length=7,
                output_chunk_length=6,
                likelihood=QuantileRegression(quantiles=[0.231, 0.5, 0.769]),
                local_dir=self.dummy_local_dir,
                **tfm_kwargs,
            )

    @pytest.mark.slow
    def test_default(self):
        # default model is deterministic
        model = PatchTSTFMModel(
            input_chunk_length=3,
            output_chunk_length=4,
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

        # default model allows autoregressive predictions (6 > 4)
        pred_ar = model.predict(n=6, series=self.series)
        assert isinstance(pred_ar, TimeSeries)
        assert len(pred_ar) == 6
        assert pred_ar.n_components == 1

    @pytest.mark.slow
    def test_probabilistic(self):
        # probabilistic model
        model = PatchTSTFMModel(
            input_chunk_length=5,
            output_chunk_length=6,
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
        model = PatchTSTFMModel(
            input_chunk_length=3,
            output_chunk_length=8,
            likelihood=(
                QuantileRegression(quantiles=[0.1, 0.5, 0.9]) if probabilistic else None
            ),
            local_dir=self.dummy_local_dir,
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
        model = PatchTSTFMModel(
            input_chunk_length=21,
            output_chunk_length=23,
            local_dir=self.dummy_local_dir,
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
        model = PatchTSTFMModel(
            input_chunk_length=2,
            output_chunk_length=3,
            local_dir=self.dummy_local_dir,
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
