import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from darts import TimeSeries
from darts.logging import get_logger
from darts.metrics import rmse
from darts.models import (
    LinearRegressionModel,
    NaiveDrift,
    NaiveSeasonal,
    RandomForest,
    RegressionEnsembleModel,
    RegressionModel,
)
from darts.tests.base_test_class import DartsBaseTestClass
from darts.tests.models.forecasting.test_ensemble_models import _make_ts
from darts.tests.models.forecasting.test_regression_models import train_test_split
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
    import torch

    from darts.models import BlockRNNModel, RNNModel

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. Some tests will be skipped.")
    TORCH_AVAILABLE = False


class RegressionEnsembleModelsTestCase(DartsBaseTestClass):

    RANDOM_SEED = 111

    sine_series = tg.sine_timeseries(
        value_frequency=(1 / 5), value_y_offset=10, length=50
    )
    lin_series = tg.linear_timeseries(length=50)

    combined = sine_series + lin_series

    seq1 = [_make_ts(0), _make_ts(10), _make_ts(20)]
    cov1 = [_make_ts(5), _make_ts(15), _make_ts(25)]

    seq2 = [_make_ts(0, 20), _make_ts(10, 20), _make_ts(20, 20)]
    cov2 = [_make_ts(5, 30), _make_ts(15, 30), _make_ts(25, 30)]

    # dummy feature and target TimeSeries instances
    ts_periodic = tg.sine_timeseries(length=500)
    ts_gaussian = tg.gaussian_timeseries(length=500)
    ts_random_walk = tg.random_walk_timeseries(length=500)

    ts_cov1 = ts_periodic.stack(ts_gaussian)
    ts_cov1 = ts_cov1.pd_dataframe()
    ts_cov1.columns = ["Periodic", "Gaussian"]
    ts_cov1 = TimeSeries.from_dataframe(ts_cov1)
    ts_sum1 = ts_periodic + ts_gaussian

    ts_cov2 = ts_sum1.stack(ts_random_walk)
    ts_sum2 = ts_sum1 + ts_random_walk

    def get_local_models(self):
        return [NaiveDrift(), NaiveSeasonal(5), NaiveSeasonal(10)]

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def get_global_models(self, output_chunk_length=5):
        return [
            RNNModel(
                input_chunk_length=20,
                output_chunk_length=output_chunk_length,
                n_epochs=1,
                random_state=42,
            ),
            BlockRNNModel(
                input_chunk_length=20,
                output_chunk_length=output_chunk_length,
                n_epochs=1,
                random_state=42,
            ),
        ]

    @staticmethod
    def get_global_ensembe_model(output_chunk_length=5):
        lags = [-1, -2, -5]
        return RegressionEnsembleModel(
            forecasting_models=[
                LinearRegressionModel(
                    lags=lags,
                    lags_past_covariates=lags,
                    output_chunk_length=output_chunk_length,
                ),
                LinearRegressionModel(
                    lags=lags,
                    lags_past_covariates=lags,
                    output_chunk_length=output_chunk_length,
                ),
            ],
            regression_train_n_points=10,
        )

    def test_accepts_different_regression_models(self):
        regr1 = LinearRegression()
        regr2 = RandomForestRegressor()
        regr3 = RandomForest(lags_future_covariates=[0])

        model0 = RegressionEnsembleModel(self.get_local_models(), 10)
        model1 = RegressionEnsembleModel(self.get_local_models(), 10, regr1)
        model2 = RegressionEnsembleModel(self.get_local_models(), 10, regr2)
        model3 = RegressionEnsembleModel(self.get_local_models(), 10, regr3)

        models = [model0, model1, model2, model3]
        for model in models:
            model.fit(series=self.combined)
            model.predict(10)

    def test_accepts_one_model(self):
        regr1 = LinearRegression()
        regr2 = RandomForest(lags_future_covariates=[0])

        model0 = RegressionEnsembleModel([self.get_local_models()[0]], 10)
        model1 = RegressionEnsembleModel([self.get_local_models()[0]], 10, regr1)
        model2 = RegressionEnsembleModel([self.get_local_models()[0]], 10, regr2)

        models = [model0, model1, model2]
        for model in models:
            model.fit(series=self.combined)
            model.predict(10)

    def test_train_n_points(self):
        regr = LinearRegressionModel(lags_future_covariates=[0])

        # same values
        _ = RegressionEnsembleModel(self.get_local_models(), 5, regr)

        # too big value to perform the split
        ensemble = RegressionEnsembleModel(self.get_local_models(), 100)
        with self.assertRaises(ValueError):
            ensemble.fit(self.combined)

        ensemble = RegressionEnsembleModel(self.get_local_models(), 50)
        with self.assertRaises(ValueError):
            ensemble.fit(self.combined)

        # too big value considering min_train_series_length
        ensemble = RegressionEnsembleModel(self.get_local_models(), 45)
        with self.assertRaises(ValueError):
            ensemble.fit(self.combined)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_torch_models_retrain(self):
        model1 = BlockRNNModel(
            input_chunk_length=12, output_chunk_length=1, random_state=0, n_epochs=2
        )
        model2 = BlockRNNModel(
            input_chunk_length=12, output_chunk_length=1, random_state=0, n_epochs=2
        )

        ensemble = RegressionEnsembleModel([model1], 5)
        ensemble.fit(self.combined)

        model1_fitted = ensemble.models[0]
        forecast1 = model1_fitted.predict(10)

        model2.fit(self.combined)
        forecast2 = model2.predict(10)

        self.assertAlmostEqual(
            sum(forecast1.values() - forecast2.values())[0], 0.0, places=2
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_train_predict_global_models_univar(self):
        ensemble_models = self.get_global_models(output_chunk_length=10)
        ensemble_models.append(RegressionModel(lags=1))
        ensemble = RegressionEnsembleModel(ensemble_models, 10)
        ensemble.fit(series=self.combined)
        ensemble.predict(10)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_train_predict_global_models_multivar_no_covariates(self):
        ensemble_models = self.get_global_models(output_chunk_length=10)
        ensemble_models.append(RegressionModel(lags=1))
        ensemble = RegressionEnsembleModel(ensemble_models, 10)
        ensemble.fit(self.seq1)
        ensemble.predict(10, self.seq1)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_train_predict_global_models_multivar_with_covariates(self):
        ensemble_models = self.get_global_models(output_chunk_length=10)
        ensemble_models.append(RegressionModel(lags=1, lags_past_covariates=[-1]))
        ensemble = RegressionEnsembleModel(ensemble_models, 10)
        ensemble.fit(self.seq1, self.cov1)
        ensemble.predict(10, self.seq2, self.cov2)

    def test_predict_with_target(self):
        series_long = self.combined
        series_short = series_long[:25]

        # train with a single series
        ensemble_model = self.get_global_ensembe_model()
        ensemble_model.fit(series_short, past_covariates=series_long)
        # predict after end of train series
        preds = ensemble_model.predict(n=5, past_covariates=series_long)
        self.assertTrue(isinstance(preds, TimeSeries))
        # predict a new target series
        preds = ensemble_model.predict(
            n=5, series=series_long, past_covariates=series_long
        )
        self.assertTrue(isinstance(preds, TimeSeries))
        # predict multiple target series
        preds = ensemble_model.predict(
            n=5, series=[series_long] * 2, past_covariates=[series_long] * 2
        )
        self.assertTrue(isinstance(preds, list) and len(preds) == 2)
        # predict single target series in list
        preds = ensemble_model.predict(
            n=5, series=[series_long], past_covariates=[series_long]
        )
        self.assertTrue(isinstance(preds, list) and len(preds) == 1)

        # train with multiple series
        ensemble_model = self.get_global_ensembe_model()
        ensemble_model.fit([series_short] * 2, past_covariates=[series_long] * 2)
        with self.assertRaises(ValueError):
            # predict without passing series should raise an error
            ensemble_model.predict(n=5, past_covariates=series_long)
        # predict a new target series
        preds = ensemble_model.predict(
            n=5, series=series_long, past_covariates=series_long
        )
        self.assertTrue(isinstance(preds, TimeSeries))
        # predict multiple target series
        preds = ensemble_model.predict(
            n=5, series=[series_long] * 2, past_covariates=[series_long] * 2
        )
        self.assertTrue(isinstance(preds, list) and len(preds) == 2)
        # predict single target series in list
        preds = ensemble_model.predict(
            n=5, series=[series_long], past_covariates=[series_long]
        )
        self.assertTrue(isinstance(preds, list) and len(preds) == 1)

    def helper_test_models_accuracy(
        self, model_instance, n, series, past_covariates, min_rmse
    ):
        # for every model, test whether it predicts the target with a minimum r2 score of `min_rmse`
        train_series, test_series = train_test_split(series, pd.Timestamp("20010101"))
        train_past_covariates, _ = train_test_split(
            past_covariates, pd.Timestamp("20010101")
        )

        model_instance.fit(series=train_series, past_covariates=train_past_covariates)
        prediction = model_instance.predict(n=n, past_covariates=past_covariates)
        current_rmse = rmse(test_series, prediction)

        self.assertTrue(
            current_rmse <= min_rmse,
            f"Model was not able to denoise data. A rmse score of {current_rmse} was recorded.",
        )

    def denoising_input(self):
        np.random.seed(self.RANDOM_SEED)

        ts_periodic = tg.sine_timeseries(length=500)
        ts_gaussian = tg.gaussian_timeseries(length=500)
        ts_random_walk = tg.random_walk_timeseries(length=500)

        ts_cov1 = ts_periodic.stack(ts_gaussian)
        ts_cov1 = ts_cov1.pd_dataframe()
        ts_cov1.columns = ["Periodic", "Gaussian"]
        ts_cov1 = TimeSeries.from_dataframe(ts_cov1)
        ts_sum1 = ts_periodic + ts_gaussian

        ts_cov2 = ts_sum1.stack(ts_random_walk)
        ts_sum2 = ts_sum1 + ts_random_walk

        return ts_sum1, ts_cov1, ts_sum2, ts_cov2

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_ensemble_models_denoising(self):
        # for every model, test whether it correctly denoises ts_sum using ts_gaussian and ts_sum as inputs
        # WARNING: this test isn't numerically stable, changing self.RANDOM_SEED can lead to exploding coefficients
        horizon = 10
        ts_sum1, ts_cov1, _, _ = self.denoising_input()
        torch.manual_seed(self.RANDOM_SEED)

        ensemble_models = [
            RNNModel(
                input_chunk_length=20,
                output_chunk_length=horizon,
                n_epochs=1,
                random_state=self.RANDOM_SEED,
            ),
            BlockRNNModel(
                input_chunk_length=20,
                output_chunk_length=horizon,
                n_epochs=1,
                random_state=self.RANDOM_SEED,
            ),
            RegressionModel(lags_past_covariates=[-1]),
        ]

        ensemble = RegressionEnsembleModel(ensemble_models, horizon)
        self.helper_test_models_accuracy(ensemble, horizon, ts_sum1, ts_cov1, 3)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_ensemble_models_denoising_multi_input(self):
        # for every model, test whether it correctly denoises ts_sum_2 using ts_random_multi and ts_sum_2 as inputs
        # WARNING: this test isn't numerically stable, changing self.RANDOM_SEED can lead to exploding coefficients
        horizon = 10
        _, _, ts_sum2, ts_cov2 = self.denoising_input()
        torch.manual_seed(self.RANDOM_SEED)

        ensemble_models = [
            RNNModel(
                input_chunk_length=20,
                output_chunk_length=horizon,
                n_epochs=1,
                random_state=self.RANDOM_SEED,
            ),
            BlockRNNModel(
                input_chunk_length=20,
                output_chunk_length=horizon,
                n_epochs=1,
                random_state=self.RANDOM_SEED,
            ),
            RegressionModel(lags_past_covariates=[-1]),
            RegressionModel(lags_past_covariates=[-1]),
        ]

        ensemble = RegressionEnsembleModel(ensemble_models, horizon)
        self.helper_test_models_accuracy(ensemble, horizon, ts_sum2, ts_cov2, 3)
