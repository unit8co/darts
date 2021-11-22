import numpy as np
import pandas as pd
import unittest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .base_test_class import DartsBaseTestClass
from ..utils import timeseries_generation as tg
from ..models import NaiveDrift, NaiveSeasonal
from ..logging import get_logger
from .test_ensemble_models import _make_ts
from ..metrics import rmse
from .. import TimeSeries
from .test_regression_models import train_test_split
logger = get_logger(__name__)

try:
    import torch
    from darts.models import RNNModel, BlockRNNModel
    from darts.models import RegressionEnsembleModel, LinearRegressionModel, RandomForest, RegressionModel
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. Some tests will be skipped.")
    TORCH_AVAILABLE = False


class RegressionEnsembleModelsTestCase(DartsBaseTestClass):

    RANDOM_SEED = 111

    sine_series = tg.sine_timeseries(value_frequency=(1 / 5), value_y_offset=10, length=50)
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
        return [RNNModel(input_chunk_length=20, output_chunk_length=output_chunk_length, n_epochs=1,
                         random_state=42),
                BlockRNNModel(input_chunk_length=20, output_chunk_length=output_chunk_length, n_epochs=1,
                              random_state=42),
                ]

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
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

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
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

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_train_n_points(self):
        regr = LinearRegressionModel(lags_future_covariates=[0])

        # same values
        ensemble = RegressionEnsembleModel(self.get_local_models(), 5, regr)

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
        model1 = BlockRNNModel(input_chunk_length=12, output_chunk_length=1, random_state=0, n_epochs=2)
        model2 = BlockRNNModel(input_chunk_length=12, output_chunk_length=1, random_state=0, n_epochs=2)

        ensemble = RegressionEnsembleModel([model1], 5)
        ensemble.fit(self.combined)

        model1_fitted = ensemble.models[0]
        forecast1 = model1_fitted.predict(10)

        model2.fit(self.combined)
        forecast2 = model2.predict(10)

        self.assertAlmostEqual(sum(forecast1.values() - forecast2.values())[0], 0., places=2)

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

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def helper_test_models_accuracy(self, model_instance, n, series, past_covariates, min_rmse):
        # for every model, test whether it predicts the target with a minimum r2 score of `min_rmse`
        train_f, train_t, test_f, test_t = train_test_split(past_covariates, series, pd.Timestamp("20010101"))

        model_instance.fit(series=train_t, past_covariates=train_f)
        prediction = model_instance.predict(n=n, past_covariates=past_covariates)
        current_rmse = rmse(prediction, test_t)

        self.assertTrue(
            current_rmse <= min_rmse,
            f"Model was not able to denoise data. A rmse score of {current_rmse} was recorded."
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
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
        horizon = 10
        ts_sum1, ts_cov1, _, _ = self.denoising_input()
        torch.manual_seed(self.RANDOM_SEED)

        ensemble_models = [
            RNNModel(input_chunk_length=20, output_chunk_length=horizon, n_epochs=1, random_state=self.RANDOM_SEED),
            BlockRNNModel(input_chunk_length=20, output_chunk_length=horizon, n_epochs=1, random_state=self.RANDOM_SEED),
            RegressionModel(lags=1, lags_past_covariates=[-1])
        ]

        ensemble = RegressionEnsembleModel(ensemble_models, horizon)
        self.helper_test_models_accuracy(ensemble, horizon, ts_sum1, ts_cov1, 1.5)

    @unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
    def test_ensemble_models_denoising_multi_input(self):
        # for every model, test whether it correctly denoises ts_sum_2 using ts_random_multi and ts_sum_2 as inputs
        horizon = 10
        _, _, ts_sum2, ts_cov2 = self.denoising_input()
        torch.manual_seed(self.RANDOM_SEED)

        ensemble_models = [
            RNNModel(input_chunk_length=20, output_chunk_length=horizon, n_epochs=1, random_state=self.RANDOM_SEED),
            BlockRNNModel(input_chunk_length=20, output_chunk_length=horizon, n_epochs=1, random_state=self.RANDOM_SEED),
            RegressionModel(lags=1, lags_past_covariates=[-1]), RegressionModel(lags=1, lags_past_covariates=[-1])
        ]

        ensemble = RegressionEnsembleModel(ensemble_models, horizon)
        self.helper_test_models_accuracy(ensemble, horizon, ts_sum2, ts_cov2, 1.9)