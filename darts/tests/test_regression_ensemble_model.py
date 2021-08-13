import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .base_test_class import DartsBaseTestClass
from ..utils import timeseries_generation as tg
from ..models import NaiveDrift, NaiveSeasonal
from ..models import RegressionEnsembleModel, LinearRegressionModel, RandomForest
from ..logging import get_logger

logger = get_logger(__name__)

try:
    from ..models import BlockRNNModel
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. Some tests will be skipped.")
    TORCH_AVAILABLE = False


class RegressionEnsembleModelsTestCase(DartsBaseTestClass):
    sine_series = tg.sine_timeseries(value_frequency=(1 / 5), value_y_offset=10, length=50)
    lin_series = tg.linear_timeseries(length=50)

    combined = sine_series + lin_series

    def get_models(self):
        return [NaiveDrift(), NaiveSeasonal(5), NaiveSeasonal(10)]

    def test_accepts_different_regression_models(self):
        regr1 = LinearRegression()
        regr2 = RandomForestRegressor()
        regr3 = RandomForest(lags_future_covariates=[0])

        model0 = RegressionEnsembleModel(self.get_models(), 10)
        model1 = RegressionEnsembleModel(self.get_models(), 10, regr1)
        model2 = RegressionEnsembleModel(self.get_models(), 10, regr2)
        model3 = RegressionEnsembleModel(self.get_models(), 10, regr3)

        models = [model0, model1, model2, model3]
        for model in models:
            model.fit(series=self.combined)
            model.predict(10)

    def test_accepts_one_model(self):
        regr1 = LinearRegression()
        regr2 = RandomForest(lags_future_covariates=[0])

        model0 = RegressionEnsembleModel([self.get_models()[0]], 10)
        model1 = RegressionEnsembleModel([self.get_models()[0]], 10, regr1)
        model2 = RegressionEnsembleModel([self.get_models()[0]], 10, regr2)

        models = [model0, model1, model2]
        for model in models:
            model.fit(series=self.combined)
            model.predict(10)

    def test_train_n_points(self):
        regr = LinearRegressionModel(lags_future_covariates=[0])

        # same values
        ensemble = RegressionEnsembleModel(self.get_models(), 5, regr)

        # too big value to perform the split
        ensemble = RegressionEnsembleModel(self.get_models(), 100)
        with self.assertRaises(ValueError):
            ensemble.fit(self.combined)

        ensemble = RegressionEnsembleModel(self.get_models(), 50)
        with self.assertRaises(ValueError):
            ensemble.fit(self.combined)

        # too big value considering min_train_series_length
        ensemble = RegressionEnsembleModel(self.get_models(), 45)
        with self.assertRaises(ValueError):
            ensemble.fit(self.combined)

    if TORCH_AVAILABLE:

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
