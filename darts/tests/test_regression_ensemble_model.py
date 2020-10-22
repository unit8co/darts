import unittest
import logging

from sklearn.ensemble import RandomForestRegressor

from ..utils import timeseries_generation as tg
from ..models import NaiveDrift, NaiveSeasonal
from ..models import StandardRegressionModel
from ..models import RegressionEnsembleModel

try:
    from ..models import RNNModel
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not available. Some tests will be skipped.')
    TORCH_AVAILABLE = False


class RegressionEnsembleModelsTestCase(unittest.TestCase):
    sine_series = tg.sine_timeseries(value_frequency=(1 / 5), value_y_offset=10, length=50)
    lin_series = tg.linear_timeseries(length=50)

    combined = sine_series + lin_series

    def get_models(self):
        return [NaiveDrift(), NaiveSeasonal(5), NaiveSeasonal(10)]

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_accepts_different_regression_models(self):
        regr1 = StandardRegressionModel()
        regr2 = RandomForestRegressor()

        model0 = RegressionEnsembleModel(self.get_models(), 10)
        model1 = RegressionEnsembleModel(self.get_models(), 10, regr1)
        model2 = RegressionEnsembleModel(self.get_models(), 10, regr2)

        models = [model0, model1, model2]
        for model in models:
            model.fit(self.combined)
            model.predict(10)

    def test_train_n_points(self):
        regr = StandardRegressionModel(train_n_points=5)

        # ambiguous values
        with self.assertRaises(ValueError):
            ensemble = RegressionEnsembleModel(self.get_models(), 10, regr)

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
            model1 = RNNModel(random_state=0)
            model2 = RNNModel(random_state=0)

            ensemble = RegressionEnsembleModel([model1], 5)
            ensemble.fit(self.combined)

            model1_fitted = ensemble.models[0]
            forecast1 = model1_fitted.predict(10)

            model2.fit(self.combined)
            forecast2 = model2.predict(10)

            self.assertEqual(forecast1, forecast2)


if __name__ == '__main__':
    unittest.main()
