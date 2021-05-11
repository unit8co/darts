import numpy as np

from .base_test_class import DartsBaseTestClass
from ..utils import timeseries_generation as tg
from ..models import NaiveDrift, NaiveSeasonal, Theta, ExponentialSmoothing
from ..models import NaiveEnsembleModel


class EnsembleModelsTestCase(DartsBaseTestClass):
    series1 = tg.sine_timeseries(value_frequency=(1 / 5), value_y_offset=10, length=50)
    series2 = tg.linear_timeseries(length=50)

    def test_input_models(self):
        with self.assertRaises(ValueError):
            NaiveEnsembleModel([])
        with self.assertRaises(ValueError):
            NaiveEnsembleModel([NaiveDrift, NaiveSeasonal, Theta, ExponentialSmoothing])
        with self.assertRaises(ValueError):
            NaiveEnsembleModel([NaiveDrift(), NaiveSeasonal, Theta(), ExponentialSmoothing()])
        NaiveEnsembleModel([NaiveDrift(), NaiveSeasonal(), Theta(), ExponentialSmoothing()])

    def test_call_predict(self):
        naive_ensemble = NaiveEnsembleModel([NaiveSeasonal(), Theta()])
        with self.assertRaises(Exception):
            naive_ensemble.predict(5)
        naive_ensemble.fit(self.series1)
        naive_ensemble.predict(5)

    def test_predict_ensemble(self):
        naive = NaiveSeasonal(K=5)
        theta = Theta()
        naive_ensemble = NaiveEnsembleModel([naive, theta])
        naive_ensemble.fit(self.series1 + self.series2)
        forecast_naive_ensemble = naive_ensemble.predict(5)
        naive.fit(self.series1 + self.series2)
        theta.fit(self.series1 + self.series2)
        forecast_mean = 0.5 * naive.predict(5) + 0.5 * theta.predict(5)

        self.assertTrue(np.array_equal(forecast_naive_ensemble.values(), forecast_mean.values()))


if __name__ == '__main__':
    unittest.main()
