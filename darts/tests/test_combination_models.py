import unittest
import logging

from ..utils import timeseries_generation as tg
from ..models import NaiveDrift, NaiveSeasonal, Theta, ExponentialSmoothing
from ..models import CombinationModel, NaiveCombinationModel
from ..models.groe_combination_model import GROECombinationModel


class CombinationModelsTestCase(unittest.TestCase):
    series1 = tg.sine_timeseries(value_frequency=(1 / 5), value_y_offset=10, length=50)
    series2 = tg.linear_timeseries(length=50)

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_input_models(self):
        with self.assertRaises(ValueError):
            NaiveCombinationModel([])
        with self.assertRaises(ValueError):
            NaiveCombinationModel([NaiveDrift, NaiveSeasonal, Theta, ExponentialSmoothing])
        with self.assertRaises(ValueError):
            NaiveCombinationModel([NaiveDrift(), NaiveSeasonal, Theta(), ExponentialSmoothing()])
        NaiveCombinationModel([NaiveDrift(), NaiveSeasonal(), Theta(), ExponentialSmoothing()])

    def test_call_predict(self):
        naive_comb = NaiveCombinationModel([NaiveSeasonal(), Theta()])
        groe_comb = GROECombinationModel([NaiveSeasonal(), Theta()])
        with self.assertRaises(Exception):
            naive_comb.predict(5)
        with self.assertRaises(Exception):
            groe_comb.predict(5)
        naive_comb.fit(self.series1)
        groe_comb.fit(self.series1)
        naive_comb.predict(5)
        groe_comb.predict(5)

    def test_predict_combination(self):
        naive = NaiveSeasonal(K=5)
        theta = Theta()
        naive_comb = NaiveCombinationModel([naive, theta])
        naive_comb.fit(self.series1 + self.series2)
        forecast_naive_comb = naive_comb.predict(5)
        naive.fit(self.series1 + self.series2)
        theta.fit(self.series1 + self.series2)
        forecast_mean = 0.5 * naive.predict(5) + 0.5 * theta.predict(5)
        self.assertEqual(forecast_naive_comb, forecast_mean)

    def test_fit_groe(self):
        naive = NaiveSeasonal(K=5)
        theta = Theta()
        comb = GROECombinationModel([naive, theta])
        comb.fit(self.series1 + self.series2)
        self.assertTrue((1 >= comb.weights).all())
        self.assertTrue((comb.weights >= 0).all())

    def test_bad_fit_groe(self):
        naive = NaiveSeasonal(K=5)
        theta = Theta()
        comb = GROECombinationModel([naive, theta])
        with self.assertRaises(ValueError):
            comb.fit(tg.constant_timeseries(0, length=50))

    def test_perfect_fit_groe(self):
        naive = NaiveSeasonal(K=5)
        theta = Theta()
        comb = GROECombinationModel([naive, theta])
        comb.fit(self.series1)
        self.assertTrue(comb.weights[0] == 1.)


if __name__ == '__main__':
    unittest.main()
