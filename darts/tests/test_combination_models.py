import unittest

from ..utils import timeseries_generation as tg
from ..models import NaiveDrift, NaiveSeasonal, Theta, ExponentialSmoothing
from ..models.combination_model import CombinationModel
from ..models.groe_combination_model import GROECombinationModel


class CombinationModelsTestCase(unittest.TestCase):
    series1 = tg.sine_timeseries(value_frequency=(1 / 5), value_y_offset=10, length=50)
    series2 = tg.linear_timeseries(length=50)

    def test_input_models(self):
        with self.assertRaises(ValueError):
            CombinationModel([])
        with self.assertRaises(ValueError):
            CombinationModel([NaiveDrift, NaiveSeasonal, Theta, ExponentialSmoothing])
        with self.assertRaises(ValueError):
            CombinationModel([NaiveDrift(), NaiveSeasonal, Theta(), ExponentialSmoothing()])
        CombinationModel([NaiveDrift(), NaiveSeasonal(), Theta(), ExponentialSmoothing()])

    def test_call_predict(self):
        comb = CombinationModel([NaiveSeasonal(), Theta()])
        groe_comb = GROECombinationModel([NaiveSeasonal(), Theta()])
        with self.assertRaises(Exception):
            comb.predict(5)
        with self.assertRaises(Exception):
            groe_comb.predict(5)
        comb.fit(self.series1)
        groe_comb.fit(self.series1)
        comb.predict(5)
        groe_comb.predict(5)

    def test_predict_combination(self):
        naive = NaiveSeasonal(K=5)
        theta = Theta()
        comb = CombinationModel([naive, theta])
        comb.fit(self.series1 + self.series2)
        forecast_comb = comb.predict(5)
        naive.fit(self.series1 + self.series2)
        theta.fit(self.series1 + self.series2)
        forecast_mean = 0.5 * naive.predict(5) + 0.5 * theta.predict(5)
        self.assertEqual(forecast_comb, forecast_mean)

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
