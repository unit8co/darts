import unittest
import numpy as np
import pandas as pd
import logging

from ..timeseries import TimeSeries
from ..metrics import metrics


class MetricsTestCase(unittest.TestCase):

    pd_series1 = pd.Series(range(10), index=pd.date_range('20130101', '20130110'))
    pd_series2 = pd.Series(np.random.rand(10) * 10, index=pd.date_range('20130101', '20130110'))
    pd_series3 = pd.Series(np.sin(np.pi * np.arange(20) / 4), index=pd.date_range('20130101', '20130120'))
    series1: TimeSeries = TimeSeries.from_series(pd_series1)
    pd_series1[:] = pd_series1.mean()
    series0: TimeSeries = TimeSeries.from_series(pd_series1)
    series2: TimeSeries = TimeSeries.from_series(pd_series2)
    series3: TimeSeries = TimeSeries.from_series(pd_series3)
    series12: TimeSeries = series1.stack(series2)
    series21: TimeSeries = series2.stack(series1)
    series1b = TimeSeries.from_times_and_values(pd.date_range('20130111', '20130120'), series1.values())
    series2b = TimeSeries.from_times_and_values(pd.date_range('20130111', '20130120'), series2.values())

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_zero(self):
        with self.assertRaises(ValueError):
            metrics.mape(self.series1, self.series1)

        with self.assertRaises(ValueError):
            metrics.mape(self.series12, self.series12)

        with self.assertRaises(ValueError):
            metrics.ope(self.series1 - self.series1.pd_series().mean(), self.series1 - self.series1.pd_series().mean())

    def test_same(self):
        self.assertEqual(metrics.mape(self.series1 + 1, self.series1 + 1), 0)
        self.assertEqual(metrics.mase(self.series1 + 1, self.series1 + 1, 1), 0)
        self.assertEqual(metrics.marre(self.series1 + 1, self.series1 + 1), 0)
        self.assertEqual(metrics.r2_score(self.series1 + 1, self.series1 + 1), 1)
        self.assertEqual(metrics.ope(self.series1 + 1, self.series1 + 1), 0)

    def helper_test_shape_equality(self, metric):
        self.assertAlmostEqual(metric(self.series12, self.series21),
                               metric(self.series1.append(self.series2b), self.series2.append(self.series1b)))

    def test_r2(self):
        from sklearn.metrics import r2_score
        self.assertEqual(metrics.r2_score(self.series1, self.series0), 0)
        self.assertEqual(metrics.r2_score(self.series1, self.series2),
                         r2_score(self.series1.values(), self.series2.values()))
        self.helper_test_shape_equality(metrics.r2_score)

    def test_marre(self):
        self.assertAlmostEqual(metrics.marre(self.series1, self.series2),
                               metrics.marre(self.series1 + 100, self.series2 + 100))
        self.helper_test_shape_equality(metrics.marre)

    def test_season(self):
        with self.assertRaises(ValueError):
            metrics.mase(self.series3, self.series3 * 1.3, 8)

    def test_ope(self):
        self.helper_test_shape_equality(metrics.ope)

    def test_mse(self):
        self.helper_test_shape_equality(metrics.mse)

    def test_rmse(self):
        self.helper_test_shape_equality(metrics.rmse)

    def test_rmsle(self):
        self.helper_test_shape_equality(metrics.rmsle)

    def test_coefficient_of_variation(self):
        self.helper_test_shape_equality(metrics.coefficient_of_variation)

    def test_mae(self):
        self.helper_test_shape_equality(metrics.mae)

    def test_different_width(self):
        with self.assertRaises(ValueError):
            metrics.mape(self.series1, self.series12)
