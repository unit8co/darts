import unittest
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from u8timeseries.metrics import metrics


class MetricsTestCase(unittest.TestCase):

    pd_series1 = pd.Series(range(10), index=pd.date_range('20130101', '20130110'))
    pd_series2 = pd.Series(np.random.rand(10)*10, index=pd.date_range('20130101', '20130110'))
    pd_series3 = pd.Series(np.sin(np.pi*np.arange(20)/4), index=pd.date_range('20130101', '20130120'))
    series1: TimeSeries = TimeSeries(pd_series1)
    pd_series1[:] = pd_series1.mean()
    series0: TimeSeries = TimeSeries(pd_series1)
    series2: TimeSeries = TimeSeries(pd_series2)
    series3: TimeSeries = TimeSeries(pd_series3)

    def test_zero(self):
        self.assertTrue(np.isnan(metrics.mape(self.series1, self.series1)))

        self.assertTrue(np.isnan(metrics.overall_percentage_error(self.series1-self.series1.mean(),
                                                                  self.series1-self.series1.mean())))

    def test_same(self):
        self.assertEqual(metrics.mape(self.series1+1, self.series1+1), 0)
        self.assertEqual(metrics.mase(self.series1+1, self.series1+1, 1), 0)
        self.assertEqual(metrics.marre(self.series1+1, self.series1+1), 0)
        self.assertEqual(metrics.r2_score(self.series1+1, self.series1+1), 1)
        self.assertEqual(metrics.overall_percentage_error(self.series1+1, self.series1+1), 0)

    def test_r2(self):
        from sklearn.metrics import r2_score
        self.assertEqual(metrics.r2_score(self.series1, self.series0), 0)
        self.assertEqual(metrics.r2_score(self.series1, self.series2),
                         r2_score(self.series1.values(), self.series2.values()))

    def test_marre(self):
        self.assertAlmostEqual(metrics.marre(self.series1, self.series2),
                               metrics.marre(self.series1+100, self.series2+100))

    def test_season(self):
        with self.assertRaises(AssertionError):
            metrics.mase(self.series3, self.series3 * 1.3, 8)


