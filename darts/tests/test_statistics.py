import numpy as np
import pandas as pd

from .base_test_class import DartsBaseTestClass
from .. import TimeSeries
from ..utils.timeseries_generation import constant_timeseries, linear_timeseries, gaussian_timeseries
from ..utils.statistics import check_seasonality, granger_causality_tests


class TimeSeriesTestCase(DartsBaseTestClass):

    def test_check_seasonality(self):
        pd_series = pd.Series(range(50), index=pd.date_range('20130101', '20130219'))
        pd_series = pd_series.map(lambda x: np.sin(x * np.pi / 3 + np.pi / 2))
        series = TimeSeries.from_series(pd_series)

        self.assertEqual((True, 6), check_seasonality(series))
        self.assertEqual((False, 3), check_seasonality(series, m=3))

        with self.assertRaises(AssertionError):
            check_seasonality(series.stack(series))

    def test_granger_causality(self):
        series_cause_1 = (
            constant_timeseries(start = 0, end = 9999)
            .stack(constant_timeseries(start = 0, end = 9999))
        )
        series_cause_2 = gaussian_timeseries(start = 0, end = 9999)
        series_effect_1  = constant_timeseries(start = 0, end = 999)
        series_effect_2  = TimeSeries.from_values(np.random.uniform(0, 1, 10000))
        series_effect_3 = TimeSeries.from_values(np.random.uniform(0, 1, (1000, 2, 1000)))
        series_effect_4 = constant_timeseries(start=pd.Timestamp('2000-01-01'), length=10000)

        #Test univariate
        with self.assertRaises(AssertionError):
            granger_causality_tests(series_cause_1, series_effect_1,  10, verbose=False)
        with self.assertRaises(AssertionError):
            granger_causality_tests(series_effect_1, series_cause_1, 10, verbose=False)

        #Test deterministic
        with self.assertRaises(AssertionError):
            granger_causality_tests(series_cause_1, series_effect_3, 10, verbose=False)
        with self.assertRaises(AssertionError):
            granger_causality_tests(series_effect_3, series_cause_1,  10, verbose=False)

        #Test Frequency
        with self.assertRaises(ValueError):
            granger_causality_tests(series_cause_2, series_effect_4,  10, verbose=False)

        #Test granger basics
        tests = granger_causality_tests(series_effect_2, series_effect_2, 10, verbose=False)
        self.assertTrue(tests[1][0]['ssr_ftest'][1]>0.99)
        tests = granger_causality_tests(series_cause_2, series_effect_2, 10, verbose=False)
        self.assertTrue(tests[1][0]['ssr_ftest'][1]>0.01)