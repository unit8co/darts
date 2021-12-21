import numpy as np
import pandas as pd

from darts.tests.base_test_class import DartsBaseTestClass
from darts import TimeSeries
from darts.utils.statistics import check_seasonality


class TimeSeriesTestCase(DartsBaseTestClass):
    def test_check_seasonality(self):
        pd_series = pd.Series(range(50), index=pd.date_range("20130101", "20130219"))
        pd_series = pd_series.map(lambda x: np.sin(x * np.pi / 3 + np.pi / 2))
        series = TimeSeries.from_series(pd_series)

        self.assertEqual((True, 6), check_seasonality(series))
        self.assertEqual((False, 3), check_seasonality(series, m=3))

        with self.assertRaises(AssertionError):
            check_seasonality(series.stack(series))
