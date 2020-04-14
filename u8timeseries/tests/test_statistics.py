import unittest
import pandas as pd
import numpy as np

from u8timeseries import TimeSeries
from u8timeseries.models.statistics import check_seasonality


class TimeSeriesTestCase(unittest.TestCase):

    def test_check_seasonality(self):
        pd_series = pd.Series(range(50), index=pd.date_range('20130101', '20130219'))
        pd_series = pd_series.map(lambda x: np.sin(x*np.pi / 3 + np.pi/2))
        series = TimeSeries(pd_series)

        self.assertEqual((True, 6), check_seasonality(series))
        self.assertEqual((False, 3), check_seasonality(series, m=3))

