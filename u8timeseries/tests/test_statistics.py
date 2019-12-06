import unittest
import pandas as pd
import numpy as np

from u8timeseries import TimeSeries
from u8timeseries.models import statistics


class TimeSeriesTestCase(unittest.TestCase):
    __test__ = True

    def test_check_seasonality(self):
        pd_series = pd.Series(range(50), index=pd.date_range('20130101', '20130219'))
        pd_series = pd_series.map(lambda x: np.sin(x*np.pi / 3))
        series = TimeSeries(pd_series)

        self.assertEqual((True, 6), statistics.check_seasonality(series))


if __name__ == '__main__':
    unittest.main()
