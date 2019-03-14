import unittest
import pandas as pd
from ..u8timeseries.timeseries import TimeSeries


class TimeSeriesTestCase(unittest.TestCase):
    __test__ = True

    pd_series1 = pd.Series(range(10), index=pd.date_range('20130101', '20130110'))
    series1: TimeSeries = TimeSeries(pd_series1)

    def test_dates(self):
        self.assertEqual(self.series1.start_time(), pd.Timestamp('20130101'))
        self.assertEqual(self.series1.end_time(), pd.Timestamp('20130110'))
        self.assertEqual(self.series1.duration(), pd.Timedelta(days=10))


if __name__ == "__main__":
    unittest.main()
