import unittest
import pandas as pd

from ..utils import retain_period_common_to_all, _with_sanity_checks
from ..timeseries import TimeSeries


class UtilsTestCase(unittest.TestCase):

    def test_retain_period_common_to_all(self):
        seriesA = TimeSeries.from_times_and_values(pd.date_range('20000101', '20000110'), range(10))
        seriesB = TimeSeries.from_times_and_values(pd.date_range('20000103', '20000108'), range(6))
        seriesC = TimeSeries.from_times_and_values(pd.date_range('20000104', '20000112'), range(9))
        seriesC = seriesC.stack(seriesC)

        common_series_list = retain_period_common_to_all([seriesA, seriesB, seriesC])

        # test start and end dates
        for common_series in common_series_list:
            self.assertEqual(common_series.start_time(), pd.Timestamp('20000104'))
            self.assertEqual(common_series.end_time(), pd.Timestamp('20000108'))

        # test widths
        self.assertEqual(common_series_list[0].width, 1)
        self.assertEqual(common_series_list[1].width, 1)
        self.assertEqual(common_series_list[2].width, 2)

    def test_sanity_check_example(self):
        class Model:
            def _sanity_check(self, *args, **kwargs):
                if kwargs['b'] != kwargs['c']:
                    raise(ValueError("b and c must be equal"))

            @_with_sanity_checks("_sanity_check")
            def fit(self, a, b=0, c=0):
                pass

        m = Model()

        # b != c should raise error
        with self.assertRaises(ValueError):
            m.fit(5, b=3, c=2)

        # b == c should not raise error
        m.fit(5, b=2, c=2)
