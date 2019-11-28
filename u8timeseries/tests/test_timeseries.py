import unittest
import pandas as pd

from timeseries import TimeSeries


class TimeSeriesTestCase(unittest.TestCase):
    __test__ = True

    pd_series1 = pd.Series(range(10), index=pd.date_range('20130101', '20130110'))
    series1: TimeSeries = TimeSeries(pd_series1)

    def test_creation(self):
        with self.assertRaises(AssertionError):
            # Conf interval must be same length as main series
            pd_lo = pd.Series(range(5, 14), index=pd.date_range('20130101', '20130109'))
            TimeSeries(self.pd_series1, pd_lo)

        with self.assertRaises(AssertionError):
            # Conf interval must have same time index as main series
            pd_lo = pd.Series(range(5, 15), index=pd.date_range('20130102', '20130111'))
            TimeSeries(self.pd_series1, pd_lo)

        with self.assertRaises(AssertionError):
            # Main series cannot have date holes
            range_ = pd.date_range('20130101', '20130104').append(pd.date_range('20130106', '20130110'))
            TimeSeries(pd.Series(range(9), index=range_))

    def test_eq(self):
        seriesA: TimeSeries = TimeSeries(self.pd_series1)
        self.assertTrue(self.series1 == seriesA)

        # with a defined CI
        seriesB: TimeSeries = TimeSeries(self.pd_series1,
                                         confidence_hi=pd.Series(range(10, 20),
                                                                 index=pd.date_range('20130101', '20130110')))
        self.assertFalse(self.series1 == seriesB)
        self.assertTrue(self.series1 != seriesB)

        # with different dates
        seriesC = TimeSeries(pd.Series(range(10), index=pd.date_range('20130102', '20130111')))
        self.assertFalse(self.series1 == seriesC)

    def test_dates(self):
        self.assertEqual(self.series1.start_time(), pd.Timestamp('20130101'))
        self.assertEqual(self.series1.end_time(), pd.Timestamp('20130110'))
        self.assertEqual(self.series1.duration(), pd.Timedelta(days=9))

    def test_split(self):
        seriesA, seriesB = self.series1.split_after(pd.Timestamp('20130104'))
        self.assertEqual(seriesA.end_time(), pd.Timestamp('20130104'))
        self.assertEqual(seriesB.start_time(), pd.Timestamp('20130105'))

        with self.assertRaises(AssertionError):
            # Timestamp must be in time series
            _, _ = self.series1.split_after(pd.Timestamp('20130103 10:30:00'))

    def test_slice(self):
        # base case
        seriesA = self.series1.slice(pd.Timestamp('20130104'), pd.Timestamp('20130107'))
        self.assertEqual(seriesA.start_time(), pd.Timestamp('20130104'))
        self.assertEqual(seriesA.end_time(), pd.Timestamp('20130107'))

        # time stamp not in series
        seriesB = self.series1.slice(pd.Timestamp('20130104 12:00:00'), pd.Timestamp('20130107'))
        self.assertEqual(seriesB.start_time(), pd.Timestamp('20130105'))
        self.assertEqual(seriesB.end_time(), pd.Timestamp('20130107'))

        # end timestamp after series
        seriesC = self.series1.slice(pd.Timestamp('20130108'), pd.Timestamp('20130201'))
        self.assertEqual(seriesC.start_time(), pd.Timestamp('20130108'))
        self.assertEqual(seriesC.end_time(), pd.Timestamp('20130110'))

        # n points, base case
        seriesD = self.series1.slice_n_points_after(pd.Timestamp('20130102'), n=3)
        self.assertEqual(seriesD.start_time(), pd.Timestamp('20130102'))
        self.assertEqual(seriesD.end_time(), pd.Timestamp('20130104'))

        seriesE = self.series1.slice_n_points_after(pd.Timestamp('20130107 12:00:10'), n=10)
        self.assertEqual(seriesE.start_time(), pd.Timestamp('20130108'))
        self.assertEqual(seriesE.end_time(), pd.Timestamp('20130110'))

    def test_ops(self):
        seriesA = TimeSeries(pd.Series([2 for _ in range(10)], index=self.pd_series1.index))
        targetAdd = TimeSeries(pd.Series(range(2, 12), index=self.pd_series1.index))
        targetSub = TimeSeries(pd.Series(range(-2, 8), index=self.pd_series1.index))
        targetMul = TimeSeries(pd.Series(range(0, 20, 2), index=self.pd_series1.index))
        targetDiv = TimeSeries(pd.Series([i / 2 for i in range(10)], index=self.pd_series1.index))
        targetPow = TimeSeries(pd.Series([float(i ** 2) for i in range(10)], index=self.pd_series1.index))

        self.assertEqual(self.series1 + seriesA, targetAdd)
        self.assertEqual(self.series1 + 2, targetAdd)
        self.assertEqual(2 + self.series1, targetAdd)
        self.assertEqual(self.series1 - seriesA, targetSub)
        self.assertEqual(self.series1 - 2, targetSub)
        self.assertEqual(self.series1 * seriesA, targetMul)
        self.assertEqual(self.series1 * 2, targetMul)
        self.assertEqual(2 * self.series1, targetMul)
        self.assertEqual(self.series1 / seriesA, targetDiv)
        self.assertEqual(self.series1 / 2, targetDiv)
        self.assertEqual(self.series1 ** 2, targetPow)

        with self.assertRaises(AssertionError):
            # Cannot divide by a TimeSeries with a value 0.
            self.series1 / self.series1

        with self.assertRaises(AssertionError):
            # Cannot divide by 0.
            self.series1 / 0


if __name__ == "__main__":
    unittest.main()
