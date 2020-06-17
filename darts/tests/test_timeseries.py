import logging
import unittest
import math

import numpy as np
import pandas as pd

from ..timeseries import TimeSeries


class TimeSeriesTestCase(unittest.TestCase):

    times = pd.date_range('20130101', '20130110')
    pd_series1 = pd.Series(range(10), index=times)
    pd_series2 = pd.Series(range(5, 15), index=times)
    pd_series3 = pd.Series(range(15, 25), index=times)
    series1: TimeSeries = TimeSeries.from_series(pd_series1)
    series2: TimeSeries = TimeSeries.from_series(pd_series2)
    series3: TimeSeries = TimeSeries.from_series(pd_series2)

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_creation(self):
        with self.assertRaises(ValueError):
            # Index is dateTimeIndex
            TimeSeries.from_series(pd.Series(range(10), range(10)))
        series_test = TimeSeries.from_series(self.pd_series1)
        self.assertTrue(series_test.pd_series().equals(self.pd_series1))

    def test_alt_creation(self):
        with self.assertRaises(ValueError):
            # Series cannot be lower than three without passing frequency as argument to constructor
            index = pd.date_range('20130101', '20130102')
            TimeSeries.from_times_and_values(index, self.pd_series1.values[:2])
        with self.assertRaises(ValueError):
            # all arrays must have same length
            TimeSeries.from_times_and_values(self.pd_series1.index,
                                             self.pd_series1.values[:-1])

        # test if reordering is correct
        rand_perm = np.random.permutation(range(1, 11))
        index = pd.to_datetime(['201301{:02d}'.format(i) for i in rand_perm])
        series_test = TimeSeries.from_times_and_values(index,
                                                       self.pd_series1.values[rand_perm - 1])

        self.assertTrue(series_test.start_time() == pd.to_datetime('20130101'))
        self.assertTrue(series_test.end_time() == pd.to_datetime('20130110'))
        self.assertTrue(series_test.pd_series().equals(self.pd_series1))
        self.assertTrue(series_test.freq() == self.series1.freq())

    # TODO test over to_dataframe when multiple features choice is decided

    def test_eq(self):
        seriesA: TimeSeries = TimeSeries.from_series(self.pd_series1)
        self.assertTrue(self.series1 == seriesA)
        self.assertFalse(self.series1 != seriesA)

        # with different dates
        seriesC = TimeSeries.from_series(pd.Series(range(10), index=pd.date_range('20130102', '20130111')))
        self.assertFalse(self.series1 == seriesC)

    def test_dates(self):
        self.assertEqual(self.series1.start_time(), pd.Timestamp('20130101'))
        self.assertEqual(self.series1.end_time(), pd.Timestamp('20130110'))
        self.assertEqual(self.series1.duration(), pd.Timedelta(days=9))

    @staticmethod
    def helper_test_slice(test_case, test_series: TimeSeries):
        # base case
        seriesA = test_series.slice(pd.Timestamp('20130104'), pd.Timestamp('20130107'))
        test_case.assertEqual(seriesA.start_time(), pd.Timestamp('20130104'))
        test_case.assertEqual(seriesA.end_time(), pd.Timestamp('20130107'))

        # time stamp not in series
        seriesB = test_series.slice(pd.Timestamp('20130104 12:00:00'), pd.Timestamp('20130107'))
        test_case.assertEqual(seriesB.start_time(), pd.Timestamp('20130105'))
        test_case.assertEqual(seriesB.end_time(), pd.Timestamp('20130107'))

        # end timestamp after series
        seriesC = test_series.slice(pd.Timestamp('20130108'), pd.Timestamp('20130201'))
        test_case.assertEqual(seriesC.start_time(), pd.Timestamp('20130108'))
        test_case.assertEqual(seriesC.end_time(), pd.Timestamp('20130110'))

        # n points, base case
        seriesD = test_series.slice_n_points_after(pd.Timestamp('20130102'), n=3)
        test_case.assertEqual(seriesD.start_time(), pd.Timestamp('20130102'))
        test_case.assertTrue(len(seriesD.values()) == 3)
        test_case.assertEqual(seriesD.end_time(), pd.Timestamp('20130104'))

        seriesE = test_series.slice_n_points_after(pd.Timestamp('20130107 12:00:10'), n=10)
        test_case.assertEqual(seriesE.start_time(), pd.Timestamp('20130108'))
        test_case.assertEqual(seriesE.end_time(), pd.Timestamp('20130110'))

        seriesF = test_series.slice_n_points_before(pd.Timestamp('20130105'), n=3)
        test_case.assertEqual(seriesF.end_time(), pd.Timestamp('20130105'))
        test_case.assertTrue(len(seriesF.values()) == 3)
        test_case.assertEqual(seriesF.start_time(), pd.Timestamp('20130103'))

        seriesG = test_series.slice_n_points_before(pd.Timestamp('20130107 12:00:10'), n=10)
        test_case.assertEqual(seriesG.start_time(), pd.Timestamp('20130101'))
        test_case.assertEqual(seriesG.end_time(), pd.Timestamp('20130107'))

    @staticmethod
    def helper_test_split(test_case, test_series: TimeSeries):
        seriesA, seriesB = test_series.split_after(pd.Timestamp('20130104'))
        test_case.assertEqual(seriesA.end_time(), pd.Timestamp('20130104'))
        test_case.assertEqual(seriesB.start_time(), pd.Timestamp('20130105'))

        seriesC, seriesD = test_series.split_before(pd.Timestamp('20130104'))
        test_case.assertEqual(seriesC.end_time(), pd.Timestamp('20130103'))
        test_case.assertEqual(seriesD.start_time(), pd.Timestamp('20130104'))

        test_case.assertEqual(test_series.freq_str(), seriesA.freq_str())
        test_case.assertEqual(test_series.freq_str(), seriesC.freq_str())

    @staticmethod
    def helper_test_drop(test_case, test_series: TimeSeries):
        seriesA = test_series.drop_after(pd.Timestamp('20130105'))
        test_case.assertEqual(seriesA.end_time(), pd.Timestamp('20130105') - test_series.freq())
        test_case.assertTrue(np.all(seriesA.time_index() < pd.Timestamp('20130105')))

        seriesB = test_series.drop_before(pd.Timestamp('20130105'))
        test_case.assertEqual(seriesB.start_time(), pd.Timestamp('20130105') + test_series.freq())
        test_case.assertTrue(np.all(seriesB.time_index() > pd.Timestamp('20130105')))

        test_case.assertEqual(test_series.freq_str(), seriesA.freq_str())
        test_case.assertEqual(test_series.freq_str(), seriesB.freq_str())

    @staticmethod
    def helper_test_intersect(test_case, test_series: TimeSeries):
        seriesA = TimeSeries.from_series(pd.Series(range(2, 8), index=pd.date_range('20130102', '20130107')))

        seriesB = test_series.slice_intersect(seriesA)
        test_case.assertEqual(seriesB.start_time(), pd.Timestamp('20130102'))
        test_case.assertEqual(seriesB.end_time(), pd.Timestamp('20130107'))

        # Outside of range
        seriesD = test_series.slice_intersect(TimeSeries.from_series(pd.Series(range(6, 13),
                                                                     index=pd.date_range('20130106', '20130112'))))
        test_case.assertEqual(seriesD.start_time(), pd.Timestamp('20130106'))
        test_case.assertEqual(seriesD.end_time(), pd.Timestamp('20130110'))

        # Small intersect
        seriesE = test_series.slice_intersect(TimeSeries.from_series(
            pd.Series(range(9, 13), index=pd.date_range('20130109', '20130112')))
        )
        test_case.assertEqual(len(seriesE), 2)

        # No intersect
        with test_case.assertRaises(ValueError):
            test_series.slice_intersect(TimeSeries(pd.Series(range(6, 13),
                                        index=pd.date_range('20130116', '20130122'))))

    def test_rescale(self):
        with self.assertRaises(ValueError):
            self.series1.rescale_with_value(1)

        seriesA = self.series3.rescale_with_value(0)
        self.assertTrue(np.all(seriesA.values() == 0))

        seriesB = self.series3.rescale_with_value(-5)
        self.assertTrue(self.series3 * -1. == seriesB)

        seriesC = self.series3.rescale_with_value(1)
        self.assertTrue(self.series3 * 0.2 == seriesC)

        seriesD = self.series3.rescale_with_value(1e+20)  # TODO: test will fail if value > 1e24 due to num imprecision
        self.assertTrue(self.series3 * 0.2e+20 == seriesD)

    @staticmethod
    def helper_test_shift(test_case, test_series: TimeSeries):
        seriesA = test_case.series1.shift(0)
        test_case.assertTrue(seriesA == test_case.series1)

        seriesB = test_series.shift(1)
        test_case.assertTrue(seriesB.time_index().equals(
            test_series.time_index()[1:].append(
                pd.DatetimeIndex([test_series.time_index()[-1] + test_series.freq()])
            )))

        seriesC = test_series.shift(-1)
        test_case.assertTrue(seriesC.time_index().equals(
            pd.DatetimeIndex([test_series.time_index()[0] - test_series.freq()]).append(
                test_series.time_index()[:-1])))

        with test_case.assertRaises(OverflowError):
            test_series.shift(1e+6)

        seriesM = TimeSeries.from_times_and_values(pd.date_range('20130101', '20130601', freq='m'), range(5))
        with test_case.assertRaises(OverflowError):
            seriesM.shift(1e+4)

        seriesD = TimeSeries.from_times_and_values(pd.date_range('20130101', '20130101'), range(1),
                                                   freq='D')
        seriesE = seriesD.shift(1)
        test_case.assertEqual(seriesE.time_index()[0], pd.Timestamp('20130102'))

    @staticmethod
    def helper_test_append(test_case, test_series: TimeSeries):
        # reconstruct series
        seriesA, seriesB = test_series.split_after(pd.Timestamp('20130106'))
        test_case.assertEqual(seriesA.append(seriesB), test_series)
        test_case.assertEqual(seriesA.append(seriesB).freq(), test_series.freq())

        # Creating a gap is not allowed
        seriesC = test_series.drop_before(pd.Timestamp('20130107'))
        with test_case.assertRaises(ValueError):
            seriesA.append(seriesC)

        # Changing frequence is not allowed
        seriesM = TimeSeries.from_times_and_values(pd.date_range('20130107', '20130507', freq='30D'), range(5))
        with test_case.assertRaises(ValueError):
            seriesA.append(seriesM)

    @staticmethod
    def helper_test_append_values(test_case, test_series: TimeSeries):
        # reconstruct series
        seriesA, seriesB = test_series.split_after(pd.Timestamp('20130106'))
        test_case.assertEqual(seriesA.append_values(seriesB.values(), seriesB.time_index()), test_series)
        test_case.assertEqual(seriesA.append_values(seriesB.values()), test_series)

        # test for equality
        test_case.assertEqual(test_series.drop_after(pd.Timestamp('20130105'))
                              .append_values(test_series.drop_before(pd.Timestamp('20130104')).values()), test_series)
        test_case.assertEqual(seriesA.append_values([]), seriesA)

        # randomize order
        rd_order = np.random.permutation(range(len(seriesB.values())))
        test_case.assertEqual(seriesA.append_values(seriesB.values()[rd_order], seriesB.time_index()[rd_order]),
                              test_series)

        # add non consecutive index
        with test_case.assertRaises(ValueError):
            test_case.assertEqual(seriesA.append_values(seriesB.values(), seriesB.time_index() + seriesB.freq()),
                                  test_series)

        # add existing indices
        with test_case.assertRaises(ValueError):
            test_case.assertEqual(seriesA.append_values(seriesB.values(), seriesB.time_index() - 3 * seriesB.freq()),
                                  test_series)

        # other frequency
        with test_case.assertRaises(ValueError):
            test_case.assertEqual(seriesA.append_values(seriesB.values(),
                                                        pd.date_range('20130107', '20130113', freq='2d')),
                                  test_series)

    def test_slice(self):
        TimeSeriesTestCase.helper_test_slice(self, self.series1)

    def test_split(self):
        TimeSeriesTestCase.helper_test_split(self, self.series1)

    def test_drop(self):
        TimeSeriesTestCase.helper_test_drop(self, self.series1)

    def test_intersect(self):
        TimeSeriesTestCase.helper_test_intersect(self, self.series1)

    def test_shift(self):
        TimeSeriesTestCase.helper_test_shift(self, self.series1)

    def test_append(self):
        TimeSeriesTestCase.helper_test_append(self, self.series1)

    def test_append_values(self):
        TimeSeriesTestCase.helper_test_append_values(self, self.series1)

    def test_update(self):
        seriesA: TimeSeries = TimeSeries.from_times_and_values(self.times, [0, 1, 1, 3, 4, 5, 6, 2, 8, 0])
        seriesB: TimeSeries = TimeSeries.from_times_and_values(self.times, range(10))

        # change nothing
        seriesC = self.series1.copy()
        with self.assertRaises(ValueError):
            seriesA.update(self.times)
        seriesC = seriesC.update(self.times, range(10))
        self.assertEqual(seriesC, self.series1)

        # different len
        with self.assertRaises(ValueError):
            seriesA.update(self.times, [])
        with self.assertRaises(ValueError):
            seriesA.update(self.times, np.arange(3))
        with self.assertRaises(ValueError):
            seriesA.update(self.times, np.arange(4))

        # change outside
        seriesC = seriesA.copy()
        with self.assertRaises(ValueError):
            seriesC.update(self.times + 100 * seriesC.freq(), range(10))
        seriesC = seriesC.update(self.times.append(pd.date_range('20140101', '20140110')),
                                 list(range(10)) + [0] * 10)
        self.assertEqual(seriesC, self.series1)

        # change random
        seriesC = seriesA.copy()
        seriesC = seriesC.update(pd.DatetimeIndex(['20130108', '20130110', '20130103']), [7, 9, 2])
        self.assertEqual(seriesC, self.series1)

        # change one of each series
        seriesD = seriesB.copy()
        seriesD = seriesD.update(self.times, seriesA.pd_series().values)
        seriesA = seriesA.update(pd.DatetimeIndex(['20130103', '20130108', '20130110']), [2, 7, 9])
        self.assertEqual(seriesA, self.series1)
        seriesB = seriesB.update(self.times[::2], range(5))
        self.assertNotEqual(seriesB, self.series2)

        # use nan
        new_series = np.empty(10)
        new_series[:] = np.nan
        new_series[[2, 7, 9]] = [2, 7, 9]
        seriesD = seriesD.update(self.times, new_series)
        self.assertEqual(seriesD, self.series1)

    def test_ops(self):
        seriesA = TimeSeries.from_series(pd.Series([2 for _ in range(10)], index=self.pd_series1.index))
        targetAdd = TimeSeries.from_series(pd.Series(range(2, 12), index=self.pd_series1.index))
        targetSub = TimeSeries.from_series(pd.Series(range(-2, 8), index=self.pd_series1.index))
        targetMul = TimeSeries.from_series(pd.Series(range(0, 20, 2), index=self.pd_series1.index))
        targetDiv = TimeSeries.from_series(pd.Series([i / 2 for i in range(10)], index=self.pd_series1.index))
        targetPow = TimeSeries.from_series(pd.Series([float(i ** 2) for i in range(10)], index=self.pd_series1.index))

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

        with self.assertRaises(ZeroDivisionError):
            # Cannot divide by a TimeSeries with a value 0.
            self.series1 / self.series1

        with self.assertRaises(ZeroDivisionError):
            # Cannot divide by 0.
            self.series1 / 0

    def test_getitem(self):
        seriesA: TimeSeries = self.series1.drop_after(pd.Timestamp("20130105"))
        self.assertEqual(self.series1[pd.date_range('20130101', ' 20130104')], seriesA)
        self.assertEqual(self.series1[:4], seriesA)
        self.assertTrue(self.series1[pd.Timestamp('20130101')].equals(self.series1.pd_dataframe()[:1]))
        self.assertEqual(self.series1[pd.Timestamp('20130101'):pd.Timestamp('20130105')], seriesA)

        with self.assertRaises(IndexError):
            self.series1[pd.date_range('19990101', '19990201')]

        with self.assertRaises(KeyError):
            self.series1['19990101']

        with self.assertRaises(IndexError):
            self.series1[::-1]

    def test_fill_missing_dates(self):
        with self.assertRaises(ValueError):
            # Series cannot have date holes without automatic filling
            range_ = pd.date_range('20130101', '20130104').append(pd.date_range('20130106', '20130110'))
            TimeSeries.from_series(pd.Series(range(9), index=range_), fill_missing_dates=False)

        with self.assertRaises(ValueError):
            # Main series should have explicit frequency in case of date holes
            range_ = pd.date_range('20130101', '20130104').append(pd.date_range('20130106', '20130110', freq='2D'))
            TimeSeries.from_series(pd.Series(range(7), index=range_))

        range_ = pd.date_range('20130101', '20130104').append(pd.date_range('20130106', '20130110'))
        series_test = TimeSeries.from_series(pd.Series(range(9), index=range_))
        self.assertEqual(series_test.freq_str(), 'D')

        range_ = pd.date_range('20130101', '20130104', freq='2D') \
            .append(pd.date_range('20130107', '20130111', freq='2D'))
        series_test = TimeSeries.from_series(pd.Series(range(5), index=range_))
        self.assertEqual(series_test.freq_str(), '2D')
        self.assertEqual(series_test.start_time(), range_[0])
        self.assertEqual(series_test.end_time(), range_[-1])
        self.assertTrue(math.isnan(series_test.pd_series().get('20130105')))

    def test_resample_timeseries(self):
        times = pd.date_range('20130101', '20130110')
        pd_series = pd.Series(range(10), index=times)
        timeseries = TimeSeries.from_series(pd_series)

        resampled_timeseries = timeseries.resample('H')
        self.assertEqual(resampled_timeseries.freq_str(), 'H')
        self.assertEqual(resampled_timeseries.pd_series().at[pd.Timestamp('20130101020000')], 0)
        self.assertEqual(resampled_timeseries.pd_series().at[pd.Timestamp('20130102020000')], 1)
        self.assertEqual(resampled_timeseries.pd_series().at[pd.Timestamp('20130109090000')], 8)

        resampled_timeseries = timeseries.resample('2D')
        self.assertEqual(resampled_timeseries.freq_str(), '2D')
        self.assertEqual(resampled_timeseries.pd_series().at[pd.Timestamp('20130101')], 0)
        with self.assertRaises(KeyError):
            resampled_timeseries.pd_series().at[pd.Timestamp('20130102')]

        self.assertEqual(resampled_timeseries.pd_series().at[pd.Timestamp('20130109')], 8)

    def test_short_series_creation(self):
        # test missing freq argument error
        with self.assertRaises(ValueError):
            TimeSeries.from_times_and_values(pd.date_range('20130101', '20130102'), range(2))
        # test empty pandas series error
        with self.assertRaises(ValueError):
            TimeSeries.from_series(pd.Series(), freq='D')
        # test frequency mismatch error
        with self.assertRaises(ValueError):
            TimeSeries.from_times_and_values(pd.date_range('20130101', '20130105'), range(5), freq='M')
        # test successful instantiation of TimeSeries with length 2
        TimeSeries.from_times_and_values(pd.date_range('20130101', '20130102'), range(2), freq='D')

    def test_short_series_slice(self):
        seriesA, seriesB = self.series1.split_after(pd.Timestamp('20130108'))
        self.assertEqual(len(seriesA), 8)
        self.assertEqual(len(seriesB), 2)
        seriesA, seriesB = self.series1.split_after(pd.Timestamp('20130109'))
        self.assertEqual(len(seriesA), 9)
        self.assertEqual(len(seriesB), 1)
        self.assertEqual(seriesB.time_index()[0], self.series1.time_index()[-1])
        seriesA, seriesB = self.series1.split_before(pd.Timestamp('20130103'))
        self.assertEqual(len(seriesA), 2)
        self.assertEqual(len(seriesB), 8)
        seriesA, seriesB = self.series1.split_before(pd.Timestamp('20130102'))
        self.assertEqual(len(seriesA), 1)
        self.assertEqual(len(seriesB), 9)
        self.assertEqual(seriesA.time_index()[-1], self.series1.time_index()[0])
        seriesC = self.series1.slice(pd.Timestamp('20130105'), pd.Timestamp('20130105'))
        self.assertEqual(len(seriesC), 1)
