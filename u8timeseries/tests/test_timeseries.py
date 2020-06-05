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
    series1: TimeSeries = TimeSeries(pd_series1)
    series2: TimeSeries = TimeSeries(pd_series2)
    series3: TimeSeries = TimeSeries(pd_series2)

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_creation(self):
        with self.assertRaises(ValueError):
            # Index is dateTimeIndex
            TimeSeries(pd.Series(range(10), range(10)))
        series_test = TimeSeries(self.pd_series1)
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
        seriesA: TimeSeries = TimeSeries(self.pd_series1)
        self.assertTrue(self.series1 == seriesA)
        self.assertFalse(self.series1 != seriesA)

        # with different dates
        seriesC = TimeSeries(pd.Series(range(10), index=pd.date_range('20130102', '20130111')))
        self.assertFalse(self.series1 == seriesC)


    def test_dates(self):
        self.assertEqual(self.series1.start_time(), pd.Timestamp('20130101'))
        self.assertEqual(self.series1.end_time(), pd.Timestamp('20130110'))
        self.assertEqual(self.series1.duration(), pd.Timedelta(days=9))

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
        self.assertTrue(len(seriesD.values()) == 3)
        self.assertEqual(seriesD.end_time(), pd.Timestamp('20130104'))

        seriesE = self.series1.slice_n_points_after(pd.Timestamp('20130107 12:00:10'), n=10)
        self.assertEqual(seriesE.start_time(), pd.Timestamp('20130108'))
        self.assertEqual(seriesE.end_time(), pd.Timestamp('20130110'))

        seriesF = self.series1.slice_n_points_before(pd.Timestamp('20130105'), n=3)
        self.assertEqual(seriesF.end_time(), pd.Timestamp('20130105'))
        self.assertTrue(len(seriesF.values()) == 3)
        self.assertEqual(seriesF.start_time(), pd.Timestamp('20130103'))

        seriesG = self.series1.slice_n_points_before(pd.Timestamp('20130107 12:00:10'), n=10)
        self.assertEqual(seriesG.start_time(), pd.Timestamp('20130101'))
        self.assertEqual(seriesG.end_time(), pd.Timestamp('20130107'))

    def test_split(self):
        seriesA, seriesB = self.series1.split_after(pd.Timestamp('20130104'))
        self.assertEqual(seriesA.end_time(), pd.Timestamp('20130104'))
        self.assertEqual(seriesB.start_time(), pd.Timestamp('20130105'))

        seriesC, seriesD = self.series1.split_before(pd.Timestamp('20130104'))
        self.assertEqual(seriesC.end_time(), pd.Timestamp('20130103'))
        self.assertEqual(seriesD.start_time(), pd.Timestamp('20130104'))

        self.assertEqual(self.series1.freq_str(), seriesA.freq_str())
        self.assertEqual(self.series1.freq_str(), seriesC.freq_str())

    def test_drop(self):
        seriesA = self.series1.drop_after(pd.Timestamp('20130105'))
        self.assertEqual(seriesA.end_time(), pd.Timestamp('20130105') - self.series1.freq())
        self.assertTrue(np.all(seriesA.time_index() < pd.Timestamp('20130105')))

        seriesB = self.series1.drop_before(pd.Timestamp('20130105'))
        self.assertEqual(seriesB.start_time(), pd.Timestamp('20130105') + self.series1.freq())
        self.assertTrue(np.all(seriesB.time_index() > pd.Timestamp('20130105')))

        self.assertEqual(self.series1.freq_str(), seriesA.freq_str())
        self.assertEqual(self.series1.freq_str(), seriesB.freq_str())

    def test_intersect(self):
        seriesA = TimeSeries(pd.Series(range(2, 8), index=pd.date_range('20130102', '20130107')))

        seriesB = self.series1.slice_intersect(seriesA)
        self.assertEqual(seriesB.start_time(), pd.Timestamp('20130102'))
        self.assertEqual(seriesB.end_time(), pd.Timestamp('20130107'))

        # Outside of range
        seriesD = self.series1.slice_intersect(TimeSeries(pd.Series(range(6, 13),
                                                          index=pd.date_range('20130106', '20130112'))))
        self.assertEqual(seriesD.start_time(), pd.Timestamp('20130106'))
        self.assertEqual(seriesD.end_time(), pd.Timestamp('20130110'))

        # No intersect or too small intersect
        with self.assertRaises(ValueError):
            self.series1.slice_intersect(TimeSeries(pd.Series(range(6, 13),
                                                    index=pd.date_range('20130116', '20130122'))))
        with self.assertRaises(ValueError):
            self.series1.slice_intersect(TimeSeries(pd.Series(range(9, 13),
                                                    index=pd.date_range('20130109', '20130112'))))

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

    def test_shift(self):
        seriesA = self.series1.shift(0)
        self.assertTrue(seriesA == self.series1)

        seriesB = self.series1.shift(1)
        self.assertTrue(seriesB.time_index().equals(
                        self.series1.time_index()[1:].append(
                            pd.DatetimeIndex([self.series1.time_index()[-1] + self.series1.freq()])
                        )))

        seriesC = self.series1.shift(-1)
        self.assertTrue(seriesC.time_index().equals(
            pd.DatetimeIndex([self.series1.time_index()[0] - self.series1.freq()]).append(
                self.series1.time_index()[:-1])))

        with self.assertRaises(OverflowError):
            self.series1.shift(1e+6)

        seriesM = TimeSeries.from_times_and_values(pd.date_range('20130101', '20130601', freq='m'), range(5))
        with self.assertRaises(OverflowError):
            seriesM.shift(1e+4)

    def test_append(self):
        # reconstruct series
        seriesA, seriesB = self.series1.split_after(pd.Timestamp('20130106'))
        self.assertEqual(seriesA.append(seriesB), self.series1)
        self.assertEqual(seriesA.append(seriesB).freq(), self.series1.freq())

        # Creating a gap is not allowed
        seriesC = self.series1.drop_before(pd.Timestamp('20130107'))
        with self.assertRaises(ValueError):
            seriesA.append(seriesC)

        # Changing frequence is not allowed
        seriesM = TimeSeries.from_times_and_values(pd.date_range('20130107', '20130507', freq='30D'), range(5))
        with self.assertRaises(ValueError):
            seriesA.append(seriesM)

    def test_append_values(self):
        # reconstruct series
        seriesA, seriesB = self.series1.split_after(pd.Timestamp('20130106'))
        self.assertEqual(seriesA.append_values(seriesB.values(), seriesB.time_index()), self.series1)
        self.assertEqual(seriesA.append_values(seriesB.values()), self.series1)

        # add only few element
        self.assertEqual(self.series1.drop_after(pd.Timestamp('20130110')).append_values([9]), self.series1)
        self.assertEqual(seriesA.append_values([]), seriesA)

        # randomize order
        rd_order = np.random.permutation(range(len(seriesB.values())))
        self.assertEqual(seriesA.append_values(seriesB.values()[rd_order], seriesB.time_index()[rd_order]),
                         self.series1)

        # add non consecutive index
        with self.assertRaises(ValueError):
            self.assertEqual(seriesA.append_values(seriesB.values(), seriesB.time_index() + seriesB.freq()),
                             self.series1)

        # add existing indices
        with self.assertRaises(ValueError):
            self.assertEqual(seriesA.append_values(seriesB.values(), seriesB.time_index() - 3 * seriesB.freq()),
                             self.series1)

        # other frequency
        with self.assertRaises(ValueError):
            self.assertEqual(seriesA.append_values(seriesB.values(), pd.date_range('20130107', '20130113', freq='2d')),
                             self.series1)

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
            TimeSeries(pd.Series(range(9), index=range_), fill_missing_dates=False)

        with self.assertRaises(ValueError):
            # Main series should have explicit frequency in case of date holes
            range_ = pd.date_range('20130101', '20130104').append(pd.date_range('20130106', '20130110', freq='2D'))
            TimeSeries(pd.Series(range(7), index=range_))

        range_ = pd.date_range('20130101', '20130104').append(pd.date_range('20130106', '20130110'))
        series_test = TimeSeries(pd.Series(range(9), index=range_))
        self.assertEqual(series_test.freq_str(), 'D')

        range_ = pd.date_range('20130101', '20130104', freq='2D') \
            .append(pd.date_range('20130107', '20130111', freq='2D'))
        series_test = TimeSeries(pd.Series(range(5), index=range_))
        self.assertEqual(series_test.freq_str(), '2D')
        self.assertEqual(series_test.start_time(), range_[0])
        self.assertEqual(series_test.end_time(), range_[-1])
        self.assertTrue(math.isnan(series_test.pd_series().get('20130105')))

    def test_resample_timeseries(self):
        times = pd.date_range('20130101', '20130110')
        pd_series = pd.Series(range(10), index=times)
        timeseries = TimeSeries(pd_series)

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
            TimeSeries(pd.Series(), freq='D')
        # test frequency mismatch error
        with self.assertRaises(ValueError):
            TimeSeries.from_times_and_values(pd.date_range('20130101', '20130105'), range(5), freq='M')
        # test successful instantiation of TimeSeries with length 2
        TimeSeries.from_times_and_values(pd.date_range('20130101', '20130102'), range(2), freq='D')
