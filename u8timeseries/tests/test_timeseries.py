import unittest
import pandas as pd
import numpy as np

from ..timeseries import TimeSeries


class TimeSeriesTestCase(unittest.TestCase):

    times = pd.date_range('20130101', '20130110')
    pd_series1 = pd.Series(range(10), index=times)
    pd_series2 = pd.Series(range(5, 15), index=times)
    pd_series3 = pd.Series(range(15, 25), index=times)
    series1: TimeSeries = TimeSeries(pd_series1)
    series2: TimeSeries = TimeSeries(pd_series1, pd_series2, pd_series3)
    series3: TimeSeries = TimeSeries(pd_series2)

    def test_creation(self):
        with self.assertRaises(AssertionError):
            # Index is dateTimeIndex
            TimeSeries(pd.Series(range(10), range(10)))

        with self.assertRaises(AssertionError):
            # Conf interval must be same length as main series
            pd_lo = pd.Series(range(5, 14), index=pd.date_range('20130101', '20130109'))
            TimeSeries(self.pd_series1, pd_lo)

        with self.assertRaises(AssertionError):
            # Conf interval must have same time index as main series
            pd_lo = pd.Series(range(5, 15), index=pd.date_range('20130102', '20130111'))
            TimeSeries(self.pd_series1, pd_lo)

        with self.assertRaises(AssertionError):
            # Conf interval must be same length as main series
            pd_hi = pd.Series(range(5, 14), index=pd.date_range('20130101', '20130109'))
            TimeSeries(self.pd_series1, None, pd_hi)

        with self.assertRaises(AssertionError):
            # Conf interval must have same time index as main series
            pd_lo = pd.Series(range(5, 15), index=pd.date_range('20130102', '20130111'))
            TimeSeries(self.pd_series1, None, pd_lo)

        with self.assertRaises(AssertionError):
            # Main series cannot have date holes
            range_ = pd.date_range('20130101', '20130104').append(pd.date_range('20130106', '20130110'))
            TimeSeries(pd.Series(range(9), index=range_))

        series_test = TimeSeries(self.pd_series1, self.pd_series2, self.pd_series3)

        self.assertTrue(series_test.pd_series().equals(self.pd_series1))
        self.assertTrue(series_test.conf_lo_pd_series().equals(self.pd_series2))
        self.assertTrue(series_test.conf_hi_pd_series().equals(self.pd_series3))

    def test_alt_creation(self):
        with self.assertRaises(AssertionError):
            # Series cannot be lower than three
            index = pd.date_range('20130101', '20130102')
            TimeSeries.from_times_and_values(index, self.pd_series1.values[:2])
        with self.assertRaises(ValueError):
            # all array must have same length
            TimeSeries.from_times_and_values(self.pd_series1.index,
                                             self.pd_series1.values[:-1],
                                             self.pd_series2[:-2],
                                             self.pd_series3[:-1])

        # test if reordering is correct
        rand_perm = np.random.permutation(range(1, 11))
        index = pd.to_datetime(['201301{:02d}'.format(i) for i in rand_perm])
        series_test = TimeSeries.from_times_and_values(index, self.pd_series1.values[rand_perm-1],
                                                       self.pd_series2[rand_perm-1],
                                                       self.pd_series3[rand_perm-1].tolist())

        self.assertTrue(series_test.start_time() == pd.to_datetime('20130101'))
        self.assertTrue(series_test.end_time() == pd.to_datetime('20130110'))
        self.assertTrue(series_test.pd_series().equals(self.pd_series1))
        self.assertTrue(series_test.conf_lo_pd_series().equals(self.pd_series2))
        self.assertTrue(series_test.conf_hi_pd_series().equals(self.pd_series3))
        self.assertTrue(series_test.freq() == self.series1.freq())

    # TODO test over to_dataframe when multiple features choice is decided

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

        # compare with both CI
        seriesD: TimeSeries = TimeSeries(self.pd_series1, self.pd_series2, self.pd_series3)
        seriesE: TimeSeries = TimeSeries(self.pd_series1, self.pd_series3, self.pd_series2)
        self.assertTrue(self.series2 == seriesD)
        self.assertFalse(self.series2 == seriesE)

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

        # with CI
        seriesH = self.series2.slice(pd.Timestamp('20130104'), pd.Timestamp('20130107'))
        self.assertEqual(seriesH.conf_lo_pd_series().index[0], pd.Timestamp('20130104'))
        self.assertEqual(seriesH.conf_lo_pd_series().index[-1], pd.Timestamp('20130107'))
        self.assertEqual(seriesH.conf_hi_pd_series().index[0], pd.Timestamp('20130104'))
        self.assertEqual(seriesH.conf_hi_pd_series().index[-1], pd.Timestamp('20130107'))

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

        seriesB = self.series1.intersect(seriesA)
        self.assertEqual(seriesB.start_time(), pd.Timestamp('20130102'))
        self.assertEqual(seriesB.end_time(), pd.Timestamp('20130107'))

        # The same, with CI
        seriesC = self.series2.intersect(seriesA)
        self.assertEqual(seriesC.conf_lo_pd_series().index[0], pd.Timestamp('20130102'))
        self.assertEqual(seriesC.conf_hi_pd_series().index[-1], pd.Timestamp('20130107'))

        # Outside of range
        seriesD = self.series1.intersect(TimeSeries(pd.Series(range(6, 13),
                                                              index=pd.date_range('20130106', '20130112'))))
        self.assertEqual(seriesD.start_time(), pd.Timestamp('20130106'))
        self.assertEqual(seriesD.end_time(), pd.Timestamp('20130110'))

        # No intersect or too small intersect
        with self.assertRaises(AssertionError):
            self.series1.intersect(TimeSeries(pd.Series(range(6, 13),
                                                        index=pd.date_range('20130116', '20130122'))))
        with self.assertRaises(AssertionError):
            self.series1.intersect(TimeSeries(pd.Series(range(9, 13),
                                                        index=pd.date_range('20130109', '20130112'))))

    def test_rescale(self):
        with self.assertRaises(AssertionError):
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
                        self.series1.time_index()[1:].append(pd.DatetimeIndex([self.series1.time_index()[-1] +
                                                                              self.series1.freq()]))))

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
        with self.assertRaises(AssertionError):
            seriesA.append(seriesC)

        # Changing frequence is not allowed
        seriesM = TimeSeries.from_times_and_values(pd.date_range('20130107', '20130507', freq='30D'), range(5))
        with self.assertRaises(AssertionError):
            seriesA.append(seriesM)

        # reconstruction with CI
        seriesD, seriesE = self.series2.split_after(pd.Timestamp('20130106'))
        self.assertEqual(seriesD.append(seriesE), self.series2)
        self.assertEqual(seriesD.append(seriesE).freq(), self.series2.freq())

    def test_append_values(self):
        # reconstruct series
        seriesA, seriesB = self.series1.split_after(pd.Timestamp('20130106'))
        self.assertEqual(seriesA.append_values(seriesB.values(), seriesB.time_index()), self.series1)
        self.assertEqual(seriesA.append_values(seriesB.values()), self.series1)

        # same with CI
        seriesC, seriesD = self.series2.split_after(pd.Timestamp('20130106'))
        self.assertEqual(seriesC.append_values(seriesD.values(), seriesD.time_index(),
                                     seriesD.conf_lo_pd_series().values,
                                     seriesD.conf_hi_pd_series().values), self.series2)

        # add only few element
        self.assertEqual(self.series1.drop_after(pd.Timestamp('20130110')).append_values([9]), self.series1)
        self.assertEqual(seriesA.append_values([]), seriesA)

        # randomize order
        rd_order = np.random.permutation(range(len(seriesB.values())))
        self.assertEqual(seriesA.append_values(seriesB.values()[rd_order], seriesB.time_index()[rd_order]),
                         self.series1)

        # add non consecutive index
        with self.assertRaises(AssertionError):
            self.assertEqual(seriesA.append_values(seriesB.values(), seriesB.time_index()+seriesB.freq()), self.series1)

        # add existing indices
        with self.assertRaises(AssertionError):
            self.assertEqual(seriesA.append_values(seriesB.values(), seriesB.time_index()-3*seriesB.freq()), self.series1)

        # other frequency
        with self.assertRaises(AssertionError):
            self.assertEqual(seriesA.append_values(seriesB.values(), pd.date_range('20130107', '20130113', freq='2d')),
                             self.series1)

    def test_update(self):
        seriesA: TimeSeries = TimeSeries.from_times_and_values(self.times, [0, 1, 1, 3, 4, 5, 6, 2, 8, 0])
        seriesB: TimeSeries = TimeSeries.from_times_and_values(self.times, range(10),
                                                               [5, 1, 7, 3, 9, 5, 11, 2, 13, 14],
                                                               [15, 16, 1, 18, 4, 20, 6, 22, 8, 24])
        # change nothing
        seriesC = self.series1.copy()
        with self.assertRaises(AssertionError):
            seriesA.update(self.times)
        seriesC.update(self.times, range(10))
        self.assertEqual(seriesC, self.series1)

        # different len
        with self.assertRaises(AssertionError):
            seriesA.update(self.times, [], None, None)
        with self.assertRaises(AssertionError):
            seriesA.update(self.times, None, np.arange(3), None)
        with self.assertRaises(AssertionError):
            seriesA.update(self.times, None, None, np.arange(4))

        # change outside
        seriesC = seriesA.copy()
        with self.assertRaises(AssertionError):
            seriesC.update(self.times+100*seriesC.freq(), range(10))
        seriesC.update(self.times.append(pd.date_range('20140101', '20140110')), list(range(10))+[0]*10)
        self.assertEqual(seriesC, self.series1)

        # change random
        seriesC = seriesA.copy()
        seriesC.update(pd.DatetimeIndex(['20130108', '20130110', '20130103']), [7, 9, 2])
        self.assertEqual(seriesC, self.series1)

        # change one of each series
        seriesD = seriesB.copy()
        seriesD.update(self.times, seriesA.pd_series())
        seriesA.update(pd.DatetimeIndex(['20130103', '20130108', '20130110']), [2, 7, 9])
        self.assertEqual(seriesA, self.series1)
        seriesB.update(self.times[::2], conf_hi=range(15, 25, 2))
        self.assertTrue(seriesB.conf_hi_pd_series().equals(self.series2.conf_hi_pd_series()))
        self.assertNotEqual(seriesB, self.series2)
        seriesB.update(self.times[1::2], conf_lo=range(6, 15, 2))
        self.assertEqual(seriesB, self.series2)

        # use nan to update all series altogether
        new_series = np.empty(10)
        new_series[:] = np.nan
        new_series[[2, 7, 9]] = [2, 7, 9]
        new_lo = np.empty(10)
        new_lo[:] = np.nan
        new_lo[1::2] = np.arange(6, 15, 2)
        new_hi = np.empty(10)
        new_hi[:] = np.nan
        new_hi[::2] = np.arange(15, 25, 2)
        seriesD.update(self.times, new_series, new_lo, new_hi)
        self.assertEqual(seriesD, self.series2)

        # raise error when update missing CI
        with self.assertRaises(AttributeError):
            self.series1.update(self.times, conf_lo=range(5, 15))

    def test_drop_values(self):
        seriesA = self.series1.append_values([1])
        self.assertEqual(seriesA.drop_values(pd.Timestamp('20130111'), inplace=False), self.series1)
        seriesA.drop_values(pd.Timestamp('20130111'))
        self.assertEqual(seriesA, self.series1)

        with self.assertRaises(KeyError):
            seriesA.drop_values(pd.Timestamp('20130112'))

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

    def test_getitem(self):
        seriesA: TimeSeries = self.series1.drop_after(pd.Timestamp("20130105"))
        self.assertEqual(self.series1[pd.date_range('20130101', ' 20130104')], seriesA)
        self.assertEqual(self.series1[:4], seriesA)
        self.assertTrue(self.series1[pd.Timestamp('20130101')].equals(self.series1.pd_series()[:1]))
        self.assertEqual(self.series1[pd.Timestamp('20130101'):pd.Timestamp('20130105')], seriesA)

        with self.assertRaises(IndexError):
            self.series1[pd.date_range('19990101', '19990201')]

        with self.assertRaises(IndexError):
            self.series1['19990101']

        with self.assertRaises(IndexError):
            self.series1[::-1]


