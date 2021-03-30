import math

import numpy as np
import pandas as pd

from .base_test_class import DartsBaseTestClass
from ..timeseries import TimeSeries
from ..utils.timeseries_generation import linear_timeseries, constant_timeseries


class TimeSeriesTestCase(DartsBaseTestClass):

    times = pd.date_range('20130101', '20130110')
    pd_series1 = pd.Series(range(10), index=times)
    pd_series2 = pd.Series(range(5, 15), index=times)
    pd_series3 = pd.Series(range(15, 25), index=times)
    series1: TimeSeries = TimeSeries.from_series(pd_series1)
    series2: TimeSeries = TimeSeries.from_series(pd_series2)
    series3: TimeSeries = TimeSeries.from_series(pd_series2)

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

        seriesE, seriesF = test_series.split_after(0.7)
        test_case.assertEqual(len(seriesE), round(0.7 * len(test_series)))
        test_case.assertEqual(len(seriesF), round(0.3 * len(test_series)))

        seriesG, seriesH = test_series.split_before(0.7)
        test_case.assertEqual(len(seriesG), round(0.7 * len(test_series)) - 1)
        test_case.assertEqual(len(seriesH), round(0.3 * len(test_series)) + 1)

        seriesI, seriesJ = test_series.split_after(5)
        test_case.assertEqual(len(seriesI), 6)
        test_case.assertEqual(len(seriesJ), len(test_series) - 6)

        seriesK, seriesL = test_series.split_before(5)
        test_case.assertEqual(len(seriesK), 5)
        test_case.assertEqual(len(seriesL), len(test_series) - 5)

        test_case.assertEqual(test_series.freq_str(), seriesA.freq_str())
        test_case.assertEqual(test_series.freq_str(), seriesC.freq_str())
        test_case.assertEqual(test_series.freq_str(), seriesE.freq_str())
        test_case.assertEqual(test_series.freq_str(), seriesG.freq_str())
        test_case.assertEqual(test_series.freq_str(), seriesI.freq_str())
        test_case.assertEqual(test_series.freq_str(), seriesK.freq_str())

        # Test split points outside of range
        for value in [-5, 1.1, pd.Timestamp('21300104')]:
            with test_case.assertRaises(ValueError):
                test_series.split_before(value)

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

    def test_diff(self):
        diff1 = TimeSeries.from_dataframe(self.series1._df.diff())
        diff2 = TimeSeries.from_dataframe(diff1._df.diff())
        diff1_no_na = TimeSeries.from_dataframe(diff1._df.dropna())
        diff2_no_na = TimeSeries.from_dataframe(diff2._df.dropna())

        diff_shift2 = TimeSeries.from_dataframe(self.series1._df.diff(periods=2))
        diff_shift2_no_na = TimeSeries.from_dataframe(self.series1._df.diff(periods=2).dropna())

        diff2_shift2 = TimeSeries.from_dataframe(diff_shift2._df.diff(periods=2))
        diff2_shift2_no_na = TimeSeries.from_dataframe(diff2_shift2._df.diff(periods=2).dropna())

        with self.assertRaises(ValueError):
            self.series1.diff(n=0)
        with self.assertRaises(ValueError):
            self.series1.diff(n=-5)
        with self.assertRaises(ValueError):
            self.series1.diff(n=0.2)
        with self.assertRaises(ValueError):
            self.series1.diff(periods=0.2)

        self.assertEqual(self.series1.diff(), diff1_no_na)
        self.assertEqual(self.series1.diff(n=2), diff2_no_na)
        self.assertEqual(self.series1.diff(dropna=False), diff1)
        self.assertEqual(self.series1.diff(n=2, dropna=0), diff2)
        self.assertEqual(self.series1.diff(periods=2), diff_shift2_no_na)
        self.assertEqual(self.series1.diff(n=2, periods=2, dropna=False), diff2_shift2)

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
        self.assertTrue(self.series1[pd.Timestamp('20130101')] == TimeSeries(self.series1.pd_dataframe()[:1],
                                                                             freq=self.series1.freq()))
        self.assertEqual(self.series1[pd.Timestamp('20130101'):pd.Timestamp('20130104')], seriesA)

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
        # test frequency mismatch case
        seriesA = TimeSeries.from_times_and_values(pd.date_range('20130101', '20130105'), range(5), freq='M')
        self.assertEqual(seriesA.freq(), 'D')
        # test successful instantiation of TimeSeries with length 2
        TimeSeries.from_times_and_values(pd.date_range('20130101', '20130102'), range(2), freq='D')

    def test_from_dataframe(self):
        data_dict = {"Time": pd.date_range(start="20180501", end="20200301", freq="MS")}
        data_dict["Values1"] = np.random.uniform(low=-10, high=10, size=len(data_dict["Time"]))
        data_dict["Values2"] = np.random.uniform(low=0, high=1, size=len(data_dict["Time"]))

        data_pd1 = pd.DataFrame(data_dict)
        data_pd2 = data_pd1.copy()
        data_pd2["Time"] = data_pd2["Time"].apply(lambda date: str(date))
        data_pd3 = data_pd1.set_index("Time")

        data_darts1 = TimeSeries.from_dataframe(df=data_pd1, time_col="Time")
        data_darts2 = TimeSeries.from_dataframe(df=data_pd2, time_col="Time")
        data_darts3 = TimeSeries.from_dataframe(df=data_pd3)

        self.assertEqual(data_darts1, data_darts2)
        self.assertEqual(data_darts1, data_darts3)

    def test_create_dummy_index(self):
        times = pd.date_range(start="20210312", periods=15, freq="MS")
        values1 = np.random.uniform(low=-10, high=10, size=len(times))
        values2 = np.random.uniform(low=0, high=1, size=len(times))

        df1 = pd.DataFrame({"V1": values1, "V2": values2})
        df2 = pd.DataFrame({"V1": values1, "V2": values2}, index=times)
        df3 = pd.DataFrame({"V1": values1, "V2": values2, "Time": times})
        series1 = pd.Series(values1)
        series2 = pd.Series(values1, index=times)

        with self.assertRaises(ValueError):
            TimeSeries(df2, dummy_index=True) # Has DatetimeIndex
        self.assertFalse(TimeSeries(df2).has_dummy_index)

        with self.assertRaises(ValueError):
            TimeSeries.from_dataframe(df3, time_col="Time", dummy_index=True)
        with self.assertRaises(ValueError):
            TimeSeries.from_dataframe(df3, dummy_index=True) # Has "Time" column which is not numerical
        self.assertFalse(TimeSeries.from_dataframe(df3, time_col="Time").has_dummy_index)

        ts_df1_1 = TimeSeries(df1, dummy_index=True)
        ts_df1_2 = TimeSeries.from_dataframe(df1, dummy_index=True)
        ts_series1 = TimeSeries.from_series(series1, dummy_index=True)

        self.assertTrue(ts_df1_1.has_dummy_index)
        self.assertTrue(ts_df1_2.has_dummy_index)
        self.assertTrue(ts_series1.has_dummy_index)

        self.assertEqual(ts_df1_1, ts_df1_2)
        self.assertEqual(
            ts_df1_1,
            TimeSeries(pd.DataFrame(
                {"V1": values1, "V2": values2},
                index=pd.date_range(start="19700101", periods=len(ts_df1_1), freq="S")
                )
            )
        )

        self.assertEqual(
            ts_series1,
            TimeSeries.from_times_and_values(
                times=pd.date_range(start="19700101", periods=len(ts_series1), freq="S"),
                values=values1
            )
        )

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

    def test_map(self):
        fn = np.sin  # noqa: E731
        series = TimeSeries.from_times_and_values(pd.date_range('20000101', '20000110'), np.random.randn(10, 3))

        df_0 = series.pd_dataframe()
        df_2 = series.pd_dataframe()
        df_01 = series.pd_dataframe()
        df_012 = series.pd_dataframe()

        df_0[["0"]] = df_0[["0"]].applymap(fn)
        df_2[["2"]] = df_2[["2"]].applymap(fn)
        df_01[["0", "1"]] = df_01[["0", "1"]].applymap(fn)
        df_012 = df_012.applymap(fn)

        series_0 = TimeSeries(df_0, 'D')
        series_2 = TimeSeries(df_2, 'D')
        series_01 = TimeSeries(df_01, 'D')
        series_012 = TimeSeries(df_012, 'D')

        self.assertEqual(series_0['0'], series['0'].map(fn))
        self.assertEqual(series_2['2'], series['2'].map(fn))
        self.assertEqual(series_01[['0', '1']], series[['0', '1']].map(fn))
        self.assertEqual(series_012, series[['0', '1', '2']].map(fn))
        self.assertEqual(series_012, series.map(fn))

        self.assertNotEqual(series_01, series[['0', '1']].map(fn))

    def test_map_with_timestamp(self):
        series = linear_timeseries(start_value=1, length=12, freq='MS', start_ts=pd.Timestamp('2000-01-01'), end_value=12)  # noqa: E501
        zeroes = constant_timeseries(value=0.0, length=12, freq='MS', start_ts=pd.Timestamp('2000-01-01'))

        def function(ts, x):
            return x - ts.month

        new_series = series.map(function)
        self.assertEqual(new_series, zeroes)

    def test_map_wrong_fn(self):
        series = linear_timeseries(start_value=1, length=12, freq='MS', start_ts=pd.Timestamp('2000-01-01'), end_value=12)  # noqa: E501

        def add(x, y, z):
            return x + y + z

        with self.assertRaises(ValueError):
            series.map(add)

        ufunc_add = np.frompyfunc(add, 3, 1)

        with self.assertRaises(ValueError):
            series.map(ufunc_add)

    def test_gaps(self):
        times1 = pd.date_range('20130101', '20130110')
        times2 = pd.date_range('20120101', '20210301', freq="Q")
        times3 = pd.date_range('20120101', '20210301', freq="AS")
        times4 = pd.date_range('20120101', '20210301', freq="2MS")

        pd_series1 = pd.Series([1, 1] + 3 * [np.nan] + [1, 1, 1] + [np.nan] * 2, index=times1)
        pd_series2 = pd.Series([1, 1] + 3 * [np.nan] + [1, 1] + [np.nan] * 3, index=times1)
        pd_series3 = pd.Series([np.nan] * 10, index=times1)
        pd_series4 = pd.Series([1]*5 + 3*[np.nan] + [1]*18 + 7*[np.nan] + [1, 1] + [np.nan], index=times2)
        pd_series5 = pd.Series([1]*3 + 2*[np.nan] + [1] + 2*[np.nan] + [1, 1], index=times3)
        pd_series6 = pd.Series([1]*10 + 1*[np.nan] + [1]*13 + 5*[np.nan] + [1]*18 + 9*[np.nan], index=times4)

        series1 = TimeSeries.from_series(pd_series1)
        series2 = TimeSeries.from_series(pd_series2)
        series3 = TimeSeries.from_series(pd_series3)
        series4 = TimeSeries.from_series(pd_series4)
        series5 = TimeSeries.from_series(pd_series5)
        series6 = TimeSeries.from_series(pd_series6)

        gaps1 = series1.gaps()
        self.assertTrue((gaps1['gap_start'] == pd.DatetimeIndex([pd.Timestamp('20130103'),
                                                                 pd.Timestamp('20130109')])).all())
        self.assertTrue((gaps1['gap_end'] == pd.DatetimeIndex([pd.Timestamp('20130105'),
                                                               pd.Timestamp('20130110')])).all())
        self.assertEqual(gaps1['gap_size'].values.tolist(), [3, 2])
        gaps2 = series2.gaps()
        self.assertEqual(gaps2['gap_size'].values.tolist(), [3, 3])
        gaps3 = series3.gaps()
        self.assertEqual(gaps3['gap_size'].values.tolist(), [10])
        gaps4 = series4.gaps()
        self.assertEqual(gaps4['gap_size'].values.tolist(), [3, 7, 1])
        gaps5 = series5.gaps()
        self.assertEqual(gaps5['gap_size'].values.tolist(), [2, 2])
        self.assertTrue((gaps5['gap_start'] == pd.DatetimeIndex([pd.Timestamp('20150101'),
                                                                 pd.Timestamp('20180101')])).all())
        self.assertTrue((gaps5['gap_end'] == pd.DatetimeIndex([pd.Timestamp('20160101'),
                                                                pd.Timestamp('20190101')])).all())
        gaps6 = series6.gaps()
        self.assertEqual(gaps6['gap_size'].values.tolist(), [1, 5, 9])
        self.assertTrue((gaps6['gap_start'] == pd.DatetimeIndex([pd.Timestamp('20130901'),
                                                                 pd.Timestamp('20160101'),
                                                                 pd.Timestamp('20191101')])).all())
        self.assertTrue((gaps6['gap_end'] == pd.DatetimeIndex([pd.Timestamp('20130901'),
                                                                pd.Timestamp('20160901'),
                                                                pd.Timestamp('20210301')])).all())

    def test_longest_contiguous_slice(self):
        times = pd.date_range('20130101', '20130111')
        pd_series1 = pd.Series([1, 1] + 3 * [np.nan] + [1, 1, 1] + [np.nan] * 2 + [1], index=times)
        series1 = TimeSeries.from_series(pd_series1)

        self.assertEqual(len(series1.longest_contiguous_slice()), 3)
        self.assertEqual(len(series1.longest_contiguous_slice(2)), 6)
