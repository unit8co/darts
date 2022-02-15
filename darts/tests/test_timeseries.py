import math
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import skew, kurtosis

from darts import TimeSeries, concatenate
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.timeseries_generation import constant_timeseries, linear_timeseries


class TimeSeriesTestCase(DartsBaseTestClass):

    times = pd.date_range("20130101", "20130110", freq="D")
    pd_series1 = pd.Series(range(10), index=times)
    pd_series2 = pd.Series(range(5, 15), index=times)
    pd_series3 = pd.Series(range(15, 25), index=times)
    series1: TimeSeries = TimeSeries.from_series(pd_series1)
    series2: TimeSeries = TimeSeries.from_series(pd_series2)
    series3: TimeSeries = TimeSeries.from_series(pd_series2)

    def test_creation(self):
        series_test = TimeSeries.from_series(self.pd_series1)
        self.assertTrue(series_test.pd_series().equals(self.pd_series1.astype(float)))

        # Creation with a well formed array:
        ar = xr.DataArray(
            np.random.randn(10, 2, 3),
            dims=("time", "component", "sample"),
            coords={"time": self.times, "component": ["a", "b"]},
            name="time series",
        )
        ts = TimeSeries(ar)
        self.assertTrue(ts.is_stochastic)

        ar = xr.DataArray(
            np.random.randn(10, 2, 1),
            dims=("time", "component", "sample"),
            coords={"time": pd.RangeIndex(0, 10, 1), "component": ["a", "b"]},
            name="time series",
        )
        ts = TimeSeries(ar)
        self.assertTrue(ts.is_deterministic)

        # creation with ill-formed arrays
        with self.assertRaises(ValueError):
            ar2 = xr.DataArray(
                np.random.randn(10, 2, 1),
                dims=("time", "wrong", "sample"),
                coords={"time": self.times, "wrong": ["a", "b"]},
                name="time series",
            )
            _ = TimeSeries(ar2)

        with self.assertRaises(ValueError):
            # duplicated column names
            ar3 = xr.DataArray(
                np.random.randn(10, 2, 1),
                dims=("time", "component", "sample"),
                coords={"time": self.times, "component": ["a", "a"]},
                name="time series",
            )
            _ = TimeSeries(ar3)

        # creation using from_xarray()
        ar = xr.DataArray(
            np.random.randn(10, 2, 1),
            dims=("time", "component", "sample"),
            coords={"time": self.times, "component": ["a", "b"]},
            name="time series",
        )
        _ = TimeSeries.from_xarray(ar)

    def test_integer_indexing(self):
        # sanity checks for the integer-indexed series
        range_indexed_data = np.random.randn(
            50,
        )
        series_int: TimeSeries = TimeSeries.from_values(range_indexed_data)

        self.assertTrue(series_int[0].values().item() == range_indexed_data[0])
        self.assertTrue(series_int[10].values().item() == range_indexed_data[10])

        self.assertTrue(
            np.all(series_int[10:20].univariate_values() == range_indexed_data[10:20])
        )
        self.assertTrue(
            np.all(series_int[10:].univariate_values() == range_indexed_data[10:])
        )

        self.assertTrue(
            np.all(
                series_int[pd.RangeIndex(start=10, stop=40, step=1)].univariate_values()
                == range_indexed_data[10:40]
            )
        )

        # check the RangeIndex when indexing with a list
        indexed_ts = series_int[[2, 3, 4, 5, 6]]
        self.assertTrue(isinstance(indexed_ts.time_index, pd.RangeIndex))
        self.assertTrue(
            list(indexed_ts.time_index) == list(pd.RangeIndex(2, 7, step=1))
        )

    def test_column_names(self):
        # test the column names resolution
        columns_before = [
            ["0", "1", "2"],
            ["v", "v", "x"],
            ["v", "v", "x", "v"],
            ["0", "0_1", "0"],
            ["0", "0_1", "0", "0_1_1"],
        ]
        columns_after = [
            ["0", "1", "2"],
            ["v", "v_1", "x"],
            ["v", "v_1", "x", "v_2"],
            ["0", "0_1", "0_1_1"],
            ["0", "0_1", "0_1_1", "0_1_1_1"],
        ]
        for cs_before, cs_after in zip(columns_before, columns_after):
            ar = xr.DataArray(
                np.random.randn(10, len(cs_before), 2),
                dims=("time", "component", "sample"),
                coords={"time": self.times, "component": cs_before},
            )
            ts = TimeSeries.from_xarray(ar)
            self.assertEqual(ts.columns.tolist(), cs_after)

    def test_quantiles(self):
        values = np.random.rand(10, 2, 1000)
        ar = xr.DataArray(
            values,
            dims=("time", "component", "sample"),
            coords={"time": self.times, "component": ["a", "b"]},
        )
        ts = TimeSeries(ar)

        for q in [0.01, 0.1, 0.5, 0.95]:
            q_ts = ts.quantile_timeseries(quantile=q)
            self.assertTrue(
                (abs(q_ts.values() - np.quantile(values, q=q, axis=2)) < 1e-3).all()
            )

    def test_alt_creation(self):
        with self.assertRaises(ValueError):
            # Series cannot be lower than three without passing frequency as argument to constructor,
            # if fill_missing_dates is True (otherwise it works)
            index = pd.date_range("20130101", "20130102")
            TimeSeries.from_times_and_values(
                index, self.pd_series1.values[:2], fill_missing_dates=True
            )
        with self.assertRaises(ValueError):
            # all arrays must have same length
            TimeSeries.from_times_and_values(
                self.pd_series1.index, self.pd_series1.values[:-1]
            )

        # test if reordering is correct
        rand_perm = np.random.permutation(range(1, 11))
        index = pd.to_datetime([f"201301{i:02d}" for i in rand_perm])
        series_test = TimeSeries.from_times_and_values(
            index, self.pd_series1.values[rand_perm - 1]
        )

        self.assertTrue(series_test.start_time() == pd.to_datetime("20130101"))
        self.assertTrue(series_test.end_time() == pd.to_datetime("20130110"))
        self.assertTrue(all(series_test.pd_series().values == self.pd_series1.values))
        self.assertTrue(series_test.freq == self.series1.freq)

    # TODO test over to_dataframe when multiple features choice is decided

    def test_eq(self):
        seriesA: TimeSeries = TimeSeries.from_series(self.pd_series1)
        self.assertTrue(self.series1 == seriesA)
        self.assertFalse(self.series1 != seriesA)

        # with different dates
        seriesC = TimeSeries.from_series(
            pd.Series(range(10), index=pd.date_range("20130102", "20130111"))
        )
        self.assertFalse(self.series1 == seriesC)

    def test_dates(self):
        self.assertEqual(self.series1.start_time(), pd.Timestamp("20130101"))
        self.assertEqual(self.series1.end_time(), pd.Timestamp("20130110"))
        self.assertEqual(self.series1.duration, pd.Timedelta(days=9))

    @staticmethod
    def helper_test_slice(test_case, test_series: TimeSeries):
        # base case
        seriesA = test_series.slice(pd.Timestamp("20130104"), pd.Timestamp("20130107"))
        test_case.assertEqual(seriesA.start_time(), pd.Timestamp("20130104"))
        test_case.assertEqual(seriesA.end_time(), pd.Timestamp("20130107"))

        # time stamp not in series
        seriesB = test_series.slice(
            pd.Timestamp("20130104 12:00:00"), pd.Timestamp("20130107")
        )
        test_case.assertEqual(seriesB.start_time(), pd.Timestamp("20130105"))
        test_case.assertEqual(seriesB.end_time(), pd.Timestamp("20130107"))

        # end timestamp after series
        seriesC = test_series.slice(pd.Timestamp("20130108"), pd.Timestamp("20130201"))
        test_case.assertEqual(seriesC.start_time(), pd.Timestamp("20130108"))
        test_case.assertEqual(seriesC.end_time(), pd.Timestamp("20130110"))

        # n points, base case
        seriesD = test_series.slice_n_points_after(pd.Timestamp("20130102"), n=3)
        test_case.assertEqual(seriesD.start_time(), pd.Timestamp("20130102"))
        test_case.assertTrue(len(seriesD.values()) == 3)
        test_case.assertEqual(seriesD.end_time(), pd.Timestamp("20130104"))

        seriesE = test_series.slice_n_points_after(
            pd.Timestamp("20130107 12:00:10"), n=10
        )
        test_case.assertEqual(seriesE.start_time(), pd.Timestamp("20130108"))
        test_case.assertEqual(seriesE.end_time(), pd.Timestamp("20130110"))

        seriesF = test_series.slice_n_points_before(pd.Timestamp("20130105"), n=3)
        test_case.assertEqual(seriesF.end_time(), pd.Timestamp("20130105"))
        test_case.assertTrue(len(seriesF.values()) == 3)
        test_case.assertEqual(seriesF.start_time(), pd.Timestamp("20130103"))

        seriesG = test_series.slice_n_points_before(
            pd.Timestamp("20130107 12:00:10"), n=10
        )
        test_case.assertEqual(seriesG.start_time(), pd.Timestamp("20130101"))
        test_case.assertEqual(seriesG.end_time(), pd.Timestamp("20130107"))

    @staticmethod
    def helper_test_split(test_case, test_series: TimeSeries):
        seriesA, seriesB = test_series.split_after(pd.Timestamp("20130104"))
        test_case.assertEqual(seriesA.end_time(), pd.Timestamp("20130104"))
        test_case.assertEqual(seriesB.start_time(), pd.Timestamp("20130105"))

        seriesC, seriesD = test_series.split_before(pd.Timestamp("20130104"))
        test_case.assertEqual(seriesC.end_time(), pd.Timestamp("20130103"))
        test_case.assertEqual(seriesD.start_time(), pd.Timestamp("20130104"))

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

        test_case.assertEqual(test_series.freq_str, seriesA.freq_str)
        test_case.assertEqual(test_series.freq_str, seriesC.freq_str)
        test_case.assertEqual(test_series.freq_str, seriesE.freq_str)
        test_case.assertEqual(test_series.freq_str, seriesG.freq_str)
        test_case.assertEqual(test_series.freq_str, seriesI.freq_str)
        test_case.assertEqual(test_series.freq_str, seriesK.freq_str)

        # Test split points outside of range
        for value in [-5, 1.1, pd.Timestamp("21300104")]:
            with test_case.assertRaises(ValueError):
                test_series.split_before(value)

    @staticmethod
    def helper_test_drop(test_case, test_series: TimeSeries):
        seriesA = test_series.drop_after(pd.Timestamp("20130105"))
        test_case.assertEqual(
            seriesA.end_time(), pd.Timestamp("20130105") - test_series.freq
        )
        test_case.assertTrue(np.all(seriesA.time_index < pd.Timestamp("20130105")))

        seriesB = test_series.drop_before(pd.Timestamp("20130105"))
        test_case.assertEqual(
            seriesB.start_time(), pd.Timestamp("20130105") + test_series.freq
        )
        test_case.assertTrue(np.all(seriesB.time_index > pd.Timestamp("20130105")))

        test_case.assertEqual(test_series.freq_str, seriesA.freq_str)
        test_case.assertEqual(test_series.freq_str, seriesB.freq_str)

    @staticmethod
    def helper_test_intersect(test_case, test_series: TimeSeries):
        seriesA = TimeSeries.from_series(
            pd.Series(range(2, 8), index=pd.date_range("20130102", "20130107"))
        )

        seriesB = test_series.slice_intersect(seriesA)
        test_case.assertEqual(seriesB.start_time(), pd.Timestamp("20130102"))
        test_case.assertEqual(seriesB.end_time(), pd.Timestamp("20130107"))

        # Outside of range
        seriesD = test_series.slice_intersect(
            TimeSeries.from_series(
                pd.Series(range(6, 13), index=pd.date_range("20130106", "20130112"))
            )
        )
        test_case.assertEqual(seriesD.start_time(), pd.Timestamp("20130106"))
        test_case.assertEqual(seriesD.end_time(), pd.Timestamp("20130110"))

        # Small intersect
        seriesE = test_series.slice_intersect(
            TimeSeries.from_series(
                pd.Series(range(9, 13), index=pd.date_range("20130109", "20130112"))
            )
        )
        test_case.assertEqual(len(seriesE), 2)

        # No intersect
        with test_case.assertRaises(ValueError):
            test_series.slice_intersect(
                TimeSeries(
                    pd.Series(range(6, 13), index=pd.date_range("20130116", "20130122"))
                )
            )

    def test_rescale(self):
        with self.assertRaises(ValueError):
            self.series1.rescale_with_value(1)

        seriesA = self.series3.rescale_with_value(0)
        self.assertTrue(np.all(seriesA.values() == 0))

        seriesB = self.series3.rescale_with_value(-5)
        self.assertTrue(self.series3 * -1.0 == seriesB)

        seriesC = self.series3.rescale_with_value(1)
        self.assertTrue(self.series3 * 0.2 == seriesC)

        seriesD = self.series3.rescale_with_value(
            1e20
        )  # TODO: test will fail if value > 1e24 due to num imprecision
        self.assertTrue(self.series3 * 0.2e20 == seriesD)

    @staticmethod
    def helper_test_shift(test_case, test_series: TimeSeries):
        seriesA = test_case.series1.shift(0)
        test_case.assertTrue(seriesA == test_case.series1)

        seriesB = test_series.shift(1)
        test_case.assertTrue(
            seriesB.time_index.equals(
                test_series.time_index[1:].append(
                    pd.DatetimeIndex([test_series.time_index[-1] + test_series.freq])
                )
            )
        )

        seriesC = test_series.shift(-1)
        test_case.assertTrue(
            seriesC.time_index.equals(
                pd.DatetimeIndex([test_series.time_index[0] - test_series.freq]).append(
                    test_series.time_index[:-1]
                )
            )
        )

        with test_case.assertRaises(OverflowError):
            test_series.shift(1e6)

        seriesM = TimeSeries.from_times_and_values(
            pd.date_range("20130101", "20130601", freq="m"), range(5)
        )
        with test_case.assertRaises(OverflowError):
            seriesM.shift(1e4)

        seriesD = TimeSeries.from_times_and_values(
            pd.date_range("20130101", "20130101"), range(1), freq="D"
        )
        seriesE = seriesD.shift(1)
        test_case.assertEqual(seriesE.time_index[0], pd.Timestamp("20130102"))

        seriesF = TimeSeries.from_times_and_values(pd.RangeIndex(2, 10), range(8))

        seriesG = seriesF.shift(4)
        test_case.assertEqual(seriesG.time_index[0], 6)

    @staticmethod
    def helper_test_append(test_case, test_series: TimeSeries):
        # reconstruct series
        seriesA, seriesB = test_series.split_after(pd.Timestamp("20130106"))
        test_case.assertEqual(seriesA.append(seriesB), test_series)
        test_case.assertEqual(seriesA.append(seriesB).freq, test_series.freq)

        # Creating a gap is not allowed
        seriesC = test_series.drop_before(pd.Timestamp("20130108"))
        with test_case.assertRaises(ValueError):
            seriesA.append(seriesC)

        # Changing frequence is not allowed
        seriesM = TimeSeries.from_times_and_values(
            pd.date_range("20130107", "20130507", freq="30D"), range(5)
        )
        with test_case.assertRaises(ValueError):
            seriesA.append(seriesM)

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

    def test_with_values(self):
        vals = np.random.rand(5, 10, 3)
        series = TimeSeries.from_values(vals)
        series2 = series.with_values(vals + 1)
        series3 = series2.with_values(series2.all_values() - 1)

        # values should work
        np.testing.assert_allclose(series3.all_values(), series.all_values())
        np.testing.assert_allclose(series2.all_values(), vals + 1)

        # should fail if shape is not the same:
        with self.assertRaises(ValueError):
            series.with_values(np.random.rand(5, 10, 2))

    def test_diff(self):
        diff1 = TimeSeries.from_dataframe(self.series1.pd_dataframe().diff())
        diff2 = TimeSeries.from_dataframe(diff1.pd_dataframe().diff())
        diff1_no_na = TimeSeries.from_dataframe(diff1.pd_dataframe().dropna())
        diff2_no_na = TimeSeries.from_dataframe(diff2.pd_dataframe().dropna())

        diff_shift2 = TimeSeries.from_dataframe(
            self.series1.pd_dataframe().diff(periods=2)
        )
        diff_shift2_no_na = TimeSeries.from_dataframe(
            self.series1.pd_dataframe().diff(periods=2).dropna()
        )

        diff2_shift2 = TimeSeries.from_dataframe(
            diff_shift2.pd_dataframe().diff(periods=2)
        )

        with self.assertRaises(ValueError):
            self.series1.diff(n=0)
        with self.assertRaises(ValueError):
            self.series1.diff(n=-5)
        with self.assertRaises(ValueError):
            self.series1.diff(n=0.2)
        with self.assertRaises(ValueError):
            self.series1.diff(periods=0.2)

        self.assertEqual(self.series1.diff(), diff1_no_na)
        self.assertEqual(self.series1.diff(n=2, dropna=True), diff2_no_na)
        self.assertEqual(self.series1.diff(dropna=False), diff1)
        self.assertEqual(self.series1.diff(n=2, dropna=0), diff2)
        self.assertEqual(self.series1.diff(periods=2, dropna=True), diff_shift2_no_na)
        self.assertEqual(self.series1.diff(n=2, periods=2, dropna=False), diff2_shift2)

    def test_ops(self):
        seriesA = TimeSeries.from_series(
            pd.Series([2 for _ in range(10)], index=self.pd_series1.index)
        )
        targetAdd = TimeSeries.from_series(
            pd.Series(range(2, 12), index=self.pd_series1.index)
        )
        targetSub = TimeSeries.from_series(
            pd.Series(range(-2, 8), index=self.pd_series1.index)
        )
        targetMul = TimeSeries.from_series(
            pd.Series(range(0, 20, 2), index=self.pd_series1.index)
        )
        targetDiv = TimeSeries.from_series(
            pd.Series([i / 2 for i in range(10)], index=self.pd_series1.index)
        )
        targetPow = TimeSeries.from_series(
            pd.Series([float(i ** 2) for i in range(10)], index=self.pd_series1.index)
        )

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
        self.assertEqual(self.series1[pd.date_range("20130101", " 20130104")], seriesA)
        self.assertEqual(self.series1[:4], seriesA)
        self.assertTrue(
            self.series1[pd.Timestamp("20130101")]
            == TimeSeries.from_dataframe(
                self.series1.pd_dataframe()[:1], freq=self.series1.freq
            )
        )
        self.assertEqual(
            self.series1[pd.Timestamp("20130101") : pd.Timestamp("20130104")], seriesA
        )

        with self.assertRaises(KeyError):
            self.series1[pd.date_range("19990101", "19990201")]

        with self.assertRaises(KeyError):
            self.series1["19990101"]

        with self.assertRaises(IndexError):
            self.series1[::-1]

    def test_fill_missing_dates(self):
        with self.assertRaises(ValueError):
            # Series cannot have date holes without automatic filling
            range_ = pd.date_range("20130101", "20130104").append(
                pd.date_range("20130106", "20130110")
            )
            TimeSeries.from_series(
                pd.Series(range(9), index=range_), fill_missing_dates=False
            )

        with self.assertRaises(ValueError):
            # Main series should have explicit frequency in case of date holes
            range_ = pd.date_range("20130101", "20130104").append(
                pd.date_range("20130106", "20130110", freq="2D")
            )
            TimeSeries.from_series(
                pd.Series(range(7), index=range_), fill_missing_dates=True
            )

        range_ = pd.date_range("20130101", "20130104").append(
            pd.date_range("20130106", "20130110")
        )
        series_test = TimeSeries.from_series(
            pd.Series(range(9), index=range_), fill_missing_dates=True
        )
        self.assertEqual(series_test.freq_str, "D")

        range_ = pd.date_range("20130101", "20130104", freq="2D").append(
            pd.date_range("20130107", "20130111", freq="2D")
        )
        series_test = TimeSeries.from_series(
            pd.Series(range(5), index=range_), fill_missing_dates=True
        )
        self.assertEqual(series_test.freq_str, "2D")
        self.assertEqual(series_test.start_time(), range_[0])
        self.assertEqual(series_test.end_time(), range_[-1])
        self.assertTrue(math.isnan(series_test.pd_series().get("20130105")))

        # ------ test infer frequency for all offset aliases from ------
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        offset_aliases = [
            "B",
            "C",
            "D",
            "W",
            "M",
            "SM",
            "BM",
            "CBM",
            "MS",
            "SMS",
            "BMS",
            "CBMS",
            "Q",
            "BQ",
            "QS",
            "BQS",
            "A",
            "Y",
            "BA",
            "BY",
            "AS",
            "YS",
            "BAS",
            "BYS",
            "BH",
            "H",
            "T",
            "min",
            "S",
            "L",
            "U",
            "us",
            "N",
        ]
        # fill_missing_dates will find multiple inferred frequencies (i.e. for 'B' it finds {'B', 'D'}) -> good
        offset_aliases_raise = [
            "B",
            "C",
            "SM",
            "BM",
            "CBM",
            "SMS",
            "BMS",
            "CBMS",
            "BQ",
            "BA",
            "BY",
            "BAS",
            "BYS",
            "BH",
            "BQS",
        ]
        # frequency cannot be inferred for these types (finds '15D' instead of 'SM')
        offset_not_supported = ["SM", "SMS"]

        ts_length = 25
        for offset_alias in offset_aliases:
            if offset_alias in offset_not_supported:
                continue

            # test with the initial full DataFrame
            df_full = pd.DataFrame(
                data={
                    "date": pd.date_range(
                        start=pd.to_datetime("01-04-1960"),
                        periods=ts_length,
                        freq=offset_alias,
                    ),
                    "value": np.arange(0, ts_length, 1),
                }
            )
            # test fill dates with DataFrame including holes
            df_holes = pd.concat([df_full[:4], df_full[5:7], df_full[9:]])

            series_target = TimeSeries.from_dataframe(df_full, time_col="date")
            for df, df_name in zip([df_full, df_holes], ["full", "holes"]):

                # fill_missing_dates will find multiple inferred frequencies (i.e. for 'B' it finds {'B', 'D'})
                if offset_alias in offset_aliases_raise:
                    with self.assertRaises(ValueError):
                        _ = TimeSeries.from_dataframe(
                            df, time_col="date", fill_missing_dates=True
                        )
                    continue

                # test with different arguments
                series_out_freq1 = TimeSeries.from_dataframe(
                    df, time_col="date", fill_missing_dates=True, freq=offset_alias
                )
                series_out_freq2 = TimeSeries.from_dataframe(
                    df, time_col="date", freq=offset_alias
                )
                series_out_fill = TimeSeries.from_dataframe(
                    df, time_col="date", fill_missing_dates=True
                )

                for series in [series_out_freq1, series_out_freq2, series_out_fill]:
                    if df_name == "full":
                        self.assertTrue(series == series_target)
                    self.assertTrue(series.time_index.equals(series_target.time_index))

    def test_fillna_value(self):
        range_ = pd.date_range("20130101", "20130108", freq="D")

        pd_series_nan = pd.Series([np.nan] * len(range_), index=range_)
        pd_series_1 = pd.Series([1] * len(range_), index=range_)
        pd_series_holes = pd.concat([pd_series_1[:2], pd_series_nan[3:]])

        series_nan = TimeSeries.from_series(pd_series_nan)
        series_1 = TimeSeries.from_series(pd_series_1)
        series_holes = TimeSeries.from_series(pd_series_holes, fill_missing_dates=True)

        series_nan_fillna = TimeSeries.from_series(pd_series_nan, fillna_value=1.0)
        series_1_fillna = TimeSeries.from_series(pd_series_nan, fillna_value=1.0)
        series_holes_fillna = TimeSeries.from_series(
            pd_series_holes, fill_missing_dates=True, fillna_value=1.0
        )

        for series_with_nan in [series_nan, series_holes]:
            self.assertTrue(np.isnan(series_with_nan.all_values(copy=False)).any())
        for series_no_nan in [
            series_1,
            series_nan_fillna,
            series_1_fillna,
            series_holes_fillna,
        ]:
            self.assertTrue(not np.isnan(series_no_nan.all_values(copy=False)).any())
            self.assertTrue(series_1 == series_no_nan)

    def test_resample_timeseries(self):
        times = pd.date_range("20130101", "20130110")
        pd_series = pd.Series(range(10), index=times)
        timeseries = TimeSeries.from_series(pd_series)

        resampled_timeseries = timeseries.resample("H")
        self.assertEqual(resampled_timeseries.freq_str, "H")
        self.assertEqual(
            resampled_timeseries.pd_series().at[pd.Timestamp("20130101020000")], 0
        )
        self.assertEqual(
            resampled_timeseries.pd_series().at[pd.Timestamp("20130102020000")], 1
        )
        self.assertEqual(
            resampled_timeseries.pd_series().at[pd.Timestamp("20130109090000")], 8
        )

        resampled_timeseries = timeseries.resample("2D")
        self.assertEqual(resampled_timeseries.freq_str, "2D")
        self.assertEqual(
            resampled_timeseries.pd_series().at[pd.Timestamp("20130101")], 0
        )
        with self.assertRaises(KeyError):
            resampled_timeseries.pd_series().at[pd.Timestamp("20130102")]

        self.assertEqual(
            resampled_timeseries.pd_series().at[pd.Timestamp("20130109")], 8
        )

    def test_short_series_creation(self):
        # test missing freq argument error when filling missing dates on short time series
        with self.assertRaises(ValueError):
            TimeSeries.from_times_and_values(
                pd.date_range("20130101", "20130102"), range(2), fill_missing_dates=True
            )
        # test empty pandas series error
        with self.assertRaises(ValueError):
            TimeSeries.from_series(pd.Series(dtype="object"), freq="D")
        # frequency should be ignored when fill_missing_dates is False
        seriesA = TimeSeries.from_times_and_values(
            pd.date_range("20130101", "20130105"),
            range(5),
            fill_missing_dates=False,
            freq="M",
        )
        self.assertEqual(seriesA.freq, "D")
        # test successful instantiation of TimeSeries with length 2
        TimeSeries.from_times_and_values(
            pd.date_range("20130101", "20130102"), range(2), freq="D"
        )

    def test_from_csv(self):
        data_dict = {"Time": pd.date_range(start="20180501", end="20200301", freq="MS")}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        data_dict["Values2"] = np.random.uniform(
            low=0, high=1, size=len(data_dict["Time"])
        )

        data_pd1 = pd.DataFrame(data_dict)

        f1 = NamedTemporaryFile()
        f2 = NamedTemporaryFile()

        # testing two separators to check later if the arguments are passed to the `pd.read_csv`
        data_pd1.to_csv(f1.name, sep=",", index=False)
        data_pd1.to_csv(f2.name, sep=".", index=False)

        # it should be possible to read data given either file object or file path
        f1.seek(0)
        data_darts1 = TimeSeries.from_csv(
            filepath_or_buffer=f1, time_col="Time", sep=","
        )
        data_darts2 = TimeSeries.from_csv(
            filepath_or_buffer=f2.name, time_col="Time", sep="."
        )

        self.assertEqual(data_darts1, data_darts2)

    def test_index_creation(self):
        times = pd.date_range(start="20210312", periods=15, freq="MS")
        values1 = np.random.uniform(low=-10, high=10, size=len(times))
        values2 = np.random.uniform(low=0, high=1, size=len(times))

        df1 = pd.DataFrame({"V1": values1, "V2": values2})
        df2 = pd.DataFrame({"V1": values1, "V2": values2}, index=times)
        df3 = pd.DataFrame({"V1": values1, "V2": values2, "Time": times})
        series1 = pd.Series(values1)
        series2 = pd.Series(values1, index=times)

        ts1 = TimeSeries.from_dataframe(df1)
        self.assertTrue(ts1.has_range_index)

        ts2 = TimeSeries.from_dataframe(df2)
        self.assertTrue(ts2.has_datetime_index)

        ts3 = TimeSeries.from_dataframe(df3, time_col="Time")
        self.assertTrue(ts3.has_datetime_index)

        ts4 = TimeSeries.from_series(series1)
        self.assertTrue(ts4.has_range_index)

        ts5 = TimeSeries.from_series(series2)
        self.assertTrue(ts5.has_datetime_index)

        ts6 = TimeSeries.from_times_and_values(times=times, values=values1)
        self.assertTrue(ts6.has_datetime_index)

        ts7 = TimeSeries.from_times_and_values(times=times, values=df1)
        self.assertTrue(ts7.has_datetime_index)

        ts8 = TimeSeries.from_values(values1)
        self.assertTrue(ts8.has_range_index)

    def test_short_series_slice(self):
        seriesA, seriesB = self.series1.split_after(pd.Timestamp("20130108"))
        self.assertEqual(len(seriesA), 8)
        self.assertEqual(len(seriesB), 2)
        seriesA, seriesB = self.series1.split_after(pd.Timestamp("20130109"))
        self.assertEqual(len(seriesA), 9)
        self.assertEqual(len(seriesB), 1)
        self.assertEqual(seriesB.time_index[0], self.series1.time_index[-1])
        seriesA, seriesB = self.series1.split_before(pd.Timestamp("20130103"))
        self.assertEqual(len(seriesA), 2)
        self.assertEqual(len(seriesB), 8)
        seriesA, seriesB = self.series1.split_before(pd.Timestamp("20130102"))
        self.assertEqual(len(seriesA), 1)
        self.assertEqual(len(seriesB), 9)
        self.assertEqual(seriesA.time_index[-1], self.series1.time_index[0])
        seriesC = self.series1.slice(pd.Timestamp("20130105"), pd.Timestamp("20130105"))
        self.assertEqual(len(seriesC), 1)

    def test_map(self):
        fn = np.sin  # noqa: E731
        series = TimeSeries.from_times_and_values(
            pd.date_range("20000101", "20000110"), np.random.randn(10, 3)
        )

        df_0 = series.pd_dataframe()
        df_2 = series.pd_dataframe()
        df_01 = series.pd_dataframe()
        df_012 = series.pd_dataframe()

        df_0[["0"]] = df_0[["0"]].applymap(fn)
        df_2[["2"]] = df_2[["2"]].applymap(fn)
        df_01[["0", "1"]] = df_01[["0", "1"]].applymap(fn)
        df_012 = df_012.applymap(fn)

        series_0 = TimeSeries.from_dataframe(df_0, freq="D")
        series_2 = TimeSeries.from_dataframe(df_2, freq="D")
        series_01 = TimeSeries.from_dataframe(df_01, freq="D")
        series_012 = TimeSeries.from_dataframe(df_012, freq="D")

        self.assertEqual(series_0["0"], series["0"].map(fn))
        self.assertEqual(series_2["2"], series["2"].map(fn))
        self.assertEqual(series_01[["0", "1"]], series[["0", "1"]].map(fn))
        self.assertEqual(series_012, series[["0", "1", "2"]].map(fn))
        self.assertEqual(series_012, series.map(fn))

        self.assertNotEqual(series_01, series[["0", "1"]].map(fn))

    def test_map_with_timestamp(self):
        series = linear_timeseries(
            start_value=1,
            length=12,
            freq="MS",
            start=pd.Timestamp("2000-01-01"),
            end_value=12,
        )  # noqa: E501
        zeroes = constant_timeseries(
            value=0.0, length=12, freq="MS", start=pd.Timestamp("2000-01-01")
        )
        zeroes = zeroes.with_columns_renamed("constant", "linear")

        def function(ts, x):
            return x - ts.month

        new_series = series.map(function)
        self.assertEqual(new_series, zeroes)

    def test_map_wrong_fn(self):
        series = linear_timeseries(
            start_value=1,
            length=12,
            freq="MS",
            start=pd.Timestamp("2000-01-01"),
            end_value=12,
        )  # noqa: E501

        def add(x, y, z):
            return x + y + z

        with self.assertRaises(ValueError):
            series.map(add)

        ufunc_add = np.frompyfunc(add, 3, 1)

        with self.assertRaises(ValueError):
            series.map(ufunc_add)

    def test_gaps(self):
        times1 = pd.date_range("20130101", "20130110")
        times2 = pd.date_range("20120101", "20210301", freq="Q")
        times3 = pd.date_range("20120101", "20210301", freq="AS")
        times4 = pd.date_range("20120101", "20210301", freq="2MS")

        pd_series1 = pd.Series(
            [1, 1] + 3 * [np.nan] + [1, 1, 1] + [np.nan] * 2, index=times1
        )
        pd_series2 = pd.Series(
            [1, 1] + 3 * [np.nan] + [1, 1] + [np.nan] * 3, index=times1
        )
        pd_series3 = pd.Series([np.nan] * 10, index=times1)
        pd_series4 = pd.Series(
            [1] * 5 + 3 * [np.nan] + [1] * 18 + 7 * [np.nan] + [1, 1] + [np.nan],
            index=times2,
        )
        pd_series5 = pd.Series(
            [1] * 3 + 2 * [np.nan] + [1] + 2 * [np.nan] + [1, 1], index=times3
        )
        pd_series6 = pd.Series(
            [1] * 10 + 1 * [np.nan] + [1] * 13 + 5 * [np.nan] + [1] * 18 + 9 * [np.nan],
            index=times4,
        )

        series1 = TimeSeries.from_series(pd_series1)
        series2 = TimeSeries.from_series(pd_series2)
        series3 = TimeSeries.from_series(pd_series3)
        series4 = TimeSeries.from_series(pd_series4)
        series5 = TimeSeries.from_series(pd_series5)
        series6 = TimeSeries.from_series(pd_series6)

        gaps1 = series1.gaps()
        self.assertTrue(
            (
                gaps1["gap_start"]
                == pd.DatetimeIndex(
                    [pd.Timestamp("20130103"), pd.Timestamp("20130109")]
                )
            ).all()
        )
        self.assertTrue(
            (
                gaps1["gap_end"]
                == pd.DatetimeIndex(
                    [pd.Timestamp("20130105"), pd.Timestamp("20130110")]
                )
            ).all()
        )
        self.assertEqual(gaps1["gap_size"].values.tolist(), [3, 2])
        gaps2 = series2.gaps()
        self.assertEqual(gaps2["gap_size"].values.tolist(), [3, 3])
        gaps3 = series3.gaps()
        self.assertEqual(gaps3["gap_size"].values.tolist(), [10])
        gaps4 = series4.gaps()
        self.assertEqual(gaps4["gap_size"].values.tolist(), [3, 7, 1])
        gaps5 = series5.gaps()
        self.assertEqual(gaps5["gap_size"].values.tolist(), [2, 2])
        self.assertTrue(
            (
                gaps5["gap_start"]
                == pd.DatetimeIndex(
                    [pd.Timestamp("20150101"), pd.Timestamp("20180101")]
                )
            ).all()
        )
        self.assertTrue(
            (
                gaps5["gap_end"]
                == pd.DatetimeIndex(
                    [pd.Timestamp("20160101"), pd.Timestamp("20190101")]
                )
            ).all()
        )
        gaps6 = series6.gaps()
        self.assertEqual(gaps6["gap_size"].values.tolist(), [1, 5, 9])
        self.assertTrue(
            (
                gaps6["gap_start"]
                == pd.DatetimeIndex(
                    [
                        pd.Timestamp("20130901"),
                        pd.Timestamp("20160101"),
                        pd.Timestamp("20191101"),
                    ]
                )
            ).all()
        )
        self.assertTrue(
            (
                gaps6["gap_end"]
                == pd.DatetimeIndex(
                    [
                        pd.Timestamp("20130901"),
                        pd.Timestamp("20160901"),
                        pd.Timestamp("20210301"),
                    ]
                )
            ).all()
        )

    def test_longest_contiguous_slice(self):
        times = pd.date_range("20130101", "20130111")
        pd_series1 = pd.Series(
            [1, 1] + 3 * [np.nan] + [1, 1, 1] + [np.nan] * 2 + [1], index=times
        )
        series1 = TimeSeries.from_series(pd_series1)

        self.assertEqual(len(series1.longest_contiguous_slice()), 3)
        self.assertEqual(len(series1.longest_contiguous_slice(2)), 6)

    def test_with_columns_renamed(self):
        series1 = linear_timeseries(
            start_value=1,
            length=12,
            freq="MS",
            start=pd.Timestamp("2000-01-01"),
            end_value=12,
        ).stack(
            linear_timeseries(
                start_value=1,
                length=12,
                freq="MS",
                start=pd.Timestamp("2000-01-01"),
                end_value=12,
            )
        )

        series1 = series1.with_columns_renamed(
            ["linear", "linear_1"], ["linear1", "linear2"]
        )
        self.assertEqual(["linear1", "linear2"], series1.columns.to_list())

        with self.assertRaises(ValueError):
            series1.with_columns_renamed(
                ["linear1", "linear2"], ["linear1", "linear3", "linear4"]
            )

        #  Linear7 doesn't exist
        with self.assertRaises(ValueError):
            series1.with_columns_renamed("linear7", "linear5")

    def test_to_csv_probabilistic_ts(self):
        samples = [
            linear_timeseries(start_value=val, length=10) for val in [10, 20, 30]
        ]
        ts = concatenate(samples, axis=2)
        with self.assertRaises(AssertionError):
            ts.to_csv("blah.csv")

    @patch("darts.timeseries.TimeSeries.pd_dataframe")
    def test_to_csv_deterministic(self, pddf_mock):
        ts = TimeSeries(
            xr.DataArray(
                np.random.rand(10, 10, 1),
                [
                    ("time", pd.date_range("2000-01-01", periods=10)),
                    ("component", ["comp_" + str(i) for i in range(10)]),
                    ("sample", [0]),
                ],
            )
        )

        ts.to_csv("test.csv")
        pddf_mock.assert_called_once()

    @patch("darts.timeseries.TimeSeries.pd_dataframe")
    def test_to_csv_stochastic(self, pddf_mock):
        ts = TimeSeries(
            xr.DataArray(
                np.random.rand(10, 10, 10),
                [
                    ("time", pd.date_range("2000-01-01", periods=10)),
                    ("component", ["comp_" + str(i) for i in range(10)]),
                    ("sample", range(10)),
                ],
            )
        )

        with self.assertRaises(AssertionError):
            ts.to_csv("test.csv")


class TimeSeriesConcatenateTestCase(DartsBaseTestClass):

    #
    # COMPONENT AXIS TESTS
    #

    def test_concatenate_component_sunny_day(self):
        samples = [
            linear_timeseries(
                start_value=10, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=20, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=30, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
        ]

        ts = concatenate(samples, axis="component")
        self.assertEqual((10, 3, 1), ts._xa.shape)

    def test_concatenate_component_different_time_axes_no_force(self):
        samples = [
            linear_timeseries(
                start_value=10, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=20, length=10, start=pd.Timestamp("2000-01-11"), freq="D"
            ),
            linear_timeseries(
                start_value=30, length=10, start=pd.Timestamp("2000-02-11"), freq="D"
            ),
        ]

        with self.assertRaises(ValueError):
            concatenate(samples, axis="component")

    def test_concatenate_component_different_time_axes_with_force(self):
        samples = [
            linear_timeseries(
                start_value=10, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=20, length=10, start=pd.Timestamp("2000-01-11"), freq="D"
            ),
            linear_timeseries(
                start_value=30, length=10, start=pd.Timestamp("2000-02-11"), freq="D"
            ),
        ]

        ts = concatenate(samples, axis="component", ignore_time_axis=True)
        self.assertEqual((10, 3, 1), ts._xa.shape)
        self.assertEqual(pd.Timestamp("2000-01-01"), ts.start_time())
        self.assertEqual(pd.Timestamp("2000-01-10"), ts.end_time())

    def test_concatenate_component_different_time_axes_with_force_uneven_series(self):
        samples = [
            linear_timeseries(
                start_value=10, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=20, length=10, start=pd.Timestamp("2000-01-11"), freq="D"
            ),
            linear_timeseries(
                start_value=30, length=20, start=pd.Timestamp("2000-02-11"), freq="D"
            ),
        ]

        with self.assertRaises(ValueError):
            concatenate(samples, axis="component", ignore_time_axis=True)

    #
    # SAMPLE AXIS TESTS
    #

    def test_concatenate_sample_sunny_day(self):
        samples = [
            linear_timeseries(
                start_value=10, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=20, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=30, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
        ]

        ts = concatenate(samples, axis="sample")
        self.assertEqual((10, 1, 3), ts._xa.shape)

    #
    # TIME AXIS TESTS
    #

    def test_concatenate_time_sunny_day(self):
        samples = [
            linear_timeseries(
                start_value=10, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=20, length=10, start=pd.Timestamp("2000-01-11"), freq="D"
            ),
            linear_timeseries(
                start_value=30, length=10, start=pd.Timestamp("2000-01-21"), freq="D"
            ),
        ]

        ts = concatenate(samples, axis="time")
        self.assertEqual((30, 1, 1), ts._xa.shape)
        self.assertEqual(pd.Timestamp("2000-01-01"), ts.start_time())
        self.assertEqual(pd.Timestamp("2000-01-30"), ts.end_time())

    def test_concatenate_time_same_time_no_force(self):
        samples = [
            linear_timeseries(
                start_value=10, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=20, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=30, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
        ]

        with self.assertRaises(ValueError):
            concatenate(samples, axis="time")

    def test_concatenate_time_same_time_force(self):
        samples = [
            linear_timeseries(
                start_value=10, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=20, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=30, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
        ]

        ts = concatenate(samples, axis="time", ignore_time_axis=True)
        self.assertEqual((30, 1, 1), ts._xa.shape)
        self.assertEqual(pd.Timestamp("2000-01-01"), ts.start_time())
        self.assertEqual(pd.Timestamp("2000-01-30"), ts.end_time())

    def test_concatenate_time_different_time_axes_no_force(self):
        samples = [
            linear_timeseries(
                start_value=10, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=20, length=10, start=pd.Timestamp("2000-01-12"), freq="D"
            ),
            linear_timeseries(
                start_value=30, length=10, start=pd.Timestamp("2000-01-18"), freq="D"
            ),
        ]

        with self.assertRaises(ValueError):
            concatenate(samples, axis="time")

    def test_concatenate_time_different_time_axes_force(self):
        samples = [
            linear_timeseries(
                start_value=10, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
            ),
            linear_timeseries(
                start_value=20, length=10, start=pd.Timestamp("2000-01-13"), freq="D"
            ),
            linear_timeseries(
                start_value=30, length=10, start=pd.Timestamp("2000-01-19"), freq="D"
            ),
        ]

        ts = concatenate(samples, axis="time", ignore_time_axis=True)
        self.assertEqual((30, 1, 1), ts._xa.shape)
        self.assertEqual(pd.Timestamp("2000-01-01"), ts.start_time())
        self.assertEqual(pd.Timestamp("2000-01-30"), ts.end_time())

    def test_concatenate_time_different_time_axes_no_force_2_day_freq(self):
        samples = [
            linear_timeseries(
                start_value=10, length=10, start=pd.Timestamp("2000-01-01"), freq="2D"
            ),
            linear_timeseries(
                start_value=20, length=10, start=pd.Timestamp("2000-01-21"), freq="2D"
            ),
            linear_timeseries(
                start_value=30, length=10, start=pd.Timestamp("2000-02-10"), freq="2D"
            ),
        ]

        ts = concatenate(samples, axis="time")
        self.assertEqual((30, 1, 1), ts._xa.shape)
        self.assertEqual(pd.Timestamp("2000-01-01"), ts.start_time())
        self.assertEqual(pd.Timestamp("2000-02-28"), ts.end_time())
        self.assertEqual("2D", ts.freq)

    def test_concatenate_timeseries_method(self):
        ts1 = linear_timeseries(
            start_value=10, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
        )
        ts2 = linear_timeseries(
            start_value=20, length=10, start=pd.Timestamp("2000-01-11"), freq="D"
        )

        result_ts = ts1.concatenate(ts2, axis="time")
        self.assertEqual((20, 1, 1), result_ts._xa.shape)
        self.assertEqual(pd.Timestamp("2000-01-01"), result_ts.start_time())
        self.assertEqual(pd.Timestamp("2000-01-20"), result_ts.end_time())
        self.assertEqual("D", result_ts.freq)


class TimeSeriesHeadTailTestCase(DartsBaseTestClass):

    ts = TimeSeries(
        xr.DataArray(
            np.random.rand(10, 10, 10),
            [
                ("time", pd.date_range("2000-01-01", periods=10)),
                ("component", ["comp_" + str(i) for i in range(10)]),
                ("sample", range(10)),
            ],
        )
    )

    def test_head_sunny_day_time_axis(self):
        result = self.ts.head()
        self.assertEqual(5, result.n_timesteps)
        self.assertEqual(pd.Timestamp("2000-01-05"), result.end_time())

    def test_head_sunny_day_component_axis(self):
        result = self.ts.head(axis=1)
        self.assertEqual(5, result.n_components)
        self.assertEqual(
            ["comp_0", "comp_1", "comp_2", "comp_3", "comp_4"],
            result._xa.coords["component"].values.tolist(),
        )

    def test_tail_sunny_day_time_axis(self):
        result = self.ts.tail()
        self.assertEqual(5, result.n_timesteps)
        self.assertEqual(pd.Timestamp("2000-01-06"), result.start_time())

    def test_tail_sunny_day_component_axis(self):
        result = self.ts.tail(axis=1)
        self.assertEqual(5, result.n_components)
        self.assertEqual(
            ["comp_5", "comp_6", "comp_7", "comp_8", "comp_9"],
            result._xa.coords["component"].values.tolist(),
        )

    def test_head_sunny_day_sample_axis(self):
        result = self.ts.tail(axis=2)
        self.assertEqual(5, result.n_samples)
        self.assertEqual(
            list(range(5, 10)), result._xa.coords["sample"].values.tolist()
        )

    def test_head_overshot_time_axis(self):
        result = self.ts.head(20)
        self.assertEqual(10, result.n_timesteps)
        self.assertEqual(pd.Timestamp("2000-01-10"), result.end_time())

    def test_head_overshot_component_axis(self):
        result = self.ts.head(20, axis="component")
        self.assertEqual(10, result.n_components)

    def test_head_overshot_sample_axis(self):
        result = self.ts.head(20, axis="sample")
        self.assertEqual(10, result.n_samples)

    def test_tail_overshot_time_axis(self):
        result = self.ts.tail(20)
        self.assertEqual(10, result.n_timesteps)
        self.assertEqual(pd.Timestamp("2000-01-01"), result.start_time())

    def test_tail_overshot_component_axis(self):
        result = self.ts.tail(20, axis="component")
        self.assertEqual(10, result.n_components)

    def test_tail_overshot_sample_axis(self):
        result = self.ts.tail(20, axis="sample")
        self.assertEqual(10, result.n_samples)


class TimeSeriesFromDataFrameTestCase(DartsBaseTestClass):
    def test_from_dataframe_sunny_day(self):
        data_dict = {"Time": pd.date_range(start="20180501", end="20200301", freq="MS")}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        data_dict["Values2"] = np.random.uniform(
            low=0, high=1, size=len(data_dict["Time"])
        )

        data_pd1 = pd.DataFrame(data_dict)
        data_pd2 = data_pd1.copy()
        data_pd2["Time"] = data_pd2["Time"].apply(lambda date: str(date))
        data_pd3 = data_pd1.set_index("Time")

        data_darts1 = TimeSeries.from_dataframe(df=data_pd1, time_col="Time")
        data_darts2 = TimeSeries.from_dataframe(df=data_pd2, time_col="Time")
        data_darts3 = TimeSeries.from_dataframe(df=data_pd3)

        self.assertEqual(data_darts1, data_darts2)
        self.assertEqual(data_darts1, data_darts3)

    def test_time_col_convert_string_integers(self):
        expected = np.array(list(range(3, 10)))
        data_dict = {"Time": expected.astype(str)}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        ts = TimeSeries.from_dataframe(df=df, time_col="Time")

        self.assertEqual(set(ts.time_index.values.tolist()), set(expected))
        self.assertEqual(ts.time_index.dtype, int)
        self.assertEqual(ts.time_index.name, "Time")

    def test_time_col_convert_integers(self):
        expected = np.array(list(range(10)))
        data_dict = {"Time": expected}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        ts = TimeSeries.from_dataframe(df=df, time_col="Time")

        self.assertEqual(set(ts.time_index.values.tolist()), set(expected))
        self.assertEqual(ts.time_index.dtype, int)
        self.assertEqual(ts.time_index.name, "Time")

    def test_fail_with_bad_integer_time_col(self):
        bad_time_col_vals = np.array([4, 0, 1, 2])
        data_dict = {"Time": bad_time_col_vals}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        with self.assertRaises(ValueError):
            TimeSeries.from_dataframe(df=df, time_col="Time")

    def test_time_col_convert_rangeindex(self):
        expected_l = [4, 0, 2, 3, 1]
        expected = np.array(expected_l)
        data_dict = {"Time": expected}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        ts = TimeSeries.from_dataframe(df=df, time_col="Time")

        # check type (should convert to RangeIndex):
        self.assertEqual(type(ts.time_index), pd.RangeIndex)

        # check values inside the index (should be sorted correctly):
        self.assertEqual(list(ts.time_index), sorted(expected))

        # check that values are sorted accordingly:
        ar1 = ts.values(copy=False)[:, 0]
        ar2 = data_dict["Values1"][
            list(expected_l.index(i) for i in range(len(expected)))
        ]
        self.assertTrue(np.all(ar1 == ar2))

    def test_time_col_convert_datetime(self):
        expected = pd.date_range(start="20180501", end="20200301", freq="MS")
        data_dict = {"Time": expected}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        ts = TimeSeries.from_dataframe(df=df, time_col="Time")

        self.assertEqual(ts.time_index.dtype, "datetime64[ns]")
        self.assertEqual(ts.time_index.name, "Time")

    def test_time_col_convert_datetime_strings(self):
        expected = pd.date_range(start="20180501", end="20200301", freq="MS")
        data_dict = {"Time": expected.values.astype(str)}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        ts = TimeSeries.from_dataframe(df=df, time_col="Time")

        self.assertEqual(ts.time_index.dtype, "datetime64[ns]")
        self.assertEqual(ts.time_index.name, "Time")

    def test_time_col_convert_garbage(self):
        expected = [
            "2312312asdfdw",
            "asdfsdf432sdf",
            "sfsdfsvf3435",
            "cdsfs45234",
            "vsdgert43534f",
        ]
        data_dict = {"Time": expected}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)

        with self.assertRaises(AttributeError):
            TimeSeries.from_dataframe(df=df, time_col="Time")


class SimpleStatisticsTestCase(DartsBaseTestClass):

    times = pd.date_range("20130101", "20130110", freq="D")
    values = np.random.rand(10, 2, 100)
    ar = xr.DataArray(
        values,
        dims=("time", "component", "sample"),
        coords={"time": times, "component": ["a", "b"]},
    )
    ts = TimeSeries(ar)

    def test_mean(self):
        for axis in range(3):
            new_ts = self.ts.mean(axis=axis)
            # check values
            self.assertTrue(
                np.isclose(
                    new_ts._xa.values, self.values.mean(axis=axis, keepdims=True)
                ).all()
            )

    def test_var(self):
        for ddof in range(5):
            new_ts = self.ts.var(ddof=ddof)
            # check values
            self.assertTrue(
                np.isclose(new_ts.values(), self.values.var(ddof=ddof, axis=2)).all()
            )

    def test_std(self):
        for ddof in range(5):
            new_ts = self.ts.std(ddof=ddof)
            # check values
            self.assertTrue(
                np.isclose(new_ts.values(), self.values.std(ddof=ddof, axis=2)).all()
            )

    def test_skew(self):
        new_ts = self.ts.skew()
        # check values
        self.assertTrue(np.isclose(new_ts.values(), skew(self.values, axis=2)).all())

    def test_kurtosis(self):
        new_ts = self.ts.kurtosis()
        # check values
        self.assertTrue(
            np.isclose(
                new_ts.values(),
                kurtosis(self.values, axis=2),
            ).all()
        )

    def test_min(self):
        for axis in range(3):
            new_ts = self.ts.min(axis=axis)
            # check values
            self.assertTrue(
                np.isclose(
                    new_ts._xa.values, self.values.min(axis=axis, keepdims=True)
                ).all()
            )

    def test_max(self):
        for axis in range(3):
            new_ts = self.ts.max(axis=axis)
            # check values
            self.assertTrue(
                np.isclose(
                    new_ts._xa.values, self.values.max(axis=axis, keepdims=True)
                ).all()
            )

    def test_sum(self):
        for axis in range(3):
            new_ts = self.ts.sum(axis=axis)
            # check values
            self.assertTrue(
                np.isclose(
                    new_ts._xa.values, self.values.sum(axis=axis, keepdims=True)
                ).all()
            )

    def test_median(self):
        for axis in range(3):
            new_ts = self.ts.median(axis=axis)
            # check values
            self.assertTrue(
                np.isclose(
                    new_ts._xa.values, np.median(self.values, axis=axis, keepdims=True)
                ).all()
            )

    def test_quantile(self):
        for q in [0.01, 0.1, 0.5, 0.95]:
            new_ts = self.ts.quantile(quantile=q)
            # check values
            self.assertTrue(
                np.isclose(
                    new_ts.values(),
                    np.quantile(self.values, q=q, axis=2),
                ).all()
            )
