import math
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import kurtosis, skew

from darts import TimeSeries, concatenate
from darts.utils.timeseries_generation import (
    constant_timeseries,
    generate_index,
    linear_timeseries,
)


class TestTimeSeries:

    times = pd.date_range("20130101", "20130110", freq="D")
    pd_series1 = pd.Series(range(10), index=times)
    pd_series2 = pd.Series(range(5, 15), index=times)
    pd_series3 = pd.Series(range(15, 25), index=times)
    series1: TimeSeries = TimeSeries.from_series(pd_series1)
    series2: TimeSeries = TimeSeries.from_series(pd_series2)
    series3: TimeSeries = TimeSeries.from_series(pd_series2)

    def test_creation(self):
        series_test = TimeSeries.from_series(self.pd_series1)
        assert series_test.pd_series().equals(self.pd_series1.astype(float))

        # Creation with a well formed array:
        ar = xr.DataArray(
            np.random.randn(10, 2, 3),
            dims=("time", "component", "sample"),
            coords={"time": self.times, "component": ["a", "b"]},
            name="time series",
        )
        ts = TimeSeries(ar)
        assert ts.is_stochastic

        ar = xr.DataArray(
            np.random.randn(10, 2, 1),
            dims=("time", "component", "sample"),
            coords={"time": pd.RangeIndex(0, 10, 1), "component": ["a", "b"]},
            name="time series",
        )
        ts = TimeSeries(ar)
        assert ts.is_deterministic

        # creation with ill-formed arrays
        with pytest.raises(ValueError):
            ar2 = xr.DataArray(
                np.random.randn(10, 2, 1),
                dims=("time", "wrong", "sample"),
                coords={"time": self.times, "wrong": ["a", "b"]},
                name="time series",
            )
            _ = TimeSeries(ar2)

        with pytest.raises(ValueError):
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

    def test_integer_range_indexing(self):
        # sanity checks for the integer-indexed series
        range_indexed_data = np.random.randn(
            50,
        )
        series_int: TimeSeries = TimeSeries.from_values(range_indexed_data)

        assert series_int[0].values().item() == range_indexed_data[0]
        assert series_int[10].values().item() == range_indexed_data[10]

        assert np.all(
            series_int[10:20].univariate_values() == range_indexed_data[10:20]
        )
        assert np.all(series_int[10:].univariate_values() == range_indexed_data[10:])

        assert np.all(
            series_int[pd.RangeIndex(start=10, stop=40, step=1)].univariate_values()
            == range_indexed_data[10:40]
        )

        # check the RangeIndex when indexing with a list
        indexed_ts = series_int[[2, 3, 4, 5, 6]]
        assert isinstance(indexed_ts.time_index, pd.RangeIndex)
        assert list(indexed_ts.time_index) == list(pd.RangeIndex(2, 7, step=1))

        # check integer indexing features when series index does not start at 0
        values = np.random.random(100)
        times = pd.RangeIndex(10, 110)
        series: TimeSeries = TimeSeries.from_times_and_values(times, values)

        # getting index for idx should return i s.t., series[i].time == idx
        assert series.get_index_at_point(101) == 91

        # slicing outside of the index range should return an empty ts
        assert len(series[120:125]) == 0
        assert series[120:125] == series.slice(120, 125)

        # slicing with a partial index overlap should return the ts subset
        assert len(series[95:105]) == 5
        # adding the 10 values index shift to compare the same values
        assert series[95:105] == series.slice(105, 115)

        # check integer indexing features when series index starts at 0 with a step > 1
        values = np.random.random(100)
        times = pd.RangeIndex(0, 200, step=2)
        series: TimeSeries = TimeSeries.from_times_and_values(times, values)

        # getting index for idx should return i s.t., series[i].time == idx
        assert series.get_index_at_point(100) == 50

        # getting index outside of the index range should raise an exception
        with pytest.raises(IndexError):
            series[100]

        # slicing should act the same irrespective of the initial time stamp
        np.testing.assert_equal(series[10:20].values().flatten(), values[10:20])

        # slicing outside of the range should return an empty ts
        assert len(series[105:110]) == 0
        # multiply the slice start and end values by 2 to compare the same values
        assert series[105:110] == series.slice(210, 220)

        # slicing with an index overlap should return the ts subset
        assert len(series[95:105]) == 5
        # multiply the slice start and end values by 2 to compare the same values
        assert series[95:105] == series.slice(190, 210)

        # drop_after should act on the timestamp
        np.testing.assert_equal(series.drop_after(20).values().flatten(), values[:10])

        # test get_index_at_point on series which does not start at 0 and with a step > 1
        values = np.random.random(10)
        times = pd.RangeIndex(10, 30, step=2)
        series: TimeSeries = TimeSeries.from_times_and_values(times, values)

        # getting index for idx should return i s.t., series[i].time == idx
        assert series.get_index_at_point(16) == 3

    def test_integer_indexing(self):
        n = 10
        int_idx = pd.Index([i for i in range(n)])
        assert not isinstance(int_idx, pd.RangeIndex)

        # test that integer index gets converted to correct RangeIndex
        vals = np.random.randn(n)
        ts_from_int_idx = TimeSeries.from_times_and_values(times=int_idx, values=vals)
        ts_from_range_idx = TimeSeries.from_values(values=vals)
        assert (
            isinstance(ts_from_int_idx.time_index, pd.RangeIndex)
            and ts_from_int_idx.freq == 1
        )
        assert ts_from_int_idx.time_index.equals(ts_from_range_idx.time_index)

        for step in [2, 3]:
            # test integer index with different step sizes, beginning at non-zero
            int_idx = pd.Index([i for i in range(2, 2 + n * step, step)])
            ts_from_int_idx = TimeSeries.from_times_and_values(
                times=int_idx, values=vals
            )
            assert isinstance(ts_from_int_idx.time_index, pd.RangeIndex)
            assert ts_from_int_idx.time_index[0] == 2
            assert ts_from_int_idx.time_index[-1] == 2 + (n - 1) * step
            assert ts_from_int_idx.freq == step

            # test integer index with unsorted indices
            idx_permuted = [n - 1] + [i for i in range(1, n - 1, 1)] + [0]
            ts_from_int_idx2 = TimeSeries.from_times_and_values(
                times=int_idx[idx_permuted], values=vals[idx_permuted]
            )
            assert ts_from_int_idx == ts_from_int_idx2

            # check other TimeSeries creation methods
            ts_from_df_time_col = TimeSeries.from_dataframe(
                pd.DataFrame({"0": vals, "time": int_idx}), time_col="time"
            )
            ts_from_df = TimeSeries.from_dataframe(pd.DataFrame(vals, index=int_idx))
            ts_from_series = TimeSeries.from_series(pd.Series(vals, index=int_idx))
            assert ts_from_df_time_col == ts_from_int_idx
            assert ts_from_df == ts_from_int_idx
            assert ts_from_series == ts_from_int_idx

        # test invalid integer index; non-constant step size
        int_idx = pd.Index([0, 2, 4, 5])
        with pytest.raises(ValueError):
            _ = TimeSeries.from_times_and_values(
                times=int_idx, values=np.random.randn(4)
            )

    def test_datetime_indexing(self):
        # checking that the DatetimeIndex slicing is behaving as described in
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html

        # getting index outside of the index range should raise an exception
        with pytest.raises(KeyError):
            self.series1[pd.Timestamp("20130111")]

        # slicing outside of the range should return an empty ts
        assert (
            len(self.series1[pd.Timestamp("20130111") : pd.Timestamp("20130115")]) == 0
        )
        assert self.series1[
            pd.Timestamp("20130111") : pd.Timestamp("20130115")
        ] == self.series1.slice(pd.Timestamp("20130111"), pd.Timestamp("20130115"))

        # slicing with an partial index overlap should return the ts subset (start and end included)
        assert (
            len(self.series1[pd.Timestamp("20130105") : pd.Timestamp("20130112")]) == 6
        )
        assert self.series1[
            pd.Timestamp("20130105") : pd.Timestamp("20130112")
        ] == self.series1.slice(pd.Timestamp("20130105"), pd.Timestamp("20130112"))

    def test_univariate_component(self):
        series = TimeSeries.from_values(np.array([10, 20, 30])).with_columns_renamed(
            "0", "component"
        )
        mseries = concatenate([series] * 3, axis="component")
        mseries = mseries.with_hierarchy(
            {"component_1": ["component"], "component_2": ["component"]}
        )

        static_cov = pd.DataFrame(
            {"dim0": [1, 2, 3], "dim1": [-2, -1, 0], "dim2": [0.0, 0.1, 0.2]}
        )

        mseries = mseries.with_static_covariates(static_cov)

        for univ_series in [
            mseries.univariate_component(1),
            mseries.univariate_component("component_1"),
        ]:
            # hierarchy should be dropped
            assert univ_series.hierarchy is None

            # only the right static covariate column should be retained
            assert univ_series.static_covariates.sum().sum() == 1.1

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
            assert ts.columns.tolist() == cs_after

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
            assert (abs(q_ts.values() - np.quantile(values, q=q, axis=2)) < 1e-3).all()

    def test_quantiles_df(self):
        q = (0.01, 0.1, 0.5, 0.95)
        values = np.random.rand(10, 1, 1000)
        ar = xr.DataArray(
            values,
            dims=("time", "component", "sample"),
            coords={"time": self.times, "component": ["a"]},
        )
        ts = TimeSeries(ar)
        q_ts = ts.quantiles_df(q)
        for col in q_ts:
            q = float(str(col).replace("a_", ""))
            assert abs(
                q_ts[col].to_numpy().reshape(10, 1) - np.quantile(values, q=q, axis=2)
                < 1e-3
            ).all()

    def test_alt_creation(self):
        with pytest.raises(ValueError):
            # Series cannot be lower than three without passing frequency as argument to constructor,
            # if fill_missing_dates is True (otherwise it works)
            index = pd.date_range("20130101", "20130102")
            TimeSeries.from_times_and_values(
                index, self.pd_series1.values[:2], fill_missing_dates=True
            )
        with pytest.raises(ValueError):
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

        assert series_test.start_time() == pd.to_datetime("20130101")
        assert series_test.end_time() == pd.to_datetime("20130110")
        assert all(series_test.pd_series().values == self.pd_series1.values)
        assert series_test.freq == self.series1.freq

    # TODO test over to_dataframe when multiple features choice is decided

    def test_eq(self):
        seriesA: TimeSeries = TimeSeries.from_series(self.pd_series1)
        assert self.series1 == seriesA
        assert not (self.series1 != seriesA)

        # with different dates
        seriesC = TimeSeries.from_series(
            pd.Series(range(10), index=pd.date_range("20130102", "20130111"))
        )
        assert not (self.series1 == seriesC)

    def test_dates(self):
        assert self.series1.start_time() == pd.Timestamp("20130101")
        assert self.series1.end_time() == pd.Timestamp("20130110")
        assert self.series1.duration == pd.Timedelta(days=9)

    @staticmethod
    def helper_test_slice(test_case, test_series: TimeSeries):
        # base case
        seriesA = test_series.slice(pd.Timestamp("20130104"), pd.Timestamp("20130107"))
        assert seriesA.start_time() == pd.Timestamp("20130104")
        assert seriesA.end_time() == pd.Timestamp("20130107")

        # time stamp not in series
        seriesB = test_series.slice(
            pd.Timestamp("20130104 12:00:00"), pd.Timestamp("20130107")
        )
        assert seriesB.start_time() == pd.Timestamp("20130105")
        assert seriesB.end_time() == pd.Timestamp("20130107")

        # end timestamp after series
        seriesC = test_series.slice(pd.Timestamp("20130108"), pd.Timestamp("20130201"))
        assert seriesC.start_time() == pd.Timestamp("20130108")
        assert seriesC.end_time() == pd.Timestamp("20130110")

        # integer-indexed series, starting at 0
        values = np.random.rand(30)
        idx = pd.RangeIndex(start=0, stop=30, step=1)
        ts = TimeSeries.from_times_and_values(idx, values)
        slice_vals = ts.slice(10, 20).values(copy=False).flatten()
        np.testing.assert_equal(slice_vals, values[10:20])

        # integer-indexed series, not starting at 0
        values = np.random.rand(30)
        idx = pd.RangeIndex(start=5, stop=35, step=1)
        ts = TimeSeries.from_times_and_values(idx, values)
        slice_vals = ts.slice(10, 20).values(copy=False).flatten()
        np.testing.assert_equal(slice_vals, values[5:15])

        # integer-indexed series, starting at 0, with step > 1
        values = np.random.rand(30)
        idx = pd.RangeIndex(start=0, stop=60, step=2)
        ts = TimeSeries.from_times_and_values(idx, values)
        slice_vals = ts.slice(10, 20).values(copy=False).flatten()
        np.testing.assert_equal(slice_vals, values[5:10])

        # integer-indexed series, not starting at 0, with step > 1
        values = np.random.rand(30)
        idx = pd.RangeIndex(start=5, stop=65, step=2)
        ts = TimeSeries.from_times_and_values(idx, values)
        slice_vals = ts.slice(11, 21).values(copy=False).flatten()
        np.testing.assert_equal(slice_vals, values[3:8])

        # test cases where start and/or stop are not in the series

        # n points, base case
        seriesD = test_series.slice_n_points_after(pd.Timestamp("20130102"), n=3)
        assert seriesD.start_time() == pd.Timestamp("20130102")
        assert len(seriesD.values()) == 3
        assert seriesD.end_time() == pd.Timestamp("20130104")

        seriesE = test_series.slice_n_points_after(
            pd.Timestamp("20130107 12:00:10"), n=10
        )
        assert seriesE.start_time() == pd.Timestamp("20130108")
        assert seriesE.end_time() == pd.Timestamp("20130110")

        seriesF = test_series.slice_n_points_before(pd.Timestamp("20130105"), n=3)
        assert seriesF.end_time() == pd.Timestamp("20130105")
        assert len(seriesF.values()) == 3
        assert seriesF.start_time() == pd.Timestamp("20130103")

        seriesG = test_series.slice_n_points_before(
            pd.Timestamp("20130107 12:00:10"), n=10
        )
        assert seriesG.start_time() == pd.Timestamp("20130101")
        assert seriesG.end_time() == pd.Timestamp("20130107")

        # test slice_n_points_after and slice_n_points_before with integer-indexed series
        s = TimeSeries.from_times_and_values(pd.RangeIndex(6, 10), np.arange(16, 20))
        sliced_idx = s.slice_n_points_after(7, 2).time_index
        assert all(sliced_idx == pd.RangeIndex(7, 9))

        sliced_idx = s.slice_n_points_before(8, 2).time_index
        assert all(sliced_idx == pd.RangeIndex(7, 9))

        # integer indexed series, step = 1, timestamps not in series
        values = np.random.rand(30)
        idx = pd.RangeIndex(start=0, stop=30, step=1)
        ts = TimeSeries.from_times_and_values(idx, values)
        # end timestamp further off, slice should be inclusive of last timestamp:
        slice_vals = ts.slice(10, 30).values(copy=False).flatten()
        np.testing.assert_equal(slice_vals, values[10:])
        slice_vals = ts.slice(10, 32).values(copy=False).flatten()
        np.testing.assert_equal(slice_vals, values[10:])

        # end timestamp within the series make it exclusive:
        slice_vals = ts.slice(10, 29).values(copy=False).flatten()
        np.testing.assert_equal(slice_vals, values[10:29])

        # integer indexed series, step > 1, timestamps not in series
        idx = pd.RangeIndex(start=0, stop=60, step=2)
        ts = TimeSeries.from_times_and_values(idx, values)
        slice_vals = ts.slice(11, 31).values(copy=False).flatten()
        np.testing.assert_equal(slice_vals, values[6:15])

        slice_ts = ts.slice(40, 60)
        assert ts.end_time() == slice_ts.end_time()

    @staticmethod
    def helper_test_split(test_case, test_series: TimeSeries):
        seriesA, seriesB = test_series.split_after(pd.Timestamp("20130104"))
        assert seriesA.end_time() == pd.Timestamp("20130104")
        assert seriesB.start_time() == pd.Timestamp("20130105")

        seriesC, seriesD = test_series.split_before(pd.Timestamp("20130104"))
        assert seriesC.end_time() == pd.Timestamp("20130103")
        assert seriesD.start_time() == pd.Timestamp("20130104")

        seriesE, seriesF = test_series.split_after(0.7)
        assert len(seriesE) == round(0.7 * len(test_series))
        assert len(seriesF) == round(0.3 * len(test_series))

        seriesG, seriesH = test_series.split_before(0.7)
        assert len(seriesG) == round(0.7 * len(test_series)) - 1
        assert len(seriesH) == round(0.3 * len(test_series)) + 1

        seriesI, seriesJ = test_series.split_after(5)
        assert len(seriesI) == 6
        assert len(seriesJ) == len(test_series) - 6

        seriesK, seriesL = test_series.split_before(5)
        assert len(seriesK) == 5
        assert len(seriesL) == len(test_series) - 5

        assert test_series.freq_str == seriesA.freq_str
        assert test_series.freq_str == seriesC.freq_str
        assert test_series.freq_str == seriesE.freq_str
        assert test_series.freq_str == seriesG.freq_str
        assert test_series.freq_str == seriesI.freq_str
        assert test_series.freq_str == seriesK.freq_str

        # Test split points outside of range
        for value in [-5, 1.1, pd.Timestamp("21300104")]:
            with pytest.raises(ValueError):
                test_series.split_before(value)

        # Test split points between series indeces
        times = pd.date_range("20130101", "20130120", freq="2D")
        pd_series = pd.Series(range(10), index=times)
        test_series2: TimeSeries = TimeSeries.from_series(pd_series)
        split_date = pd.Timestamp("20130110")
        seriesM, seriesN = test_series2.split_before(split_date)
        seriesO, seriesP = test_series2.split_after(split_date)
        assert seriesM.end_time() < split_date
        assert seriesN.start_time() >= split_date
        assert seriesO.end_time() <= split_date
        assert seriesP.start_time() > split_date

    @staticmethod
    def helper_test_drop(test_case, test_series: TimeSeries):
        seriesA = test_series.drop_after(pd.Timestamp("20130105"))
        assert seriesA.end_time() == pd.Timestamp("20130105") - test_series.freq
        assert np.all(seriesA.time_index < pd.Timestamp("20130105"))

        seriesB = test_series.drop_before(pd.Timestamp("20130105"))
        assert seriesB.start_time() == pd.Timestamp("20130105") + test_series.freq
        assert np.all(seriesB.time_index > pd.Timestamp("20130105"))

        assert test_series.freq_str == seriesA.freq_str
        assert test_series.freq_str == seriesB.freq_str

    @staticmethod
    def helper_test_intersect(test_case, test_series: TimeSeries):
        seriesA = TimeSeries.from_series(
            pd.Series(range(2, 8), index=pd.date_range("20130102", "20130107"))
        )

        seriesB = test_series.slice_intersect(seriesA)
        assert seriesB.start_time() == pd.Timestamp("20130102")
        assert seriesB.end_time() == pd.Timestamp("20130107")

        # Outside of range
        seriesD = test_series.slice_intersect(
            TimeSeries.from_series(
                pd.Series(range(6, 13), index=pd.date_range("20130106", "20130112"))
            )
        )
        assert seriesD.start_time() == pd.Timestamp("20130106")
        assert seriesD.end_time() == pd.Timestamp("20130110")

        # Small intersect
        seriesE = test_series.slice_intersect(
            TimeSeries.from_series(
                pd.Series(range(9, 13), index=pd.date_range("20130109", "20130112"))
            )
        )
        assert len(seriesE) == 2

        # No intersect
        with pytest.raises(ValueError):
            test_series.slice_intersect(
                TimeSeries(
                    pd.Series(range(6, 13), index=pd.date_range("20130116", "20130122"))
                )
            )

    def test_rescale(self):
        with pytest.raises(ValueError):
            self.series1.rescale_with_value(1)

        seriesA = self.series3.rescale_with_value(0)
        assert np.all(seriesA.values() == 0)

        seriesB = self.series3.rescale_with_value(-5)
        assert self.series3 * -1.0 == seriesB

        seriesC = self.series3.rescale_with_value(1)
        assert self.series3 * 0.2 == seriesC

        seriesD = self.series3.rescale_with_value(
            1e20
        )  # TODO: test will fail if value > 1e24 due to num imprecision
        assert self.series3 * 0.2e20 == seriesD

    @staticmethod
    def helper_test_shift(test_case, test_series: TimeSeries):
        seriesA = test_case.series1.shift(0)
        assert seriesA == test_case.series1

        seriesB = test_series.shift(1)
        assert seriesB.time_index.equals(
            test_series.time_index[1:].append(
                pd.DatetimeIndex([test_series.time_index[-1] + test_series.freq])
            )
        )

        seriesC = test_series.shift(-1)
        assert seriesC.time_index.equals(
            pd.DatetimeIndex([test_series.time_index[0] - test_series.freq]).append(
                test_series.time_index[:-1]
            )
        )

        with pytest.raises(Exception):
            test_series.shift(1e6)

        seriesM = TimeSeries.from_times_and_values(
            pd.date_range("20130101", "20130601", freq="m"), range(5)
        )
        with pytest.raises(OverflowError):
            seriesM.shift(1e4)

        seriesD = TimeSeries.from_times_and_values(
            pd.date_range("20130101", "20130101"), range(1), freq="D"
        )
        seriesE = seriesD.shift(1)
        assert seriesE.time_index[0] == pd.Timestamp("20130102")

        seriesF = TimeSeries.from_times_and_values(pd.RangeIndex(2, 10), range(8))

        seriesG = seriesF.shift(4)
        assert seriesG.time_index[0] == 6

    @staticmethod
    def helper_test_append(test_case, test_series: TimeSeries):
        # reconstruct series
        seriesA, seriesB = test_series.split_after(pd.Timestamp("20130106"))
        assert seriesA.append(seriesB) == test_series
        assert seriesA.append(seriesB).freq == test_series.freq
        assert test_series.time_index.equals(seriesA.append(seriesB).time_index)

        # Creating a gap is not allowed
        seriesC = test_series.drop_before(pd.Timestamp("20130108"))
        with pytest.raises(ValueError):
            seriesA.append(seriesC)

        # Changing frequency is not allowed
        seriesM = TimeSeries.from_times_and_values(
            pd.date_range("20130107", "20130507", freq="30D"), range(5)
        )
        with pytest.raises(ValueError):
            seriesA.append(seriesM)

    @staticmethod
    def helper_test_append_values(test_case, test_series: TimeSeries):
        # reconstruct series
        seriesA, seriesB = test_series.split_after(pd.Timestamp("20130106"))
        arrayB = seriesB.all_values()
        assert seriesA.append_values(arrayB) == test_series
        assert test_series.time_index.equals(seriesA.append_values(arrayB).time_index)

        # arrayB shape shouldn't affect append_values output:
        squeezed_arrayB = arrayB.squeeze()
        assert seriesA.append_values(squeezed_arrayB) == test_series
        assert test_series.time_index.equals(
            seriesA.append_values(squeezed_arrayB).time_index
        )

    @staticmethod
    def helper_test_prepend(test_case, test_series: TimeSeries):
        # reconstruct series
        seriesA, seriesB = test_series.split_after(pd.Timestamp("20130106"))
        assert seriesB.prepend(seriesA) == test_series
        assert seriesB.prepend(seriesA).freq == test_series.freq
        assert test_series.time_index.equals(seriesB.prepend(seriesA).time_index)

        # Creating a gap is not allowed
        seriesC = test_series.drop_before(pd.Timestamp("20130108"))
        with pytest.raises(ValueError):
            seriesC.prepend(seriesA)

        # Changing frequency is not allowed
        seriesM = TimeSeries.from_times_and_values(
            pd.date_range("20130107", "20130507", freq="30D"), range(5)
        )
        with pytest.raises(ValueError):
            seriesM.prepend(seriesA)

    @staticmethod
    def helper_test_prepend_values(test_case, test_series: TimeSeries):
        # reconstruct series
        seriesA, seriesB = test_series.split_after(pd.Timestamp("20130106"))
        arrayA = seriesA.data_array().values
        assert seriesB.prepend_values(arrayA) == test_series
        assert test_series.time_index.equals(seriesB.prepend_values(arrayA).time_index)

        # arrayB shape shouldn't affect append_values output:
        squeezed_arrayA = arrayA.squeeze()
        assert seriesB.prepend_values(squeezed_arrayA) == test_series
        assert test_series.time_index.equals(
            seriesB.prepend_values(squeezed_arrayA).time_index
        )

    def test_slice(self):
        TestTimeSeries.helper_test_slice(self, self.series1)

    def test_split(self):
        TestTimeSeries.helper_test_split(self, self.series1)

    def test_drop(self):
        TestTimeSeries.helper_test_drop(self, self.series1)

    def test_intersect(self):
        TestTimeSeries.helper_test_intersect(self, self.series1)

    def test_shift(self):
        TestTimeSeries.helper_test_shift(self, self.series1)

    def test_append(self):
        TestTimeSeries.helper_test_append(self, self.series1)
        # Check `append` deals with `RangeIndex` series correctly:
        series_1 = linear_timeseries(start=1, length=5, freq=2)
        series_2 = linear_timeseries(start=11, length=2, freq=2)
        appended = series_1.append(series_2)
        expected_vals = np.concatenate(
            [series_1.all_values(), series_2.all_values()], axis=0
        )
        expected_idx = pd.RangeIndex(start=1, stop=15, step=2)
        assert np.allclose(appended.all_values(), expected_vals)
        assert appended.time_index.equals(expected_idx)

    def test_append_values(self):
        TestTimeSeries.helper_test_append_values(self, self.series1)
        # Check `append_values` deals with `RangeIndex` series correctly:
        series = linear_timeseries(start=1, length=5, freq=2)
        appended = series.append_values(np.ones((2, 1, 1)))
        expected_vals = np.concatenate(
            [series.all_values(), np.ones((2, 1, 1))], axis=0
        )
        expected_idx = pd.RangeIndex(start=1, stop=15, step=2)
        assert np.allclose(appended.all_values(), expected_vals)
        assert appended.time_index.equals(expected_idx)

    def test_prepend(self):
        TestTimeSeries.helper_test_prepend(self, self.series1)
        # Check `prepend` deals with `RangeIndex` series correctly:
        series_1 = linear_timeseries(start=1, length=5, freq=2)
        series_2 = linear_timeseries(start=11, length=2, freq=2)
        prepended = series_2.prepend(series_1)
        expected_vals = np.concatenate(
            [series_1.all_values(), series_2.all_values()], axis=0
        )
        expected_idx = pd.RangeIndex(start=1, stop=15, step=2)
        assert np.allclose(prepended.all_values(), expected_vals)
        assert prepended.time_index.equals(expected_idx)

    def test_prepend_values(self):
        TestTimeSeries.helper_test_prepend_values(self, self.series1)
        # Check `prepend_values` deals with `RangeIndex` series correctly:
        series = linear_timeseries(start=1, length=5, freq=2)
        prepended = series.prepend_values(np.ones((2, 1, 1)))
        expected_vals = np.concatenate(
            [np.ones((2, 1, 1)), series.all_values()], axis=0
        )
        expected_idx = pd.RangeIndex(start=-3, stop=11, step=2)
        assert np.allclose(prepended.all_values(), expected_vals)
        assert prepended.time_index.equals(expected_idx)

    def test_with_values(self):
        vals = np.random.rand(5, 10, 3)
        series = TimeSeries.from_values(vals)
        series2 = series.with_values(vals + 1)
        series3 = series2.with_values(series2.all_values() - 1)

        # values should work
        np.testing.assert_allclose(series3.all_values(), series.all_values())
        np.testing.assert_allclose(series2.all_values(), vals + 1)

        # should fail if nr components is not the same:
        with pytest.raises(ValueError):
            series.with_values(np.random.rand(5, 11, 3))

        # should not fail if nr samples is not the same:
        series.with_values(np.random.rand(5, 10, 2))

    def test_cumsum(self):
        cumsum_expected = TimeSeries.from_dataframe(
            self.series1.pd_dataframe().cumsum()
        )
        # univariate deterministic
        assert self.series1.cumsum() == TimeSeries.from_dataframe(
            self.series1.pd_dataframe().cumsum()
        )
        # multivariate deterministic
        assert self.series1.stack(self.series1).cumsum() == cumsum_expected.stack(
            cumsum_expected
        )
        # multivariate stochastic
        # shape = (time steps, components, samples)
        ts = TimeSeries.from_values(np.random.random((10, 2, 10)))
        np.testing.assert_array_equal(
            ts.cumsum().all_values(copy=False),
            np.cumsum(ts.all_values(copy=False), axis=0),
        )

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

        with pytest.raises(ValueError):
            self.series1.diff(n=0)
        with pytest.raises(ValueError):
            self.series1.diff(n=-5)
        with pytest.raises(ValueError):
            self.series1.diff(n=0.2)
        with pytest.raises(ValueError):
            self.series1.diff(periods=0.2)

        assert self.series1.diff() == diff1_no_na
        assert self.series1.diff(n=2, dropna=True) == diff2_no_na
        assert self.series1.diff(dropna=False) == diff1
        assert self.series1.diff(n=2, dropna=0) == diff2
        assert self.series1.diff(periods=2, dropna=True) == diff_shift2_no_na
        assert self.series1.diff(n=2, periods=2, dropna=False) == diff2_shift2

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
            pd.Series([float(i**2) for i in range(10)], index=self.pd_series1.index)
        )

        assert self.series1 + seriesA == targetAdd
        assert self.series1 + 2 == targetAdd
        assert 2 + self.series1 == targetAdd
        assert self.series1 - seriesA == targetSub
        assert self.series1 - 2 == targetSub
        assert self.series1 * seriesA == targetMul
        assert self.series1 * 2 == targetMul
        assert 2 * self.series1 == targetMul
        assert self.series1 / seriesA == targetDiv
        assert self.series1 / 2 == targetDiv
        assert self.series1**2 == targetPow

        with pytest.raises(ZeroDivisionError):
            # Cannot divide by a TimeSeries with a value 0.
            self.series1 / self.series1

        with pytest.raises(ZeroDivisionError):
            # Cannot divide by 0.
            self.series1 / 0

    def test_getitem_datetime_index(self):
        seriesA: TimeSeries = self.series1.drop_after(pd.Timestamp("20130105"))
        assert self.series1[pd.date_range("20130101", " 20130104")] == seriesA
        assert self.series1[:4] == seriesA
        assert self.series1[pd.Timestamp("20130101")] == TimeSeries.from_dataframe(
            self.series1.pd_dataframe()[:1], freq=self.series1.freq
        )
        assert (
            self.series1[pd.Timestamp("20130101") : pd.Timestamp("20130104")] == seriesA
        )

        with pytest.raises(KeyError):
            self.series1[pd.date_range("19990101", "19990201")]

        with pytest.raises(KeyError):
            self.series1["19990101"]

        with pytest.raises(IndexError):
            self.series1[::-1]

    def test_getitem_integer_index(self):
        freq = 3
        start = 1
        end = start + (len(self.series1) - 1) * freq
        idx_int = pd.RangeIndex(start=start, stop=end + freq, step=freq)
        series = TimeSeries.from_times_and_values(
            times=idx_int, values=self.series1.values()
        )
        assert series.freq == freq
        assert series.start_time() == start
        assert series.end_time() == end
        assert series[idx_int] == series == series[0 : len(series)]

        series_single = series.drop_after(start + 2 * freq)
        assert (
            series[pd.RangeIndex(start=start, stop=start + 2 * freq, step=freq)]
            == series_single
        )
        assert series[:2] == series_single
        assert series_single.freq == freq
        assert series_single.start_time() == start
        assert series_single.end_time() == start + freq

        idx_single = pd.RangeIndex(start=start + freq, stop=start + 2 * freq, step=freq)
        assert series[idx_single].time_index == idx_single
        assert series[idx_single].pd_series().equals(series.pd_series()[1:2])
        assert series[idx_single] == series[1:2] == series[1]

        # cannot slice with two RangeIndex
        with pytest.raises(IndexError):
            _ = series[idx_single : idx_single + freq]

        # RangeIndex not in time_index
        with pytest.raises(KeyError):
            _ = series[idx_single - 1]

        # RangeIndex start is out of bounds
        with pytest.raises(KeyError):
            _ = series[pd.RangeIndex(start - freq, stop=end + freq, step=freq)]

        # RangeIndex end is out of bounds
        with pytest.raises(KeyError):
            _ = series[pd.RangeIndex(start, stop=end + 2 * freq, step=freq)]

    def test_fill_missing_dates(self):
        with pytest.raises(ValueError):
            # Series cannot have date holes without automatic filling
            range_ = pd.date_range("20130101", "20130104").append(
                pd.date_range("20130106", "20130110")
            )
            TimeSeries.from_series(
                pd.Series(range(9), index=range_), fill_missing_dates=False
            )

        with pytest.raises(ValueError):
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
        assert series_test.freq_str == "D"

        range_ = pd.date_range("20130101", "20130104", freq="2D").append(
            pd.date_range("20130107", "20130111", freq="2D")
        )
        series_test = TimeSeries.from_series(
            pd.Series(range(5), index=range_), fill_missing_dates=True
        )
        assert series_test.freq_str == "2D"
        assert series_test.start_time() == range_[0]
        assert series_test.end_time() == range_[-1]
        assert math.isnan(series_test.pd_series().get("20130105"))

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
                    with pytest.raises(ValueError):
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
                        assert series == series_target
                    assert series.time_index.equals(series_target.time_index)

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
            assert np.isnan(series_with_nan.all_values(copy=False)).any()
        for series_no_nan in [
            series_1,
            series_nan_fillna,
            series_1_fillna,
            series_holes_fillna,
        ]:
            assert not np.isnan(series_no_nan.all_values(copy=False)).any()
            assert series_1 == series_no_nan

    def test_resample_timeseries(self):
        times = pd.date_range("20130101", "20130110")
        pd_series = pd.Series(range(10), index=times)
        timeseries = TimeSeries.from_series(pd_series)

        resampled_timeseries = timeseries.resample("H")
        assert resampled_timeseries.freq_str == "H"
        assert resampled_timeseries.pd_series().at[pd.Timestamp("20130101020000")] == 0
        assert resampled_timeseries.pd_series().at[pd.Timestamp("20130102020000")] == 1
        assert resampled_timeseries.pd_series().at[pd.Timestamp("20130109090000")] == 8

        resampled_timeseries = timeseries.resample("2D")
        assert resampled_timeseries.freq_str == "2D"
        assert resampled_timeseries.pd_series().at[pd.Timestamp("20130101")] == 0
        with pytest.raises(KeyError):
            resampled_timeseries.pd_series().at[pd.Timestamp("20130102")]

        assert resampled_timeseries.pd_series().at[pd.Timestamp("20130109")] == 8

        # using offset to avoid nan in the first value
        times = pd.date_range(
            start=pd.Timestamp("20200101233000"), periods=10, freq="15T"
        )
        pd_series = pd.Series(range(10), index=times)
        timeseries = TimeSeries.from_series(pd_series)
        resampled_timeseries = timeseries.resample(
            freq="1h", offset=pd.Timedelta("30T")
        )
        assert resampled_timeseries.pd_series().at[pd.Timestamp("20200101233000")] == 0

    def test_short_series_creation(self):
        # test missing freq argument error when filling missing dates on short time series
        with pytest.raises(ValueError):
            TimeSeries.from_times_and_values(
                pd.date_range("20130101", "20130102"), range(2), fill_missing_dates=True
            )
        # test empty pandas series with DatetimeIndex
        freq = "D"
        # fails without freq
        with pytest.raises(ValueError):
            TimeSeries.from_series(pd.Series(index=pd.DatetimeIndex([])))
        # works with index having freq, or setting freq at TimeSeries creation
        series_a = TimeSeries.from_series(
            pd.Series(index=pd.DatetimeIndex([], freq=freq))
        )
        assert series_a.freq == freq
        assert len(series_a) == 0
        series_b = TimeSeries.from_series(
            pd.Series(index=pd.DatetimeIndex([])), freq=freq
        )
        assert series_a == series_b

        # test empty pandas series with DatetimeIndex
        freq = 2
        # fails pd.Index (IntIndex)
        with pytest.raises(ValueError):
            TimeSeries.from_series(pd.Series(index=pd.Index([])))
        # works with pd.RangeIndex as freq (step) is given by default (step=1)
        series_a = TimeSeries.from_series(pd.Series(index=pd.RangeIndex(start=0)))
        assert series_a.freq == 1
        # works with RangeIndex of different freq, or setting freq at TimeSeries creation
        series_a = TimeSeries.from_series(
            pd.Series(index=pd.RangeIndex(start=0, step=freq))
        )
        assert series_a.freq == freq
        assert len(series_a) == 0
        series_b = TimeSeries.from_series(
            pd.Series(index=pd.RangeIndex(start=0)), freq=freq
        )
        assert series_a == series_b

        # frequency should be ignored when fill_missing_dates is False
        seriesA = TimeSeries.from_times_and_values(
            pd.date_range("20130101", "20130105"),
            range(5),
            fill_missing_dates=False,
            freq="M",
        )
        assert seriesA.freq == "D"
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

        assert data_darts1 == data_darts2

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
        assert ts1.has_range_index

        ts2 = TimeSeries.from_dataframe(df2)
        assert ts2.has_datetime_index

        ts3 = TimeSeries.from_dataframe(df3, time_col="Time")
        assert ts3.has_datetime_index

        ts4 = TimeSeries.from_series(series1)
        assert ts4.has_range_index

        ts5 = TimeSeries.from_series(series2)
        assert ts5.has_datetime_index

        ts6 = TimeSeries.from_times_and_values(times=times, values=values1)
        assert ts6.has_datetime_index

        ts7 = TimeSeries.from_times_and_values(times=times, values=df1)
        assert ts7.has_datetime_index

        ts8 = TimeSeries.from_values(values1)
        assert ts8.has_range_index

    def test_short_series_slice(self):
        seriesA, seriesB = self.series1.split_after(pd.Timestamp("20130108"))
        assert len(seriesA) == 8
        assert len(seriesB) == 2
        seriesA, seriesB = self.series1.split_after(pd.Timestamp("20130109"))
        assert len(seriesA) == 9
        assert len(seriesB) == 1
        assert seriesB.time_index[0] == self.series1.time_index[-1]
        seriesA, seriesB = self.series1.split_before(pd.Timestamp("20130103"))
        assert len(seriesA) == 2
        assert len(seriesB) == 8
        seriesA, seriesB = self.series1.split_before(pd.Timestamp("20130102"))
        assert len(seriesA) == 1
        assert len(seriesB) == 9
        assert seriesA.time_index[-1] == self.series1.time_index[0]
        seriesC = self.series1.slice(pd.Timestamp("20130105"), pd.Timestamp("20130105"))
        assert len(seriesC) == 1

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

        assert series_0["0"] == series["0"].map(fn)
        assert series_2["2"] == series["2"].map(fn)
        assert series_01[["0", "1"]] == series[["0", "1"]].map(fn)
        assert series_012 == series[["0", "1", "2"]].map(fn)
        assert series_012 == series.map(fn)

        assert series_01 != series[["0", "1"]].map(fn)

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
        assert new_series == zeroes

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

        with pytest.raises(ValueError):
            series.map(add)

        ufunc_add = np.frompyfunc(add, 3, 1)

        with pytest.raises(ValueError):
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
        pd_series7 = pd.Series(
            [1] * 10 + 1 * [0] + [1] * 13 + 5 * [2] + [1] * 18 + 9 * [6],
            index=times4,
        )

        series1 = TimeSeries.from_series(pd_series1)
        series2 = TimeSeries.from_series(pd_series2)
        series3 = TimeSeries.from_series(pd_series3)
        series4 = TimeSeries.from_series(pd_series4)
        series5 = TimeSeries.from_series(pd_series5)
        series6 = TimeSeries.from_series(pd_series6)
        series7 = TimeSeries.from_series(pd_series7)

        gaps1 = series1.gaps()
        assert (
            gaps1["gap_start"]
            == pd.DatetimeIndex([pd.Timestamp("20130103"), pd.Timestamp("20130109")])
        ).all()
        assert (
            gaps1["gap_end"]
            == pd.DatetimeIndex([pd.Timestamp("20130105"), pd.Timestamp("20130110")])
        ).all()
        assert gaps1["gap_size"].values.tolist() == [3, 2]
        gaps2 = series2.gaps()
        assert gaps2["gap_size"].values.tolist() == [3, 3]
        gaps3 = series3.gaps()
        assert gaps3["gap_size"].values.tolist() == [10]
        gaps4 = series4.gaps()
        assert gaps4["gap_size"].values.tolist() == [3, 7, 1]
        gaps5 = series5.gaps()
        assert gaps5["gap_size"].values.tolist() == [2, 2]
        assert (
            gaps5["gap_start"]
            == pd.DatetimeIndex([pd.Timestamp("20150101"), pd.Timestamp("20180101")])
        ).all()
        assert (
            gaps5["gap_end"]
            == pd.DatetimeIndex([pd.Timestamp("20160101"), pd.Timestamp("20190101")])
        ).all()
        gaps6 = series6.gaps()
        assert gaps6["gap_size"].values.tolist() == [1, 5, 9]
        assert (
            gaps6["gap_start"]
            == pd.DatetimeIndex(
                [
                    pd.Timestamp("20130901"),
                    pd.Timestamp("20160101"),
                    pd.Timestamp("20191101"),
                ]
            )
        ).all()
        assert (
            gaps6["gap_end"]
            == pd.DatetimeIndex(
                [
                    pd.Timestamp("20130901"),
                    pd.Timestamp("20160901"),
                    pd.Timestamp("20210301"),
                ]
            )
        ).all()
        gaps7 = series7.gaps()
        assert gaps7.empty

        # test gaps detection on integer-indexed series
        values = np.array([1, 2, np.nan, np.nan, 3, 4, np.nan, 6])
        times = pd.RangeIndex(8)
        ts = TimeSeries.from_times_and_values(times, values)
        np.testing.assert_equal(ts.gaps().values, np.array([[2, 3, 2], [6, 6, 1]]))

        values = np.array([1, 2, 7, 8, 3, 4, 0, 6])
        times = pd.RangeIndex(8)
        ts = TimeSeries.from_times_and_values(times, values)
        assert ts.gaps().empty

    def test_longest_contiguous_slice(self):
        times = pd.date_range("20130101", "20130111")
        pd_series1 = pd.Series(
            [1, 1] + 3 * [np.nan] + [1, 1, 1] + [np.nan] * 2 + [1], index=times
        )
        series1 = TimeSeries.from_series(pd_series1)

        assert len(series1.longest_contiguous_slice()) == 3
        assert len(series1.longest_contiguous_slice(2)) == 6

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
        assert ["linear1", "linear2"] == series1.columns.to_list()

        with pytest.raises(ValueError):
            series1.with_columns_renamed(
                ["linear1", "linear2"], ["linear1", "linear3", "linear4"]
            )

        #  Linear7 doesn't exist
        with pytest.raises(ValueError):
            series1.with_columns_renamed("linear7", "linear5")

    def test_to_csv_probabilistic_ts(self):
        samples = [
            linear_timeseries(start_value=val, length=10) for val in [10, 20, 30]
        ]
        ts = concatenate(samples, axis=2)
        with pytest.raises(AssertionError):
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

        with pytest.raises(AssertionError):
            ts.to_csv("test.csv")


class TestTimeSeriesConcatenate:

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
        assert (10, 3, 1) == ts._xa.shape

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

        with pytest.raises(ValueError):
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
        assert (10, 3, 1) == ts._xa.shape
        assert pd.Timestamp("2000-01-01") == ts.start_time()
        assert pd.Timestamp("2000-01-10") == ts.end_time()

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

        with pytest.raises(ValueError):
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
        assert (10, 1, 3) == ts._xa.shape

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
        assert (30, 1, 1) == ts._xa.shape
        assert pd.Timestamp("2000-01-01") == ts.start_time()
        assert pd.Timestamp("2000-01-30") == ts.end_time()

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

        with pytest.raises(ValueError):
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
        assert (30, 1, 1) == ts._xa.shape
        assert pd.Timestamp("2000-01-01") == ts.start_time()
        assert pd.Timestamp("2000-01-30") == ts.end_time()

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

        with pytest.raises(ValueError):
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
        assert (30, 1, 1) == ts._xa.shape
        assert pd.Timestamp("2000-01-01") == ts.start_time()
        assert pd.Timestamp("2000-01-30") == ts.end_time()

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
        assert (30, 1, 1) == ts._xa.shape
        assert pd.Timestamp("2000-01-01") == ts.start_time()
        assert pd.Timestamp("2000-02-28") == ts.end_time()
        assert "2D" == ts.freq

    def test_concatenate_timeseries_method(self):
        ts1 = linear_timeseries(
            start_value=10, length=10, start=pd.Timestamp("2000-01-01"), freq="D"
        )
        ts2 = linear_timeseries(
            start_value=20, length=10, start=pd.Timestamp("2000-01-11"), freq="D"
        )

        result_ts = ts1.concatenate(ts2, axis="time")
        assert (20, 1, 1) == result_ts._xa.shape
        assert pd.Timestamp("2000-01-01") == result_ts.start_time()
        assert pd.Timestamp("2000-01-20") == result_ts.end_time()
        assert "D" == result_ts.freq


class TestTimeSeriesHierarchy:

    components = ["total", "a", "b", "x", "y", "ax", "ay", "bx", "by"]

    hierarchy = {
        "ax": ["a", "x"],
        "ay": ["a", "y"],
        "bx": ["b", "x"],
        "by": ["b", "y"],
        "a": ["total"],
        "b": ["total"],
        "x": ["total"],
        "y": ["total"],
    }

    base_series = TimeSeries.from_values(
        values=np.random.rand(50, len(components), 5), columns=components
    )

    def test_creation_with_hierarchy_sunny_day(self):
        hierarchical_series = TimeSeries.from_values(
            values=np.random.rand(50, len(self.components), 5),
            columns=self.components,
            hierarchy=self.hierarchy,
        )
        assert hierarchical_series.hierarchy == self.hierarchy

    def test_with_hierarchy_sunny_day(self):
        hierarchical_series = self.base_series.with_hierarchy(self.hierarchy)
        assert hierarchical_series.hierarchy == self.hierarchy

    def test_with_hierarchy_rainy_day(self):
        # wrong type
        with pytest.raises(ValueError):
            self.base_series.with_hierarchy(set())

        # wrong keys
        with pytest.raises(ValueError):
            hierarchy = {"ax": ["a", "x"]}
            self.base_series.with_hierarchy(hierarchy)

        with pytest.raises(ValueError):
            hierarchy = {"unknown": ["a", "x"]}
            self.base_series.with_hierarchy(hierarchy)

        with pytest.raises(ValueError):
            hierarchy = {
                "unknown": ["a", "x"],
                "ay": ["a", "y"],
                "bx": ["b", "x"],
                "by": ["b", "y"],
                "a": ["total"],
                "b": ["total"],
                "x": ["total"],
                "y": ["total"],
            }
            self.base_series.with_hierarchy(hierarchy)

        with pytest.raises(ValueError):
            hierarchy = {
                "total": ["a", "x"],
                "ay": ["a", "y"],
                "bx": ["b", "x"],
                "by": ["b", "y"],
                "a": ["total"],
                "b": ["total"],
                "x": ["total"],
                "y": ["total"],
            }
            self.base_series.with_hierarchy(hierarchy)

        # wrong values
        with pytest.raises(ValueError):
            hierarchy = {
                "ax": ["unknown", "x"],
                "ay": ["a", "y"],
                "bx": ["b", "x"],
                "by": ["b", "y"],
                "a": ["total"],
                "b": ["total"],
                "x": ["total"],
                "y": ["total"],
            }
            self.base_series.with_hierarchy(hierarchy)

    def test_hierarchy_processing(self):
        hierarchical_series = self.base_series.with_hierarchy(self.hierarchy)
        assert hierarchical_series.has_hierarchy
        assert not self.base_series.has_hierarchy
        assert hierarchical_series.bottom_level_components == ["ax", "ay", "bx", "by"]
        assert hierarchical_series.top_level_component == "total"

        top_level_idx = self.components.index("total")
        np.testing.assert_equal(
            hierarchical_series.top_level_series.all_values(copy=False)[:, 0, :],
            self.base_series.all_values(copy=False)[:, top_level_idx, :],
        )

        np.testing.assert_equal(
            hierarchical_series.bottom_level_series.all_values(copy=False),
            hierarchical_series[["ax", "ay", "bx", "by"]].all_values(copy=False),
        )

    def test_concat(self):
        series1 = self.base_series.with_hierarchy(self.hierarchy)
        series2 = self.base_series.with_hierarchy(self.hierarchy)

        # concat on time or samples should preserve hierarchy:
        concat_s = concatenate([series1, series2], axis=0, ignore_time_axis=True)
        assert concat_s.hierarchy == self.hierarchy

        concat_s = concatenate([series1, series2], axis=2)
        assert concat_s.hierarchy == self.hierarchy

        # concat on components should fail when not ignoring hierarchy
        with pytest.raises(ValueError):
            concat_s = concatenate([series1, series2], axis=1, drop_hierarchy=False)

        # concat on components should work when dropping hierarchy
        concat_s = concatenate([series1, series2], axis=1, drop_hierarchy=True)
        assert not concat_s.has_hierarchy

        # hierarchy should be dropped when selecting components:
        subs1 = series1[["ax", "ay", "bx", "by"]]
        assert not subs1.has_hierarchy
        subs2 = series1["total"]
        assert not subs2.has_hierarchy

    def test_ops(self):
        # another hierarchy different than the original
        hierarchy2 = {
            "ax": ["b", "y"],
            "ay": ["b", "x"],
            "bx": ["a", "y"],
            "by": ["a", "x"],
            "a": ["total"],
            "b": ["total"],
            "x": ["total"],
            "y": ["total"],
        }

        # Ops not touching components should preserve hierarchy
        series1 = self.base_series.with_hierarchy(self.hierarchy)
        series2 = self.base_series.with_hierarchy(hierarchy2)

        assert series1[:10].hierarchy == self.hierarchy
        assert (series1 + 10).hierarchy == self.hierarchy

        # combining series should keep hierarchy of first series
        assert (series1 / series2).hierarchy == self.hierarchy
        assert (series1.slice_intersect(series2[10:20])).hierarchy == self.hierarchy

    def test_with_string_items(self):
        # Single parents may be specified as string rather than [string]
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        nr_dates = len(dates)
        t1 = TimeSeries.from_times_and_values(
            dates, 3 * np.ones(nr_dates), columns=["T1"]
        )
        t2 = TimeSeries.from_times_and_values(
            dates, 5 * np.ones(nr_dates), columns=["T2"]
        )
        t3 = TimeSeries.from_times_and_values(dates, np.ones(nr_dates), columns=["T3"])
        tsum = TimeSeries.from_times_and_values(
            dates, 9 * np.ones(nr_dates), columns=["T_sum"]
        )

        ts = concatenate([t1, t2, t3, tsum], axis="component")
        string_hierarchy = {"T1": "T_sum", "T2": "T_sum", "T3": "T_sum"}
        ts_with_string_hierarchy = ts.with_hierarchy(string_hierarchy)
        hierarchy_as_list = {k: [v] for k, v in string_hierarchy.items()}
        assert ts_with_string_hierarchy.hierarchy == hierarchy_as_list
        list_hierarchy = {"T1": ["T_sum"], "T2": ["T_sum"], "T3": ["T_sum"]}
        ts_with_list_hierarchy = ts.with_hierarchy(list_hierarchy)
        assert ts_with_string_hierarchy.hierarchy == ts_with_list_hierarchy.hierarchy


class TestTimeSeriesHeadTail:

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
        assert 5 == result.n_timesteps
        assert pd.Timestamp("2000-01-05") == result.end_time()

    def test_head_sunny_day_component_axis(self):
        result = self.ts.head(axis=1)
        assert 5 == result.n_components
        assert ["comp_0", "comp_1", "comp_2", "comp_3", "comp_4"] == result._xa.coords[
            "component"
        ].values.tolist()

    def test_tail_sunny_day_time_axis(self):
        result = self.ts.tail()
        assert 5 == result.n_timesteps
        assert pd.Timestamp("2000-01-06") == result.start_time()

    def test_tail_sunny_day_component_axis(self):
        result = self.ts.tail(axis=1)
        assert 5 == result.n_components
        assert ["comp_5", "comp_6", "comp_7", "comp_8", "comp_9"] == result._xa.coords[
            "component"
        ].values.tolist()

    def test_head_sunny_day_sample_axis(self):
        result = self.ts.tail(axis=2)
        assert 5 == result.n_samples
        assert list(range(5, 10)) == result._xa.coords["sample"].values.tolist()

    def test_head_overshot_time_axis(self):
        result = self.ts.head(20)
        assert 10 == result.n_timesteps
        assert pd.Timestamp("2000-01-10") == result.end_time()

    def test_head_overshot_component_axis(self):
        result = self.ts.head(20, axis="component")
        assert 10 == result.n_components

    def test_head_overshot_sample_axis(self):
        result = self.ts.head(20, axis="sample")
        assert 10 == result.n_samples

    def test_head_numeric_time_index(self):
        s = TimeSeries.from_values(self.ts.values())
        # taking the head should not crash
        s.head()

    def test_tail_overshot_time_axis(self):
        result = self.ts.tail(20)
        assert 10 == result.n_timesteps
        assert pd.Timestamp("2000-01-01") == result.start_time()

    def test_tail_overshot_component_axis(self):
        result = self.ts.tail(20, axis="component")
        assert 10 == result.n_components

    def test_tail_overshot_sample_axis(self):
        result = self.ts.tail(20, axis="sample")
        assert 10 == result.n_samples

    def test_tail_numeric_time_index(self):
        s = TimeSeries.from_values(self.ts.values())
        # taking the tail should not crash
        s.tail()


class TestTimeSeriesFromDataFrame:
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

        assert data_darts1 == data_darts2
        assert data_darts1 == data_darts3

    def test_time_col_convert_string_integers(self):
        expected = np.array(list(range(3, 10)))
        data_dict = {"Time": expected.astype(str)}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        ts = TimeSeries.from_dataframe(df=df, time_col="Time")

        assert set(ts.time_index.values.tolist()) == set(expected)
        assert ts.time_index.dtype == int
        assert ts.time_index.name == "Time"

    def test_time_col_convert_integers(self):
        expected = np.array(list(range(10)))
        data_dict = {"Time": expected}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        ts = TimeSeries.from_dataframe(df=df, time_col="Time")

        assert set(ts.time_index.values.tolist()) == set(expected)
        assert ts.time_index.dtype == int
        assert ts.time_index.name == "Time"

    def test_fail_with_bad_integer_time_col(self):
        bad_time_col_vals = np.array([4, 0, 1, 2])
        data_dict = {"Time": bad_time_col_vals}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        with pytest.raises(ValueError):
            TimeSeries.from_dataframe(df=df, time_col="Time")

    def test_time_col_convert_rangeindex(self):
        for expected_l, step in zip([[4, 0, 2, 3, 1], [8, 0, 4, 6, 2]], [1, 2]):
            expected = np.array(expected_l)
            data_dict = {"Time": expected}
            data_dict["Values1"] = np.random.uniform(
                low=-10, high=10, size=len(data_dict["Time"])
            )
            df = pd.DataFrame(data_dict)
            ts = TimeSeries.from_dataframe(df=df, time_col="Time")

            # check type (should convert to RangeIndex):
            assert type(ts.time_index) == pd.RangeIndex

            # check values inside the index (should be sorted correctly):
            assert list(ts.time_index) == sorted(expected)

            # check that values are sorted accordingly:
            ar1 = ts.values(copy=False)[:, 0]
            ar2 = data_dict["Values1"][
                list(expected_l.index(i * step) for i in range(len(expected)))
            ]
            assert np.all(ar1 == ar2)

    def test_time_col_convert_datetime(self):
        expected = pd.date_range(start="20180501", end="20200301", freq="MS")
        data_dict = {"Time": expected}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        ts = TimeSeries.from_dataframe(df=df, time_col="Time")

        assert ts.time_index.dtype == "datetime64[ns]"
        assert ts.time_index.name == "Time"

    def test_time_col_convert_datetime_strings(self):
        expected = pd.date_range(start="20180501", end="20200301", freq="MS")
        data_dict = {"Time": expected.values.astype(str)}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        ts = TimeSeries.from_dataframe(df=df, time_col="Time")

        assert ts.time_index.dtype == "datetime64[ns]"
        assert ts.time_index.name == "Time"

    def test_time_col_with_tz(self):
        # numpy and xarray don't support "timezone aware" pd.DatetimeIndex
        # the BUGFIX removes timezone information without conversion

        time_range_MS = pd.date_range(
            start="20180501", end="20200301", freq="MS", tz="CET"
        )
        values = np.random.uniform(low=-10, high=10, size=len(time_range_MS))
        # pd.DataFrame loses the tz information unless it is contained in its index
        # (other columns are silently converted to UTC, with tz attribute set to None)
        df = pd.DataFrame(data=values, index=time_range_MS)
        ts = TimeSeries.from_dataframe(df=df)
        assert list(ts.time_index) == list(time_range_MS.tz_localize(None))
        assert list(ts.time_index.tz_localize("CET")) == list(time_range_MS)
        assert ts.time_index.tz is None

        serie = pd.Series(data=values, index=time_range_MS)
        ts = TimeSeries.from_series(pd_series=serie)
        assert list(ts.time_index) == list(time_range_MS.tz_localize(None))
        assert list(ts.time_index.tz_localize("CET")) == list(time_range_MS)
        assert ts.time_index.tz is None

        ts = TimeSeries.from_times_and_values(times=time_range_MS, values=values)
        assert list(ts.time_index) == list(time_range_MS.tz_localize(None))
        assert list(ts.time_index.tz_localize("CET")) == list(time_range_MS)
        assert ts.time_index.tz is None

        time_range_H = pd.date_range(
            start="20200518", end="20200521", freq="H", tz="CET"
        )
        values = np.random.uniform(low=-10, high=10, size=len(time_range_H))

        df = pd.DataFrame(data=values, index=time_range_H)
        ts = TimeSeries.from_dataframe(df=df)
        assert list(ts.time_index) == list(time_range_H.tz_localize(None))
        assert list(ts.time_index.tz_localize("CET")) == list(time_range_H)
        assert ts.time_index.tz is None

        serie = pd.Series(data=values, index=time_range_H)
        ts = TimeSeries.from_series(pd_series=serie)
        assert list(ts.time_index) == list(time_range_H.tz_localize(None))
        assert list(ts.time_index.tz_localize("CET")) == list(time_range_H)
        assert ts.time_index.tz is None

        ts = TimeSeries.from_times_and_values(times=time_range_H, values=values)
        assert list(ts.time_index) == list(time_range_H.tz_localize(None))
        assert list(ts.time_index.tz_localize("CET")) == list(time_range_H)
        assert ts.time_index.tz is None

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

        with pytest.raises(AttributeError):
            TimeSeries.from_dataframe(df=df, time_col="Time")

    def test_df_named_columns_index(self):
        time_index = generate_index(
            start=pd.Timestamp("2000-01-01"), length=4, freq="D", name="index"
        )
        df = pd.DataFrame(
            data=np.arange(4),
            index=time_index,
            columns=["y"],
        )
        df.columns.name = "id"
        ts = TimeSeries.from_dataframe(df)

        exp_ts = TimeSeries.from_times_and_values(
            times=time_index,
            values=np.arange(4),
            columns=["y"],
        )
        # check that series are exactly identical
        assert ts == exp_ts
        # check that the original df was not changed
        assert df.columns.name == "id"


class TestSimpleStatistics:

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
            assert np.isclose(
                new_ts._xa.values, self.values.mean(axis=axis, keepdims=True)
            ).all()

    def test_var(self):
        for ddof in range(5):
            new_ts = self.ts.var(ddof=ddof)
            # check values
            assert np.isclose(new_ts.values(), self.values.var(ddof=ddof, axis=2)).all()

    def test_std(self):
        for ddof in range(5):
            new_ts = self.ts.std(ddof=ddof)
            # check values
            assert np.isclose(new_ts.values(), self.values.std(ddof=ddof, axis=2)).all()

    def test_skew(self):
        new_ts = self.ts.skew()
        # check values
        assert np.isclose(new_ts.values(), skew(self.values, axis=2)).all()

    def test_kurtosis(self):
        new_ts = self.ts.kurtosis()
        # check values
        assert np.isclose(
            new_ts.values(),
            kurtosis(self.values, axis=2),
        ).all()

    def test_min(self):
        for axis in range(3):
            new_ts = self.ts.min(axis=axis)
            # check values
            assert np.isclose(
                new_ts._xa.values, self.values.min(axis=axis, keepdims=True)
            ).all()

    def test_max(self):
        for axis in range(3):
            new_ts = self.ts.max(axis=axis)
            # check values
            assert np.isclose(
                new_ts._xa.values, self.values.max(axis=axis, keepdims=True)
            ).all()

    def test_sum(self):
        for axis in range(3):
            new_ts = self.ts.sum(axis=axis)
            # check values
            assert np.isclose(
                new_ts._xa.values, self.values.sum(axis=axis, keepdims=True)
            ).all()

    def test_median(self):
        for axis in range(3):
            new_ts = self.ts.median(axis=axis)
            # check values
            assert np.isclose(
                new_ts._xa.values, np.median(self.values, axis=axis, keepdims=True)
            ).all()

    def test_quantile(self):
        for q in [0.01, 0.1, 0.5, 0.95]:
            new_ts = self.ts.quantile(quantile=q)
            # check values
            assert np.isclose(
                new_ts.values(),
                np.quantile(self.values, q=q, axis=2),
            ).all()
