import itertools
import logging
import math
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import kurtosis, skew

from darts import TimeSeries, concatenate, slice_intersect
from darts.tests.conftest import POLARS_AVAILABLE
from darts.utils.timeseries_generation import constant_timeseries, linear_timeseries
from darts.utils.utils import expand_arr, freqs, generate_index

TEST_BACKENDS = ["pandas"]

if POLARS_AVAILABLE:
    import polars as pl

    TEST_BACKENDS.append("polars")
else:
    pl = None


class TestTimeSeries:
    times = pd.date_range("20130101", "20130110", freq="D")
    pd_series1 = pd.Series(range(10), index=times)
    pd_series2 = pd.Series(range(5, 15), index=times)
    pd_series3 = pd.Series(range(15, 25), index=times)
    series1: TimeSeries = TimeSeries.from_series(pd_series1)
    series2: TimeSeries = TimeSeries.from_series(pd_series2)
    series3: TimeSeries = TimeSeries.from_series(pd_series3)

    def test_creation(self):
        series_test = TimeSeries.from_series(self.pd_series1)
        assert series_test.to_series().equals(self.pd_series1.astype(float))

        # Creation with a well-formed array:
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

    def test_pandas_creation(self):
        pd_series = pd.Series(range(10), name="test_name", dtype="float32")
        ts = TimeSeries.from_series(pd_series)
        ts_pd_series = ts.to_series()
        assert ts_pd_series.equals(pd_series)
        assert ts_pd_series.name == pd_series.name

        pd_df = pd_series.to_frame()
        ts = TimeSeries.from_dataframe(pd_df)
        ts_pd_df = ts.to_dataframe()
        assert ts_pd_df.equals(pd_df)

        ts_pd_df = ts.to_dataframe(time_as_index=False)
        assert ts_pd_df.equals(pd_df.reset_index())

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="requires polars")
    def test_polars_creation(self, caplog):
        expected_idx = pl.Series("time", range(10))
        pl_series = pl.Series("test_name", range(1, 11), dtype=pl.Float32)

        # without time_col, Darts generates a RangeIndex and raises a warning
        warning_expected = "No time column specified (`time_col=None`) and no index found in the `DataFrame`."
        with caplog.at_level(logging.WARNING):
            ts = TimeSeries.from_series(pl_series)
            assert warning_expected in caplog.text
        caplog.clear()

        ts_pl_series = ts.to_series(backend="polars")
        assert ts_pl_series[:, 0].equals(expected_idx)
        assert ts_pl_series.columns[0] == expected_idx.name
        assert ts_pl_series[:, 1].equals(pl_series)
        assert ts_pl_series.columns[1] == pl_series.name

        pl_df = pl.DataFrame(
            data={
                "time": pd.date_range(start="2023-01-01", periods=10, freq="D"),
                "test_float": [float(i) for i in range(10)],
            }
        )
        # with a `time_col` no warning is raised
        with caplog.at_level(logging.WARNING):
            ts = TimeSeries.from_dataframe(pl_df, time_col="time")
            assert caplog.text == ""
        caplog.clear()
        ts_pl_df = ts.to_dataframe(backend="polars", time_as_index=False)
        assert ts_pl_df.equals(pl_df)
        assert ts_pl_df.dtypes == pl_df.dtypes

        # setting time_as_index=True has no effect but raises a warning
        # with a `time_col` no warning is raised
        warning_expected = '`time_as_index=True` is only supported with `backend="pandas"`, and will be ignored.'
        with caplog.at_level(logging.WARNING):
            ts_pl_df_2 = ts.to_dataframe(backend="polars", time_as_index=True)
            assert warning_expected in caplog.text
        caplog.clear()
        assert ts_pl_df_2.equals(pl_df)
        assert ts_pl_df_2.dtypes == pl_df.dtypes

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
        mseries = mseries.with_hierarchy({
            "component_1": ["component"],
            "component_2": ["component"],
        })

        static_cov = pd.DataFrame({
            "dim0": [1, 2, 3],
            "dim1": [-2, -1, 0],
            "dim2": [0.0, 0.1, 0.2],
        })

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
        assert all(series_test.to_series().values == self.pd_series1.values)
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

        # Test split points between series indices
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

    def test_rescale(self):
        with pytest.raises(ValueError):
            self.series1.rescale_with_value(1)

        seriesA = self.series2.rescale_with_value(0)
        assert np.all(seriesA.values() == 0)

        seriesB = self.series2.rescale_with_value(-5)
        assert self.series2 * -1.0 == seriesB

        seriesC = self.series2.rescale_with_value(1)
        assert self.series2 * 0.2 == seriesC

        seriesD = self.series2.rescale_with_value(
            1e20
        )  # TODO: test will fail if value > 1e24 due to num imprecision
        assert self.series2 * 0.2e20 == seriesD

    @staticmethod
    def helper_test_intersect(freq, is_mixed_freq: bool, is_univariate: bool):
        start = pd.Timestamp("20130101") if isinstance(freq, str) else 0
        freq = pd.tseries.frequencies.to_offset(freq) if isinstance(freq, str) else freq

        # handle identical and mixed frequency setup
        if not is_mixed_freq:
            freq_other = freq
            n_steps = 11
        elif "2" not in str(freq):  # 1 or "1D"
            freq_other = freq * 2
            n_steps = 21
        else:  # 2 or "2D"
            freq_other = freq / 2
            n_steps = 11
        freq_other = int(freq_other) if isinstance(freq_other, float) else freq_other
        # if freq_other has a higher freq, we expect the slice to have the higher freq
        freq_expected = freq if freq > freq_other else freq_other
        idx = generate_index(start=start, freq=freq, length=n_steps)
        end = idx[-1]

        n_cols = 1 if is_univariate else 2
        series = TimeSeries.from_times_and_values(
            values=np.random.randn(n_steps, n_cols), times=idx
        )

        def check_intersect(other, start_, end_, freq_):
            s_int = series.slice_intersect(other)
            assert s_int.components.equals(series.components)
            assert s_int.freq == freq_

            if start_ is None:  # empty slice
                assert len(s_int) == 0
                return

            assert s_int.start_time() == start_
            assert s_int.end_time() == end_

            s_int_vals = series.slice_intersect_values(other, copy=False)
            np.testing.assert_array_equal(s_int.all_values(), s_int_vals)
            # check that first and last values are as expected
            start_ = series.get_index_at_point(start_)
            end_ = series.get_index_at_point(end_)
            np.testing.assert_array_equal(
                series[start_].all_values(), s_int_vals[0:1, :, :]
            )
            np.testing.assert_array_equal(
                series[end_].all_values(), s_int_vals[-1:, :, :]
            )
            # check that the time index is the same with `slice_intersect_times`
            s_int_idx = series.slice_intersect_times(other, copy=False)
            assert s_int.time_index.equals(s_int_idx)

            assert slice_intersect([series, other]) == [
                series.slice_intersect(other),
                other.slice_intersect(series),
            ]

        # slice with exact range
        startA = start
        endA = end
        idxA = generate_index(startA, endA, freq=freq_other)
        seriesA = TimeSeries.from_series(pd.Series(range(len(idxA)), index=idxA))
        check_intersect(seriesA, startA, endA, freq_expected)

        # entire slice within the range
        startB = start + freq
        endB = startB + 6 * freq_other
        idxB = generate_index(startB, endB, freq=freq_other)
        seriesB = TimeSeries.from_series(pd.Series(range(len(idxB)), index=idxB))
        check_intersect(seriesB, startB, endB, freq_expected)

        # start outside of range
        startC = start - 4 * freq
        endC = start + 4 * freq_other
        idxC = generate_index(startC, endC, freq=freq_other)
        seriesC = TimeSeries.from_series(pd.Series(range(len(idxC)), index=idxC))
        check_intersect(seriesC, start, endC, freq_expected)

        # end outside of range
        startD = start + 4 * freq
        endD = end + 4 * freq_other
        idxD = generate_index(startD, endD, freq=freq_other)
        seriesD = TimeSeries.from_series(pd.Series(range(len(idxD)), index=idxD))
        check_intersect(seriesD, startD, end, freq_expected)

        # small intersect
        startE = start + (n_steps - 1) * freq
        endE = startE + 2 * freq_other
        idxE = generate_index(startE, endE, freq=freq_other)
        seriesE = TimeSeries.from_series(pd.Series(range(len(idxE)), index=idxE))
        check_intersect(seriesE, startE, end, freq_expected)

        # No intersect
        startF = end + 3 * freq
        endF = startF + 6 * freq_other
        idxF = generate_index(startF, endF, freq=freq_other)
        seriesF = TimeSeries.from_series(pd.Series(range(len(idxF)), index=idxF))
        # for empty slices, we expect the original freq
        check_intersect(seriesF, None, None, freq)

        # sequence with zero or one element
        assert slice_intersect([]) == []
        assert slice_intersect([series]) == [series]

        # sequence with more than 2 elements
        intersected_series = slice_intersect([series, seriesA, seriesE])
        s1_int = intersected_series[0]
        s2_int = intersected_series[1]
        s3_int = intersected_series[2]

        assert s1_int.time_index.equals(s2_int.time_index) and s1_int.time_index.equals(
            s3_int.time_index
        )
        assert s1_int.start_time() == startE
        assert s1_int.end_time() == endA

        # check treatment different time index types
        if series.has_datetime_index:
            seriesF = TimeSeries.from_series(
                pd.Series(range(len(idxF)), index=pd.to_numeric(idxF))
            )
        else:
            seriesF = TimeSeries.from_series(
                pd.Series(range(len(idxF)), index=pd.to_datetime(idxF))
            )

        with pytest.raises(IndexError):
            slice_intersect([series, seriesF])

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
            pd.date_range("20130101", "20130601", freq=freqs["ME"]), range(5)
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
        appended = seriesA.append(seriesB)
        assert appended == test_series
        assert appended.freq == test_series.freq
        assert test_series.time_index.equals(appended.time_index)
        assert appended.components.equals(seriesA.components)

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
        appended = seriesA.append_values(arrayB)
        assert appended == test_series
        assert test_series.time_index.equals(appended.time_index)

        # arrayB shape shouldn't affect append_values output:
        squeezed_arrayB = arrayB.squeeze()
        appended_sq = seriesA.append_values(squeezed_arrayB)
        assert appended_sq == test_series
        assert test_series.time_index.equals(appended_sq.time_index)
        assert appended_sq.components.equals(seriesA.components)

    @staticmethod
    def helper_test_prepend(test_case, test_series: TimeSeries):
        # reconstruct series
        seriesA, seriesB = test_series.split_after(pd.Timestamp("20130106"))
        prepended = seriesB.prepend(seriesA)
        assert prepended == test_series
        assert prepended.freq == test_series.freq
        assert test_series.time_index.equals(prepended.time_index)
        assert prepended.components.equals(seriesB.components)

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
        prepended = seriesB.prepend_values(arrayA)
        assert prepended == test_series
        assert test_series.time_index.equals(prepended.time_index)
        assert prepended.components.equals(test_series.components)

        # arrayB shape shouldn't affect append_values output:
        squeezed_arrayA = arrayA.squeeze()
        prepended_sq = seriesB.prepend_values(squeezed_arrayA)
        assert prepended_sq == test_series
        assert test_series.time_index.equals(prepended_sq.time_index)
        assert prepended_sq.components.equals(test_series.components)

        # component and sample dimension should match
        assert prepended._xa.shape[1:] == test_series._xa.shape[1:]

    def test_slice(self):
        TestTimeSeries.helper_test_slice(self, self.series1)

    def test_split(self):
        TestTimeSeries.helper_test_split(self, self.series1)

    def test_drop(self):
        TestTimeSeries.helper_test_drop(self, self.series1)

    @pytest.mark.parametrize(
        "config", itertools.product(["D", "2D", 1, 2], [False, True])
    )
    def test_intersect(self, config):
        """Tests slice intersection between two series with datetime or range index with identical and
        mixed frequencies."""
        freq, mixed_freq = config
        self.helper_test_intersect(freq, mixed_freq, is_univariate=True)

    def test_shift(self):
        TestTimeSeries.helper_test_shift(self, self.series1)

    def test_append(self):
        TestTimeSeries.helper_test_append(self, self.series1)
        # Check `append` deals with `RangeIndex` series correctly:
        series_1 = linear_timeseries(start=1, length=5, freq=2, column_name=freqs["YE"])
        series_2 = linear_timeseries(start=11, length=2, freq=2, column_name="B")
        appended = series_1.append(series_2)
        expected_vals = np.concatenate(
            [series_1.all_values(), series_2.all_values()], axis=0
        )
        expected_idx = pd.RangeIndex(start=1, stop=15, step=2)
        assert np.allclose(appended.all_values(), expected_vals)
        assert appended.time_index.equals(expected_idx)
        assert appended.components.equals(series_1.components)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [
                (  # univariate array
                    np.array([0, 1, 2]).reshape((3, 1, 1)),
                    np.array([0, 1]).reshape((2, 1, 1)),
                ),
                (  # multivariate array
                    np.array([0, 1, 2, 3, 4, 5]).reshape((3, 2, 1)),
                    np.array([0, 1, 2, 3]).reshape((2, 2, 1)),
                ),
                (  # empty array
                    np.array([0, 1, 2]).reshape((3, 1, 1)),
                    np.array([]).reshape((0, 1, 1)),
                ),
                (
                    # wrong number of components
                    np.array([0, 1, 2]).reshape((3, 1, 1)),
                    np.array([0, 1, 2, 3]).reshape((2, 2, 1)),
                ),
                (
                    # wrong number of samples
                    np.array([0, 1, 2]).reshape((3, 1, 1)),
                    np.array([0, 1, 2, 3]).reshape((2, 1, 2)),
                ),
                (  # univariate list with times
                    np.array([0, 1, 2]).reshape((3, 1, 1)),
                    [0, 1],
                ),
                (  # univariate list with times and components
                    np.array([0, 1, 2]).reshape((3, 1, 1)),
                    [[0], [1]],
                ),
                (  # univariate list with times, components and samples
                    np.array([0, 1, 2]).reshape((3, 1, 1)),
                    [[[0]], [[1]]],
                ),
                (  # multivar with list has wrong shape
                    np.array([0, 1, 2, 3]).reshape((2, 2, 1)),
                    [[1, 2], [3, 4]],
                ),
                (  # list with wrong number of components
                    np.array([0, 1, 2]).reshape((3, 1, 1)),
                    [[1, 2], [3, 4]],
                ),
                (  # list with wrong number of samples
                    np.array([0, 1, 2]).reshape((3, 1, 1)),
                    [[[0, 1]], [[1, 2]]],
                ),
                (  # multivar input but list has wrong shape
                    np.array([0, 1, 2, 3]).reshape((2, 2, 1)),
                    [1, 2],
                ),
            ],
            [True, False],
            ["append_values", "prepend_values"],
        ),
    )
    def test_append_and_prepend_values(self, config):
        (series_vals, vals), is_datetime, method = config
        start = "20240101" if is_datetime else 1
        series_idx = generate_index(
            start=start, length=len(series_vals), name="some_name"
        )
        series = TimeSeries.from_times_and_values(
            times=series_idx,
            values=series_vals,
        )

        # expand if it's a list
        vals_arr = np.array(vals) if isinstance(vals, list) else vals
        vals_arr = expand_arr(vals_arr, ndim=3)

        ts_method = getattr(TimeSeries, method)

        if vals_arr.shape[1:] != series_vals.shape[1:]:
            with pytest.raises(ValueError) as exc:
                _ = ts_method(series, vals)
            assert str(exc.value).startswith(
                "The (expanded) values must have the same number of components and samples"
            )
            return

        appended = ts_method(series, vals)

        if method == "append_values":
            expected_vals = np.concatenate([series_vals, vals_arr], axis=0)
            expected_idx = generate_index(
                start=series.start_time(),
                length=len(series_vals) + len(vals),
                freq=series.freq,
            )
        else:
            expected_vals = np.concatenate([vals_arr, series_vals], axis=0)
            expected_idx = generate_index(
                end=series.end_time(),
                length=len(series_vals) + len(vals),
                freq=series.freq,
            )

        assert np.allclose(appended.all_values(), expected_vals)
        assert appended.time_index.equals(expected_idx)
        assert appended.components.equals(series.components)
        assert appended._xa.shape[1:] == series._xa.shape[1:]
        assert appended.time_index.name == series.time_index.name

    def test_prepend(self):
        TestTimeSeries.helper_test_prepend(self, self.series1)
        # Check `prepend` deals with `RangeIndex` series correctly:
        series_1 = linear_timeseries(start=1, length=5, freq=2, column_name=freqs["YE"])
        series_2 = linear_timeseries(start=11, length=2, freq=2, column_name="B")
        prepended = series_2.prepend(series_1)
        expected_vals = np.concatenate(
            [series_1.all_values(), series_2.all_values()], axis=0
        )
        expected_idx = pd.RangeIndex(start=1, stop=15, step=2)
        assert np.allclose(prepended.all_values(), expected_vals)
        assert prepended.time_index.equals(expected_idx)
        assert prepended.components.equals(series_1.components)

    @pytest.mark.parametrize(
        "config",
        [
            ("with_values", True),
            ("with_times_and_values", True),
            ("with_times_and_values", False),
        ],
    )
    def test_with_x_values(self, config):
        """Test `with_values`, and `with_times_and_values`, where the latter can have identical or different times."""
        method, use_entire_index = config
        mask = slice(None) if use_entire_index else slice(1, 4)

        vals = np.random.rand(5, 10, 3)
        series = TimeSeries.from_values(vals)

        vals = vals[mask]
        series[::2]
        kwargs = (
            {"times": series.time_index[mask]}
            if method == "with_times_and_values"
            else dict()
        )
        series2 = getattr(series, method)(values=vals + 1, **kwargs)
        series3 = getattr(series2, method)(values=series2.all_values() - 1, **kwargs)

        # values should work
        np.testing.assert_allclose(series3.all_values(), series[mask].all_values())
        np.testing.assert_allclose(series2.all_values(), vals + 1)

        # should fail if nr components is not the same:
        with pytest.raises(ValueError):
            getattr(series, method)(values=np.random.rand(len(vals), 11, 3), **kwargs)

        # should not fail if nr samples is not the same:
        getattr(series, method)(values=np.random.rand(len(vals), 10, 2), **kwargs)

        # should not fail if nr samples is not the same:
        getattr(series, method)(values=np.random.rand(len(vals), 10, 2), **kwargs)

        # should not fail for univariate deterministic series if values is a 1D array
        getattr(series[series.columns[0]], method)(
            values=np.random.rand(len(vals)), **kwargs
        )

    def test_cumsum(self):
        cumsum_expected = TimeSeries.from_dataframe(
            self.series1.to_dataframe().cumsum()
        )
        # univariate deterministic
        assert self.series1.cumsum() == TimeSeries.from_dataframe(
            self.series1.to_dataframe().cumsum()
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
        diff1 = TimeSeries.from_dataframe(self.series1.to_dataframe().diff())
        diff2 = TimeSeries.from_dataframe(diff1.to_dataframe().diff())
        diff1_no_na = TimeSeries.from_dataframe(diff1.to_dataframe().dropna())
        diff2_no_na = TimeSeries.from_dataframe(diff2.to_dataframe().dropna())

        diff_shift2 = TimeSeries.from_dataframe(
            self.series1.to_dataframe().diff(periods=2)
        )
        diff_shift2_no_na = TimeSeries.from_dataframe(
            self.series1.to_dataframe().diff(periods=2).dropna()
        )

        diff2_shift2 = TimeSeries.from_dataframe(
            diff_shift2.to_dataframe().diff(periods=2)
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

    def test_ops_array(self):
        # can work with xarray directly
        series2_x = self.series2.data_array(copy=False)
        assert self.series1 + self.series2 == self.series1 + series2_x
        assert self.series1 - self.series2 == self.series1 - series2_x
        assert self.series1 * self.series2 == self.series1 * series2_x
        assert self.series1 / self.series2 == self.series1 / series2_x
        assert self.series1**self.series2 == self.series1**series2_x
        # can work with ndarray directly
        series2_nd = self.series2.all_values(copy=False)
        assert self.series1 + self.series2 == self.series1 + series2_nd
        assert self.series1 - self.series2 == self.series1 - series2_nd
        assert self.series1 * self.series2 == self.series1 * series2_nd
        assert self.series1 / self.series2 == self.series1 / series2_nd
        assert self.series1**self.series2 == self.series1**series2_nd

    @pytest.mark.parametrize(
        "broadcast_components,broadcast_samples",
        itertools.product([True, False], [True, False]),
    )
    def test_ops_broadcasting(self, broadcast_components, broadcast_samples):
        # generate random time-series
        t, c, s = 10, 5, 3
        arrayA = np.random.rand(t, c, s)
        arrayB = np.random.rand(
            t, 1 if broadcast_components else c, 1 if broadcast_samples else s
        )

        seriesA = TimeSeries.from_times_and_values(self.times, arrayA)
        seriesB = TimeSeries.from_times_and_values(self.times, arrayB)

        seriesAdd = TimeSeries.from_times_and_values(self.times, arrayA + arrayB)
        seriesSub = TimeSeries.from_times_and_values(self.times, arrayA - arrayB)
        seriesMul = TimeSeries.from_times_and_values(self.times, arrayA * arrayB)
        seriesDiv = TimeSeries.from_times_and_values(self.times, arrayA / arrayB)
        seriesPow = TimeSeries.from_times_and_values(self.times, arrayA**arrayB)

        # assert different operations; must be equivalent to operations with scalar
        assert seriesA + seriesB == seriesAdd
        assert seriesA - seriesB == seriesSub
        assert seriesA * seriesB == seriesMul
        assert seriesA / seriesB == seriesDiv
        assert seriesA**seriesB == seriesPow

        # it also works with numpy arrays directly
        assert seriesA + arrayB == seriesAdd
        assert seriesA - arrayB == seriesSub
        assert seriesA * arrayB == seriesMul
        assert seriesA / arrayB == seriesDiv
        assert seriesA**arrayB == seriesPow

    def test_getitem_datetime_index(self):
        series_short: TimeSeries = self.series1.drop_after(pd.Timestamp("20130105"))
        series_stride_2: TimeSeries = self.series1.with_times_and_values(
            times=self.series1.time_index[::2],
            values=self.series1.all_values()[::2],
        )
        # getitem from slice
        assert self.series1[:] == self.series1[::] == self.series1[::1] == self.series1
        assert self.series1[::2] == series_stride_2
        assert self.series1[::2].freq == self.series1.freq * 2
        assert self.series1[:4] == series_short
        # getitem from dates
        assert self.series1[pd.date_range("20130101", " 20130104")] == series_short
        assert self.series1[pd.Timestamp("20130101")] == TimeSeries.from_dataframe(
            self.series1.to_dataframe()[:1],
            freq=self.series1.freq,
        )
        assert (
            self.series1[pd.Timestamp("20130101") : pd.Timestamp("20130104")]
            == series_short
        )

        # not all dates in index
        with pytest.raises(KeyError):
            self.series1[pd.date_range("19990101", "19990201")]
        # date not in index
        with pytest.raises(KeyError):
            self.series1["19990101"]
        # cannot reverse series
        with pytest.raises(ValueError):
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

        # getitem from slice
        series_stride_2 = self.series1.with_times_and_values(
            times=series.time_index[::2],
            values=series.all_values()[::2],
        )
        assert series[:] == series[::] == series[::1] == series
        assert series[::2] == series_stride_2
        assert series[::2].freq == series.freq * 2

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
        assert series[idx_single].to_series().equals(series.to_series()[1:2])
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

    def test_getitem_frequency_inferrence(self):
        ts = self.series1
        assert ts.freq == "D"
        assert ts[::2].freq == ts[1::2].freq == ts[:-1:2].freq == "2D"
        assert ts[pd.Timestamp("20130103") :: 2].freq == "2D"

        idx = pd.DatetimeIndex(["20130102", "20130105", "20130108"])
        ts_idx = ts[idx]
        assert ts_idx.freq == "3D"

        # With BusinessDay frequency
        offset = pd.offsets.BusinessDay()  # Closed on Saturdays & Sundays
        dates1 = pd.date_range("20231101", "20231126", freq=offset)
        values1 = np.ones(len(dates1))
        ts = TimeSeries.from_times_and_values(dates1, values1)
        assert ts.freq == ts[-4:].freq

        # Using a step parameter
        assert ts[1::3].freq == 3 * ts.freq
        assert ts[pd.Timestamp("20231102") :: 4].freq == 4 * ts.freq

        # Indexing with datetime index
        idx = pd.date_range("20231101", "20231126", freq=offset)
        assert ts[idx].freq == idx.freq

    def test_getitem_frequency_inferrence_integer_index(self):
        start = 2
        freq = 3
        ts = TimeSeries.from_times_and_values(
            times=pd.RangeIndex(
                start=start, stop=start + freq * len(self.series1), step=freq
            ),
            values=self.series1.values(),
        )

        assert ts.freq == freq
        assert ts[::2].freq == ts[1::2].freq == ts[:-1:2].freq == 2 * freq
        assert ts[1::2].start_time() == start + freq

        idx = pd.RangeIndex(
            start=start + 2 * freq, stop=start + 4 * freq, step=2 * freq
        )
        ts_idx = ts[idx]
        assert ts_idx.start_time() == idx[0]
        assert ts_idx.freq == 2 * freq

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
        assert math.isnan(series_test.to_series().get("20130105"))

        # ------ test infer frequency for all offset aliases from ------
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        offset_aliases = [
            "B",
            "C",
            "D",
            "W",
            freqs["ME"],
            freqs["SME"],
            freqs["BME"],
            freqs["CBME"],
            "MS",
            "SMS",
            "BMS",
            "CBMS",
            freqs["QE"],
            freqs["BQE"],
            "QS",
            "BQS",
            freqs["YE"],
            freqs["BYE"],
            freqs["YS"],
            "YS",
            freqs["BYS"],
            "BYS",
            freqs["bh"],
            freqs["h"],
            freqs["min"],
            freqs["s"],
            freqs["ms"],
            freqs["us"],
            freqs["ns"],
        ]
        # fill_missing_dates will find multiple inferred frequencies (i.e. for 'B' it finds {'B', 'D'}) -> good
        offset_aliases_raise = [
            "B",
            "C",
            freqs["SME"],
            freqs["BME"],
            freqs["CBME"],
            "SMS",
            "BMS",
            "CBMS",
            freqs["BQE"],
            freqs["BYE"],
            freqs["BYS"],
            "BYS",
            freqs["bh"],
            "BQS",
        ]
        # frequency cannot be inferred for these types (finds '15D' instead of 'SM')
        offset_not_supported = [freqs["SME"], "SMS"]

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
        # 01/01/2013 -> 10/01/2013, one value per day: 0 1 2 3 … 9
        times = pd.date_range("20130101", "20130110")
        pd_series = pd.Series(range(10), index=times)
        timeseries = TimeSeries.from_series(pd_series)

        # up-sample with pad
        # one value per hour -> same value for the whole day
        resampled_timeseries = timeseries.resample(freqs["h"])
        assert resampled_timeseries.freq_str == freqs["h"]
        # day 1: -> 0
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101020000")] == 0
        # day 2: -> 1
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130102020000")] == 1
        # day 9: -> 8
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130109090000")] == 8

        # down-sample with pad
        # one value per 2 days -> entries for every other days do not exist, value of the first day is kept
        resampled_timeseries = timeseries.resample("2D")
        assert resampled_timeseries.freq_str == "2D"
        # day 1: -> 0
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 0
        # day 2: -> does not exist
        with pytest.raises(KeyError):
            resampled_timeseries.to_series().at[pd.Timestamp("20130102")]
        # day 9: -> 8
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130109")] == 8

        # down-sample with all
        # one value per 2 days -> if all scalar in group are > 0 then 1 else 0
        resampled_timeseries = timeseries.resample("2D", "all")
        # group: [0,1] -> 0
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 0
        # group: [2,3] -> 1
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130103")] == 1

        # down-sample with any
        # one value per 2 days -> if any scalar in group is > 0 then 1 else 0
        resampled_timeseries = timeseries.resample("2D", "any")
        # group: [0,1] -> 1
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 1
        # group: [2,3] -> 1
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130103")] == 1

        # up-sample with asfreq
        # two values per day -> holes are filled with nan
        resampled_timeseries = timeseries.resample("12h", "asfreq")
        # day 1, 0h -> 0
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101000000")] == 0
        # day 1, 12h -> nan
        assert pd.isna(
            resampled_timeseries.to_series().at[pd.Timestamp("20130101120000")]
        )

        # up-sample with backfill
        # two values per day -> holes are filled with next value
        resampled_timeseries = timeseries.resample("12h", "backfill")
        # hole in day 1 -> 1, from day 2
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101120000")] == 1
        # day 2 -> 1
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130102000000")] == 1

        # up-sample with bfill (same as backfill)
        # two values per day -> holes are filled with next value
        resampled_timeseries = timeseries.resample("12h", "bfill")
        # hole in day 1 -> 1, from day 2
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101120000")] == 1
        # day 2 -> 1
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130102000000")] == 1

        # down-sample with count
        # two values per day -> count number of values per group
        resampled_timeseries = timeseries.resample("2D", "count")
        # days 1,2 grouped -> 2
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 2
        # days 3,4 grouped -> 2
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130103")] == 2

        # up-sample with ffill
        # two values per day -> holes are filled with previous value
        resampled_timeseries = timeseries.resample("12h", "ffill")
        # day 1 -> 0
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101000000")] == 0
        # hole in day 1 -> 0, from day 1
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101120000")] == 0

        # down-sample with first
        # one value per 2 days -> keep first value of the group
        resampled_timeseries = timeseries.resample("2D", "first")
        # days 1,2 grouped -> 0, from day 1
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 0
        # days 3,4 grouped -> 2, from day 3
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130103")] == 2

        # up-sample with interpolate
        # two values per day -> holes are filled with linearly interpolated values
        resampled_timeseries = timeseries.resample("12h", "interpolate")
        # day 1, 0h -> 0
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101000000")] == 0
        # between [0,1] -> 0.5
        assert (
            resampled_timeseries.to_series().at[pd.Timestamp("20130101120000")] == 0.5
        )

        # down-sample with last
        # one value per 2 days -> keep last value of the group
        resampled_timeseries = timeseries.resample("2D", "last")
        # days 1,2 grouped -> 1, from day 2
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 1
        # days 3,4 grouped -> 3, from day 4
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130103")] == 3

        # down-sample with max
        # one value per 2 days -> keep the max value of the group
        resampled_timeseries = timeseries.resample("2D", "max")
        # days 1,2 group: [0,1] -> 1
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 1
        # days 3,4 group: [2,3] -> 3
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130103")] == 3

        # down-sample with mean
        # one value per 2 days -> keep the mean of the values of the group
        resampled_timeseries = timeseries.resample("2D", "mean")
        # days 1,2 group: [0,1] -> 0.5
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 0.5
        # days 3,4 group: [2,3] -> 2.5
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130103")] == 2.5

        # down-sample with median
        # one value per 3 days -> keep the median of the values of the group
        resampled_timeseries = timeseries.resample("3D", "median")
        # days 1,2,3 group: [0,1,2] -> 1
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 1
        # days 4,5,6 group: [3,4,5] -> 4
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130104")] == 4

        # down-sample with min
        # one value per 2 days -> keep the min value of the group
        resampled_timeseries = timeseries.resample("2D", "min")
        # days 1,2 group: [0,1] -> 0
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 0
        # days 3,4 group: [2,3] -> 2
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130103")] == 2

        # up-sample with nearest (next is the nearest if equals)
        # two values per day -> holes are filled with nearest value
        resampled_timeseries = timeseries.resample("12h", "nearest")
        # days 1.5 -> 1 from day 2
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101120000")] == 1
        # days 2 -> 1
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130102000000")] == 1

        # down-sample with quantile
        # one value per 2 days -> keep the quantile of the values of the group
        resampled_timeseries = timeseries.resample(
            "2D", "quantile", method_kwargs={"q": 0.05}
        )
        # days 1,2 group: [0,1] -> 0.05
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 0.05
        # days 3,4 group: [2,3] -> 2.05
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130103")] == 2.05

        # down-sample with std
        # one value per 2 days -> keep the std of the values of the group
        resampled_timeseries = timeseries.resample("2D", "std")
        # days 1,2 group: [0,1] -> 0.5
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 0.5
        # days 3,4 group: [2,3] -> 0.5
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130103")] == 0.5

        # down-sample with sum using reduce
        # one value per 2 days -> keep the sum of the values of the group
        resampled_timeseries = timeseries.resample(
            "2D", "reduce", method_kwargs={"func": np.sum}
        )
        # days 1,2 group: [0,1] -> 1
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 1
        # days 3,4 group: [2,3] -> 5
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130103")] == 5

        # down-sample with sum
        # one value per 2 days -> keep the sum of the values of the group
        resampled_timeseries = timeseries.resample("2D", "sum")
        # days 1,2 group: [0,1] -> 1
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 1
        # days 3,4 group: [2,3] -> 5
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130103")] == 5

        # down-sample with var
        # one value per 2 days -> keep the sum of the values of the group
        resampled_timeseries = timeseries.resample("2D", "var")
        # days 1,2 group: [0,1] -> 0.25
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130101")] == 0.25
        # days 3,4 group: [2,4] -> 0.25
        assert resampled_timeseries.to_series().at[pd.Timestamp("20130103")] == 0.25

        # unsupported method: apply
        with pytest.raises(ValueError):
            _ = timeseries.resample("2D", "apply")

        # using offset to avoid nan in the first value
        times = pd.date_range(
            start=pd.Timestamp("20200101233000"), periods=10, freq="15" + freqs["min"]
        )
        pd_series = pd.Series(range(10), index=times)
        timeseries = TimeSeries.from_series(pd_series)
        resampled_timeseries = timeseries.resample(
            freq="1" + freqs["h"], offset=pd.Timedelta("30" + freqs["min"])
        )
        assert resampled_timeseries.to_series().at[pd.Timestamp("20200101233000")] == 0

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
            freq=freqs["ME"],
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

        df_0 = series.to_dataframe()
        df_2 = series.to_dataframe()
        df_01 = series.to_dataframe()
        df_012 = series.to_dataframe()

        PANDAS_210 = pd.__version__ >= "2.1.0"
        select_map = "map"
        if not PANDAS_210:
            select_map = "applymap"

        df_0[["0"]] = getattr(df_0[["0"]], select_map)(fn)
        df_2[["2"]] = getattr(df_2[["2"]], select_map)(fn)
        df_01[["0", "1"]] = getattr(df_01[["0", "1"]], select_map)(fn)
        df_012 = getattr(df_012, select_map)(fn)

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
        times2 = pd.date_range("20120101", "20210301", freq=freqs["QE"])
        times3 = pd.date_range("20120101", "20210301", freq=freqs["YS"])
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
            == pd.DatetimeIndex([
                pd.Timestamp("20130901"),
                pd.Timestamp("20160101"),
                pd.Timestamp("20191101"),
            ])
        ).all()
        assert (
            gaps6["gap_end"]
            == pd.DatetimeIndex([
                pd.Timestamp("20130901"),
                pd.Timestamp("20160901"),
                pd.Timestamp("20210301"),
            ])
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
        with pytest.raises(ValueError):
            ts.to_csv("blah.csv")

    @patch("darts.timeseries.TimeSeries.to_dataframe")
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

    @patch("darts.timeseries.TimeSeries.to_dataframe")
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

        with pytest.raises(ValueError):
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
    def pd_to_backend(self, df, backend, index=False):
        if backend == "pandas":
            return df
        elif backend == "polars":
            if index:
                return pl.from_pandas(df.reset_index())
            return pl.from_pandas(df)

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_from_dataframe_sunny_day(self, backend):
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

        data_darts1 = TimeSeries.from_dataframe(
            df=self.pd_to_backend(data_pd1, backend), time_col="Time"
        )
        data_darts2 = TimeSeries.from_dataframe(
            df=self.pd_to_backend(data_pd2, backend), time_col="Time"
        )
        data_darts3 = TimeSeries.from_dataframe(
            df=self.pd_to_backend(data_pd3, backend, index=True),
            time_col=None if backend == "pandas" else "Time",
        )

        assert data_darts1 == data_darts2
        assert data_darts1 == data_darts3

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_time_col_convert_string_integers(self, backend):
        expected = np.array(list(range(3, 10)))
        data_dict = {"Time": expected.astype(str)}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        ts = TimeSeries.from_dataframe(
            df=self.pd_to_backend(df, backend), time_col="Time"
        )

        assert set(ts.time_index.values.tolist()) == set(expected)
        assert ts.time_index.dtype == int
        assert ts.time_index.name == "Time"

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_time_col_convert_integers(self, backend):
        expected = np.array(list(range(10)))
        data_dict = {"Time": expected}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )

        df = pd.DataFrame(data_dict)
        ts = TimeSeries.from_dataframe(
            df=self.pd_to_backend(df, backend), time_col="Time"
        )

        assert set(ts.time_index.values.tolist()) == set(expected)
        assert ts.time_index.dtype == int
        assert ts.time_index.name == "Time"

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_fail_with_bad_integer_time_col(self, backend):
        bad_time_col_vals = np.array([4, 0, 1, 2])
        data_dict = {"Time": bad_time_col_vals}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        with pytest.raises(ValueError):
            TimeSeries.from_dataframe(
                df=self.pd_to_backend(df, backend), time_col="Time"
            )

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_time_col_convert_rangeindex(self, backend):
        for expected_l, step in zip([[4, 0, 2, 3, 1], [8, 0, 4, 6, 2]], [1, 2]):
            expected = np.array(expected_l)
            data_dict = {"Time": expected}
            data_dict["Values1"] = np.random.uniform(
                low=-10, high=10, size=len(data_dict["Time"])
            )
            df = pd.DataFrame(data_dict)
            ts = TimeSeries.from_dataframe(
                df=self.pd_to_backend(df, backend), time_col="Time"
            )

            # check type (should convert to RangeIndex):
            assert type(ts.time_index) is pd.RangeIndex

            # check values inside the index (should be sorted correctly):
            assert list(ts.time_index) == sorted(expected)

            # check that values are sorted accordingly:
            ar1 = ts.values(copy=False)[:, 0]
            ar2 = data_dict["Values1"][
                list(expected_l.index(i * step) for i in range(len(expected)))
            ]
            assert np.all(ar1 == ar2)

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_time_col_convert_datetime(self, backend):
        expected = pd.date_range(start="20180501", end="20200301", freq="MS")
        data_dict = {"Time": expected}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        ts = TimeSeries.from_dataframe(
            df=self.pd_to_backend(df, backend), time_col="Time"
        )

        assert ts.time_index.dtype == "datetime64[ns]"
        assert ts.time_index.name == "Time"

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_time_col_convert_datetime_strings(self, backend):
        expected = pd.date_range(start="20180501", end="20200301", freq="MS")
        data_dict = {"Time": expected.values.astype(str)}
        data_dict["Values1"] = np.random.uniform(
            low=-10, high=10, size=len(data_dict["Time"])
        )
        df = pd.DataFrame(data_dict)
        ts = TimeSeries.from_dataframe(
            df=self.pd_to_backend(df, backend), time_col="Time"
        )

        assert ts.time_index.dtype == "datetime64[ns]"
        assert ts.time_index.name == "Time"

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_time_col_with_tz_df(self, backend):
        # numpy and xarray don't support "timezone aware" pd.DatetimeIndex
        # the BUGFIX removes timezone information without conversion

        time_range_MS = pd.date_range(
            start="20180501", end="20200301", freq="MS", tz="CET"
        )
        values = np.random.uniform(low=-10, high=10, size=len(time_range_MS))
        # pd.DataFrame loses the tz information unless it is contained in its index
        # (other columns are silently converted to UTC, with tz attribute set to None)
        df = pd.DataFrame(data=values, index=time_range_MS)
        ts = TimeSeries.from_dataframe(
            df=self.pd_to_backend(df, backend, index=True),
            time_col=None if backend == "pandas" else "index",
        )
        assert list(ts.time_index) == list(time_range_MS.tz_localize(None))
        assert list(ts.time_index.tz_localize("CET")) == list(time_range_MS)
        assert ts.time_index.tz is None

        ts = TimeSeries.from_times_and_values(times=time_range_MS, values=values)
        assert list(ts.time_index) == list(time_range_MS.tz_localize(None))
        assert list(ts.time_index.tz_localize("CET")) == list(time_range_MS)
        assert ts.time_index.tz is None

        time_range_H = pd.date_range(
            start="20200518", end="20200521", freq=freqs["h"], tz="CET"
        )
        values = np.random.uniform(low=-10, high=10, size=len(time_range_H))

        df = pd.DataFrame(data=values, index=time_range_H)
        ts = TimeSeries.from_dataframe(
            df=self.pd_to_backend(df, backend, index=True),
            time_col=None if backend == "pandas" else "index",
        )
        assert list(ts.time_index) == list(time_range_H.tz_localize(None))
        assert list(ts.time_index.tz_localize("CET")) == list(time_range_H)
        assert ts.time_index.tz is None

        ts = TimeSeries.from_times_and_values(times=time_range_H, values=values)
        assert list(ts.time_index) == list(time_range_H.tz_localize(None))
        assert list(ts.time_index.tz_localize("CET")) == list(time_range_H)
        assert ts.time_index.tz is None

    def test_time_col_with_tz_series(self):
        time_range_MS = pd.date_range(
            start="20180501", end="20200301", freq="MS", tz="CET"
        )
        values = np.random.uniform(low=-10, high=10, size=len(time_range_MS))
        serie = pd.Series(data=values, index=time_range_MS)
        ts = TimeSeries.from_series(pd_series=serie)
        assert list(ts.time_index) == list(time_range_MS.tz_localize(None))
        assert list(ts.time_index.tz_localize("CET")) == list(time_range_MS)
        assert ts.time_index.tz is None

        time_range_H = pd.date_range(
            start="20200518", end="20200521", freq=freqs["h"], tz="CET"
        )
        values = np.random.uniform(low=-10, high=10, size=len(time_range_H))
        series = pd.Series(data=values, index=time_range_H)
        ts = TimeSeries.from_series(pd_series=series)
        assert list(ts.time_index) == list(time_range_H.tz_localize(None))
        assert list(ts.time_index.tz_localize("CET")) == list(time_range_H)
        assert ts.time_index.tz is None

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_time_col_convert_garbage(self, backend):
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
            TimeSeries.from_dataframe(
                df=self.pd_to_backend(df, backend), time_col="Time"
            )

    @pytest.mark.parametrize("backend", TEST_BACKENDS)
    def test_df_named_columns_index(self, backend):
        time_index = generate_index(
            start=pd.Timestamp("2000-01-01"), length=4, freq="D", name="index"
        )
        df = pd.DataFrame(
            data=np.arange(4),
            index=time_index,
            columns=["y"],
        )
        df.columns.name = "id"
        ts = TimeSeries.from_dataframe(
            df=self.pd_to_backend(df, backend, index=True),
            time_col=None if backend == "pandas" else "index",
        )

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
