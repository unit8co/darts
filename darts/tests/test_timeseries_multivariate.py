import itertools

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.tests.conftest import POLARS_AVAILABLE
from darts.tests.test_timeseries import TestTimeSeries
from darts.utils.utils import freqs

if POLARS_AVAILABLE:
    import polars as pl
else:
    pl = None


class TestTimeSeriesMultivariate:
    times1 = pd.date_range("20130101", "20130110")
    times2 = pd.date_range("20130206", "20130215")
    dataframe1 = pd.DataFrame(
        {"0": range(10), "1": range(5, 15), "2": range(10, 20)}, index=times1
    )
    dataframe2 = pd.DataFrame(
        {"0": np.arange(1, 11), "1": np.arange(1, 11) * 3, "2": np.arange(1, 11) * 5},
        index=times1,
    )
    dataframe3 = pd.DataFrame(
        {
            "0": np.arange(1, 11),
            "1": np.arange(11, 21),
        },
        index=times2,
    )
    dataframe4 = pd.DataFrame(
        {
            "0": [1, 1, np.nan, 1, 1, 1, 1, 1, 1, 1],
            "1": [1, 1, np.nan, 1, 1, np.nan, np.nan, 1, 1, 1],
            "2": [1, 1, np.nan, 1, 1, np.nan, np.nan, np.nan, np.nan, 1],
        },
        index=times2,
    )
    series1 = TimeSeries.from_dataframe(dataframe1)
    series2 = TimeSeries.from_dataframe(dataframe2)
    series3 = TimeSeries.from_dataframe(dataframe3)
    series4 = TimeSeries.from_dataframe(dataframe4)

    def test_creation(self):
        series_test = TimeSeries.from_dataframe(self.dataframe1)
        assert np.all(series_test.to_dataframe().values == self.dataframe1.values)

        # Series cannot be lower than three without passing frequency as argument to constructor
        with pytest.raises(ValueError):
            TimeSeries(self.dataframe1.iloc[:2, :])
        TimeSeries.from_dataframe(self.dataframe1.iloc[:2, :], freq="D")

    @pytest.mark.skipif(not POLARS_AVAILABLE, reason="requires polars")
    def test_polars_creation(self):
        pl_df = pl.DataFrame(
            data={
                "time": pd.date_range(start="2023-01-01", periods=10, freq="D"),
                "test_float": [float(i) for i in range(10)],
                "test_int": range(10),
            }
        )
        # with a `time_col` no warning is raised
        ts = TimeSeries.from_dataframe(pl_df, time_col="time")
        ts_pl_df = ts.to_dataframe(backend="polars", time_as_index=False)
        assert ts_pl_df.equals(pl_df)

        # darts converts everything to float (test_int)
        assert ts_pl_df.dtypes != pl_df.dtypes
        dtypes_expected = pl_df.dtypes[:2] + [pl_df.dtypes[1]]
        assert ts_pl_df.dtypes == dtypes_expected

    def test_eq(self):
        seriesA = TimeSeries.from_dataframe(self.dataframe1)
        assert self.series1 == seriesA
        assert not (self.series1 != seriesA)

        # with different dates
        dataframeB = self.dataframe1.copy()
        dataframeB.index = pd.date_range("20130102", "20130111")
        seriesB = TimeSeries.from_dataframe(dataframeB)
        assert not (self.series1 == seriesB)

        # with one different value
        dataframeC = self.dataframe1.copy()
        dataframeC.iloc[2, 2] = 0
        seriesC = TimeSeries.from_dataframe(dataframeC)
        assert not (self.series1 == seriesC)

    def test_rescale(self):
        with pytest.raises(ValueError):
            self.series1.rescale_with_value(1)

        seriesA = self.series2.rescale_with_value(0)
        assert np.all(seriesA.values() == 0).all()

        seriesB = self.series2.rescale_with_value(1)
        assert seriesB == TimeSeries.from_dataframe(
            pd.DataFrame(
                {
                    "0": np.arange(1, 11),
                    "1": np.arange(1, 11),
                    "2": np.arange(1, 11),
                },
                index=self.dataframe2.index,
            ).astype(float)
        )

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
        freq, mixed_freq = config
        TestTimeSeries.helper_test_intersect(freq, mixed_freq, is_univariate=False)

    def test_shift(self):
        TestTimeSeries.helper_test_shift(self, self.series1)

    def test_append(self):
        TestTimeSeries.helper_test_append(self, self.series1)

    def test_append_values(self):
        TestTimeSeries.helper_test_append_values(self, self.series1)

    def test_prepend(self):
        TestTimeSeries.helper_test_prepend(self, self.series1)

    def test_prepend_values(self):
        TestTimeSeries.helper_test_prepend_values(self, self.series1)

    def test_strip(self):
        dataframe1 = pd.DataFrame(
            {
                "0": 2 * [np.nan] + list(range(7)) + [np.nan],
                "1": [np.nan] + list(range(7)) + 2 * [np.nan],
            },
            index=self.times1,
        )
        series1 = TimeSeries.from_dataframe(dataframe1)

        assert (series1.strip().time_index == self.times1[1:-1]).all()
        assert (series1.strip(how="any").time_index == self.times1[2:-2]).all()

    """
    Testing new multivariate methods.
    """

    def test_stack(self):
        with pytest.raises(ValueError):
            self.series1.stack(self.series3)
        seriesA = self.series1.stack(self.series2)
        dataframeA = pd.concat([self.dataframe1, self.dataframe2], axis=1)
        dataframeA.columns = [
            "0",
            "1",
            "2",
            "0_1",
            "1_1",
            "2_1",
        ]  # the names to expect after stacking
        assert (seriesA.to_dataframe() == dataframeA).all().all()
        assert seriesA.values().shape == (
            len(self.dataframe1),
            len(self.dataframe1.columns) + len(self.dataframe2.columns),
        )

    def test_univariate_component(self):
        with pytest.raises(IndexError):
            self.series1.univariate_component(-5)
        with pytest.raises(IndexError):
            self.series1.univariate_component(3)
        seriesA = self.series1.univariate_component(1)
        assert seriesA == TimeSeries.from_times_and_values(
            self.times1, range(5, 15), columns=["1"]
        )
        seriesB = (
            self.series1.univariate_component(0)
            .stack(seriesA)
            .stack(self.series1.univariate_component(2))
        )
        assert self.series1 == seriesB

    def test_add_datetime_attribute(self):
        """datetime_attributes are 0-indexed (shift is applied when necessary)"""
        seriesA = self.series1.add_datetime_attribute("day")
        assert seriesA.width == self.series1.width + 1
        assert set(
            seriesA.to_dataframe().iloc[:, seriesA.width - 1].values.flatten()
        ) == set(range(0, 10))
        seriesB = self.series3.add_datetime_attribute("day", True)
        assert seriesB.width == self.series3.width + 31
        assert set(
            seriesB.to_dataframe().iloc[:, self.series3.width :].values.flatten()
        ) == {0, 1}
        seriesC = self.series1.add_datetime_attribute("month", True)
        assert seriesC.width == self.series1.width + 12
        seriesD = TimeSeries.from_times_and_values(
            pd.date_range("20130206", "20130430"), range(84)
        )
        seriesD = seriesD.add_datetime_attribute("month", True)
        assert seriesD.width == 13
        assert sum(seriesD.values().flatten()) == sum(range(84)) + 84
        assert sum(seriesD.values()[:, 1 + 3]) == 30
        assert sum(seriesD.values()[:, 1 + 1]) == 23

        # test cyclic
        times_month = pd.date_range("20130101", "20140610")

        seriesE = TimeSeries.from_times_and_values(
            times_month, np.repeat(0.1, len(times_month))
        )
        seriesF = seriesE.add_datetime_attribute("day", cyclic=True)

        values_sin = seriesF.values()[:, 1]
        values_cos = seriesF.values()[:, 2]

        assert np.allclose(np.add(np.square(values_sin), np.square(values_cos)), 1)

        df = seriesF.to_dataframe()
        # first day is equivalent to t=0
        df = df[df.index.day == 1]
        assert np.allclose(df["day_sin"].values, 0, atol=0.03)
        assert np.allclose(df["day_cos"].values, 1, atol=0.03)

        # second day is equivalent to t=1
        df = df[df.index.day == 2]
        assert np.allclose(df["day_sin"].values, 0.2, atol=0.03)
        assert np.allclose(df["day_cos"].values, 0.97, atol=0.03)

    def test_add_holidays(self):
        times = pd.date_range(start=pd.Timestamp("20201201"), periods=30, freq="D")
        seriesA = TimeSeries.from_times_and_values(times, range(len(times)))

        # testing for christmas and non-holiday in US
        seriesA = seriesA.add_holidays("US")
        last_column = seriesA.to_dataframe().iloc[:, seriesA.width - 1]
        assert last_column.at[pd.Timestamp("20201225")] == 1
        assert last_column.at[pd.Timestamp("20201210")] == 0
        assert last_column.at[pd.Timestamp("20201226")] == 0

        # testing for christmas and non-holiday in PL
        seriesA = seriesA.add_holidays("PL")
        last_column = seriesA.to_dataframe().iloc[:, seriesA.width - 1]
        assert last_column.at[pd.Timestamp("20201225")] == 1
        assert last_column.at[pd.Timestamp("20201210")] == 0
        assert last_column.at[pd.Timestamp("20201226")] == 1
        assert seriesA.width == 3

        # testing hourly time series
        times = pd.date_range(
            start=pd.Timestamp("20201224"), periods=50, freq=freqs["h"]
        )
        seriesB = TimeSeries.from_times_and_values(times, range(len(times)))
        seriesB = seriesB.add_holidays("US")
        last_column = seriesB.to_dataframe().iloc[:, seriesB.width - 1]
        assert last_column.at[pd.Timestamp("2020-12-25 01:00:00")] == 1
        assert last_column.at[pd.Timestamp("2020-12-24 23:00:00")] == 0

    def test_assert_univariate(self):
        with pytest.raises(AssertionError):
            self.series1._assert_univariate()
        self.series1.univariate_component(0)._assert_univariate()

    def test_first_last_values(self):
        assert self.series1.first_values().tolist() == [0, 5, 10]
        assert self.series3.last_values().tolist() == [10, 20]
        assert self.series1.univariate_component(1).first_values().tolist() == [5]
        assert self.series3.univariate_component(1).last_values().tolist() == [20]

    def test_drop_column(self):
        # testing dropping a single column
        seriesA = self.series1.drop_columns("0")
        assert "0" not in seriesA.columns.values
        assert seriesA.columns.tolist() == ["1", "2"]
        assert len(seriesA.columns) == 2

        # testing dropping multiple columns
        seriesB = self.series1.drop_columns(["0", "1"])
        assert "2" in seriesB.columns.values
        assert len(seriesB.columns) == 1

    def test_gaps(self):
        gaps1_all = self.series1.gaps(mode="all")
        assert gaps1_all.empty
        gaps1_any = self.series1.gaps(mode="any")
        assert gaps1_any.empty

        gaps4_all = self.series4.gaps(mode="all")
        assert (
            gaps4_all["gap_start"] == pd.DatetimeIndex([pd.Timestamp("20130208")])
        ).all()
        assert (
            gaps4_all["gap_end"] == pd.DatetimeIndex([pd.Timestamp("20130208")])
        ).all()
        assert gaps4_all["gap_size"].values.tolist() == [1]

        gaps4_any = self.series4.gaps(mode="any")
        assert (
            gaps4_any["gap_start"]
            == pd.DatetimeIndex([pd.Timestamp("20130208"), pd.Timestamp("20130211")])
        ).all()
        assert (
            gaps4_any["gap_end"]
            == pd.DatetimeIndex([pd.Timestamp("20130208"), pd.Timestamp("20130214")])
        ).all()
        assert gaps4_any["gap_size"].values.tolist() == [1, 4]
