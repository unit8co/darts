import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.tests.base_test_class import DartsBaseTestClass
from darts.tests.test_timeseries import TimeSeriesTestCase


class TimeSeriesMultivariateTestCase(DartsBaseTestClass):

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
        self.assertTrue(
            np.all(series_test.pd_dataframe().values == (self.dataframe1.values))
        )

        # Series cannot be lower than three without passing frequency as argument to constructor
        with self.assertRaises(ValueError):
            TimeSeries(self.dataframe1.iloc[:2, :])
        TimeSeries.from_dataframe(self.dataframe1.iloc[:2, :], freq="D")

    def test_eq(self):
        seriesA = TimeSeries.from_dataframe(self.dataframe1)
        self.assertTrue(self.series1 == seriesA)
        self.assertFalse(self.series1 != seriesA)

        # with different dates
        dataframeB = self.dataframe1.copy()
        dataframeB.index = pd.date_range("20130102", "20130111")
        seriesB = TimeSeries.from_dataframe(dataframeB)
        self.assertFalse(self.series1 == seriesB)

        # with one different value
        dataframeC = self.dataframe1.copy()
        dataframeC.iloc[2, 2] = 0
        seriesC = TimeSeries.from_dataframe(dataframeC)
        self.assertFalse(self.series1 == seriesC)

    def test_rescale(self):
        with self.assertRaises(ValueError):
            self.series1.rescale_with_value(1)

        seriesA = self.series2.rescale_with_value(0)
        self.assertTrue(np.all(seriesA.values() == 0).all())

        seriesB = self.series2.rescale_with_value(1)
        self.assertEqual(
            seriesB,
            TimeSeries.from_dataframe(
                pd.DataFrame(
                    {
                        "0": np.arange(1, 11),
                        "1": np.arange(1, 11),
                        "2": np.arange(1, 11),
                    },
                    index=self.dataframe2.index,
                ).astype(float)
            ),
        )

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

    def test_prepend(self):
        TimeSeriesTestCase.helper_test_prepend(self, self.series1)

    def test_prepend_values(self):
        TimeSeriesTestCase.helper_test_prepend_values(self, self.series1)

    def test_strip(self):
        dataframe1 = pd.DataFrame(
            {
                "0": 2 * [np.nan] + list(range(7)) + [np.nan],
                "1": [np.nan] + list(range(7)) + 2 * [np.nan],
            },
            index=self.times1,
        )
        series1 = TimeSeries.from_dataframe(dataframe1)

        self.assertTrue((series1.strip().time_index == self.times1[1:-1]).all())

    """
    Testing new multivariate methods.
    """

    def test_stack(self):
        with self.assertRaises(ValueError):
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
        self.assertTrue((seriesA.pd_dataframe() == dataframeA).all().all())
        self.assertEqual(
            seriesA.values().shape,
            (
                len(self.dataframe1),
                len(self.dataframe1.columns) + len(self.dataframe2.columns),
            ),
        )

    def test_univariate_component(self):
        with self.assertRaises(IndexError):
            self.series1.univariate_component(-5)
        with self.assertRaises(IndexError):
            self.series1.univariate_component(3)
        seriesA = self.series1.univariate_component(1)
        self.assertTrue(
            seriesA
            == TimeSeries.from_times_and_values(
                self.times1, range(5, 15), columns=["1"]
            )
        )
        seriesB = (
            self.series1.univariate_component(0)
            .stack(seriesA)
            .stack(self.series1.univariate_component(2))
        )
        self.assertTrue(self.series1 == seriesB)

    def test_add_datetime_attribute(self):
        seriesA = self.series1.add_datetime_attribute("day")
        self.assertEqual(seriesA.width, self.series1.width + 1)
        self.assertTrue(
            set(seriesA.pd_dataframe().iloc[:, seriesA.width - 1].values.flatten())
            == set(range(1, 11))
        )
        seriesB = self.series3.add_datetime_attribute("day", True)
        self.assertEqual(seriesB.width, self.series3.width + 31)
        self.assertEqual(
            set(seriesB.pd_dataframe().iloc[:, self.series3.width :].values.flatten()),
            {0, 1},
        )
        seriesC = self.series1.add_datetime_attribute("month", True)
        self.assertEqual(seriesC.width, self.series1.width + 12)
        seriesD = TimeSeries.from_times_and_values(
            pd.date_range("20130206", "20130430"), range(84)
        )
        seriesD = seriesD.add_datetime_attribute("month", True)
        self.assertEqual(seriesD.width, 13)
        self.assertEqual(sum(seriesD.values().flatten()), sum(range(84)) + 84)
        self.assertEqual(sum(seriesD.values()[:, 1 + 3]), 30)
        self.assertEqual(sum(seriesD.values()[:, 1 + 1]), 23)

        # test cyclic
        times_month = pd.date_range("20130101", "20140610")

        seriesE = TimeSeries.from_times_and_values(
            times_month, np.repeat(0.1, len(times_month))
        )
        seriesF = seriesE.add_datetime_attribute("day", cyclic=True)

        values_sin = seriesF.values()[:, 1]
        values_cos = seriesF.values()[:, 2]

        self.assertTrue(
            np.allclose(np.add(np.square(values_sin), np.square(values_cos)), 1)
        )

        df = seriesF.pd_dataframe()
        df = df[df.index.day == 1]
        self.assertTrue(np.allclose(df["day_sin"].values, 0.2, atol=0.03))
        self.assertTrue(np.allclose(df["day_cos"].values, 0.97, atol=0.03))

    def test_add_holidays(self):
        times = pd.date_range(start=pd.Timestamp("20201201"), periods=30, freq="D")
        seriesA = TimeSeries.from_times_and_values(times, range(len(times)))

        # testing for christmas and non-holiday in US
        seriesA = seriesA.add_holidays("US")
        last_column = seriesA.pd_dataframe().iloc[:, seriesA.width - 1]
        self.assertEqual(last_column.at[pd.Timestamp("20201225")], 1)
        self.assertEqual(last_column.at[pd.Timestamp("20201210")], 0)
        self.assertEqual(last_column.at[pd.Timestamp("20201226")], 0)

        # testing for christmas and non-holiday in PL
        seriesA = seriesA.add_holidays("PL")
        last_column = seriesA.pd_dataframe().iloc[:, seriesA.width - 1]
        self.assertEqual(last_column.at[pd.Timestamp("20201225")], 1)
        self.assertEqual(last_column.at[pd.Timestamp("20201210")], 0)
        self.assertEqual(last_column.at[pd.Timestamp("20201226")], 1)
        self.assertEqual(seriesA.width, 3)

        # testing hourly time series
        times = pd.date_range(start=pd.Timestamp("20201224"), periods=50, freq="H")
        seriesB = TimeSeries.from_times_and_values(times, range(len(times)))
        seriesB = seriesB.add_holidays("US")
        last_column = seriesB.pd_dataframe().iloc[:, seriesB.width - 1]
        self.assertEqual(last_column.at[pd.Timestamp("2020-12-25 01:00:00")], 1)
        self.assertEqual(last_column.at[pd.Timestamp("2020-12-24 23:00:00")], 0)

    def test_assert_univariate(self):
        with self.assertRaises(AssertionError):
            self.series1._assert_univariate()
        self.series1.univariate_component(0)._assert_univariate()

    def test_first_last_values(self):
        self.assertEqual(self.series1.first_values().tolist(), [0, 5, 10])
        self.assertEqual(self.series3.last_values().tolist(), [10, 20])
        self.assertEqual(
            self.series1.univariate_component(1).first_values().tolist(), [5]
        )
        self.assertEqual(
            self.series3.univariate_component(1).last_values().tolist(), [20]
        )

    def test_drop_column(self):
        # testing dropping a single column
        seriesA = self.series1.drop_columns("0")
        self.assertNotIn("0", seriesA.columns.values)
        self.assertEqual(seriesA.columns.tolist(), ["1", "2"])
        self.assertEqual(len(seriesA.columns), 2)

        # testing dropping multiple columns
        seriesB = self.series1.drop_columns(["0", "1"])
        self.assertIn("2", seriesB.columns.values)
        self.assertEqual(len(seriesB.columns), 1)

    def test_gaps(self):
        gaps1_all = self.series1.gaps(mode="all")
        self.assertTrue(gaps1_all.empty)
        gaps1_any = self.series1.gaps(mode="any")
        self.assertTrue(gaps1_any.empty)

        gaps4_all = self.series4.gaps(mode="all")
        self.assertTrue(
            (
                gaps4_all["gap_start"] == pd.DatetimeIndex([pd.Timestamp("20130208")])
            ).all()
        )
        self.assertTrue(
            (gaps4_all["gap_end"] == pd.DatetimeIndex([pd.Timestamp("20130208")])).all()
        )
        self.assertEqual(gaps4_all["gap_size"].values.tolist(), [1])

        gaps4_any = self.series4.gaps(mode="any")
        self.assertTrue(
            (
                gaps4_any["gap_start"]
                == pd.DatetimeIndex(
                    [pd.Timestamp("20130208"), pd.Timestamp("20130211")]
                )
            ).all()
        )
        self.assertTrue(
            (
                gaps4_any["gap_end"]
                == pd.DatetimeIndex(
                    [pd.Timestamp("20130208"), pd.Timestamp("20130214")]
                )
            ).all()
        )
        self.assertEqual(gaps4_any["gap_size"].values.tolist(), [1, 4])
