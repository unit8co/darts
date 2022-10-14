import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import _with_sanity_checks, retain_period_common_to_all
from darts.utils.missing_values import extract_subseries


class UtilsTestCase(DartsBaseTestClass):
    def test_retain_period_common_to_all(self):
        seriesA = TimeSeries.from_times_and_values(
            pd.date_range("20000101", "20000110"), range(10)
        )
        seriesB = TimeSeries.from_times_and_values(
            pd.date_range("20000103", "20000108"), range(6)
        )
        seriesC = TimeSeries.from_times_and_values(
            pd.date_range("20000104", "20000112"), range(9)
        )
        seriesC = seriesC.stack(seriesC)

        common_series_list = retain_period_common_to_all([seriesA, seriesB, seriesC])

        # test start and end dates
        for common_series in common_series_list:
            self.assertEqual(common_series.start_time(), pd.Timestamp("20000104"))
            self.assertEqual(common_series.end_time(), pd.Timestamp("20000108"))

        # test widths
        self.assertEqual(common_series_list[0].width, 1)
        self.assertEqual(common_series_list[1].width, 1)
        self.assertEqual(common_series_list[2].width, 2)

    def test_sanity_check_example(self):
        class Model:
            def _sanity_check(self, *args, **kwargs):
                if kwargs["b"] != kwargs["c"]:
                    raise (ValueError("b and c must be equal"))

            @_with_sanity_checks("_sanity_check")
            def fit(self, a, b=0, c=0):
                pass

        m = Model()

        # b != c should raise error
        with self.assertRaises(ValueError):
            m.fit(5, b=3, c=2)

        # b == c should not raise error
        m.fit(5, b=2, c=2)

    def test_extract_subseries(self):
        start_times = ["2020-01-01", "2020-06-01", "2020-09-01"]
        end_times = ["2020-01-31", "2020-07-31", "2020-09-28"]

        # Form a series without missing values between start_times and end_times
        time_index = pd.date_range(periods=365, freq="D", start=start_times[0])
        pd_series = pd.Series(np.nan, index=time_index)
        for start, end in zip(start_times, end_times):
            pd_series[start:end] = 42
        series = TimeSeries.from_series(pd_series)

        subseries = extract_subseries(series)

        self.assertEqual(len(subseries), len(start_times))
        for sub, start, end in zip(subseries, start_times, end_times):
            self.assertEqual(sub.start_time(), pd.to_datetime(start))
            self.assertEqual(sub.end_time(), pd.to_datetime(end))

        # Multivariate timeserie
        times = pd.date_range("20130206", "20130215")
        dataframe = pd.DataFrame(
            {
                "0": [1, 1, np.nan, 1, 2, 1, 1, 1, 1, 1],
                "1": [1, 1, np.nan, 1, 3, np.nan, np.nan, 1, 1, 1],
                "2": [1, 1, np.nan, 1, 4, np.nan, np.nan, np.nan, np.nan, 1],
            },
            index=times,
        )
        series = TimeSeries.from_dataframe(dataframe)

        # gaps is characterized by NaN in all the covariate columns
        subseries_all = extract_subseries(series, mode="all")
        self.assertEqual(len(subseries_all), 2)
        self.assertEqual(subseries_all[0], series[:2])
        self.assertEqual(subseries_all[1], series[3:])

        # gaps is characterized by NaN in any of the covariate columns
        subseries_any = extract_subseries(series, mode="any")
        self.assertEqual(len(subseries_any), 3)
        self.assertEqual(subseries_any[0], series[:2])
        self.assertEqual(subseries_any[1], series[3:5])
        self.assertEqual(subseries_any[2], series[-1])
