import unittest

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import MIDAS


class MIDASTestCase(unittest.TestCase):
    monthly_values = np.arange(1, 10)
    monthly_times = pd.date_range(start="01-2020", periods=9, freq="M")
    monthly_ts = TimeSeries.from_times_and_values(
        times=monthly_times, values=monthly_values, columns=["values"]
    )

    monthly_not_complete_ts = monthly_ts[2:-1]

    quarterly_values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    quarterly_times = pd.date_range(start="01-2020", periods=3, freq="QS")
    quarterly_ts = TimeSeries.from_times_and_values(
        times=quarterly_times,
        values=quarterly_values,
        columns=["values_0", "values_1", "values_2"],
    )

    quarterly_values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    quarterly_end_times = pd.date_range(start="01-2020", periods=3, freq="Q")
    quarterly_with_quarter_end_index_ts = TimeSeries.from_times_and_values(
        times=quarterly_end_times,
        values=quarterly_values,
        columns=["values_0", "values_1", "values_2"],
    )

    quarterly_not_complete_values = np.array(
        [[np.nan, np.nan, 3], [4, 5, 6], [7, 8, np.nan]]
    )
    quarterly_times = pd.date_range(start="01-2020", periods=3, freq="QS")
    quarterly_not_complete_ts = TimeSeries.from_times_and_values(
        times=quarterly_times,
        values=quarterly_not_complete_values,
        columns=["values_0", "values_1", "values_2"],
    )

    daily_times = pd.date_range(start="01-2020", end="09-30-2020", freq="D")
    daily_values = np.arange(1, len(daily_times) + 1)
    daily_ts = TimeSeries.from_times_and_values(
        times=daily_times, values=daily_values, columns=["values"]
    )

    second_times = pd.date_range(start="01-2020", periods=120, freq="S")
    second_values = np.arange(1, len(second_times) + 1)
    second_ts = TimeSeries.from_times_and_values(
        times=second_times, values=second_values, columns=["values"]
    )

    minute_times = pd.date_range(start="01-2020", periods=2, freq="T")
    minute_values = np.array([[i for i in range(1, 61)], [i for i in range(61, 121)]])
    minute_ts = TimeSeries.from_times_and_values(
        times=minute_times,
        values=minute_values,
        columns=[f"values_{i}" for i in range(60)],
    )

    def test_complete_monthly_to_quarterly(self):
        """
        Tests if monthly series is transformed into a quarterly series in the expected way.
        """
        # 'complete' monthly series
        midas_1 = MIDAS(rule="QS")
        quarterly_midas_ts = midas_1.transform(self.monthly_ts)
        self.assertEqual(
            quarterly_midas_ts,
            self.quarterly_ts,
            "Monthly TimeSeries is not correctly transformed "
            "into a quarterly TimeSeries.",
        )

        # 'complete' monthly series
        midas_2 = MIDAS(rule="Q")
        quarterly_midas_ts = midas_2.transform(self.monthly_ts)
        self.assertEqual(
            quarterly_midas_ts,
            self.quarterly_with_quarter_end_index_ts,
            "Monthly TimeSeries is not correctly transformed "
            "into a quarterly TimeSeries. Specifically, when the rule requires an QuarterEnd index.",
        )

    def test_not_complete_monthly_to_quarterly(self):
        """
        Tests if a not 'complete' monthly series is transformed into a quarterly series in the expected way.
        """
        # not 'complete' monthly series
        midas = MIDAS(rule="QS", strip=False)
        quarterly_midas_not_complete_ts = midas.transform(self.monthly_not_complete_ts)
        self.assertEqual(
            quarterly_midas_not_complete_ts,
            self.quarterly_not_complete_ts,
            "Monthly TimeSeries is not "
            "correctly transformed when"
            " it is not 'complete'.",
        )

    def test_error_when_from_low_to_high(self):
        """
        Tests if the transformer raises an error when the user asks for a transform in the wrong direction.
        """
        # wrong direction / low to high freq
        midas_1 = MIDAS(rule="M")
        self.assertRaises(ValueError, midas_1.transform, self.quarterly_ts)

        # transform to same index requested
        midas_2 = MIDAS(rule="Q")
        self.assertRaises(ValueError, midas_2.transform, self.quarterly_ts)

    def test_error_when_frequency_not_suitable_for_midas(self):
        """
        MIDAS can only be performed when the high frequency is the same and the exact multiple of the low frequency.
        For example, there are always exactly three months in a quarter, but the number of days in a month differs.
        So the monthly to quarterly transformation is possible, while the daily to monthly MIDAS transform is
        impossible.
        """
        midas = MIDAS(rule="M")
        self.assertRaises(ValueError, midas.transform, self.daily_ts)

    def test_from_second_to_minute(self):
        """
        Test to see if other frequencies transforms like second to minute work as well.
        """
        midas = MIDAS(rule="T")
        self.assertEqual(midas.transform(self.second_ts), self.minute_ts)
