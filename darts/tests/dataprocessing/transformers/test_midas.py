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

    monthly_not_complete_ts = monthly_ts[2:]

    quarterly_values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    quarterly_times = pd.date_range(start="01-2020", periods=3, freq="QS")
    quarterly_ts = TimeSeries.from_times_and_values(
        times=quarterly_times,
        values=quarterly_values,
        columns=["values_0", "values_1", "values_2"],
    )

    quarterly_not_complete_values = np.array(
        [[np.nan, np.nan, 3], [4, 5, 6], [7, 8, 9]]
    )
    quarterly_times = pd.date_range(start="01-2020", periods=3, freq="QS")
    quarterly_not_complete_ts = TimeSeries.from_times_and_values(
        times=quarterly_times,
        values=quarterly_not_complete_values,
        columns=["values_0", "values_1", "values_2"],
    )

    def test_complete_monthly_to_quarterly(self):
        """
        Tests if monthly series is transformed into a quarterly series in the expected way.
        """
        # 'complete' monthly series
        midas = MIDAS(rule="QS")
        quarterly_midas_ts = midas.transform(self.monthly_ts)
        self.assertEqual(
            quarterly_midas_ts,
            self.quarterly_ts,
            "Monthly TimeSeries is not correctly transformed "
            "into a quarterly TimeSeries.",
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

    # def assert_error_when_from_low_to_high(self):
    # """
    # Tests if the transformer raises an error when the user asks for a transform in the wrong direction.
    # """
    # wrong direction / low to high freq
    # midas = MIDAS(rule="Q")
    # self.assertRaises(ValueError):
    #    midas.transform(self.quarterly_ts)
    # self.assertRaises(ValueError, midas.transform, self.monthly_ts)


monthly_values = np.arange(1, 10)
monthly_times = pd.date_range(start="01-2020", periods=9, freq="M")
monthly_ts = TimeSeries.from_times_and_values(
    times=monthly_times, values=monthly_values, columns=["values"]
)

monthly_not_complete_ts = monthly_ts[2:]

quarterly_values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
quarterly_times = pd.date_range(start="01-2020", periods=3, freq="QS")
quarterly_ts = TimeSeries.from_times_and_values(
    times=quarterly_times,
    values=quarterly_values,
    columns=["values_0", "values_1", "values_2"],
)

quarterly_not_complete_values = np.array([[np.nan, np.nan, 3], [4, 5, 6], [7, 8, 9]])
quarterly_times = pd.date_range(start="01-2020", periods=3, freq="QS")
quarterly_not_complete_ts = TimeSeries.from_times_and_values(
    times=quarterly_times,
    values=quarterly_not_complete_values,
    columns=["values_0", "values_1", "values_2"],
)

# midas = MIDAS(rule="M")
# midas.transform(quarterly_ts)
