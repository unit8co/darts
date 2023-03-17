import unittest

import numpy as np
import pandas as pd

from darts import TimeSeries


class MIDASTestCase(unittest.TestCase):
    monthly_start_values = np.arange(1, 10)
    monthly_start_times = pd.date_range(start="01-2020", periods=9, freq="M")
    monthly_start_ts = TimeSeries.from_times_and_values(
        times=monthly_start_times, values=monthly_start_values, columns=["values"]
    )

    quarterly_end_values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    quarterly_end_times = pd.date_range(start="01-2020", periods=3, freq="Q")
    quarterly_end_ts = TimeSeries.from_times_and_values(
        times=quarterly_end_times,
        values=quarterly_end_values,
        columns=["values_0", "values_1", "values_2"],
    )
    # def assert_monthly_to_quarterly


monthly_values = np.arange(1, 10)
monthly_times = pd.date_range(start="01-2020", periods=9, freq="M")
monthly_ts = TimeSeries.from_times_and_values(
    times=monthly_times, values=monthly_values, columns=["values"]
)

values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
times = pd.date_range(start="01-2020", periods=9, freq="M")
monthly_ts = TimeSeries.from_times_and_values(
    times=times, values=values, columns=["values"]
)
