import unittest
import pandas as pd
import numpy as np

from ..timeseries import TimeSeries
from u8timeseries.utils.timeseries_generation import (
    constant_timeseries, 
    linear_timeseries, 
    periodic_timeseries,
    white_noise_timeseries, 
    random_walk_timeseries, 
    us_holiday_timeseries
)

class TimeSeriesGenerationTestCase(unittest.TestCase):

    def test_constant_timeseries(self):

        # testing parameters
        length = 100
        value = 5

        # testing for constant value
        constant_ts = constant_timeseries(value=value, length=length)
        value_set = set(constant_ts._series.values)
        self.assertTrue(len(value_set) == 1)

    def test_linear_timeseries(self):

        # testing parameters
        length = 100
        start_value = 5
        value_delta = 3

        # testing for correct start and end values
        linear_ts = linear_timeseries(start_value=start_value, value_delta=value_delta, length=length)
        self.assertEqual(linear_ts.values()[0], start_value)
        self.assertEqual(linear_ts.values()[-1], start_value + (length - 1) * value_delta)

    def test_periodic_timeseries(self):

        # testing parameters
        length = 100
        amplitude = 5
        y_offset = -3

        # testing for correct value range
        periodic_ts = periodic_timeseries(length=length, amplitude=amplitude, y_offset=y_offset)
        self.assertTrue((periodic_ts <= y_offset + amplitude).all())
        self.assertTrue((periodic_ts >= y_offset - amplitude).all())

    def test_white_noise_timeseries(self):

        # testing parameters
        length = 100

        # testing for correct length
        white_noise_ts = white_noise_timeseries(length=length)
        self.assertEqual(len(white_noise_ts), length)

    def test_random_walk_timeseries(self):

        # testing parameters
        length = 100

        # testing for correct length
        random_walk_ts = random_walk_timeseries(length=length)
        self.assertEqual(len(random_walk_ts), length)

    def test_us_holiday_timeseries(self):

        # testing parameters
        length = 30
        start_date = pd.Timestamp('20201201')

        # testing for christmas and non-holiday
        us_holiday_ts = us_holiday_timeseries(length=length, start_date=start_date)
        self.assertEqual(us_holiday_ts._series.at[pd.Timestamp('20201225')], 1)
        self.assertEqual(us_holiday_ts._series.at[pd.Timestamp('20201210')], 0)
