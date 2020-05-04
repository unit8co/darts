import unittest
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..models.fft import find_relevant_timestamp_attributes
from ..utils import timeseries_generation as tg


class FFTTestCase(unittest.TestCase):

    def test_find_relevant_timestamp_attributes(self):

        np.random.seed(0)
        ts_length_1 = 1000
        ts_length_2 = 150


        ### daily frequency ###

        # random walk
        random_walk_ts = tg.random_walk_timeseries(length=ts_length_1)
        self.assertEqual(find_relevant_timestamp_attributes(random_walk_ts), set())

        # yearly period with no noise 
        yearly_ts = tg.sine_timeseries(value_frequency=1/365, length=ts_length_1)
        self.assertEqual(find_relevant_timestamp_attributes(yearly_ts), {'month', 'day'})

        # yearly period with noise 
        yearly_noisy_ts = yearly_ts + tg.gaussian_timeseries(length=ts_length_1)
        self.assertEqual(find_relevant_timestamp_attributes(yearly_noisy_ts), {'month', 'day'})

        # monthly period with no noise
        monthly_ts = tg.sine_timeseries(value_frequency=1/30, length=ts_length_1)
        self.assertEqual(find_relevant_timestamp_attributes(monthly_ts), {'day'})

        # monthly period with noise
        monthly_noisy_ts = monthly_ts + tg.gaussian_timeseries(length=ts_length_1)
        self.assertEqual(find_relevant_timestamp_attributes(monthly_noisy_ts), {'day'})


        ### monthly frequency ###

        # random walk
        random_walk_ts = tg.random_walk_timeseries(freq='M', length=ts_length_2)
        self.assertEqual(find_relevant_timestamp_attributes(random_walk_ts), set())

        # yearly period with no noise
        yearly_ts_2 = tg.sine_timeseries(freq='M', value_frequency=1/12, length=ts_length_2)
        self.assertEqual(find_relevant_timestamp_attributes(yearly_ts_2), {'month'})

        # yearly period with noise 
        yearly_noisy_ts_2 = yearly_ts_2 + tg.gaussian_timeseries(freq='M', length=ts_length_2)
        self.assertEqual(find_relevant_timestamp_attributes(yearly_noisy_ts_2), {'month'})

        

