import pandas as pd

from ..utils.timeseries_generation import (
    constant_timeseries,
    holidays_timeseries,
    linear_timeseries,
    sine_timeseries,
    gaussian_timeseries,
    random_walk_timeseries,
)
from .base_test_class import DartsBaseTestClass


class TimeSeriesGenerationTestCase(DartsBaseTestClass):

    def test_constant_timeseries(self):
        # testing parameters
        value = 5

        def test_routine(length):
            # testing for constant value
            constant_ts = constant_timeseries(value=value, length=length)
            value_set = set(constant_ts.values().flatten())
            self.assertTrue(len(value_set) == 1)

        for length in [1, 2, 5, 10, 100]:
            test_routine(length)

    def test_linear_timeseries(self):

        # testing parameters
        start_value = 5
        end_value = 12

        def test_routine(length):
            # testing for start value, end value and delta between two adjacent entries
            linear_ts = linear_timeseries(start_value=start_value, end_value=end_value, length=length)
            self.assertEqual(linear_ts.values()[0][0], start_value)
            self.assertEqual(linear_ts.values()[-1][0], end_value)
            self.assertAlmostEqual(linear_ts.values()[-1][0] - linear_ts.values()[-2][0],
                                   (end_value - start_value) / (length - 1))

        for length in [2, 5, 10, 100]:
            test_routine(length)

    def test_sine_timeseries(self):

        # testing parameters
        value_amplitude = 5
        value_y_offset = -3

        def test_routine(length):
            # testing for correct value range
            sine_ts = sine_timeseries(length=length, value_amplitude=value_amplitude, value_y_offset=value_y_offset)
            self.assertTrue((sine_ts <= value_y_offset + value_amplitude).all().all())
            self.assertTrue((sine_ts >= value_y_offset - value_amplitude).all().all())

        for length in [1, 2, 5, 10, 100]:
            test_routine(length)

    def test_gaussian_timeseries(self):

        # testing for correct length
        def test_routine(length):
            gaussian_ts = gaussian_timeseries(length=length)
            self.assertEqual(len(gaussian_ts), length)

        for length in [1, 2, 5, 10, 100]:
            test_routine(length)

    def test_random_walk_timeseries(self):

        # testing for correct length
        def test_routine(length):
            random_walk_ts = random_walk_timeseries(length=length)
            self.assertEqual(len(random_walk_ts), length)

        for length in [1, 2, 5, 10, 100]:
            test_routine(length)

    def test_holidays_timeseries(self):
        time_index_1 = pd.date_range(periods=365*3, freq='D', start=pd.Timestamp('2012-01-01'))
        time_index_2 = pd.date_range(periods=365 * 3, freq='D', start=pd.Timestamp('2014-12-24'))
        time_index_3 = pd.date_range(periods=10, freq='Y', start=pd.Timestamp('1950-01-01')) + pd.Timedelta(days=1)

        # testing we have at least one holiday flag in each year
        def test_routine(time_index, country_code, until=0, add_length=0):
            ts = holidays_timeseries(time_index, country_code, until=until, add_length=add_length)
            self.assertTrue(
                all(ts.pd_dataframe().groupby(pd.Grouper(freq="y")).sum().values)
            )

        for time_index in [time_index_1, time_index_2, time_index_3]:
            for country_code in ['US', 'CH', 'AR']:
                test_routine(time_index, country_code)

        # test extend time index
        test_routine(time_index_1, 'US',  add_length=365)
        test_routine(time_index_1, 'CH',  until='2016-01-01')
        test_routine(time_index_1, 'CH',  until='20160101')
        test_routine(time_index_1, 'AR',  until=pd.Timestamp('2016-01-01'))

        # test overflow
        with self.assertRaises(ValueError):
            holidays_timeseries(time_index_1, 'US', add_length=99999)

        # test date is too short
        with self.assertRaises(ValueError):
            holidays_timeseries(time_index_2, 'US', until='2016-01-01')

        # test wrong timestamp
        with self.assertRaises(ValueError):
            holidays_timeseries(time_index_3, 'US', until=163)

