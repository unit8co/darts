import unittest

from ..utils.timeseries_generation import (
    constant_timeseries,
    linear_timeseries,
    sine_timeseries,
    gaussian_timeseries,
    random_walk_timeseries,
)


class TimeSeriesGenerationTestCase(unittest.TestCase):

    def test_constant_timeseries(self):

        # testing parameters
        length = 100
        value = 5

        # testing for constant value
        constant_ts = constant_timeseries(value=value, length=length)
        value_set = set(constant_ts._df.values.flatten())
        self.assertTrue(len(value_set) == 1)

    def test_linear_timeseries(self):

        # testing parameters
        length = 100
        start_value = 5
        end_value = 12

        # testing for start value, end value and delta between two adjacent entries
        linear_ts = linear_timeseries(start_value=start_value, end_value=end_value, length=length)
        self.assertEqual(linear_ts.values()[0][0], start_value)
        self.assertEqual(linear_ts.values()[-1][0], end_value)
        self.assertAlmostEqual(linear_ts.values()[-1][0] - linear_ts.values()[-2][0],
                               (end_value - start_value) / (length - 1))

    def test_sine_timeseries(self):

        # testing parameters
        length = 100
        value_amplitude = 5
        value_y_offset = -3

        # testing for correct value range
        sine_ts = sine_timeseries(length=length, value_amplitude=value_amplitude, value_y_offset=value_y_offset)
        self.assertTrue((sine_ts <= value_y_offset + value_amplitude).all().all())
        self.assertTrue((sine_ts >= value_y_offset - value_amplitude).all().all())

    def test_gaussian_timeseries(self):

        # testing parameters
        length = 100

        # testing for correct length
        gaussian_ts = gaussian_timeseries(length=length)
        self.assertEqual(len(gaussian_ts), length)

    def test_random_walk_timeseries(self):

        # testing parameters
        length = 100

        # testing for correct length
        random_walk_ts = random_walk_timeseries(length=length)
        self.assertEqual(len(random_walk_ts), length)
