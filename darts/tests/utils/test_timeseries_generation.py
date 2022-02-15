from typing import Union

import numpy as np
import pandas as pd

from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.timeseries_generation import (
    _generate_index,
    autoregressive_timeseries,
    constant_timeseries,
    gaussian_timeseries,
    holidays_timeseries,
    linear_timeseries,
    random_walk_timeseries,
    sine_timeseries,
)


class TimeSeriesGenerationTestCase(DartsBaseTestClass):
    def test_constant_timeseries(self):
        # testing parameters
        value = 5

        def test_routine(start, end=None, length=None):
            # testing for constant value
            constant_ts = constant_timeseries(
                start=start, end=end, value=value, length=length
            )
            value_set = set(constant_ts.values().flatten())
            self.assertTrue(len(value_set) == 1)
            self.assertEqual(len(constant_ts), length_assert)

        for length_assert in [1, 2, 5, 10, 100]:
            test_routine(start=0, length=length_assert)
            test_routine(start=0, end=length_assert - 1)
            test_routine(start=pd.Timestamp("2000-01-01"), length=length_assert)
            end_date = _generate_index(
                start=pd.Timestamp("2000-01-01"), length=length_assert
            )[-1]
            test_routine(start=pd.Timestamp("2000-01-01"), end=end_date)

    def test_linear_timeseries(self):

        # testing parameters
        start_value = 5
        end_value = 12

        def test_routine(start, end=None, length=None):
            # testing for start value, end value and delta between two adjacent entries
            linear_ts = linear_timeseries(
                start=start,
                end=end,
                length=length,
                start_value=start_value,
                end_value=end_value,
            )
            self.assertEqual(linear_ts.values()[0][0], start_value)
            self.assertEqual(linear_ts.values()[-1][0], end_value)
            self.assertAlmostEqual(
                linear_ts.values()[-1][0] - linear_ts.values()[-2][0],
                (end_value - start_value) / (length_assert - 1),
            )
            self.assertEqual(len(linear_ts), length_assert)

        for length_assert in [2, 5, 10, 100]:
            test_routine(start=0, length=length_assert)
            test_routine(start=0, end=length_assert - 1)
            test_routine(start=pd.Timestamp("2000-01-01"), length=length_assert)
            end_date = _generate_index(
                start=pd.Timestamp("2000-01-01"), length=length_assert
            )[-1]
            test_routine(start=pd.Timestamp("2000-01-01"), end=end_date)

    def test_sine_timeseries(self):

        # testing parameters
        value_amplitude = 5
        value_y_offset = -3

        def test_routine(start, end=None, length=None):
            # testing for correct value range
            sine_ts = sine_timeseries(
                start=start,
                end=end,
                length=length,
                value_amplitude=value_amplitude,
                value_y_offset=value_y_offset,
            )
            self.assertTrue((sine_ts <= value_y_offset + value_amplitude).all().all())
            self.assertTrue((sine_ts >= value_y_offset - value_amplitude).all().all())
            self.assertEqual(len(sine_ts), length_assert)

        for length_assert in [1, 2, 5, 10, 100]:
            test_routine(start=0, length=length_assert)
            test_routine(start=0, end=length_assert - 1)
            test_routine(start=pd.Timestamp("2000-01-01"), length=length_assert)
            end_date = _generate_index(
                start=pd.Timestamp("2000-01-01"), length=length_assert
            )[-1]
            test_routine(start=pd.Timestamp("2000-01-01"), end=end_date)

    def test_gaussian_timeseries(self):

        # testing for correct length
        def test_routine(start, end=None, length=None):
            gaussian_ts = gaussian_timeseries(start=start, end=end, length=length)
            self.assertEqual(len(gaussian_ts), length_assert)

        for length_assert in [1, 2, 5, 10, 100]:
            test_routine(start=0, length=length_assert)
            test_routine(start=0, end=length_assert - 1)
            test_routine(start=pd.Timestamp("2000-01-01"), length=length_assert)
            end_date = _generate_index(
                start=pd.Timestamp("2000-01-01"), length=length_assert
            )[-1]
            test_routine(start=pd.Timestamp("2000-01-01"), end=end_date)

    def test_random_walk_timeseries(self):

        # testing for correct length
        def test_routine(start, end=None, length=None):
            random_walk_ts = random_walk_timeseries(start=start, end=end, length=length)
            self.assertEqual(len(random_walk_ts), length_assert)

        for length_assert in [1, 2, 5, 10, 100]:
            test_routine(start=0, length=length_assert)
            test_routine(start=0, end=length_assert - 1)
            test_routine(start=pd.Timestamp("2000-01-01"), length=length_assert)
            end_date = _generate_index(
                start=pd.Timestamp("2000-01-01"), length=length_assert
            )[-1]
            test_routine(start=pd.Timestamp("2000-01-01"), end=end_date)

    def test_holidays_timeseries(self):
        time_index_1 = pd.date_range(
            periods=365 * 3, freq="D", start=pd.Timestamp("2012-01-01")
        )
        time_index_2 = pd.date_range(
            periods=365 * 3, freq="D", start=pd.Timestamp("2014-12-24")
        )
        time_index_3 = pd.date_range(
            periods=10, freq="Y", start=pd.Timestamp("1950-01-01")
        ) + pd.Timedelta(days=1)

        # testing we have at least one holiday flag in each year
        def test_routine(
            time_index,
            country_code,
            until: Union[int, pd.Timestamp, str] = 0,
            add_length=0,
        ):
            ts = holidays_timeseries(
                time_index, country_code, until=until, add_length=add_length
            )
            self.assertTrue(
                all(ts.pd_dataframe().groupby(pd.Grouper(freq="y")).sum().values)
            )

        for time_index in [time_index_1, time_index_2, time_index_3]:
            for country_code in ["US", "CH", "AR"]:
                test_routine(time_index, country_code)

        # test extend time index
        test_routine(time_index_1, "US", add_length=365)
        test_routine(time_index_1, "CH", until="2016-01-01")
        test_routine(time_index_1, "CH", until="20160101")
        test_routine(time_index_1, "AR", until=pd.Timestamp("2016-01-01"))

        # test overflow
        with self.assertRaises(ValueError):
            holidays_timeseries(time_index_1, "US", add_length=99999)

        # test date is too short
        with self.assertRaises(ValueError):
            holidays_timeseries(time_index_2, "US", until="2016-01-01")

        # test wrong timestamp
        with self.assertRaises(ValueError):
            holidays_timeseries(time_index_3, "US", until=163)

    def test_generate_index(self):
        def test_routine(start, end=None, length=None, freq="D"):
            # testing length, correct start and if sorted (monotonic increasing)
            index = _generate_index(start=start, end=end, length=length, freq=freq)
            self.assertEqual(len(index), length_assert)
            self.assertTrue(index.is_monotonic_increasing)
            self.assertTrue(index[0] == start_assert)
            self.assertTrue(index[-1] == end_assert)

        for length_assert in [1, 2, 5, 10, 100]:
            for start_pos in [0, 1]:
                # pandas.RangeIndex
                start_assert, end_assert = start_pos, start_pos + length_assert - 1
                test_routine(start=start_assert, length=length_assert, freq="")
                test_routine(start=start_assert, length=length_assert, freq="D")
                test_routine(start=start_assert, end=end_assert)
                test_routine(start=start_assert, end=end_assert, freq="D")
                test_routine(
                    start=None, end=end_assert, length=length_assert, freq="BH"
                )
                # pandas.DatetimeIndex
                start_date = pd.DatetimeIndex(["2000-01-01"], freq="D")
                start_date += start_date.freq * start_pos
                # dates = _generate_index(start=start_date[0], length=length_assert)
                dates = _generate_index(start=start_date[0], length=length_assert)
                start_assert, end_assert = dates[0], dates[-1]
                test_routine(start=start_assert, length=length_assert)
                test_routine(start=start_assert, end=end_assert)
                test_routine(start=None, end=end_assert, length=length_assert, freq="D")

        # `start`, `end` and `length` cannot both be set simultaneously
        with self.assertRaises(ValueError):
            _generate_index(start=0, end=9, length=10)
        # same as above but `start` defaults to timestamp '2000-01-01' in all timeseries generation functions
        with self.assertRaises(ValueError):
            linear_timeseries(end=9, length=10)

        # exactly two of [`start`, `end`, `length`] must be set
        with self.assertRaises(ValueError):
            test_routine(start=0)
        with self.assertRaises(ValueError):
            test_routine(start=None, end=1)
        with self.assertRaises(ValueError):
            test_routine(start=None, end=None, length=10)

        # `start` and `end` must have same type
        with self.assertRaises(ValueError):
            test_routine(start=0, end=pd.Timestamp("2000-01-01"))
        with self.assertRaises(ValueError):
            test_routine(start=pd.Timestamp("2000-01-01"), end=10)

    def test_autoregressive_timeseries(self):
        # testing for correct length
        def test_length(start, end=None, length=None):
            autoregressive_ts = autoregressive_timeseries(
                coef=[-1, 1.618], start=start, end=end, length=length
            )
            self.assertEqual(len(autoregressive_ts), length_assert)

        # testing for correct calculation
        def test_calculation(coef):
            autoregressive_values = autoregressive_timeseries(
                coef=coef, length=100
            ).values()
            for idx, val in enumerate(autoregressive_values[len(coef) :]):
                self.assertTrue(
                    val
                    == np.dot(
                        coef, autoregressive_values[idx : idx + len(coef)].ravel()
                    )
                )

        for length_assert in [1, 2, 5, 10, 100]:
            test_length(start=0, length=length_assert)
            test_length(start=0, end=length_assert - 1)
            test_length(start=pd.Timestamp("2000-01-01"), length=length_assert)
            end_date = _generate_index(
                start=pd.Timestamp("2000-01-01"), length=length_assert
            )[-1]
            test_length(start=pd.Timestamp("2000-01-01"), end=end_date)

        for coef_assert in [[-1], [-1, 1.618], [1, 2, 3], list(range(10))]:
            test_calculation(coef=coef_assert)
