from math import lcm

import pandas as pd

from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.data.tabularization import get_shared_times
from darts.utils.timeseries_generation import linear_timeseries


class GetSharedTimesTestCase(DartsBaseTestClass):

    """
    Tests `get_shared_times` function defined in `darts.utils.data.tabularization`.
    """

    def test_shared_times_equal_freq_range_idx(self):
        series_1 = linear_timeseries(start=1, end=11, freq=2)
        series_2 = linear_timeseries(start=3, end=13, freq=2)
        series_3 = linear_timeseries(start=5, end=15, freq=2)
        self.assertTrue(series_1.time_index.equals(get_shared_times(series_1)))
        self.assertTrue(series_2.time_index.equals(get_shared_times(series_2)))
        self.assertTrue(series_3.time_index.equals(get_shared_times(series_3)))
        expected_12 = linear_timeseries(
            start=series_2.start_time(), end=series_1.end_time(), freq=series_1.freq
        )
        self.assertTrue(
            expected_12.time_index.equals(get_shared_times(series_1, series_2))
        )
        expected_23 = linear_timeseries(
            start=series_3.start_time(), end=series_2.end_time(), freq=series_2.freq
        )
        self.assertTrue(
            expected_23.time_index.equals(get_shared_times(series_2, series_3))
        )
        expected_13 = linear_timeseries(
            start=series_3.start_time(), end=series_1.end_time(), freq=series_1.freq
        )
        self.assertTrue(
            expected_13.time_index.equals(get_shared_times(series_1, series_3))
        )
        expected_123 = linear_timeseries(
            start=series_3.start_time(), end=series_1.end_time(), freq=series_1.freq
        )
        self.assertTrue(
            expected_123.time_index.equals(
                get_shared_times(series_1, series_2, series_3)
            )
        )

    def test_shared_times_equal_freq_datetime_idx(self):
        series_1 = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), end=pd.Timestamp("1/11/2000"), freq="2d"
        )
        series_2 = linear_timeseries(
            start=pd.Timestamp("1/3/2000"), end=pd.Timestamp("1/13/2000"), freq="2d"
        )
        series_3 = linear_timeseries(
            start=pd.Timestamp("1/5/2000"), end=pd.Timestamp("1/15/2000"), freq="2d"
        )
        self.assertTrue(series_1.time_index.equals(get_shared_times(series_1)))
        self.assertTrue(series_2.time_index.equals(get_shared_times(series_2)))
        self.assertTrue(series_3.time_index.equals(get_shared_times(series_3)))
        expected_12 = linear_timeseries(
            start=series_2.start_time(), end=series_1.end_time(), freq=series_1.freq
        )
        self.assertTrue(
            expected_12.time_index.equals(get_shared_times(series_1, series_2))
        )
        expected_23 = linear_timeseries(
            start=series_3.start_time(), end=series_2.end_time(), freq=series_2.freq
        )
        self.assertTrue(
            expected_23.time_index.equals(get_shared_times(series_2, series_3))
        )
        expected_13 = linear_timeseries(
            start=series_3.start_time(), end=series_1.end_time(), freq=series_1.freq
        )
        self.assertTrue(
            expected_13.time_index.equals(get_shared_times(series_1, series_3))
        )
        expected_123 = linear_timeseries(
            start=series_3.start_time(), end=series_1.end_time(), freq=series_1.freq
        )
        self.assertTrue(
            expected_123.time_index.equals(
                get_shared_times(series_1, series_2, series_3)
            )
        )

    def test_shared_times_unequal_freq_range_idx(self):
        series_1 = linear_timeseries(start=1, end=11, freq=1)
        series_2 = linear_timeseries(start=3, end=13, freq=2)
        series_3 = linear_timeseries(start=5, end=17, freq=3)
        self.assertTrue(series_1.time_index.equals(get_shared_times(series_1)))
        self.assertTrue(series_2.time_index.equals(get_shared_times(series_2)))
        self.assertTrue(series_3.time_index.equals(get_shared_times(series_3)))
        expected_12 = linear_timeseries(
            start=series_2.start_time(),
            end=series_1.end_time(),
            freq=lcm(series_1.freq, series_2.freq),
        )
        if expected_12.time_index[-1] > series_1.end_time():
            expected_12 = expected_12.drop_after(expected_12.time_index[-1])
        self.assertTrue(
            expected_12.time_index.equals(get_shared_times(series_1, series_2))
        )
        expected_23 = linear_timeseries(
            start=series_3.start_time(),
            end=series_2.end_time(),
            freq=lcm(series_2.freq, series_3.freq),
        )
        if expected_23.time_index[-1] > series_2.end_time():
            expected_23 = expected_23.drop_after(expected_23.time_index[-1])
        self.assertTrue(
            expected_23.time_index.equals(get_shared_times(series_2, series_3))
        )
        expected_13 = linear_timeseries(
            start=series_3.start_time(),
            end=series_1.end_time(),
            freq=lcm(series_1.freq, series_3.freq),
        )
        if expected_13.time_index[-1] > series_1.end_time():
            expected_13 = expected_13.drop_after(expected_13.time_index[-1])
        self.assertTrue(
            expected_13.time_index.equals(get_shared_times(series_1, series_3))
        )
        expected_123 = linear_timeseries(
            start=series_3.start_time(),
            end=series_1.end_time(),
            freq=lcm(series_1.freq, series_2.freq, series_3.freq),
        )
        if expected_123.time_index[-1] > series_1.end_time():
            expected_123 = expected_123.drop_after(expected_123.time_index[-1])
        self.assertTrue(
            expected_123.time_index.equals(
                get_shared_times(series_1, series_2, series_3)
            )
        )

    def test_shared_times_unequal_freq_datetime_idx(self):
        series_1 = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), end=pd.Timestamp("1/11/2000"), freq="2d"
        )
        series_2 = linear_timeseries(
            start=pd.Timestamp("1/3/2000"), end=pd.Timestamp("1/13/2000"), freq="2d"
        )
        series_3 = linear_timeseries(
            start=pd.Timestamp("1/5/2000"), end=pd.Timestamp("1/15/2000"), freq="2d"
        )
        self.assertTrue(series_1.time_index.equals(get_shared_times(series_1)))
        self.assertTrue(series_2.time_index.equals(get_shared_times(series_2)))
        self.assertTrue(series_3.time_index.equals(get_shared_times(series_3)))
        freq_12 = lcm(series_1.freq.n, series_2.freq.n) * series_1.freq.base
        expected_12 = linear_timeseries(
            start=series_2.start_time(), end=series_1.end_time(), freq=freq_12
        )
        if expected_12.time_index[-1] > series_1.end_time():
            expected_12 = expected_12.drop_after(expected_12.time_index[-1])
        self.assertTrue(
            expected_12.time_index.equals(get_shared_times(series_1, series_2))
        )
        freq_23 = lcm(series_2.freq.n, series_3.freq.n) * series_2.freq.base
        expected_23 = linear_timeseries(
            start=series_3.start_time(), end=series_2.end_time(), freq=freq_23
        )
        if expected_23.time_index[-1] > series_2.end_time():
            expected_23 = expected_23.drop_after(expected_23.time_index[-1])
        self.assertTrue(
            expected_23.time_index.equals(get_shared_times(series_2, series_3))
        )
        freq_13 = lcm(series_1.freq.n, series_3.freq.n) * series_1.freq.base
        expected_13 = linear_timeseries(
            start=series_3.start_time(), end=series_1.end_time(), freq=freq_13
        )
        if expected_13.time_index[-1] > series_1.end_time():
            expected_13 = expected_13.drop_after(expected_13.time_index[-1])
        self.assertTrue(
            expected_13.time_index.equals(get_shared_times(series_1, series_3))
        )
        freq_123 = (
            lcm(series_1.freq.n, series_2.freq.n, series_3.freq.n) * series_1.freq.base
        )
        expected_123 = linear_timeseries(
            start=series_3.start_time(), end=series_1.end_time(), freq=freq_123
        )
        if expected_123.time_index[-1] > series_1.end_time():
            expected_123 = expected_123.drop_after(expected_123.time_index[-1])
        self.assertTrue(
            expected_123.time_index.equals(
                get_shared_times(series_1, series_2, series_3)
            )
        )

    def test_shared_times_no_overlap_range_idx(self):
        series_1 = linear_timeseries(start=1, end=11, freq=2)
        series_2 = linear_timeseries(start=series_1.end_time() + 1, length=5, freq=3)
        self.assertEqual(get_shared_times(series_1, series_2), None)
        self.assertEqual(get_shared_times(series_1, series_1, series_2), None)
        self.assertEqual(get_shared_times(series_1, series_2, series_2), None)
        self.assertEqual(get_shared_times(series_1, series_1, series_2, series_2), None)

    def test_shared_times_no_overlap_datetime_idx(self):
        series_1 = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), end=pd.Timestamp("1/11/2000"), freq="2d"
        )
        series_2 = linear_timeseries(
            start=series_1.end_time() + pd.Timedelta(1, "d"), length=5, freq="3d"
        )
        self.assertEqual(get_shared_times(series_1, series_2), None)
        self.assertEqual(get_shared_times(series_1, series_1, series_2), None)
        self.assertEqual(get_shared_times(series_1, series_2, series_2), None)
        self.assertEqual(get_shared_times(series_1, series_1, series_2, series_2), None)

    def test_shared_times_single_time_point_overlap_range_idx(self):
        series_1 = linear_timeseries(start=1, end=11, freq=2)
        series_2 = linear_timeseries(start=series_1.end_time(), length=5, freq=3)
        overlap_val = series_1.end_time()
        self.assertEqual(get_shared_times(series_1, series_2), overlap_val)
        self.assertEqual(get_shared_times(series_1, series_1, series_2), overlap_val)
        self.assertEqual(get_shared_times(series_1, series_2, series_2), overlap_val)
        self.assertEqual(
            get_shared_times(series_1, series_1, series_2, series_2), overlap_val
        )

    def test_shared_times_single_time_point_overlap_datetime_idx(self):
        series_1 = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), end=pd.Timestamp("1/11/2000"), freq="2d"
        )
        series_2 = linear_timeseries(start=series_1.end_time(), length=5, freq="3d")
        overlap_val = series_1.end_time()
        self.assertEqual(get_shared_times(series_1, series_2), overlap_val)
        self.assertEqual(get_shared_times(series_1, series_1, series_2), overlap_val)
        self.assertEqual(get_shared_times(series_1, series_2, series_2), overlap_val)
        self.assertEqual(
            get_shared_times(series_1, series_1, series_2, series_2), overlap_val
        )

    def test_shared_times_different_time_index_types_error(self):
        series_1 = linear_timeseries(start=1, length=5, freq=1)
        series_2 = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=5, freq="d")
        with self.assertRaises(ValueError) as e:
            get_shared_times(series_1, series_2)
        self.assertEqual(
            (
                "Specified series and/or times must all "
                "have the same type of `time_index` (i.e. all "
                "`pd.RangeIndex` or all `pd.DatetimeIndex`)."
            ),
            str(e.exception),
        )

    def test_shared_times_identical_inputs_range_idx(self):
        series = linear_timeseries(start=0, length=5, freq=1)
        self.assertTrue(series.time_index.equals(get_shared_times(series)))
        self.assertTrue(series.time_index.equals(get_shared_times(series, series)))
        self.assertTrue(
            series.time_index.equals(get_shared_times(series, series, series))
        )

    def test_shared_times_identical_inputs_datetime_idx(self):
        series = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=5, freq="d")
        self.assertTrue(series.time_index.equals(get_shared_times(series)))
        self.assertTrue(series.time_index.equals(get_shared_times(series, series)))
        self.assertTrue(
            series.time_index.equals(get_shared_times(series, series, series))
        )

    def test_shared_times_unspecified_inputs(self):
        series = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=5, freq="d")
        self.assertEqual(get_shared_times(None), None)
        self.assertTrue(series.time_index.equals(get_shared_times(series, None)))
        self.assertTrue(series.time_index.equals(get_shared_times(None, series, None)))
        self.assertTrue(
            series.time_index.equals(get_shared_times(None, series.time_index, None))
        )

    def test_shared_times_time_index_inputs(self):
        series_1 = linear_timeseries(start=0, end=10, freq=1)
        series_2 = linear_timeseries(start=0, end=20, freq=2)
        # `stop=10+1` since `stop` is exclusive
        intersection = pd.RangeIndex(start=0, stop=10 + 1, step=2)
        self.assertTrue(
            intersection.equals(get_shared_times(series_1.time_index, series_2))
        )
        self.assertTrue(
            intersection.equals(get_shared_times(series_1, series_2.time_index))
        )
        self.assertTrue(
            intersection.equals(
                get_shared_times(series_1.time_index, series_2.time_index)
            )
        )

    def test_shared_times_empty_input(self):
        series = linear_timeseries(start=0, length=0, freq=1)
        self.assertEqual(get_shared_times(series), None)
        self.assertEqual(get_shared_times(series.time_index), None)
        self.assertEqual(get_shared_times(series, series.time_index), None)
