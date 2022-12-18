import pandas as pd

from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.data.tabularization import get_shared_times_bounds
from darts.utils.timeseries_generation import linear_timeseries


class GetSharedTimesBoundsTestCase(DartsBaseTestClass):
    def test_series_with_overlap_range_index(self):
        long_series_1 = linear_timeseries(start=2, end=20, freq=2)
        long_series_2 = linear_timeseries(start=1, end=15, freq=3)
        lb_series = linear_timeseries(start=5, length=20, freq=4)
        ub_series = linear_timeseries(start=None, end=10, length=20, freq=5)
        self.assertEqual(
            get_shared_times_bounds(lb_series, ub_series, long_series_1, long_series_2),
            (lb_series.start_time(), ub_series.end_time()),
        )

    def test_series_with_overlap_datetime_index(self):
        long_series_1 = linear_timeseries(
            start=pd.Timestamp("1/2/2000"), end=pd.Timestamp("1/20/2000"), freq="2d"
        )
        long_series_2 = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), end=pd.Timestamp("1/15/2000"), freq="3d"
        )
        lb_series = linear_timeseries(
            start=pd.Timestamp("1/5/2000"), length=20, freq="4d"
        )
        ub_series = linear_timeseries(
            start=None, end=pd.Timestamp("1/10/2000"), length=20, freq="5d"
        )
        self.assertEqual(
            get_shared_times_bounds(lb_series, ub_series, long_series_1, long_series_2),
            (lb_series.start_time(), ub_series.end_time()),
        )

    def test_time_index_inputs(self):
        ub_series = linear_timeseries(start=0, end=10, freq=1)
        lb_series = linear_timeseries(start=2, end=16, freq=2)
        self.assertEqual(
            get_shared_times_bounds(ub_series.time_index),
            (ub_series.start_time(), ub_series.end_time()),
        )
        self.assertEqual(
            get_shared_times_bounds(lb_series.time_index),
            (lb_series.start_time(), lb_series.end_time()),
        )
        self.assertEqual(
            get_shared_times_bounds(lb_series.time_index, ub_series),
            (lb_series.start_time(), ub_series.end_time()),
        )
        self.assertEqual(
            get_shared_times_bounds(lb_series, ub_series.time_index),
            (lb_series.start_time(), ub_series.end_time()),
        )
        self.assertEqual(
            get_shared_times_bounds(lb_series.time_index, ub_series.time_index),
            (lb_series.start_time(), ub_series.end_time()),
        )

    def test_long_series_with_short_series_range_index(self):
        long_series = linear_timeseries(start=0, length=20, freq=3)
        med_series = linear_timeseries(start=1, length=10, freq=2)
        short_series = linear_timeseries(start=2, length=5, freq=1)
        self.assertEqual(
            get_shared_times_bounds(long_series, med_series, short_series),
            (short_series.start_time(), short_series.end_time()),
        )

    def test_long_series_with_short_series_datetime_index(self):
        long_series = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), length=20, freq="3d"
        )
        med_series = linear_timeseries(
            start=pd.Timestamp("1/2/2000"), length=10, freq="2d"
        )
        short_series = linear_timeseries(
            start=pd.Timestamp("1/3/2000"), length=5, freq="d"
        )
        self.assertEqual(
            get_shared_times_bounds(long_series, med_series, short_series),
            (short_series.start_time(), short_series.end_time()),
        )

    def test_identical_inputs_range_index(self):
        series = linear_timeseries(start=0, length=5, freq=1)
        expected = (series.start_time(), series.end_time())
        self.assertEqual(get_shared_times_bounds(series), expected)
        self.assertEqual(get_shared_times_bounds(series, series), expected)
        self.assertEqual(get_shared_times_bounds(series, series, series), expected)

    def test_identical_inputs_datetime_index(self):
        series = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=5, freq="d")
        expected = (series.start_time(), series.end_time())
        self.assertEqual(get_shared_times_bounds(series), expected)
        self.assertEqual(get_shared_times_bounds(series, series), expected)
        self.assertEqual(get_shared_times_bounds(series, series, series), expected)

    def test_unspecified_inputs(self):
        series = linear_timeseries(start=0, length=5, freq=1)
        expected = (series.start_time(), series.end_time())
        self.assertEqual(get_shared_times_bounds(series, None), expected)
        self.assertEqual(get_shared_times_bounds(None, series), expected)
        self.assertEqual(get_shared_times_bounds(None, series, None), expected)

        # `None` should be returned if no series specified:
        self.assertEqual(get_shared_times_bounds(None), None)
        self.assertEqual(get_shared_times_bounds(None, None, None), None)

    def test_single_index_overlap_range_index(self):
        series_1 = linear_timeseries(start=0, length=5, freq=1)
        series_2 = linear_timeseries(start=None, end=1, length=6, freq=2)
        series_3 = linear_timeseries(start=0, length=7, freq=3)
        self.assertEqual(get_shared_times_bounds(series_1, series_2, series_3), (0, 1))

    def test_single_index_overlap_datetime_index(self):
        series_1 = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=5, freq="d")
        series_2 = linear_timeseries(
            start=None, end=pd.Timestamp("1/2/2000"), length=6, freq="2d"
        )
        series_3 = linear_timeseries(
            start=pd.Timestamp("1/1/2000"), length=7, freq="3d"
        )
        self.assertEqual(
            get_shared_times_bounds(series_1, series_2, series_3),
            (series_1.start_time(), series_2.end_time()),
        )

    def test_no_overlap_range_index(self):
        series_1 = linear_timeseries(start=0, length=5, freq=1)
        series_2 = linear_timeseries(start=series_1.end_time(), length=6, freq=2)
        self.assertEqual(get_shared_times_bounds(series_1, series_2), None)
        self.assertEqual(get_shared_times_bounds(series_2, series_1, series_2), None)

    def test_no_overlap_datetime_index(self):
        series_1 = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=5, freq="d")
        series_2 = linear_timeseries(start=series_1.end_time(), length=6, freq="2d")
        self.assertEqual(get_shared_times_bounds(series_1, series_2), None)
        self.assertEqual(get_shared_times_bounds(series_2, series_1, series_2), None)

    def test_different_time_index_types_error(self):
        series_1 = linear_timeseries(start=1, length=5, freq=1)
        series_2 = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=5, freq="d")
        with self.assertRaises(ValueError) as e:
            get_shared_times_bounds(series_1, series_2)
        self.assertEqual(
            (
                "Specified series and/or times must all "
                "have the same type of `time_index` (i.e. all "
                "`pd.RangeIndex` or all `pd.DatetimeIndex`)."
            ),
            str(e.exception),
        )
