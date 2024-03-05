import pandas as pd
import pytest

from darts.utils.data.tabularization import get_shared_times_bounds
from darts.utils.timeseries_generation import linear_timeseries


class TestGetSharedTimesBounds:
    """
    Tests `get_shared_times_bounds` function defined in `darts.utils.data.tabularization`.
    """

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_shared_times_bounds_overlapping_range_idx_series(self, series_type):
        """
        Tests that `get_shared_times_bounds` correctly computes bounds
        of two overlapping time index timeseries.
        """
        # Defined so `series_1` starts and ends before `series_2` does:
        if series_type == "integer":
            series_1 = linear_timeseries(start=1, end=15, freq=3)
            series_2 = linear_timeseries(start=2, end=20, freq=2)
        else:
            series_1 = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), end=pd.Timestamp("1/15/2000"), freq="3d"
            )
            series_2 = linear_timeseries(
                start=pd.Timestamp("1/2/2000"), end=pd.Timestamp("1/20/2000"), freq="2d"
            )
        expected_bounds = (series_2.start_time(), series_1.end_time())
        assert get_shared_times_bounds(series_1, series_2) == expected_bounds

    def test_shared_times_bounds_time_idx_inputs(self):
        """
        Tests that `get_shared_times_bounds` behaves correctly
        when passed `pd.Index` inputs instead of `TimeSeries`
        inputs. Mixtures of `pd.Index` and `TimeSeries` inputs
        are also checked.
        """
        # Defined so `series_1` starts and ends before `series_2` does:
        series_1 = linear_timeseries(start=0, end=10, freq=1)
        series_2 = linear_timeseries(start=2, end=16, freq=2)
        expected_bounds = (series_2.start_time(), series_1.end_time())
        # Pass only `series_1.time_index` - bounds are just start and
        # end times of this series:
        assert get_shared_times_bounds(series_1.time_index) == (
            series_1.start_time(),
            series_1.end_time(),
        )
        # Pass only `series_2.time_index` - bounds are just start and
        # end times of this series:
        assert get_shared_times_bounds(series_2.time_index) == (
            series_2.start_time(),
            series_2.end_time(),
        )
        assert get_shared_times_bounds(series_1.time_index, series_2) == expected_bounds
        assert get_shared_times_bounds(series_1, series_2.time_index) == expected_bounds
        assert (
            get_shared_times_bounds(series_1.time_index, series_2.time_index)
            == expected_bounds
        )

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_shared_times_bounds_subset_series(self, series_type):
        """
        Tests that `get_shared_times_bounds` correctly handles case where
        the provided series are formed by taking successive subsets of an
        initial series (i.e. `series_2` is formed by taking a subset of
        `series_1`, and `series_3` is formed by taking a subset of `series_2`).
        In such cases, the bounds are simply the start and end times of the
        shortest series.
        """
        if series_type == "integer":
            series = linear_timeseries(start=0, length=10, freq=3)
        else:
            series = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=10, freq="3d"
            )
        subseries = (
            series.copy()
            .drop_after(series.time_index[-1])
            .drop_before(series.time_index[1])
        )
        subsubseries = (
            subseries.copy()
            .drop_after(subseries.time_index[-1])
            .drop_before(subseries.time_index[1])
        )
        expected_bounds = (subsubseries.start_time(), subsubseries.end_time())
        assert (
            get_shared_times_bounds(series, subseries, subsubseries) == expected_bounds
        )

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_shared_times_bounds_identical_inputs(self, series_type):
        """
        Tests that `get_shared_times_bounds` correctly handles case where
        multiple copies of the same series is passed as an input; we expect
        the return bounds to just be the start and end times of that repeated
        series.
        """
        if series_type == "integer":
            series = linear_timeseries(start=0, length=5, freq=1)
        else:
            series = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=5, freq="d"
            )
        expected = (series.start_time(), series.end_time())
        assert get_shared_times_bounds(series) == expected
        assert get_shared_times_bounds(series, series) == expected
        assert get_shared_times_bounds(series, series, series) == expected

    def test_shared_times_bounds_unspecified_inputs(self):
        """
        Tests that `get_shared_times_bounds` correctly handles case unspecified
        inputs (i.e. `None`) are passed. If passed with a specified series, the
        `None` input should be ignored, meaning that the returned bounds should
        be the start and end times of the only specified series. If only `None`
        inputs are passed, `None` should be returned.
        """
        # `None` is passed alonside a non-`None` input:
        series = linear_timeseries(start=0, length=5, freq=1)
        expected = (series.start_time(), series.end_time())
        assert get_shared_times_bounds(series, None) == expected
        assert get_shared_times_bounds(None, series) == expected
        assert get_shared_times_bounds(None, series, None) == expected

        # `None` should be returned if no series specified:
        assert get_shared_times_bounds(None) is None
        assert get_shared_times_bounds(None, None, None) is None

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_shared_times_bounds_single_idx_overlap(self, series_type):
        """
        Tests that `get_shared_times_bounds` correctly handles cases
        where the bounds contains a single time index value.
        """
        # Pass multiple copies of timeseries with single time
        # value - bounds should be start time and end time of
        # this single-valued series:
        # `series_1` and `series_2` share only a single overlap point
        # at the end of `series_1`:
        if series_type == "integer":
            series = linear_timeseries(start=0, length=1, freq=1)
            series_1 = linear_timeseries(start=0, length=3, freq=1)
            series_2 = linear_timeseries(start=series_1.end_time(), length=2, freq=2)
        else:
            series = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=1, freq="d"
            )
            series_1 = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=3, freq="d"
            )
            series_2 = linear_timeseries(start=series_1.end_time(), length=2, freq="2d")
        assert get_shared_times_bounds(series, series) == (
            series.start_time(),
            series.end_time(),
        )
        assert get_shared_times_bounds(series_1, series_2) == (
            series_1.end_time(),
            series_2.start_time(),
        )

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_shared_times_bounds_no_overlap(self, series_type):
        """
        Tests that `get_shared_times_bounds` returns `None` when provided
        with two series that share no overlap.
        """
        # Have `series_2` begin after the end of `series_1`:
        if series_type == "integer":
            series_1 = linear_timeseries(start=0, length=5, freq=1)
            series_2 = linear_timeseries(
                start=series_1.end_time() + 1, length=6, freq=2
            )
        else:
            series_1 = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=5, freq="d"
            )
            series_2 = linear_timeseries(
                start=series_1.end_time() + pd.Timedelta("1d"), length=6, freq="2d"
            )
        assert get_shared_times_bounds(series_1, series_2) is None
        assert get_shared_times_bounds(series_2, series_1, series_2) is None

    def test_shared_times_bounds_different_time_idx_types_error(self):
        """
        Tests that `get_shared_times_bounds` throws correct error
        when a range time index series and a datetime index series
        are specified as inputs together.
        """
        series_1 = linear_timeseries(start=1, length=5, freq=1)
        series_2 = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=5, freq="d")
        with pytest.raises(ValueError) as err:
            get_shared_times_bounds(series_1, series_2)
        assert (
            "Specified series and/or times must all "
            "have the same type of `time_index` (i.e. all "
            "`pd.RangeIndex` or all `pd.DatetimeIndex`)."
        ) == str(err.value)

    def test_shared_times_bounds_empty_input(self):
        """
        Tests that `get_shared_times_bounds` returns `None` when
        handed a non-`None` input that has no timesteps.
        """
        series = linear_timeseries(start=0, length=0, freq=1)
        assert get_shared_times_bounds(series) is None
        assert get_shared_times_bounds(series.time_index) is None
        assert get_shared_times_bounds(series, series.time_index) is None
