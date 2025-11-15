from math import lcm

import pandas as pd
import pytest

from darts.utils.data.tabularization import get_shared_times
from darts.utils.timeseries_generation import linear_timeseries


class TestGetSharedTimes:
    """
    Tests `get_shared_times` function defined in `darts.utils.data.tabularization`.
    """

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_shared_times_equal_freq(self, series_type):
        """
        Tests that `get_shared_times` correctly handles time index series that are of equal frequency.
        """
        # `series_1` begins before `series_2` does and ends
        # before `series_2` does, and `series_2` begins before
        # `series_3` does and ends before `series_3` does:
        if series_type == "integer":
            series_1 = linear_timeseries(start=1, end=11, freq=2)
            series_2 = linear_timeseries(start=3, end=13, freq=2)
            series_3 = linear_timeseries(start=5, end=15, freq=2)
        else:
            series_1 = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), end=pd.Timestamp("1/11/2000"), freq="2d"
            )
            series_2 = linear_timeseries(
                start=pd.Timestamp("1/3/2000"), end=pd.Timestamp("1/13/2000"), freq="2d"
            )
            series_3 = linear_timeseries(
                start=pd.Timestamp("1/5/2000"), end=pd.Timestamp("1/15/2000"), freq="2d"
            )

        # Intersection of a single time index is just the original time index:
        assert series_1.time_index.equals(get_shared_times(series_1))
        assert series_2.time_index.equals(get_shared_times(series_2))
        assert series_3.time_index.equals(get_shared_times(series_3))

        # Intersection of two time indices begins at start time of later series
        # and stops at end time of earlier series.
        # Since `series_1` is before `series_2`:
        expected_12 = linear_timeseries(
            start=series_2.start_time(), end=series_1.end_time(), freq=series_1.freq
        )
        assert expected_12.time_index.equals(get_shared_times(series_1, series_2))
        # Since `series_2` is before `series_3`:
        expected_23 = linear_timeseries(
            start=series_3.start_time(), end=series_2.end_time(), freq=series_2.freq
        )
        assert expected_23.time_index.equals(get_shared_times(series_2, series_3))
        # Since `series_1` is before `series_3`:
        expected_13 = linear_timeseries(
            start=series_3.start_time(), end=series_1.end_time(), freq=series_1.freq
        )
        assert expected_13.time_index.equals(get_shared_times(series_1, series_3))

        # Intersection of all three time series should begin at start of series_3 (i.e.
        # the last series to begin) and end at the end of series_1 (i.e. the first series
        # to end):
        expected_123 = linear_timeseries(
            start=series_3.start_time(), end=series_1.end_time(), freq=series_1.freq
        )
        assert expected_123.time_index.equals(
            get_shared_times(series_1, series_2, series_3)
        )

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_shared_times_unequal_freq(self, series_type):
        """
        Tests that `get_shared_times` correctly handles time index series that are of different frequencies.
        """
        # `series_1` begins before `series_2` does and ends
        # before `series_2` does, and `series_2` begins before
        # `series_3` does and ends before `series_3` does. Each
        # series is of a different frequency:
        if series_type == "integer":
            series_1 = linear_timeseries(start=1, end=11, freq=1)
            series_2 = linear_timeseries(start=3, end=13, freq=2)
            series_3 = linear_timeseries(start=5, end=17, freq=3)
            freq_12 = lcm(series_1.freq, series_2.freq)
            freq_23 = lcm(series_2.freq, series_3.freq)
            freq_13 = lcm(series_1.freq, series_3.freq)
            freq_123 = lcm(series_1.freq, series_2.freq, series_3.freq)
        else:
            series_1 = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), end=pd.Timestamp("1/11/2000"), freq="2d"
            )
            series_2 = linear_timeseries(
                start=pd.Timestamp("1/3/2000"), end=pd.Timestamp("1/13/2000"), freq="2d"
            )
            series_3 = linear_timeseries(
                start=pd.Timestamp("1/5/2000"), end=pd.Timestamp("1/15/2000"), freq="2d"
            )
            freq_12 = f"{lcm(series_1.freq.n, series_2.freq.n)}d"
            freq_23 = f"{lcm(series_2.freq.n, series_3.freq.n)}d"
            freq_13 = f"{lcm(series_1.freq.n, series_3.freq.n)}d"
            freq_123 = f"{lcm(series_1.freq.n, series_2.freq.n, series_3.freq.n)}d"
        # Intersection of a single time index is just the original time index:
        assert series_1.time_index.equals(get_shared_times(series_1))
        assert series_2.time_index.equals(get_shared_times(series_2))
        assert series_3.time_index.equals(get_shared_times(series_3))

        # Intersection of two time indices begins at start time of later series
        # and stops at end time of earlier series. The frequency of the intersection
        # is the lowest common multiple between the frequencies of the two series:

        # `series_1` is before `series_2`:
        expected_12 = linear_timeseries(
            start=series_2.start_time(),
            end=series_1.end_time(),
            freq=freq_12,
        )
        # `linear_timeseries` may have added point beyond specified `end`;
        # remove this point if present:
        if expected_12.time_index[-1] > series_1.end_time():
            expected_12 = expected_12.drop_after(expected_12.time_index[-1])
        assert expected_12.time_index.equals(get_shared_times(series_1, series_2))
        # `series_2` is before `series_3`:
        expected_23 = linear_timeseries(
            start=series_3.start_time(),
            end=series_2.end_time(),
            freq=freq_23,
        )
        # `linear_timeseries` may have added point beyond specified `end`;
        # remove this point if present:
        if expected_23.time_index[-1] > series_2.end_time():
            expected_23 = expected_23.drop_after(expected_23.time_index[-1])
        assert expected_23.time_index.equals(get_shared_times(series_2, series_3))
        # `series_1` is before `series_3`:
        expected_13 = linear_timeseries(
            start=series_3.start_time(),
            end=series_1.end_time(),
            freq=freq_13,
        )
        # `linear_timeseries` may have added point beyond specified `end`;
        # remove this point if present:
        if expected_13.time_index[-1] > series_1.end_time():
            expected_13 = expected_13.drop_after(expected_13.time_index[-1])
        assert expected_13.time_index.equals(get_shared_times(series_1, series_3))

        # Intersection of all three time series should begin at start of series_3 (i.e.
        # the last series to begin) and end at the end of series_1 (i.e. the first series
        # to end). The frequency of the intersection should be the lowest common multiple
        # shared by all three frequencies:
        expected_123 = linear_timeseries(
            start=series_3.start_time(),
            end=series_1.end_time(),
            freq=freq_123,
        )
        if expected_123.time_index[-1] > series_1.end_time():
            expected_123 = expected_123.drop_after(expected_123.time_index[-1])
        assert expected_123.time_index.equals(
            get_shared_times(series_1, series_2, series_3)
        )

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_shared_times_no_overlap(self, series_type):
        """
        Tests that `get_shared_times` returns `None` when supplied time index series share no temporal overlap.
        """
        # Define `series_2` so that it starts after `series_1` ends:
        if series_type == "integer":
            series_1 = linear_timeseries(start=1, end=11, freq=2)
            series_2 = linear_timeseries(
                start=series_1.end_time() + 1, length=5, freq=3
            )
        else:
            series_1 = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), end=pd.Timestamp("1/11/2000"), freq="2d"
            )
            series_2 = linear_timeseries(
                start=series_1.end_time() + pd.Timedelta(1, "d"), length=5, freq="3d"
            )
        assert get_shared_times(series_1, series_2) is None
        assert get_shared_times(series_1, series_1, series_2) is None
        assert get_shared_times(series_1, series_2, series_2) is None
        assert get_shared_times(series_1, series_1, series_2, series_2) is None

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_shared_times_single_time_point_overlap(self, series_type):
        """
        Tests that `get_shared_times` returns correct bounds when given time index series that overlap
        at a single time point.
        """
        # `series_1` and `series_2` only overlap at `series_1.end_time()`:
        if series_type == "integer":
            series_1 = linear_timeseries(start=1, end=11, freq=2)
            series_2 = linear_timeseries(start=series_1.end_time(), length=5, freq=3)
        else:
            series_1 = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), end=pd.Timestamp("1/11/2000"), freq="2d"
            )
            series_2 = linear_timeseries(start=series_1.end_time(), length=5, freq="3d")
        overlap_val = series_1.end_time()
        assert get_shared_times(series_1, series_2) == overlap_val
        assert get_shared_times(series_1, series_1, series_2) == overlap_val
        assert get_shared_times(series_1, series_2, series_2) == overlap_val
        assert get_shared_times(series_1, series_1, series_2, series_2) == overlap_val

    @pytest.mark.parametrize(
        "series_type",
        ["datetime", "integer"],
    )
    def test_shared_times_identical_inputs(self, series_type):
        """
        Tests that `get_shared_times` correctly handles case where
        multiple copies of same time index timeseries is passed;
        we expect that the unaltered time index of the series is returned.
        """
        if series_type == "integer":
            series = linear_timeseries(start=0, length=5, freq=1)
        else:
            series = linear_timeseries(
                start=pd.Timestamp("1/1/2000"), length=5, freq="d"
            )
        assert series.time_index.equals(get_shared_times(series))
        assert series.time_index.equals(get_shared_times(series, series))
        assert series.time_index.equals(get_shared_times(series, series, series))

    def test_shared_times_unspecified_inputs(self):
        """
        Tests that `get_shared_times` correctly handles unspecified
        (i.e. `None` value) inputs. If `None` is passed with another
        series/time index, then `None` should be ignored and the time
        index of the other series should be returned. If only `None`
        values are passed, `None` should be returned.
        """
        # Pass `None` with series/time index input:
        series = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=5, freq="d")
        assert get_shared_times(None) is None
        assert series.time_index.equals(get_shared_times(series, None))
        assert series.time_index.equals(get_shared_times(None, series, None))
        assert series.time_index.equals(get_shared_times(None, series.time_index, None))
        # Pass only `None` as input:
        assert get_shared_times(None) is None

    def test_shared_times_time_index_inputs(self):
        """
        Tests that `get_shared_times` can accept time index
        inputs instead of `TimeSeries` inputs; combinations
        of time index and `TimeSeries` inputs are also tested.
        """
        series_1 = linear_timeseries(start=0, end=10, freq=1)
        series_2 = linear_timeseries(start=0, end=20, freq=2)
        # `stop=10+1` since `stop` is exclusive
        intersection = pd.RangeIndex(
            start=series_2.start_time(), stop=series_1.end_time() + 1, step=2
        )
        assert intersection.equals(get_shared_times(series_1.time_index, series_2))
        assert intersection.equals(get_shared_times(series_1, series_2.time_index))
        assert intersection.equals(
            get_shared_times(series_1.time_index, series_2.time_index)
        )

    def test_shared_times_empty_input(self):
        """
        Tests that `get_shared_times` returns `None` when
        given a non-`None` input with no timesteps.
        """
        series = linear_timeseries(start=0, length=0, freq=1)
        assert get_shared_times(series) is None
        assert get_shared_times(series.time_index) is None
        assert get_shared_times(series, series.time_index) is None

    def test_shared_times_different_time_index_types_error(self):
        """
        Tests that `get_shared_times` throws correct error when
        provided with series with different types of time indices.
        """
        series_1 = linear_timeseries(start=1, length=5, freq=1)
        series_2 = linear_timeseries(start=pd.Timestamp("1/1/2000"), length=5, freq="d")
        with pytest.raises(ValueError) as err:
            get_shared_times(series_1, series_2)
        assert (
            "Specified series and/or times must all "
            "have the same type of `time_index` (i.e. all "
            "`pd.RangeIndex` or all `pd.DatetimeIndex`)."
        ) == str(err.value)
