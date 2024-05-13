import numpy as np
import pandas as pd
import pytest
from pandas.tseries.offsets import CustomBusinessDay

from darts import TimeSeries
from darts.utils import _with_sanity_checks
from darts.utils.missing_values import extract_subseries
from darts.utils.ts_utils import retain_period_common_to_all
from darts.utils.utils import freqs, generate_index, n_steps_between


class TestUtils:
    def test_retain_period_common_to_all(self):
        seriesA = TimeSeries.from_times_and_values(
            pd.date_range("20000101", "20000110"), range(10)
        )
        seriesB = TimeSeries.from_times_and_values(
            pd.date_range("20000103", "20000108"), range(6)
        )
        seriesC = TimeSeries.from_times_and_values(
            pd.date_range("20000104", "20000112"), range(9)
        )
        seriesC = seriesC.stack(seriesC)

        common_series_list = retain_period_common_to_all([seriesA, seriesB, seriesC])

        # test start and end dates
        for common_series in common_series_list:
            assert common_series.start_time() == pd.Timestamp("20000104")
            assert common_series.end_time() == pd.Timestamp("20000108")

        # test widths
        assert common_series_list[0].width == 1
        assert common_series_list[1].width == 1
        assert common_series_list[2].width == 2

    def test_sanity_check_example(self):
        class Model:
            def _sanity_check(self, *args, **kwargs):
                if kwargs["b"] != kwargs["c"]:
                    raise (ValueError("b and c must be equal"))

            @_with_sanity_checks("_sanity_check")
            def fit(self, a, b=0, c=0):
                pass

        m = Model()

        # b != c should raise error
        with pytest.raises(ValueError):
            m.fit(5, b=3, c=2)

        # b == c should not raise error
        m.fit(5, b=2, c=2)

    def test_extract_subseries(self):
        start_times = ["2020-01-01", "2020-06-01", "2020-09-01"]
        end_times = ["2020-01-31", "2020-07-31", "2020-09-28"]

        # Form a series without missing values between start_times and end_times
        time_index = pd.date_range(periods=365, freq="D", start=start_times[0])
        pd_series = pd.Series(np.nan, index=time_index)
        for start, end in zip(start_times, end_times):
            pd_series[start:end] = 42
        series = TimeSeries.from_series(pd_series)

        subseries = extract_subseries(series)

        assert len(subseries) == len(start_times)
        for sub, start, end in zip(subseries, start_times, end_times):
            assert sub.start_time() == pd.to_datetime(start)
            assert sub.end_time() == pd.to_datetime(end)

        # Multivariate timeserie
        times = pd.date_range("20130206", "20130215")
        dataframe = pd.DataFrame(
            {
                "0": [1, 1, np.nan, 1, 2, 1, 1, 1, 1, 1],
                "1": [1, 1, np.nan, 1, 3, np.nan, np.nan, 1, 1, 1],
                "2": [1, 1, np.nan, 1, 4, np.nan, np.nan, np.nan, np.nan, 1],
            },
            index=times,
        )
        series = TimeSeries.from_dataframe(dataframe)

        # gaps is characterized by NaN in all the covariate columns
        subseries_all = extract_subseries(series, mode="all")
        assert len(subseries_all) == 2
        assert subseries_all[0] == series[:2]
        assert subseries_all[1] == series[3:]

        # gaps is characterized by NaN in any of the covariate columns
        subseries_any = extract_subseries(series, mode="any")
        assert len(subseries_any) == 3
        assert subseries_any[0] == series[:2]
        assert subseries_any[1] == series[3:5]
        assert subseries_any[2] == series[-1]

    @pytest.mark.parametrize(
        "config",
        [
            # regular date offset frequencies
            # day
            ("2000-01-02", "2000-01-01", None, None, "D", 0),  # empty time index
            ("2000-01-01", "2000-01-01", None, None, "D", 1),  # increasing time index
            ("2000-01-01", "2000-01-02", None, None, "D", 2),
            ("2000-01-01 01:00:00", "2000-01-02 02:00:00", None, None, "D", 2),
            # 2 * day
            ("2000-01-01", "1999-12-31", None, None, "2D", 0),
            ("2000-01-01", "2000-01-02", None, None, "2D", 1),
            ("2000-01-01", "2000-01-03", None, None, "2D", 2),
            # hour
            ("2000-01-01", "2000-01-01", None, None, "h", 1),
            ("2000-01-01", "2000-01-02", None, None, "h", 25),
            ("2000-01-01 01:00:00", "2000-01-02 02:00:00", None, None, "h", 26),
            ("2000-01-01 01:30:00", "2000-01-02 02:00:00", None, None, "h", 25),
            # 2 * hour
            ("2000-01-01", "2000-01-01", None, None, "2h", 1),
            ("2000-01-01", "2000-01-02", None, None, "2h", 13),
            ("2000-01-01 01:00:00", "2000-01-02 02:00:00", None, None, "2h", 13),
            ("2000-01-01 01:30:00", "2000-01-02 02:00:00", None, None, "2h", 13),
            # ambiguous frequencies
            # week-monday
            (
                "2000-01-01",  # saturday
                "2000-01-03",  # first monday
                "2000-01-03",  # first monday
                None,  # first wednesday
                "W-MON",
                1,
            ),
            # week-monday, start and end are not part of freq (two mondays)
            (
                "2000-01-01",  # saturday
                "2000-01-12",  # second wednesday
                "2000-01-03",  # first monday
                "2000-01-10",  # second monday
                "W-MON",
                2,
            ),
            # week-monday, start is part of freq (two mondays)
            (
                "2000-01-03",  # saturday
                "2000-01-12",  # second wednesday
                "2000-01-03",  # first monday
                "2000-01-10",  # second monday
                "W-MON",
                2,
            ),
            # week-monday, end is part of freq (one monday, end exclusive)
            (
                "2000-01-01",  # saturday
                "2000-01-10",  # second monday
                "2000-01-03",  # first monday
                None,  # second wednesday
                "W-MON",
                2,
            ),
            # week-monday, start and end are part of freq (one monday, end exclusive)
            (
                "2000-01-03",  # saturday
                "2000-01-10",  # second monday
                "2000-01-03",  # first monday
                None,  # second wednesday
                "W-MON",
                2,
            ),
            # month start
            ("2000-01-31", "2000-01-31", None, None, "MS", 0),
            ("2000-01-01", "2000-01-02", None, "2000-01-01", "MS", 1),
            ("2000-01-01", "2000-01-01", None, None, "MS", 1),
            ("2000-01-01", "2000-02-01", None, None, "MS", 2),
            ("2000-01-01", "2000-03-01", None, None, "MS", 3),
            # month end
            ("2000-01-01", "2000-01-02", None, None, freqs["ME"], 0),
            ("2000-01-31", "2000-02-29", None, None, freqs["ME"], 2),
            # 2 * months
            ("2000-01-01", "2000-01-01", None, None, "2MS", 1),
            ("2000-01-01", "2000-02-11", None, "2000-01-01", "2MS", 1),
            ("2000-01-01", "2000-03-01", None, None, "2MS", 2),
            ("2000-01-01", "2000-05-01", None, None, "2MS", 3),
            # quarter
            ("2000-01-01", "2000-04-01", None, None, "QS", 2),
            # year
            ("2000-01-01", "2001-04-01", None, "2001-01-01", "YS", 2),
            # 2*year
            ("2001-01-01", "2010-04-01", None, "2009-01-01", "2YS", 5),
            (0, -1, None, None, 1, 0),  # empty int index
            (0, -1, None, None, -1, 2),  # decreasing int index
            (0, 0, None, None, 1, 1),  # increasing int index
            (0, 0, None, None, 2, 1),
            (0, 1, None, None, 1, 2),
            (0, 1, None, None, 2, 1),
            (0, 2, None, None, 1, 3),
            (0, 2, None, None, 2, 2),
        ],
    )
    def test_generate_index_with_start_end(self, config):
        """Test that generate index returns the expected length, start, and end points
        using `start`, `end`, and `freq` as input.
        Also tests the reverse index generation with a negative frequency.
        """
        start, end, expected_start, expected_start_rev, freq, expected_n_steps = config
        if isinstance(start, str):
            start = pd.Timestamp(start)
            end = pd.Timestamp(end)
            expected_start = (
                pd.Timestamp(expected_start) if expected_start is not None else start
            )
            expected_start_rev = (
                pd.Timestamp(expected_start_rev)
                if expected_start_rev is not None
                else end
            )
            freq = pd.tseries.frequencies.to_offset(freq)
        else:
            expected_start = expected_start if expected_start is not None else start
            expected_start_rev = (
                expected_start_rev if expected_start_rev is not None else end
            )

        idx = generate_index(start=start, end=end, freq=freq)

        if isinstance(freq, int):
            assert idx.step == freq
        else:
            assert idx.freq == freq

        # idx has expected length
        assert len(idx) == expected_n_steps

        if expected_n_steps == 0:
            return

        # start and end are as expected
        assert idx[0] == expected_start
        assert idx[-1] == expected_start + freq * (expected_n_steps - 1)

        # reversed operations generates expected index
        idx_rev = generate_index(start=end, end=start, freq=-freq)
        assert idx_rev[0] == expected_start_rev
        assert idx_rev[-1] == expected_start_rev - freq * (expected_n_steps - 1)

    @pytest.mark.parametrize(
        "config",
        [
            # regular date offset frequencies
            # day
            ("2000-01-02", None, "D", 0),  # empty time index
            ("2000-01-01", "2000-01-01", "D", 1),  # increasing time index
            ("2000-01-01", "2000-01-02", "D", 2),
            ("2000-01-01 01:00:00", "2000-01-02 01:00:00", "D", 2),
            # 2 * day
            ("2000-01-01", None, "2D", 0),
            ("2000-01-01", "2000-01-01", "2D", 1),
            ("2000-01-01", "2000-01-03", "2D", 2),
            # hour
            ("2000-01-01", "2000-01-01", "h", 1),
            ("2000-01-01", "2000-01-02", "h", 25),
            ("2000-01-01 01:00:00", "2000-01-02 02:00:00", "h", 26),
            ("2000-01-01 01:30:00", "2000-01-02 01:30:00", "h", 25),
            # 2 * hour
            ("2000-01-01", "2000-01-01", "2h", 1),
            ("2000-01-01", "2000-01-02", "2h", 13),
            ("2000-01-01 01:00:00", "2000-01-02 01:00:00", "2h", 13),
            ("2000-01-01 01:30:00", "2000-01-02 01:30:00", "2h", 13),
            # ambiguous frequencies
            # week-monday
            (
                "2000-01-01",  # saturday
                "2000-01-03",  # first monday
                "W-MON",
                1,
            ),
            # week-monday, start is not part of freq (two mondays)
            (
                "2000-01-01",  # saturday
                "2000-01-10",  # second monday
                "W-MON",
                2,
            ),
            # week-monday, start and end are part of freq (two mondays)
            (
                "2000-01-03",  # saturday
                "2000-01-10",  # second monday
                "W-MON",
                2,
            ),
            # month start
            ("2000-01-31", None, "MS", 0),
            ("2000-01-01", "2000-01-01", "MS", 1),
            ("2000-01-01", "2000-02-01", "MS", 2),
            ("2000-01-01", "2000-03-01", "MS", 3),
            # month end
            ("2000-01-01", None, freqs["ME"], 0),
            ("2000-01-31", "2000-02-29", freqs["ME"], 2),
            # 2 * months
            ("2000-01-01", "2000-01-01", "2MS", 1),
            ("2000-01-01", "2000-03-01", "2MS", 2),
            ("2000-01-01", "2000-05-01", "2MS", 3),
            # quarter
            ("2000-01-01", "2000-04-01", "QS", 2),
            # year
            ("2000-01-01", "2001-01-01", "YS", 2),
            # 2*year
            ("2001-01-01", "2009-01-01", "2YS", 5),
            (0, None, 1, 0),  # empty int index
            (0, -1, -1, 2),  # decreasing int index
            (0, 0, 1, 1),  # increasing int index
            (0, 0, 2, 1),
            (0, 1, 1, 2),
            (0, 2, 1, 3),
            (0, 2, 2, 2),
        ],
    )
    def test_generate_index_with_start_length(self, config):
        """Test that generate index returns the expected length, start, and end points
        using `start`, `length`, and `freq` as input.
        """
        start, expected_end, freq, n_steps = config
        if isinstance(start, str):
            freq = pd.tseries.frequencies.to_offset(freq)
            start = pd.Timestamp(start)
            expected_end = (
                pd.Timestamp(expected_end) if expected_end is not None else None
            )
        idx = generate_index(start=start, length=n_steps, freq=freq)
        assert len(idx) == n_steps
        if n_steps == 0:
            return

        assert idx[-1] == expected_end
        assert idx[0] == expected_end - (n_steps - 1) * freq

    @pytest.mark.parametrize(
        "config",
        [
            # regular date offset frequencies
            # day
            (None, "2000-01-02", "D", 0),  # empty time index
            ("2000-01-01", "2000-01-01", "D", 1),  # increasing time index
            ("2000-01-01", "2000-01-02", "D", 2),
            ("2000-01-01 01:00:00", "2000-01-02 01:00:00", "D", 2),
            # 2 * day
            (None, "2000-01-01", "2D", 0),
            ("2000-01-01", "2000-01-01", "2D", 1),
            ("2000-01-01", "2000-01-03", "2D", 2),
            # hour
            ("2000-01-01", "2000-01-01", "h", 1),
            ("2000-01-01", "2000-01-02", "h", 25),
            ("2000-01-01 01:00:00", "2000-01-02 02:00:00", "h", 26),
            ("2000-01-01 01:30:00", "2000-01-02 01:30:00", "h", 25),
            # 2 * hour
            ("2000-01-01", "2000-01-01", "2h", 1),
            ("2000-01-01", "2000-01-02", "2h", 13),
            ("2000-01-01 01:00:00", "2000-01-02 01:00:00", "2h", 13),
            ("2000-01-01 01:30:00", "2000-01-02 01:30:00", "2h", 13),
            # ambiguous frequencies
            # week-monday, end is not part of freq
            (
                "1999-12-27",  # saturday
                "2000-01-02",  # first monday
                "W-MON",
                1,
            ),
            # week-monday, end is part of freq
            (
                "2000-01-03",  # saturday
                "2000-01-10",  # second monday
                "W-MON",
                2,
            ),
            # month start
            (None, "2000-01-31", "MS", 0),
            ("2000-01-01", "2000-01-01", "MS", 1),
            ("2000-01-01", "2000-02-01", "MS", 2),
            ("2000-01-01", "2000-03-01", "MS", 3),
            # month end
            (None, "2000-01-01", freqs["ME"], 0),
            ("2000-01-31", "2000-02-29", freqs["ME"], 2),
            # 2 * months
            ("2000-01-01", "2000-01-01", "2MS", 1),
            ("2000-01-01", "2000-03-01", "2MS", 2),
            ("2000-01-01", "2000-05-01", "2MS", 3),
            # quarter
            ("2000-01-01", "2000-04-01", "QS", 2),
            # year
            ("2000-01-01", "2001-01-01", "YS", 2),
            # 2*year
            ("2001-01-01", "2009-01-01", "2YS", 5),
            (None, 0, 1, 0),  # empty int index
            (0, -1, -1, 2),  # decreasing int index
            (0, 0, 1, 1),  # increasing int index
            (0, 0, 2, 1),
            (0, 1, 1, 2),
            (0, 2, 1, 3),
            (0, 2, 2, 2),
        ],
    )
    def test_generate_index_with_end_length(self, config):
        """Test that generate index returns the expected length, start, and end points
        using `end`, `length`, and `freq` as input.
        """
        expected_start, end, freq, n_steps = config

        if isinstance(end, str):
            freq = pd.tseries.frequencies.to_offset(freq)
            expected_start = (
                pd.Timestamp(expected_start) if expected_start is not None else None
            )
            end = pd.Timestamp(end)
        idx = generate_index(end=end, length=n_steps, freq=freq)
        assert len(idx) == n_steps
        if n_steps == 0:
            return

        assert idx[0] == expected_start
        assert idx[-1] == expected_start + (n_steps - 1) * freq

    @pytest.mark.parametrize(
        "config",
        [
            # regular date offset frequencies
            # day
            ("2000-01-01", "2000-01-01", "D", 0),
            ("2000-01-01", "2000-01-02", "D", 1),
            ("2000-01-01", "2005-02-05", "D", 1862),
            # 2*days
            ("2000-01-01", "2000-01-01", "2D", 0),
            ("2000-01-01", "2000-01-02", "2D", 0),
            ("2000-01-01", "2000-01-03", "2D", 1),
            # hour
            ("2000-01-01", "2000-01-01", "h", 0),
            ("2000-01-01", "2000-01-01 06:00:00", "h", 6),
            ("2000-01-01", "2000-01-02", "h", 24),
            # ambiguous frequencies
            # week-monday, start and end are not part of freq (two mondays)
            (
                "2000-01-01",  # saturday
                "2000-01-12",  # second wednesday
                "W-MON",
                2,
            ),
            # week-monday, start is part of freq (two mondays)
            (
                "2000-01-03",  # monday
                "2000-01-12",  # second wednesday
                "W-MON",
                2,
            ),
            # week-monday, end is part of freq (one monday, end exclusive)
            (
                "2000-01-01",  # saturday
                "2000-01-10",  # second monday
                "W-MON",
                1,
            ),
            # week-monday, start and end are part of freq (one monday, end exclusive)
            (
                "2000-01-03",  # saturday
                "2000-01-10",  # second monday
                "W-MON",
                1,
            ),
            # month
            ("2000-01-01", "2000-01-02", freqs["ME"], 0),
            ("2000-01-01", "2000-01-01", freqs["ME"], 0),
            ("2000-01-01", "2000-02-01", freqs["ME"], 1),
            ("2000-01-01", "2000-03-01", freqs["ME"], 2),
            # 2 * months
            ("2000-01-01", "2000-01-01", "2" + freqs["ME"], 0),
            ("2000-01-01", "2000-02-11", "2" + freqs["ME"], 0),
            ("2000-01-01", "2000-03-01", "2" + freqs["ME"], 1),
            ("2000-01-01", "2000-05-01", "2" + freqs["ME"], 2),
            # quarter
            ("2000-01-01", "2000-04-01", freqs["QE"], 1),
            # year
            ("2000-01-01", "2001-04-01", freqs["YE"], 1),
            # 2*year
            ("2000-01-01", "2010-04-01", "2" + freqs["YE"], 5),
            # custom frequencies
            # business day
            (
                "2000-01-01",  # saturday (no business)
                "2000-01-01",
                CustomBusinessDay(weekmask="Mon Tue Wed Thu Fri"),
                0,
            ),
            (
                "2000-01-01",  # saturday (no business)
                "2000-01-02",  # sunday (no business)
                CustomBusinessDay(weekmask="Mon Tue Wed Thu Fri"),
                0,
            ),
            (
                "2000-01-01",  # saturday (no business)
                "2000-01-03",  # monday (first business day)
                CustomBusinessDay(weekmask="Mon Tue Wed Thu Fri"),
                0,
            ),
            (
                "2000-01-01",  # saturday (no business)
                "2000-01-08",  # second saturday (first business day)
                CustomBusinessDay(weekmask="Mon Tue Wed Thu Fri"),
                4,
            ),
            (
                "2000-01-03",  # monday
                "2000-01-07",  # friday
                CustomBusinessDay(weekmask="Mon Tue Wed Thu Fri"),
                4,
            ),
            # 2 * business days
            (
                "2000-01-01",  # saturday (no business)
                "2000-01-08",  # second saturday (first business day)
                2 * CustomBusinessDay(weekmask="Mon Tue Wed Thu Fri"),
                2,
            ),
            # integer steps/frequencies
            (0, -1, 1, -1),
            (0, 0, 1, 0),
            (0, 0, 2, 0),
            (0, 1, 1, 1),
            (0, 1, 2, 0),
            (0, 2, 1, 2),
            (0, 2, 2, 1),
        ],
    )
    def test_n_steps_between(self, config):
        """Test the number of frequency steps/periods between two time steps."""
        start, end, freq, expected_n_steps = config
        if isinstance(start, str):
            start = pd.Timestamp(start)
            end = pd.Timestamp(end)
            freq = pd.tseries.frequencies.to_offset(freq)
        n_steps = n_steps_between(end=end, start=start, freq=freq)
        assert n_steps == expected_n_steps
        n_steps_reversed = n_steps_between(end=start, start=end, freq=freq)
        assert n_steps_reversed == -expected_n_steps
