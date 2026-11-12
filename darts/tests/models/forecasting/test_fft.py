import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.models import FFT
from darts.models.forecasting.fft import (
    _crop_to_match_seasons,
    _find_relevant_timestamp_attributes,
)
from darts.utils import timeseries_generation as tg


class TestFFT:
    def helper_relevant_attributes(self, freq, length, period_attributes_tuples):
        # test random walk
        random_walk_ts = tg.random_walk_timeseries(freq=freq, length=length)
        assert _find_relevant_timestamp_attributes(random_walk_ts) == set()

        for period, relevant_attributes in period_attributes_tuples:
            # test seasonal period with no noise
            seasonal_ts = tg.sine_timeseries(
                freq=freq, value_frequency=1 / period, length=length
            )
            assert (
                _find_relevant_timestamp_attributes(seasonal_ts) == relevant_attributes
            ), "failed to recognize season in non-noisy timeseries"

            # test seasonal period with no noise
            seasonal_noisy_ts = seasonal_ts + tg.gaussian_timeseries(
                freq=freq, length=length
            )
            assert (
                _find_relevant_timestamp_attributes(seasonal_noisy_ts)
                == relevant_attributes
            ), "failed to recognize season in noisy timeseries"

    def test_find_relevant_timestamp_attributes(self):
        np.random.seed(0)

        # monthly frequency
        self.helper_relevant_attributes("ME", 150, [(12, {"month"})])

        # daily frequency
        self.helper_relevant_attributes(
            "D", 1000, [(365, {"month", "day"}), (30, {"day"}), (7, {"day_of_week"})]
        )

        # hourly frequency
        self.helper_relevant_attributes(
            "h",
            3000,
            [(730, {"day", "hour"}), (168, {"day_of_week", "hour"}), (24, {"hour"})],
        )

        # minutely frequency
        self.helper_relevant_attributes(
            "min",
            5000,
            [
                (1440, {"hour", "minute"}),
                (60, {"minute"}),
            ],
        )

    def test_crop_to_match_day_of_week(self):
        # daily series with 73 points: the first prediction point is a Tuesday
        idx = pd.date_range("2024-01-01", periods=73, freq="D")
        series = TimeSeries.from_times_and_values(idx, np.arange(73.0))
        pred_ts = series.end_time() + series.freq

        cropped = _crop_to_match_seasons(series, required_matches={"day_of_week"})
        assert len(cropped) < len(series)
        assert cropped.start_time().day_of_week == pred_ts.day_of_week

    def test_weekly_seasonality_is_aligned(self):
        idx = pd.date_range("2024-01-01", periods=73, freq="D")
        pattern = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 12.0])
        series = TimeSeries.from_times_and_values(idx, np.tile(pattern, 11)[:73])
        assert _find_relevant_timestamp_attributes(series) == {"day_of_week"}

        # the automatically detected weekly seasonality must crop the training series
        # like an explicit `required_matches={"day_of_week"}`
        model_auto = FFT()
        model_auto.fit(series)
        model_explicit = FFT(required_matches={"day_of_week"})
        model_explicit.fit(series)
        np.testing.assert_allclose(
            model_auto.predict(7).values(), model_explicit.predict(7).values()
        )

    @pytest.mark.parametrize("required_matches", [{"weekday"}, {"month", "foo"}])
    def test_invalid_required_matches(self, required_matches):
        with pytest.raises(ValueError, match="properties of `pd.Timestamp`"):
            FFT(required_matches=required_matches)

    @pytest.mark.parametrize(
        "freq, period, required_matches",
        [("D", 7, {"day_of_week"}), ("MS", 12, {"month"})],
    )
    def test_periodic_series_is_predicted_exactly(self, freq, period, required_matches):
        # the cropped training series must start at the same phase as the first prediction point
        n = 10 * period + 3
        idx = pd.date_range("2024-01-01", periods=n, freq=freq)
        pattern = np.arange(1.0, period + 1.0) ** 2
        series = TimeSeries.from_times_and_values(idx, np.tile(pattern, 11)[:n])
        expected = np.roll(pattern, -(n % period))

        model = FFT(nr_freqs_to_keep=None, required_matches=required_matches)
        model.fit(series)
        np.testing.assert_allclose(
            model.predict(period).values().ravel(), expected, atol=1e-8
        )
