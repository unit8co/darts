import numpy as np

from darts.models.forecasting.fft import _find_relevant_timestamp_attributes
from darts.utils import timeseries_generation as tg
from darts.utils.utils import freqs


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
        self.helper_relevant_attributes(freqs["ME"], 150, [(12, {"month"})])

        # daily frequency
        self.helper_relevant_attributes(
            "D", 1000, [(365, {"month", "day"}), (30, {"day"}), (7, {"weekday"})]
        )

        # hourly frequency
        self.helper_relevant_attributes(
            freqs["h"],
            3000,
            [(730, {"day", "hour"}), (168, {"weekday", "hour"}), (24, {"hour"})],
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
