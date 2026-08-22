import numpy as np
import pytest

from darts import TimeSeries
from darts.metrics import mae
from darts.models import LinearRegressionModel
from darts.utils.statistics import effective_sample_size


def _ar1(n: int, rho: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0, 1)
    return x


class TestEffectiveSampleSize:
    def test_independent_sample_is_worth_its_own_size(self):
        """Known value: with no autocorrelation, n_eff must come back as n."""
        n = 4000
        x = np.random.default_rng(0).normal(0, 1, n)

        assert effective_sample_size(x) == pytest.approx(n, rel=0.15)

    @pytest.mark.parametrize("rho", [0.3, 0.6, 0.85])
    def test_matches_the_analytic_ar1_value(self, rho):
        """Known value from a different formula than the one implemented.

        For an AR(1) process the effective size is ``n * (1 - rho) / (1 + rho)``.
        The implementation instead sums the estimated autocorrelations, so
        agreement here is two routes meeting, not one route restated.
        """
        n = 8000
        x = _ar1(n, rho)
        analytic = n * (1 - rho) / (1 + rho)

        assert effective_sample_size(x) == pytest.approx(analytic, rel=0.25)

    def test_never_exceeds_the_sample_size_and_never_drops_below_one(self):
        rng = np.random.default_rng(3)
        samples = (
            rng.normal(0, 1, 500),
            _ar1(500, 0.95),
            # rho < 0 alternates sign, which drives the correction below 1 and
            # would report *more* independent observations than were collected.
            # Negating a positively correlated series would not do this: the
            # autocorrelation of -x equals that of x.
            _ar1(2000, -0.7),
        )
        for x in samples:
            n_eff = effective_sample_size(x)
            assert 1.0 <= n_eff <= len(x)

    def test_anti_correlated_sample_is_not_credited_with_extra_observations(self):
        x = _ar1(2000, -0.7)

        assert effective_sample_size(x) == pytest.approx(len(x))

    def test_result_does_not_drift_when_max_lag_is_raised(self):
        """Long lags carry estimation noise, not information.

        Summing them instead of stopping at the first non-positive
        autocorrelation makes the answer depend on where the sum was cut off.
        """
        x = _ar1(3000, 0.6)
        at_default = effective_sample_size(x)
        at_high_lag = effective_sample_size(x, max_lag=400)

        assert at_high_lag == pytest.approx(at_default, rel=0.05)

    def test_more_correlation_means_fewer_independent_observations(self):
        sizes = [effective_sample_size(_ar1(4000, rho)) for rho in (0.0, 0.5, 0.9)]

        assert sizes[0] > sizes[1] > sizes[2]

    def test_constant_input_is_rejected_rather_than_silently_scored(self):
        with pytest.raises(ValueError):
            effective_sample_size(np.full(100, 2.5))

    @pytest.mark.parametrize("values", [[1.0], [], [1.0, np.nan, 2.0]])
    def test_unusable_input_raises(self, values):
        with pytest.raises(ValueError):
            effective_sample_size(np.asarray(values, dtype=float))

    def test_accepts_a_plain_sequence(self):
        x = list(_ar1(600, 0.5))

        assert effective_sample_size(x) == pytest.approx(
            effective_sample_size(np.asarray(x))
        )


class TestEffectiveSampleSizeOnBacktestWindows:
    """The reason this function is in darts: overlapping backtest windows."""

    @staticmethod
    def _window_metrics(stride: int) -> np.ndarray:
        n = 400
        t = np.arange(n)
        rng = np.random.default_rng(7)
        series = TimeSeries.from_values(
            10 * np.sin(2 * np.pi * t / 24) + 0.02 * t + rng.normal(0, 1, n)
        )
        model = LinearRegressionModel(lags=24)
        model.fit(series[: n // 2])
        return np.asarray(
            model.backtest(
                series,
                start=0.5,
                forecast_horizon=12,
                stride=stride,
                metric=mae,
                reduction=None,
                retrain=False,
                last_points_only=False,
                verbose=False,
            ),
            dtype=float,
        ).ravel()

    def test_overlapping_windows_are_worth_far_fewer_observations(self):
        values = self._window_metrics(stride=1)

        assert effective_sample_size(values) < len(values) / 3

    def test_non_overlapping_windows_keep_most_of_their_size(self):
        """Control: the shrinkage above is caused by the overlap, not by the
        metric or the series."""
        values = self._window_metrics(stride=12)

        assert effective_sample_size(values) > len(values) / 2
