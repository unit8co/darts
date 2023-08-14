import numpy as np
import pandas as pd
from nfoursid import kalman, state_space
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared

from darts import TimeSeries
from darts.metrics import rmse
from darts.models.filtering.gaussian_process_filter import GaussianProcessFilter
from darts.models.filtering.kalman_filter import KalmanFilter
from darts.models.filtering.moving_average_filter import MovingAverageFilter
from darts.utils import timeseries_generation as tg


class FilterBaseTestClass:
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        np.random.seed(42)


class TestKalmanFilter(FilterBaseTestClass):
    def test_kalman(self):
        """KalmanFilter test.
        Creates an increasing sequence of numbers, adds noise and
        assumes the kalman filter predicts values closer to real values
        """
        testing_signal = np.arange(1, 5, 0.1)

        noise = np.random.normal(0, 0.7, testing_signal.shape)
        testing_signal_with_noise = testing_signal + noise

        df = pd.DataFrame(data=testing_signal_with_noise, columns=["signal"])
        testing_signal_with_noise_ts = TimeSeries.from_dataframe(
            df, value_cols=["signal"]
        )

        kf = KalmanFilter(dim_x=1)
        kf.fit(testing_signal_with_noise_ts)
        filtered_ts = kf.filter(testing_signal_with_noise_ts, num_samples=1)
        filtered_values = filtered_ts.univariate_values()

        noise_distance = testing_signal_with_noise - testing_signal
        prediction_distance = filtered_values - testing_signal

        assert noise_distance.std() > prediction_distance.std()
        assert filtered_ts.width == 1
        assert filtered_ts.n_samples == 1

    def test_kalman_covariates(self):
        kf = KalmanFilter(dim_x=2)

        series = tg.sine_timeseries(length=30, value_frequency=0.1)
        covariates = -series.copy()

        kf.fit(series, covariates=covariates)
        prediction = kf.filter(series, covariates=covariates)

        assert prediction.width == 1
        assert prediction.n_samples == 1

    def test_kalman_covariates_multivariate(self):
        kf = KalmanFilter(dim_x=3)

        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        noise_ts = tg.gaussian_timeseries(length=30) * 0.1
        series = sine_ts.stack(noise_ts)

        covariates = -series.copy()

        kf.fit(series, covariates=covariates)
        prediction = kf.filter(series, covariates=covariates)

        assert kf.dim_u == 2
        assert prediction.width == 2
        assert prediction.n_samples == 1

    def test_kalman_multivariate(self):
        kf = KalmanFilter(dim_x=3)

        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        noise_ts = tg.gaussian_timeseries(length=30) * 0.1
        series = sine_ts.stack(noise_ts)

        kf.fit(series)
        prediction = kf.filter(series)

        assert prediction.width == 2
        assert prediction.n_samples == 1

    def test_kalman_samples(self):
        kf = KalmanFilter(dim_x=1)

        series = tg.sine_timeseries(length=30, value_frequency=0.1)

        kf.fit(series)
        prediction = kf.filter(series, num_samples=10)

        assert prediction.width == 1
        assert prediction.n_samples == 10

    def test_kalman_missing_values(self):
        sine = tg.sine_timeseries(
            length=100, value_frequency=0.05
        ) + 0.1 * tg.gaussian_timeseries(length=100)
        values = sine.values()
        values[20:22] = np.nan
        values[28:40] = np.nan
        sine_holes = TimeSeries.from_values(values)
        sine = TimeSeries.from_values(sine.values())

        kf = KalmanFilter(dim_x=2)
        kf.fit(sine_holes[-50:])  # fit on the part with no holes

        # reconstructruction should succeed
        filtered_series = kf.filter(sine_holes, num_samples=100)

        # reconstruction error should be sufficiently small
        assert rmse(filtered_series, sine) < 0.1

    def test_kalman_given_kf(self):
        nfoursid_ss = state_space.StateSpace(
            a=np.eye(2), b=np.ones((2, 1)), c=np.ones((1, 2)), d=np.ones((1, 1))
        )
        nfoursid_kf = kalman.Kalman(nfoursid_ss, np.ones((3, 3)) * 0.1)
        kf = KalmanFilter(dim_x=1, kf=nfoursid_kf)

        series = tg.sine_timeseries(length=30, value_frequency=0.1)

        prediction = kf.filter(series, covariates=-series.copy())

        assert kf.dim_u == 1
        assert kf.dim_x == 2
        assert prediction.width == 1
        assert prediction.n_samples == 1


class TestMovingAverage(FilterBaseTestClass):
    def test_moving_average_univariate(self):
        ma = MovingAverageFilter(window=3, centered=False)
        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        sine_filtered = ma.filter(sine_ts)
        assert np.mean(np.abs(sine_ts.values())) > np.mean(
            np.abs(sine_filtered.values())
        )

    def test_moving_average_multivariate(self):
        ma = MovingAverageFilter(window=3)
        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        noise_ts = tg.gaussian_timeseries(length=30) * 0.1
        ts = sine_ts.stack(noise_ts)
        ts_filtered = ma.filter(ts)

        assert np.mean(np.abs(ts.values()[:, 0])) > np.mean(
            np.abs(ts_filtered.values()[:, 0])
        )
        assert np.mean(np.abs(ts.values()[:, 1])) > np.mean(
            np.abs(ts_filtered.values()[:, 1])
        )


class TestGaussianProcessFilter(FilterBaseTestClass):
    def test_gaussian_process(self):
        """GaussianProcessFilter test.
        Creates a sine wave, adds noise and assumes the GP filter
        predicts values closer to real values
        """
        theta = np.radians(np.linspace(0, 360 * 5, 200))
        testing_signal = TimeSeries.from_values(np.cos(theta))

        noise = TimeSeries.from_values(np.random.normal(0, 0.4, len(testing_signal)))
        testing_signal_with_noise = testing_signal + noise

        kernel = ExpSineSquared(length_scale_bounds=(1e-3, 1e3))
        gpf = GaussianProcessFilter(
            kernel=kernel, alpha=0.2, n_restarts_optimizer=10, random_state=42
        )
        filtered_ts = gpf.filter(testing_signal_with_noise, num_samples=1)

        noise_diff = testing_signal_with_noise - testing_signal
        prediction_diff = filtered_ts - testing_signal
        assert noise_diff.values().std() > prediction_diff.values().std()

        filtered_ts_median = gpf.filter(
            testing_signal_with_noise, num_samples=100
        ).quantile_timeseries()
        median_prediction_diff = filtered_ts_median - testing_signal
        assert noise_diff.values().std() > median_prediction_diff.values().std()

    def test_gaussian_process_multivariate(self):
        gpf = GaussianProcessFilter()

        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        noise_ts = tg.gaussian_timeseries(length=30) * 0.1
        ts = sine_ts.stack(noise_ts)

        prediction = gpf.filter(ts)

        assert prediction.width == 2

    def test_gaussian_process_missing_values(self):
        ts = TimeSeries.from_values(np.ones(6))

        kernel = RBF(length_scale_bounds=(1e-3, 1e10))
        gpf = GaussianProcessFilter(kernel=kernel)
        filtered_values = gpf.filter(ts).values()
        np.testing.assert_allclose(filtered_values, np.ones_like(filtered_values))


if __name__ == "__main__":
    TestKalmanFilter().test_kalman()
    TestKalmanFilter().test_kalman_multivariate()
    TestKalmanFilter().test_kalman_covariates()
    TestKalmanFilter().test_kalman_covariates_multivariate()
    TestKalmanFilter().test_kalman_samples()
    TestKalmanFilter().test_kalman_given_kf()
    TestMovingAverage().test_moving_average_univariate()
    TestMovingAverage().test_moving_average_multivariate()
    TestGaussianProcessFilter().test_gaussian_process()
