import numpy as np
import pandas as pd
from nfoursid import kalman, state_space
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared

from darts import TimeSeries
from darts.metrics import rmse
from darts.models.filtering.gaussian_process_filter import GaussianProcessFilter
from darts.models.filtering.kalman_filter import KalmanFilter
from darts.models.filtering.moving_average import MovingAverage
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg


class FilterBaseTestClass(DartsBaseTestClass):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        np.random.seed(42)


class KalmanFilterTestCase(FilterBaseTestClass):
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

        self.assertGreater(noise_distance.std(), prediction_distance.std())
        self.assertEqual(filtered_ts.width, 1)
        self.assertEqual(filtered_ts.n_samples, 1)

    def test_kalman_covariates(self):
        kf = KalmanFilter(dim_x=2)

        series = tg.sine_timeseries(length=30, value_frequency=0.1)
        covariates = -series.copy()

        kf.fit(series, covariates=covariates)
        prediction = kf.filter(series, covariates=covariates)

        self.assertEqual(prediction.width, 1)
        self.assertEqual(prediction.n_samples, 1)

    def test_kalman_covariates_multivariate(self):
        kf = KalmanFilter(dim_x=3)

        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        noise_ts = tg.gaussian_timeseries(length=30) * 0.1
        series = sine_ts.stack(noise_ts)

        covariates = -series.copy()

        kf.fit(series, covariates=covariates)
        prediction = kf.filter(series, covariates=covariates)

        self.assertEqual(kf.dim_u, 2)
        self.assertEqual(prediction.width, 2)
        self.assertEqual(prediction.n_samples, 1)

    def test_kalman_multivariate(self):
        kf = KalmanFilter(dim_x=3)

        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        noise_ts = tg.gaussian_timeseries(length=30) * 0.1
        series = sine_ts.stack(noise_ts)

        kf.fit(series)
        prediction = kf.filter(series)

        self.assertEqual(prediction.width, 2)
        self.assertEqual(prediction.n_samples, 1)

    def test_kalman_samples(self):
        kf = KalmanFilter(dim_x=1)

        series = tg.sine_timeseries(length=30, value_frequency=0.1)

        kf.fit(series)
        prediction = kf.filter(series, num_samples=10)

        self.assertEqual(prediction.width, 1)
        self.assertEqual(prediction.n_samples, 10)

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
        self.assertLess(rmse(filtered_series, sine), 0.1)

    def test_kalman_given_kf(self):
        nfoursid_ss = state_space.StateSpace(
            a=np.eye(2), b=np.ones((2, 1)), c=np.ones((1, 2)), d=np.ones((1, 1))
        )
        nfoursid_kf = kalman.Kalman(nfoursid_ss, np.ones((3, 3)) * 0.1)
        kf = KalmanFilter(dim_x=1, kf=nfoursid_kf)

        series = tg.sine_timeseries(length=30, value_frequency=0.1)

        prediction = kf.filter(series, covariates=-series.copy())

        self.assertEqual(kf.dim_u, 1)
        self.assertEqual(kf.dim_x, 2)
        self.assertEqual(prediction.width, 1)
        self.assertEqual(prediction.n_samples, 1)


class MovingAverageTestCase(FilterBaseTestClass):
    def test_moving_average_univariate(self):
        ma = MovingAverage(window=3, centered=False)
        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        sine_filtered = ma.filter(sine_ts)
        self.assertGreater(
            np.mean(np.abs(sine_ts.values())), np.mean(np.abs(sine_filtered.values()))
        )

    def test_moving_average_multivariate(self):
        ma = MovingAverage(window=3)
        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        noise_ts = tg.gaussian_timeseries(length=30) * 0.1
        ts = sine_ts.stack(noise_ts)
        ts_filtered = ma.filter(ts)

        self.assertGreater(
            np.mean(np.abs(ts.values()[:, 0])),
            np.mean(np.abs(ts_filtered.values()[:, 0])),
        )
        self.assertGreater(
            np.mean(np.abs(ts.values()[:, 1])),
            np.mean(np.abs(ts_filtered.values()[:, 1])),
        )


class GaussianProcessFilterTestCase(FilterBaseTestClass):
    def test_gaussian_process(self):
        """GaussianProcessFilter test.
        Creates a sine wave, adds noise and assumes the GP filter
        predicts values closer to real values
        """
        theta = np.radians(np.linspace(0, 360 * 5, 200))
        testing_signal = TimeSeries.from_values(np.cos(theta))

        noise = TimeSeries.from_values(np.random.normal(0, 0.4, len(testing_signal)))
        testing_signal_with_noise = testing_signal + noise

        kernel = ExpSineSquared()
        gpf = GaussianProcessFilter(
            kernel=kernel, alpha=0.2, n_restarts_optimizer=100, random_state=42
        )
        filtered_ts = gpf.filter(testing_signal_with_noise, num_samples=1)

        noise_diff = testing_signal_with_noise - testing_signal
        prediction_diff = filtered_ts - testing_signal
        self.assertGreater(noise_diff.values().std(), prediction_diff.values().std())

        filtered_ts_median = gpf.filter(
            testing_signal_with_noise, num_samples=100
        ).quantile_timeseries()
        median_prediction_diff = filtered_ts_median - testing_signal
        self.assertGreater(
            noise_diff.values().std(), median_prediction_diff.values().std()
        )

    def test_gaussian_process_multivariate(self):
        gpf = GaussianProcessFilter()

        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        noise_ts = tg.gaussian_timeseries(length=30) * 0.1
        ts = sine_ts.stack(noise_ts)

        prediction = gpf.filter(ts)

        self.assertEqual(prediction.width, 2)

    def test_gaussian_process_missing_values(self):
        ts = TimeSeries.from_values(np.ones(6))

        gpf = GaussianProcessFilter(RBF())
        filtered_values = gpf.filter(ts).values()
        np.testing.assert_allclose(filtered_values, np.ones_like(filtered_values))


if __name__ == "__main__":
    KalmanFilterTestCase().test_kalman()
    KalmanFilterTestCase().test_kalman_multivariate()
    KalmanFilterTestCase().test_kalman_covariates()
    KalmanFilterTestCase().test_kalman_covariates_multivariate()
    KalmanFilterTestCase().test_kalman_samples()
    KalmanFilterTestCase().test_kalman_given_kf()
    MovingAverageTestCase().test_moving_average_univariate()
    MovingAverageTestCase().test_moving_average_multivariate()
    GaussianProcessFilterTestCase().test_gaussian_process()
