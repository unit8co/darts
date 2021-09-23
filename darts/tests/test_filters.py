import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared

from ..models import GaussianProcessFilter
from darts.models.filtering.moving_average import MovingAverage
from darts.models.filtering.kalman_filter import KalmanFilter
from ..timeseries import TimeSeries
from ..utils import timeseries_generation as tg
from .base_test_class import DartsBaseTestClass


class FilterBaseTestClass(DartsBaseTestClass):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        np.random.seed(42)


class KalmanFilterTestCase(FilterBaseTestClass):

    def test_kalman(self):
        """ KalmanFilter test.
        Creates an increasing sequence of numbers, adds noise and 
        assumes the kalman filter predicts values closer to real values
        """
        testing_signal = np.arange(1, 5, 0.1)

        noise = np.random.normal(0, 0.7, testing_signal.shape)
        testing_signal_with_noise = testing_signal + noise

        df = pd.DataFrame(data=testing_signal_with_noise, columns=['signal'])
        testing_signal_with_noise_ts = TimeSeries.from_dataframe(df, value_cols=['signal'])

        kf = KalmanFilter(dim_x=1)
        filtered_ts = kf.filter(testing_signal_with_noise_ts, num_samples=1).univariate_values()
        
        noise_distance = testing_signal_with_noise - testing_signal
        prediction_distance = filtered_ts - testing_signal
        
        self.assertGreater(noise_distance.std(), prediction_distance.std())

    def test_kalman_multivariate(self):
        kf = KalmanFilter(dim_x=3)

        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        noise_ts = tg.gaussian_timeseries(length=30) * 0.1
        ts = sine_ts.stack(noise_ts)

        prediction = kf.filter(ts)

        self.assertEqual(prediction.width, 3)


class MovingAverageTestCase(FilterBaseTestClass):

    def test_moving_average_univariate(self):
        ma = MovingAverage(window=3, centered=False)
        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        sine_filtered = ma.filter(sine_ts)
        self.assertGreater(np.mean(np.abs(sine_ts.values())), np.mean(np.abs(sine_filtered.values())))

    def test_moving_average_multivariate(self):
        ma = MovingAverage(window=3)
        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        noise_ts = tg.gaussian_timeseries(length=30) * 0.1
        ts = sine_ts.stack(noise_ts)
        ts_filtered = ma.filter(ts)

        self.assertGreater(np.mean(np.abs(ts.values()[:, 0])), np.mean(np.abs(ts_filtered.values()[:, 0])))
        self.assertGreater(np.mean(np.abs(ts.values()[:, 1])), np.mean(np.abs(ts_filtered.values()[:, 1])))


class GaussianProcessFilterTestCase(FilterBaseTestClass):

    def test_gaussian_process(self):
        """ GaussianProcessFilter test.
        Creates a sine wave, adds noise and assumes the GP filter
        predicts values closer to real values
        """
        theta = np.radians(np.linspace(0, 360*5, 200))
        testing_signal = TimeSeries.from_values(np.cos(theta))

        noise = TimeSeries.from_values(np.random.normal(0, 0.4, len(testing_signal)))
        testing_signal_with_noise = testing_signal + noise

        kernel = ExpSineSquared()
        gpf = GaussianProcessFilter(kernel=kernel, alpha=0.2, n_restarts_optimizer=100, random_state=42)
        filtered_ts = gpf.filter(testing_signal_with_noise, num_samples=1)
        
        noise_diff = testing_signal_with_noise - testing_signal
        prediction_diff = filtered_ts - testing_signal
        self.assertGreater(noise_diff.values().std(), prediction_diff.values().std())

        filtered_ts_median = gpf.filter(testing_signal_with_noise, num_samples=100).quantile_timeseries()
        median_prediction_diff = filtered_ts_median - testing_signal
        self.assertGreater(noise_diff.values().std(), median_prediction_diff.values().std())

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


if __name__ == '__main__':
    KalmanFilterTestCase().test_kalman()
    MovingAverageTestCase().test_moving_average_univariate()
    MovingAverageTestCase().test_moving_average_multivariate()
    GaussianProcessFilterTestCase().test_gaussian_process()
