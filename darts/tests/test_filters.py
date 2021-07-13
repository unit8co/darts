import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF

from ..models import GaussianProcessFilter
from ..models.filtering_model import MovingAverage
from ..models.kalman_filter import KalmanFilter
from ..timeseries import TimeSeries
from ..utils import timeseries_generation as tg
from .base_test_class import DartsBaseTestClass


class KalmanFilterTestCase(DartsBaseTestClass):

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


class MovingAverageTestCase(DartsBaseTestClass):

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


class GaussianProcessFilterTestCase(DartsBaseTestClass):

    def test_gaussian_process(self):
        """ GaussianProcessFilter test.
        Creates a sine wave, adds noise and assumes the GP filter
        predicts values closer to real values
        """
        theta = np.radians(np.linspace(0, 360*5, 200))
        testing_signal = TimeSeries.from_values(np.cos(theta))

        noise = TimeSeries.from_values(np.random.normal(0, 0.5, len(testing_signal)) * 0.5)
        testing_signal_with_noise = testing_signal + noise

        kernel = ExpSineSquared()
        gpf = GaussianProcessFilter(kernel=kernel, alpha=0.5, n_restarts_optimizer=100)
        filtered_ts = gpf.filter(testing_signal_with_noise, num_samples=1)
        
        noise_distance = testing_signal_with_noise - testing_signal
        prediction_distance = filtered_ts - testing_signal
        
        self.assertGreater(noise_distance.values().std(), prediction_distance.values().std())

    def test_gaussian_process_multivariate(self):
        gpf = GaussianProcessFilter()

        sine_ts = tg.sine_timeseries(length=30, value_frequency=0.1)
        noise_ts = tg.gaussian_timeseries(length=30) * 0.1
        ts = sine_ts.stack(noise_ts)

        prediction = gpf.filter(ts)

        self.assertEqual(prediction.width, 2)

    def test_gaussian_process_missing_values(self):
        times = pd.DatetimeIndex(np.array([0,1,3,4,5]).astype('datetime64[ns]'))
        ts = TimeSeries.from_times_and_values(times, np.ones(len(times)))

        gpf = GaussianProcessFilter(RBF())
        filtered_values = gpf.filter(ts).values()
        np.testing.assert_allclose(filtered_values, np.ones_like(filtered_values))


if __name__ == '__main__':
    KalmanFilterTestCase().test_kalman()
    MovingAverageTestCase().test_moving_average_univariate()
    MovingAverageTestCase().test_moving_average_multivariate()
    GaussianProcessFilterTestCase().test_gaussian_process()
