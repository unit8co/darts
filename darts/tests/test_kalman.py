import numpy as np
import pandas as pd

from .base_test_class import DartsBaseTestClass
from ..models.kalman import Kalman
from ..timeseries import TimeSeries



class KalmanTestCase(DartsBaseTestClass):


    def test_kalman(self):
        """ Kalman test.
        Creates an increasing sequence of numbers, adds noise and 
        assumes the kalman filter predicts values closer to real values
        """
        kf = Kalman()
        testing_signal = np.arange(1,5,0.1)

        noise = np.random.normal(0, 0.7, testing_signal.shape)
        testing_signal_with_noise = testing_signal + noise

        df = pd.DataFrame(data=testing_signal_with_noise, columns = ['signal'])
        testing_signal_with_noise_ts = TimeSeries.from_dataframe(df, value_cols=['signal'])
        kf.fit(testing_signal_with_noise_ts)
        prediction = kf.filter().univariate_values()
        
        noise_distance = testing_signal_with_noise - testing_signal
        prediction_distance = prediction - testing_signal
        
        assert noise_distance.std() > prediction_distance.std()


if __name__ == '__main__':
    KalmanTestCase().test_kalman()