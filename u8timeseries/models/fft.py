from .autoregressive_model import AutoRegressiveModel
from ..timeseries import TimeSeries
from ..custom_logging import raise_if_not, time_log, get_logger
import numpy as np

logger = get_logger(__name__)


### helper functions that might have to be relocated ###

def compare_seasonality(ts_1, ts_2, required_matches):
    is_matching = True
    if 'month' in required_matches:
        is_matching = is_matching and (ts_1.month == ts_2.month)
    if 'day' in required_matches:
        is_matching = is_matching and (ts_1.day == ts_2.day)
    if 'weekday' in required_matches:
        is_matching = is_matching and (ts_1.weekday() == ts_2.weekday())
    if 'hour' in required_matches:
        is_matching = is_matching and (ts_1.hour == ts_2.hour)
    if 'minute' in required_matches:
        is_matching = is_matching and (ts_1.minute == ts_2.minute)
    if 'second' in required_matches:
        is_matching = is_matching and (ts_1.second == ts_2.second)
    return is_matching



def crop_to_match_seasons(series: 'TimeSeries', required_matches={'day', 'month'}):
    
    first_ts = series._series.index[0]
    freq = first_ts.freq
    pred_ts = series._series.index[-1] + freq

    curr_ts = first_ts
    while (curr_ts < pred_ts - 3 * freq):
        curr_ts += freq
        if compare_seasonality(pred_ts, curr_ts, required_matches):
            new_series = series.drop_before(curr_ts)
            return new_series
    
    logger.warning("No matchins time stamp could be found, returning original TimeSeries.")
    return series


class FFT(AutoRegressiveModel):

    def __init__(self, percentile=80):
        self.freq_percentile = percentile

    def __str__(self):
        return 'FFT'


    def ifft_at(self, n: int):
        N = len(self.fft_values)
        arr = np.exp(np.array(range(N)) * 2j * np.pi * n / N)
        return (np.dot(self.fft_values_filtered, arr) / N).real

    @time_log(logger=logger)
    def fit(self, series: 'TimeSeries'):
        super().fit(series)

        self.cropped_series = crop_to_match_seasons(series)

        self.fft_values = np.fft.fft(self.cropped_series.values())
        threshold = np.percentile(abs(self.fft_values), self.freq_percentile)

        def filter_amplitude(x):
            if (abs(x) < threshold):
                return 0
            else:
                return x

        f = np.vectorize(filter_amplitude)
        self.fft_values_filtered = f(self.fft_values)


    def predict(self, n: int):
        super().predict(n)
        first_prediction_index = len(self.fft_values)
        forecast = np.array([self.ifft_at(i + first_prediction_index) for i in range(n)])
        return self._build_forecast_series(forecast)