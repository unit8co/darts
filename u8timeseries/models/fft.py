from .autoregressive_model import AutoRegressiveModel
from ..timeseries import TimeSeries
from ..custom_logging import raise_if_not, time_log, get_logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
    logger.warning("No matching time stamp could be found, returning original TimeSeries.")
    return series

class FFT(AutoRegressiveModel):

    def __init__(self, filter_first_n=-1, required_matches={}, detrend=False, detrend_poly_degree=3):
        self.filter_first_n = filter_first_n
        self.required_matches = required_matches
        self.detrend = detrend
        self.detrend_poly_degree = detrend_poly_degree

    def __str__(self):
        return 'FFT'

    def chosen_frequencies(self):
        if (not self._fit_called):
            raise_log(Exception('fit() must be called before chosen_frequencies()'), logger)
        frequencies = np.fft.fftfreq(len(self.fft_values))
        chosen_frequencies = frequencies[self.filtered_indices]
        return chosen_frequencies

    @time_log(logger=logger)
    def fit(self, series: 'TimeSeries'):
        super().fit(series)

        # determine trend
        if (self.detrend):
            trend_coefficients = np.polyfit(range(len(self.training_series)), series.values(), 3)
            self.trend = np.poly1d(trend_coefficients)
        else:
            self.trend = lambda x: 0

        # crop training set to match the seasonality of the first prediction point
        self.cropped_series = crop_to_match_seasons(series, required_matches=self.required_matches)

        # subtract trend
        detrended_values = self.cropped_series.values() - self.trend(range(len(series) - len(self.cropped_series), len(series)))

        # perform dft
        self.fft_values = np.fft.fft(detrended_values)

        # filter out low-amplitude frequencies
        first_n = self.filter_first_n
        if (first_n < 0 or first_n > len(self.fft_values)): first_n = len(self.fft_values)
        self.filtered_indices = np.argpartition(abs(self.fft_values), -first_n)[-first_n:]
        self.fft_values_filtered = np.zeros(len(self.fft_values), dtype=np.complex_)
        self.fft_values_filtered[self.filtered_indices] = self.fft_values[self.filtered_indices]

        # precompute all possible predicted values
        self.predicted_values = np.fft.ifft(self.fft_values_filtered).real


    def predict(self, n: int):
        super().predict(n)
        trend_forecast = np.array([self.trend(i + len(self.training_series)) for i in range(n)])
        periodic_forecast = np.array([self.predicted_values[i % len(self.predicted_values)] for i in range(n)])
        return self._build_forecast_series(periodic_forecast + trend_forecast)