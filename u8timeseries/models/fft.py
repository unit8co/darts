from .autoregressive_model import AutoRegressiveModel
from ..timeseries import TimeSeries
from ..custom_logging import raise_if_not, time_log, get_logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = get_logger(__name__)


### helper functions that might have to be relocated ###

def compare_seasonality(ts_1: pd.Timestamp, ts_2: pd.Timestamp, required_matches: set):
    """
    Compares two timestamps according two a given set of attributes (such as minute, hour, day, etc.).
    It returns true if and only if the two timestamps are matching in all given attributes.

    :param ts_1: First timestamp that will be compared.
    :param ts_2: Second timestamp that will be compared.
    :required_matches: A set of pd.Timestamp attributes which ts_1 and ts_2 will be checked on.
    :return: True if and only if 'ts_1' and 'ts_2' match in all attributes given in 'required_matches'.
    """
    is_matching = True
    for required_match in required_matches:
        is_matching = is_matching and (getattr(ts_1, required_match) == getattr(ts_2, required_match))
    return is_matching

def crop_to_match_seasons(series: 'TimeSeries', required_matches: set):
    """
    Crops a given TimeSeries 'series' that will be used as a training set in such
    a way that its first entry has a timestamp that matches the first timestamp
    right after the end of 'series' in all attributes given in 'required_matches'.
    If no such timestamp can be found, the original TimeSeries instance is returned.

    :param series: TimeSeries instance to be cropped.
    :param ts_2: Second timestamp that will be compared.
    :required_matches: A set of pd.Timestamp attributes which will be used to choose the cropping point.
    :return: New TimeSeries instance that is cropped as described above.
    """
    first_ts = series._series.index[0]
    freq = first_ts.freq
    pred_ts = series._series.index[-1] + freq

    # start at first timestamp of given series and move forward until a matching timestamp is found
    curr_ts = first_ts
    while (curr_ts < pred_ts - 4 * freq):
        curr_ts += freq
        if compare_seasonality(pred_ts, curr_ts, required_matches):
            new_series = series.drop_before(curr_ts)
            return new_series
    
    logger.warning("No matching timestamp could be found, returning original TimeSeries.")
    return series

class FFT(AutoRegressiveModel):

    def __init__(self, filter_first_n: int = -1, required_matches: set = {}, detrend: bool = False, detrend_poly_degree: int = 3):
        """
        Forecasting based on a discrete fourier transform using FFT of the (cropped and detrended) training sequence
        with subsequent selection of the most significant frequencies to remove noise from the prediction.

        :param filter_first_n: The total number of frequencies that will be used for forecasting.
        :param required_matches: The attributes of pd.Timestamp that will be used to create a training
                                 sequence that is cropped at the beginning such that the first timestamp 
                                 of the training sequence and the first prediction point have matching 'phases'.
                                 If the series has a yearly seasonality, include 'month', if it has a monthly 
                                 seasonality, include 'day', etc.
        :param detrend: Boolean value indicating whether or not detrending will be applied before performing DFT.
        :detrend_poly_degree: The degree of the polynomial that will be used for detrending.

        """
        self.filter_first_n = filter_first_n
        self.required_matches = required_matches
        self.detrend = detrend
        self.detrend_poly_degree = detrend_poly_degree

    def __str__(self):
        return 'FFT'

    def chosen_frequencies(self):
        """
        Returns the most significant frequencies that were chosen when training the model.
        """
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