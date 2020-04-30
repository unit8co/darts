from .autoregressive_model import AutoRegressiveModel
from ..timeseries import TimeSeries
from ..custom_logging import raise_if_not, time_log, get_logger
from ..models.statistics import check_seasonality
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from typing import List, Optional

logger = get_logger(__name__)

def find_relevant_timestamp_attributes(series: TimeSeries):
    """
    Analyzes the given TimeSeries instance for relevant pd.Timestamp attributes
    in terms of the autocorrelation of their length within the series with the 
    goal of finding the periods of the seasonal trends present in the series.

    :param series: The TimeSeries instance to be analyzed.
    :return: A set of pd.Timestamp attributes with high autocorrelation within 'series'.
    """
    timestamp_attributes = ['month', 'day', 'weekday', 'hour', 'minute', 'second']
    relevant_attributes = set()
    r = acf(series.values())

    if (series.freq() == pd.tseries.offsets.Day):
        if (len(series) > 450):
            


    # 365 seasonality -> month + day
    # 30 day seasonality -> day
    # 7 day seasonality -> weekday 

def compare_timestamps_on_attributes(ts_1: pd.Timestamp, ts_2: pd.Timestamp, required_matches: set) -> bool:
    """
    Compares two timestamps according two a given set of attributes (such as minute, hour, day, etc.).
    It returns true if and only if the two timestamps are matching in all given attributes.

    :param ts_1: First timestamp that will be compared.
    :param ts_2: Second timestamp that will be compared.
    :required_matches: A set of pd.Timestamp attributes which ts_1 and ts_2 will be checked on.
    :return: True if and only if 'ts_1' and 'ts_2' match in all attributes given in 'required_matches'.
    """
    return all(map(lambda attr: getattr(ts_1, attr) == getattr(ts_2, attr), required_matches))

def crop_to_match_seasons(series: TimeSeries, required_matches: Optional[set]) -> TimeSeries:
    """
    Crops a given TimeSeries 'series' that will be used as a training set in such
    a way that its first entry has a timestamp that matches the first timestamp
    right after the end of 'series' in all attributes given in 'required_matches'.
    If no such timestamp can be found, the original TimeSeries instance is returned.
    If the value of 'required_matches' is 'None', the original TimeSeries instance is returned.

    :param series: TimeSeries instance to be cropped.
    :param ts_2: Second timestamp that will be compared.
    :required_matches: A set of pd.Timestamp attributes which will be used to choose the cropping point.
    :return: New TimeSeries instance that is cropped as described above.
    """
    if (required_matches is None): return series

    first_ts = series._series.index[0]
    freq = first_ts.freq
    pred_ts = series._series.index[-1] + freq

    # start at first timestamp of given series and move forward until a matching timestamp is found
    curr_ts = first_ts
    while (curr_ts < pred_ts - 4 * freq):
        curr_ts += freq
        if compare_timestamps_on_attributes(pred_ts, curr_ts, required_matches):
            new_series = series.drop_before(curr_ts)
            return new_series
    
    logger.warning("No matching timestamp could be found, returning original TimeSeries.")
    return series

class FFT(AutoRegressiveModel):

    def __init__(self, nr_freqs_to_keep: Optional[int] = None, required_matches: Optional[set] = None, trend: bool = False, trend_poly_degree: int = 3):
        """
        Forecasting based on a discrete fourier transform using FFT of the (cropped and detrended) training sequence
        with subsequent selection of the most significant frequencies to remove noise from the prediction.

        :param nr_freqs_to_keep: The total number of frequencies that will be used for forecasting.
        :param required_matches: The attributes of pd.Timestamp that will be used to create a training
                                 sequence that is cropped at the beginning such that the first timestamp 
                                 of the training sequence and the first prediction point have matching 'phases'.
                                 If the series has a yearly seasonality, include 'month', if it has a monthly 
                                 seasonality, include 'day', etc.
        :param trend: Boolean value indicating whether or not detrending will be applied before performing DFT.
        :param trend_poly_degree: The degree of the polynomial that will be used for detrending.

        """
        self.nr_freqs_to_keep = nr_freqs_to_keep
        self.required_matches = required_matches
        self.trend = trend
        self.trend_poly_degree = trend_poly_degree

    def __str__(self):
        return 'FFT'

    def chosen_frequencies(self) -> List[float]:
        """
        Returns the most significant frequencies that were chosen when training the model.
        """
        if (not self._fit_called):
            raise_log(Exception('fit() must be called before chosen_frequencies()'), logger)
        frequencies = np.fft.fftfreq(len(self.fft_values))
        chosen_frequencies = frequencies[self.filtered_indices]
        return chosen_frequencies

    @time_log(logger=logger)
    def fit(self, series: TimeSeries):
        super().fit(series)

        # determine trend
        if (self.trend):
            trend_coefficients = np.polyfit(range(len(self.training_series)), series.values(), self.trend_poly_degree)
            self.trend = np.poly1d(trend_coefficients)
        else:
            self.trend = lambda x: 0

        # crop training set to match the seasonality of the first prediction point
        self.cropped_series = crop_to_match_seasons(series, required_matches=self.required_matches)

        # subtract trend
        detrended_values = self.cropped_series.values() - self.trend(range(len(series) - len(self.cropped_series), len(series)))

        # perform dft
        self.fft_values = np.fft.fft(detrended_values)

        # get indices of 'nr_freqs_to_keep' (if a correct value was provied) frequencies with the highest amplitudes
        # by partitioning around the element with sorted index -nr_freqs_to_keep instead of reduntantly sorting the whole array
        first_n = self.nr_freqs_to_keep
        if (first_n is None or first_n < 1 or first_n > len(self.fft_values)): first_n = len(self.fft_values)
        self.filtered_indices = np.argpartition(abs(self.fft_values), -first_n)[-first_n:]

        # set all other values in the frequency domain to 0
        self.fft_values_filtered = np.zeros(len(self.fft_values), dtype=np.complex_)
        self.fft_values_filtered[self.filtered_indices] = self.fft_values[self.filtered_indices]

        # precompute all possible predicted values using inverse dft
        self.predicted_values = np.fft.ifft(self.fft_values_filtered).real


    def predict(self, n: int):
        super().predict(n)
        trend_forecast = np.array([self.trend(i + len(self.training_series)) for i in range(n)])
        periodic_forecast = np.array([self.predicted_values[i % len(self.predicted_values)] for i in range(n)])
        return self._build_forecast_series(periodic_forecast + trend_forecast)