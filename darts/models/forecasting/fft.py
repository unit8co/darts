"""
Fast Fourier Transform
----------------------
"""

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts.timeseries import TimeSeries
from darts.utils.missing_values import fill_missing_values

logger = get_logger(__name__)


def _check_approximate_seasonality(
    series: TimeSeries,
    seasonality_period: int,
    period_error_margin: int,
    max_seasonality_order: int,
) -> bool:
    """Checks whether the given series has a given seasonality.

    Analyzes the given TimeSeries instance for seasonality of the given period
    while taking into account potential noise of the autocorrelation function.
    This is done by averaging all AC values that are within `period_error_margin`
    steps from the index `seasonality_period` in the ACF domain.

    Parameters
    ----------
    series
        The TimeSeries instance to be analyzed.
    seasonality_period
        The (approximate) period to be checked for seasonality.
    period_error_margin
        The radius around the `seasonality_period` that is taken into consideration when computing the autocorrelation.
    max_seasonality_order
        The maximum number of lags (or inputs to the acf) that can exceed the ac value computed over the interval
        around `seasonality_period`. The lower this number, the stricter the criterion for seasonality.

    Returns
    -------
    bool
        Boolean value indicating whether the seasonality is significant given the parameters passed.
    """
    # fraction of seasonality_period that will skipped when looking at acf values due to high
    # autocorrelation for small lags
    frac = 1 / 4

    # return False if there are not enough entries in the TimeSeries instance
    if len(series) < seasonality_period * (1 + frac):
        return False

    # compute relevant autocorrelation values
    r = acf(
        series.univariate_values(),
        nlags=int(seasonality_period * (1 + frac)),
        fft=False,
    )

    # compute the approximate autocorrelation value for the given period
    left_bound = seasonality_period - period_error_margin
    right_bound = seasonality_period + period_error_margin
    approximation_interval = range(left_bound, right_bound + 1)
    approximated_period_ac = np.mean(r[approximation_interval])

    # compute the number of ac values larger than the approximated ac value for the given period
    indices = list(range(int(frac * seasonality_period), left_bound)) + list(
        range(right_bound + 1, len(r))
    )
    order = sum(
        map(lambda ac_value: int(ac_value > approximated_period_ac), r[indices])
    )

    return order <= max_seasonality_order


def _find_relevant_timestamp_attributes(series: TimeSeries) -> set:
    """Finds pd.Timestamp attributes relevant for seasonality.

    Analyzes the given TimeSeries instance for relevant pd.Timestamp attributes
    in terms of the autocorrelation of their length within the series with the
    goal of finding the periods of the seasonal trends present in the series.

    Parameters
    ----------
    series
        The TimeSeries instance to be analyzed.

    Returns
    -------
    set
        A set of pd.Timestamp attributes with high autocorrelation within `series`.
    """
    relevant_attributes = set()

    if type(series.freq) in {
        pd.tseries.offsets.MonthBegin,
        pd.tseries.offsets.MonthEnd,
    }:
        # check for yearly seasonality
        if _check_approximate_seasonality(series, 12, 1, 0):
            relevant_attributes.add("month")
    elif type(series.freq) == pd.tseries.offsets.Day:
        # check for yearly seasonality
        if _check_approximate_seasonality(series, 365, 5, 20):
            relevant_attributes.update({"month", "day"})
        # check for monthly seasonality
        elif _check_approximate_seasonality(series, 30, 2, 2):
            relevant_attributes.add("day")
        # check for weekly seasonality
        elif _check_approximate_seasonality(series, 7, 0, 0):
            relevant_attributes.add("weekday")
    elif type(series.freq) == pd.tseries.offsets.Hour:
        # check for yearly seasonality
        if _check_approximate_seasonality(series, 8760, 100, 100):
            relevant_attributes.update({"month", "day", "hour"})
        # check for monthly seasonality
        elif _check_approximate_seasonality(series, 730, 10, 30):
            relevant_attributes.update({"day", "hour"})
        # check for weekly seasonality
        elif _check_approximate_seasonality(series, 168, 3, 10):
            relevant_attributes.update({"weekday", "hour"})
        # check for daily seasonality
        elif _check_approximate_seasonality(series, 24, 1, 1):
            relevant_attributes.add("hour")
    elif type(series.freq) == pd.tseries.offsets.Minute:
        # check for daily seasonality
        if _check_approximate_seasonality(series, 1440, 20, 50):
            relevant_attributes.update({"hour", "minute"})
        # check for hourly seasonality
        elif _check_approximate_seasonality(series, 60, 4, 3):
            relevant_attributes.add("minute")

    return relevant_attributes


def _compare_timestamps_on_attributes(
    ts_1: pd.Timestamp, ts_2: pd.Timestamp, required_matches: set
) -> bool:
    """Compares pd.Timestamp instances on attributes.

    Compares two timestamps according two a given set of attributes (such as minute, hour, day, etc.).
    It returns True if and only if the two timestamps are matching in all given attributes.

    Parameters
    ----------
    ts_1
        First timestamp that will be compared.
    ts_2
        Second timestamp that will be compared.
    required_matches
        A set of pd.Timestamp attributes which ts_1 and ts_2 will be checked on.

    Returns
    -------
    bool
        True if and only if `ts_1` and `ts_2` match in all attributes given in `required_matches`.
    """
    return all(
        map(lambda attr: getattr(ts_1, attr) == getattr(ts_2, attr), required_matches)
    )


def _crop_to_match_seasons(
    series: TimeSeries, required_matches: Optional[set]
) -> TimeSeries:
    """Crops TimeSeries instance to contain full periods.

    Crops a given TimeSeries `series` that will be used as a training set in such
    a way that its first entry has a timestamp that matches the first timestamp
    right after the end of `series` in all attributes given in `required_matches`.
    If no such timestamp can be found, the original TimeSeries instance is returned.
    If the value of `required_matches` is `None`, the original TimeSeries instance is returned.

    Parameters
    ----------
    series
        TimeSeries instance to be cropped.
    required_matches
        A set of pd.Timestamp attributes which will be used to choose the cropping point.

    Returns
    -------
    TimeSeries
        New TimeSeries instance that is cropped as described above.
    """
    if required_matches is None or len(required_matches) == 0:
        return series

    first_ts = series.time_index[0]
    freq = series.freq
    pred_ts = series.time_index[-1] + freq

    # start at first timestamp of given series and move forward until a matching timestamp is found
    curr_ts = first_ts
    while curr_ts < pred_ts - 4 * freq:
        curr_ts += freq
        if _compare_timestamps_on_attributes(pred_ts, curr_ts, required_matches):
            new_series = series.drop_before(curr_ts)
            return new_series

    logger.warning(
        "No matching timestamp could be found, returning original TimeSeries."
    )
    return series


class FFT(LocalForecastingModel):
    def __init__(
        self,
        nr_freqs_to_keep: Optional[int] = 10,
        required_matches: Optional[set] = None,
        trend: Optional[str] = None,
        trend_poly_degree: int = 3,
    ):
        """Fast Fourier Transform Model

        This model performs forecasting on a TimeSeries instance using FFT, subsequent frequency filtering
        (controlled by the `nr_freqs_to_keep` argument) and  inverse FFT, combined with the option to detrend
        the data (controlled by the `trend` argument) and to crop the training sequence to full seasonal periods
        Note that if the training series contains any NaNs (missing values), these will be filled using
        :func:`darts.utils.missing_values.fill_missing_values()`.

        Parameters
        ----------
        nr_freqs_to_keep
            The total number of frequencies that will be used for forecasting.
        required_matches
            The attributes of pd.Timestamp that will be used to create a training sequence that is cropped at the
            beginning such that the first timestamp of the training sequence and the first prediction point have
            matching phases. If the series has a yearly seasonality, include `month`, if it has a monthly
            seasonality, include `day`, etc. If not set, or explicitly set to None, the model tries to find the
            pd.Timestamp attributes that are relevant for the seasonality automatically.
        trend
            If set, indicates what kind of detrending will be applied before performing DFT.
            Possible values: 'poly' or 'exp', for polynomial trend, or exponential trend, respectively.
        trend_poly_degree
            The degree of the polynomial that will be used for detrending, if `trend='poly'`.

        Examples
        --------
        Automatically detect the seasonal periods, uses the 10 most significant frequencies for
        forecasting and expect no global trend to be present in the data:

        >>> FFT(nr_freqs_to_keep=10)

        Assume the provided TimeSeries instances will have a monthly seasonality and an exponential
        global trend, and do not perform any frequency filtering:

        >>> FFT(required_matches={'month'}, trend='exp')
        """
        super().__init__()
        self.nr_freqs_to_keep = nr_freqs_to_keep
        self.required_matches = required_matches
        self.trend = trend
        self.trend_poly_degree = trend_poly_degree

    def __str__(self):
        return (
            "FFT(nr_freqs_to_keep="
            + str(self.nr_freqs_to_keep)
            + ", trend="
            + str(self.trend)
            + ")"
        )

    def fit(self, series: TimeSeries):
        series = fill_missing_values(series)
        super().fit(series)
        self._assert_univariate(series)
        series = self.training_series

        # determine trend
        if self.trend == "poly":
            trend_coefficients = np.polyfit(
                range(len(series)), series.univariate_values(), self.trend_poly_degree
            )
            self.trend_function = np.poly1d(trend_coefficients)
        elif self.trend == "exp":
            trend_coefficients = np.polyfit(
                range(len(series)), np.log(series.univariate_values()), 1
            )
            self.trend_function = lambda x: np.exp(trend_coefficients[1]) * np.exp(
                trend_coefficients[0] * x
            )
        else:
            self.trend_function = lambda x: 0

        # subtract trend
        detrended_values = series.univariate_values() - self.trend_function(
            range(len(series))
        )
        detrended_series = TimeSeries.from_times_and_values(
            series.time_index, detrended_values
        )

        # crop training set to match the seasonality of the first prediction point
        if self.required_matches is None:
            curr_required_matches = _find_relevant_timestamp_attributes(
                detrended_series
            )
        else:
            curr_required_matches = self.required_matches
        cropped_series = _crop_to_match_seasons(
            detrended_series, required_matches=curr_required_matches
        )

        # perform dft
        self.fft_values = np.fft.fft(cropped_series.univariate_values())

        # get indices of `nr_freqs_to_keep` (if a correct value was provided) frequencies with the highest amplitudes
        # by partitioning around the element with sorted index -nr_freqs_to_keep instead of sorting the whole array
        first_n = self.nr_freqs_to_keep
        if first_n is None or first_n < 1 or first_n > len(self.fft_values):
            first_n = len(self.fft_values)
        self.filtered_indices = np.argpartition(abs(self.fft_values), -first_n)[
            -first_n:
        ]

        # set all other values in the frequency domain to 0
        self.fft_values_filtered = np.zeros(len(self.fft_values), dtype=np.complex_)
        self.fft_values_filtered[self.filtered_indices] = self.fft_values[
            self.filtered_indices
        ]

        # precompute all possible predicted values using inverse dft
        self.predicted_values = np.fft.ifft(self.fft_values_filtered).real

        return self

    def predict(self, n: int, num_samples: int = 1, verbose: bool = False):
        super().predict(n, num_samples)
        trend_forecast = np.array(
            [self.trend_function(i + len(self.training_series)) for i in range(n)]
        )
        periodic_forecast = np.array(
            [self.predicted_values[i % len(self.predicted_values)] for i in range(n)]
        )
        return self._build_forecast_series(periodic_forecast + trend_forecast)
