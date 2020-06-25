"""
Utils for time series statistics
--------------------------------
"""

import math
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

from ..logging import raise_log, get_logger
from ..timeseries import TimeSeries

logger = get_logger(__name__)


def check_seasonality(ts: TimeSeries,
                      m: Optional[int] = None,
                      max_lag: int = 24,
                      alpha: float = 0.05):
    """
    Returns whether the TimeSeries `ts` is seasonal with period `m` or not.

    If `m` is None, we work under the assumption that there is a unique seasonality period, which is inferred
    from the Auto-correlation Function (ACF).

    Parameters
    ----------
    ts
        The time series to check for seasonality.
    m
        The seasonality period to check.
    max_lag
        The maximal lag allowed in the ACF.
    alpha
        The desired confidence level (default 5%).

    Returns
    -------
    Tuple[bool, int]
        A tuple `(season, m)`, where season is a boolean indicating whether the series has seasonality or not
        and `m` is the seasonality period.
    """

    ts._assert_univariate()

    if m is not None and (m < 2 or not isinstance(m, int)):
        raise_log(ValueError('m must be an integer greater than 1.'), logger)

    if m is not None and m > max_lag:
        raise_log(ValueError('max_lag must be greater than or equal to m.'), logger)

    n_unique = np.unique(ts.values()).shape[0]

    if n_unique == 1:  # Check for non-constant TimeSeries
        return False, 0

    r = acf(ts.values(), nlags=max_lag, fft=False)  # In case user wants to check for seasonality higher than 24 steps.

    gradient = np.gradient(r)
    gradient_signs_changes = np.diff(np.sign(gradient))

    # Tries to infer seasonality from Auto-Correlation Function if no value of m has been provided.
    # We look for the first positive significant local maximum of the ACF by checking the sign changes
    # in the gradient.

    # Local maximum is indicated by signs_change == -2.
    if len(np.nonzero((gradient_signs_changes == -2))[0]) == 0:
        logger.info('The ACF has no local maximum for m < max_lag = {}.'.format(max_lag))
        return False, 0

    # Building a list of candidates for local maximum.
    candidates = np.nonzero((gradient_signs_changes == -2))[0].tolist()

    # If a -2 value appears in gradient_signs_changes at index i, then the local
    # maximum of r occurs either at index i or i+1. We check manually and change the candidates accordingly.
    candidates = [i if r[i] >= r[i + 1] else i + 1 for i in candidates]

    if m is not None:
        # Check for local maximum when m is user defined.
        test = m not in candidates

        if test:
            return False, m

        candidates = [m]

    # Remove r[0], the auto-correlation at lag order 0, that introduces bias.
    r = r[1:]

    # The non-adjusted upper limit of the significance interval.
    band_upper = r.mean() + norm.ppf(1 - alpha / 2) * r.var()

    # Significance test, stops at first admissible value. The two '-1' below
    # compensate for the index change due to the restriction of the original r to r[1:].
    for candidate in candidates:
        stat = _bartlett_formula(r, candidate - 1, len(ts))
        if r[candidate - 1] > stat * band_upper:
            return True, candidate
    return False, 0


def _bartlett_formula(r: np.ndarray,
                      m: int,
                      length: int) -> float:
    """
    Computes the standard error of `r` at order `m` with respect to `length` according to Bartlett's formula.

    Parameters
    ----------
    r
        The array whose standard error is to be computed.
    m
        The order of the standard error.
    length
        The size of the underlying sample to be used.

    Returns
    -------
    float
        The standard error of `r` with order `m`.
    """

    if m == 1:
        return math.sqrt(1 / length)
    else:
        return math.sqrt((1 + 2 * sum(map(lambda x: x ** 2, r[:m - 1]))) / length)


def extract_trend_and_seasonality(ts: TimeSeries,
                                  freq: int = None,
                                  model: str = 'multiplicative') -> Tuple[TimeSeries, TimeSeries]:
    """
    Extracts trend and seasonality from a time series using `statsmodels.seasonal_decompose`.

    Parameters
    ----------
    ts
        The series to decompose
    freq
        The seasonality period to use.
    model
        The type of decomposition to use ('additive' or 'multiplicative').

    Returns
    -------
        A tuple of (trend, seasonal) time series.
    """

    ts._assert_univariate()

    decomp = seasonal_decompose(ts.pd_series(), period=freq, model=model, extrapolate_trend='freq')

    season = TimeSeries.from_times_and_values(ts.time_index(), decomp.seasonal)
    trend = TimeSeries.from_times_and_values(ts.time_index(), decomp.trend)

    return trend, season


def remove_from_series(ts: TimeSeries,
                       other: TimeSeries,
                       model: str) -> TimeSeries:
    """
    Removes the TimeSeries `other` from the TimeSeries `ts` as specified by `model`.
    Use e.g. to remove an additive or multiplicative trend from a series.

    Parameters
    ----------
    ts
        The TimeSeries to be modified.
    other
        The TimeSeries to remove.
    model
        The type of model considered (either 'additive' or 'multiplicative').

    Returns
    -------
    TimeSeries
        A TimeSeries defined by removing `other` from `ts`.
    """

    ts._assert_univariate()

    if model == 'multiplicative':
        new_ts = ts / other
    elif model == 'additive':
        new_ts = ts - other
    else:
        raise_log(ValueError('Invalid parameter; must be either "additive" or "multiplicative". Was: {}'.format(model)))
    return new_ts


def remove_seasonality(ts: TimeSeries,
                       freq: int = None,
                       model: str = 'multiplicative') -> TimeSeries:
    """
    Adjusts the TimeSeries `ts` for a seasonality of order `frequency` using the `model` decomposition.

    Parameters
    ----------
    ts
        The TimeSeries to adjust.
    freq
        The seasonality period to use.
    model
        The type of decomposition to use ('additive' or 'multiplicative').
    Returns
    -------
    TimeSeries
        A new time series that is the adjusted original time series
    """

    ts._assert_univariate()

    _, seasonality = extract_trend_and_seasonality(ts, freq, model)
    new_ts = remove_from_series(ts, seasonality, model)
    return new_ts


def remove_trend(ts: TimeSeries,
                 model='multiplicative') -> TimeSeries:
    """
    Adjusts the TimeSeries `ts` for a trend using the `model` decomposition.

    Parameters
    ----------
    ts
        The TimeSeries to adjust.
    model
        The type of decomposition to use (additive or multiplicative).
    Returns
    -------
    TimeSeries
        A new time series that is the adjusted original time series
    """

    ts._assert_univariate()

    trend, _ = extract_trend_and_seasonality(ts, model=model)
    new_ts = remove_from_series(ts, trend, model)
    return new_ts


def plot_acf(ts: TimeSeries,
             m: Optional[int] = None,
             max_lag: int = 24,
             alpha: float = 0.05,
             fig_size: Tuple[int, int] = (10, 5),
             axis: Optional[plt.axis] = None) -> None:
    """
    Plots the ACF of `ts`, highlighting it at lag `m`, with corresponding significance interval.

    Parameters
    ----------
    ts
        The TimeSeries whose ACF should be plotted.
    m
        Optionally, a time lag to highlight on the plot.
    max_lag
        The maximal lag order to consider.
    alpha
        The confidence interval to display.
    fig_size
        The size of the figure to be displayed.
    axis
        Optionally, an axis object to plot the ACF on.
    """

    ts._assert_univariate()

    r = acf(ts.values(), nlags=max_lag, fft=False)  # , alpha=alpha) and confint as output too

    # Computes the confidence interval at level alpha for all lags.
    stats = []
    for i in range(1, max_lag + 1):
        stats.append(_bartlett_formula(r[1:], i, len(ts)))

    if (axis is None):
        plt.figure(figsize=fig_size)
        axis = plt

    for i in range(len(r)):
        axis.plot((i, i), (0, r[i]), color=('red' if m is not None and i == m else 'black'), lw=.5)

    upp_band = r[1:].mean() + norm.ppf(1 - alpha / 2) * r[1:].var()
    acf_band = [upp_band * stat for stat in stats]

    axis.fill_between(np.arange(1, max_lag + 1), acf_band, [-x for x in acf_band], color='blue', alpha=.25)
    axis.plot((0, max_lag + 1), (0, 0), color='black')
