from ..timeseries import TimeSeries
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from typing import Tuple
import math


def check_seasonality(ts: 'TimeSeries', m: int = None, max_lag: int = 24, alpha: float = 0.05):
    """
    Returns whether the TimeSeries [ts] is seasonal with period [m] or not.

    If [m] is None, we work under the assumption that there is a unique seasonality period, which is inferred
    from the Auto-correlation Function (ACF).

    TODO : Generalize to multiple seasonality periods.

    :param ts: The TimeSeries to check for seasonality.
    :param m: The seasonality period to check.
    :param max_lag: The maximal lag allowed in the ACF.
    :param alpha: The desired confidence level (default 5%).
    :return: A tuple (season, m), where s is a boolean indicating whether the TimeSeries has seasonality or not
             and m is the seasonality period.
    """
    if m is not None and m < 0:
        raise ValueError('m must be a positive integer.')

    n_unique = np.unique(ts.values()).shape[0]

    if n_unique == 1:  # Check for non-constant TimeSeries
        return False, 0

    r = acf(ts.values(), nlags=max_lag)  # In case user wants to check for seasonality higher than 24 steps.

    gradient = np.gradient(r)
    gradient_signs_changes = np.diff(np.sign(gradient))

    if m is None:
        # Tries to infer seasonality from AutoCorrelation Function if no value of m has been provided.
        # We look for the first positive significant local maximum of the ACF by checking the sign changes
        # in the gradient.

        # Local maximum is indicated by signs_change == -2.
        if len(np.nonzero((gradient_signs_changes == -2))[0]) == 0:
            return False, 0

        candidate = np.nonzero((gradient_signs_changes == -2))[0].tolist()

    else:
        # Check for local maximum when m is user defined.
        if m - 1 not in np.nonzero((gradient_signs_changes == -2))[0]:
            return False, m

        candidate = [m - 1]

    # Remove r[0], the auto-correlation at lag order 0, that introduces bias.
    r = r[1:]

    # The non-adjusted upper limit of the significance interval.
    band_upper = r.mean() + norm.ppf(1 - alpha/2)*r.var()

    # Significance test, stops at first admissible value.
    is_significant = False
    i = 0
    c = 0

    while not is_significant and i < len(candidate):
        c = candidate[i] + 1
        stat = _bartlett_formula(r, c - 1, len(ts))

        is_significant = r[c-1] > stat * band_upper
        i += 1

    return is_significant, c


def _bartlett_formula(r, m, length) -> float:
    """
    Computes the standard error of [r] at order [m] with respect to [length] according to Bartlett's formula.

    :param r: The array whose standard error is to be computed.
    :param m: The order of the standard error.
    :param length: The size of the underlying sample to be used.
    :return: The standard error of [r] with order [m].
    """
    if m == 1:
        return math.sqrt(1/length)
    else:
        return math.sqrt((1 + 2 * sum(map(lambda x: x ** 2, r[:m-1]))) / length)


def extract_trend_and_seasonality(ts: 'TimeSeries', freq: int = None, model: str = 'multiplicative') \
                                                                -> Tuple['TimeSeries', 'TimeSeries']:
    """
    Extracts trend and seasonality from the TimeSeries [ts] using statsmodels.seasonal_decompose.

    :param ts: The TimeSeries to study.
    :param freq: The seasonality period to use.
    :param model: The type of decomposition to use ('additive' or 'multiplicative').
    :return: A tuple (trend, seasonal) of TimeSeries containing the trend and seasonality respectively.
    """

    decomp = seasonal_decompose(ts.pd_series(), freq=freq, model=model, extrapolate_trend='freq')

    season = TimeSeries.from_times_and_values(ts.time_index(), decomp.seasonal)
    trend = TimeSeries.from_times_and_values(ts.time_index(), decomp.trend)

    return trend, season


def remove_from_series(ts: 'TimeSeries', other: 'TimeSeries', model: str) -> 'TimeSeries':
    """
    Removes the TimeSeries [other] from the TimeSeries [ts] according to the specified model.

    :param ts: The TimeSeries to be modified.
    :param other: The TimeSeries to remove.
    :param model: The type of model considered (either 'additive' or 'multiplicative').
    :return: A TimeSeries [new_ts], defined by removing [other] from [ts].
    """
    if (other.values() < 1e-6).any() and model == 'multiplicative':
        print("WARNING indexes equal to zero, cannot remove for a multiplicative model.")
        return ts
    else:
        if model == 'multiplicative':
            new_ts = ts / other
        else:
            new_ts = ts - other

    return new_ts


def remove_seasonality(ts: 'TimeSeries', freq: int = None, model: str = 'multiplicative') \
                                                        -> 'TimeSeries':
    """
    Adjusts the TimeSeries [ts] for a seasonality of order [frequency] using the [model] decomposition.

    :param ts: The TimeSeries to adjust.
    :param freq: The seasonality period to use.
    :param model: The type of decomposition to use (additive or multiplicative).
    :return: A tuple (new_ts, seasonality) of TimeSeries where new_ts is the adjusted original TimeSeries and
            seasonality is a TimeSeries containing the removed seasonality.
    """

    _, seasonality = extract_trend_and_seasonality(ts, freq, model)

    new_ts = remove_from_series(ts, seasonality, model)

    return new_ts


def remove_trend(ts: 'TimeSeries', model='multiplicative') -> 'TimeSeries':
    """
    Adjusts the TimeSeries [ts] for a trend using the [model] decomposition.

    :param ts: The TimeSeries to adjust.
    :param model: The type of decomposition to use (additive or multiplicative).
    :return: A tuple (new_ts, trend) of TimeSeries where new_ts is the adjusted original TimeSeries and
            seasonality is a TimeSeries containing the removed trend.
    """
    trend, _ = extract_trend_and_seasonality(ts, model=model)

    new_ts = remove_from_series(ts, trend, model)

    return new_ts


def plot_acf(ts: 'TimeSeries', m: int = None, max_lag: int = 24, alpha: float = 0.05,
             fig_size: Tuple[int, int] = (10, 5)):
    """
    Plots the ACF of [ts], highlighting it at lag [m], with corresponding significance interval.

    :param ts: The TimeSeries whose ACF should be plotted.
    :param m:  The lag to highlight.
    :param max_lag: The maximal lag order to consider.
    :param alpha: The confidence interval.
    :param fig_size: The size of the figure to be displayed.
    :return:
    """
    r = acf(ts.values(), nlags=max_lag)

    # Computes the confidence interval at level alpha for all lags.
    stats = []
    for i in range(1, max_lag+1):
        stats.append(_bartlett_formula(r[1:], i, len(ts)))

    plt.figure(figsize=fig_size)

    for i in range(len(r)):
        plt.plot((i, i), (0, r[i]), color=('red' if m is not None and i == m else 'black'), lw=.5)

    upp_band = r[1:].mean() + norm.ppf(1 - alpha/2) * r[1:].var()
    acf_band = [upp_band * stat for stat in stats]

    plt.fill_between(np.arange(1, max_lag+1), acf_band, [-x for x in acf_band], color='blue', alpha=.25)
    plt.plot((0, max_lag+1), (0, 0), color='black')

    plt.show()
