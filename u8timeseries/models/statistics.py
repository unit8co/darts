from ..timeseries import TimeSeries
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from typing import Tuple
import math


def check_seasonality(ts: 'TimeSeries', m: int = None, max_lag: int = 24, alpha: float = 5):
    """
    Returns whether the TimeSeries [ts] is seasonal with period [m] or not.

    If [m] is None, the seasonality period is inferred from the Auto-correlation Function (ACF).

    :param ts: The TimeSeries to check for seasonality.
    :param m: The seasonality period to check.
    :param max_lag: The maximal lag allowed in the ACF.
    :param alpha: The desired confidence level (default 5%).
    :return: A tuple (season, m), where s is a boolean indicating whether the TimeSeries has seasonality or not
             and m is the seasonality period.
    """
    if m is not None:
        if m < 0:
            raise ValueError('m must be a positive integer.')

    n_unique = np.unique(ts.values()).shape[0]

    if n_unique > 1:  # Check for non-constant TimeSeries

        r = acf(ts.values(), nlags=max_lag)  # In case user wants to check for seasonality higher than 24 steps.

        grad = np.gradient(r)
        signs_changes = np.diff(np.sign(grad))

        if m is None:
            # Tries to infer seasonality from AutoCorrelation Function if no value of m has been provided.
            # We look for the first positive local maximum of the ACF by checking the sign changes in the gradient.

            # Local maximum is indicated by signs_change == -2
            if len(np.nonzero((signs_changes == -2))[0]) > 0:
                m = np.nonzero((signs_changes == -2))[0][0] + 1
            else:
                return False, 0

        # Check for local maximum when m is user defined
        if np.nonzero((signs_changes == -2))[0][0] != m-1:
            return False, m

        # Remove r[0], the auto-correlation at order 0, that introduces bias.
        r = r[1:]

        stat = bartlett_formula(r, m - 1, len(ts))

        # The upper limit of the significance interval.
        band_upper = r.mean() + norm.ppf(1 - alpha/200)*r.var()

        # Only local maximum with positive values are candidates to be tested against significance level.
        season = r[m-1] > stat * band_upper

        return season, m

    return False, 0


def bartlett_formula(r, m, length):
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


def seasonal_adjustment(ts: 'TimeSeries', frequency, model='multiplicative') -> Tuple['TimeSeries', 'TimeSeries']:
    """
    Adjusts the TimeSeries [ts] for a seasonality of order [frequency] using the [model] decomposition.

    :param ts: The TimeSeries to adjust.
    :param frequency: The seasonality period.
    :param model: -the type of decomposition to use (additive or multipilcative)
    :return: A tuple (ts, seasonality) of TimeSeries where ts is the adjusted original TimeSeries ad seasonality
            is a TimeSeries containing the extracted seasonality.
    """
    decomp = seasonal_decompose(ts.values(), model=model, freq=frequency)
    seasonality = TimeSeries.from_times_and_values(ts.time_index(), decomp.seasonal)

    if (seasonality.values() < 1e-6).any():
        print("WARNING seasonal indexes equal to zero, using non-seasonal Theta method ")
    else:
        if model == 'multiplicative':
            ts = ts / seasonality
        else:
            ts = ts - seasonality
    return ts, seasonality


def plot_acf(ts: 'TimeSeries', m: int = None, max_lag: int = 24, alpha: float = 5):
    """
    Plots the ACF of [ts], highlighting the seasonality at lag [m] and the significance interval.

    :param ts: The TimeSeries whose ACF should be plotted.
    :param m:  The seasonality period.
    :param max_lag: The maximal lag order to consider.
    :param alpha: The confidence interval.
    :return: Shows the plot.
    """

    r = acf(ts.values(), nlags=max_lag)

    fig = plt.figure(figsize=(10, 5))

    stat = bartlett_formula(r[1:], m - 1, len(ts))

    for i in range(len(r)):
        if i == m:
            pass
        else:
            plt.plot((i, i), (0, r[i]), color='black', lw=.5)

    if m is not None:
        plt.plot((m, m), (0, r[m]), color='red', lw=.5)

    band_up = (r[1:].mean() + norm.ppf(1 - alpha/200) * r[1:].var()) * stat
    plt.plot((0, len(r) - 1), (band_up, band_up), linestyle='--', color='blue')
    plt.plot((0, len(r) - 1), (-band_up, -band_up), linestyle='--', color='blue')
    plt.plot((0, len(r) - 1), (0, 0), color='black')

    plt.show(fig)
