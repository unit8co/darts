"""
Backtesting Functions
---------------------
"""

from typing import Iterable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm

from ..timeseries import TimeSeries
from ..models.forecasting_model import ForecastingModel
from ..models.regression_model import RegressionModel
from ..utils import _build_tqdm_iterator
from ..logging import raise_if_not, get_logger
from ..utils.statistics import plot_acf

logger = get_logger(__name__)


# TODO parameterize the moving window

def backtest_forecasting(series: TimeSeries,
                         model: ForecastingModel,
                         start: pd.Timestamp,
                         fcast_horizon_n: int,
                         trim_to_series: bool = True,
                         verbose: bool = False) -> TimeSeries:
    """ A function for backtesting `ForecastingModel`'s.

    This function computes the time series of historical predictions
    that would have been obtained, if `model` had been used to predict `series`
    with a certain time horizon.

    To this end, it repeatedly builds a training set from the beginning of `series`.
    It trains `model` on the training set, emits a (point) prediction for a fixed
    forecast horizon, and then moves the end of the training set forward by one
    time step. The resulting predictions are then returned.

    This always re-trains the models on the entire available history,
    corresponding an expending window strategy.

    Parameters
    ----------
    series
        The time series on which to backtest
    model
        The forecasting model to be backtested
    start
        The first prediction time, at which a prediction is computed for a future time
    fcast_horizon_n
        The forecast horizon for the point predictions
    trim_to_series
        Whether the predicted series has the end trimmed to match the end of the main series
    verbose
        Whether to print progress

    Returns
    -------
    TimeSeries
        A time series containing the forecast values for `series`, when successively applying
        the specified model with the specified forecast horizon.
    """

    raise_if_not(start in series, 'The provided start timestamp is not in the time series.', logger)
    raise_if_not(start != series.end_time(), 'The provided start timestamp is the last timestamp of the time series',
                 logger)

    last_pred_time = series.time_index()[-fcast_horizon_n - 1] if trim_to_series else series.time_index()[-1]

    # build the prediction times in advance (to be able to use tqdm)
    pred_times = [start]
    while pred_times[-1] <= last_pred_time:
        pred_times.append(pred_times[-1] + series.freq())

    # what we'll return
    values = []
    times = []

    iterator = _build_tqdm_iterator(pred_times, verbose)

    for pred_time in iterator:
        train = series.drop_after(pred_time)  # build the training series
        model.fit(train)
        pred = model.predict(fcast_horizon_n)
        values.append(pred.values()[-1])  # store the N-th point
        times.append(pred.end_time())  # store the N-th timestamp

    return TimeSeries.from_times_and_values(pd.DatetimeIndex(times), np.array(values))


def backtest_regression(feature_series: Iterable[TimeSeries],
                        target_series: TimeSeries,
                        model: RegressionModel,
                        start: pd.Timestamp,
                        fcast_horizon_n: int,
                        trim_to_series: bool = True,
                        verbose=False) -> TimeSeries:
    """ A function for backtesting `RegressionModel`'s.

    This function computes the time series of historical predictions
    that would have been obtained, if the `model` had been used to predict `series`
    using the `feature_series`, with a certain time horizon.

    To this end, it repeatedly builds a training set composed of both features and targets,
    from `feature_series` and `target_series`, respectively.
    It trains `model` on the training set, emits a (point) prediction for a fixed
    forecast horizon, and then moves the end of the training set forward by one
    time step. The resulting predictions are then returned.

    This always re-trains the models on the entire available history,
    corresponding an expending window strategy.

    Parameters
    ----------
    feature_series
        A list of time series representing the features for the regression model (independent variables)
    target_series
        The target time series for the regression model (dependent variable)
    model
        The regression model to be backtested
    start
        The first prediction time, at which a prediction is computed for a future time
    fcast_horizon_n
        The forecast horizon for the point predictions
    trim_to_series
        Whether the predicted series has the end trimmed to match the end of the main series
    verbose
        Whether to print progress

    Returns
    -------
    TimeSeries
        A time series containing the forecast values when successively applying
        the specified model with the specified forecast horizon.
    """

    raise_if_not(all([s.has_same_time_as(target_series) for s in feature_series]), 'All provided time series must '
                 'have the same time index', logger)
    raise_if_not(start in target_series, 'The provided start timestamp is not in the time series.', logger)
    raise_if_not(start != target_series.end_time(), 'The provided start timestamp is the '
                 'last timestamp of the time series', logger)

    last_pred_time = (target_series.time_index()[-fcast_horizon_n - 2] if trim_to_series
                      else target_series.time_index()[-2])

    # build the prediction times in advance (to be able to use tqdm)
    pred_times = [start]
    while pred_times[-1] <= last_pred_time:
        pred_times.append(pred_times[-1] + target_series.freq())

    # what we'll return
    values = []
    times = []

    iterator = _build_tqdm_iterator(pred_times, verbose)

    for pred_time in iterator:
        # build train/val series
        train_features = [s.drop_after(pred_time) for s in feature_series]
        train_target = target_series.drop_after(pred_time)
        val_features = [s.slice_n_points_after(pred_time + target_series.freq(), fcast_horizon_n)
                        for s in feature_series]

        model.fit(train_features, train_target)
        pred = model.predict(val_features)
        values.append(pred.values()[-1])  # store the N-th point
        times.append(pred.end_time())  # store the N-th timestamp

    return TimeSeries.from_times_and_values(pd.DatetimeIndex(times), np.array(values))


def forecasting_residuals(model: ForecastingModel, series: TimeSeries, fcast_horizon_n: int = 1,
                          verbose: bool = True) -> TimeSeries:
    """ A function for computing the residuals produced by a given model and time series.

    This function computes the difference between the actual observations from `series`
    and the fitted values vector p obtained by training `model` on `series`.

    For every index i in `series`, p[i] is computed by training `model` on
    series[:(i - `fcast_horizon_n`)] and forecasting `fcast_horizon_n` into the future.
    (p[i] will be set to the last value of the predicted vector.)
    The vector of residuals will be shorter than `series` due to the minimum
    training series length required by `model` and the gap introduced by `fcast_horizon_n`.

    Note that the common usage of the term residuals implies a value for `fcast_horizon_n` of 1.

    Parameters
    ----------
    model
        Instance of ForecastingModel used to compute the fitted values p.
    series
        The TimeSeries instance which the residuals will be computed for.
    fcast_horizon_n
        The forecasting horizon used to predict each fitted value.
    verbose
        Whether to print progress.

    Returns
    -------
    TimeSeries
        The vector of residuals.
    """

    # get first index not contained in the first training set
    first_index = series.time_index()[model.min_train_series_length]

    # compute fitted values
    p = backtest_forecasting(series, model, first_index, fcast_horizon_n, True, verbose=verbose)

    # compute residuals
    series_trimmed = series.slice_intersect(p)
    residuals = series_trimmed - p

    return residuals


def plot_residuals_analysis(residuals: TimeSeries, num_bins: int = 20):
    """ Plots data relevant to residuals.

    This function takes a TimeSeries instance of residuals and plots their values,
    their distribution and their ACF.

    Parameters
    ----------
    residuals
        TimeSeries instance representing residuals.
    num_bins
        Optionally, an integer value determining the number of bins in the histogram.
    """

    fig = plt.figure(constrained_layout=True, figsize=(8, 6))
    gs = fig.add_gridspec(2, 2)

    # plot values
    ax1 = fig.add_subplot(gs[:1, :])
    residuals.plot(ax=ax1)
    ax1.set_ylabel('value')
    ax1.set_title('Residual values')

    # plot distribution
    res_mean, res_std = np.mean(residuals.values()), np.std(residuals.values())
    res_min, res_max = min(residuals.values()), max(residuals.values())
    x = np.linspace(res_min, res_max, 100)
    ax2 = fig.add_subplot(gs[1:, 1:])
    ax2.hist(residuals.values(), bins=num_bins)
    ax2.plot(x, norm(res_mean, res_std).pdf(x) * len(residuals) * (res_max - res_min) / num_bins)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title('Distribution')
    ax2.set_ylabel('count')
    ax2.set_xlabel('value')

    # plot ACF
    ax3 = fig.add_subplot(gs[1:, :1])
    plot_acf(residuals, axis=ax3)
    ax3.set_ylabel('ACF value')
    ax3.set_xlabel('lag')
    ax3.set_title('ACF')
