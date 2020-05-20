"""
Backtesting Functions
---------------------
"""

from typing import Iterable
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from ..timeseries import TimeSeries
from ..models.forecasting_model import ForecastingModel
from ..models.regression_model import RegressionModel
from ..models import NaiveSeasonal, AutoARIMA, ExponentialSmoothing, FFT, Prophet, Theta
from .. import metrics
from ..utils import _build_tqdm_iterator
from ..logging import raise_if_not, get_logger
from typing import Iterable, Optional
from itertools import product

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
    raise_if_not(fcast_horizon_n > 0, 'The provided forecasting horizon must be a positive integer.', logger)

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


def backtest_gridsearch(model_class: type, parameters: dict, 
                        train_series: TimeSeries, 
                        val_series: Optional[TimeSeries] = None,  # only used by split mode
                        fcast_horizon_n: int = 5, num_predictions: int = 10,  # only used by expanding window mode
                        metric='mape', verbose=False):
    """ A function for finding the best hyperparameters.

    This function has 2 modes of operation: Expanding window mode and split mode.
    Both modes of operation evaluate every possible combination of hyperparameter values
    provided in the `parameters` dictionary by instantiating the `model_class` subclass
    of ForecastingModel with each combination and returning the best-performing model with regards
    to the `metric` function. The relationship of the training data and test data depends
    on the mode of operation.

    Expanding window mode (default):
    For every model, `num_predictions` predictions with horizon `fcast_horizon_n` are computed
    by using the `backtest_forecast` function as a subroutine on `train_series`. This means that
    the model is both trained and evaluated on `train_series`.
    Note that the model is retrained for every single prediction, thus this mode is slower.

    Split window mode (activated when `val_series` is passed):
    This mode will be used when the `val_series` argument is passed.
    For every hyperparameter combination, the model is trained on `train_series` and
    evaluated on `val_series`.


    Parameters
    ----------
    model
        The ForecastingModel subclass to be tuned for 'series'.
    parameters
        A dictionary containing as keys hyperparameter names, and as values lists of values for the
        respective hyperparameter.
    train_series
        The TimeSeries instance used for training (and also validation in split mode).
    test_series
        The TimeSeries instance used for validation in split mode.
    fcast_horizon_n
        The integer value of the forecasting horizon used in expanding window mode.
    num_predictions:
        The number of train/prediction cycles performed in one iteration of expanding window mode.
    metric:
        The function name (as string) of a metrics function from the metrics module used for evaluating performance.
    verbose:
        Whether to print progress.

    Returns
    -------
    ForecastingModel
        A 'model_class' instance with the best-performing hyperparameters from the given selection.
    """

    raise_if_not(hasattr(metrics, metric), "'metric' must be an attribute of u8timeseries.metrics.", logger)

    backtest_start_time = train_series.end_time() - (num_predictions + fcast_horizon_n) * train_series.freq()
    metric_function = getattr(metrics, metric)
    min_error = float('inf')
    best_param_combination = {}

    # compute all hyperparameter combinations from selection
    params_cross_product = list(product(*parameters.values()))

    # iterate through all combinations of the provided parameters and choose the best one
    iterator = _build_tqdm_iterator(params_cross_product, verbose)
    for param_combination in iterator:
        param_combination_dict = dict(list(zip(parameters.keys(), param_combination)))
        model = model_class(**param_combination_dict)
        if (val_series is None):  # expanding window mode
            backtest_forecast = backtest_forecasting(train_series, model, backtest_start_time, fcast_horizon_n)
            error = metric_function(backtest_forecast, train_series)
        else:  # split mode
            model.fit(train_series)
            error = metric_function(model.predict(len(val_series)), val_series)
        if (error < min_error):
            min_error = error
            best_param_combination = param_combination_dict
    logger.info('Chosen parameters: ' + str(best_param_combination))
    return model_class(**best_param_combination)


def explore_models(train_series: TimeSeries, val_series: TimeSeries, test_series: TimeSeries,
                   width: int = 3, verbose: bool = True):
    """ A function for exploring the suitability of multiple models on a given train/validation/test split.

    This funtion iterates through a list of models, training each on 'train_series' and 'val_series'
    and evaluating them on 'test_series'. Models with free hyperparameters are first
    tuned by trying out a 'resonable' set of hyperparameter combinations. A model for
    each combination is trained on 'train_series' and evaluated on 'val_series'.
    The best-performing set of hyperparameters is then chosen for that type of model.
    This way, none of the models have 'looked at' the test set before the final evaluation.

    The performance of every model type is then plotted against the series of actual observations.
    Additionally, the MAPE values and the runtimes of every model type are plotted in bar charts.

    Parameters
    ----------
    train_series
        A TimeSeries instance used for training during model tuning and evaluation.
    val_series
        A TimeSeries instance used for validation during model tuning and for training during evaluation.
    test_series
        A TimeSeries instance used for validation when evaluating a model.
    width
        An integer indicating the number of plots that are displayed in one row.
    verbose:
        Whether to print progress.
    """

    raise_if_not(width > 1, "Please choose an integer 'width' value larger than 1", logger)

    # list of tuples containing model classes and hyperparameter selection, if required
    model_parameter_tuples = [
        (NaiveSeasonal, {
            'K': list(range(1, 31))
        }),
        (FFT, {
            'nr_freqs_to_keep': [5, 10, 25, 50, 100, 150, 200],
            'trend': [None, 'poly', 'exp']
        }),
        (Theta, {
            'theta': [0, 1] + list(range(3, 50))
        }),
        (Prophet, {}),
        (AutoARIMA, {})
    ]

    # set up plot grid
    height = math.ceil(len(model_parameter_tuples) / width) + 1
    fig = plt.figure(constrained_layout=True, figsize=(width * 4, height * 4))
    gs = fig.add_gridspec(height, width)

    # arrays to store data for bar charts
    mape_vals = []
    times = []
    index = []

    # iterate through model type selection
    iterator = _build_tqdm_iterator(model_parameter_tuples, verbose)
    for i, (model_class, params) in enumerate(iterator):
        
        # if necessary, tune hyperparameters using train_series and val_series
        if (len(params.keys()) > 0):
            model = backtest_gridsearch(model_class, params, train_series, val_series)
        else:
            model = model_class()

        # fit on train and val series, predict test series
        train_and_val_series = train_series.append(val_series)
        start_time = time.time()
        model.fit(train_and_val_series)
        end_time = time.time()
        predictions = model.predict(len(test_series))

        # plot predictions against observations
        y = int(i / width)
        x = i - y * width
        ax = fig.add_subplot(gs[y, x])
        mape_val = metrics.mape(predictions, test_series)
        ax.set_title(str(model) + "\nMAPE: " + str(round(mape_val, 2)))
        train_and_val_series[-len(test_series):].plot(label='train and val', ax=ax)
        test_series.plot(label='test', ax=ax)
        predictions.plot(label='pred', ax=ax)
        plt.legend()

        # record data for bar charts
        mape_vals.append(mape_val)
        times.append(end_time - start_time)
        index.append(str(model)[:3] + "..")

    # plot bar charts
    ax_mapes = fig.add_subplot(gs[-1, 0])
    ax_mapes.set_title("Performance")
    ax_mapes.set_ylabel('MAPE value')
    pd.Series(mape_vals, index=index).plot.bar(ax=ax_mapes, rot=0)
    ax_times = fig.add_subplot(gs[-1, 1])
    ax_times.set_title("Runtime")
    ax_times.set_ylabel('seconds')
    pd.Series(times, index=index).plot.bar(ax=ax_times, rot=0)

    plt.show()
