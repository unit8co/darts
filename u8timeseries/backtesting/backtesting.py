"""
Backtesting Functions
---------------------
"""

import pandas as pd
import numpy as np
from ..timeseries import TimeSeries
from ..models.forecasting_model import ForecastingModel
from ..models.regression_model import RegressionModel
from .. import metrics
from ..utils import _build_tqdm_iterator
from ..logging import raise_if_not, get_logger
from typing import Iterable
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

    last_pred_time = series.time_index()[-fcast_horizon_n - 2] if trim_to_series else series.time_index()[-2]

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


def backtest_gridsearch(model_class: type, parameters: dict, series: TimeSeries, fcast_horizon_n: int,
                        num_predictions: int = 10, metric='mape', verbose=False):
    """ A function for finding the best hyperparameters.
    
    Computes 'num_predictions' predictions with horizon 'fcast_horizon_n' for every combination
    of hyperparameter values provided in the 'parameters' dictionary using an instance of the 
    given 'model_class' subclass of ForecastingModel. An instance with the best-performing
    hyperparameters (with respect to 'metric') will be returned.

    Parameters
    ----------
    model
        A ForecastingModel subclass to be tuned for 'series'.
    parameters
        A dictionary containg as keys hyperparameter names, and as values lists of values for the
        respective hyperparameter.
    series
        The TimeSeries instance used for backtesting.
    fcast_horizon_n
        The integer value of the forecasting horizon used during backtesting.
    num_predictions:
        The number of train/predictions cycles performed when testing one hyperparameter combination.
    metric:
        The function name (as string) of a metrics function from the metrics module.
    verbose:
        Whether to print progress.
    
    Returns
    -------
    ForecastingModel
        A 'model_cls' instance with the best-performing hyperparameters from the given selection.
    """

    raise_if_not(hasattr(metrics, metric), "'metric' must be an attribute of u8timeseries.metriecs.", logger)

    backtest_start_time = series.end_time() - (num_predictions + fcast_horizon_n) * series.freq()
    metric_function = getattr(metrics, metric)

    min_error = float('inf')
    best_param_combination = {}
    params_cross_product = list(product(*parameters.values()))
    iterator = _build_tqdm_iterator(params_cross_product, verbose)

    # iterate through all combinations of the provided parameters and choose the best one
    for param_combination in iterator:
        param_combination_dict = dict(list(zip(parameters.keys(), param_combination)))
        model = model_class(**param_combination_dict)
        backtest_forecast = backtest_forecasting(series, model, backtest_start_time, fcast_horizon_n)
        error = metric_function(backtest_forecast, series)
        if (error < min_error):
            min_error = error
            best_param_combination = param_combination_dict
    logger.info('Chosen parameters: ' + str(best_param_combination))
    return model_class(**best_param_combination)
