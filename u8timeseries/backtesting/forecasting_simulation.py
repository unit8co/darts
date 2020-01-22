import pandas as pd
import numpy as np
from IPython import get_ipython
from tqdm import tqdm, tqdm_notebook
from u8timeseries.timeseries import TimeSeries
from u8timeseries.models.autoregressive_model import AutoRegressiveModel
from u8timeseries.models.regressive_model import RegressiveModel
from typing import List


def _build_iterator(iterable, verbose):
    def _isnotebook():
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter

    if verbose:
        if _isnotebook():
            iterator = tqdm_notebook(iterable)
        else:
            iterator = tqdm(iterable)
    else:
        iterator = iterable
    return iterator


def simulate_forecast_ar(series: 'TimeSeries',
                         model: 'AutoRegressiveModel',
                         start: 'pd.Timestamp',
                         fcast_horizon_n: int,
                         trim_to_series: bool = True,
                         verbose=False) -> 'TimeSeries':
    """
    Provides an environment for forecasting future values of the TimeSeries 'series`.

    This function predicts the `fcast_horizon_n` values for the TimeSeries `series` starting from the date `start`
    according to the auto-regressive model `model`.

    :param series: The TimeSeries to forecast.
    :param model: The AutoRegressiveModel to use.
    :param start: The first time at which a prediction is produced for a future time.
    :param fcast_horizon_n: The number of future values to predict.
    :param trim_to_series: Whether the predicted series has the end trimmed to match the end of the main series or not.
    :param verbose: Whether to print progress or not.
    :return: A TimeSeries containing the fore-casted values of `series` over the horizon with respect to the model \
    `model`.

    """
    assert start in series, 'The provided start timestamp is not in the time series.'
    assert start != series.end_time(), 'The provided start timestamp is the last timestamp of the time series'

    last_pred_time = series.time_index()[-fcast_horizon_n - 2] if trim_to_series else series.time_index()[-2]

    # build the prediction times in advance (to be able to use tqdm)
    pred_times = [start]
    while pred_times[-1] <= last_pred_time:
        pred_times.append(pred_times[-1] + series.freq())

    # what we'll return
    values = []
    times = []

    iterator = _build_iterator(pred_times, verbose)

    for pred_time in iterator:
        if not verbose:
            print('.', end='')
        train = series.drop_after(pred_time)  # build the training series

        model.fit(train)
        pred = model.predict(fcast_horizon_n)
        values.append(pred.values()[-1])  # store the N-th point
        times.append(pred.end_time())  # store the N-th timestamp

    return TimeSeries.from_times_and_values(pd.DatetimeIndex(times), np.array(values))


def simulate_forecast_regr(feature_series: List[TimeSeries],
                           target_series: TimeSeries,
                           model: RegressiveModel,
                           start: pd.Timestamp,
                           fcast_horizon_n: int,
                           trim_to_series: bool = True,
                           verbose=False) -> TimeSeries:
    """
    Returns a TimeSeries containing the forecasts that would have been obtained from a given RegressiveModel,
    on a given forecast time horizon.

    .. todo: review and add to documentation.
    .. todo: optionally also return weights, when those are available in model
    .. todo: (getattr(model.model, 'coef_', None) is not None)

    :param feature_series: the feature time series of the regressive model
    :param target_series: the target time series of the regressive model (i.e., the series to predict)
    :param model: the RegressiveModel to use
    :param start: when the forecasts start (i.e., the first time at which a prediction is produced for a future time)
    :param fcast_horizon_n: the forecast horizon
    :param trim_to_series: whether the returned predicted series has the end trimmed to match the end of the main series
    :param verbose: whether to print progress
    :return:
    """
    assert all([s.has_same_time_as(target_series) for s in feature_series]), 'All provided time series must ' \
                                                                             'have the same time index'
    assert start in target_series, 'The provided start timestamp is not in the time series.'
    assert start != target_series.end_time(), 'The provided start timestamp is the last timestamp of the time series'

    last_pred_time = target_series.time_index()[-fcast_horizon_n - 2] if trim_to_series else target_series.time_index()[-2]

    # build the prediction times in advance (to be able to use tqdm)
    pred_times = [start]
    while pred_times[-1] <= last_pred_time:
        pred_times.append(pred_times[-1] + target_series.freq())

    # what we'll return
    values = []
    times = []

    iterator = _build_iterator(pred_times, verbose)

    for pred_time in iterator:
        if not verbose:
            print('.', end='')
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