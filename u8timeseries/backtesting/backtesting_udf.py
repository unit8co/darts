import pandas as pd
from u8timeseries.timeseries import TimeSeries
from u8timeseries.models.autoregressive_model import AutoRegressiveModel
from u8timeseries.models.regressive_model import RegressiveModel
from typing import Tuple, List, Callable, Any
from ..utils.types import RegrFeatures, RegrDataset


def get_train_val_series(series: TimeSeries, start: pd.Timestamp, nr_points_val: int,
                         nr_steps_iter: int = 1) -> List[Tuple[TimeSeries, TimeSeries]]:
    """
    Returns a list of (training_set, validation_set) pairs for backtesting.

    TODO: this is expanding training window, implement optional sliding window

    :param series: The full time series needs to be split
    :param start: the start time of the earliest validation set
    :param nr_points_val: the number of points in the validation sets
    :param nr_steps_iter: the number of time steps to iterate between the successive validation sets
    :return: a list of (training_set, validation_set) pairs
    """

    assert start in series, 'The provided start timestamp is not in the time series.'
    assert start != series.end_time(), 'The provided start timestamp is the last timestamp of the time series'
    # TODO: maybe also check that valset_duration >= series frequency

    curr_val_start: pd.Timestamp = start

    def _get_train_val_and_increase_pointer() -> Tuple[TimeSeries, TimeSeries]:
        nonlocal curr_val_start

        train_series, val_series_all = series.split_after(curr_val_start)
        val_series = val_series_all.slice_n_points_after(val_series_all.start_time(), nr_points_val)

        curr_val_start = curr_val_start + nr_steps_iter * series.freq()
        return train_series, val_series

    series_pairs = []
    curr_train_series, curr_val_series = _get_train_val_and_increase_pointer()

    while len(curr_val_series) >= nr_points_val:
        series_pairs.append((curr_train_series, curr_val_series))
        curr_train_series, curr_val_series = _get_train_val_and_increase_pointer()

    return series_pairs


def backtest_autoregressive_model(model: AutoRegressiveModel, train_val_series: List[Tuple[TimeSeries, TimeSeries]],
                                  eval_fn: Callable[[TimeSeries, TimeSeries], Any]) -> List[Any]:
    """
    Returns a list of results obtained when calling [eval_fn] on the validation time series in [train_val_series] as
    well as on the corresponding prediction obtained from [model]

    TODO: option for point-prediction
    """

    results = []
    for train, val in train_val_series:
        model.fit(train)
        pred = model.predict(len(val))
        results.append(eval_fn(val, pred))
    return results


def get_train_val_series_for_regr_model(ar_models: List[AutoRegressiveModel],
                                        series: TimeSeries,
                                        external_series: List[TimeSeries],
                                        start_regr_training: pd.Timestamp,
                                        nr_points_training: int,
                                        nr_points_val: int,
                                        nr_steps_iter: int = 1) -> List[Tuple[RegrDataset]]:
    """
    Prepares training & validation datasets needed for backtesting regressive models, supporting the case
    where some features can come from other (auto-regressive) models, and some are "external" and known in advance.


    This function returns a list of (training_set, validation_set) to be consumed
    by regressive models, sliding over time.
    Each {training_}, {validation_} set is a "RegrDataset", i.e., some features series and a target series

    :param ar_models: A list of auto-regressive models that we want to use to produce features
                      TODO: Could we find a simple way to extend this method to support RegressiveModel's as well?
    :param series: the main target time series that we are interested in predicting
    :param external_series: some external regression series to be used by the regressive model. Those series are *not*
                            the output of some other model. Rather they are known in advance at prediction time.
    :param start_regr_training: the timestamp at which the first *training set* starts
                                the training sets and validation sets are contiguous
    :param nr_points_training: the number of points in the (regression) training sets
    :param nr_points_val: the number of points in the (regression) validation sets
    :param nr_steps_iter: the number of time steps to iterate between the successive validation sets
    :return:
    """

    curr_train_start: pd.Timestamp = start_regr_training
    curr_val_start: pd.Timestamp = curr_train_start + nr_points_training * series.freq()

    for s in [series] + external_series:
        assert curr_train_start in s, 'The specified start of training set must be in all provided time series'
        assert curr_val_start in s, 'The start of train set and training set duration must be such that the first' \
                                    'validation set start time is in all provided time series'

    def _get_train_val_and_increase_pointer() -> Tuple[TimeSeries, TimeSeries]:
        """
        Returns new (auto-regressive) training series, new (regressive) training series and
        (regressive) validation series; updates training set pointer
        """
        nonlocal curr_train_start, curr_val_start

        # split main series
        train_ar_series, rest = series.split_after(curr_train_start)
        train_regr_series = rest.slice_n_points_after(curr_train_start, nr_points_training)
        val_regr_series = rest.slice_n_points_after(curr_val_start, nr_points_val)

        # split external series
        external_train: List[TimeSeries] = []  # (regressive) training features
        external_val: List[TimeSeries] = []  # (regressive) validation features

        for es in external_series:
            external_train.append(es.slice_n_points_after(curr_train_start, nr_points_training))
            external_val.append(es.slice_n_points_after(curr_val_start, nr_points_val))

        increment = nr_steps_iter * series.freq()  # we'll iterate both pointer by the same duration
        curr_train_start = curr_train_start + increment
        curr_val_start = curr_val_start + increment

        # TODO from here

        return train_series, val_series


def backtest_regressive_model(model: RegressiveModel,
                              train_features: List[RegrFeatures],
                              train_targets: List[TimeSeries],
                              val_features: List[RegrFeatures],
                              val_targets: List[TimeSeries],
                              eval_fn: Callable[[TimeSeries, TimeSeries], Any]) -> List[Any]:
    """
    Performs backtesting of a RegressiveModel
    :param model: the model to backtest

    # To use for training:
    :param train_features: A list of features time series used for training.
                           The outer list contains training sets for each validation sets.
    :param train_targets: A list of target time series that contains targets for each validation sets

    # To use for validation:
    :param val_features: A list of features time series used for validation.
    :param val_targets: A list of target time series used for validation.

    :return: the result of calling eval_fn() on all the predictions obtained for each validation sets
    """

    assert len(train_features) == len(train_targets) == len(val_features) == len(val_targets), \
        'All time series list must be of same length (one entry per validation set)'

    results = []
    for train_X, train_y, val_X, val_y in zip(train_features, train_targets, val_features, val_targets):
        model.fit(train_X, train_y)
        pred = model.predict(val_X)
        results.append(eval_fn(val_y, pred))
    return results
