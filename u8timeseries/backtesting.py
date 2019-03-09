import pandas as pd
from .timeseries import TimeSeries
from .models.autoregressive_model import AutoRegressiveModel
from typing import Tuple, List, Callable


def get_train_and_val_series(series: TimeSeries, start: pd.Timestamp, valset_duration: pd.Timedelta,
                             nr_steps_iter: int = 1) -> List[Tuple[TimeSeries, TimeSeries]]:
    """
    Returns a list of (training_set, validation_set) pairs for backtesting.
    :param series: The full time series needs to be split
    :param start: the start time of the earliest validation set
    :param valset_duration: the duration of the validation sets
    :param nr_steps_iter: the number of time steps to iterate between the successive validation sets
    :return: a list of (training_set, validation_set) pairs
    """

    assert start in series.time_index(), 'The provided start timestamp is not in the time series.'
    assert start != series.time_index()[-1], 'The provided start timestamp is the last timestamp of the time series'
    # TODO: maybe also check that valset_duration >= series frequency

    curr_val_start: pd.Timestamp = start

    def _get_train_val_and_increase_pointer() -> Tuple[TimeSeries, TimeSeries]:
        nonlocal curr_val_start

        train_series, val_series_all = series.split(curr_val_start)
        val_series = val_series_all.slice_duration(val_series_all.start_time(), valset_duration)

        curr_val_start = curr_val_start + nr_steps_iter * series.freq()
        return train_series, val_series

    series_pairs = []
    curr_train_series, curr_val_series = _get_train_val_and_increase_pointer()

    while curr_val_series.duration() >= valset_duration:
        series_pairs.append((curr_train_series, curr_val_series))
        curr_train_series, curr_val_series = _get_train_val_and_increase_pointer()

    return series_pairs


def backtest_autoregressive_model(model: AutoRegressiveModel, train_val_series: List[Tuple[TimeSeries, TimeSeries]],
                                  eval_fn: Callable[[TimeSeries, TimeSeries], float]) -> List[float]:
    """
    Returns a list of results obtained when calling [eval_fn] on the validation time series in [train_val_series] as
    well as on the corresponding prediction obtained from [model]

    TODO: option for point-prediction
    """

    results = []
    for train, val in train_val_series:
        model.fit(train)
        pred = model.predict_interval(val.duration())
        results.append(eval_fn(val, pred))
    return results
