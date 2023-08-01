from types import SimpleNamespace
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from darts.logging import get_logger, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts.utils.timeseries_generation import generate_index
from darts.utils.utils import series2seq

logger = get_logger(__name__)

TimeIndex = Union[
    pd.DatetimeIndex,
    pd.RangeIndex,
    Tuple[int, int],
    Tuple[pd.Timestamp, pd.Timestamp],
]


def _historical_forecasts_general_checks(model, series, kwargs):
    """
    Performs checks common to ForecastingModel and RegressionModel backtest() methods
    Parameters
    ----------
    model
        The forecasting model.
    series
        Either series when called from ForecastingModel, or target_series if called from RegressionModel
    signature_params
        A dictionary of the signature parameters of the calling method, to get the default values
        Typically would be signature(self.backtest).parameters
    kwargs
        Params specified by the caller of backtest(), they take precedence over the arguments' default values
    """
    # parse kwargs
    n = SimpleNamespace(**kwargs)

    # check forecast horizon
    raise_if_not(
        n.forecast_horizon > 0,
        "The provided forecasting horizon must be a positive integer.",
        logger,
    )

    # check stride
    raise_if_not(
        n.stride > 0,
        "The provided stride parameter must be a positive integer.",
        logger,
    )

    series = series2seq(series)

    if n.start is not None:
        # check start parameter in general (non series dependent)
        if not isinstance(n.start, (float, int, np.int64, pd.Timestamp)):
            raise_log(
                TypeError(
                    "`start` needs to be either `float`, `int`, `pd.Timestamp` or `None`"
                ),
                logger,
            )
        if isinstance(n.start, float):
            raise_if_not(
                0.0 <= n.start <= 1.0, "`start` should be between 0.0 and 1.0.", logger
            )
        elif isinstance(n.start, (int, np.int64)):
            raise_if_not(
                n.start >= 0, "if `start` is an integer, must be `>= 0`.", logger
            )

        # verbose error messages
        if not isinstance(n.start, pd.Timestamp):
            start_value_msg = f"`start` value `{n.start}` corresponding to timestamp"
        else:
            start_value_msg = "`start` time"
        for idx, series_ in enumerate(series):
            # check specifically for int and Timestamp as error by `get_timestamp_at_point` is too generic
            if isinstance(n.start, pd.Timestamp):
                if n.start > series_.end_time():
                    raise_log(
                        ValueError(
                            f"`start` time `{n.start}` is after the last timestamp `{series_.end_time()}` of the "
                            f"series at index: {idx}."
                        ),
                        logger,
                    )
                elif n.start < series_.start_time():
                    raise_log(
                        ValueError(
                            f"`start` time `{n.start}` is before the first timestamp `{series_.start_time()}` of the "
                            f"series at index: {idx}."
                        ),
                        logger,
                    )
            elif isinstance(n.start, (int, np.int64)):
                if (
                    series_.has_datetime_index
                    or (series_.has_range_index and series_.freq == 1)
                ) and n.start >= len(series_):
                    raise_log(
                        ValueError(
                            f"`start` index `{n.start}` is out of bounds for series of length {len(series_)} "
                            f"at index: {idx}."
                        ),
                        logger,
                    )
                elif (
                    series_.has_range_index and series_.freq > 1
                ) and n.start > series_.time_index[-1]:
                    raise_log(
                        ValueError(
                            f"`start` index `{n.start}` is larger than the last index `{series_.time_index[-1]}` "
                            f"for series at index: {idx}."
                        ),
                        logger,
                    )

            start = series_.get_timestamp_at_point(n.start)
            if n.retrain is not False and start == series_.start_time():
                raise_log(
                    ValueError(
                        f"{start_value_msg} `{start}` is the first timestamp of the series {idx}, resulting in an "
                        f"empty training set."
                    ),
                    logger,
                )

            # check that overlap_end and start together form a valid combination
            overlap_end = n.overlap_end
            if not overlap_end and not (
                start + (series_.freq * (n.forecast_horizon - 1)) in series_
            ):
                raise_log(
                    ValueError(
                        f"{start_value_msg} `{start}` is too late in the series {idx} to make any predictions with "
                        f"`overlap_end` set to `False`."
                    ),
                    logger,
                )

    # check direct likelihood parameter prediction before fitting a model
    if n.predict_likelihood_parameters:
        if not model.supports_likelihood_parameter_prediction:
            raise_log(
                ValueError(
                    f"Model `{model.__class__.__name__}` does not support `predict_likelihood_parameters=True`. "
                    f"Either the model does not support likelihoods, or no `likelihood` was used at model "
                    f"creation."
                )
            )
        if n.num_samples != 1:
            raise_log(
                ValueError(
                    f"`predict_likelihood_parameters=True` is only supported for `num_samples=1`, "
                    f"received {n.num_samples}."
                ),
                logger,
            )

        if (
            model.output_chunk_length is not None
            and n.forecast_horizon > model.output_chunk_length
        ):
            raise_log(
                ValueError(
                    "`predict_likelihood_parameters=True` is only supported for `forecast_horizon` smaller than or "
                    "equal to model's `output_chunk_length`."
                ),
                logger,
            )


def _historical_forecasts_start_warnings(
    idx: int,
    start: Union[pd.Timestamp, int],
    start_time_: Union[int, pd.Timestamp],
    historical_forecasts_time_index: TimeIndex,
):
    """Warnings when start value provided by user is not within the forecastable indexes boundaries"""
    if not isinstance(start, pd.Timestamp):
        start_value_msg = f"value `{start}` corresponding to timestamp `{start_time_}`"
    else:
        start_value_msg = f"time `{start_time_}`"

    if start_time_ < historical_forecasts_time_index[0]:
        logger.warning(
            f"`start` {start_value_msg} is before the first predictable/trainable historical "
            f"forecasting point for series at index: {idx}. Ignoring `start` for this series and "
            f"beginning at first trainable/predictable time: {historical_forecasts_time_index[0]}. "
            f"To hide these warnings, set `show_warnings=False`."
        )
    else:
        logger.warning(
            f"`start` {start_value_msg} is after the last trainable/predictable historical "
            f"forecasting point for series at index: {idx}. This would results in empty historical "
            f"forecasts. Ignoring `start` for this series and beginning at first trainable/"
            f"predictable time: {historical_forecasts_time_index[0]}. Non-empty forecasts can be "
            f"generated by setting `start` value to times between (including): "
            f"{historical_forecasts_time_index[0], historical_forecasts_time_index[-1]}. "
            f"To hide these warnings, set `show_warnings=False`."
        )


def _get_historical_forecastable_time_index(
    model,
    series: TimeSeries,
    past_covariates: Optional[TimeSeries] = None,
    future_covariates: Optional[TimeSeries] = None,
    is_training: Optional[bool] = False,
    reduce_to_bounds: bool = False,
) -> Union[
    pd.DatetimeIndex,
    pd.RangeIndex,
    Tuple[int, int],
    Tuple[pd.Timestamp, pd.Timestamp],
    None,
]:
    """
    Private function that returns the largest time_index representing the subset of each timestamps
    for which historical forecasts can be made, given the model's properties, the training series
    and the covariates.
        - If ``None`` is returned, there is no point where a forecast can be made.

        - If ``is_training=False``, it returns the time_index subset of predictable timestamps.

        - If ``is_training=True``, it returns the time_index subset of trainable timestamps. A trainable
        timestamp is a timestamp that has a training sample of length at least ``self.training_sample_length``
            preceding it.


    Parameters
    ----------
    series
        A target series.
    past_covariates
        Optionally, a past covariates.
    future_covariates
        Optionally, a future covariates.
    is_training
        Whether the returned time_index should be taking into account the training.
    reduce_to_bounds
        Whether to only return the minimum and maximum historical forecastable index

    Returns
    -------
    Union[pd.DatetimeIndex, pd.RangeIndex, Tuple[int, int], Tuple[pd.Timestamp, pd.Timestamp], None]
        The longest time_index that can be used for historical forecasting, either as a range or a tuple.

    Examples
    --------
    >>> model = LinearRegressionModel(lags=3, output_chunk_length=2)
    >>> model.fit(train_series)
    >>> series = TimeSeries.from_times_and_values(pd.date_range('2000-01-01', '2000-01-10'), np.arange(10))
    >>> model._get_historical_forecastable_time_index(series=series, is_training=False)
    DatetimeIndex(['2000-01-04', '2000-01-05', '2000-01-06', '2000-01-07',
            '2000-01-08', '2000-01-09', '2000-01-10'],
            dtype='datetime64[ns]', freq='D')
    >>> model._get_historical_forecastable_time_index(series=series, is_training=True)
    DatetimeIndex(['2000-01-06', '2000-01-08', '2000-01-09', '2000-01-10'],
            dtype='datetime64[ns]', freq='D')

    >>> model = NBEATSModel(input_chunk_length=3, output_chunk_length=3)
    >>> model.fit(train_series, train_past_covariates)
    >>> series = TimeSeries.from_times_and_values(pd.date_range('2000-10-01', '2000-10-09'), np.arange(8))
    >>> past_covariates = TimeSeries.from_times_and_values(pd.date_range('2000-10-03', '2000-10-20'),
    np.arange(18))
    >>> model._get_historical_forecastable_time_index(series=series, past_covariates=past_covariates,
    is_training=False)
    DatetimeIndex(['2000-10-06', '2000-10-07', '2000-10-08', '2000-10-09'], dtype='datetime64[ns]', freq='D')
    >>> model._get_historical_forecastable_time_index(series=series, past_covariates=past_covariates,
    is_training=True)
    DatetimeIndex(['2000-10-09'], dtype='datetime64[ns]', freq='D') # Only one point is trainable, and
    # corresponds to the first point after we reach a common subset of timestamps of training_sample_length length.
    """

    (
        min_target_lag,
        max_target_lag,
        min_past_cov_lag,
        max_past_cov_lag,
        min_future_cov_lag,
        max_future_cov_lag,
    ) = model.extreme_lags

    if min_target_lag is None:
        min_target_lag = 0

    # longest possible time index for target
    start = (
        series.start_time() + (max_target_lag - min_target_lag + 1) * series.freq
        if is_training
        else series.start_time() - min_target_lag * series.freq
    )
    end = series.end_time() + 1 * series.freq

    intersect_ = (start, end)

    # longest possible time index for past covariates
    if (min_past_cov_lag is not None) and (past_covariates is not None):
        start_pc = (
            past_covariates.start_time()
            - (min_past_cov_lag - max_target_lag - 1) * past_covariates.freq
            if is_training
            else past_covariates.start_time() - min_past_cov_lag * past_covariates.freq
        )
        end_pc = past_covariates.end_time() - max_past_cov_lag * past_covariates.freq
        tmp_ = (start_pc, end_pc)

        if intersect_ is not None:
            intersect_ = (
                max([intersect_[0], tmp_[0]]),
                min([intersect_[1], tmp_[1]]),
            )
        else:
            intersect_ = tmp_

    # longest possible time index for future covariates
    if (min_future_cov_lag is not None) and (future_covariates is not None):
        start_fc = (
            future_covariates.start_time()
            - (min_future_cov_lag - max_target_lag - 1) * future_covariates.freq
            if is_training
            else future_covariates.start_time()
            - min_future_cov_lag * future_covariates.freq
        )
        end_fc = (
            future_covariates.end_time() - max_future_cov_lag * future_covariates.freq
        )
        tmp_ = (start_fc, end_fc)

        if intersect_ is not None:
            intersect_ = (
                max([intersect_[0], tmp_[0]]),
                min([intersect_[1], tmp_[1]]),
            )
        else:
            intersect_ = tmp_

    # end comes before the start
    if intersect_[1] < intersect_[0]:
        return None

    # if RegressionModel is not multi_models, it looks further in the past
    is_multi_models = getattr(model, "multi_models", None)
    if is_multi_models is not None and not is_multi_models:
        intersect_ = (
            intersect_[0] + (model.output_chunk_length - 1) * series.freq,
            intersect_[1],
        )

    # generate an index
    if not reduce_to_bounds:
        intersect_ = generate_index(
            start=intersect_[0], end=intersect_[1], freq=series.freq
        )

    return intersect_ if len(intersect_) > 0 else None


def _adjust_historical_forecasts_time_index(
    model,
    series: TimeSeries,
    series_idx: int,
    historical_forecasts_time_index: TimeIndex,
    forecast_horizon: int,
    overlap_end: bool,
    start: Optional[Union[pd.Timestamp, float, int]],
    show_warnings: bool,
) -> TimeIndex:
    """
    Shrink the beginning and end of the historical forecasts time index based on the values of `start`,
    `forecast_horizon` and `overlap_end`.
    """
    # shift the end of the forecastable index based on `overlap_end`` and `forecast_horizon``
    last_valid_pred_time = model._get_last_prediction_time(
        series,
        forecast_horizon,
        overlap_end,
    )

    historical_forecasts_time_index = (
        historical_forecasts_time_index[0],
        min(historical_forecasts_time_index[-1], last_valid_pred_time),
    )

    # when applicable, shift the start of the forecastable index based on `start`
    if start is not None:
        start_time_ = series.get_timestamp_at_point(start)
        # ignore user-defined `start`
        if (
            not historical_forecasts_time_index[0]
            <= start_time_
            <= historical_forecasts_time_index[-1]
        ):
            if show_warnings:
                _historical_forecasts_start_warnings(
                    idx=series_idx,
                    start=start,
                    start_time_=start_time_,
                    historical_forecasts_time_index=historical_forecasts_time_index,
                )
        else:
            historical_forecasts_time_index = (
                max(historical_forecasts_time_index[0], start_time_),
                historical_forecasts_time_index[1],
            )

    return historical_forecasts_time_index


def _get_historical_forecast_predict_index(
    model,
    series: TimeSeries,
    series_idx: int,
    past_covariates: Optional[TimeSeries],
    future_covariates: Optional[TimeSeries],
) -> TimeIndex:
    """Obtain the boundaries of the predictable time indices, raise an exception if None"""
    historical_forecasts_time_index = _get_historical_forecastable_time_index(
        model,
        series,
        past_covariates,
        future_covariates,
        is_training=False,
        reduce_to_bounds=True,
    )

    if historical_forecasts_time_index is None:
        raise_log(
            ValueError(
                "Cannot build a single input for prediction with the provided model, "
                f"`series` and `*_covariates` at series index: {series_idx}. The minimum "
                "prediction input time index requirements were not met. "
                "Please check the time index of `series` and `*_covariates`."
            )
        )

    return historical_forecasts_time_index


def _get_historical_forecast_train_index(
    model,
    series: TimeSeries,
    series_idx: int,
    past_covariates: Optional[TimeSeries],
    future_covariates: Optional[TimeSeries],
) -> TimeIndex:
    """
    Obtain the boundaries of the time indices usable for training, raise an exception if training is required and
    no indices are available.
    """
    historical_forecasts_time_index = _get_historical_forecastable_time_index(
        model,
        series,
        past_covariates,
        future_covariates,
        is_training=True,
        reduce_to_bounds=True,
    )

    if not model._fit_called and historical_forecasts_time_index is None:
        raise_log(
            ValueError(
                "Cannot build a single input for training with the provided untrained model, "
                f"`series` and `*_covariates` at series index: {series_idx}. The minimum "
                "training input time index requirements were not met. "
                "Please check the time index of `series` and `*_covariates`."
            ),
            logger,
        )

    return historical_forecasts_time_index


def _reconciliate_historical_time_indices(
    model,
    historical_forecasts_time_index_predict: TimeIndex,
    historical_forecasts_time_index_train: TimeIndex,
    series: TimeSeries,
    series_idx: int,
    retrain: Union[bool, int, Callable[..., bool]],
    train_length: Optional[int],
    show_warnings: bool,
) -> Tuple[TimeIndex, Optional[int]]:
    """Depending on the value of retrain, select which time indices will be used during the historical forecasts."""
    train_length_ = None
    if isinstance(retrain, Callable):
        # retain the longer time index, anything can happen
        if (
            historical_forecasts_time_index_train is not None
            and historical_forecasts_time_index_train[0]
            < historical_forecasts_time_index_predict[0]
        ):
            historical_forecasts_time_index = historical_forecasts_time_index_train
        else:
            historical_forecasts_time_index = historical_forecasts_time_index_predict
    elif retrain:
        historical_forecasts_time_index = historical_forecasts_time_index_train
    else:
        historical_forecasts_time_index = historical_forecasts_time_index_predict

    # compute the maximum forecasts time index assuming that `start=None`
    if retrain or (not model._fit_called):
        if train_length and train_length <= len(series):
            train_length_ = train_length
            # we have to start later for larger `train_length`
            step_ahead = max(train_length - model._training_sample_time_index_length, 0)
            if step_ahead:
                historical_forecasts_time_index = (
                    historical_forecasts_time_index[0] + step_ahead * series.freq,
                    historical_forecasts_time_index[-1],
                )

        # if not we start training right away; some models (sklearn) require more than 1
        # training samples, so we start after the first trainable point.
        else:
            if train_length and train_length > len(series) and show_warnings:
                logger.warning(
                    f"`train_length` is larger than the length of series at index: {series_idx}. "
                    f"Ignoring `train_length` and using default behavior where all available time steps up "
                    f"until the end of the expanding training set. "
                    f"To hide these warnings, set `show_warnings=False`."
                )

            train_length_ = None
            if model.min_train_samples > 1:
                historical_forecasts_time_index = (
                    historical_forecasts_time_index[0]
                    + (model.min_train_samples - 1) * series.freq,
                    historical_forecasts_time_index[1],
                )

    return historical_forecasts_time_index, train_length_


def _get_historical_forecast_boundaries(
    model,
    series: TimeSeries,
    series_idx: int,
    past_covariates: Optional[TimeSeries],
    future_covariates: Optional[TimeSeries],
    start: Optional[Union[pd.Timestamp, float, int]],
    forecast_horizon: int,
    overlap_end: bool,
    freq: pd.DateOffset,
    show_warnings: bool = True,
) -> Tuple[Any, ...]:
    """
    Based on the boundaries of the forecastable time index, generates the boundaries of each covariates using the lags.

    For TimeSeries with a RangeIndex, the boundaries are converted to positional indexes to slice the array
    appropriately when start > 0.

    When applicable, move the start boundaries to the value provided by the user.
    """
    # obtain forecastable indexes boundaries, as values from the time index
    historical_forecasts_time_index = _get_historical_forecast_predict_index(
        model, series, series_idx, past_covariates, future_covariates
    )

    # adjust boundaries based on start, forecast_horizon and overlap_end
    historical_forecasts_time_index = _adjust_historical_forecasts_time_index(
        model=model,
        series=series,
        series_idx=series_idx,
        historical_forecasts_time_index=historical_forecasts_time_index,
        forecast_horizon=forecast_horizon,
        overlap_end=overlap_end,
        start=start,
        show_warnings=show_warnings,
    )

    # re-adjust the slicing indexes to account for the lags
    (
        min_target_lag,
        _,
        min_past_cov_lag,
        max_past_cov_lag,
        min_future_cov_lag,
        max_future_cov_lag,
    ) = model.extreme_lags

    # target lags are <= 0
    hist_fct_tgt_start, hist_fct_tgt_end = historical_forecasts_time_index
    if min_target_lag is not None:
        hist_fct_tgt_start += min_target_lag * freq
    hist_fct_tgt_end -= 1 * freq
    # past lags are <= 0
    hist_fct_pc_start, hist_fct_pc_end = historical_forecasts_time_index
    if min_past_cov_lag is not None:
        hist_fct_pc_start += min_past_cov_lag * freq
    if max_past_cov_lag is not None:
        hist_fct_pc_end += max_past_cov_lag * freq
    # future lags can be anything
    hist_fct_fc_start, hist_fct_fc_end = historical_forecasts_time_index
    if min_future_cov_lag is not None:
        hist_fct_fc_start += min_future_cov_lag * freq
    if max_future_cov_lag is not None:
        hist_fct_fc_end += max_future_cov_lag * freq

    # convert actual integer index values (points) to positional index, make end bound inclusive
    if series.has_range_index:
        hist_fct_tgt_start = series.get_index_at_point(hist_fct_tgt_start)
        hist_fct_tgt_end = series.get_index_at_point(hist_fct_tgt_end) + 1
        hist_fct_pc_start = series.get_index_at_point(hist_fct_pc_start)
        hist_fct_pc_end = series.get_index_at_point(hist_fct_pc_end) + 1
        hist_fct_fc_start = series.get_index_at_point(hist_fct_fc_start)
        hist_fct_fc_end = series.get_index_at_point(hist_fct_fc_end) + 1

    return (
        historical_forecasts_time_index[0],
        historical_forecasts_time_index[1],
        hist_fct_tgt_start,
        hist_fct_tgt_end,
        hist_fct_pc_start,
        hist_fct_pc_end,
        hist_fct_fc_start,
        hist_fct_fc_end,
    )
