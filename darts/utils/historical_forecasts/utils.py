from types import SimpleNamespace
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd

from darts.logging import get_logger, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts.utils.utils import series2seq

logger = get_logger(__name__)


def _historical_forecasts_general_checks(series: TimeSeries, kwargs):
    """
    Performs checks common to ForecastingModel and RegressionModel backtest() methods
    Parameters
    ----------
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


def _historical_forecasts_start_warnings(
    idx: int,
    start: Union[pd.Timestamp, int],
    start_time_: Union[int, pd.Timestamp],
    historical_forecasts_time_index: Union[
        pd.DatetimeIndex, pd.RangeIndex, Tuple[Any, Any]
    ],
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

    For TimeSeries with a RangeIndex, the boundaries are converted to absolute indexes to slice the array appropriately
    when start > 0.

    When applicable, move the start boundaries to the value provided by the user.
    """
    # obtain forecastable indexes boundaries, as values from the time index
    historical_forecasts_time_index = model._get_historical_forecastable_time_index(
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
        return ()

    # shift the end of the forecastable index based on `overlap_end`` and `forecast_horizon``
    last_valid_pred_time = model._get_last_prediction_time(
        series,
        forecast_horizon,
        overlap_end,
    )

    historical_forecasts_time_index = (
        historical_forecasts_time_index[0],
        min(historical_forecasts_time_index[1], last_valid_pred_time),
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

    # re-adjust the slicing indexes to account for the lags
    (
        min_target_lag,
        _,
        min_past_cov_lag,
        _,
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
    hist_fct_pc_end = hist_fct_tgt_end
    # future lags can be anything
    hist_fct_fc_start, hist_fct_fc_end = historical_forecasts_time_index
    if min_future_cov_lag is not None and min_future_cov_lag < 0:
        hist_fct_fc_start += min_future_cov_lag * freq
    if max_future_cov_lag is not None and max_future_cov_lag > 0:
        hist_fct_fc_end += max_future_cov_lag * freq

    # convert relative integer index to absolute, make end bound inclusive
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
