import inspect
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import (
    BaseDataTransformer,
    FittableDataTransformer,
)
from darts.logging import get_logger, raise_log
from darts.timeseries import TimeSeries
from darts.utils.ts_utils import SeriesType, get_series_seq_type, series2seq
from darts.utils.utils import generate_index, n_steps_between

logger = get_logger(__name__)

TimeIndex = Union[
    pd.DatetimeIndex,
    pd.RangeIndex,
    tuple[int, int],
    tuple[pd.Timestamp, pd.Timestamp],
]


def _historical_forecasts_general_checks(
    model, series, kwargs, is_conformal: bool = False
):
    """
    Performs checks common to ForecastingModel and RegressionModel backtest() methods

    Parameters
    ----------
    model
        The forecasting model.
    series
        Either series when called from ForecastingModel, or target_series if called from RegressionModel
    kwargs
        Params specified by the caller of backtest(), they take precedence over the arguments' default values
    """
    # parse kwargs
    n = SimpleNamespace(**kwargs)

    # check forecast horizon
    if not n.forecast_horizon > 0:
        raise_log(
            ValueError("The provided forecasting horizon must be a positive integer."),
            logger,
        )

    # check stride
    if not n.stride > 0:
        raise_log(
            ValueError("The provided stride parameter must be a positive integer."),
            logger,
        )

    # check stride for ConformalModel
    if is_conformal and (
        n.stride < model.cal_stride or n.stride % model.cal_stride > 0
    ):
        raise_log(
            ValueError(
                f"The provided `stride` parameter must be a round-multiple of `cal_stride={model.cal_stride}` "
                f"and `>=cal_stride`. Received `stride={n.stride}`"
            ),
            logger,
        )

    series = series2seq(series)

    if n.start is not None:
        # check start parameter in general (non series dependent)
        if not isinstance(n.start, (float, int, np.int64, pd.Timestamp)):
            raise_log(
                TypeError(
                    "`start` must be either `float`, `int`, `pd.Timestamp` or `None`."
                ),
                logger,
            )

        if n.start_format not in ["position", "value"]:
            raise_log(
                ValueError(
                    f"`start_format` must be on of ['position', 'value']. Received '{n.start_format}'."
                )
            )
        if n.start_format == "position" and not isinstance(n.start, (int, np.int64)):
            raise_log(
                ValueError(
                    f"Since `start_format='position'`, `start` must be an integer, received {type(n.start)}."
                ),
                logger,
            )
        if isinstance(n.start, float):
            if is_conformal:
                raise_log(
                    ValueError(
                        "`start` of type float is not supported for `ConformalModel`."
                    ),
                    logger,
                )
            if not 0.0 <= n.start <= 1.0:
                raise_log(
                    ValueError("if `start` is a float, must be between 0.0 and 1.0."),
                    logger,
                )

        series_freq = None
        for idx, series_ in enumerate(series):
            start_is_value = False
            # check specifically for int and Timestamp as error by `get_timestamp_at_point` is too generic
            if isinstance(n.start, pd.Timestamp):
                if not series_._has_datetime_index:
                    raise_log(
                        ValueError(
                            "if `start` is a `pd.Timestamp`, all series must be indexed with a `pd.DatetimeIndex`"
                        ),
                        logger,
                    )
                if n.start > series_.end_time():
                    raise_log(
                        ValueError(
                            f"`start` time `{n.start}` is after the last timestamp `{series_.end_time()}` of the "
                            f"series at index: {idx}."
                        ),
                        logger,
                    )
                start_is_value = True
            elif isinstance(n.start, (int, np.int64)):
                if n.start_format == "position" or series_.has_datetime_index:
                    if n.start >= len(series_):
                        raise_log(
                            ValueError(
                                f"`start` position `{n.start}` is out of bounds for series of length {len(series_)} "
                                f"at index: {idx}."
                            ),
                            logger,
                        )
                else:
                    if (
                        n.start > series_.time_index[-1]
                    ):  # format "value" and range index
                        raise_log(
                            ValueError(
                                f"`start` time `{n.start}` is larger than the last index `{series_.time_index[-1]}` "
                                f"for series at index: {idx}."
                            ),
                            logger,
                        )
                    start_is_value = True

            # `ConformalModel` with `start_format='value'` requires all series to have the same frequency
            if is_conformal and start_is_value:
                if series_freq is None:
                    series_freq = series_.freq

                if series_freq != series_.freq:
                    raise_log(
                        ValueError(
                            f"Found mismatching `series` time index frequencies `{series_freq}` and `{series_.freq}`. "
                            f"`start_format='value'` with `ConformalModel` is only supported if all series in "
                            f"`series` have the same frequency."
                        ),
                        logger=logger,
                    )

            # find valid start position relative to the series start time, otherwise raise an error
            start_idx, _ = _get_start_index(
                series_, idx, n.start, n.start_format, n.stride
            )

            # check that `overlap_end` and `start` are a valid combination
            overlap_end = n.overlap_end
            if (
                not overlap_end
                and start_idx + n.forecast_horizon + model.output_chunk_shift
                > len(series_)
            ):
                # verbose error messages
                if n.start_format == "position" or (
                    not isinstance(n.start, pd.Timestamp)
                    and series_._has_datetime_index
                ):
                    start_value_msg = (
                        f"`start` position `{n.start}` corresponding to time"
                    )
                else:
                    start_value_msg = "`start` time"
                start = series_._time_index[start_idx]
                raise_log(
                    ValueError(
                        f"{start_value_msg} `{start}` is too late in the series {idx} to make any predictions with "
                        f"`overlap_end` set to `False`."
                    ),
                    logger,
                )

    # duplication of ForecastingModel.predict() check for the optimized historical forecasts implementations
    if not model.supports_probabilistic_prediction and n.num_samples > 1:
        raise_log(
            ValueError("`num_samples > 1` is only supported for probabilistic models."),
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

    if n.data_transformers is not None:
        # check the type
        if not isinstance(n.data_transformers, dict):
            raise_log(
                ValueError(
                    "`data_transformers` should either `None` or a dictionary.", logger
                )
            )
        # check the keys
        supported_keys = {"series", "past_covariates", "future_covariates"}
        incorrect_keys = set(n.data_transformers.keys()) - supported_keys
        if len(incorrect_keys) > 0:
            raise_log(
                ValueError(
                    f"The keys supported by `data_transformers` are {supported_keys}, received the following "
                    f"incorrect keys: {incorrect_keys}."
                ),
                logger,
            )

        # convert to Pipelines
        data_pipelines = _convert_data_transformers(
            data_transformers=n.data_transformers, copy=False
        )
        # extract pipelines containing at least one fittable element
        fittable_pipelines = [
            transf_ for transf_ in data_pipelines.values() if transf_.fittable
        ]
        # extract pipelines where all the fittable transformer are fitted globally
        global_fit_pipelines = [
            transf_ for transf_ in fittable_pipelines if transf_._global_fit
        ]

        if n.retrain:
            # if more than one series is passed and the pipelines are retrained, they cannot be global
            if n.show_warnings and len(series) > 1 and len(global_fit_pipelines) > 0:
                logger.warning(
                    "When `retrain=True` and multiple series are provided, the fittable `data_transformers` "
                    "are trained on each series independently (`global_fit=True` will be ignored)."
                )
        else:
            # must already be fitted without retraining
            not_fitted_pipelines = [
                name_
                for name_, transf_ in data_pipelines.items()
                if transf_.fittable and not transf_._fit_called
            ]
            if len(not_fitted_pipelines) > 0:
                raise_log(
                    ValueError(
                        "All the fittable entries in `data_transformers` must already be fitted when "
                        f"`retrain=False`, the following entries were not fitted: {', '.join(not_fitted_pipelines)}."
                    ),
                    logger,
                )
            # extract the number of fitted params in each pipeline (already fitted)
            fitted_params_pipelines = [
                max(
                    len(t._fitted_params)
                    for t in pipeline
                    if isinstance(t, FittableDataTransformer)
                )
                for pipeline in data_pipelines.values()
            ]

            if len(series) > 1:
                # if multiple series are passed and the pipelines are not all globally fitted, the number of series must
                # match the number of fitted params in the pipelines
                if len(global_fit_pipelines) != len(fittable_pipelines) and len(
                    series
                ) != max(fitted_params_pipelines):
                    raise_log(
                        ValueError(
                            f"When multiple series are provided, their number should match the number of "
                            f"`TimeSeries` used to fit the data transformers `n={max(fitted_params_pipelines)}` "
                            f"(only relevant for fittable transformers that use `global_fit=False`)."
                        ),
                        logger,
                    )
            else:
                # at least one pipeline was fitted on several series with `global_fit=False` but only
                # one series was passed
                if n.show_warnings and max(fitted_params_pipelines) > 1:
                    logger.warning(
                        "Provided only a single series, but at least one of the `data_transformers` "
                        "that use `global_fit=False` was fitted on multiple `TimeSeries`."
                    )

    if (
        n.sample_weight is not None
        and not isinstance(n.sample_weight, str)
        and model.supports_sample_weight
    ):
        sample_weight = series2seq(n.sample_weight)
        for idx, (series_, sample_weight_) in enumerate(zip(series, sample_weight)):
            is_valid = (
                sample_weight_.freq == series_.freq
                and sample_weight_.start_time() <= series_.start_time()
                and len(sample_weight_) >= len(series_)
            )
            if not is_valid:
                raise_log(
                    ValueError(
                        f"`sample_weight` at series index {idx} must contain at least all times "
                        f"of the corresponding target `series`."
                    ),
                    logger=logger,
                )


def _historical_forecasts_sanitize_kwargs(
    model,
    fit_kwargs: Optional[dict[str, Any]],
    predict_kwargs: Optional[dict[str, Any]],
    retrain: bool,
    show_warnings: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert kwargs to dictionary, check that their content is compatible with called methods."""
    hfc_args = set(inspect.signature(model.historical_forecasts).parameters)
    # replace `forecast_horizon` with `n`
    hfc_args = hfc_args - {"forecast_horizon"}
    hfc_args = hfc_args.union({"n"})

    if fit_kwargs is None:
        fit_kwargs = dict()
    elif retrain:
        fit_kwargs = _historical_forecasts_check_kwargs(
            hfc_args=hfc_args,
            name_kwargs="fit_kwargs",
            dict_kwargs=fit_kwargs,
        )
    elif show_warnings:
        logger.warning(
            "`fit_kwargs` was provided with `retrain=False`, the argument will be ignored."
        )

    if predict_kwargs is None:
        predict_kwargs = dict()
    else:
        predict_kwargs = _historical_forecasts_check_kwargs(
            hfc_args=hfc_args,
            name_kwargs="predict_kwargs",
            dict_kwargs=predict_kwargs,
        )
    return fit_kwargs, predict_kwargs


def _historical_forecasts_check_kwargs(
    hfc_args: set[str],
    name_kwargs: str,
    dict_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Return the kwargs dict without the arguments unsupported by the model method.

    Raise a warning if some argument are not supported and an exception if some arguments interfere with
    historical_forecasts logic.
    """
    invalid_args = set(dict_kwargs).intersection(hfc_args)
    if len(invalid_args) > 0:
        raise_log(
            ValueError(
                f"The following parameters cannot be passed in `{name_kwargs}`: {invalid_args}. "
                f"Make sure to pass them explicitly to the function/method."
            ),
            logger,
        )
    return dict_kwargs


def _get_start_index(
    series: TimeSeries,
    series_idx: int,
    start: Union[pd.Timestamp, int, float],
    start_format: Literal["value", "position"],
    stride: int,
    historical_forecasts_time_index: Optional[TimeIndex] = None,
):
    """Finds a valid historical forecast start point within either `series` or `historical_forecasts_time_index`
    (depending on whether `historical_forecasts_time_index` is passed, denoted as `ref`).

    - If `start` is larger or equal to the first index of `ref`, uses `start` directly.
    - If `start` is before the first index of `ref`, tries to find a start point within `ref` that lies a
      round-multiple `stride` time steps ahead of `start`.

    Raises an error if the new start index from above is larger than the last index in `ref`.

    Parameters
    ----------
    series
        A time series. If `historical_forecasts_time_index` is `None`, will use this series' time index as a reference
        index.
    series_idx
        The sequence index of the `series`.
    start
        The start point for historical forecasts.
    start_format
        The start format for historical forecasts.
    stride
        The stride for historical forecasts.
    historical_forecasts_time_index
        Optionally, the historical forecast index (or the boundaries only) to use as the reference index.
    """
    series_start, series_end = series._time_index[0], series._time_index[-1]
    has_dti = series._has_datetime_index
    # find start position relative to the series start time
    if isinstance(start, float):
        # fraction of series
        rel_start = series.get_index_at_point(start)
    elif start_format == "value" and not (isinstance(start, int) and has_dti):
        # start is a time stamp for DatetimeIndex, and integer for RangeIndex
        rel_start = n_steps_between(start, series_start, freq=series.freq)
    else:
        # start is a positional index
        start: int
        rel_start = start if start >= 0 else len(series) - abs(start)

    # find actual start time
    start_idx = _adjust_start(rel_start, stride)
    _check_start(
        series=series,
        start_idx=start_idx,
        start=start,
        start_format=start_format,
        series_start=series_start,
        ref_start=series_start,
        ref_end=series_end,
        stride=stride,
        series_idx=series_idx,
        is_historical_forecast=False,
    )
    if historical_forecasts_time_index is not None:
        hfc_start, hfc_end = (
            historical_forecasts_time_index[0],
            historical_forecasts_time_index[-1],
        )
        # at this point, we know that `start_idx` is within `series`. Now, find the position of that time step
        # relative to the first forecastable point
        rel_start_hfc = n_steps_between(
            series._time_index[start_idx], hfc_start, freq=series.freq
        )
        # get the positional index of `hfc_start` in `series`
        hfc_start_idx = start_idx - rel_start_hfc
        # potentially, adjust the position to be inside the forecastable points
        hfc_start_idx += _adjust_start(rel_start_hfc, stride)
        _check_start(
            series=series,
            start_idx=hfc_start_idx,
            start=start,
            start_format=start_format,
            series_start=series_start,
            ref_start=hfc_start,
            ref_end=hfc_end,
            stride=stride,
            series_idx=series_idx,
            is_historical_forecast=True,
        )
        start_idx = hfc_start_idx
    return start_idx, rel_start


def _adjust_start(rel_start, stride):
    """If relative start position `rel_start` is negative, then adjust it to the first non-negative index that lies a
    round-multiple of `stride` ahead of `rel_start`
    """
    if rel_start >= 0:
        start_idx = rel_start
    else:
        # if `start` lies before the start time of `series` -> check if there is a valid start point in
        # `series` that is a round-multiple of `stride` ahead of `start`
        start_idx = (
            rel_start
            + (abs(rel_start) // stride + int(abs(rel_start) % stride > 0)) * stride
        )
    return start_idx


def _check_start(
    series: TimeSeries,
    start_idx: int,
    start: Union[pd.Timestamp, int, float],
    start_format: Literal["value", "position"],
    series_start: Union[pd.Timestamp, int],
    ref_start: Union[pd.Timestamp, int],
    ref_end: Union[pd.Timestamp, int],
    stride: int,
    series_idx: int,
    is_historical_forecast: bool,
):
    """Raises an error if the start index (position) is not within the series."""
    if start_idx < len(series):
        return

    if start_format == "position" or (
        not isinstance(start, pd.Timestamp) and series._has_datetime_index
    ):
        start_format_msg = f"position `{start}` corresponding to time "
        if isinstance(start, float):
            # fraction of series
            start = series.get_index_at_point(start)
        elif start >= 0:
            # start >= 0 is relative to the start
            start = series.start_time() + start * series.freq
        else:
            # start < 0 is relative to the end
            start = series.end_time() + (start + 1) * series.freq
    else:
        start_format_msg = "time "
    ref_msg = "" if not is_historical_forecast else "historical forecastable "
    start_new = series_start + start_idx * series.freq
    raise_log(
        ValueError(
            f"`start` {start_format_msg}`{start}` is smaller than the first {ref_msg}time index "
            f"`{ref_start}` for series at index: {series_idx}, and could not find a valid start "
            f"point within the {ref_msg}time index that lies a round-multiple of `stride={stride}` "
            f"ahead of `start` (first inferred start is `{start_new}`, but last {ref_msg}time index "
            f"is `{ref_end}`."
        ),
        logger=logger,
    )


def _get_historical_forecastable_time_index(
    model,
    series: TimeSeries,
    forecast_horizon: int,
    overlap_end: bool,
    past_covariates: Optional[TimeSeries] = None,
    future_covariates: Optional[TimeSeries] = None,
    is_training: Optional[bool] = False,
    reduce_to_bounds: bool = False,
) -> Union[
    pd.DatetimeIndex,
    pd.RangeIndex,
    tuple[int, int],
    tuple[pd.Timestamp, pd.Timestamp],
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

    Additionally, it accounts for auto-regression (forecast_horizon > model.output_chunk_length , and overlap_end.
        - In case of auto-regression, we have to step back the last possible predictable/trainable time step if the
          covariates are too short
        - In case overlap_end=False, the latest possible predictable/trainable time step is shifted back if a
          prediction starting from that point would go beyond the end of `series`.


    Parameters
    ----------
    series
        A target series.
    forecast_horizon
        The forecast horizon for the predictions.
    overlap_end
        Whether the returned forecasts can go beyond the series' end or not.
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
    Union[pd.DatetimeIndex, pd.RangeIndex, tuple[int, int], tuple[pd.Timestamp, pd.Timestamp], None]
        The longest time_index that can be used for historical forecasting, either as a range or a tuple.

    Examples
    --------
    >>> model = LinearRegressionModel(lags=3, output_chunk_length=2)
    >>> model.fit(train_series)
    >>> series = TimeSeries.from_times_and_values(pd.date_range('2000-01-01', '2000-01-10'), np.arange(10))
    >>> model._get_historical_forecastable_time_index(series=series, is_training=False, forecast_horizon=1)
    DatetimeIndex(
            ['2000-01-04', '2000-01-05', '2000-01-06', '2000-01-07', '2000-01-08', '2000-01-09', '2000-01-10'],
            dtype='datetime64[ns]', freq='D'
    )
    >>> model._get_historical_forecastable_time_index(series=series, is_training=True)
    DatetimeIndex(['2000-01-06', '2000-01-08', '2000-01-09', '2000-01-10'], dtype='datetime64[ns]', freq='D')
    >>> model = NBEATSModel(input_chunk_length=3, output_chunk_length=3)
    >>> model.fit(train_series, train_past_covariates)
    >>> series = TimeSeries.from_times_and_values(pd.date_range('2000-10-01', '2000-10-09'), np.arange(8))
    >>> past_covariates = TimeSeries.from_times_and_values(
    >>>     pd.date_range('2000-10-03', '2000-10-20'),
    >>>     np.arange(18)
    >>> )
    >>> model._get_historical_forecastable_time_index(
    >>>     series=series,
    >>>     past_covariates=past_covariates,
    >>>     is_training=False,
    >>>     forecast_horizon=1,
    >>> )
    DatetimeIndex(['2000-10-06', '2000-10-07', '2000-10-08', '2000-10-09'], dtype='datetime64[ns]', freq='D')
    >>>  # Only one point is trainable; it corresponds to the first point after we reach a common subset of
    >>> # timestamps of training_sample_length length.
    >>> model._get_historical_forecastable_time_index(
    >>>     series=series,
    >>>     past_covariates=past_covariates,
    >>>     is_training=True,
    >>> )
    DatetimeIndex(['2000-10-09'], dtype='datetime64[ns]', freq='D')
    """

    (
        min_target_lag,
        max_target_lag,
        min_past_cov_lag,
        max_past_cov_lag,
        min_future_cov_lag,
        max_future_cov_lag,
        output_chunk_shift,
        max_target_lag_train,
    ) = model.extreme_lags

    # max_target_lag < 0 are local models which can predict for n (horizon) -> infinity (no auto-regression)
    is_autoregression = (
        max_target_lag >= 0
        and forecast_horizon > max_target_lag - output_chunk_shift + 1
    )

    if min_target_lag is None:
        min_target_lag = 0

    if is_training and max_target_lag_train is not None:
        # the output lag/window can be different for train and predict modes
        output_lag = max_target_lag_train
    else:
        output_lag = max_target_lag

    # longest possible time index for target
    if is_training:
        start = (
            series.start_time()
            + (output_lag - output_chunk_shift - min_target_lag + 1) * series.freq
        )
    else:
        start = series.start_time() - min_target_lag * series.freq
    end = series.end_time() + 1 * series.freq

    intersect_ = (start, end)

    # longest possible time index for past covariates
    if (min_past_cov_lag is not None) and (past_covariates is not None):
        if is_training:
            start_pc = (
                past_covariates.start_time()
                + (output_lag - output_chunk_shift - min_past_cov_lag + 1)
                * past_covariates.freq
            )
        else:
            start_pc = (
                past_covariates.start_time() - min_past_cov_lag * past_covariates.freq
            )

        shift_pc_end = max_past_cov_lag
        if is_autoregression:
            # we step back in case of auto-regression
            shift_pc_end += forecast_horizon - (max_target_lag - output_chunk_shift + 1)
        end_pc = past_covariates.end_time() - shift_pc_end * past_covariates.freq

        intersect_ = (
            max([intersect_[0], start_pc]),
            min([intersect_[1], end_pc]),
        )

    # longest possible time index for future covariates
    if (min_future_cov_lag is not None) and (future_covariates is not None):
        if is_training:
            start_fc = (
                future_covariates.start_time()
                + (output_lag - output_chunk_shift - min_future_cov_lag + 1)
                * future_covariates.freq
            )
        else:
            start_fc = (
                future_covariates.start_time()
                - min_future_cov_lag * future_covariates.freq
            )

        shift_fc_end = max_future_cov_lag
        if is_autoregression:
            # we step back in case of auto-regression
            shift_fc_end += forecast_horizon - (max_target_lag - output_chunk_shift + 1)
        end_fc = future_covariates.end_time() - shift_fc_end * future_covariates.freq

        intersect_ = (
            max([intersect_[0], start_fc]),
            min([intersect_[1], end_fc]),
        )

    # overlap_end = False -> predictions must not go beyond end of target series
    if (
        not overlap_end
        and intersect_[1] + (forecast_horizon + output_chunk_shift - 1) * series.freq
        > series.end_time()
    ):
        intersect_ = (
            intersect_[0],
            end - (forecast_horizon + output_chunk_shift) * series.freq,
        )

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
    series: TimeSeries,
    series_idx: int,
    historical_forecasts_time_index: TimeIndex,
    start: Optional[Union[pd.Timestamp, float, int]],
    start_format: Literal["position", "value"],
    stride: int,
    show_warnings: bool,
) -> TimeIndex:
    """
    Shrink the beginning and end of the historical forecasts time index based on the value of `start`.
    """
    # retrieve actual start
    # when applicable, shift the start of the forecastable index based on `start`
    if start is not None:
        # find valid start position relative to the hfc start time, otherwise raise an error
        start_idx, start_idx_orig = _get_start_index(
            series=series,
            series_idx=series_idx,
            start=start,
            start_format=start_format,
            stride=stride,
            historical_forecasts_time_index=historical_forecasts_time_index,
        )
        start_time = series._time_index[start_idx]

        if start_idx != start_idx_orig and show_warnings:
            if start_idx_orig >= 0:
                start_time_orig = series._time_index[start_idx_orig]
            else:
                start_time_orig = series.start_time() + start_idx_orig * series.freq

            if start_format == "position" or (
                not isinstance(start, pd.Timestamp) and series._has_datetime_index
            ):
                start_value_msg = (
                    f"position `{start}` corresponding to time `{start_time_orig}`"
                )
            else:
                start_value_msg = f"time `{start_time_orig}`"
            logger.warning(
                f"`start` {start_value_msg} is before the first predictable/trainable historical "
                f"forecasting point for series at index: {series_idx}. Using the first historical forecasting "
                f"point `{start_time}` that lies a round-multiple of `stride={stride}` "
                f"ahead of `start`. To hide these warnings, set `show_warnings=False`."
            )
        historical_forecasts_time_index = (
            max(historical_forecasts_time_index[0], start_time),
            historical_forecasts_time_index[1],
        )
    return historical_forecasts_time_index


def _get_historical_forecast_predict_index(
    model,
    series: TimeSeries,
    series_idx: int,
    past_covariates: Optional[TimeSeries],
    future_covariates: Optional[TimeSeries],
    forecast_horizon: int,
    overlap_end: bool,
) -> TimeIndex:
    """Obtain the boundaries of the predictable time indices, raise an exception if None"""
    historical_forecasts_time_index = _get_historical_forecastable_time_index(
        model=model,
        series=series,
        forecast_horizon=forecast_horizon,
        overlap_end=overlap_end,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
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
    forecast_horizon: int,
    overlap_end: bool,
) -> TimeIndex:
    """
    Obtain the boundaries of the time indices usable for training, raise an exception if training is required and
    no indices are available.
    """
    historical_forecasts_time_index = _get_historical_forecastable_time_index(
        model=model,
        series=series,
        forecast_horizon=forecast_horizon,
        overlap_end=overlap_end,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
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
) -> tuple[TimeIndex, Optional[int]]:
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
    start_format: Literal["position", "value"],
    forecast_horizon: int,
    overlap_end: bool,
    stride: int,
    freq: pd.DateOffset,
    show_warnings: bool = True,
) -> tuple[Any, ...]:
    """
    Based on the boundaries of the forecastable time index, generates the boundaries of each covariates using the lags.

    For TimeSeries with a RangeIndex, the boundaries are converted to positional indexes to slice the array
    appropriately when start > 0.

    When applicable, move the start boundaries to the value provided by the user.
    """
    # obtain forecastable indexes boundaries, as values from the time index
    historical_forecasts_time_index = _get_historical_forecast_predict_index(
        model,
        series,
        series_idx,
        past_covariates,
        future_covariates,
        forecast_horizon,
        overlap_end,
    )

    # adjust boundaries based on start
    historical_forecasts_time_index = _adjust_historical_forecasts_time_index(
        series=series,
        series_idx=series_idx,
        historical_forecasts_time_index=historical_forecasts_time_index,
        start=start,
        start_format=start_format,
        stride=stride,
        show_warnings=show_warnings,
    )

    # re-adjust the slicing indexes to account for the lags
    # `max_target_lag_train` is redundant, since optimized hist fc is running in predict mode only
    (
        min_target_lag,
        _,
        min_past_cov_lag,
        max_past_cov_lag,
        min_future_cov_lag,
        max_future_cov_lag,
        output_chunk_shift,
        max_target_lag_train,
    ) = model.extreme_lags

    # target lags are <= 0
    hist_fct_tgt_start, hist_fct_tgt_end = historical_forecasts_time_index
    if min_target_lag is not None:
        hist_fct_tgt_start += min_target_lag * freq

    # target lag has a gap between the max lag and the present
    if hasattr(model, "lags") and model._get_lags("target"):
        hist_fct_tgt_end += 1 * freq * model._get_lags("target")[-1]
    else:
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
        if past_covariates is not None:
            hist_fct_pc_start = past_covariates.get_index_at_point(hist_fct_pc_start)
            hist_fct_pc_end = past_covariates.get_index_at_point(hist_fct_pc_end) + 1
        else:
            hist_fct_pc_start, hist_fct_pc_end = None, None
        if future_covariates is not None:
            hist_fct_fc_start = future_covariates.get_index_at_point(hist_fct_fc_start)
            hist_fct_fc_end = future_covariates.get_index_at_point(hist_fct_fc_end) + 1
        else:
            hist_fct_fc_start, hist_fct_fc_end = None, None

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


def _check_optimizable_historical_forecasts_global_models(
    model,
    forecast_horizon: int,
    retrain: Union[bool, int, Callable[..., bool]],
    show_warnings: bool,
    allow_autoregression: bool,
) -> bool:
    """
    Historical forecast can be optimized only if `retrain=False`. If `allow_autoregression=False`, historical forecasts
    can be optimized only if `forecast_horizon <= model.output_chunk_length` (no auto-regression required).
    """

    retrain_off = (retrain is False) or (retrain == 0)
    is_autoregressive = forecast_horizon > model.output_chunk_length
    if retrain_off and (
        not is_autoregressive or (is_autoregressive and allow_autoregression)
    ):
        return True

    if show_warnings:
        if not retrain_off:
            logger.warning(
                "`enable_optimization=True` is ignored because `retrain` is not `False` or `0`. "
                "To hide this warning, set `show_warnings=False` or `enable_optimization=False`."
            )
        if is_autoregressive:
            logger.warning(
                "`enable_optimization=True` is ignored because `forecast_horizon > model.output_chunk_length`. "
                "To hide this warning, set `show_warnings=False` or `enable_optimization=False`."
            )

    return False


def _process_historical_forecast_input(
    model,
    series: Union[TimeSeries, Sequence[TimeSeries]],
    past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    forecast_horizon: int = 1,
    allow_autoregression: bool = False,
) -> Union[
    Sequence[TimeSeries],
    Optional[Sequence[TimeSeries]],
    Optional[Sequence[TimeSeries]],
    int,
]:
    if not model._fit_called:
        raise_log(
            ValueError("Model has not been fit yet."),
            logger,
        )

    if not allow_autoregression and forecast_horizon > model.output_chunk_length:
        raise_log(
            ValueError(
                "`forecast_horizon > model.output_chunk_length` requires auto-regression which is not "
                "supported in this optimized routine."
            ),
            logger,
        )
    series_seq_type = get_series_seq_type(series)
    series = series2seq(series)
    past_covariates = series2seq(past_covariates)
    future_covariates = series2seq(future_covariates)

    # manage covariates, usually handled by RegressionModel.predict()
    if past_covariates is None and model.past_covariate_series is not None:
        past_covariates = [model.past_covariate_series] * len(series)
    if future_covariates is None and model.future_covariate_series is not None:
        future_covariates = [model.future_covariate_series] * len(series)

    if model.uses_static_covariates:
        model._verify_static_covariates(series[0].static_covariates)

    if model.encoders.encoding_available:
        past_covariates, future_covariates = model.generate_fit_predict_encodings(
            n=forecast_horizon,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
    return series, past_covariates, future_covariates, series_seq_type


def _process_predict_start_points_bounds(
    series: Sequence[TimeSeries], bounds: ArrayLike, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Processes the historical forecastable time index bounds (earliest, and latest possible prediction
    start points).

    Parameters
    ----------
    bounds
        An array of shape (n series, 2), with the left and right prediction start point bounds per series.
    stride
        The number of time steps between two consecutive predictions.

    Returns
    -------
    (np.ndarray, np.ndarray)
        The adjusted bounds: the right bounds are adjusted to be a multiple of 'stride' ahead of the left bounds.
        The number of resulting predicted series per input series respecting stride and bounds.
    """
    bounds = bounds if isinstance(bounds, np.ndarray) else np.array(bounds)
    if not bounds.shape == (len(series), 2):
        raise_log(
            ValueError(
                "`bounds` must be an array like with shape `(n target series, 2)`, "
                "with the start and end bounds of each series"
            ),
            logger=logger,
        )
    # we might have some steps that are too long considering stride
    steps_too_long = (bounds[:, 1] - bounds[:, 0]) % stride
    bounds[:, 1] -= steps_too_long
    cum_lengths = np.cumsum(np.diff(bounds) // stride + 1)
    return bounds, cum_lengths


def _convert_data_transformers(
    data_transformers: Optional[dict[str, Union[BaseDataTransformer, Pipeline]]],
    copy: bool,
) -> dict[str, Pipeline]:
    if data_transformers is None:
        return dict()
    else:
        return {
            key_: val_
            if isinstance(val_, Pipeline)
            else Pipeline(transformers=[val_], copy=copy)
            for key_, val_ in data_transformers.items()
        }


def _apply_data_transformers(
    series: Union[TimeSeries, list[TimeSeries]],
    past_covariates: Optional[Union[TimeSeries, list[TimeSeries]]],
    future_covariates: Optional[Union[TimeSeries, list[TimeSeries]]],
    data_transformers: dict[str, Pipeline],
    max_future_cov_lag: int,
    fit_transformers: bool,
) -> tuple[
    Union[TimeSeries, list[TimeSeries]],
    Union[TimeSeries, list[TimeSeries]],
    Union[TimeSeries, list[TimeSeries]],
]:
    """Transform each series using the corresponding Pipeline.

    If the Pipeline is fittable and `fit_transformers=True`, the series are sliced to correspond
    to the information available at model training time
    """
    # `global_fit`` is not supported, requires too complex time indexes manipulation across series (slice and align)
    if fit_transformers and any(
        not (isinstance(ts, TimeSeries) or ts is None)
        for ts in [series, past_covariates, future_covariates]
    ):
        raise_log(
            ValueError(
                "Fitting the data transformers on multiple series is not supported, either provide trained "
                "`data_transformers` or a single series (including for the covariates).",
                logger,
            )
        )
    transformed_ts = []
    for ts_type, ts in zip(
        ["series", "past_covariates", "future_covariates"],
        [series, past_covariates, future_covariates],
    ):
        if ts is None or data_transformers.get(ts_type) is None:
            transformed_ts.append(ts)
        else:
            if fit_transformers and data_transformers[ts_type].fittable:
                # must slice the ts to distinguish accessible information from future information
                if ts_type == "past_covariates":
                    # known information is aligned with the target series
                    tmp_ts = ts.drop_after(series.end_time())
                elif ts_type == "future_covariates":
                    # known information goes up to the first forecasts iteration (in case of autoregression)
                    tmp_ts = ts.drop_after(
                        series.end_time() + max(0, max_future_cov_lag + 1) * series.freq
                    )
                else:
                    # nothing to do, the target series is already sliced appropriately
                    tmp_ts = ts
                data_transformers[ts_type].fit(tmp_ts)
            # transforming the series
            transformed_ts.append(data_transformers[ts_type].transform(ts))
    return tuple(transformed_ts)


def _apply_inverse_data_transformers(
    series: Union[TimeSeries, Sequence[TimeSeries]],
    forecasts: Union[TimeSeries, list[TimeSeries], list[list[TimeSeries]]],
    data_transformers: dict[str, Pipeline],
    series_idx: Optional[int] = None,
) -> Union[TimeSeries, list[TimeSeries], list[list[TimeSeries]]]:
    """
    Apply the inverse transform to the forecasts when defined.

    `series_idx` is used to retrieve the appropriate transformer when the data transformer was
    fitted with several series and global_fit=False.
    """
    if "series" in data_transformers and data_transformers["series"].invertible:
        called_with_single_series = get_series_seq_type(series) == SeriesType.SINGLE
        if called_with_single_series:
            forecasts = [forecasts]
        forecasts = data_transformers["series"].inverse_transform(
            forecasts, series_idx=series_idx
        )
        return forecasts[0] if called_with_single_series else forecasts
    else:
        return forecasts


def _process_historical_forecast_for_backtest(
    series: Union[TimeSeries, Sequence[TimeSeries]],
    historical_forecasts: Union[
        TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]
    ],
    last_points_only: bool,
):
    """Checks that the `historical_forecasts` have the correct format based on the input `series` and
    `last_points_only`. If all checks have passed, it converts `series` and `historical_forecasts` format into a
    multiple series case with `last_points_only=False`.
    """
    # remember input series type
    series_seq_type = get_series_seq_type(series)
    series = series2seq(series)

    # check that `historical_forecasts` have correct type
    expected_seq_type = None
    forecast_seq_type = get_series_seq_type(historical_forecasts)
    if last_points_only and not series_seq_type == forecast_seq_type:
        # lpo=True -> fc sequence type must be the same
        expected_seq_type = series_seq_type
    elif not last_points_only and forecast_seq_type != series_seq_type + 1:
        # lpo=False -> fc sequence type must be one order higher
        expected_seq_type = series_seq_type + 1

    if expected_seq_type is not None:
        raise_log(
            ValueError(
                f"Expected `historical_forecasts` of type {expected_seq_type} "
                f"with `last_points_only={last_points_only}` and `series` of type "
                f"{series_seq_type}. However, received `historical_forecasts` of type "
                f"{forecast_seq_type}. Make sure to pass the same `last_points_only` "
                f"value that was used to generate the historical forecasts."
            ),
            logger=logger,
        )

    # we must wrap each fc in a list if `last_points_only=True`
    nested = last_points_only and forecast_seq_type == SeriesType.SEQ
    historical_forecasts = series2seq(
        historical_forecasts, seq_type_out=SeriesType.SEQ_SEQ, nested=nested
    )

    # check that the number of series-specific forecasts corresponds to the
    # number of series in `series`
    if len(series) != len(historical_forecasts):
        error_msg = (
            f"Mismatch between the number of series-specific `historical_forecasts` "
            f"(n={len(historical_forecasts)}) and the number of  `TimeSeries` in `series` "
            f"(n={len(series)}). For `last_points_only={last_points_only}`, expected "
        )
        expected_seq_type = series_seq_type if last_points_only else series_seq_type + 1
        if expected_seq_type == SeriesType.SINGLE:
            error_msg += f"a single `historical_forecasts` of type {expected_seq_type}."
        else:
            error_msg += f"`historical_forecasts` of type {expected_seq_type} with length n={len(series)}."
        raise_log(
            ValueError(error_msg),
            logger=logger,
        )
    return series, historical_forecasts


def _extend_series_for_overlap_end(
    series: Sequence[TimeSeries],
    historical_forecasts: Sequence[Sequence[TimeSeries]],
):
    """Extends each target `series` to the end of the last historical forecast for that series.
    Fills the values all missing dates with `np.nan`.

    Assumes the input meets the multiple `series` case with `last_points_only=False` (e.g. the output of
    `darts.utils.historical_forecasts.utils_process_historical_forecast_for_backtest()`).
    """
    series_extended = []
    append_vals = [np.nan] * series[0].n_components
    for series_, hfcs_ in zip(series, historical_forecasts):
        # find number of missing target time steps based on the last forecast
        missing_steps = n_steps_between(
            hfcs_[-1].end_time(), series[0].end_time(), freq=series[0].freq
        )
        # extend the target if it is too short
        if missing_steps > 0:
            series_extended.append(
                series_.append_values(np.array([append_vals] * missing_steps))
            )
        else:
            series_extended.append(series_)
    return series_extended
