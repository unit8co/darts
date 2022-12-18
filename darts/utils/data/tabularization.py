import warnings
from functools import reduce
from math import inf
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided

from darts.logging import get_logger, raise_if, raise_if_not
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


def _create_lagged_data(
    target_series: Union[TimeSeries, Sequence[TimeSeries]],
    output_chunk_length: int,
    past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    lags: Optional[Sequence[int]] = None,
    lags_past_covariates: Optional[Sequence[int]] = None,
    lags_future_covariates: Optional[Sequence[int]] = None,
    max_samples_per_ts: Optional[int] = None,
    is_training: Optional[bool] = True,  # other option: 'inference
    multi_models: Optional[bool] = True,
):
    """
    Helper function that creates training/validation matrices (X and y as required in sklearn), given series and
    max_samples_per_ts.

    X has the following structure:
    lags_target | lags_past_covariates | lags_future_covariates

    Where each lags_X has the following structure (lags_X=[-2,-1] and X has 2 components):
    lag_-2_comp_1_X | lag_-2_comp_2_X | lag_-1_comp_1_X | lag_-1_comp_2_X

    y has the following structure (output_chunk_length=4 and target has 2 components):
    lag_+0_comp_1_target | lag_+0_comp_2_target | ... | lag_+3_comp_1_target | lag_+3_comp_2_target

    Parameters
    ----------
    target_series
        The target series of the regression model.
    output_chunk_length
        The output_chunk_length of the regression model.
    past_covariates
        Optionally, the past covariates of the regression model.
    future_covariates
        Optionally, the future covariates of the regression model.
    lags
        Optionally, the lags of the target series to be used as features.
    lags_past_covariates
        Optionally, the lags of the past covariates to be used as features.
    lags_future_covariates
        Optionally, the lags of the future covariates to be used as features.
    max_samples_per_ts
        Optionally, the maximum number of samples to be drawn for training/validation
        The kept samples are the most recent ones.
    is_training
        Optionally, whether the data is used for training or inference.
        If inference, the rows where the future_target_lags are NaN are not removed from X,
        as we are only interested in the X matrix to infer the future target values.
    """

    # ensure list of TimeSeries format
    if isinstance(target_series, TimeSeries):
        target_series = [target_series]
        past_covariates = [past_covariates] if past_covariates else None
        future_covariates = [future_covariates] if future_covariates else None

    Xs, ys, Ts = [], [], []

    # iterate over series
    for idx, target_ts in enumerate(target_series):
        covariates = [
            (
                past_covariates[idx].pd_dataframe(copy=False)
                if past_covariates
                else None,
                lags_past_covariates,
            ),
            (
                future_covariates[idx].pd_dataframe(copy=False)
                if future_covariates
                else None,
                lags_future_covariates,
            ),
        ]

        df_X = []
        df_y = []
        df_target = target_ts.pd_dataframe(copy=False)

        # y: output chunk length lags of target
        if multi_models:
            for future_target_lag in range(output_chunk_length):
                df_y.append(
                    df_target.shift(-future_target_lag).rename(
                        columns=lambda x: f"{x}_output_chunk_length_lag{future_target_lag}"
                    )
                )
        else:
            df_y.append(
                df_target.shift(-output_chunk_length + 1).rename(
                    columns=lambda x: f"{x}_output_chunk_length_lag{output_chunk_length-1}"
                )
            )

        if lags:
            for lag in lags:
                df_X.append(
                    df_target.shift(-lag).rename(
                        columns=lambda x: f"{x}_target_lag{lag}"
                    )
                )

        # X: covariate lags
        for covariate_name, (df_cov, lags_cov) in zip(["past", "future"], covariates):
            if lags_cov:
                if not is_training:
                    # We extend the covariates dataframe
                    # so that when we create the lags with shifts
                    # we don't have nan on the last (or first) rows. Only useful for inference.
                    df_cov = df_cov.reindex(df_target.index.union(df_cov.index))

                for lag in lags_cov:
                    df_X.append(
                        df_cov.shift(-lag).rename(
                            columns=lambda x: f"{x}_{covariate_name}_cov_lag{lag}"
                        )
                    )

        # combine lags
        df_X = pd.concat(df_X, axis=1)
        df_y = pd.concat(df_y, axis=1)
        df_X_y = pd.concat([df_X, df_y], axis=1)

        if is_training:
            df_X_y = df_X_y.dropna()
        # We don't need to drop where y are none for inference, as we just care for X
        else:
            df_X_y = df_X_y.dropna(subset=df_X.columns)

        Ts.append(df_X_y.index)
        X_y = df_X_y.values

        # keep most recent max_samples_per_ts samples
        if max_samples_per_ts:
            X_y = X_y[-max_samples_per_ts:]
            Ts[-1] = Ts[-1][-max_samples_per_ts:]

        raise_if(
            X_y.shape[0] == 0,
            "Unable to build any training samples of the target series "
            + (f"at index {idx} " if len(target_series) > 1 else "")
            + "and the corresponding covariate series; "
            "There is no time step for which all required lags are available and are not NaN values.",
        )

        X, y = np.split(X_y, [df_X.shape[1]], axis=1)

        Xs.append(X)
        ys.append(y)

    # combine samples from all series
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y, Ts


#
#   Refactored Implementation:
#


def create_lagged_data(
    target_series: Optional[TimeSeries] = None,
    past_covariates: Optional[TimeSeries] = None,
    future_covariates: Optional[TimeSeries] = None,
    lags: Optional[Sequence[int]] = None,
    lags_past_covariates: Optional[Sequence[int]] = None,
    lags_future_covariates: Optional[Sequence[int]] = None,
    output_chunk_length: int = 1,
    max_samples_per_ts: Optional[int] = None,
    multi_models: bool = True,
    check_inputs: bool = True,
    use_moving_windows: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:

    """
    Creates the features array `X` and labels array `y` to train a lagged-variables regression model (e.g. an
    `sklearn` model); the time index values of each observation is also returned. In order for the lagged features
    of a series to be added to `X`, *both* that series and the corresponding lags must be specified; if a series is
    specified without the corresponding lags, that series will be ignored and not added to `X`. Observations (i.e.
    rows along the `0`th axis of `X` and `y`) are only created for those time points which are common to all of the
    specified series. `X` and `y` arrays are constructed independently over the samples dimension (i.e. the second axis)
    of each series.

    Notes
    -----
    The `X` array is constructed from the lagged values of up to three separate timeseries:
        1. The `target_series`, which contains the values we're trying to predict. A regression model that
        uses previous values
        of the target its predicting is referred to as *auto-regressive*; please refer to [1]_ for further
        details
        about auto-regressive timeseries models.
        2. The past covariates series, which contains values that are *not* known into the future. Unlike
        the target series, however,
        past covariates are *not* to be predicted by the regression model.
        3. The future covariates (AKA 'exogenous' covariates) series, which contains values that are known
        into the future, even beyond
        the data in `target_series` and `past_covariates`.
    See [2]_ for a more detailed discussion about target, past, and future covariates. Conversely, `y` is
    comprised only of the lagged values of `target_series`.

    The shape of `X` is:
        `X.shape = (n_observations, n_lagged_features, n_samples)`,
    where `n_observations` equals either the number of time points shared between all specified series,
    or `max_samples_per_ts`, whichever is smallest. The shape of `y` is:
        `y.shape = (n_observations, output_chunk_length, n_samples)`,
    if `multi_models = True`, otherwise:
        `y.shape = (n_observations, 1, n_samples)`.

    Along the `n_lagged_features` axis, `X` has the following structure (for `*_lags=[-2,-1]` and
    `*_series.n_components = 2`):
        lagged_target | lagged_past_covariates | lagged_future_covariates
    where each `lagged_*` has the following structure:
        lag_-2_comp_1_* | lag_-2_comp_2_* | lag_-1_comp_1_* | lag_-1_comp_2_*

    Along the `n_lagged_labels` axis, `y` has the following structure (for `output_chunk_length=4` and
    `target_series.n_components=2`):
        lag_+0_comp_1_target | lag_+0_comp_2_target | ... | lag_+3_comp_1_target | lag_+3_comp_2_target

    The exact method used to construct `X` and `y` depends on whether all of the specified timeseries are
    of the same frequency or not:
        - If all specified timeseries are of the same frequency, `strided_moving_window` is used to extract
        contiguous time blocks between time `t - max_lag` and `t`
        - If all specified timeseries are *not* of the same frequency, `find_intersecting_times` is first used
        to find those times common to all three timeseries, after which the lagged features are extracted by
        offsetting the time indices of these common times by the requested lags.
    In cases where it can be validly applied, the 'moving window' method is expected to be faster than the
    'intersecting time' method.

    Parameters
    ----------
    target_series
        The series for the regression model to predict.
    output_chunk_length
        The number of timesteps ahead into the future the regression model is to predict.
    past_covariates
        Optionally, the past covariates series that the regression model will use as inputs. Unlike the
        `target_series`, `past_covariates` are *not* to be predicted by the regression model.
    future_covariates
        Optionally, the future covariates (i.e. exogenous covariates) series that the regression model will
        use as inputs.
    lags
        Optionally, the lags of the target series to be used as (auto-regressive) features. If not specified,
        auto-regressive features will *not* be added to `X`.
    lags_past_covariates
        Optionally, the lags of `past_covariates` to be used as features.
    lags_future_covariates
        Optionally, the lags of `future_covariates` to be used as features.
    max_samples_per_ts
        Optionally, the maximum number of samples to be drawn for training/validation; only the most recent
        samples are kept. In theory, specifying a smaller `max_samples_per_ts` should reduce computation time,
        especially in cases where many observations could be generated.
    multi_models
        Optionally, specifies whether the regression model predicts multiple timesteps into the future. If `True`,
        then the regression model is assumed to predict all of the timesteps from time `t` to `t+output_chunk_length`.
        If `False`, then the regression model is assumed to predict *only* the timestep at `t+output_chunk_length`.
    check_inputs
        Optionally, specifies that the `lags_*` and `series_*` inputs should be checked for validity. Caution should be
        exercised when setting `check_inputs = False`.
    use_moving_windows
        Optionally, specifies that the 'moving window method' should be used to construct `X` and `y` if all of the
        provided series are of the same frequency. If `use_moving_windows = False`, the 'time intersection' method
        will always be used, even when all of the provided series are of the same frequency. See Notes for further
        details.

    Returns
    -------
    X
        A `(n_observations, n_lagged_features, n_samples)`-shaped array of lagged features for training a
        lagged-variables model.
    y
        If `multi_models = True`, a `(n_observations, output_chunk_length, n_samples)`-shaped array of lagged
        labels for training a lagged-variables model. Othewise, a `(n_observations, 1, n_samples)`-shaped
        array of lagged labels.
    times
        The `time_index` of each observation in `X` and `y`.

    Raises
    ------
    ValueError
        If there are no times shared by all of the specified series, or if no lags have been specified.
    TypeError
        If the provided series do not share the same type of `time_index` (e.g. `target_series` uses a
        pd.RangeIndex, but `future_covariates` uses a `pd.DatetimeIndex`).

    References
    ----------
    .. [1] https://otexts.com/fpp2/AR.html#AR
    .. [2] https://unit8.com/resources/time-series-forecasting-using-past-and-future-external-data-with-darts/
    """
    if max_samples_per_ts is None:
        max_samples_per_ts = inf
    if use_moving_windows and _all_equal_freq(
        target_series, past_covariates, future_covariates
    ):
        X, y, times = _create_lagged_data_by_moving_window(
            target_series,
            output_chunk_length,
            past_covariates,
            future_covariates,
            lags,
            lags_past_covariates,
            lags_future_covariates,
            max_samples_per_ts,
            multi_models,
            check_inputs,
        )
    else:
        X, y, times = _create_lagged_data_by_intersecting_times(
            target_series,
            output_chunk_length,
            past_covariates,
            future_covariates,
            lags,
            lags_past_covariates,
            lags_future_covariates,
            max_samples_per_ts,
            multi_models,
            check_inputs,
        )
    return X, y, times


def _create_lagged_data_by_moving_window(
    target_series: TimeSeries,
    output_chunk_length: int,
    past_covariates: Optional[TimeSeries] = None,
    future_covariates: Optional[TimeSeries] = None,
    lags: Optional[Sequence[int]] = None,
    lags_past_covariates: Optional[Sequence[int]] = None,
    lags_future_covariates: Optional[Sequence[int]] = None,
    max_samples_per_ts: Optional[int] = inf,
    multi_models: bool = True,
    check_inputs: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Helper function called by `_create_lagged_data` that computes `X`, `y`, and `times` by
    first extracting 'moving window' slices of each series; the size of these windows is
    `max_lag` when constructing `X`, where `max_lag` is the largest lag value specified
    among all of the `lags*` inputs, and `output_chunk_length` (if `multi_models = True`)
    or `1` (if `multi_models = False`) when constructing `y`. The `lags*` requested by the
    user are extracted from the `X` windows. One formed, these windows are then reshaped
    into the correct form. This approach is used if we *can* assume that the specified series
    are all of the same frequency.

    Notes
    -----

    """
    feature_times, max_lags = get_feature_times(
        target_series,
        past_covariates,
        future_covariates,
        lags,
        lags_past_covariates,
        lags_future_covariates,
        output_chunk_length,
        is_training=True,
        check_inputs=check_inputs,
        return_max_lags=True,
    )
    time_bounds = get_shared_times_bounds(*feature_times)
    raise_if(
        time_bounds is None,
        "Specified series do not share any common times for which features can be created.",
    )
    if isinstance(target_series.time_index, pd.DatetimeIndex):
        times = pd.date_range(
            start=time_bounds[0], end=time_bounds[1], freq=target_series.freq
        )
    else:
        freq = target_series.freq
        # Since `stop` is exclusive:
        stop = time_bounds[1] + freq
        times = pd.RangeIndex(start=time_bounds[0], stop=stop, step=freq)
    num_samples = len(times)
    if num_samples > max_samples_per_ts:
        times = times[-max_samples_per_ts:]
        num_samples = max_samples_per_ts
    window_start_time = times[0]
    # Construct features:
    X = []
    for i, (series_i, lags_i, max_lag_i) in enumerate(
        zip(
            [target_series, past_covariates, future_covariates],
            [lags, lags_past_covariates, lags_future_covariates],
            max_lags,
        )
    ):
        series_and_lags_specified = max_lag_i is not None
        is_target_series = i == 0
        if series_and_lags_specified or is_target_series:
            window_start_idx = np.searchsorted(series_i.time_index, window_start_time)
            window_end_idx = window_start_idx + num_samples
        if series_and_lags_specified:
            vals = series_i.all_values(copy=False)[
                window_start_idx - max_lag_i : window_end_idx, :, :
            ]
            windows = strided_moving_window(
                vals, window_len=max_lag_i + 1, stride=1, axis=0, check_inputs=False
            )
            # Last point in each window corresponds to time `t`, so need
            lagged_vals = _extract_lagged_vals_from_windows(
                windows, lags_to_extract=np.array(lags_i, dtype=int) - 1
            )
            X.append(lagged_vals)
        # Required for creating labels:
        if is_target_series:
            # Add back `max_lag_i`, since we don't want lagged values when creating labels:
            label_window_start_idx = window_start_idx
    X = np.concatenate(X, axis=1)
    if multi_models:
        label_window_end_idx = label_window_start_idx + num_samples
        # Need to include end point, so `+1`:
        vals = target_series.all_values(copy=False)[
            label_window_start_idx : label_window_end_idx + output_chunk_length - 1,
            :,
            :,
        ]
        windows = strided_moving_window(
            vals, window_len=output_chunk_length, stride=1, axis=0, check_inputs=False
        )
        y = _extract_lagged_vals_from_windows(windows)
    else:
        label_window_start_idx += output_chunk_length - 1
        label_window_end_idx = label_window_start_idx + num_samples
        y = target_series.all_values(copy=False)[
            label_window_start_idx:label_window_end_idx, :, :
        ]
    return X, y, times


def _extract_lagged_vals_from_windows(
    windows: np.ndarray, lags_to_extract: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Helper function called by `_create_lagged_data_by_moving_window` that
    reshapes the `windows` formed by `strided_moving_window` from the
    shape `(num_windows, num_components, num_series, window_len)` to the
    shape `(num_windows, num_components*window_len, num_series)`. This reshaping
    is done such that the order of elements along axis 1 matches the pattern
    described in the docstring of `create_lagged_data`.

    If `lags_to_extract` is specified, only the values within each extracted window
    wi. For example, if `lags_to_extract = [-2]`, only the second-to-last values within
    each window will be extracted. If `lags_to_extract` is not specified, all of the values
    within each window is extracted.
    """
    # windows.shape = (num_windows, num_components, num_samples, window_len):
    if lags_to_extract is not None:
        windows = windows[:, :, :, lags_to_extract]
    # windows.shape = (num_windows, window_len, num_components, num_samples):
    windows = np.moveaxis(windows, (0, 3, 1, 2), (0, 1, 2, 3))
    # lagged_vals.shape = (num_windows, num_components*window_len, num_samples):
    lagged_vals = windows.reshape(windows.shape[0], -1, windows.shape[-1])
    return lagged_vals


def _create_lagged_data_by_intersecting_times(
    target_series: TimeSeries,
    output_chunk_length: int,
    past_covariates: Optional[TimeSeries] = None,
    future_covariates: Optional[TimeSeries] = None,
    lags: Optional[Sequence[int]] = None,
    lags_past_covariates: Optional[Sequence[int]] = None,
    lags_future_covariates: Optional[Sequence[int]] = None,
    max_samples_per_ts: Optional[int] = inf,
    multi_models: bool = True,
    check_inputs: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Union[pd.RangeIndex, pd.DatetimeIndex]]:
    """
    Helper function called by `_create_lagged_data` that computes `X`, `y`, and `times` by
    finding the time points that are shared by all specified series, offsetting the
    index of these shared time points by the specified `lags` (for `X`) or
    `output_chunk_length` (for `y`), and then finally accessing the feature or label
    values in each timeseries by using the offset indexing arrays. This approach is used if
    we *cannot* assume that the specified series are of the same frequency.
    """
    feature_times, max_lags = get_feature_times(
        target_series,
        past_covariates,
        future_covariates,
        lags,
        lags_past_covariates,
        lags_future_covariates,
        output_chunk_length,
        is_training=True,
        check_inputs=check_inputs,
        return_max_lags=True,
    )
    shared_times = get_shared_times(*feature_times, sort=True)
    raise_if(
        shared_times is None,
        "Specified series do not share any common times for which features can be created.",
    )
    if len(shared_times) > max_samples_per_ts:
        shared_times = shared_times[-max_samples_per_ts:]
    X = []
    for i, (series_i, lags_i, max_lag_i) in enumerate(
        zip(
            [target_series, past_covariates, future_covariates],
            [lags, lags_past_covariates, lags_future_covariates],
            max_lags,
        )
    ):
        series_and_lags_specified = max_lag_i is not None
        is_target_series = i == 0
        if series_and_lags_specified or is_target_series:
            shared_time_idx = np.searchsorted(series_i.time_index, shared_times)
        if series_and_lags_specified:
            idx_to_get = shared_time_idx.reshape(-1, 1) + np.array(lags_i, dtype=int)
            # Before reshaping: lagged_vals.shape = (n_observations, num_lags, n_components, n_samples)
            lagged_vals = series_i.all_values(copy=False)[idx_to_get, :, :]
            # After reshaping: lagged_vals.shape = (n_observations, num_lags*n_components, n_samples)
            lagged_vals = lagged_vals.reshape(
                lagged_vals.shape[0], -1, lagged_vals.shape[-1]
            )
            X.append(lagged_vals)
        # `target_series` indices required for creating labels:
        if is_target_series:
            label_shared_time_idx = shared_time_idx
    X = np.concatenate(X, axis=1)
    if multi_models:
        # All points between time `t` and `t + output_chunk_length` are labels:
        time_idx_to_get = label_shared_time_idx.reshape(-1, 1) + np.arange(
            output_chunk_length
        )
    else:
        # Only point at time `t + output_chunk_length` is a label:
        time_idx_to_get = label_shared_time_idx.reshape(-1, 1) + output_chunk_length - 1
    # Before reshaping: lagged_vals.shape = (n_observations, num_lags, n_components, n_samples)
    lagged_vals = target_series.all_values(copy=False)[time_idx_to_get, :, :]
    # After reshaping: lagged_vals.shape = (n_observations, num_lags*n_components, n_samples)
    y = lagged_vals.reshape(lagged_vals.shape[0], -1, lagged_vals.shape[-1])
    return X, y, shared_times


FeatureTimes = Tuple[
    Union[pd.Index, None], Union[pd.Index, None], Union[pd.Index, None]
]
MaxLags = Tuple[Union[int, None], Union[int, None], Union[int, None]]


def get_feature_times(
    target_series: Optional[TimeSeries] = None,
    past_covariates: Optional[TimeSeries] = None,
    future_covariates: Optional[TimeSeries] = None,
    lags: Optional[Sequence[int]] = None,
    lags_past_covariates: Optional[Sequence[int]] = None,
    lags_future_covariates: Optional[Sequence[int]] = None,
    output_chunk_length: Optional[int] = None,
    is_training: bool = True,
    check_inputs: bool = True,
    return_max_lags: bool = False,
) -> Union[FeatureTimes, Tuple[FeatureTimes, MaxLags]]:
    """
    Returns a tuple containing the times in `target_series`, the times in `past_covariates`, and the times in
    `future_covariates` that *could* be used to create features. More specifically, we note that:
        1. Features cannot be created from times that have fewer than `-min(lags)` preceding values.
        2. When creating training data, labels cannot be created for times points that have fewer than
        `(output_chunk_length - 1)` values ahead of them. Thus, there's no point creating features for these
        time points.
    `get_feature_times` returns all of the times in each series that satisfy:
        - Only the first condition, if `is_training = False`,
        - Both the first and the second condition, if `is_training = True`.
    The returned tuple of times can then be passed to `get_shared_times` to compute the 'eligible time points'
    shared by all of the specified series.

    Notes
    -----
    If `return_max_lags = True`, the largest lag value for each series is also returned as a tuple.

    For those series which are either unspecified, a `None` value takes the place of that series' feature time
    and maximum lag value.

    If `is_training = True`, then `target_series`, `multi_models`, and `output_chunk_length` must be provided.

    Parameters
    ----------
    target_series
        Optionally, the series for the regression model to predict.
    past_covariates
        Optionally, the past covariates series that the regression model will use as inputs. Unlike the
        `target_series`, `past_covariates` are *not* to be predicted by the regression model.
    future_covariates
        Optionally, the future covariates (i.e. exogenous covariates) series that the regression model will
        use as inputs.
    lags
        Optionally, the lags of the target series to be used as (auto-regressive) features. If not specified,
        auto-regressive features will *not* be added to `X`.
    lags_past_covariates
        Optionally, the lags of `past_covariates` to be used as features.
    lags_future_covariates
        Optionally, the lags of `future_covariates` to be used as features.
    output_chunk_length
        Optionally, the number of timesteps ahead into the future the regression model is to predict. This is ignored
        if `is_training = False`.
    multi_models
        Optionally, specifies whether the regression model predicts multiple timesteps into the future. If `True`,
        then the regression model is assumed to predict all of the timesteps from time `t` to `t+output_chunk_length`.
        If `False`, then the regression model is assumed to predict *only* the timestep at `t+output_chunk_length`.
        This is ignored if `is_training = False`.
    is_training
        Optionally, specifies that training data is to be generated from the specified series. If `True`,
        `target_series`, `output_chunk_length`, and `multi_models` must all be specified.
    check_inputs
        Optionally, specifies that the `lags_*` and `series_*` inputs should be checked for validity. Caution should be
        exercised when setting `check_inputs = False`.
    return_max_lags
        Optionally, specifies whether the largest magnitude lag value for each series should also be returned along with
        the 'eligible' feature times

    Returns
    -------
    feature_times
        A tuple containing all the 'eligible feature times' in `target_series`, in `past_covariates`, and in
        `future_covariates`, in that order. If a particular series-lag pair isn't fully specified, then a `None`
        will take the place of that series' eligible times.
    max_lags
        Optionally, a tuple containing the largest lag value in `lags`, `lags_past_covariates`, and
        `lags_future_covariates`, in that order. If a particular series-lag pair isn't fully specified, then a `None`
        will take the place of that series' maximum lag values.

    Raises
    ------
    ValueError
        If `target_series` and `output_chunk_length` are not both specified if `is_training = True`.
    ValueError
        If any of the `lags` inputs contain non-negative values or if none of the `lags` inputs have been specified.
    ValueError
        If any of the series are too short for the requested `lags` and/or `output_chunk_length` values.
    UserWarning
        If a `lags_*` input is specified without the accompanying time series or vice versa. The only expection to this
        is when `lags` isn't specified alongside `target_series` when `is_training = True`, since one may wish to fit
        a regression model without using auto-regressive features.

    """
    raise_if(
        is_training and ((target_series is None) or (output_chunk_length is None)),
        "Must specify `target_series` and `output_chunk_length` when `is_training = True`.",
    )
    if check_inputs:
        _check_lags(lags, lags_past_covariates, lags_future_covariates)
    feature_times, max_lags = [], []
    for name_i, series_i, lags_i in zip(
        ["target_series", "past_covariates", "future_covariates"],
        [target_series, past_covariates, future_covariates],
        [lags, lags_past_covariates, lags_future_covariates],
    ):
        if check_inputs and (series_i is not None):
            _check_series_length(
                series_i,
                lags_i,
                output_chunk_length,
                is_training,
                name_i,
            )
        times_i = series_i.time_index if series_i is not None else None
        max_lag_i = -min(lags_i) if lags_i is not None else None
        series_specified = times_i is not None
        lags_specified = max_lag_i is not None
        is_label_series = is_training and name_i == "target_series"
        if is_label_series:
            end_idx = -output_chunk_length + 1 if output_chunk_length > 1 else None
            times_i = times_i[:end_idx]
        if series_specified and lags_specified:
            times_i = times_i[max_lag_i:]
        elif (not is_label_series) and (series_specified ^ lags_specified):
            times_i = max_lag_i = None
            lags_name = "lags" if name_i == "target_series" else f"lags_{name_i}"
            specified = lags_name if lags_specified else name_i
            unspecified = name_i if lags_specified else lags_name
            warnings.warn(
                f"`{specified}` was specified without accompanying `{unspecified}` and, thus, will be ignored."
            )
        feature_times.append(times_i)
        max_lags.append(max_lag_i)
    return (feature_times, max_lags) if return_max_lags else feature_times


def get_shared_times(
    *series_or_times: Sequence[Union[TimeSeries, pd.Index, None]], sort: bool = True
) -> pd.Index:
    """
    Returns the times found within all of the specified `TimeSeries` or time indexes (i.e. the intersection of all
    these times). If `sort = True`, then these shared times are sorted from earliest to latest. Any `TimeSeries` or
    time indices in `series_or_times` that aren't specified (i.e. are `None`) are simply ignored.

    Parameters
    ----------
    series_or_times
        The `TimeSeries` and/or time indices that should 'intersected'/searched for shared times.
    sort
        Optionally, specifies that the returned should be sorted from earliest to latest.

    Returns
    -------
    shared_times
        The time indices present in all of the specified `TimeSeries` and/or time indices.

    Raises
    ------
    TypeError
        If the specified `TimeSeries` and/or time indices do not all share the same type of time index (i.e. must
        either be all `pd.DatetimeIndex` or all `pd.RangeIndex`).

    """

    # `sort = None` specifies to `pd.Index.intersection` that values should be sorted:
    sort = None if sort else False

    def intersection_func(series_or_times_1, series_or_times_2):
        times_1 = (
            series_or_times_1.time_index
            if isinstance(series_or_times_1, TimeSeries)
            else series_or_times_1
        )
        times_2 = (
            series_or_times_2.time_index
            if isinstance(series_or_times_2, TimeSeries)
            else series_or_times_2
        )
        return times_1.intersection(times_2, sort=sort)

    shared_times = reduce(
        intersection_func, [series for series in series_or_times if series is not None]
    )
    # Empty intersection may result from intersecting time indices being of different types - throw error if so:
    if shared_times.empty:
        times_types = [
            type(ts.time_index if isinstance(ts, TimeSeries) else ts)
            for ts in series_or_times
        ]
        raise_if_not(
            len(set(times_types)) == 1,
            (
                "Time indices must be of the same type"
                "(i.e. all `pd.DaterangeIndex` or all `pd.RangeIndex`)."
            ),
        )
    return shared_times


def get_shared_times_bounds(
    *series_or_times: Sequence[Union[TimeSeries, pd.Index, None]]
) -> Union[Tuple[pd.Index, pd.Index], None]:
    """
    Returns the latest `start_time` and the earliest `end_time` among all of the non-`None` `series_or_times`;
    these are (non-tight) lower and upper `bounds` on the intersection of all these `series_or_times` respectively.
    If no potential overlap exists between all of the specified series, `None` is returned instead.

    Notes
    -----
    If all of the specified `series_or_times` are of the same frequency, then `get_shared_times_bounds`
    returns tight `bounds` (i.e. the earliest and latest time within the intersection of all the timeseries
    is returned). To see this, suppose we have three equal-frequency series with observations made at different
    times:
        Series 1: ------
        Series 2:    ------
        Series 3:  ------
    Here, each `-` denotes as observation at a specific time. In this example, `find_time_overlap_bounds` will
    return the times at `LB` and `UB`:
                    LB
        Series 1: ---|---|
        Series 2:    |---|---
        Series 3:  --|---|-
                         UB
    If the specified timeseries are *not* all of the same frequency, then the returned `bounds` is potentially non-tight
    (i.e. `LB <= intersection.start_time() < intersection.end_time() <= UB`, where `intersection` are the times shared
    by all specified timeseries)

    Parameters
    ----------
    series_or_times
        The `TimeSeries` and/or `pd.Index` values to compute intersection `bounds` for; any provided `None` values
        are ignored.

    Returns
    -------
    bounds
        Tuple containing the latest `start_time` and earliest `end time` among all specified `timeseries`, in that
        order. If no potential overlap exists between the specified series, then `None` is returned instead. Similarly,
        if no non-`None` `series_or_times` were specified, `None` is returned.

    Raises
    ------
    TypeError
        If the series and/or times in `series_or_times` don't all share the same type of `time_index`
        (i.e. either all `pd.DatetimeIndex` or `pd.RangeIndex`).

    """
    start_times, end_times = [], []
    for val in series_or_times:
        if val is not None:
            start_times.append(
                val.start_time() if isinstance(val, TimeSeries) else val[0]
            )
            end_times.append(val.end_time() if isinstance(val, TimeSeries) else val[-1])
    times_types = [type(time) for time in start_times]
    raise_if_not(
        len(set(times_types)) == 1,
        "Specified series and/or times must all have the same type of `time_index`.",
    )
    # If `start_times` empty, no series were specified -> `bounds = (1, -1)` will
    # be 'converted' to `None` in next line:
    bounds = (max(start_times), min(end_times)) if start_times else (1, -1)
    # Specified timeseries share no overlapping periods.
    if bounds[1] <= bounds[0]:
        bounds = None
    return bounds


def strided_moving_window(
    x: np.ndarray,
    window_len: int,
    stride: int = 1,
    axis: int = 0,
    check_inputs: bool = True,
) -> np.ndarray:
    """
    Extracts moving window views of an `x` array along a specified `axis`, where each window is of length `window_len`
    and consecutive windows are separated by `stride` indices. The total number of extracted windows equals
    `num_windows = (x.shape[axis] - window_len)//stride + 1`.

    Notes
    -----
    This function is similar to `sliding_window_view` in `np.lib.stride_tricks`, except that:
        1. `strided_moving_window` allows for consecutive windows to be separated by a specified `stride`,
        whilst `sliding_window_view` does not.
        2. `strided_moving_window` can only operate along a single axis, whereas `sliding_window_view` can
        operate along multiple axes.
    Additionally, unlike `sliding_window_view`, using `strided_moving_window` doesn't require `numpy >= 1.20.0`.

    Parameters
    ----------
    x
        The array from which to extract moving windows.
    window_len
        The size of the extracted moving windows.
    stride
        Optionally, the separation between consecutive windows.
    axis
        Optionally, the axis along which the moving windows should be extracted.
    check_inputs
        Optionally, specifies whether inputs should be checked for validity. Caution should be exercised when
        setting to `False`; see [1]_ for further details.

    Returns
    -------
    windows
        The moving windows extracted from `x`. The extracted windows are stacked along the last axis, and the
        `axis` along which the windows were extracted is 'trimmed' such that its length equals the number of
        extracted windows. More specifically, `windows.shape = x_trimmed_shape + (window_len,)`, where
        `x_trimmed_shape` equals `x.shape`, except that `x_trimmed_shape[axis] = num_windows`.

    Raises
    ------
    ValueError
        If `check_inputs = True` and `window_len` is not positive.
    ValueError
        If `check_inputs = True` and `stride` is not positive.
    ValueError
        If `check_inputs = True` and `axis` is greater than `x.ndim`.
    ValueError
        If `check_inputs = True` and `window_len` is larger than `x.shape[axis]`.

    References
    ----------
    .. [1] https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html
    """
    if check_inputs:
        raise_if(stride < 1, f"`stride` (= {window_len}) must be positive.")
        raise_if(
            window_len < 1,
            f"`window_len` (= {window_len}) must be positive.",
        )
        raise_if_not(
            axis < x.ndim,
            (f"`axis` (= {axis}) must be " "less than `x.ndim` (= {x.ndim})."),
        )
        raise_if(
            window_len > x.shape[axis],
            (
                f"`window_len` (= {window_len}) must "
                "be less than or equal to"
                f"x.shape[axis] (= {x.shape[axis]})."
            ),
        )
    num_windows = (x.shape[axis] - window_len) // stride + 1
    new_shape = list(x.shape)
    new_shape[axis] = num_windows
    new_shape = tuple(new_shape) + (window_len,)
    out_strides = list(x.strides) + [x.strides[axis]]
    out_strides[axis] = stride * out_strides[axis]
    out_strides = tuple(out_strides)
    return as_strided(x, shape=new_shape, strides=out_strides)


#
#   Private Input Checker Functions
#


def _all_equal_freq(*series: Sequence[Union[TimeSeries, None]]) -> bool:
    """
    Returns `True` is all of the specified (i.e. non-`None`) `series` have the same frequency.
    """
    freqs = []
    for ts in series:
        if ts is not None:
            freqs.append(ts.freq)
    return len(set(freqs)) == 1


def _check_lags(
    lags: Sequence[int],
    lags_past_covariates: Sequence[int],
    lags_future_covariates: Sequence[int],
) -> None:
    """
    Throws `ValueError` if any `lag` values aren't negative OR if no lags have been specified.
    """
    all_lags = [lags, lags_past_covariates, lags_future_covariates]
    suffixes = ["", "_past_covariates", "_future_covariates"]
    lags_is_none = []
    for suffix, lags_i in zip(suffixes, all_lags):
        lags_is_none.append(lags_i is None)
        if not lags_is_none[-1]:
            raise_if(
                any(lag >= 0 for lag in lags_i),
                f"`lags{suffix}` must contain only negative values.",
            )
    raise_if(
        all(lags_is_none),
        "Must specify at least one of: `lags`, `lags_past_covariates`, `lags_future_covariates`.",
    )
    return None


def _check_series_length(
    series: TimeSeries,
    lags: Union[None, Sequence[int]],
    output_chunk_length: int,
    is_training: bool,
    name: Literal["target_series", "past_covariates", "future_covariates"],
) -> None:
    """
    Throws `ValueError` if `series` is too short for specified `lags` and, when `is_training`, `output_chunk_length`.
    """
    is_target = name == "target_series"
    is_label_series = is_training and is_target
    lags_specified = lags is not None
    if is_label_series:
        minimum_len_str = (
            "-min(lags) + 1 + output_chunk_length"
            if lags_specified
            else "output_chunk_length"
        )
        minimum_len = (
            -min(lags) + 1 + output_chunk_length
            if lags_specified
            else output_chunk_length
        )
    elif lags_specified:
        lags_name = "lags" if name == "target_series" else f"lags_{name}"
        minimum_len_str = f"-min({lags_name}) + 1"
        minimum_len = -min(lags) + 1
    if lags_specified:
        raise_if(
            series.n_timesteps < minimum_len,
            (
                f"`{name}` must have at least "
                f"`{minimum_len_str}` = {minimum_len} timesteps; "
                f"instead, it only has {series.n_timesteps}."
            ),
        )
    return None
