import warnings
from math import inf
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from darts.logging import raise_if
from darts.timeseries import TimeSeries


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
                        columns=lambda x: f"{x}_horizon_lag{future_target_lag}"
                    )
                )
        else:
            df_y.append(
                df_target.shift(-output_chunk_length + 1).rename(
                    columns=lambda x: f"{x}_horizon_lag{output_chunk_length-1}"
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


def create_lagged_features_and_labels(
    target_series: TimeSeries,
    horizon: int,
    past_series: Optional[TimeSeries] = None,
    future_series: Optional[TimeSeries] = None,
    target_lags: Optional[Sequence[int]] = None,
    past_lags: Optional[Sequence[int]] = None,
    future_lags: Optional[Sequence[int]] = None,
    max_samples: Optional[int] = inf,
    multiple_outputs: Optional[bool] = True,
) -> Tuple[np.ndarray, np.ndarray, Union[pd.RangeIndex, pd.DatetimeIndex]]:

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
        the data in `target_series` and `past_series`.
    See [2]_ for a more detailed discussion about target, past, and future covariates. Conversely, `y` is
    comprised only of the lagged values of `target_series`.

    The shape of `X` is:
        `X.shape = (n_observations, n_lagged_features, n_samples)`,
    where `n_observations` equals either the number of time points shared between all specified series,
    or `max_samples`, whichever is smallest. The shape of `y` is:
        `y.shape = (n_observations, horizon, n_samples)`,
    if `multiple_outputs = True`, otherwise:
        `y.shape = (n_observations, 1, n_samples)`.

    Along the `n_lagged_features` axis, `X` has the following structure (for `*_lags=[-2,-1]` and
    `*_series.n_components = 2`):
        lagged_target | lagged_past_covariates | lagged_future_covariates
    where each `lagged_*` has the following structure:
        lag_-2_comp_1_* | lag_-2_comp_2_* | lag_-1_comp_1_* | lag_-1_comp_2_*

    Along the `n_lagged_labels` axis, `y` has the following structure (for `horizon=4` and
    `target_series.n_components=2`):
        lag_+0_comp_1_target | lag_+0_comp_2_target | ... | lag_+3_comp_1_target | lag_+3_comp_2_target

    Parameters
    ----------
    target_series
        The target series for the regression model to predict.
    horizon
        The number of timesteps ahead into the future the regression model is to predict.
    past_series
        Optionally, the past covariates series that the regression model will use as inputs. Unlike the
        `target_series`, `past_covariates` are *not* to be predicted by the regression model.
    future_series
        Optionally, the future covariates (i.e. exogenous covariates) series that the regression model will
        use as inputs.
    target_lags
        Optionally, the lags of the target series to be used as (auto-regressive) features. If not specified,
        auto-regressive features will *not* be added to `X`.
    lags_past_covariates
        Optionally, the lags of `past_series` to be used as features.
    lags_future_covariates
        Optionally, the lags of `future_series` to be used as features.
    lags_past_covariates
        Optionally, the lags of `past_series` to be used as features.
    lags_future_covariates
        Optionally, the lags of `future_series` to be used as features.
    max_samples
        Optionally, the maximum number of samples to be drawn for training/validation; only the most recent
        samples are kept. In theory, specifying a smaller `max_samples` should reduce computation time,
        especially in cases where many observations could be generated.
    multiple_outputs
        Optionally, specifies whether the regression model predicts multiple timesteps into the future. If `True`,
        then the regression model is assumed to predict all of the timesteps from time `t` to `t+horizon`. If
        `False`, then the regression model is assumed to predict *only* the timestep at `t+horizon`.

    Returns
    -------
    X
        A `(n_observations, n_lagged_features, n_samples)`-shaped array of lagged features for training a
        lagged-variables model.
    y
        If `multiple_outputs = True`, a `(n_observations, horizon, n_samples)`-shaped array of lagged
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
        pd.RangeIndex, but `future_series` uses a `pd.DatetimeIndex`).

    References
    ----------
    .. [1] https://otexts.com/fpp2/AR.html#AR
    .. [2] https://unit8.com/resources/time-series-forecasting-using-past-and-future-external-data-with-darts/
    """

    all_series = [target_series, past_series, future_series]
    all_lags = [target_lags, past_lags, future_lags]
    _check_lags(all_lags)
    times, times_idxs = _find_intersecting_times(
        all_series, all_lags, max_samples, horizon=horizon, is_training=True
    )
    X = []
    for i, time_idx in enumerate(times_idxs):
        if time_idx is not None:
            # Offset shared time indices by requested lags:
            idx_to_get = time_idx.reshape(-1, 1) + all_lags[i]
            X.append(_create_lagged_vals(all_series[i], idx_to_get))
    X = np.concatenate(X, axis=1)
    if multiple_outputs:
        # All points between time `t` and `t + horizon` are labels:
        idx_to_get = times_idxs[0].reshape(-1, 1) + np.arange(horizon)
    else:
        # Only point at time `t + horizon` is a label:
        idx_to_get = times_idxs[0].reshape(-1, 1) + horizon - 1
    y = _create_lagged_vals(target_series, idx_to_get)
    return X, y, times


def create_lagged_features(
    target_series: Optional[TimeSeries] = None,
    past_series: Optional[TimeSeries] = None,
    future_series: Optional[TimeSeries] = None,
    target_lags: Optional[Sequence[int]] = None,
    past_lags: Optional[Sequence[int]] = None,
    future_lags: Optional[Sequence[int]] = None,
    max_samples: Optional[int] = inf,
) -> Tuple[np.ndarray, Union[pd.RangeIndex, pd.DatetimeIndex]]:

    """

    Creates the features array `X` required to generate a prediction from a trained lagged-variables
    regression model (e.g. an `sklearn` model); the time index values of each observation is also returned.
    In order for the lagged features of a series to be added to `X`, *both* that series and the corresponding
    lags must be specified; if a series is specified without the corresponding lags, that series will be
    ignored and not added to `X`. Observations (i.e. rows along the `0`th axis of `X` and `y`) are only created
    for those time points which are common to all of the specified series. `X` and `y` arrays are constructed
    independently over the samples dimension (i.e. the second axis) of each series.

    Notes
    -----
    The precise meaning of the `target_series`, `past_series` and `future_series`, as well as the structure
    of the `X` matrix, are detailed in `create_training_X_and_y`.

    The same `*_series` and `*_lags` provided to `create_lagged_features_and_labels` should also
    be provided to `create_lagged_features`, with the exception of `target_series`, which doesn't need
    to be provided if the trained model does *not* take auto-regressive features as inputs.

    To generate only the feature corresponding to the latest possible (predictable) time point, specify
    `max_samples=1`.

    Parameters
    ----------
    target_series
        Optionally, the target series for the regression model to predict. This series only needs to be specified
        if the trained lagged-feature model uses auto-regressive features.
    past_series
        Optionally, the past covariates series that the regression model will use as inputs.
    future_series
        Optionally, the future covariates (i.e. exogenous covariates) series that the regression model will use
        as inputs.
    target_lags
        Optionally, the lags of the target series to be used as (auto-regressive) features.
    lags_past_covariates
        Optionally, the lags of `past_series` to be used as features.
    lags_future_covariates
        Optionally, the lags of `future_series` to be used as features.
    lags_past_covariates
        Optionally, the lags of `past_series` to be used as features.
    lags_future_covariates
        Optionally, the lags of `future_series` to be used as features.
    max_samples
        Optionally, the maximum number of samples to be drawn for training/validation; only the most recent samples
        are kept. In theory, specifying a smaller `max_samples` should reduce computation time, especially in
        cases where many observations could be generated.

    Returns
    -------
    X
        A `(n_observations, n_lagged_features, n_samples)`-shaped array of lagged features for training a
        lagged-variables model.
    times
        The `time_index` of each observation in `X`.

    Raises
    ------
    ValueError
        If there are no times shared by all of the specified series, or if no lags have been specified.
    TypeError
        If the provided series do not share the same type of `time_index` (e.g. `target_series` uses a
        `pandas.RangeIndex`, but `future_series` uses a `pandas.DatetimeIndex`).

    """

    all_series = [target_series, past_series, future_series]
    all_lags = [target_lags, past_lags, future_lags]
    _check_lags(all_lags)
    times, times_idxs = _find_intersecting_times(
        all_series, all_lags, max_samples, horizon=0, is_training=False
    )
    X = []
    for i, time_idx in enumerate(times_idxs):
        if time_idx is not None:
            # Offset shared time indices by requested lags w/ + 1
            # to account for fact that value at time `t` can also
            # be used as a feature when predicting:
            idx_to_get = time_idx.reshape(-1, 1) + all_lags[i] + 1
            X.append(_create_lagged_vals(all_series[i], idx_to_get))
    X = np.concatenate(X, axis=1)
    return X, times


def _check_lags(lags):
    """
    Raises `ValueError` if not at least one `*_lags` argument is specified.
    """
    raise_if(
        all(lag is None for lag in lags),
        ("Must specify at least one of: " "`target_lags`, `past_lags`, `future_lags`."),
    )
    return None


def _find_intersecting_times(
    all_series: Tuple[
        Union[TimeSeries, None], Union[TimeSeries, None], Union[TimeSeries, None]
    ],
    all_lags: Tuple[
        Union[Sequence[int], None],
        Union[Sequence[int], None],
        Union[Sequence[int], None],
    ],
    max_samples: int,
    horizon: int,
    is_training: bool,
):
    """
    Returns `pd.RangeIndex`/`pd.DatetimeIndex` shared by all three series (i.e. `target_series`,
    `past_series`, and `future_series`); the index of these shared times in each series (i.e. the
    indexes that need to be accessed in `*_series.time_index` to retrieve the shared times) is also
    computed.

    Notes
    -----
    Series which haven't been specified OR which are specified without corresponding `lags` are
    ignored, except `target_series`, whose times are still considered if `is_training = True`.

    Parameters
    ----------
    all_series
        Tuple that stores `(target_series, past_series, future_series)`; some of these series
        values may be `None`.
    all_lags
        Tuple that stores `(target_lags, past_lags, future_lags)`; some of these lag values
        may be `None`.
    max_samples
        The maximum number of samples to be drawn for training/validation; only the most recent
        samples are kept.
    horizon
        The number of timesteps ahead into the future the regression model is to predict; is
        ignored if `is_training = False`.
    is_training
        Specifies whether intersecting times are to be used for creating training data.

    Returns
    -------
    shared_times
        The `time_index` values shared by all specified time series.
    shared_times_idx
        Tuple of array indices to access the `shared_times` values in `target_series`, `past_series`,
        and `future_series`, in that order. If a series was not specified, `None` appears in its
        place in lieu of an array index.

    Raises
    ------
    ValueError
        If there are no times shared by all of the specified series, or if no lags have been specified.
    TypeError
        If the provided series do not share the same type of `time_index` (e.g. `target_series` uses
        a pd.RangeIndex, but `future_series` uses a `pd.DatetimeIndex`).
    """
    names = ["target", "past", "future"]
    start_time_idxs = []
    shared_times = None
    for i, (ts, lags) in enumerate(zip(all_series, all_lags)):
        if ts is None:
            start_time_idxs.append(None)
            continue
        # *Don't* skip `target_series` if `is_training` even when `target_lags` is `None`,
        # since `target_series` is required to construct `y`:
        if is_training and (shared_times is None):
            start_time_idx = -np.min(lags) if lags is not None else 0
            end_time_idx = -horizon + 1 if horizon > 1 else None
            shared_times = ts.time_index[start_time_idx:end_time_idx]
        else:
            # Skip series without corresponding lag specified:
            if lags is None:
                not_specified = names[i]
                warn_msg = f"`{not_specified}_series` specified without specifying `{not_specified}_lags`; "
                warn_msg += f"`{not_specified}_series` will not be added to `X`."
                warnings.warn(warn_msg)
                start_time_idxs.append(None)
                continue
            start_time_idx = -np.min(lags)
            # For prediction, we can use series at time `t` as a feature rather than a label:
            if not is_training:
                start_time_idx -= 1
            if shared_times is None:
                shared_times = ts.time_index[start_time_idx:]
            else:
                # `sort=None` tells `pd` to sort merged times; sorting is required for
                # `np.searchsorted` later on:
                shared_times = shared_times.intersection(
                    ts.time_index[start_time_idx:], sort=None
                )
        start_time_idxs.append(start_time_idx)
    # Throw errors if no shared times:
    if shared_times.empty:
        is_empty = True
        index_types = [type(ts.time_index) for ts in all_series]
        same_index_type = len(set(index_types)) == 1
    else:
        is_empty = False
        same_index_type = True
    raise_if(
        is_empty and same_index_type,
        "No common times found for specified `lag` values.",
    )
    raise_if(
        is_empty and not same_index_type,
        (
            "All series must have the same `time_index` type"
            "(i.e. all `pandas.DateTimeIndex)` or all `pandas.RangeIndex`)."
        ),
    )
    if len(shared_times) > max_samples:
        shared_times = shared_times[-max_samples:]
    shared_times_idx = []
    for ts, start_time_idx in zip(all_series, start_time_idxs):
        if start_time_idx is None:
            shared_times_idx.append(None)
        else:
            # Specify `ts.time_index[start_time_idx:]` as first arg since this array
            # will contain more values, so binary search will be faster:
            time_idx = (
                np.searchsorted(ts.time_index[start_time_idx:], shared_times)
                + start_time_idx
            )
            shared_times_idx.append(time_idx)
    return shared_times, shared_times_idx


def _create_lagged_vals(series, idx_to_get):
    """
    Extracts lagged features specified by `idx_to_get` and reshapes indexed values
    into correct shape.
    """
    # Before reshaping: lagged_vals.shape = (n_observations, num_lags, n_components, n_samples)
    lagged_vals = series.all_values()[idx_to_get, :, :]
    # After reshaping: lagged_vals.shape = (n_observations, num_lags*n_components, n_samples)
    lagged_vals = lagged_vals.reshape(lagged_vals.shape[0], -1, lagged_vals.shape[-1])
    return lagged_vals
