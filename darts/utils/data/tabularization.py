import warnings
from functools import reduce
from math import inf
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from itertools import chain

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided

from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts.utils.utils import get_single_series, series2seq

logger = get_logger(__name__)

ArrayOrArraySequence = Union[np.ndarray, Sequence[np.ndarray]]


def create_lagged_data(
    target_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    lags: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    lags_past_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    lags_future_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    output_chunk_length: int = 1,
    uses_static_covariates: bool = True,
    last_static_covariates_shape: Optional[Tuple[int, int]] = None,
    max_samples_per_ts: Optional[int] = None,
    multi_models: bool = True,
    check_inputs: bool = True,
    use_moving_windows: bool = True,
    is_training: bool = True,
    concatenate: bool = True,
) -> Tuple[
    ArrayOrArraySequence,
    Union[None, ArrayOrArraySequence],
    Sequence[pd.Index],
    Optional[Tuple[int, int]],
]:
    """
    Creates the features array `X` and labels array `y` to train a lagged-variables regression model (e.g. an
    `sklearn` model) when `is_training = True`; alternatively, creates the features array `X` to produce a series
    of prediction from an already-trained regression model when `is_training = False`. In both cases, a list of time
    indices corresponding to each generated observation is also returned.

    Notes
    -----
    Instead of calling `create_lagged_data` directly, it is instead recommended that:
        - `create_lagged_training_data` be called if one wishes to create the `X` and `y` arrays
        to train a regression model.
        - `create_lagged_prediction_data` be called if one wishes to create the `X` array required
        to generate a prediction from an already-trained regression model.
    This is because even though both of these functions are merely wrappers around `create_lagged_data`, their
    call signatures are more easily interpreted than `create_lagged_data`. For example,
    `create_lagged_prediction_data` does not accept `output_chunk_length` nor `multi_models` as inputs, since
    these inputs are not used when constructing prediction data. Similarly, `create_lagged_prediction_data`
    returns only `X` and `times` as outputs, as opposed to returning `y` as `None` along with `X` and `times`.

    The `X` array is constructed from the lagged values of up to three separate timeseries:
        1. The `target_series`, which contains the values we're trying to predict. A regression model that
        uses previous values of the target its predicting is referred to as *auto-regressive*; please refer to
        [1]_ for further details about auto-regressive timeseries models.
        2. The past covariates series, which contains values that are *not* known into the future. Unlike
        the target series, however, past covariates are *not* to be predicted by the regression model.
        3. The future covariates (AKA 'exogenous' covariates) series, which contains values that are known
        into the future, even beyond the data in `target_series` and `past_covariates`.
    See [2]_ for a more detailed discussion about target, past, and future covariates. Conversely, `y` is
    comprised only of the lagged values of `target_series`.

    The shape of `X` is:
        `X.shape = (n_observations, n_lagged_features, n_samples)`,
    where `n_observations` equals either the number of time points shared between all specified series,
    or `max_samples_per_ts`, whichever is smallest.
    The shape of `y` is:
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

    The `lags` and `lags_past_covariates` must contain only values less than or equal to -1. In other words, one
    cannot use the value of either of these series at time `t` to predict the value of the target series at the
    same time `t`; this is because the values of `target_series` and `past_covariates` at time `t` aren't available
    at prediction time, by definition. Conversely, since the values of `future_covariates` are known into the future,
    `lags_future_covariates` can contain negative, positive, and/or zero lag values (i.e. we *can* use the values of
    `future_covariates` at time `t` or beyond to predict the value of `target_series` at time `t`).

    The exact method used to construct `X` and `y` depends on whether all of the specified timeseries are
    of the same frequency or not:
        - If all specified timeseries are of the same frequency, `strided_moving_window` is used to extract
        contiguous time blocks from each timeseries; the lagged variables are then extracted from each window.
        - If all specified timeseries are *not* of the same frequency, then `find_shared_times` is first used
        to find those times common to all three timeseries, after which the lagged features are extracted by
        offsetting the time indices of these common times by the requested lags.
    In cases where it can be validly applied, the 'moving window' method is expected to be faster than the
    'intersecting time' method. However, in exceptional cases where only a small number of lags are being
    extracted, but the difference between the lag values is large (e.g. `lags = [-1, -1000]`), the 'moving
    window' method is expected to consume significantly more memory, since it extracts all of the series values
    between the maximum and minimum lags as 'windows', before actually extracting the specific requested lag values.

    In order for the lagged features of a series to be added to `X`, *both* that series and the corresponding lags
    must be specified; if a series is specified without the corresponding lags, that series will be ignored and not
    added to `X`. `X` and `y` arrays are constructed independently over the samples dimension (i.e. the second axis)
    of each series.

    If the provided series are stochastic (i.e. `series.n_components > 1`), then an `X` and `y` array will be
    constructed for each sample; the arrays corresponding to each sample are concatenated togather along the `2`nd
    axis of `X` and `y`. In other words, `create_lagged_data` is vectorised over the sample axis of the `target_series`,
    `past_covariates`, and `future_covariates` inputs. Importantly, if stochastic series are provided, each series must
    have the same number of samples, otherwise an error will be thrown.

    Each series input (i.e. `target_series`, `past_covariates`, and `future_covariates`) can be specified either as
    a single `TimeSeries`, or as a `Sequence` of `TimeSeries`; the specified series must all be of the same type,
    however (i.e. either all `TimeSeries` or all `Sequence[TimeSeries]`). If `Sequence[TimeSeries]` are specified,
    then a feature matrix `X` and labels array `y` will be constructed using the corresponding `TimeSeries` in
    each `Sequence` (i.e. the first `TimeSeries` in each `Sequence` are used to create an `X` and `y`, then
    the second `TimeSeries` in each `Sequence` are used to create an `X` and `y`, etc.). If `concatenate = True`,
    these `X`'s and `y`'s will be concatenated along the `0`th axis; otherwise, a list of `X` and `y` array will
    be returned. Note that `times` is always returned as a `Sequence[pd.Index]`, however, even when
    `concatenate = True`.

    Parameters
    ----------
    target_series
        Optionally, the series for the regression model to predict. Must be specified if `is_training = True`.
        Can be specified as either a `TimeSeries` or as a `Sequence[TimeSeries]`.
    output_chunk_length
        Optionally, the number of timesteps ahead into the future the regression model is to predict. Must
        best specified if `is_training = True`.
    past_covariates
        Optionally, the past covariates series that the regression model will use as inputs. Unlike the
        `target_series`, `past_covariates` are *not* to be predicted by the regression model. Can be
        specified as either a `TimeSeries` or as a `Sequence[TimeSeries]`.
    future_covariates
        Optionally, the future covariates (i.e. exogenous covariates) series that the regression model will
        use as inputs. Can be specified as either a `TimeSeries` or as a `Sequence[TimeSeries]`.
    lags
        Optionally, the lags of the target series to be used as (auto-regressive) features. If not specified,
        auto-regressive features will *not* be added to `X`. Each lag value is assumed to be negative (e.g.
        `lags = [-3, -1]` will extract `target_series` values which are 3 timesteps and 1 timestep away from
        the current value). If the lags are provided as a dictionary, the lags values are specific to each
        component in the target series.
    lags_past_covariates
        Optionally, the lags of `past_covariates` to be used as features. Like `lags`, each lag value is assumed to
        be less than or equal to -1. If the lags are provided as a dictionary, the lags values are specific to each
        component in the past covariates series.
    lags_future_covariates
        Optionally, the lags of `future_covariates` to be used as features. Unlike `lags` and
        `lags_past_covariates`, `lags_future_covariates` values can be positive (i.e. use values *after* time `t`
        to predict target at time `t`), zero (i.e. use values *at* time `t` to predict target at time `t`), and/or
        negative (i.e. use values *before* time `t` to predict target at time `t`). If the lags are provided as
        a dictionary, the lags values are specific to each component in the future covariates series.
    uses_static_covariates
        Whether the model uses/expects static covariates. If `True`, it enforces that static covariates must
        have identical shapes across all target series.
    last_static_covariates_shape
        Optionally, the last observed shape of the static covariates. This is ``None`` before fitting, or when
        `uses_static_covariates` is ``False``.
    max_samples_per_ts
        Optionally, the maximum number of samples to be drawn for training/validation; only the most recent
        samples are kept. In theory, specifying a smaller `max_samples_per_ts` should reduce computation time,
        especially in cases where many observations could be generated.
    multi_models
        Optionally, specifies whether the regression model predicts multiple timesteps into the future. If `True`,
        then the regression model is assumed to predict all of the timesteps from time `t` to `t+output_chunk_length`.
        If `False`, then the regression model is assumed to predict *only* the timestep at `t+output_chunk_length`.
        This input is ignored if `is_training = False`.
    check_inputs
        Optionally, specifies that the `lags_*` and `series_*` inputs should be checked for validity. Should be set
        to `False` if inputs have already been checked for validity (e.g. inside the `__init__` of a class), otherwise
        should be set to `True`.
    use_moving_windows
        Optionally, specifies that the 'moving window' method should be used to construct `X` and `y` if all of the
        provided series are of the same frequency. If `use_moving_windows = False`, the 'time intersection' method
        will always be used, even when all of the provided series are of the same frequency. In general, setting
        to `True` results in faster tabularization at the potential cost of higher memory usage. See Notes for further
        details.
    is_training
        Optionally, specifies whether the constructed lagged data are to be used for training a regression model
        (i.e. `is_training = True`), or for generating predictions from an already-trained regression model (i.e.
        `is_training = False`). If `is_training = True`, `target_series` and `output_chunk_length` must be specified,
        the `multi_models` input is utilised, and a label array `y` is returned. Conversely, if `is_training = False`,
        then `target_series` and `output_chunk_length` do not need to be specified, the `multi_models` input is ignored,
        and the returned `y` value is `None`.
    concatenate
        Optionally, specifies that `X` and `y` should both be returned as single `np.ndarray`s, instead of as
        a `Sequence[np.ndarray]`. If each series input is specified as a `Sequence[TimeSeries]` and
        `concatenate = False`, `X` and `y` will be lists whose `i`th element corresponds to the feature matrix or label
        array formed by the `i`th `TimeSeries` in each `Sequence[TimeSeries]` input. Conversely, if `concatenate = True`
        when `Sequence[TimeSeries]` are provided, then `X` and `y` will be arrays created by concatenating all of the
        feature/label arrays formed by each `TimeSeries` along the `0`th axis. Note that `times` is still returned as
        `Sequence[pd.Index]`, even when `concatenate = True`.

    Returns
    -------
    X
        The constructed features array(s), with shape `(n_observations, n_lagged_features, n_samples)`.
        If the series inputs were specified as `Sequence[TimeSeries]` and `concatenate = False`, then `X`
        is returned as a `Sequence[np.array]`; otherwise, `X` is returned as a single `np.array`.
    y
        The constructed labels array. If `multi_models = True`, then `y` is a
        `(n_observations, output_chunk_length, n_samples)`-shaped array; conversely, if
        `multi_models =  False`, then `y` is a `(n_observations, 1, n_samples)`-shaped array.
        If the series inputs were specified as `Sequence[TimeSeries]` and `concatenate = False`, then `y`
        is returned as a `Sequence[np.array]`; otherwise, `y` is returned as a single `np.array`.
    times
        The `time_index` of each observation in `X` and `y`, returned as a `Sequence` of `pd.Index`es.
        If the series inputs were specified as `Sequence[TimeSeries]`, then the `i`th list element
        gives the times of those observations formed using the `i`th `TimeSeries` object in each
        `Sequence`. Otherwise, if the series inputs were specified as `TimeSeries`, the only
        element is the times of those observations formed from the lone `TimeSeries` inputs.
    last_static_covariates_shape
        The last observed shape of the static covariates. This is ``None`` when `uses_static_covariates`
        is ``False``.


    Raises
    ------
    ValueError
        If the specified time series do not share any times for which features (and labels if `is_training = True`) can
        be constructed.
    ValueError
        If no lags are specified, or if any of the specified lag values are non-negative.
    ValueError
        If any of the series are too short to create features and/or labels for the requested lags and
        `output_chunk_length` values.
    ValueError
        If `target_series` and/or `output_chunk_length` are *not* specified when `is_training = True`.
    ValueError
        If the provided series do not share the same type of `time_index` (e.g. `target_series` uses a
        pd.RangeIndex, but `future_covariates` uses a `pd.DatetimeIndex`).

    References
    ----------
    .. [1] https://otexts.com/fpp2/AR.html#AR
    .. [2] https://unit8.com/resources/time-series-forecasting-using-past-and-future-external-data-with-darts/

    See Also
    --------
        tabularization.create_lagged_component_names : return the lagged features names as a list of strings.

    """
    raise_if(
        is_training and (target_series is None),
        "Must specify `target_series` if `is_training = True`.",
    )

    # ensure list of TimeSeries format
    target_series = series2seq(target_series)
    past_covariates = series2seq(past_covariates)
    future_covariates = series2seq(future_covariates)
    seq_ts_lens = [
        len(seq_ts)
        for seq_ts in (target_series, past_covariates, future_covariates)
        if seq_ts is not None
    ]
    seq_ts_lens = set(seq_ts_lens)
    raise_if(
        len(seq_ts_lens) > 1,
        "Must specify the same number of `TimeSeries` for each series input.",
    )
    if max_samples_per_ts is None:
        max_samples_per_ts = inf
    X, y, times = [], [], []
    for i in range(max(seq_ts_lens)):
        target_i = target_series[i] if target_series else None
        past_i = past_covariates[i] if past_covariates else None
        future_i = future_covariates[i] if future_covariates else None
        if use_moving_windows and _all_equal_freq(target_i, past_i, future_i):
            X_i, y_i, times_i = _create_lagged_data_by_moving_window(
                target_i,
                output_chunk_length,
                past_i,
                future_i,
                lags,
                lags_past_covariates,
                lags_future_covariates,
                max_samples_per_ts,
                multi_models,
                check_inputs,
                is_training,
            )
        else:
            X_i, y_i, times_i = _create_lagged_data_by_intersecting_times(
                target_i,
                output_chunk_length,
                past_i,
                future_i,
                lags,
                lags_past_covariates,
                lags_future_covariates,
                max_samples_per_ts,
                multi_models,
                check_inputs,
                is_training,
            )
        X_i, last_static_covariates_shape = add_static_covariates_to_lagged_data(
            features=X_i,
            target_series=target_i,
            uses_static_covariates=uses_static_covariates,
            last_shape=last_static_covariates_shape,
        )
        X.append(X_i)
        y.append(y_i)
        times.append(times_i)

    if concatenate:
        X = np.concatenate(X, axis=0)
    if not is_training:
        y = None
    elif concatenate:
        y = np.concatenate(y, axis=0)
    return X, y, times, last_static_covariates_shape


def create_lagged_training_data(
    target_series: Union[TimeSeries, Sequence[TimeSeries]],
    output_chunk_length: int,
    past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    lags: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    lags_past_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    lags_future_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    uses_static_covariates: bool = True,
    last_static_covariates_shape: Optional[Tuple[int, int]] = None,
    max_samples_per_ts: Optional[int] = None,
    multi_models: bool = True,
    check_inputs: bool = True,
    use_moving_windows: bool = True,
    concatenate: bool = True,
) -> Tuple[
    ArrayOrArraySequence,
    Union[None, ArrayOrArraySequence],
    Sequence[pd.Index],
    Optional[Tuple[int, int]],
]:
    """
    Creates the features array `X` and labels array `y` to train a lagged-variables regression model (e.g. an
    `sklearn` model); the time index values of each observation is also returned.

    Notes
    -----
    This function is simply a wrapper around `create_lagged_data`; for further details on the structure of `X`, please
    refer to `help(create_lagged_data)`.

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
        auto-regressive features will *not* be added to `X`. Each lag value is assumed to be negative (e.g.
        `lags = [-3, -1]` will extract `target_series` values which are 3 timesteps and 1 timestep away from
        the current value). If the lags are provided as a dictionary, the lags values are specific to each
        component in the target series.
    lags_past_covariates
        Optionally, the lags of `past_covariates` to be used as features. Like `lags`, each lag value is assumed to
        be less than or equal to -1. If the lags are provided as a dictionary, the lags values are specific to each
        component in the past covariates series.
    lags_future_covariates
        Optionally, the lags of `future_covariates` to be used as features. Unlike `lags` and `lags_past_covariates`,
        `lags_future_covariates` values can be positive (i.e. use values *after* time `t` to predict target at
        time `t`), zero (i.e. use values *at* time `t` to predict target at time `t`), and/or negative (i.e. use values
        *before* time `t` to predict target at time `t`). If the lags are provided as a dictionary, the lags values
        are specific to each component in the future covariates series.
    uses_static_covariates
        Whether the model uses/expects static covariates. If `True`, it enforces that static covariates must
        have identical shapes across all target series.
    last_static_covariates_shape
        Optionally, the last observed shape of the static covariates. This is ``None`` before fitting, or when
        `uses_static_covariates` is ``False``.
    max_samples_per_ts
        Optionally, the maximum number of samples to be drawn for training/validation; only the most recent
        samples are kept. In theory, specifying a smaller `max_samples_per_ts` should reduce computation time,
        especially in cases where many observations could be generated.
    multi_models
        Optionally, specifies whether the regression model predicts multiple timesteps into the future. If `True`,
        then the regression model is assumed to predict all of the timesteps from time `t` to `t+output_chunk_length`.
        If `False`, then the regression model is assumed to predict *only* the timestep at `t+output_chunk_length`.
    check_inputs
        Optionally, specifies that the `lags_*` and `series_*` inputs should be checked for validity. Should be set
        to `False` if inputs have already been checked for validity (e.g. inside the `__init__` of a class), otherwise
        should be set to `True`.
    use_moving_windows
        Optionally, specifies that the 'moving window' method should be used to construct `X` and `y` if all of the
        provided series are of the same frequency. If `use_moving_windows = False`, the 'time intersection' method
        will always be used, even when all of the provided series are of the same frequency. In general, setting
        to `True` results in faster tabularization at the potential cost of higher memory usage. See Notes for further
        details.
    concatenate
        Optionally, specifies that `X` and `y` should both be returned as single `np.ndarray`s, instead of as
        a `Sequence[np.ndarray]`. If each series input is specified as a `Sequence[TimeSeries]` and
        `concatenate = False`, `X` and `y` will be lists whose `i`th element corresponds to the feature matrix or label
        array formed by the `i`th `TimeSeries` in each `Sequence[TimeSeries]` input. Conversely, if `concatenate = True`
        when `Sequence[TimeSeries]` are provided, then `X` and `y` will be arrays created by concatenating all of the
        feature/label arrays formed by each `TimeSeries` along the `0`th axis. Note that `times` is still returned as
        `Sequence[pd.Index]`, even when `concatenate = True`.

    Returns
    -------
    X
        The constructed features array(s), with shape `(n_observations, n_lagged_features, n_samples)`.
        If the series inputs were specified as `Sequence[TimeSeries]` and `concatenate = False`, then `X`
        is returned as a `Sequence[np.array]`; otherwise, `X` is returned as a single `np.array`.
    y
        The constructed labels array. If `multi_models = True`, then `y` is a
        `(n_observations, output_chunk_length, n_samples)`-shaped array; conversely, if
        `multi_models =  False`, then `y` is a `(n_observations, 1, n_samples)`-shaped array.
        If the series inputs were specified as `Sequence[TimeSeries]` and `concatenate = False`, then `y`
        is returned as a `Sequence[np.array]`; otherwise, `y` is returned as a single `np.array`.
    times
        The `time_index` of each observation in `X` and `y`, returned as a `Sequence` of `pd.Index`es.
        If the series inputs were specified as `Sequence[TimeSeries]`, then the `i`th list element
        gives the times of those observations formed using the `i`th `TimeSeries` object in each
        `Sequence`. Otherwise, if the series inputs were specified as `TimeSeries`, the only
        element is the times of those observations formed from the lone `TimeSeries` inputs.

    Raises
    ------
    ValueError
        If the specified time series do not share any times for which features and labels can be constructed.
    ValueError
        If no lags are specified, or if any of the specified lag values are non-negative.
    ValueError
        If any of the series are too short to create features and labels for the requested lags and
        `output_chunk_length` values.
    ValueError
        If the provided series do not share the same type of `time_index` (e.g. `target_series` uses a
        pd.RangeIndex, but `future_covariates` uses a `pd.DatetimeIndex`).
    """
    return create_lagged_data(
        target_series=target_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        lags_future_covariates=lags_future_covariates,
        output_chunk_length=output_chunk_length,
        uses_static_covariates=uses_static_covariates,
        last_static_covariates_shape=last_static_covariates_shape,
        max_samples_per_ts=max_samples_per_ts,
        multi_models=multi_models,
        check_inputs=check_inputs,
        use_moving_windows=use_moving_windows,
        is_training=True,
        concatenate=concatenate,
    )


def create_lagged_prediction_data(
    target_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    lags: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    lags_past_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    lags_future_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    uses_static_covariates: bool = True,
    last_static_covariates_shape: Optional[Tuple[int, int]] = None,
    max_samples_per_ts: Optional[int] = None,
    check_inputs: bool = True,
    use_moving_windows: bool = True,
    concatenate: bool = True,
) -> Tuple[ArrayOrArraySequence, Sequence[pd.Index]]:
    """
    Creates the features array `X` to produce a series of prediction from an already-trained regression model; the
    time index values of each observation is also returned.

    Notes
    -----
    This function is simply a wrapper around `create_lagged_data`; for further details on the structure of `X`, please
    refer to `help(create_lagged_data)`.

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
        auto-regressive features will *not* be added to `X`. Each lag value is assumed to be negative (e.g.
        `lags = [-3, -1]` will extract `target_series` values which are 3 timesteps and 1 timestep away from
        the current value). If the lags are provided as a dictionary, the lags values are specific to each
        component in the target series.
    lags_past_covariates
        Optionally, the lags of `past_covariates` to be used as features. Like `lags`, each lag value is assumed to
        be less than or equal to -1. If the lags are provided as a dictionary, the lags values are specific to each
        component in the past covariates series.
    lags_future_covariates
        Optionally, the lags of `future_covariates` to be used as features. Unlike `lags` and `lags_past_covariates`,
        `lags_future_covariates` values can be positive (i.e. use values *after* time `t` to predict target at
        time `t`), zero (i.e. use values *at* time `t` to predict target at time `t`), and/or negative (i.e. use
        values *before* time `t` to predict target at time `t`). If the lags are provided as a dictionary, the lags
        values are specific to each component in the future covariates series.
    uses_static_covariates
        Whether the model uses/expects static covariates. If `True`, it enforces that static covariates must
        have identical shapes across all target series.
    last_static_covariates_shape
        Optionally, the last observed shape of the static covariates. This is ``None`` before fitting, or when
        `uses_static_covariates` is ``False``.
    max_samples_per_ts
        Optionally, the maximum number of samples to be drawn for training/validation; only the most recent
        samples are kept. In theory, specifying a smaller `max_samples_per_ts` should reduce computation time,
        especially in cases where many observations could be generated.
    check_inputs
        Optionally, specifies that the `lags_*` and `series_*` inputs should be checked for validity. Should be set
        to `False` if inputs have already been checked for validity (e.g. inside the `__init__` of a class), otherwise
        should be set to `True`.
    use_moving_windows
        Optionally, specifies that the 'moving window' method should be used to construct `X` and `y` if all of the
        provided series are of the same frequency. If `use_moving_windows = False`, the 'time intersection' method
        will always be used, even when all of the provided series are of the same frequency. In general, setting
        to `True` results in faster tabularization at the potential cost of higher memory usage. See Notes for further
        details.
    concatenate
        Optionally, specifies that `X` should be returned as a single `np.ndarray`, instead of as a
        `Sequence[np.ndarray]`. If each series input is specified as a `Sequence[TimeSeries]` and `concatenate = False`,
        `X` will be a list whose `i`th element corresponds to the feature matrix or label array formed by the `i`th
        `TimeSeries` in each `Sequence[TimeSeries]` input. Conversely, if `concatenate = True` when
        `Sequence[TimeSeries]` are provided, then `X` will be an array created by concatenating all of the feature
        arrays formed by each `TimeSeries` along the `0`th axis. Note that `times` is still returned as
        `Sequence[pd.Index]`, even when `concatenate = True`.

    Returns
    -------
    X
        The constructed features array(s), with shape `(n_observations, n_lagged_features, n_samples)`.
        If the series inputs were specified as `Sequence[TimeSeries]` and `concatenate = False`, then `X`
        is returned as a `Sequence[np.array]`; otherwise, `X` is returned as a single `np.array`.
    times
        The `time_index` of each observation in `X` and `y`, returned as a `Sequence` of `pd.Index`es.
        If the series inputs were specified as `Sequence[TimeSeries]`, then the `i`th list element
        gives the times of those observations formed using the `i`th `TimeSeries` object in each
        `Sequence`. Otherwise, if the series inputs were specified as `TimeSeries`, the only
        element is the times of those observations formed from the lone `TimeSeries` inputs.

    Raises
    ------
    ValueError
        If the specified time series do not share any times for which features can be constructed.
    ValueError
        If no lags are specified, or if any of the specified lag values are non-negative.
    ValueError
        If any of the series are too short to create features for the requested lag values.
    ValueError
        If the provided series do not share the same type of `time_index` (e.g. `target_series` uses a
        pd.RangeIndex, but `future_covariates` uses a `pd.DatetimeIndex`).
    """
    X, _, times, _ = create_lagged_data(
        target_series=target_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        lags_future_covariates=lags_future_covariates,
        uses_static_covariates=uses_static_covariates,
        last_static_covariates_shape=last_static_covariates_shape,
        max_samples_per_ts=max_samples_per_ts,
        check_inputs=check_inputs,
        use_moving_windows=use_moving_windows,
        is_training=False,
        concatenate=concatenate,
    )
    return X, times


def add_static_covariates_to_lagged_data(
    features: Union[np.ndarray, Sequence[np.ndarray]],
    target_series: Union[TimeSeries, Sequence[TimeSeries]],
    uses_static_covariates: bool = True,
    last_shape: Optional[Tuple[int, int]] = None,
) -> Union[np.ndarray, Sequence[np.ndarray]]:
    """
    Add static covariates to the features' table for RegressionModels.
    If `uses_static_covariates=True`, all target series used in `fit()` and `predict()` must have static
    covariates with identical dimensionality. Otherwise, will not consider static covariates.

    The static covariates are added to the right of the lagged features following the convention:
    with a 2 component series, and 2 static covariates per component ->
    scov_1_comp_1 | scov_1_comp_2 | scov_2_comp_1 | scov_2_comp_2

    Parameters
    ----------
    features
        The features' numpy array(s) to which the static covariates will be added. Can either be a lone feature
        matrix or a `Sequence` of feature matrices; in the latter case, static covariates will be appended to
        each feature matrix in this `Sequence`.
    target_series
        The target series from which to read the static covariates.
    uses_static_covariates
        Whether the model uses/expects static covariates. If `True`, it enforces that static covariates must
        have identical shapes across all of target series.
    last_shape
        Optionally, the last observed shape of the static covariates. This is ``None`` before fitting, or when
        `uses_static_covariates` is ``False``.

    Returns
    -------
    (features, last_shape)
        The features' array(s) with appended static covariates columns. If the `features` input was passed as a
        `Sequence` of `np.array`s, then a `Sequence` is also returned; if `features` was passed as an `np.array`,
        a `np.array` is returned.
        `last_shape` is the shape of the static covariates.

    """
    # uses_static_covariates=True enforces that all series must have static covs of same dimensionality
    if not uses_static_covariates:
        return features, last_shape

    input_not_list = not isinstance(features, Sequence)
    if input_not_list:
        features = [features]
    target_series = series2seq(target_series)

    # go through series, check static covariates, and stack them to the right of the lagged features
    # try to abort early in case there is a mismatch in static covariates
    for idx, ts in enumerate(target_series):
        if not ts.has_static_covariates:
            raise_log(
                ValueError(
                    "Static covariates mismatch across the sequence of target series. Some of the series "
                    "contain static covariates and others do not."
                ),
                logger,
            )
        else:
            if last_shape is None:
                last_shape = ts.static_covariates.shape
            if ts.static_covariates.shape != last_shape:
                raise_log(
                    ValueError(
                        "Static covariates dimension mismatch across the sequence of target series. The static "
                        "covariates must have the same number of columns and rows across all target series."
                    ),
                    logger,
                )
            # flatten static covariates along columns -> results in [scov0_comp0, scov0_comp1, scov1_comp0, ...]
            static_covs = ts.static_covariates.values.flatten(order="F")
            # we stack the static covariates to the right of lagged features
            # the broadcasting repeats the static covariates along axis=0 to match the number of feature rows
            shape_out = (
                (len(features[idx]), len(static_covs))
                if len(features[idx].shape) == 2
                else (len(features[idx]), len(static_covs), 1)
            )
            features[idx] = np.hstack(
                [
                    features[idx],
                    np.broadcast_to(static_covs, shape_out[:2]).reshape(shape_out),
                ]
            )

    if input_not_list:
        features = features[0]
    return features, last_shape


def create_lagged_component_names(
    target_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    lags: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    lags_past_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    lags_future_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    output_chunk_length: int = 1,
    concatenate: bool = True,
    use_static_covariates: bool = False,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Helper function called to retrieve the name of the features and labels arrays created with
    `create_lagged_data()`. The order of the features is the following:

    Along the `n_lagged_features` axis, `X` has the following structure:
        lagged_target | lagged_past_covariates | lagged_future_covariates | static covariates

    For `*_lags=[-2,-1]` and `*_series.n_components = 2` (lags shared across all the components),
    each `lagged_*` has the following structure (grouped by lags):
        comp0_*_lag-2 | comp1_*_lag-2 | comp0_*_lag_-1 | comp1_*_lag-1
    For `*_lags={'comp0':[-2, -1], 'comp1':[-5, -3]}` and `*_series.n_components = 2` (component-
    specific lags), each `lagged_*` has the following structure (grouped by components):
        comp0_*_lag-2 | comp0_*_lag-1 | comp1_*_lag_-5 | comp1_*_lag-3

    and for static covariates (2 static covariates acting on 2 target components):
        cov0_*_target_comp0 | cov0_*_target_comp1 | cov1_*_target_comp0 | cov1_*_target_comp1

    Along the `n_lagged_labels` axis, `y` has the following structure (for `output_chunk_length=4` and
    `target_series.n_components=2`):
        comp0_target_lag0 | comp1_target_lag0 | ... | comp0_target_lag3 | comp1_target_lag3

    Note : will only use the component names of the first series from `target_series`, `past_covariates`,
    `future_covariates`, and static_covariates.

    The naming convention for target, past and future covariates is: ``"{name}_{type}_lag{i}"``, where:

        - ``{name}`` the component name of the (first) series
        - ``{type}`` is the feature type, one of "target", "pastcov", and "futcov"
        - ``{i}`` is the lag value

    The naming convention for static covariates is: ``"{name}_statcov_target_{comp}"``, where:

        - ``{name}`` the static covariate name of the (first) series
        - ``{comp}`` the target component name of the (first) that the static covariate act on. If the static
            covariate acts globally on a multivariate target series, will show "global".

    Returns
    -------
    features_cols_name
        The names of the lagged features in the `X` array generated by `create_lagged_data()`
        as a `List[str]`. If `concatenate=True`, also contains the columns names for
        the `y` array (on the right).
    labels_cols_name
        The names of the lagged features in the `y` array generated by `create_lagged_data()`
         as a `List[str]`.

    See Also
    --------
        tabularization.create_lagged_data : generate the lagged features and labels as (list of) Arrays.
    """
    target_series = series2seq(target_series)
    past_covariates = series2seq(past_covariates)
    future_covariates = series2seq(future_covariates)

    lagged_feature_names = []
    label_feature_names = []
    for variate, variate_lags, variate_type in zip(
        [target_series, past_covariates, future_covariates],
        [lags, lags_past_covariates, lags_future_covariates],
        ["target", "pastcov", "futcov"],
    ):
        if variate is None or variate_lags is None:
            continue

        components = get_single_series(variate).components.tolist()
        if isinstance(variate_lags, dict):
            for name in components:
                lagged_feature_names += [
                    f"{name}_{variate_type}_lag{lag}" for lag in variate_lags[name]
                ]
        else:
            lagged_feature_names += [
                f"{name}_{variate_type}_lag{lag}"
                for lag in variate_lags
                for name in components
            ]

        if variate_type == "target" and lags:
            label_feature_names = [
                f"{name}_target_lag{lag}"
                for lag in range(output_chunk_length)
                for name in components
            ]

    # static covariates
    if use_static_covariates:
        static_covs = get_single_series(target_series).static_covariates
        # static covariate names
        names = static_covs.columns.tolist()
        # target components that the static covariates reference to
        comps = static_covs.index.tolist()
        lagged_feature_names += [
            f"{name}_statcov_target_{comp}" for name in names for comp in comps
        ]

    if concatenate:
        lagged_feature_names += label_feature_names

    return lagged_feature_names, label_feature_names


def _create_lagged_data_by_moving_window(
    target_series: Optional[TimeSeries],
    output_chunk_length: int,
    past_covariates: Optional[TimeSeries],
    future_covariates: Optional[TimeSeries],
    lags: Optional[Union[Sequence[int], Dict[str, List[int]]]],
    lags_past_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]],
    lags_future_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]],
    max_samples_per_ts: Optional[int],
    multi_models: bool,
    check_inputs: bool,
    is_training: bool,
) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Helper function called by `create_lagged_data` that computes `X`, `y`, and `times` by
    extracting 'moving windows' from each series using the `strided_moving_window`
    function. More specifically, to extract the features of a particular series for an
    arbitrary time `t`, a 'window' between times `t - max_lag` and `t - min_lag` is
    extracted, where `max_lag` and `min_lag` are the largest and smallest magnitude lags
    requested for that particular series. After extracting this window, the requested lag
    values between these two minimum and maximum lag values can be extracted. Similarly,
    the labels for time `t` are formed simply by extracting a window between times `t`
    and `t + output_chunk_length - 1` from the target series. In both cases, the extracted
    windows can then be reshaped into the correct shape. This approach can only be used if
    we *can* assume that the specified series are all of the same frequency.
    """
    feature_times, min_lags, max_lags = _get_feature_times(
        target_series,
        past_covariates,
        future_covariates,
        lags,
        lags_past_covariates,
        lags_future_covariates,
        output_chunk_length,
        is_training=is_training,
        return_min_and_max_lags=True,
        check_inputs=check_inputs,
    )
    if check_inputs:
        series_and_lags_not_specified = [max_lag is None for max_lag in max_lags]
        raise_if(
            all(series_and_lags_not_specified),
            "Must specify at least one series-lags pair.",
        )
    time_bounds = get_shared_times_bounds(*feature_times)
    raise_if(
        time_bounds is None,
        "Specified series do not share any common times for which features can be created.",
    )
    freq = _get_freqs(target_series, past_covariates, future_covariates)[0]
    if isinstance(time_bounds[0], int):
        # `stop` is exclusive, so need `+ freq` to include end-point:
        times = pd.RangeIndex(
            start=time_bounds[0], stop=time_bounds[1] + freq, step=freq
        )
    else:
        times = pd.date_range(start=time_bounds[0], end=time_bounds[1], freq=freq)
    num_samples = len(times)
    if num_samples > max_samples_per_ts:
        times = times[-max_samples_per_ts:]
        num_samples = max_samples_per_ts
    # Time index of 'earliest' constructed observation:
    start_time = times[0]
    # Construct features array X:
    X = []
    start_time_idx = None
    target_start_time_idx = None
    for i, (series_i, lags_i, min_lag_i, max_lag_i) in enumerate(
        zip(
            [target_series, past_covariates, future_covariates],
            [lags, lags_past_covariates, lags_future_covariates],
            min_lags,
            max_lags,
        )
    ):
        series_and_lags_specified = min_lag_i is not None
        is_target_series = is_training and (i == 0)
        if is_target_series or series_and_lags_specified:
            time_index_i = series_i.time_index
            # If lags are sufficiently large, `series_i` may not contain all
            # feature times. For example, if `lags_past_covariates = [-50]`,
            # then we can construct features for time `51` using the value
            # of `past_covariates` at time `1`, but `past_covariates` may
            # only go up to time `30`. This does *not* occur when considering
            # the target series, however, since this series must have values
            # for all feature times - these values will become labels.
            # If `start_time` not included in `time_index_i`, can 'manually' calculate
            # what its index *would* be if `time_index_i` were extended to include that time:
            if not is_target_series and (time_index_i[-1] < start_time):
                # Series frequency represents a non-ambiguous timedelta value (not ‘M’, ‘Y’ or ‘y’)
                if pd.to_timedelta(series_i.freq, errors="coerce") is not pd.NaT:
                    start_time_idx = (
                        len(time_index_i)
                        - 1
                        + (start_time - time_index_i[-1]) // series_i.freq
                    )
                else:
                    # Create a temporary DatetimeIndex to extract the actual start index.
                    start_time_idx = (
                        len(time_index_i)
                        - 1
                        + len(
                            pd.date_range(
                                start=time_index_i[-1] + series_i.freq,
                                end=start_time,
                                freq=series_i.freq,
                            )
                        )
                    )
            elif not is_target_series and (time_index_i[0] >= start_time):
                start_time_idx = max_lag_i
            # If `start_time` *is* included in `time_index_i`, need to binary search `time_index_i`
            # for its position:
            else:
                start_time_idx = np.searchsorted(time_index_i, start_time)
        if series_and_lags_specified:
            # Windows taken between times `t - max_lag_i` and `t - min_lag_i`
            window_len = max_lag_i - min_lag_i + 1
            first_window_start_idx = start_time_idx - max_lag_i
            first_window_end_idx = first_window_start_idx + window_len
            # Other windows are formed by sequentially shifting first window forward
            # by 1 index position each time; to create `(num_samples - 1)` more windows
            # in addition to the first window, need to take `(num_samples - 1)` values
            # after `first_window_end_idx`:
            vals = series_i.all_values(copy=False)[
                first_window_start_idx : first_window_end_idx + num_samples - 1, :, :
            ]
            windows = strided_moving_window(
                vals, window_len, stride=1, axis=0, check_inputs=False
            )
            # Within each window, the `-1` indexed value (i.e. the value at the very end of
            # the window) corresponds to time `t - min_lag_i`. The negative index of the time
            # `t + lag_i` within this window is, therefore, `-1 + lag_i + min_lag_i`:
            if isinstance(lags_i, list):
                lags_to_extract = np.array(lags_i, dtype=int) + min_lag_i - 1
            else:
                # Lags are grouped by component, extracted from the same window
                lags_to_extract = [
                    np.array(comp_lags, dtype=int) + min_lag_i - 1
                    for comp_lags in lags_i.values()
                ]
            lagged_vals = _extract_lagged_vals_from_windows(windows, lags_to_extract)
            X.append(lagged_vals)
        # Cache `start_time_idx` for label creation:
        if is_target_series:
            target_start_time_idx = start_time_idx
    X = np.concatenate(X, axis=1)
    # Construct labels array `y`:
    if is_training:
        # All values between times `t` and `t + output_chunk_length` used as labels:
        # Window taken between times `t` and `t + output_chunk_length - 1`:
        first_window_start_idx = target_start_time_idx
        # Add `+ 1` since end index is exclusive in Python:
        first_window_end_idx = target_start_time_idx + output_chunk_length
        # To create `(num_samples - 1)` other windows in addition to first window,
        # must take `(num_samples - 1)` values ahead of `first_window_end_idx`
        vals = target_series.all_values(copy=False)[
            first_window_start_idx : first_window_end_idx + num_samples - 1,
            :,
            :,
        ]
        windows = strided_moving_window(
            vals,
            window_len=output_chunk_length,
            stride=1,
            axis=0,
            check_inputs=False,
        )
        lags_to_extract = None if multi_models else -np.ones((1,), dtype=int)
        y = _extract_lagged_vals_from_windows(windows, lags_to_extract)
        # Only values at times `t + output_chunk_length - 1` used as labels:
    else:
        y = None
    return X, y, times


def _extract_lagged_vals_from_windows(
    windows: np.ndarray,
    lags_to_extract: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
) -> np.ndarray:
    """
    Helper function called by `_create_lagged_data_by_moving_window` that
    reshapes the `windows` formed by `strided_moving_window` from the
    shape `(num_windows, num_components, num_series, window_len)` to the
    shape `(num_windows, num_components * window_len, num_series)`. This reshaping
    is done such that the order of elements along axis 1 matches the pattern
    described in the docstring of `create_lagged_data`.

    If `lags_to_extract` is not specified, all of the values within each window is extracted.
    If `lags_to_extract` is specified as an np.ndarray, then only those values within each window that
    are indexed by `lags_to_extract` will be returned. In such cases, the shape of the returned
    lagged values is `(num_windows, num_components * lags_to_extract.size, num_series)`. For example,
    if `lags_to_extract = [-2]`, only the second-to-last values within each window will be extracted.
    If `lags_to_extract` is specified as a list of np.ndarray, the values will be extracted using the
    lags provided for each component. In such cases, the shape of the returned lagged values is
    `(num_windows, sum([comp_lags.size for comp_lags in lags_to_extract]), num_series)`. For example,
    if `lags_to_extract = [[-2, -1], [-1]]`, the second-to-last and last values of the first component
    and the last values of the second component within each window will be extracted.
    """
    # windows.shape = (num_windows, num_components, num_samples, window_len):
    if isinstance(lags_to_extract, list):
        # iterate over the components-specific lags
        comp_windows = [
            windows[:, i, :, comp_lags_to_extract]
            for i, comp_lags_to_extract in enumerate(lags_to_extract)
        ]
        # windows.shape = (sum(lags_len) across components, num_windows, num_samples):
        windows = np.concatenate(comp_windows, axis=0)
        lagged_vals = np.moveaxis(windows, (1, 0, 2), (0, 1, 2))
    else:
        if lags_to_extract is not None:
            windows = windows[:, :, :, lags_to_extract]
        # windows.shape = (num_windows, window_len, num_components, num_samples):
        windows = np.moveaxis(windows, (0, 3, 1, 2), (0, 1, 2, 3))
        # lagged_vals.shape = (num_windows, num_components*window_len, num_samples):
        lagged_vals = windows.reshape((windows.shape[0], -1, windows.shape[-1]))
    return lagged_vals


def _create_lagged_data_by_intersecting_times(
    target_series: TimeSeries,
    output_chunk_length: int,
    past_covariates: Optional[TimeSeries],
    future_covariates: Optional[TimeSeries],
    lags: Optional[Sequence[int]],
    lags_past_covariates: Optional[Sequence[int]],
    lags_future_covariates: Optional[Sequence[int]],
    max_samples_per_ts: Optional[int],
    multi_models: bool,
    check_inputs: bool,
    is_training: bool,
) -> Tuple[np.ndarray, np.ndarray, Union[pd.RangeIndex, pd.DatetimeIndex]]:
    """
    Helper function called by `_create_lagged_data` that computes `X`, `y`, and `times` by
    first finding the time points in each series that *could* be used to create features/labels,
    and then finding which of these 'available' times is shared by all specified series. The lagged
    values are then extracted by finding the index of each of these 'shared times' in each series,
    and then offsetting this index by the requested lag value (if constructing `X`) or the requested
    `output_chunk_length` (if constructing `y`). This approach is used if we *cannot* assume that the
    specified series are of the same frequency.
    """
    feature_times, min_lags, _ = _get_feature_times(
        target_series,
        past_covariates,
        future_covariates,
        lags,
        lags_past_covariates,
        lags_future_covariates,
        output_chunk_length,
        is_training=is_training,
        return_min_and_max_lags=True,
        check_inputs=check_inputs,
    )
    if check_inputs:
        series_and_lags_not_specified = [min_lag is None for min_lag in min_lags]
        raise_if(
            all(series_and_lags_not_specified),
            "Must specify at least one series-lags pair.",
        )
    shared_times = get_shared_times(*feature_times, sort=True)
    raise_if(
        shared_times is None,
        "Specified series do not share any common times for which features can be created.",
    )
    if len(shared_times) > max_samples_per_ts:
        shared_times = shared_times[-max_samples_per_ts:]
    X = []
    shared_time_idx = None
    label_shared_time_idx = None
    for i, (series_i, lags_i, min_lag_i) in enumerate(
        zip(
            [target_series, past_covariates, future_covariates],
            [lags, lags_past_covariates, lags_future_covariates],
            min_lags,
        )
    ):
        series_and_lags_specified = min_lag_i is not None
        is_target_series = is_training and (i == 0)
        if series_and_lags_specified or is_target_series:
            time_index_i = series_i.time_index
            add_to_start = (not is_target_series) and (
                time_index_i[0] > shared_times[0]
            )
            add_to_end = (not is_target_series) and (
                time_index_i[-1] < shared_times[-1]
            )
            if add_to_start or add_to_end:
                new_start = shared_times[0] if add_to_start else None
                new_end = shared_times[-1] if add_to_end else None
                num_prepended = (
                    (time_index_i[0] - shared_times[0]) // series_i.freq
                    if add_to_start
                    else 0
                )
                time_index_i = _extend_time_index(
                    time_index_i, series_i.freq, new_start=new_start, new_end=new_end
                )
            else:
                num_prepended = 0
            shared_time_idx = (
                np.searchsorted(time_index_i, shared_times).reshape(-1, 1)
                - num_prepended
            )
        if series_and_lags_specified:
            idx_to_get = shared_time_idx + np.array(lags_i, dtype=int)
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
    if is_training:
        if multi_models:
            # All points between time `t` and `t + output_chunk_length - 1` are labels:
            idx_to_get = label_shared_time_idx + np.arange(output_chunk_length)
        else:
            # Only point at time `t + output_chunk_length - 1` is a label:
            idx_to_get = label_shared_time_idx + output_chunk_length - 1
        # Before reshaping: lagged_vals.shape = (n_observations, num_lags, n_components, n_samples)
        lagged_vals = target_series.all_values(copy=False)[idx_to_get, :, :]
        # After reshaping: lagged_vals.shape = (n_observations, num_lags*n_components, n_samples)
        y = lagged_vals.reshape(lagged_vals.shape[0], -1, lagged_vals.shape[-1])
    else:
        y = None
    return X, y, shared_times


# For convenience, define following types for `_get_feature_times`:
FeatureTimes = Tuple[
    Optional[Union[pd.Index, pd.DatetimeIndex, pd.RangeIndex]],
    Optional[Union[pd.Index, pd.DatetimeIndex, pd.RangeIndex]],
    Optional[Union[pd.Index, pd.DatetimeIndex, pd.RangeIndex]],
]
MinLags = Tuple[Optional[int], Optional[int], Optional[int]]
MaxLags = Tuple[Optional[int], Optional[int], Optional[int]]


def _get_feature_times(
    target_series: Optional[TimeSeries] = None,
    past_covariates: Optional[TimeSeries] = None,
    future_covariates: Optional[TimeSeries] = None,
    lags: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    lags_past_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    lags_future_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]] = None,
    output_chunk_length: int = 1,
    is_training: bool = True,
    return_min_and_max_lags: bool = False,
    check_inputs: bool = True,
) -> Union[FeatureTimes, Tuple[FeatureTimes, MinLags, MaxLags]]:
    """
    Returns a tuple containing the times in `target_series`, the times in `past_covariates`, and the times in
    `future_covariates` that *could* be used to create features. The returned tuple of times can then be passed
    to `get_shared_times` to compute the 'eligible time points' shared by all of the specified series.

    Notes
    -----
    For the purposes of extracting feature times from each series, we define the `min_lag` and `max_lag` of
    each series to be:
            `min_lag = -max(lags_*)`,
            `max_lag = -min(lags_*)`
    where `lags_*` denotes either `lags`, `lags_past_covariates`, or `lags_future_covariates`.

    For both `lags` and `lags_past_covariates`, `min_lag` and `max_lag` are guaranteed to be positive values,
    since the values in `lags` and `lags_past_covariates` must all be negative. For these two series then,
    `min_lag` and `max_lag` represent the smallest and largest magnitude lags requested by the user. For example:
            `lags = [-3, -2, -1] -> min_lag = 1, max_lag = 3`

    The values contained in `lags_future_covariates`, on the other hand, can be negative, zero, or positive; this
    means that there are three cases to consider:
        1. Both `min_lag` and `max_lag` are positive, which means that all the values in `lags_future_covariates`
        are negative. In this case, `min_lag` and `max_lag` correspond to the to the smallest and largest
        lag magnitudes respectively. For example:
                `lags_future_covariates = [-3, -2, -1] -> min_lag = 1, max_lag = 3`
        2. `min_lag` is non-positive (i.e. zero or negative), but `max_lag` is positive, which means that
        `lags_future_covariates` contains both negative and non-negative (i.e. zero or positive) lag values.
        In this case, `abs(min_lag)` corresponds to the magnitude of the largest *non-negative* lag value in
        `lags_future_covariates`, whilst `max_lag` corresponds to the largest *negative* lag value in
        `lags_future_covariates`. For example:
                `lags_future_covariates = [-2, -1, 0, 1, 3] -> min_lag = -3, max_lag = 2`
        3. Both `min_lag` and `max_lag` are non-positive, which means that `lags_future_covariates` contains
        only non-negative lag values. In this case, `abs(min_lag)` and `abs(max_lag)`, rather confusingly,
        correspond to the largest and smallest lag magnitudes respectively. For example:
                `lags_future_covariates = [1, 2, 3] -> min_lag = -3, max_lag = -1`
    In all three cases, we have `min_lag <= max_lag`. As a direct consequence:
        1. `min_lag > 0` is a sufficient condition for `min_lag` and `max_lag` both being positive (i.e. Case 1).
        2. `max_lag <= 0` is a sufficient condition for `min_lag` and `max_lag` both being non-positive (i.e. Case 2).

    To extract feature times from a `target_series` when `is_training = True`, the following steps are performed:
        1. The first `max_lag` times of the series are excluded; these times have too few preceeding values to
        construct features from.
        2. The last `output_chunk_length - 1` times are excluded; these times have too few succeeding times
        to construct labels from.

    To extract feature times from a `target_series` when `is_training = False`, the following steps are performed:
        1. An additional `min_lag` times are appended to the end of the series; although these times are not contained
        in the original series, we're able to construct features for them since we only need the values of the series
        from time `t - max_lag` to `t - min_lag` to construct a feature for time `t`.
        2. The first `max_lag` times of the series are then excluded; these times have too few preceeding values to
        construct features from.
    The exact same procedure is performed to extract the feature times from a `past_covariates` series.

    To extract feature times from `future_covariates`, we perform the following steps:
        1. Depending on the signs of `min_lag` and `max_lag`, additional times are either prepended or appended
        to the original series. More specifically:
            a) If `min_lag` and `max_lag` are both positive (i.e. `min_lag > 0`), then an additional `min_lag` times
            are appended to the end of the series; as previously mentioned, we only need values up to time `t - min_lag`
            to construct a feature for time `t`.
            b) If `min_lag` and `max_lag` are both non-positive (i.e. `max_lag < 0`), then an additional `abs(max_lag)`
            times are prepended to the start of the series; this is because we only need to know the values of the
            series *after* time `t + abs(max_lag)` to construct a feature for time `t` when we're only extracting
            positive lags from `future_covariates`.
            c) If `min_lag` is non-positive and `max_lag` is positive, then *no additional times* are added to the
            series, since constructing a feature for time `t` requires knowing values from time `t - max_lag` to
            time `t + abs(min_lag)`; in other words, we need to have access to time `t` itself.
        2. If `min_lag < 0`, the last `abs(min_lag)` times are excluded, since these values have fewer
        than `abs(min_lag)` values after them, which means we're unable to construct features for these times.
        3. If `max_lag > 0`, the first `max_lag` times are excluded, since these values have fewer than `max_lag` values
        before them, which means we're unable to construct features for these times.

    Some additional behaviours to note about the `_get_feature_times` function are:
        1. If `return_min_and_max_lags = True`, the smallest and largest lag value for each
        series is also returned as a pair of tuples.
        2. For those series which are either unspecified, a `None` value takes the place of
        that series' feature time, minimum lag values, and maximum lag value.
        3. If `is_training = True`, then `target_series` and `output_chunk_length` must
        be provided.

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
    is_training
        Optionally, specifies that training data is to be generated from the specified series. If `True`,
        `target_series`, `output_chunk_length`, and `multi_models` must all be specified.
    check_inputs
        Optionally, specifies that the `lags_*` and `series_*` inputs should be checked for validity. Should be set
        to `False` if inputs have already been checked for validity (e.g. inside the `__init__` of a class), otherwise
        should be set to `True`.
    return_min_and_max_lags
        Optionally, specifies whether the largest magnitude lag value for each series should also be returned along with
        the 'eligible' feature times

    Note: if the lags are provided as a dictionary for the target series or any of the covariates series, the
    component-specific lags are grouped into a single list to compute the corresponding feature time.

    Returns
    -------
    feature_times
        A tuple containing all the 'eligible feature times' in `target_series`, in `past_covariates`, and in
        `future_covariates`, in that order. If a particular series-lag pair isn't fully specified, then a `None`
        will take the place of that series' eligible times.
    min_lags
        Optionally, a tuple containing the smallest lag value in `lags`, `lags_past_covariates`, and
        `lags_future_covariates`, in that order. If a particular series-lag pair isn't fully specified, then a `None`
        will take the place of that series' minimum lag values.
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
        is_training and (target_series is None),
        "Must specify `target_series` when `is_training = True`.",
    )
    if check_inputs:
        raise_if(
            not isinstance(output_chunk_length, int) or output_chunk_length < 1,
            "`output_chunk_length` must be a positive `int`.",
        )
        _check_lags(lags, lags_past_covariates, lags_future_covariates)
    feature_times, min_lags, max_lags = [], [], []
    for name_i, series_i, lags_i in zip(
        ["target_series", "past_covariates", "future_covariates"],
        [target_series, past_covariates, future_covariates],
        [lags, lags_past_covariates, lags_future_covariates],
    ):
        # union of the component-specific lags, unsorted
        if isinstance(lags_i, dict):
            lags_i = list(set(chain(*lags_i.values())))

        if check_inputs and (series_i is not None):
            _check_series_length(
                series_i,
                lags_i,
                output_chunk_length,
                is_training,
                name_i,
            )
        series_specified = series_i is not None
        lags_specified = lags_i is not None
        is_label_series = is_training and name_i == "target_series"
        times_i = series_i.time_index if series_specified else None
        max_lag_i = -min(lags_i) if lags_specified else None
        min_lag_i = -max(lags_i) if lags_specified else None
        if is_label_series:
            # Exclude last `output_chunk_length - 1` times:
            end_idx = -output_chunk_length + 1 if output_chunk_length > 1 else None
            times_i = times_i[:end_idx]
        elif series_specified and lags_specified:
            # Prepend times to start of series - see Step 1a for extracting
            # feature times from `future_covariates` in `Notes`:
            new_start = (
                times_i[0] + series_i.freq * max_lag_i if max_lag_i < 0 else None
            )
            # Append times to end of series - see Step 1b for extracting features
            # times from `future_covariates`, or Step 1 for extracting features
            # from `target_series`/`past_covariates` in `Notes`:
            new_end = times_i[-1] + series_i.freq * min_lag_i if min_lag_i > 0 else None
            times_i = _extend_time_index(
                times_i, series_i.freq, new_start=new_start, new_end=new_end
            )
        if series_specified and lags_specified:
            # Exclude last `abs(min_lag)` times - see Step 2 for extracting feature
            # times from `future_covariates` in `Notes`:
            if min_lag_i < 0:
                times_i = times_i[:min_lag_i]
            # Exclude first `max_lag` times - see Step 3 for extracting feature times
            # from `future_covariates`, or Step 2 in extracting feature times from
            # `target_series`/`past_covariates` in `Notes`:
            if max_lag_i > 0:
                times_i = times_i[max_lag_i:]
        elif (not is_label_series) and (series_specified ^ lags_specified):
            # Warn user that series/lags input will be ignored:
            times_i = max_lag_i = None
            lags_name = "lags" if name_i == "target_series" else f"lags_{name_i}"
            specified = lags_name if lags_specified else name_i
            unspecified = name_i if lags_specified else lags_name
            warnings.warn(
                f"`{specified}` was specified without accompanying `{unspecified}` and, thus, will be ignored."
            )
        feature_times.append(times_i)
        # Note `max_lag_i` and `min_lag_i` if requested:
        if series_specified and lags_specified:
            min_lags.append(min_lag_i)
            max_lags.append(max_lag_i)
        else:
            min_lags.append(None)
            max_lags.append(None)
    return (
        (feature_times, min_lags, max_lags)
        if return_min_and_max_lags
        else feature_times
    )


def get_shared_times(
    *series_or_times: Union[TimeSeries, pd.Index, None], sort: bool = True
) -> pd.Index:
    """
    Returns the times shared by all of the specified `TimeSeries` or time indexes (i.e. the intersection of all
    these times). If `sort = True`, then these shared times are sorted from earliest to latest. Any `TimeSeries` or
    time indices in `series_or_times` that aren't specified (i.e. are `None`) are simply ignored.

    Parameters
    ----------
    series_or_times
        The `TimeSeries` and/or time indices that should 'intersected'.
    sort
        Optionally, specifies that the returned shared times should be sorted from earliest to latest.

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

    specified_inputs = [series for series in series_or_times if series is not None]

    if not specified_inputs:
        shared_times = None
    elif len(specified_inputs) == 1:
        shared_times = (
            specified_inputs[0].time_index
            if isinstance(specified_inputs[0], TimeSeries)
            else specified_inputs[0]
        )
        shared_times = None if len(shared_times) == 0 else shared_times
    else:
        shared_times = reduce(intersection_func, specified_inputs)
        # Empty intersection may result from intersecting time indices being of different types - throw error if so:
        if shared_times.empty:
            shared_times = None
            times_types = [
                type(ts.time_index if isinstance(ts, TimeSeries) else ts)
                for ts in specified_inputs
            ]
            raise_if_not(
                len(set(times_types)) == 1,
                (
                    "Specified series and/or times must all "
                    "have the same type of `time_index` (i.e. all "
                    "`pd.RangeIndex` or all `pd.DatetimeIndex`)."
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
    Here, each `-` denotes an observation at a specific time. In this example, `find_time_overlap_bounds` will
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
        if (val is not None) and (len(val) > 0):
            start_times.append(
                val.start_time() if isinstance(val, TimeSeries) else val[0]
            )
            end_times.append(val.end_time() if isinstance(val, TimeSeries) else val[-1])
    if not start_times:
        bounds = None
    else:
        times_types = [type(time) for time in start_times]
        raise_if_not(
            len(set(times_types)) == 1,
            (
                "Specified series and/or times must all "
                "have the same type of `time_index` "
                "(i.e. all `pd.RangeIndex` or all `pd.DatetimeIndex`)."
            ),
        )
        # If `start_times` empty, no series were specified -> `bounds = (1, -1)` will
        # be 'converted' to `None` in next line:
        bounds = (max(start_times), min(end_times)) if start_times else (1, -1)
        # Specified timeseries share no overlapping periods.
        if bounds[1] < bounds[0]:
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
        Optionally, specifies whether inputs should be checked for validity. Should be set
        to `False` if inputs have already been checked for validity (e.g. inside the `__init__`
        of a class), otherwise should be set to `True`. See [1]_ for further details.

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
        raise_if(
            not isinstance(stride, int) or stride < 1,
            "`stride` must be a positive `int`.",
        )
        raise_if(
            not isinstance(window_len, int) or window_len < 1,
            "`window_len` must be a positive `int`.",
        )
        raise_if(
            not isinstance(axis, int) or axis > x.ndim - 1 or axis < -x.ndim,
            "`axis` must be an `int` that is less than `x.ndim`.",
        )
        raise_if(
            window_len > x.shape[axis],
            "`window_len` must be less than or equal to x.shape[axis].",
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
#   Private Functions
#


def _extend_time_index(
    time_index: pd.Index,
    freq: Union[int, str],
    new_start: Optional[pd.Timestamp] = None,
    new_end: Optional[pd.Timestamp] = None,
):
    """
    Extends a `time_index` of frequency `freq` such that it now ends at time `new_end`;
    the fastest way to do this is actually to create a new time index from scratch.
    """
    is_range_idx = isinstance(freq, int)
    if new_start is None:
        new_start = time_index[0]
    if new_end is None:
        new_end = time_index[-1]
    if is_range_idx:
        time_index = pd.RangeIndex(start=new_start, stop=new_end + freq, step=freq)
    else:
        time_index = pd.date_range(start=new_start, end=new_end, freq=freq)
    return time_index


def _get_freqs(*series: Union[TimeSeries, None]):
    """
    Returns list with the frequency of all of the specified (i.e. non-`None`) `series`.
    """
    freqs = []
    for ts in series:
        if ts is not None:
            freqs.append(ts.freq)
    return freqs


def _all_equal_freq(*series: Union[TimeSeries, None]) -> bool:
    """
    Returns `True` is all of the specified (i.e. non-`None`) `series` have the same frequency.
    """
    freqs = _get_freqs(*series)
    return len(set(freqs)) == 1


def _check_lags(
    lags: Optional[Union[Sequence[int], Dict[str, List[int]]]],
    lags_past_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]],
    lags_future_covariates: Optional[Union[Sequence[int], Dict[str, List[int]]]],
) -> None:
    """
    Throws `ValueError` if any `lag` values aren't negative OR if no lags have been specified.
    """
    all_lags = [lags, lags_past_covariates, lags_future_covariates]
    suffixes = ["", "_past_covariates", "_future_covariates"]
    lags_is_none = []
    for i, (suffix, lags_i) in enumerate(zip(suffixes, all_lags)):
        lags_is_none.append(lags_i is None)
        if not lags_is_none[-1]:
            is_target_or_past = i < 2
            max_lag = -1 if is_target_or_past else inf

            if isinstance(lags_i, dict):
                lags_i = list(set(chain(*lags_i.values())))

            raise_if(
                any((lag > max_lag or not isinstance(lag, int)) for lag in lags_i),
                f"`lags{suffix}` must be a `Sequence` or `Dict` containing only `int` values less than {max_lag + 1}.",
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
    minimum_len, minimum_len_str = None, None
    if is_label_series:
        minimum_len_str = (
            "-min(lags) + output_chunk_length"
            if lags_specified
            else "output_chunk_length"
        )
        minimum_len = (
            -min(lags) + output_chunk_length if lags_specified else output_chunk_length
        )
    elif lags_specified:
        lags_name = "lags" if name == "target_series" else f"lags_{name}"
        minimum_len_str = f"-min({lags_name}) + max({lags_name}) + 1"
        minimum_len = -min(lags) + max(lags) + 1
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
