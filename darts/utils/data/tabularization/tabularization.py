"""
Data Tabularization Methods
---------------------------

Internal tabularization methods to convert time series data into tabular format for usage with
SKLearn-like models. This module is not part of the public Darts API.
"""

import warnings
from collections.abc import Sequence
from functools import reduce
from math import inf
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided

from darts import TimeSeries
from darts.logging import raise_log
from darts.typing import Frequency, TimeIndex, TimeSeriesLike, TimeStamp
from darts.utils.data.tabularization.data import LaggedData
from darts.utils.data.tabularization.features import LaggedFeatures
from darts.utils.data.utils import _process_sample_weight
from darts.utils.ts_utils import get_single_series, series2seq
from darts.utils.utils import n_steps_between

if TYPE_CHECKING:
    from darts.utils.data.tabularization.lags import Lags
    from darts.utils.forecast_config import ForecastConfig

NP_2_OR_ABOVE = int(np.__version__.split(".")[0]) >= 2
STABLE_SORT_KWARGS = {"stable": True} if NP_2_OR_ABOVE else {"kind": "stable"}

ArrayOrArraySequence = np.ndarray | Sequence[np.ndarray]

_TARGET = "target_series"
_PAST_COV = "past_covariates"
_FUTURE_COV = "future_covariates"


def _create_lagged_data_core(
    config: "ForecastConfig",
    target_series: TimeSeriesLike | None = None,
    past_covariates: TimeSeriesLike | None = None,
    future_covariates: TimeSeriesLike | None = None,
    uses_static_covariates: bool = True,
    last_static_covariates_shape: tuple[int, int] | None = None,
    max_samples_per_ts: int | None = None,
    check_inputs: bool = True,
    use_moving_windows: bool = True,
    is_training: bool = True,
    concatenate: bool = True,
    sample_weight: str | TimeSeriesLike | None = None,
    stride: int = 1,
    show_warnings: bool = True,
) -> tuple[
    ArrayOrArraySequence,
    None | ArrayOrArraySequence,
    Sequence[pd.Index],
    tuple[int, int] | None,
    ArrayOrArraySequence | None,
]:
    """Implementation of :func:`create_lagged_data`."""
    if is_training and (target_series is None):
        raise_log(
            ValueError("Must specify `target_series` if `is_training = True`."),
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
    if len(seq_ts_lens) > 1:
        raise_log(
            ValueError(
                "Must specify the same number of `TimeSeries` for each series input."
            ),
        )

    # process / check sample weight and generate series in case of built-in weight generator
    sample_weight = _process_sample_weight(sample_weight, target_series)

    lags_passed_as_dict = any(
        config.is_component_specific(kind) for kind in ("target", "past", "future")
    )
    if (not use_moving_windows) and lags_passed_as_dict:
        raise_log(
            ValueError(
                "`use_moving_windows=False` is not supported when any of the lags is "
                "provided as a dictionary."
            ),
        )

    if max_samples_per_ts is None:
        max_samples_per_ts = inf

    # lags are identical for multiple series: pre-compute lagged features and reordered lagged features
    lags_extract, lags_order = _get_lagged_indices(config)
    # the feature matrix is shared by all horizons with `multi_models=True`; with
    # `multi_models=False` the estimator predicts the last step of the output chunk
    n_horizons = config.strategy.pred_dim

    X, y, times, sample_weights = [], [], [], []
    for i in range(max(seq_ts_lens)):
        target_i = target_series[i] if target_series else None
        past_i = past_covariates[i] if past_covariates else None
        future_i = future_covariates[i] if future_covariates else None
        sample_weight_i = sample_weight[i] if sample_weight else None
        series_equal_freq = _all_equal_freq(target_i, past_i, future_i)
        # component-wise lags extraction is not support with times intersection at the moment
        if use_moving_windows and lags_passed_as_dict and (not series_equal_freq):
            raise_log(
                ValueError(
                    f"Cannot create tabularized data for the {i}th series because target and covariates don't have "
                    "the same frequency and some of the lags are provided as a dictionary. Either resample the "
                    "series or change the lags definition."
                ),
            )
        if use_moving_windows and series_equal_freq:
            X_i, y_i, times_i, weights_i = _create_lagged_data_by_moving_window(
                target_series=target_i,
                past_covariates=past_i,
                future_covariates=future_i,
                sample_weight=sample_weight_i,
                config=config,
                lags_extract=lags_extract,
                lags_order=lags_order,
                max_samples_per_ts=max_samples_per_ts,
                check_inputs=check_inputs,
                is_training=is_training,
                stride=stride,
                show_warnings=show_warnings,
            )
        else:
            X_i, y_i, times_i, weights_i = _create_lagged_data_by_intersecting_times(
                target_series=target_i,
                past_covariates=past_i,
                future_covariates=future_i,
                sample_weight=sample_weight_i,
                config=config,
                max_samples_per_ts=max_samples_per_ts,
                check_inputs=check_inputs,
                is_training=is_training,
                stride=stride,
                show_warnings=show_warnings,
            )
        n_components = target_i.width if target_i is not None else 1
        X_i = LaggedFeatures(
            array=X_i,
            n_horizons=n_horizons,
            output_horizons=config.strategy.output_horizons(n_components),
        )
        X_i, last_static_covariates_shape = _add_static_covariates_to_lagged_data(
            features=X_i,
            target_series=target_i,
            uses_static_covariates=uses_static_covariates,
            last_shape=last_static_covariates_shape,
        )
        X.append(X_i)
        y.append(y_i)
        times.append(times_i)
        if weights_i is not None:
            sample_weights.append(weights_i)

    if concatenate:
        X = LaggedFeatures.concatenate(X)
    if not is_training:
        y = None
    elif concatenate:
        y = np.concatenate(y, axis=0)

    if sample_weights and concatenate:
        sample_weights = np.concatenate(sample_weights, axis=0)
    elif not sample_weights:
        sample_weights = None
    return X, y, times, last_static_covariates_shape, sample_weights


def _create_lagged_data_from_config(
    config: "ForecastConfig",
    target_series: TimeSeriesLike | None = None,
    past_covariates: TimeSeriesLike | None = None,
    future_covariates: TimeSeriesLike | None = None,
    uses_static_covariates: bool = True,
    last_static_covariates_shape: tuple[int, int] | None = None,
    max_samples_per_ts: int | None = None,
    check_inputs: bool = False,
    is_training: bool = True,
    concatenate: bool = True,
    sample_weight: str | TimeSeriesLike | None = None,
    stride: int = 1,
    use_moving_windows: bool = True,
    show_warnings: bool = True,
) -> LaggedData:
    """Create lagged data from an already resolved forecast config.

    This is the config-driven entry point of the tabularization core. It does not
    re-validate the lags (the config is already resolved) and passes the same
    config to the internal window helpers.
    """
    target_sequence = None if target_series is None else series2seq(target_series)
    X, y, times, static_shape, sample_weights = _create_lagged_data_core(
        config,
        target_series=target_sequence,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        uses_static_covariates=uses_static_covariates,
        last_static_covariates_shape=last_static_covariates_shape,
        max_samples_per_ts=max_samples_per_ts,
        check_inputs=check_inputs,
        use_moving_windows=use_moving_windows,
        is_training=is_training,
        concatenate=concatenate,
        sample_weight=sample_weight,
        stride=stride,
        show_warnings=show_warnings,
    )
    return LaggedData(
        X=X,
        y=y,
        times=times,
        sample_weights=sample_weights,
        series_lengths=(
            None if target_sequence is None else [len(ts) for ts in target_sequence]
        ),
        last_static_covariates_shape=static_shape,
    )


def _series_components(series: TimeSeriesLike | None) -> Sequence[str] | None:
    series = get_single_series(series)
    return None if series is None else series.components


def _resolve_public_config(
    config: "ForecastConfig",
    target_series: TimeSeriesLike | None,
    past_covariates: TimeSeriesLike | None,
    future_covariates: TimeSeriesLike | None,
) -> "ForecastConfig":
    """Resolve a config against the components of the series passed to a public helper."""
    target_components = _series_components(target_series)
    if target_components is None:
        if config.has_lags("target"):
            raise_log(
                ValueError(
                    "`target_series` must be specified when target lags are defined."
                )
            )
        target_components = ()
    return config.resolve(
        target_components=target_components,
        past_components=_series_components(past_covariates),
        future_components=_series_components(future_covariates),
    )


def create_lagged_data(
    config: "ForecastConfig",
    target_series: TimeSeriesLike | None = None,
    past_covariates: TimeSeriesLike | None = None,
    future_covariates: TimeSeriesLike | None = None,
    *,
    uses_static_covariates: bool = True,
    last_static_covariates_shape: tuple[int, int] | None = None,
    max_samples_per_ts: int | None = None,
    check_inputs: bool = True,
    use_moving_windows: bool = True,
    is_training: bool = True,
    concatenate: bool = True,
    sample_weight: str | TimeSeriesLike | None = None,
    stride: int = 1,
    show_warnings: bool = True,
) -> tuple[
    LaggedFeatures | Sequence[LaggedFeatures],
    None | np.ndarray | Sequence[np.ndarray],
    Sequence[pd.Index],
    tuple[int, int] | None,
    np.ndarray | Sequence[np.ndarray] | None,
]:
    """Create lagged features and labels from a :class:`ForecastConfig`.

    The config is resolved against the components of the series before tabularizing. The
    returned tuple is ``(X, y, times, last_static_covariates_shape, sample_weights)``.
    """
    config = _resolve_public_config(
        config, target_series, past_covariates, future_covariates
    )
    data = _create_lagged_data_from_config(
        config,
        target_series=target_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        uses_static_covariates=uses_static_covariates,
        last_static_covariates_shape=last_static_covariates_shape,
        max_samples_per_ts=max_samples_per_ts,
        check_inputs=check_inputs,
        use_moving_windows=use_moving_windows,
        is_training=is_training,
        concatenate=concatenate,
        sample_weight=sample_weight,
        stride=stride,
        show_warnings=show_warnings,
    )
    return (
        data.X,
        data.y,
        data.times,
        data.last_static_covariates_shape,
        data.sample_weights,
    )


def create_lagged_training_data(
    config: "ForecastConfig",
    target_series: TimeSeriesLike,
    past_covariates: TimeSeriesLike | None = None,
    future_covariates: TimeSeriesLike | None = None,
    *,
    uses_static_covariates: bool = True,
    last_static_covariates_shape: tuple[int, int] | None = None,
    max_samples_per_ts: int | None = None,
    check_inputs: bool = True,
    use_moving_windows: bool = True,
    concatenate: bool = True,
    sample_weight: str | TimeSeriesLike | None = None,
    stride: int = 1,
    show_warnings: bool = True,
) -> tuple[
    LaggedFeatures | Sequence[LaggedFeatures],
    None | np.ndarray | Sequence[np.ndarray],
    Sequence[pd.Index],
    tuple[int, int] | None,
    np.ndarray | Sequence[np.ndarray] | None,
]:
    """Config-first wrapper around :func:`create_lagged_data` for training data."""
    return create_lagged_data(
        config,
        target_series=target_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        uses_static_covariates=uses_static_covariates,
        last_static_covariates_shape=last_static_covariates_shape,
        max_samples_per_ts=max_samples_per_ts,
        check_inputs=check_inputs,
        use_moving_windows=use_moving_windows,
        is_training=True,
        concatenate=concatenate,
        sample_weight=sample_weight,
        stride=stride,
        show_warnings=show_warnings,
    )


def create_lagged_prediction_data(
    config: "ForecastConfig",
    target_series: TimeSeriesLike | None = None,
    past_covariates: TimeSeriesLike | None = None,
    future_covariates: TimeSeriesLike | None = None,
    *,
    uses_static_covariates: bool = True,
    last_static_covariates_shape: tuple[int, int] | None = None,
    max_samples_per_ts: int | None = None,
    check_inputs: bool = True,
    use_moving_windows: bool = True,
    concatenate: bool = True,
    stride: int = 1,
    show_warnings: bool = True,
) -> tuple[LaggedFeatures | Sequence[LaggedFeatures], Sequence[pd.Index]]:
    """Config-first wrapper around :func:`create_lagged_data` for prediction data."""
    X, _, times, _, _ = create_lagged_data(
        config,
        target_series=target_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        uses_static_covariates=uses_static_covariates,
        last_static_covariates_shape=last_static_covariates_shape,
        max_samples_per_ts=max_samples_per_ts,
        check_inputs=check_inputs,
        use_moving_windows=use_moving_windows,
        is_training=False,
        concatenate=concatenate,
        stride=stride,
        show_warnings=show_warnings,
    )
    return X, times


def create_lagged_component_names(
    config: "ForecastConfig",
    target_series: TimeSeriesLike | None = None,
    past_covariates: TimeSeriesLike | None = None,
    future_covariates: TimeSeriesLike | None = None,
    *,
    concatenate: bool = True,
    use_static_covariates: bool = False,
) -> tuple[list[str], list[str]]:
    """Config-first wrapper around :func:`_create_lagged_component_names`."""
    config = _resolve_public_config(
        config, target_series, past_covariates, future_covariates
    )
    return _create_lagged_component_names(
        config,
        target_series=target_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        concatenate=concatenate,
        use_static_covariates=use_static_covariates,
    )


def _add_static_covariates_to_lagged_data(
    features: LaggedFeatures | np.ndarray | Sequence[LaggedFeatures | np.ndarray],
    target_series: TimeSeriesLike,
    uses_static_covariates: bool = True,
    last_shape: tuple[int, int] | None = None,
) -> LaggedFeatures | np.ndarray | Sequence[LaggedFeatures | np.ndarray]:
    """
    Add static covariates to the features' table for SKLearnModels.
    If `uses_static_covariates=True`, all target series used in `fit()` and `predict()` must have static
    covariates with identical dimensionality. Otherwise, will not consider static covariates.

    The static covariates are added to the right of the lagged features following the convention:
    with a 2 component series, and 2 static covariates per component:

    .. highlight:: md
    .. code-block:: md

        scov_1_comp_1 | scov_1_comp_2 | scov_2_comp_1 | scov_2_comp_2
    ..

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
                )
            # flatten static covariates along columns -> results in [scov0_comp0, scov0_comp1, scov1_comp0, ...]
            static_covs = ts.static_covariates.values.flatten(order="F")
            feature = features[idx]
            feature_array = (
                feature.array
                if isinstance(feature, LaggedFeatures)
                else np.asarray(feature)
            )
            # we stack the static covariates to the right of lagged features
            # the broadcasting repeats the static covariates along axis=0 to match the number of feature rows
            shape_out = (
                (len(feature_array), len(static_covs))
                if feature_array.ndim == 2
                else (len(feature_array), len(static_covs), 1)
            )
            feature_with_static = np.hstack([
                feature_array,
                np.broadcast_to(static_covs, shape_out[:2]).reshape(shape_out),
            ])
            features[idx] = (
                feature.with_base(feature_with_static)
                if isinstance(feature, LaggedFeatures)
                else feature_with_static
            )

    if input_not_list:
        features = features[0]
    return features, last_shape


def _create_lagged_component_names(
    config: "ForecastConfig",
    target_series: TimeSeriesLike | None = None,
    past_covariates: TimeSeriesLike | None = None,
    future_covariates: TimeSeriesLike | None = None,
    concatenate: bool = True,
    use_static_covariates: bool = False,
) -> tuple[list[str], list[str]]:
    """
    Helper function called to retrieve the name of the features and labels arrays created with
    `create_lagged_data()`. The order of the features is the following:

    Along the `n_lagged_features` axis, `X` has the following structure:

    .. highlight:: md
    .. code-block:: md

        lagged_target | lagged_past_covariates | lagged_future_covariates | static covariates
    ..

    For `*_lags=[-2,-1]` and `*_series.n_components = 2` (lags shared across all the components),
    each `lagged_*` has the following structure (grouped by lags):

    .. highlight:: md
    .. code-block:: md

        comp0_*_lag-2 | comp1_*_lag-2 | comp0_*_lag_-1 | comp1_*_lag-1
    ..

    For `*_lags={'comp0':[-3, -1], 'comp1':[-5, -3]}` and `*_series.n_components = 2` (component-
    specific lags), each `lagged_*` has the following structure (sorted by lags, then by components):

    .. highlight:: md
    .. code-block:: md

        comp1_*_lag-5 | comp0_*_lag-3 | comp1_*_lag_-3 | comp0_*_lag-1
    ..

    and for static covariates (2 static covariates acting on 2 target components):

    .. highlight:: md
    .. code-block:: md

        cov0_*_target_comp0 | cov0_*_target_comp1 | cov1_*_target_comp0 | cov1_*_target_comp1
    ..

    Along the `n_lagged_labels` axis, `y` has the following structure (for `output_chunk_length=4` and
    `target_series.n_components=2`):

    .. highlight:: md
    .. code-block:: md

        comp0_target_lag0 | comp1_target_lag0 | ... | comp0_target_lag3 | comp1_target_lag3
    ..

    Note : will only use the component names of the first series from `target_series`, `past_covariates`,
    `future_covariates`, and static_covariates.

    The naming convention for target, past and future covariates lags is: ``"{name}_{type}_lag{i}"``, where:

    - ``{name}`` the component name of the (first) series
    - ``{type}`` is the feature type, one of "target", "pastcov", and "futcov"
    - ``{i}`` is the lag value

    The naming convention for static covariates is: ``"{name}_statcov_target_{comp}"``, where:

    - ``{name}`` the static covariate name of the (first) series
    - ``{comp}`` the target component name of the (first) that the static covariate act on. If the static
      covariate acts globally on a multivariate target series, will show "global_components".

    The naming convention for labels is: ``"{name}_target_hrz{i}"``, where:

    - ``{name}`` the component name of the (first) series
    - ``{i}`` is the step in the forecast horizon

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
        tabularization._create_lagged_data : generate the lagged features and labels as (list of) Arrays.
    """
    target_series = series2seq(target_series)
    past_covariates = series2seq(past_covariates)
    future_covariates = series2seq(future_covariates)
    output_chunk_length = config.output_chunk_length

    lagged_feature_names = []
    label_feature_names = []
    for variate, variate_lags, variate_type in zip(
        [target_series, past_covariates, future_covariates],
        [config.lags_for(kind) for kind in ("target", "past", "future")],
        ["target", "pastcov", "futcov"],
    ):
        if variate is None:
            continue

        components = get_single_series(variate).components.tolist()
        # target labels
        if variate_type == "target":
            label_feature_names = [
                f"{name}_target_hrz{lag}"
                for lag in range(output_chunk_length)
                for name in components
            ]

        if variate_lags is None:
            continue

        if variate_lags.is_component_specific:
            per_component = variate_lags.resolved_per_component
            # combine all the lags and sort them in ascending order across all the components
            comp_lags_reordered = np.concatenate([
                np.array(per_component[comp_name], dtype=int)
                for comp_name in components
            ]).argsort(**STABLE_SORT_KWARGS)
            tmp_lagged_feats_names = []
            for name in components:
                tmp_lagged_feats_names += [
                    f"{name}_{variate_type}_lag{lag}" for lag in per_component[name]
                ]

            # adding feats names reordered across components
            lagged_feature_names += [
                tmp_lagged_feats_names[idx] for idx in comp_lags_reordered
            ]
        else:
            lagged_feature_names += [
                f"{name}_{variate_type}_lag{lag}"
                for lag in variate_lags.shared
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


def _get_lagged_indices(config: "ForecastConfig"):
    """Computes and returns:

    - the lagged feature indices for extraction from windows
    - the reordered indices to apply after the window extraction (in case of component specific lags)

    Assumes that all input series share identical component order.
    """
    lags_extract = []
    lags_order = []
    for kind in ("target", "past", "future"):
        lags_i = config.lags_for(kind)
        if lags_i is None:
            lags_extract.append(None)
            lags_order.append(None)
            continue

        # Within each window, the `-1` indexed value (i.e. the value at the very end of
        # the window) corresponds to time `t - min_lag_i`. The negative index of the time
        # `t + lag_i` within this window is, therefore, `-1 + lag_i + min_lag_i`:
        if not lags_i.is_component_specific:
            lags_extract_i = np.array(lags_i.shared, dtype=int)
            # Feats are already grouped by lags and ordered
            lags_order_i = slice(None)
        else:
            # Assume keys are in the same order as the series components
            # Lags are grouped by component, extracted from the same window
            lags_extract_i = [
                np.array(c_lags, dtype=int)
                for c_lags in lags_i.resolved_per_component.values()
            ]
            # Sort the lags across the components in ascending order
            lags_order_i = np.concatenate(lags_extract_i).argsort(**STABLE_SORT_KWARGS)
        lags_extract.append(lags_extract_i)
        lags_order.append(lags_order_i)
    return lags_extract, lags_order


def _create_lagged_data_by_moving_window(
    target_series: TimeSeries | None,
    past_covariates: TimeSeries | None,
    future_covariates: TimeSeries | None,
    sample_weight: TimeSeries | None,
    config: "ForecastConfig",
    lags_extract: list[np.ndarray | None],
    lags_order: list[np.ndarray | None],
    max_samples_per_ts: int | None,
    check_inputs: bool,
    is_training: bool,
    stride: int,
    show_warnings: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, pd.Index, np.ndarray | None]:
    """
    Helper function called by `_create_lagged_data` that computes `X`, `y`, and `times` by
    extracting 'moving windows' from each series using the `_strided_moving_window`
    function. More specifically, to extract the features of a particular series for an
    arbitrary time `t`, a 'window' between times `t - max_lag` and `t - min_lag` is
    extracted, where `max_lag` and `min_lag` are the largest and smallest magnitude lags
    requested for that particular series. After extracting this window, the requested lag
    values between these two minimum and maximum lag values can be extracted. Similarly,
    the labels for time `t` are formed simply by extracting a window between times `t`
    and `t + output_chunk_length - 1` from the target series. In both cases, the extracted
    windows can then be reshaped into the correct shape. This approach can only be used if
    we *can* assume that the specified series are all of the same frequency.

    Assumes that all the lags are sorted in ascending order.
    """
    output_chunk_length = config.output_chunk_length
    output_chunk_shift = config.output_chunk_shift
    multi_models = config.strategy.name == "direct"

    feature_times, min_lags, max_lags = _get_feature_times(
        config,
        target_series=target_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        is_training=is_training,
        return_min_and_max_lags=True,
        check_inputs=check_inputs,
        show_warnings=show_warnings,
    )
    if check_inputs:
        series_and_lags_not_specified = [max_lag is None for max_lag in max_lags]
        if all(series_and_lags_not_specified):
            raise_log(
                ValueError("Must specify at least one series-lags pair."),
            )
        if not (isinstance(stride, int) and stride > 0):
            raise_log(
                ValueError("`stride` must be a positive integer greater than 0."),
            )
    sample_weight_vals = _extract_sample_weight(sample_weight, target_series)

    time_bounds = _get_shared_times_bounds(*feature_times)
    if time_bounds is None:
        raise_log(
            ValueError(
                "Specified series do not share any common times for which features can be created."
            ),
        )
    freq = _get_freqs(target_series, past_covariates, future_covariates)[0]
    if isinstance(time_bounds[0], int):
        # `stop` is exclusive, so need `+ freq` to include end-point:
        times = pd.RangeIndex(
            start=time_bounds[0], stop=time_bounds[1] + freq, step=freq
        )
    else:
        times = pd.date_range(start=time_bounds[0], end=time_bounds[1], freq=freq)

    if stride > 1:
        # calculate the starting index so that the last element is included after applying the stride
        # equivalent to times[::-stride][::-1]
        times = times[(len(times) - 1) % stride :: stride]
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
    for i, (series_i, lags_extract_i, lags_order_i, min_lag_i, max_lag_i) in enumerate(
        zip(
            [target_series, past_covariates, future_covariates],
            lags_extract,
            lags_order,
            min_lags,
            max_lags,
        )
    ):
        series_and_lags_specified = min_lag_i is not None
        is_target_series = is_training and (i == 0)
        if is_target_series or series_and_lags_specified:
            # get the position of `start_time` relative to the beginning of the current series
            start_time_idx = n_steps_between(
                end=start_time, start=series_i.start_time(), freq=series_i.freq
            )
        if series_and_lags_specified:
            # Windows taken between times `t - max_lag_i` and `t - min_lag_i`
            window_len = max_lag_i - min_lag_i + 1
            first_window_start_idx = start_time_idx - max_lag_i
            first_window_end_idx = first_window_start_idx + window_len
            # Other windows are formed by sequentially shifting first window forward
            # by `stride` position each time; to create `(num_samples - 1)` more windows
            # in addition to the first window, need to take `(num_samples - 1) * stride`
            # values after `first_window_end_idx`:
            vals = series_i.all_values(copy=False)[
                first_window_start_idx : first_window_end_idx
                + (num_samples - 1) * stride,
                :,
                :,
            ]
            windows = _strided_moving_window(
                x=vals, window_len=window_len, stride=stride, axis=0, check_inputs=False
            )

            # Within each window, the `-1` indexed value (i.e. the value at the very end of
            # the window) corresponds to time `t - min_lag_i`. The negative index of the time
            # `t + lag_i` within this window is, therefore, `-1 + lag_i + min_lag_i`:
            # extract lagged values
            lagged_vals = _extract_lagged_vals_from_windows(
                windows, lags_extract_i, lags_shift=min_lag_i - 1
            )
            # extract and append the reordered lagged values
            X.append(lagged_vals[:, lags_order_i])
        # Cache `start_time_idx` for label creation:
        if is_target_series:
            target_start_time_idx = start_time_idx
    X = np.concatenate(X, axis=1)
    # Construct labels array `y`:
    if is_training:
        # All values between times `t` and `t + output_chunk_length` used as labels / weights:
        # Window taken between times `t` and `t + output_chunk_length - 1`:
        first_window_start_idx = target_start_time_idx + output_chunk_shift
        # Add `+ 1` since end index is exclusive in Python:
        first_window_end_idx = (
            target_start_time_idx + output_chunk_length + output_chunk_shift
        )
        lags_to_extract = None if multi_models else -np.ones((1,), dtype=int)

        # extract target labels and sample weights
        y_and_weights = []
        for vals in [target_series.all_values(copy=False), sample_weight_vals]:
            if vals is None:
                y_and_weights.append(None)
                continue

            # To create `(num_samples - 1)` other windows in addition to first window,
            # must take `(num_samples - 1) * stride` values ahead of `first_window_end_idx`
            # to also take the stride into consideration
            vals = vals[
                first_window_start_idx : first_window_end_idx
                + (num_samples - 1) * stride,
                :,
                :,
            ]
            windows = _strided_moving_window(
                x=vals,
                window_len=output_chunk_length,
                stride=stride,
                axis=0,
                check_inputs=False,
            )
            # Only values at times `t + output_chunk_length - 1` used as labels:
            vals = _extract_lagged_vals_from_windows(windows, lags_to_extract)
            y_and_weights.append(vals)

        y, weights = y_and_weights
    else:
        y, weights = None, None
    return X, y, times, weights


def _extract_lagged_vals_from_windows(
    windows: np.ndarray,
    lags_to_extract: np.ndarray | list[np.ndarray] | None = None,
    lags_shift: int = 0,
) -> np.ndarray:
    """
    Helper function called by `_create_lagged_data_by_moving_window` that
    reshapes the `windows` formed by `_strided_moving_window` from the
    shape `(num_windows, num_components, num_series, window_len)` to the
    shape `(num_windows, num_components * window_len, num_series)`. This reshaping
    is done such that the order of elements along axis 1 matches the pattern
    described in the docstring of `_create_lagged_data`.

    If `lags_to_extract` is not specified, all values within each window is extracted.
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
            windows[:, i, :, comp_lags_to_extract + lags_shift]
            for i, comp_lags_to_extract in enumerate(lags_to_extract)
        ]
        # windows.shape = (sum(lags_len) across components, num_windows, num_samples):
        windows = np.concatenate(comp_windows, axis=0)
        lagged_vals = np.moveaxis(windows, (1, 0, 2), (0, 1, 2))
    else:
        if lags_to_extract is not None:
            windows = windows[:, :, :, lags_to_extract + lags_shift]
        # windows.shape = (num_windows, window_len, num_components, num_samples):
        windows = np.moveaxis(windows, (0, 3, 1, 2), (0, 1, 2, 3))
        # lagged_vals.shape = (num_windows, num_components*window_len, num_samples):
        lagged_vals = windows.reshape((windows.shape[0], -1, windows.shape[-1]))
    return lagged_vals


def _create_lagged_data_by_intersecting_times(
    target_series: TimeSeries,
    past_covariates: TimeSeries | None,
    future_covariates: TimeSeries | None,
    sample_weight: TimeSeries | None,
    config: "ForecastConfig",
    max_samples_per_ts: int | None,
    check_inputs: bool,
    is_training: bool,
    stride: int,
    show_warnings: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    pd.RangeIndex | pd.DatetimeIndex,
    np.ndarray | None,
]:
    """
    Helper function called by `_create_lagged_data` that computes `X`, `y`, and `times` by
    first finding the time points in each series that *could* be used to create features/labels,
    and then finding which of these 'available' times is shared by all specified series. The lagged
    values are then extracted by finding the index of each of these 'shared times' in each series,
    and then offsetting this index by the requested lag value (if constructing `X`) or the requested
    `output_chunk_length` (if constructing `y`). This approach is used if we *cannot* assume that the
    specified series are of the same frequency.
    """
    output_chunk_length = config.output_chunk_length
    output_chunk_shift = config.output_chunk_shift
    multi_models = config.strategy.name == "direct"

    feature_times, min_lags, _ = _get_feature_times(
        config,
        target_series=target_series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
        is_training=is_training,
        return_min_and_max_lags=True,
        check_inputs=check_inputs,
        show_warnings=show_warnings,
    )
    if check_inputs:
        series_and_lags_not_specified = [min_lag is None for min_lag in min_lags]
        if all(series_and_lags_not_specified):
            raise_log(
                ValueError("Must specify at least one series-lags pair."),
            )
        if not (isinstance(stride, int) and stride > 0):
            raise_log(
                ValueError("`stride` must be a positive integer greater than 0."),
            )
    sample_weight_vals = _extract_sample_weight(sample_weight, target_series)
    shared_times = _get_shared_times(*feature_times, sort=True)
    if shared_times is None:
        raise_log(
            ValueError(
                "Specified series do not share any common times for which features can be created."
            ),
        )
    if stride > 1:
        # calculate the starting index so that the last element is included after applying the stride
        # equivalent to shared_times[::-stride][::-1]
        shared_times = shared_times[(len(shared_times) - 1) % stride :: stride]
    if len(shared_times) > max_samples_per_ts:
        shared_times = shared_times[-max_samples_per_ts:]
    X = []
    shared_time_idx = None
    label_shared_time_idx = None
    for i, (series_i, lags_i, min_lag_i) in enumerate(
        zip(
            [target_series, past_covariates, future_covariates],
            [
                None if config.lags_for(kind) is None else config.lags_for(kind).shared
                for kind in ("target", "past", "future")
            ],
            min_lags,
        )
    ):
        series_and_lags_specified = min_lag_i is not None
        is_target_series = is_training and (i == 0)
        if series_and_lags_specified or is_target_series:
            start_i, end_i = series_i.start_time(), series_i.end_time()
            time_index_i = series_i.time_index
            add_to_start = (not is_target_series) and (start_i > shared_times[0])
            add_to_end = (not is_target_series) and (end_i < shared_times[-1])
            if add_to_start or add_to_end:
                new_start = shared_times[0] if add_to_start else None
                new_end = shared_times[-1] if add_to_end else None
                num_prepended = (
                    n_steps_between(
                        start=shared_times[0], end=start_i, freq=series_i.freq
                    )
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
            idx_to_get = (
                label_shared_time_idx
                + np.arange(output_chunk_length)
                + output_chunk_shift
            )
        else:
            # Only point at time `t + output_chunk_length - 1` is a label:
            idx_to_get = (
                label_shared_time_idx + output_chunk_length + output_chunk_shift - 1
            )

        # extract target labels and sample weights
        y_and_weights = []
        for vals in [target_series.all_values(copy=False), sample_weight_vals]:
            if vals is None:
                y_and_weights.append(None)
                continue

            # Before reshaping: lagged_vals.shape = (n_observations, num_lags, n_components, n_samples)
            vals = vals[idx_to_get, :, :]
            # After reshaping: lagged_vals.shape = (n_observations, num_lags*n_components, n_samples)
            vals = vals.reshape(vals.shape[0], -1, vals.shape[-1])
            y_and_weights.append(vals)
        y, weights = y_and_weights
    else:
        y, weights = None, None
    return X, y, shared_times, weights


def _resolve_autoregression_lags(
    config: "ForecastConfig",
) -> tuple[
    dict[str, list[int]],
    dict[str, dict[str, list[int]]],
    dict[str, np.ndarray],
]:
    """Materialize the lag views used by autoregressive prediction.

    The config is constant across the steps of a forecast, so callers that iterate
    over ``t_pred`` can resolve this once and pass it to
    :func:`_create_lagged_data_autoregression`.
    """
    lags: dict[str, list[int]] = {}
    component_lags: dict[str, dict[str, list[int]]] = {}
    for kind in ("target", "past", "future"):
        lags_for_kind = config.lags_for(kind)
        if lags_for_kind is None:
            continue
        if lags_for_kind.is_component_specific:
            lags[kind] = list(lags_for_kind.extreme_lags)
            component_lags[kind] = {
                component: list(component_lags)
                for component, component_lags in lags_for_kind.resolved_per_component.items()
            }
        else:
            lags[kind] = list(lags_for_kind.shared)
    relative_cov_lags = {
        kind: np.array(lags[kind]) - lags[kind][0]
        for kind in ("past", "future")
        if kind in lags
    }
    return lags, component_lags, relative_cov_lags


def _create_lagged_data_autoregression(
    target_series: TimeSeriesLike,
    t_pred: int,
    last_step_shift: int,
    series_matrix: np.ndarray,
    covariate_matrices: dict[str, np.ndarray],
    config: "ForecastConfig",
    uses_static_covariates: bool,
    last_static_covariates_shape: tuple[int, int] | None,
    num_samples: int,
    lag_data: tuple[
        dict[str, list[int]],
        dict[str, dict[str, list[int]]],
        dict[str, np.ndarray],
    ]
    | None = None,
) -> LaggedFeatures:
    """Extract lagged data from target, past covariates and future covariates for auto-regression
    with SKLearnModels.

    The features are returned as a :class:`LaggedFeatures` container so call sites do not depend
    on the concrete feature representation.
    """
    series_length = len(target_series)
    shift = config.strategy.feature_lag_offset
    lags, component_lags, relative_cov_lags = (
        lag_data if lag_data is not None else _resolve_autoregression_lags(config)
    )

    X = []
    for series_type in ["target", "past", "future"]:
        if series_type not in lags:
            continue

        # extract series specific data
        values_matrix = (
            series_matrix
            if series_type == "target"
            else covariate_matrices[series_type]
        )

        if series_type not in component_lags:
            # for global lags over all components, directly extract lagged values from the data
            if series_type == "target":
                relative_lags = [
                    lag - (shift + last_step_shift) for lag in lags[series_type]
                ]
            else:
                relative_lags = relative_cov_lags[series_type] + t_pred

            lagged_data = values_matrix[:, relative_lags].reshape(
                series_length * num_samples, -1
            )
        else:
            # for component-specific lags, sort by lags and components and then extract
            tmp_X = _extract_component_lags_autoregression(
                series_type=series_type,
                values_matrix=values_matrix,
                shift=shift,
                last_step_shift=last_step_shift,
                t_pred=t_pred,
                lags=lags,
                component_lags=component_lags,
            )
            lagged_data = tmp_X.reshape(series_length * num_samples, -1)
        X.append(lagged_data)
    # concatenate retrieved lags
    X = np.concatenate(X, axis=1)

    if uses_static_covariates:
        # Need to split up `X` into three equally-sized sub-blocks
        # corresponding to each timeseries in `series`, so that
        # static covariates can be added to each block; valid since
        # each block contains same number of observations:
        X = np.split(X, series_length, axis=0)
        X, _ = _add_static_covariates_to_lagged_data(
            features=X,
            target_series=target_series,
            uses_static_covariates=uses_static_covariates,
            last_shape=last_static_covariates_shape,
        )

        # concatenate retrieved lags
        X = np.concatenate(X, axis=0)

    n_components = series2seq(target_series)[0].width
    return LaggedFeatures(
        array=X,
        n_horizons=config.strategy.pred_dim,
        output_horizons=config.strategy.output_horizons(n_components),
    )


def _extract_component_lags_autoregression(
    series_type: str,
    values_matrix: np.ndarray,
    shift: int,
    last_step_shift: int,
    t_pred: int,
    lags: dict[str, list[int]],
    component_lags: dict[str, dict[str, list[int]]],
) -> np.ndarray:
    """Extract, concatenate and reorder component-wise lags to obtain a feature order
    identical to tabularization.
    """
    # prepare index to reorder features by lags across components
    comp_lags_reordered = np.concatenate([
        comp_lags for comp_lags in component_lags[series_type].values()
    ]).argsort(**STABLE_SORT_KWARGS)

    # convert relative lags to absolute
    if series_type == "target":
        lags_shift = -shift - last_step_shift
    else:
        lags_shift = -lags[series_type][0] + t_pred

    # extract features
    tmp_X = [
        values_matrix[
            :,
            [lag + lags_shift for lag in comp_lags],
            comp_i,
        ]
        for comp_i, comp_lags in enumerate(component_lags[series_type].values())
    ]

    # concatenate on features dimension and reorder
    return np.concatenate(tmp_X, axis=1)[:, comp_lags_reordered]


# For convenience, define following types for `_get_feature_times`:
FeatureTimes = tuple[
    pd.Index | TimeIndex | None,
    pd.Index | TimeIndex | None,
    pd.Index | TimeIndex | None,
]
MinLags = tuple[int | None, int | None, int | None]
MaxLags = tuple[int | None, int | None, int | None]


def _get_feature_times(
    config: "ForecastConfig",
    target_series: TimeSeries | None = None,
    past_covariates: TimeSeries | None = None,
    future_covariates: TimeSeries | None = None,
    is_training: bool = True,
    return_min_and_max_lags: bool = False,
    check_inputs: bool = True,
    show_warnings: bool = True,
) -> FeatureTimes | tuple[FeatureTimes, MinLags, MaxLags]:
    """
    Returns a tuple containing the times in `target_series`, the times in `past_covariates`, and the times in
    `future_covariates` that *could* be used to create features. The returned tuple of times can then be passed
    to `_get_shared_times` to compute the 'eligible time points' shared by all specified series.

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
        are negative. In this case, `min_lag` and `max_lag` correspond to the smallest and largest
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
        1. The first `max_lag` times of the series are excluded; these times have too few preceding values to
        construct features from.
        2. The last `output_chunk_length - output_chunk_shift - 1` times are excluded; these times have too few
        succeeding times to construct labels from.

    To extract feature times from a `target_series` when `is_training = False`, the following steps are performed:
        1. An additional `min_lag` times are appended to the end of the series; although these times are not contained
        in the original series, we're able to construct features for them since we only need the values of the series
        from time `t - max_lag` to `t - min_lag` to construct a feature for time `t`.
        2. The first `max_lag` times of the series are then excluded; these times have too few preceding values to
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
        3. If `is_training = True`, then `target_series` must be provided.

    Parameters
    ----------
    config
        The resolved forecast config with the lags, output chunk length and output chunk shift.
    target_series
        Optionally, the series for the `SKLearnModel` to predict.
    past_covariates
        Optionally, the past covariates series that the `SKLearnModel` will use as inputs. Unlike the
        `target_series`, `past_covariates` are *not* to be predicted by the `SKLearnModel`.
    future_covariates
        Optionally, the future covariates (i.e. exogenous covariates) series that the `SKLearnModel` will
        use as inputs.
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
    show_warnings
        Whether to show warnings.

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
        a `SKLearnModel` without using autoregressive features.

    """
    if is_training and (target_series is None):
        raise_log(
            ValueError("Must specify `target_series` when `is_training = True`."),
        )
    output_chunk_length = config.output_chunk_length
    output_chunk_shift = config.output_chunk_shift
    feature_times, min_lags, max_lags = [], [], []
    for name_i, series_i, lags_i in zip(
        [_TARGET, _PAST_COV, _FUTURE_COV],
        [target_series, past_covariates, future_covariates],
        [config.lags_for(kind) for kind in ("target", "past", "future")],
    ):
        if check_inputs and (series_i is not None):
            _check_series_length(
                series=series_i,
                lags=lags_i,
                output_chunk_length=output_chunk_length,
                output_chunk_shift=output_chunk_shift,
                is_training=is_training,
                name=name_i,
            )
        series_specified = series_i is not None
        lags_specified = lags_i is not None
        is_label_series = is_training and name_i == _TARGET
        times_i = series_i.time_index if series_specified else None
        lags_extent = lags_i.extreme_lags if lags_specified else None
        max_lag_i = -lags_extent[0] if lags_specified else None
        min_lag_i = -lags_extent[1] if lags_specified else None
        if is_label_series:
            # Exclude last `output_chunk_length - 1` times:
            if not output_chunk_shift:
                end_idx = -output_chunk_length + 1 if output_chunk_length > 1 else None
            else:
                end_idx = -output_chunk_length - output_chunk_shift + 1
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
            new_end = (
                times_i[-1] + series_i.freq * (min_lag_i) if min_lag_i > 0 else None
            )
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
            lags_name = "lags" if name_i == _TARGET else f"lags_{name_i}"
            specified = lags_name if lags_specified else name_i
            unspecified = name_i if lags_specified else lags_name
            if show_warnings:
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


def _get_shared_times(
    *series_or_times: TimeSeries | pd.Index | None, sort: bool = True
) -> pd.Index:
    """
    Returns the times shared by all specified `TimeSeries` or time indexes (i.e. the intersection of all
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
        The time indices present in all specified `TimeSeries` and/or time indices.

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
            series_or_times_1._time_index
            if isinstance(series_or_times_1, TimeSeries)
            else series_or_times_1
        )
        times_2 = (
            series_or_times_2._time_index
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
            if not len(set(times_types)) == 1:
                raise_log(
                    ValueError(
                        "Specified series and/or times must all have the same type of "
                        "`time_index` (i.e. all `pd.RangeIndex` or all `pd.DatetimeIndex`)."
                    ),
                )
    return shared_times


def _get_shared_times_bounds(
    *series_or_times: Sequence[TimeSeries | pd.Index | None],
) -> tuple[pd.Index, pd.Index] | None:
    """
    Returns the latest `start_time` and the earliest `end_time` among all non-`None` `series_or_times`;
    these are (non-tight) lower and upper `bounds` on the intersection of all these `series_or_times` respectively.
    If no potential overlap exists between all specified series, `None` is returned instead.

    Notes
    -----
    If all specified `series_or_times` are of the same frequency, then `_get_shared_times_bounds`
    returns tight `bounds` (i.e. the earliest and latest time within the intersection of all the timeseries
    is returned). To see this, suppose we have three equal-frequency series with observations made at different
    times:

    .. highlight:: md
    .. code-block:: md

        Series 1: ------
        Series 2:    ------
        Series 3:  ------
    ..

    Here, each `-` denotes an observation at a specific time. In this example, `find_time_overlap_bounds` will
    return the times at `LB` and `UB`:

    .. highlight:: md
    .. code-block:: md

                    LB
        Series 1: ---|---|
        Series 2:    |---|---
        Series 3:  --|---|-
                         UB
    ..

    If the specified timeseries are *not* of the same frequency, then the returned `bounds` is potentially non-tight
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
        if not len(set(times_types)) == 1:
            raise_log(
                ValueError(
                    "Specified series and/or times must all have the same type of "
                    "`time_index` (i.e. all `pd.RangeIndex` or all `pd.DatetimeIndex`)."
                ),
            )
        # If `start_times` empty, no series were specified -> `bounds = (1, -1)` will
        # be 'converted' to `None` in next line:
        bounds = (max(start_times), min(end_times)) if start_times else (1, -1)
        # Specified timeseries share no overlapping periods.
        if bounds[1] < bounds[0]:
            bounds = None
    return bounds


def _strided_moving_window(
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

    - `_strided_moving_window` allows for consecutive windows to be separated by a specified `stride`,
      whilst `sliding_window_view` does not.
    - `_strided_moving_window` can only operate along a single axis, whereas `sliding_window_view` can
      operate along multiple axes.

    Additionally, unlike `sliding_window_view`, using `_strided_moving_window` doesn't require `numpy >= 1.20.0`.

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
        if not isinstance(stride, int) or stride < 1:
            raise_log(ValueError("`stride` must be a positive `int`."))
        if not isinstance(window_len, int) or window_len < 1:
            raise_log(
                ValueError("`window_len` must be a positive `int`."),
            )
        if not isinstance(axis, int) or axis > x.ndim - 1 or axis < -x.ndim:
            raise_log(
                ValueError("`axis` must be an `int` that is less than `x.ndim`."),
            )
        if window_len > x.shape[axis]:
            raise_log(
                ValueError("`window_len` must be less than or equal to x.shape[axis]."),
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
    time_index: TimeIndex,
    freq: Frequency,
    new_start: TimeStamp | None = None,
    new_end: TimeStamp | None = None,
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


def _get_freqs(*series: TimeSeries | None):
    """
    Returns list with the frequency of all specified (i.e. non-`None`) `series`.
    """
    freqs = []
    for ts in series:
        if ts is not None:
            freqs.append(ts.freq)
    return freqs


def _all_equal_freq(*series: TimeSeries | None) -> bool:
    """
    Returns `True` if all specified (i.e. non-`None`) `series` have the same frequency.
    """
    freqs = _get_freqs(*series)
    return len(set(freqs)) == 1


def _check_series_length(
    series: TimeSeries,
    lags: "Lags | None",
    output_chunk_length: int,
    output_chunk_shift: int,
    is_training: bool,
    name: Literal["target_series", "past_covariates", "future_covariates"],
) -> None:
    """
    Throws `ValueError` if `series` is too short for specified `lags` and, when `is_training`, `output_chunk_length`.
    """
    is_target = name == _TARGET
    is_label_series = is_training and is_target
    lags_specified = lags is not None
    if not (is_label_series or lags_specified):
        return

    lags_extent = lags.extreme_lags if lags_specified else None
    if is_label_series:
        minimum_len = (
            (-lags_extent[0] if lags_specified else 0)
            + output_chunk_shift
            + output_chunk_length
        )
    else:
        minimum_len = -lags_extent[0] + lags_extent[1] + 1

    if len(series) >= minimum_len:
        return

    if is_label_series:
        minimum_len_str = (
            "-min(lags) + " if lags_specified else ""
        ) + "output_chunk_shift + output_chunk_length"
    else:
        lags_name = "lags" if name == _TARGET else f"lags_{name}"
        minimum_len_str = f"-min({lags_name}) + max({lags_name}) + 1"

    raise_log(
        ValueError(
            f"`{name}` must have at least `{minimum_len_str}` = {minimum_len} time "
            f"steps; instead, it only has {series.n_timesteps}."
        ),
    )


def _extract_sample_weight(sample_weight, target_series):
    """Extracts sample weights values from the time intersection with the target labels."""
    if sample_weight is None:
        return None

    sample_weight_vals = sample_weight.slice_intersect_values(target_series, copy=False)
    if len(sample_weight_vals) != len(target_series):
        raise_log(
            ValueError(
                "The `sample_weight` series must have at least the same times as the target `series`."
            ),
        )

    weight_n_comp = sample_weight_vals.shape[1]
    series_n_comp = target_series.n_components
    if weight_n_comp > 1 and weight_n_comp != series_n_comp:
        raise_log(
            ValueError(
                "The number of components in `sample_weight` must either be `1` or match "
                f"the number of target series components `{series_n_comp}`."
            ),
        )
    elif weight_n_comp != series_n_comp:
        sample_weight_vals = sample_weight_vals.repeat(series_n_comp, axis=1)
    return sample_weight_vals
