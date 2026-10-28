import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.utils.data.tabularization import (
    create_lagged_data,
    create_lagged_prediction_data,
    create_lagged_training_data,
)
from darts.utils.data.tabularization.data import LaggedData
from darts.utils.data.tabularization.tabularization import (
    _create_lagged_data_from_config,
)
from darts.utils.forecast_config import ForecastConfig


def make_series(static: bool = False):
    n = 48
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    target = TimeSeries.from_times_and_values(
        index,
        np.stack([np.arange(n), np.arange(n) * 2], axis=1),
        columns=["a", "b"],
    )
    if static:
        target = target.with_static_covariates(
            pd.DataFrame({"state": [0, 1]}, index=["a", "b"])
        )
    past = TimeSeries.from_times_and_values(
        index,
        np.stack([np.arange(n) + 1, np.arange(n) + 2], axis=1),
        columns=["p_a", "p_b"],
    )
    future = TimeSeries.from_times_and_values(
        index,
        np.stack([np.arange(n), np.arange(n) + 3, np.arange(n) + 5], axis=1),
        columns=["f_a", "f_b", "f_c"],
    )
    return target, past, future


def legacy_lags(config: ForecastConfig, kind: str):
    component_lags = config.lags.legacy_component_lags_dict()
    if kind in component_lags:
        return component_lags[kind]
    return config.lags.legacy_lags_dict().get(kind)


def assert_same_data(actual: LaggedData, expected: tuple, series_length: int):
    expected_X, expected_y, expected_times, expected_static_shape, expected_weights = (
        expected
    )

    if isinstance(actual.X, list):
        assert isinstance(expected_X, list)
        assert len(actual.X) == len(expected_X)
        for left, right in zip(actual.X, expected_X):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    else:
        np.testing.assert_array_equal(np.asarray(actual.X), np.asarray(expected_X))

    if expected_y is None:
        assert actual.y is None
    elif isinstance(actual.y, list):
        assert len(actual.y) == len(expected_y)
        for left, right in zip(actual.y, expected_y):
            np.testing.assert_array_equal(left, right)
    else:
        np.testing.assert_array_equal(actual.y, expected_y)

    assert len(actual.times) == len(expected_times)
    for left, right in zip(actual.times, expected_times):
        assert left.equals(right)

    assert actual.last_static_covariates_shape == expected_static_shape
    assert actual.series_lengths == [series_length]

    if expected_weights is None:
        assert actual.sample_weights is None
    elif isinstance(actual.sample_weights, list):
        assert len(actual.sample_weights) == len(expected_weights)
        for left, right in zip(actual.sample_weights, expected_weights):
            np.testing.assert_array_equal(left, right)
    else:
        np.testing.assert_array_equal(actual.sample_weights, expected_weights)


def config_for(
    component_specific: bool, output_chunk_shift: int, multi_models: bool
) -> ForecastConfig:
    if component_specific:
        lags = {"a": [-3, -1], "b": [-2]}
        past_lags = {"p_a": [-2], "p_b": [-1]}
        future_lags = {"f_a": [0], "f_b": [-1, 1], "f_c": [2]}
    else:
        lags = 3
        past_lags = 2
        future_lags = (2, 3)

    return ForecastConfig.from_user(
        lags=lags,
        lags_past_covariates=past_lags,
        lags_future_covariates=future_lags,
        output_chunk_length=3,
        output_chunk_shift=output_chunk_shift,
        multi_models=multi_models,
    )


@pytest.mark.parametrize("multi_models", [True, False])
@pytest.mark.parametrize("component_specific", [False, True])
@pytest.mark.parametrize("output_chunk_shift", [0, 2])
@pytest.mark.parametrize("concatenate", [True, False])
@pytest.mark.parametrize("with_static", [False, True])
def test_training_data_from_config_matches_public_wrapper(
    multi_models, component_specific, output_chunk_shift, concatenate, with_static
):
    target, past, future = make_series(static=with_static)
    config = config_for(component_specific, output_chunk_shift, multi_models)
    resolved = config.resolve(
        target_components=target.components,
        past_components=past.components,
        future_components=future.components,
    )

    expected = create_lagged_training_data(
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        lags=legacy_lags(config, "target"),
        lags_past_covariates=legacy_lags(config, "past"),
        lags_future_covariates=legacy_lags(config, "future"),
        output_chunk_length=config.strategy.output_chunk_length,
        output_chunk_shift=config.output_chunk_shift,
        uses_static_covariates=with_static,
        multi_models=multi_models,
        concatenate=concatenate,
        sample_weight=target,
        stride=2,
        max_samples_per_ts=10,
    )
    actual = _create_lagged_data_from_config(
        resolved,
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        uses_static_covariates=with_static,
        concatenate=concatenate,
        sample_weight=target,
        stride=2,
        max_samples_per_ts=10,
    )

    assert isinstance(actual, LaggedData)
    assert_same_data(actual, expected, len(target))


@pytest.mark.parametrize("component_specific", [False, True])
@pytest.mark.parametrize("concatenate", [True, False])
def test_prediction_data_from_config_matches_public_wrapper(
    component_specific, concatenate
):
    target, past, future = make_series()
    config = config_for(component_specific, 0, multi_models=True)
    resolved = config.resolve(
        target_components=target.components,
        past_components=past.components,
        future_components=future.components,
    )

    expected_X, expected_times = create_lagged_prediction_data(
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        lags=legacy_lags(config, "target"),
        lags_past_covariates=legacy_lags(config, "past"),
        lags_future_covariates=legacy_lags(config, "future"),
        uses_static_covariates=False,
        concatenate=concatenate,
    )
    actual = _create_lagged_data_from_config(
        resolved,
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        uses_static_covariates=False,
        is_training=False,
        concatenate=concatenate,
    )

    assert actual.y is None
    if concatenate:
        np.testing.assert_array_equal(np.asarray(actual.X), np.asarray(expected_X))
    else:
        assert len(actual.X) == len(expected_X)
        for left, right in zip(actual.X, expected_X):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    assert len(actual.times) == len(expected_times)
    for left, right in zip(actual.times, expected_times):
        assert left.equals(right)
    assert actual.series_lengths == [len(target)]
    assert actual.last_static_covariates_shape is None
    assert actual.sample_weights is None


def test_prediction_data_from_config_uses_output_chunk_shift():
    target, past, future = make_series()
    config = config_for(
        component_specific=False, output_chunk_shift=2, multi_models=True
    )
    resolved = config.resolve(
        target_components=target.components,
        past_components=past.components,
        future_components=future.components,
    )

    expected_X, _, expected_times, _, _ = create_lagged_data(
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        lags=legacy_lags(config, "target"),
        lags_past_covariates=legacy_lags(config, "past"),
        lags_future_covariates=legacy_lags(config, "future"),
        output_chunk_length=config.strategy.output_chunk_length,
        output_chunk_shift=config.output_chunk_shift,
        uses_static_covariates=False,
        multi_models=True,
        check_inputs=False,
        is_training=False,
        concatenate=True,
    )
    actual = _create_lagged_data_from_config(
        resolved,
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        uses_static_covariates=False,
        is_training=False,
        concatenate=True,
    )

    np.testing.assert_array_equal(np.asarray(actual.X), np.asarray(expected_X))
    assert len(actual.times) == len(expected_times)
    for left, right in zip(actual.times, expected_times):
        assert left.equals(right)


def test_training_data_from_config_uses_resolved_output_chunk_shift():
    target, past, future = make_series()
    config = config_for(
        component_specific=False, output_chunk_shift=2, multi_models=True
    )
    resolved = config.resolve(
        target_components=target.components,
        past_components=past.components,
        future_components=future.components,
    )

    expected = create_lagged_training_data(
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        lags=legacy_lags(config, "target"),
        lags_past_covariates=legacy_lags(config, "past"),
        lags_future_covariates=legacy_lags(config, "future"),
        output_chunk_length=config.strategy.output_chunk_length,
        output_chunk_shift=config.output_chunk_shift,
        uses_static_covariates=False,
        multi_models=True,
        concatenate=False,
    )
    actual = _create_lagged_data_from_config(
        resolved,
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        uses_static_covariates=False,
        concatenate=False,
    )

    assert_same_data(actual, expected, len(target))


def test_training_data_from_config_matches_intersection_path():
    target, past, future = make_series()
    config = config_for(
        component_specific=False, output_chunk_shift=0, multi_models=True
    )
    resolved = config.resolve(
        target_components=target.components,
        past_components=past.components,
        future_components=future.components,
    )

    expected = create_lagged_training_data(
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        lags=legacy_lags(config, "target"),
        lags_past_covariates=legacy_lags(config, "past"),
        lags_future_covariates=legacy_lags(config, "future"),
        output_chunk_length=config.strategy.output_chunk_length,
        output_chunk_shift=config.output_chunk_shift,
        uses_static_covariates=False,
        multi_models=True,
        use_moving_windows=False,
        concatenate=False,
    )
    actual = _create_lagged_data_from_config(
        resolved,
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        uses_static_covariates=False,
        use_moving_windows=False,
        concatenate=False,
    )

    assert_same_data(actual, expected, len(target))


def test_from_config_expands_default_lags():
    target, past, future = make_series()
    config = ForecastConfig.from_user(
        lags={"a": [-3, -1], "default_lags": [-2]},
        lags_past_covariates={"p_a": [-2], "default_lags": [-1]},
        lags_future_covariates={
            "f_a": [0],
            "f_b": [-1, 1],
            "default_lags": [2],
        },
        output_chunk_length=2,
    )
    resolved = config.resolve(
        target_components=target.components,
        past_components=past.components,
        future_components=future.components,
    )

    expected = create_lagged_training_data(
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        lags={"a": [-3, -1], "b": [-2]},
        lags_past_covariates={"p_a": [-2], "p_b": [-1]},
        lags_future_covariates={"f_a": [0], "f_b": [-1, 1], "f_c": [2]},
        output_chunk_length=2,
        output_chunk_shift=0,
        uses_static_covariates=False,
        multi_models=True,
        concatenate=False,
    )
    actual = _create_lagged_data_from_config(
        resolved,
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        uses_static_covariates=False,
        concatenate=False,
    )

    assert_same_data(actual, expected, len(target))


def test_lagged_data_to_tuple_matches_positional_api():
    X = np.zeros((2, 3))
    times = [pd.RangeIndex(2)]
    data = LaggedData(X=X, y=None, times=times)

    assert data.to_tuple() == (X, None, times, None, None)
