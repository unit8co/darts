import numpy as np
import pandas as pd
import pytest

from darts.utils.data.tabularization import (
    ForecastConfig,
    LaggedFeatures,
    create_lagged_component_names,
    create_lagged_data,
    create_lagged_prediction_data,
    create_lagged_training_data,
)
from darts.utils.timeseries_generation import linear_timeseries


def make_series(length: int = 30):
    target = linear_timeseries(length=length)
    past = linear_timeseries(start_value=1, length=length, column_name="past")
    future = linear_timeseries(start_value=2, length=length, column_name="future")
    return target, past, future


def test_create_lagged_data_returns_lagged_features():
    target, _, _ = make_series()
    config = ForecastConfig(lags=3, output_chunk_length=4)

    X, y, times, last_shape, weights = create_lagged_data(
        config,
        target_series=target,
        uses_static_covariates=False,
    )

    assert isinstance(X, LaggedFeatures)
    assert isinstance(X.to_numpy(), np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(times) == 1
    assert last_shape is None
    assert weights is None
    np.testing.assert_array_equal(X.to_numpy()[0, :, 0], target.values()[:3, 0])


def test_create_lagged_training_data_wrapper():
    target, past, future = make_series()
    config = ForecastConfig(
        lags=2,
        lags_past_covariates=2,
        lags_future_covariates=(1, 1),
        output_chunk_length=3,
    )

    X, y, times, _, _ = create_lagged_training_data(
        config,
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        uses_static_covariates=False,
    )

    assert isinstance(X, LaggedFeatures)
    assert X.shape[0] == len(y) == len(times[0])


def test_create_lagged_prediction_data_wrapper():
    target, _, _ = make_series()
    config = ForecastConfig(lags=2)

    X, times = create_lagged_prediction_data(
        config,
        target_series=target,
        uses_static_covariates=False,
    )

    assert isinstance(X, LaggedFeatures)
    assert len(times) == 1


def test_create_lagged_component_names_wrapper():
    target, past, future = make_series()
    config = ForecastConfig(
        lags=2,
        lags_past_covariates=2,
        lags_future_covariates=(1, 1),
        output_chunk_length=3,
    )

    feature_names, label_names = create_lagged_component_names(
        config,
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        concatenate=False,
        use_static_covariates=False,
    )

    assert len(feature_names) == 2 + 2 + 2
    assert label_names == [f"{target.components[0]}_target_hrz{i}" for i in range(3)]


def test_component_specific_and_sequence_outputs():
    target = linear_timeseries(length=20).stack(
        linear_timeseries(length=20, start_value=10)
    )
    config = ForecastConfig(
        lags={
            target.components[0]: [-2, -1],
            target.components[1]: [-3],
        },
        output_chunk_length=2,
    )

    X, _, _, _, _ = create_lagged_data(
        config,
        target_series=target,
        uses_static_covariates=False,
        concatenate=False,
    )

    assert isinstance(X, list)
    assert len(X) == 1
    assert isinstance(X[0], LaggedFeatures)
    assert X[0].shape[1] == 3

    # the helper resolves a copy; the caller's config stays unresolved
    with pytest.raises(ValueError, match="have not been resolved yet"):
        config.resolved_per_component("target")


@pytest.mark.parametrize(
    "lags",
    [
        {"a": [-2], "b": [-1], "c": [-1]},
        {"c": [-1], "b": [-1], "a": [-2]},
        {"default_lags": [-1], "a": [-2]},
        {"a": [-2], "default_lags": [-1]},
    ],
)
def test_component_lags_are_matched_by_name(lags):
    # the lags are matched to the components by name, not by their position in the dictionary,
    # and the `default_lags` apply to every component without explicit lags
    target = (
        linear_timeseries(length=10, column_name="a")
        .stack(linear_timeseries(length=10, start_value=10, column_name="b"))
        .stack(linear_timeseries(length=10, start_value=20, column_name="c"))
    )

    X, _ = create_lagged_prediction_data(
        ForecastConfig(lags=lags), target_series=target, uses_static_covariates=False
    )

    # one row per time from the 3rd step until one step after the end of the series; the
    # features are ordered by lag, then by component: a(t-2), b(t-1), c(t-1)
    values = target.values()
    expected = np.stack([values[:-1, 0], values[1:, 1], values[1:, 2]], axis=1)
    np.testing.assert_array_equal(X.to_numpy()[:, :, 0], expected)


def _call_public_helper(helper, config, target):
    kwargs = {"target_series": target}
    if helper is not create_lagged_component_names:
        kwargs["uses_static_covariates"] = False
    return helper(config, **kwargs)


@pytest.mark.parametrize(
    "helper",
    [
        create_lagged_data,
        create_lagged_training_data,
        create_lagged_prediction_data,
        create_lagged_component_names,
    ],
)
def test_public_helpers_do_not_modify_the_config(helper):
    target = linear_timeseries(length=20, column_name="a").stack(
        linear_timeseries(length=20, start_value=10, column_name="b")
    )
    config = ForecastConfig(lags={"a": [-2, -1], "b": [-3]}, output_chunk_length=2)

    # an unresolved config stays unresolved
    _call_public_helper(helper, config, target)
    with pytest.raises(ValueError, match="have not been resolved yet"):
        config.resolved_per_component("target")

    # a resolved config (e.g. the one of a fitted model) keeps its resolution, also with
    # series whose components are in another order
    config.resolve(target.components)
    expected = list(config.resolved_per_component("target").items())
    reordered = target[["b", "a"]]
    _call_public_helper(helper, config, reordered)
    assert list(config.resolved_per_component("target").items()) == expected

    # and a failed resolution leaves it untouched as well
    with pytest.raises(ValueError, match="not present in the series"):
        _call_public_helper(helper, config, target["a"])
    assert list(config.resolved_per_component("target").items()) == expected


def test_static_covariates_preserve_container():
    target, _, _ = make_series()
    target = target.with_static_covariates(
        pd.DataFrame({"sc": [5.0]}, index=target.columns)
    )
    config = ForecastConfig(lags=2, output_chunk_length=2)

    X, _, _, last_shape, _ = create_lagged_data(
        config,
        target_series=target,
        uses_static_covariates=True,
    )

    assert isinstance(X, LaggedFeatures)
    assert last_shape == (1, 1)
    assert np.all(X.to_numpy()[:, -1, 0] == 5.0)


def test_target_lags_require_target_series():
    config = ForecastConfig(lags=2)

    with pytest.raises(ValueError) as exc:
        create_lagged_prediction_data(config)

    assert str(exc.value) == (
        "`target_series` must be specified when target lags are defined."
    )
