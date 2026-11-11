import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.utils.data.tabularization import (
    ForecastConfig,
    create_lagged_component_names,
    create_lagged_training_data,
)
from darts.utils.data.tabularization.tabularization import (
    _create_lagged_data_autoregression,
)


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


@pytest.mark.parametrize("multi_models,shift", [(True, 0), (False, 1)])
def test_autoregression_returns_the_feature_array(multi_models, shift):
    n = 12
    values = np.arange(n, dtype=float).reshape(n, 1)
    target = TimeSeries.from_values(values)
    config = ForecastConfig(lags=3, output_chunk_length=2, multi_models=multi_models)
    resolved = config.resolve(target_components=target.components)

    actual = _create_lagged_data_autoregression(
        target_series=[target],
        t_pred=0,
        last_step_shift=0,
        series_matrix=values.reshape(1, n, 1),
        covariate_matrices={},
        config=resolved,
        uses_static_covariates=False,
        last_static_covariates_shape=None,
        num_samples=1,
    )

    expected = np.array([[values[lag - shift, 0] for lag in [-3, -2, -1]]])
    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, expected)


def test_autoregression_component_specific_lags():
    n = 12
    values = np.stack([np.arange(n), np.arange(n) * 10], axis=1)
    target = TimeSeries.from_values(values, columns=["a", "b"])
    config = ForecastConfig(lags={"a": [-3, -1], "b": [-2]}, output_chunk_length=2)
    resolved = config.resolve(target_components=target.components)

    actual = _create_lagged_data_autoregression(
        target_series=[target],
        t_pred=0,
        last_step_shift=0,
        series_matrix=values.reshape(1, n, 2),
        covariate_matrices={},
        config=resolved,
        uses_static_covariates=False,
        last_static_covariates_shape=None,
        num_samples=1,
    )

    # target columns are reordered lag-major, then component: a[-3], b[-2], a[-1]
    expected = np.array([
        [
            values[-3, 0],
            values[-2, 1],
            values[-1, 0],
        ]
    ])
    np.testing.assert_array_equal(actual, expected)


def test_autoregression_adds_static_covariates():
    target, _, _ = make_series(static=True)
    config = ForecastConfig(lags=3, output_chunk_length=2)
    resolved = config.resolve(target_components=target.components)

    actual = _create_lagged_data_autoregression(
        target_series=[target],
        t_pred=0,
        last_step_shift=0,
        series_matrix=target.values(copy=False).reshape(1, len(target), -1),
        covariate_matrices={},
        config=resolved,
        uses_static_covariates=True,
        last_static_covariates_shape=None,
        num_samples=1,
    )

    assert actual.shape == (1, 2 * 3 + 2)
    np.testing.assert_array_equal(actual[:, -2], np.zeros(1))
    np.testing.assert_array_equal(actual[:, -1], np.ones(1))


def test_from_config_expands_default_lags():
    target, past, future = make_series()
    config = ForecastConfig(
        lags={"a": [-3, -1], "default_lags": [-2]},
        lags_past_covariates={"p_a": [-2], "default_lags": [-1]},
        lags_future_covariates={
            "f_a": [0],
            "f_b": [-1, 1],
            "default_lags": [2],
        },
        output_chunk_length=2,
    )

    X, y, times, _, _ = create_lagged_training_data(
        config,
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        uses_static_covariates=False,
        concatenate=False,
    )

    # `default_lags` fill the components without explicit lags (`b`, `p_b` and `f_c`)
    feature_names, _ = create_lagged_component_names(
        config,
        target_series=target,
        past_covariates=past,
        future_covariates=future,
        concatenate=False,
    )
    assert feature_names == [
        "a_target_lag-3",
        "b_target_lag-2",
        "a_target_lag-1",
        "p_a_pastcov_lag-2",
        "p_b_pastcov_lag-1",
        "f_b_futcov_lag-1",
        "f_a_futcov_lag0",
        "f_b_futcov_lag1",
        "f_c_futcov_lag2",
    ]
    assert isinstance(X, list)
    assert X[0].shape[1] == 9
    assert len(y) == 1
