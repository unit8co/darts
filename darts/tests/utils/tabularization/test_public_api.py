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
    assert X.n_horizons == 4
    assert X.output_horizons is not None
    assert len(X.output_horizons) == 4
    assert isinstance(y, np.ndarray)
    assert len(times) == 1
    assert last_shape is None
    assert weights is None
    assert config.resolved_per_component("target") == {
        target.components[0]: (-3, -2, -1)
    }


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
    assert X.n_horizons == 3
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
    assert X.n_horizons == 1
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
    assert len(X[0].output_horizons) == 4

    # the config can be re-resolved for the same series
    assert config.resolved_per_component("target") == {
        target.components[0]: (-2, -1),
        target.components[1]: (-3,),
    }


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
    assert np.all(X.array[:, -1, 0] == 5.0)


def test_target_lags_require_target_series():
    config = ForecastConfig(lags=2)

    with pytest.raises(ValueError) as exc:
        create_lagged_prediction_data(config)

    assert str(exc.value) == (
        "`target_series` must be specified when target lags are defined."
    )
