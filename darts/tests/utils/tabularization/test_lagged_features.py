import numpy as np
import pandas as pd
import pytest

from darts.utils.data.tabularization import (
    ForecastConfig,
    LaggedFeatures,
    create_lagged_prediction_data,
    create_lagged_training_data,
)
from darts.utils.data.tabularization.tabularization import (
    _add_static_covariates_to_lagged_data,
)
from darts.utils.timeseries_generation import linear_timeseries


class TestLaggedFeatures:
    def test_construction_and_properties(self):
        array = np.arange(12).reshape(4, 3)
        features = LaggedFeatures(array)

        assert features.shape == array.shape
        assert features.ndim == array.ndim
        assert features.dtype == array.dtype
        assert len(features) == len(array)
        assert features.to_numpy() is features.array
        assert np.shares_memory(features.to_numpy(), array)

    def test_invalid_array_ndim(self):
        with pytest.raises(ValueError):
            LaggedFeatures(np.zeros(3))


class TestTabularizationContainers:
    @staticmethod
    def target(length: int = 30):
        return linear_timeseries(length=length)

    @pytest.mark.parametrize("multi_models", [True, False])
    def test_training_data_returns_container(self, multi_models):
        target = self.target()
        features, labels, _, _, _ = create_lagged_training_data(
            ForecastConfig(
                lags=[-2, -1], output_chunk_length=4, multi_models=multi_models
            ),
            target_series=target,
            uses_static_covariates=False,
        )

        assert isinstance(features, LaggedFeatures)
        assert type(features.to_numpy()) is np.ndarray
        assert features.shape[:2] == (len(labels), 2)
        assert isinstance(labels, np.ndarray)

    def test_sequence_without_concatenation(self):
        target = self.target()
        features, _, _, _, _ = create_lagged_training_data(
            ForecastConfig(lags=[-2, -1], output_chunk_length=3),
            target_series=[target, target],
            concatenate=False,
            uses_static_covariates=False,
        )

        assert isinstance(features, list)
        assert len(features) == 2
        assert all(isinstance(item, LaggedFeatures) for item in features)

    def test_prediction_data_returns_container(self):
        target = self.target()
        features, _ = create_lagged_prediction_data(
            ForecastConfig(lags=[-2, -1]),
            target_series=target,
            uses_static_covariates=False,
        )

        assert isinstance(features, LaggedFeatures)
        assert type(features.to_numpy()) is np.ndarray

    def test_training_data_with_static_covariates(self):
        target = self.target()
        static = pd.DataFrame({"sc1": [1.0], "sc2": [2.0]}, index=target.columns)
        target = target.with_static_covariates(static)

        features, _, _, static_shape, _ = create_lagged_training_data(
            ForecastConfig(lags=[-2, -1], output_chunk_length=2),
            target_series=target,
            uses_static_covariates=True,
        )

        assert isinstance(features, LaggedFeatures)
        assert static_shape == (1, 2)
        assert np.array_equal(
            features.to_numpy()[:, -2:, 0],
            np.tile([1.0, 2.0], (len(features), 1)),
        )

    def test_add_static_covariates_preserves_container(self):
        target = self.target()
        features, _, _, _, _ = create_lagged_training_data(
            ForecastConfig(lags=[-2, -1], output_chunk_length=3),
            target_series=target,
            uses_static_covariates=False,
        )
        static = pd.DataFrame({"sc": [5.0]}, index=target.columns)
        target = target.with_static_covariates(static)

        features_with_static, static_shape = _add_static_covariates_to_lagged_data(
            features=features,
            target_series=target,
            uses_static_covariates=True,
        )

        assert isinstance(features_with_static, LaggedFeatures)
        assert static_shape == (1, 1)
        assert np.array_equal(
            features_with_static.to_numpy()[:, -1, 0],
            np.full(len(features_with_static), 5.0),
        )
        assert np.array_equal(
            features_with_static.to_numpy()[:, :-1], features.to_numpy()
        )

    def test_add_static_covariates_keeps_plain_arrays(self):
        target = self.target()
        features = np.arange(12.0).reshape(6, 2)
        static = pd.DataFrame({"sc": [5.0]}, index=target.columns)
        target = target.with_static_covariates(static)

        features_with_static, _ = _add_static_covariates_to_lagged_data(
            features=features,
            target_series=target,
            uses_static_covariates=True,
        )

        assert type(features_with_static) is np.ndarray
        assert features_with_static.shape == (6, 3)

    def test_disabled_static_covariates_returns_input(self):
        features = LaggedFeatures(np.zeros((4, 2)))
        target = self.target()

        returned, _ = _add_static_covariates_to_lagged_data(
            features=features,
            target_series=target,
            uses_static_covariates=False,
        )

        assert returned is features
