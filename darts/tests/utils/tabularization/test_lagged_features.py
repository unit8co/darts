import numpy as np
import pandas as pd
import pytest

from darts.utils.data.tabularization.features import ArrayLaggedFeatures
from darts.utils.data.tabularization.tabularization import (
    _add_static_covariates_to_lagged_data,
    _create_lagged_prediction_data,
    _create_lagged_training_data,
)
from darts.utils.timeseries_generation import linear_timeseries


class TestArrayLaggedFeatures:
    def test_construction_is_zero_copy(self):
        array = np.arange(12).reshape(4, 3)
        features = ArrayLaggedFeatures(array, n_horizons=3)

        assert isinstance(features, np.ndarray)
        assert features.n_horizons == 3
        assert features.shape == array.shape
        assert features.ndim == array.ndim
        assert features.dtype == array.dtype
        assert len(features) == len(array)
        assert features.format_fn is None
        assert np.shares_memory(features, array)
        assert isinstance(features.array, np.ndarray)
        assert np.shares_memory(features.array, array)

    @pytest.mark.parametrize("n_horizons", [0, -1, 1.5, True])
    def test_invalid_n_horizons(self, n_horizons):
        with pytest.raises(ValueError):
            ArrayLaggedFeatures(np.zeros((2, 2)), n_horizons=n_horizons)

    def test_wrap(self):
        array = np.arange(12).reshape(4, 3)
        features = ArrayLaggedFeatures.wrap(array, n_horizons=2)

        assert isinstance(features, ArrayLaggedFeatures)
        assert np.shares_memory(features, array)
        assert ArrayLaggedFeatures.wrap(features, n_horizons=2) is features

        rewrapped = ArrayLaggedFeatures.wrap(features, n_horizons=3)
        assert rewrapped is not features
        assert rewrapped.n_horizons == 3
        assert np.shares_memory(rewrapped, array)

    def test_horizon_identity(self):
        features = ArrayLaggedFeatures(np.zeros((3, 4)), n_horizons=3)

        for h in range(features.n_horizons):
            assert features.horizon(h) is features

        with pytest.raises(IndexError):
            features.horizon(-1)
        with pytest.raises(IndexError):
            features.horizon(features.n_horizons)

    def test_slicing_preserves_container(self):
        array = np.arange(24).reshape(4, 6)
        features = ArrayLaggedFeatures(array, n_horizons=2)

        sliced = features[1:3]
        assert isinstance(sliced, ArrayLaggedFeatures)
        assert sliced.n_horizons == 2
        assert np.shares_memory(sliced, array)

        column_slice = features[:, [0, 2]]
        assert isinstance(column_slice, ArrayLaggedFeatures)
        assert column_slice.n_horizons == 2
        assert np.array_equal(column_slice, array[:, [0, 2]])

    def test_with_base(self):
        features = ArrayLaggedFeatures(np.zeros((4, 3)), n_horizons=3)
        base = np.ones((2, 3))

        with_base = features.with_base(base)

        assert isinstance(with_base, ArrayLaggedFeatures)
        assert with_base is not features
        assert with_base.n_horizons == 3
        assert np.shares_memory(with_base, base)
        assert with_base.horizon(0) is with_base

    def test_with_format_fn(self):
        array = np.arange(12).reshape(4, 3)
        features = ArrayLaggedFeatures(array, n_horizons=2)
        formatted = features.with_format_fn(pd.DataFrame)

        assert isinstance(formatted, ArrayLaggedFeatures)
        assert formatted is not features
        assert np.shares_memory(formatted, array)
        assert isinstance(formatted.horizon(0), pd.DataFrame)
        assert isinstance(formatted.horizon(1), pd.DataFrame)
        # the original container is unchanged
        assert features.horizon(0) is features

        cleared = formatted.with_format_fn(None)
        assert cleared.format_fn is None
        assert cleared.horizon(0) is cleared

    def test_repeat_preserves_container_and_metadata(self):
        array = np.arange(12).reshape(4, 3)
        features = ArrayLaggedFeatures(array, n_horizons=2)

        repeated = features.repeat(2, axis=0)

        assert isinstance(repeated, ArrayLaggedFeatures)
        assert repeated.n_horizons == 2
        assert repeated.shape == (8, 3)
        assert np.array_equal(repeated, np.repeat(array, 2, axis=0))

    def test_concatenate_preserves_container_and_metadata(self):
        array = np.arange(12).reshape(4, 3)
        features = ArrayLaggedFeatures(array, n_horizons=2)

        concatenated = ArrayLaggedFeatures.concatenate([features, features])

        assert isinstance(concatenated, ArrayLaggedFeatures)
        assert concatenated.n_horizons == 2
        assert concatenated.shape == (8, 3)
        assert np.array_equal(concatenated, np.concatenate([array, array], axis=0))

    def test_concatenate_validation(self):
        with pytest.raises(ValueError):
            ArrayLaggedFeatures.concatenate([])

        first = ArrayLaggedFeatures(np.zeros((2, 3)), n_horizons=2)
        second = ArrayLaggedFeatures(np.zeros((2, 3)), n_horizons=3)
        with pytest.raises(ValueError):
            ArrayLaggedFeatures.concatenate([first, second])

        third = first.with_format_fn(pd.DataFrame)
        with pytest.raises(ValueError):
            ArrayLaggedFeatures.concatenate([first, third])


class TestTabularizationContainers:
    @staticmethod
    def target(length: int = 30):
        return linear_timeseries(length=length)

    def test_training_data_returns_container(self):
        target = self.target()
        features, labels, _, _, _ = _create_lagged_training_data(
            target_series=target,
            output_chunk_length=4,
            output_chunk_shift=0,
            lags=[-2, -1],
            uses_static_covariates=False,
        )

        assert isinstance(features, ArrayLaggedFeatures)
        assert features.n_horizons == 4
        assert features.horizon(0) is features
        assert features.horizon(3) is features
        assert np.shares_memory(features, features.array)
        assert not isinstance(labels, ArrayLaggedFeatures)
        assert isinstance(labels, np.ndarray)

    def test_recursive_training_data_has_one_horizon(self):
        target = self.target()
        features, _, _, _, _ = _create_lagged_training_data(
            target_series=target,
            output_chunk_length=4,
            output_chunk_shift=0,
            lags=[-2, -1],
            multi_models=False,
            uses_static_covariates=False,
        )

        assert isinstance(features, ArrayLaggedFeatures)
        assert features.n_horizons == 1

    def test_sequence_without_concatenation(self):
        target = self.target()
        features, _, _, _, _ = _create_lagged_training_data(
            target_series=[target, target],
            output_chunk_length=3,
            output_chunk_shift=0,
            lags=[-2, -1],
            concatenate=False,
            uses_static_covariates=False,
        )

        assert isinstance(features, list)
        assert len(features) == 2
        assert all(isinstance(item, ArrayLaggedFeatures) for item in features)
        assert all(item.n_horizons == 3 for item in features)

    def test_prediction_data_returns_container(self):
        target = self.target()
        features, _ = _create_lagged_prediction_data(
            target_series=target,
            lags=[-2, -1],
            uses_static_covariates=False,
        )

        assert isinstance(features, ArrayLaggedFeatures)
        assert features.n_horizons == 1
        assert features.horizon(0) is features

    def test_training_data_with_static_covariates(self):
        target = self.target()
        static = pd.DataFrame({"sc1": [1.0], "sc2": [2.0]}, index=target.columns)
        target = target.with_static_covariates(static)

        features, _, _, static_shape, _ = _create_lagged_training_data(
            target_series=target,
            output_chunk_length=2,
            output_chunk_shift=0,
            lags=[-2, -1],
            uses_static_covariates=True,
        )

        assert isinstance(features, ArrayLaggedFeatures)
        assert features.n_horizons == 2
        assert static_shape == (1, 2)
        assert np.array_equal(
            features.array[:, -2:, 0],
            np.tile([1.0, 2.0], (len(features), 1)),
        )

    def test_add_static_covariates_preserves_container(self):
        target = self.target()
        features, _, _, _, _ = _create_lagged_training_data(
            target_series=target,
            output_chunk_length=3,
            output_chunk_shift=0,
            lags=[-2, -1],
            uses_static_covariates=False,
        )
        static = pd.DataFrame({"sc": [5.0]}, index=target.columns)
        target = target.with_static_covariates(static)

        features_with_static, static_shape = _add_static_covariates_to_lagged_data(
            features=features,
            target_series=target,
            uses_static_covariates=True,
        )

        assert isinstance(features_with_static, ArrayLaggedFeatures)
        assert features_with_static.n_horizons == 3
        assert static_shape == (1, 1)
        assert np.array_equal(
            features_with_static.array[:, -1, 0],
            np.full(len(features_with_static), 5.0),
        )
        assert np.array_equal(features_with_static.array[:, :-1], features.array)

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
        features = ArrayLaggedFeatures(np.zeros((4, 2)), n_horizons=2)
        target = self.target()

        returned, _ = _add_static_covariates_to_lagged_data(
            features=features,
            target_series=target,
            uses_static_covariates=False,
        )

        assert returned is features
