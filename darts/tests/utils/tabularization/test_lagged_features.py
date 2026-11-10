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
        features = LaggedFeatures(array, n_horizons=3)

        assert features.n_horizons == 3
        assert features.output_horizons is None
        assert features.shape == array.shape
        assert features.ndim == array.ndim
        assert features.dtype == array.dtype
        assert len(features) == len(array)
        assert features.to_numpy() is features.array
        assert np.shares_memory(features.to_numpy(), array)

    @pytest.mark.parametrize("n_horizons", [0, -1, 1.5, True])
    def test_invalid_n_horizons(self, n_horizons):
        with pytest.raises(ValueError):
            LaggedFeatures(np.zeros((2, 2)), n_horizons=n_horizons)

    def test_invalid_array_ndim(self):
        with pytest.raises(ValueError):
            LaggedFeatures(np.zeros(3))

    @pytest.mark.parametrize(
        "output_horizons",
        [
            np.array([[0, 1]]),
            np.array([0.0, 1.0]),
            np.array([0, 3]),
            np.array([-1, 0]),
        ],
    )
    def test_invalid_output_horizons(self, output_horizons):
        with pytest.raises(ValueError):
            LaggedFeatures(
                np.zeros((4, 2)),
                n_horizons=2,
                output_horizons=output_horizons,
            )

    def test_for_horizon_returns_the_base_array(self):
        array = np.arange(12).reshape(4, 3)
        features = LaggedFeatures(array, n_horizons=3)

        for h in range(features.n_horizons):
            assert features.for_horizon(h) is features.array

        with pytest.raises(IndexError):
            features.for_horizon(-1)
        with pytest.raises(IndexError):
            features.for_horizon(features.n_horizons)

    def test_for_output(self):
        array = np.arange(12).reshape(4, 3)
        features = LaggedFeatures(
            array,
            n_horizons=2,
            output_horizons=np.array([1, 0]),
        )

        assert features.for_output(0) is features.array
        assert features.for_output(1) is features.array

        with pytest.raises(IndexError):
            features.for_output(2)

        without_mapping = LaggedFeatures(array, n_horizons=2)
        assert without_mapping.for_output(10) is without_mapping.array

    def test_with_base_preserves_metadata(self):
        features = LaggedFeatures(
            np.zeros((4, 3)),
            n_horizons=3,
            output_horizons=np.array([2, 1, 0]),
        )
        base = np.ones((2, 3))

        with_base = features.with_base(base)

        assert isinstance(with_base, LaggedFeatures)
        assert with_base is not features
        assert with_base.n_horizons == 3
        assert np.array_equal(with_base.output_horizons, [2, 1, 0])
        assert np.shares_memory(with_base.array, base)

    def test_repeat_preserves_metadata(self):
        array = np.arange(12).reshape(4, 3)
        features = LaggedFeatures(
            array,
            n_horizons=2,
            output_horizons=np.array([0, 1]),
        )

        repeated = features.repeat(2, axis=0)

        assert isinstance(repeated, LaggedFeatures)
        assert repeated.n_horizons == 2
        assert np.array_equal(repeated.output_horizons, [0, 1])
        assert repeated.shape == (8, 3)
        assert np.array_equal(repeated.to_numpy(), np.repeat(array, 2, axis=0))

        with pytest.raises(ValueError):
            features.repeat(2, axis=1)

    def test_select_samples(self):
        array = np.arange(24).reshape(4, 3, 2)
        features = LaggedFeatures(
            array,
            n_horizons=2,
            output_horizons=np.array([0, 1]),
        )

        selected = features.select_samples(1)

        assert isinstance(selected, LaggedFeatures)
        assert selected.shape == (4, 3)
        assert selected.n_horizons == 2
        assert np.array_equal(selected.output_horizons, [0, 1])
        assert np.array_equal(selected.to_numpy(), array[..., 1])

        with pytest.raises(ValueError):
            LaggedFeatures(np.zeros((4, 3))).select_samples(0)

    def test_concatenate_preserves_metadata(self):
        array = np.arange(12).reshape(4, 3)
        features = LaggedFeatures(
            array,
            n_horizons=2,
            output_horizons=np.array([0, 1]),
        )

        concatenated = LaggedFeatures.concatenate([features, features])

        assert isinstance(concatenated, LaggedFeatures)
        assert concatenated.n_horizons == 2
        assert np.array_equal(concatenated.output_horizons, [0, 1])
        assert concatenated.shape == (8, 3)
        assert np.array_equal(
            concatenated.to_numpy(), np.concatenate([array, array], axis=0)
        )

    def test_concatenate_validation(self):
        with pytest.raises(ValueError):
            LaggedFeatures.concatenate([])

        first = LaggedFeatures(np.zeros((2, 3)), n_horizons=2)
        second = LaggedFeatures(np.zeros((2, 3)), n_horizons=3)
        with pytest.raises(ValueError):
            LaggedFeatures.concatenate([first, second])

        third = LaggedFeatures(
            np.zeros((2, 3)),
            n_horizons=2,
            output_horizons=np.array([1, 0]),
        )
        with pytest.raises(ValueError):
            LaggedFeatures.concatenate([first, third])


class TestTabularizationContainers:
    @staticmethod
    def target(length: int = 30):
        return linear_timeseries(length=length)

    def test_training_data_returns_container(self):
        target = self.target()
        features, labels, _, _, _ = create_lagged_training_data(
            ForecastConfig(lags=[-2, -1], output_chunk_length=4),
            target_series=target,
            uses_static_covariates=False,
        )

        assert isinstance(features, LaggedFeatures)
        assert features.n_horizons == 4
        assert features.for_horizon(0) is features.array
        assert features.for_horizon(3) is features.array
        assert not isinstance(labels, LaggedFeatures)
        assert isinstance(labels, np.ndarray)

    def test_recursive_training_data_has_one_horizon(self):
        target = self.target()
        features, _, _, _, _ = create_lagged_training_data(
            ForecastConfig(lags=[-2, -1], output_chunk_length=4, multi_models=False),
            target_series=target,
            uses_static_covariates=False,
        )

        assert isinstance(features, LaggedFeatures)
        assert features.n_horizons == 1

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
        assert all(item.n_horizons == 3 for item in features)

    def test_prediction_data_returns_container(self):
        target = self.target()
        features, _ = create_lagged_prediction_data(
            ForecastConfig(lags=[-2, -1]),
            target_series=target,
            uses_static_covariates=False,
        )

        assert isinstance(features, LaggedFeatures)
        assert features.n_horizons == 1
        assert features.for_horizon(0) is features.array

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
        assert features.n_horizons == 2
        assert static_shape == (1, 2)
        assert np.array_equal(
            features.array[:, -2:, 0],
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
        features = LaggedFeatures(np.zeros((4, 2)), n_horizons=2)
        target = self.target()

        returned, _ = _add_static_covariates_to_lagged_data(
            features=features,
            target_series=target,
            uses_static_covariates=False,
        )

        assert returned is features
