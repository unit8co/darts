import numpy as np
import pandas as pd
import pytest

from darts.utils.data.tabularization import (
    ArrayLaggedFeatures,
    add_static_covariates_to_lagged_data,
    create_lagged_prediction_data,
    create_lagged_training_data,
)
from darts.utils.data.tabularization.features import StepwiseLaggedFeatures
from darts.utils.timeseries_generation import linear_timeseries


def make_stepwise(ndim: int = 2):
    shape = (5, 6) if ndim == 2 else (5, 6, 2)
    array = np.arange(np.prod(shape), dtype=float).reshape(shape)
    step_cols = [1, 4]
    step_values = np.stack([
        array[:, step_cols],
        array[:, step_cols] + 100,
        array[:, step_cols] - 100,
    ])
    return (
        StepwiseLaggedFeatures(
            array=array,
            step_values=step_values,
            step_cols=step_cols,
        ),
        array,
        step_values,
    )


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


class TestStepwiseLaggedFeatures:
    def test_construction_and_properties(self):
        features, array, step_values = make_stepwise()

        assert features.n_horizons == 3
        assert features.shape == array.shape
        assert features.ndim == array.ndim
        assert features.dtype == array.dtype
        assert len(features) == len(array)
        assert features.array is array
        assert np.array_equal(features.step_values[0], array[:, features.step_cols])
        assert np.array_equal(features.step_values, step_values)

    def test_horizon(self):
        features, array, step_values = make_stepwise()

        assert features.horizon(0) is array
        for h in range(1, features.n_horizons):
            horizon = features.horizon(h)
            assert horizon is not array
            assert np.array_equal(horizon[:, features.step_cols], step_values[h])
            assert np.array_equal(horizon[:, [0, 2, 3, 5]], array[:, [0, 2, 3, 5]])

        with pytest.raises(IndexError):
            features.horizon(-1)
        with pytest.raises(IndexError):
            features.horizon(features.n_horizons)

    def test_base_updates_are_visible_in_every_horizon(self):
        features, array, _ = make_stepwise()
        features.horizon(1)  # materialize once

        array[0, 0] = 999.0
        array[1, 2] = -999.0

        assert features.horizon(0)[0, 0] == 999.0
        for h in range(features.n_horizons):
            horizon = features.horizon(h)
            assert horizon[0, 0] == 999.0
            assert horizon[1, 2] == -999.0

    def test_with_format_fn(self):
        features, array, step_values = make_stepwise()
        formatted = features.with_format_fn(pd.DataFrame)

        assert formatted is not features
        assert formatted.array is array
        assert isinstance(formatted.horizon(1), pd.DataFrame)
        assert np.array_equal(
            formatted.horizon(1).to_numpy()[:, formatted.step_cols],
            step_values[1],
        )
        assert features.format_fn is None
        assert isinstance(features.horizon(1), np.ndarray)

        cleared = formatted.with_format_fn(None)
        assert cleared.format_fn is None
        assert isinstance(cleared.horizon(1), np.ndarray)

    def test_with_base(self):
        features, array, step_values = make_stepwise()
        static = np.arange(10).reshape(5, 2)
        new_array = np.hstack([array, static])

        with_base = features.with_base(new_array)

        assert with_base is not features
        assert with_base.array is new_array
        assert with_base.shape == new_array.shape
        assert np.array_equal(with_base.step_values, step_values)
        assert np.array_equal(
            with_base.horizon(1)[:, with_base.step_cols], step_values[1]
        )
        assert np.array_equal(with_base.horizon(1)[:, -2:], static)

    def test_repeat(self):
        features, array, step_values = make_stepwise()
        repeated = features.repeat(2, axis=0)

        assert isinstance(repeated, StepwiseLaggedFeatures)
        assert repeated.shape == (10, 6)
        assert repeated.step_values.shape == (3, 10, 2)
        assert np.array_equal(repeated.array, np.repeat(array, 2, axis=0))
        assert np.array_equal(repeated.step_values, np.repeat(step_values, 2, axis=1))
        assert np.array_equal(
            repeated.step_values[0],
            repeated.array[:, repeated.step_cols],
        )

        with pytest.raises(ValueError):
            features.repeat(2, axis=1)

    def test_concatenate(self):
        features, array, step_values = make_stepwise()
        concatenated = StepwiseLaggedFeatures.concatenate([features, features])

        assert isinstance(concatenated, StepwiseLaggedFeatures)
        assert concatenated.shape == (10, 6)
        assert concatenated.step_values.shape == (3, 10, 2)
        assert np.array_equal(concatenated.array, np.concatenate([array, array]))
        assert np.array_equal(
            concatenated.step_values,
            np.concatenate([step_values, step_values], axis=1),
        )
        assert np.array_equal(
            concatenated.step_values[0],
            concatenated.array[:, concatenated.step_cols],
        )

    def test_concatenate_validation(self):
        with pytest.raises(ValueError):
            StepwiseLaggedFeatures.concatenate([])

        first, _, _ = make_stepwise()
        other_array = np.zeros((5, 6))
        other_step_values = np.zeros((3, 5, 2))
        other = StepwiseLaggedFeatures(
            array=other_array,
            step_values=other_step_values,
            step_cols=[0, 1],
        )

        with pytest.raises(ValueError):
            StepwiseLaggedFeatures.concatenate([first, other])

    def test_getitem_observations(self):
        features, array, step_values = make_stepwise()

        sliced = features[1:4]
        assert isinstance(sliced, StepwiseLaggedFeatures)
        assert sliced.n_horizons == 3
        assert np.array_equal(sliced.array, array[1:4])
        assert np.array_equal(sliced.step_values, step_values[:, 1:4])
        assert np.array_equal(sliced.step_values[0], sliced.array[:, sliced.step_cols])

        with pytest.raises(IndexError):
            features[0]
        with pytest.raises(IndexError):
            features[:, [0, 1]]
        with pytest.raises(IndexError):
            features[0, 0, 0]

    def test_getitem_samples(self):
        features, array, step_values = make_stepwise(ndim=3)

        sliced = features[1:4, :, 0]
        assert isinstance(sliced, StepwiseLaggedFeatures)
        assert np.array_equal(sliced.array, array[1:4, :, 0])
        assert np.array_equal(sliced.step_values, step_values[:, 1:4, :, 0])
        assert np.array_equal(sliced.step_values[0], sliced.array[:, sliced.step_cols])

    def test_repr(self):
        features, _, _ = make_stepwise()

        assert repr(features) == (
            "StepwiseLaggedFeatures(shape=(5, 6), n_horizons=3, step_cols=[1, 4])"
        )

    @pytest.mark.parametrize(
        "array_shape,step_values_shape,step_cols",
        [
            ((5, 6), (3, 5), [1, 4]),
            ((5, 6), (3, 4, 2), [1, 4]),
            ((5, 6), (3, 5, 1), [1, 4]),
            ((5, 6, 2), (3, 5, 2, 3), [1, 4]),
        ],
    )
    def test_invalid_layout(self, array_shape, step_values_shape, step_cols):
        with pytest.raises(ValueError):
            StepwiseLaggedFeatures(
                array=np.zeros(array_shape),
                step_values=np.zeros(step_values_shape),
                step_cols=step_cols,
            )


class TestTabularizationContainers:
    @staticmethod
    def target(length: int = 30):
        return linear_timeseries(length=length)

    def test_training_data_returns_container(self):
        target = self.target()
        features, labels, _, _, _ = create_lagged_training_data(
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
        features, _, _, _, _ = create_lagged_training_data(
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
        features, _, _, _, _ = create_lagged_training_data(
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
        features, _ = create_lagged_prediction_data(
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

        features, _, _, static_shape, _ = create_lagged_training_data(
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
        features, _, _, _, _ = create_lagged_training_data(
            target_series=target,
            output_chunk_length=3,
            output_chunk_shift=0,
            lags=[-2, -1],
            uses_static_covariates=False,
        )
        static = pd.DataFrame({"sc": [5.0]}, index=target.columns)
        target = target.with_static_covariates(static)

        features_with_static, static_shape = add_static_covariates_to_lagged_data(
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

    def test_add_static_covariates_preserves_stepwise_layout(self):
        features, array, step_values = make_stepwise()
        target = self.target(length=len(array))
        static = pd.DataFrame({"sc": [5.0]}, index=target.columns)
        target = target.with_static_covariates(static)

        features_with_static, static_shape = add_static_covariates_to_lagged_data(
            features=features,
            target_series=target,
            uses_static_covariates=True,
        )

        assert isinstance(features_with_static, StepwiseLaggedFeatures)
        assert features_with_static.shape == (len(array), array.shape[1] + 1)
        assert features_with_static.n_horizons == features.n_horizons
        assert np.array_equal(features_with_static.step_values, step_values)
        assert np.array_equal(features_with_static.step_cols, features.step_cols)
        assert np.array_equal(features_with_static.array[:, :-1], array)
        assert np.array_equal(
            features_with_static.horizon(1)[:, -1],
            np.full(len(array), 5.0),
        )

    def test_add_static_covariates_keeps_plain_arrays(self):
        target = self.target()
        features = np.arange(12.0).reshape(6, 2)
        static = pd.DataFrame({"sc": [5.0]}, index=target.columns)
        target = target.with_static_covariates(static)

        features_with_static, _ = add_static_covariates_to_lagged_data(
            features=features,
            target_series=target,
            uses_static_covariates=True,
        )

        assert type(features_with_static) is np.ndarray
        assert features_with_static.shape == (6, 3)

    def test_disabled_static_covariates_returns_input(self):
        features = ArrayLaggedFeatures(np.zeros((4, 2)), n_horizons=2)
        target = self.target()

        returned, _ = add_static_covariates_to_lagged_data(
            features=features,
            target_series=target,
            uses_static_covariates=False,
        )

        assert returned is features
