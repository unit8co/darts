import pytest

from darts.utils.output_chunk import (
    DirectOutputStrategy,
    DirRecOutputStrategy,
    RecursiveOutputStrategy,
)


class TestDirectOutputStrategy:
    def test_properties(self):
        strategy = DirectOutputStrategy(output_chunk_length=4)

        assert strategy.name == "direct"
        assert strategy.n_horizons == 4
        assert strategy.pred_dim == 4
        assert strategy.feature_lag_offset == 0
        assert strategy.n_outputs(n_components=3) == 12

    def test_estimator_index(self):
        strategy = DirectOutputStrategy(output_chunk_length=4)
        expected = {
            (0, 0): 0,
            (0, 2): 2,
            (1, 0): 3,
            (3, 2): 11,
        }

        for (horizon, component), index in expected.items():
            assert strategy.estimator_index(horizon, component, 3) == index

    def test_horizon_and_component_of(self):
        strategy = DirectOutputStrategy(output_chunk_length=4)

        for output_idx in range(12):
            horizon = strategy.horizon_of(output_idx, 3)
            component = strategy.component_of(output_idx, 3)
            assert horizon == output_idx // 3
            assert component == output_idx % 3
            assert strategy.estimator_index(horizon, component, 3) == output_idx

    def test_orders(self):
        strategy = DirectOutputStrategy(output_chunk_length=4)

        assert strategy.fit_order(12, 3) == [list(range(12))]
        assert strategy.predict_order(12, 3) == [list(range(12))]
        assert strategy.chain_columns(0, 3) == ()

    @pytest.mark.parametrize(
        "native_multioutput,sample_weight,per_horizon_features,expected",
        [
            (True, False, False, False),
            (False, False, False, True),
            (True, True, False, True),
            (True, False, True, True),
        ],
    )
    def test_requires_wrapper(
        self,
        native_multioutput,
        sample_weight,
        per_horizon_features,
        expected,
    ):
        strategy = DirectOutputStrategy(output_chunk_length=4)

        assert (
            strategy.requires_wrapper(
                native_multioutput=native_multioutput,
                sample_weight=sample_weight,
                per_horizon_features=per_horizon_features,
            )
            is expected
        )

    @pytest.mark.parametrize("output_chunk_length", [0, -1, 1.5, True])
    def test_invalid_output_chunk_length(self, output_chunk_length):
        with pytest.raises(ValueError):
            DirectOutputStrategy(output_chunk_length)

    def test_invalid_indices(self):
        strategy = DirectOutputStrategy(output_chunk_length=4)

        with pytest.raises(ValueError):
            strategy.estimator_index(-1, 0, 3)
        with pytest.raises(ValueError):
            strategy.estimator_index(4, 0, 3)
        with pytest.raises(ValueError):
            strategy.estimator_index(0, -1, 3)
        with pytest.raises(ValueError):
            strategy.estimator_index(0, 3, 3)
        with pytest.raises(ValueError):
            strategy.horizon_of(12, 3)
        with pytest.raises(ValueError):
            strategy.component_of(-1, 3)
        with pytest.raises(ValueError):
            strategy.fit_order(12, 0)
        with pytest.raises(ValueError):
            strategy.fit_order(11, 3)


class TestRecursiveOutputStrategy:
    def test_properties(self):
        strategy = RecursiveOutputStrategy(output_chunk_length=4)

        assert strategy.name == "recursive"
        assert strategy.n_horizons == 1
        assert strategy.pred_dim == 1
        assert strategy.feature_lag_offset == 3
        assert strategy.n_outputs(n_components=3) == 3

    def test_estimator_index(self):
        strategy = RecursiveOutputStrategy(output_chunk_length=4)

        for horizon in range(4):
            for component in range(3):
                assert strategy.estimator_index(horizon, component, 3) == component

    def test_horizon_and_component_of(self):
        strategy = RecursiveOutputStrategy(output_chunk_length=4)

        for output_idx in range(3):
            assert strategy.horizon_of(output_idx, 3) == 0
            assert strategy.component_of(output_idx, 3) == output_idx

    def test_orders(self):
        strategy = RecursiveOutputStrategy(output_chunk_length=4)

        assert strategy.fit_order(3, 3) == [[0, 1, 2]]
        assert strategy.predict_order(3, 3) == [[0, 1, 2]]
        assert strategy.chain_columns(0, 3) == ()

    @pytest.mark.parametrize(
        "native_multioutput,sample_weight,per_horizon_features,expected",
        [
            (True, False, False, False),
            (False, False, False, True),
            (True, True, False, True),
            (True, False, True, False),
        ],
    )
    def test_requires_wrapper(
        self,
        native_multioutput,
        sample_weight,
        per_horizon_features,
        expected,
    ):
        strategy = RecursiveOutputStrategy(output_chunk_length=4)

        assert (
            strategy.requires_wrapper(
                native_multioutput=native_multioutput,
                sample_weight=sample_weight,
                per_horizon_features=per_horizon_features,
            )
            is expected
        )

    @pytest.mark.parametrize("output_chunk_length", [0, -1, 1.5, True])
    def test_invalid_output_chunk_length(self, output_chunk_length):
        with pytest.raises(ValueError):
            RecursiveOutputStrategy(output_chunk_length)

    def test_invalid_indices(self):
        strategy = RecursiveOutputStrategy(output_chunk_length=4)

        with pytest.raises(ValueError):
            strategy.estimator_index(-1, 0, 3)
        with pytest.raises(ValueError):
            strategy.estimator_index(4, 0, 3)
        with pytest.raises(ValueError):
            strategy.estimator_index(0, 3, 3)
        with pytest.raises(ValueError):
            strategy.horizon_of(3, 3)
        with pytest.raises(ValueError):
            strategy.component_of(-1, 3)
        with pytest.raises(ValueError):
            strategy.fit_order(4, 3)


class TestDirRecOutputStrategy:
    def test_reserved_attributes(self):
        strategy = DirRecOutputStrategy(output_chunk_length=4)

        assert strategy.name == "dirrec"
        assert strategy.n_horizons == 4
        assert strategy.pred_dim == 4
        assert strategy.feature_lag_offset == 0
        assert strategy.requires_wrapper(
            native_multioutput=True,
            sample_weight=False,
            per_horizon_features=False,
        )

    @pytest.mark.parametrize(
        "call",
        [
            lambda strategy: strategy.estimator_index(0, 0, 1),
            lambda strategy: strategy.horizon_of(0, 1),
            lambda strategy: strategy.component_of(0, 1),
            lambda strategy: strategy.chain_columns(0, 1),
            lambda strategy: strategy.fit_order(4, 1),
            lambda strategy: strategy.predict_order(4, 1),
        ],
    )
    def test_reserved_methods_are_not_implemented(self, call):
        strategy = DirRecOutputStrategy(output_chunk_length=4)

        with pytest.raises(NotImplementedError):
            call(strategy)
