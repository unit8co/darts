import pytest

from darts.utils.data.tabularization.output_chunk import (
    DirectOutputStrategy,
    RecursiveOutputStrategy,
)


class TestDirectOutputStrategy:
    def test_properties(self):
        strategy = DirectOutputStrategy(output_chunk_length=4)

        assert strategy.name == "direct"
        assert strategy.pred_dim == 4
        assert strategy.feature_lag_offset == 0

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


class TestRecursiveOutputStrategy:
    def test_properties(self):
        strategy = RecursiveOutputStrategy(output_chunk_length=4)

        assert strategy.name == "recursive"
        assert strategy.pred_dim == 1
        assert strategy.feature_lag_offset == 3

    def test_estimator_index(self):
        strategy = RecursiveOutputStrategy(output_chunk_length=4)

        for horizon in range(4):
            for component in range(3):
                assert strategy.estimator_index(horizon, component, 3) == component

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


@pytest.mark.parametrize(
    "strategy_cls", [DirectOutputStrategy, RecursiveOutputStrategy]
)
@pytest.mark.parametrize("bad_value", [1.5, True, "1"])
def test_invalid_index_types(strategy_cls, bad_value):
    strategy = strategy_cls(output_chunk_length=4)

    with pytest.raises(ValueError):
        strategy.estimator_index(bad_value, 0, 3)
    with pytest.raises(ValueError):
        strategy.estimator_index(0, bad_value, 3)
    with pytest.raises(ValueError):
        strategy.estimator_index(0, 0, bad_value)
