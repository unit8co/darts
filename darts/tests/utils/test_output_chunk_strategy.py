import numpy as np
import pytest

from darts.utils.output_chunk import (
    DirectOutputStrategy,
    RecursiveOutputStrategy,
)


class TestDirectOutputStrategy:
    def test_properties(self):
        strategy = DirectOutputStrategy(output_chunk_length=4)

        assert strategy.name == "direct"
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

    def test_output_horizons(self):
        strategy = DirectOutputStrategy(output_chunk_length=4)

        assert np.array_equal(
            strategy.output_horizons(3),
            np.repeat(np.arange(4), 3),
        )

    @pytest.mark.parametrize(
        "native_multioutput,sample_weight,expected",
        [
            (True, False, False),
            (False, False, True),
            (True, True, True),
        ],
    )
    def test_requires_wrapper(self, native_multioutput, sample_weight, expected):
        strategy = DirectOutputStrategy(output_chunk_length=4)

        assert (
            strategy.requires_wrapper(
                native_multioutput=native_multioutput,
                sample_weight=sample_weight,
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


class TestRecursiveOutputStrategy:
    def test_properties(self):
        strategy = RecursiveOutputStrategy(output_chunk_length=4)

        assert strategy.name == "recursive"
        assert strategy.pred_dim == 1
        assert strategy.feature_lag_offset == 3
        assert strategy.n_outputs(n_components=3) == 3

    def test_estimator_index(self):
        strategy = RecursiveOutputStrategy(output_chunk_length=4)

        for horizon in range(4):
            for component in range(3):
                assert strategy.estimator_index(horizon, component, 3) == component

    def test_output_horizons(self):
        strategy = RecursiveOutputStrategy(output_chunk_length=4)

        assert np.array_equal(strategy.output_horizons(3), np.zeros(3, dtype=int))

    @pytest.mark.parametrize(
        "native_multioutput,sample_weight,expected",
        [
            (True, False, False),
            (False, False, True),
            (True, True, True),
        ],
    )
    def test_requires_wrapper(self, native_multioutput, sample_weight, expected):
        strategy = RecursiveOutputStrategy(output_chunk_length=4)

        assert (
            strategy.requires_wrapper(
                native_multioutput=native_multioutput,
                sample_weight=sample_weight,
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


@pytest.mark.parametrize(
    "strategy_cls", [DirectOutputStrategy, RecursiveOutputStrategy]
)
@pytest.mark.parametrize("bad_value", [1.5, True, "1"])
def test_invalid_index_types(strategy_cls, bad_value):
    strategy = strategy_cls(output_chunk_length=4)

    with pytest.raises(ValueError):
        strategy.n_outputs(bad_value)
    with pytest.raises(ValueError):
        strategy.estimator_index(bad_value, 0, 3)
    with pytest.raises(ValueError):
        strategy.estimator_index(0, bad_value, 3)
