import pytest

from darts.utils.forecast_config import ForecastConfig, ResolvedForecastConfig
from darts.utils.output_chunk import (
    DirectOutputStrategy,
    RecursiveOutputStrategy,
)


@pytest.mark.parametrize(
    "multi_models,output_chunk_length,expected",
    [
        (True, 3, DirectOutputStrategy),
        (False, 3, RecursiveOutputStrategy),
        (False, 1, DirectOutputStrategy),
        (None, 3, RecursiveOutputStrategy),
        (None, 1, DirectOutputStrategy),
    ],
)
def test_strategy_selection(multi_models, output_chunk_length, expected):
    config = ForecastConfig.from_user(
        lags=2,
        output_chunk_length=output_chunk_length,
        multi_models=multi_models,
    )

    assert isinstance(config.strategy, expected)


def test_resolve_uses_the_same_strategy_and_resolves_lags():
    config = ForecastConfig.from_user(
        lags={"a": [-2, -1], "b": [-3]},
        lags_future_covariates={"f": [0, 1], "default_lags": [2]},
        output_chunk_length=4,
        output_chunk_shift=2,
        multi_models=False,
    )

    resolved = config.resolve(["a", "b"], future_components=["f", "g"])

    assert isinstance(resolved, ResolvedForecastConfig)
    assert resolved.strategy is config.strategy
    assert resolved.lags.legacy_lags_dict() == {
        "target": [-3, -1],
        "future": [2, 4],
    }
    assert resolved.lags.legacy_component_lags_dict() == {
        "target": {"a": [-2, -1], "b": [-3]},
        "future": {"f": [2, 3], "g": [4]},
    }


def test_output_chunk_properties_delegate_to_lags():
    config = ForecastConfig.from_user(
        lags=2,
        output_chunk_length=6,
        output_chunk_shift=3,
    )

    assert config.output_chunk_length == 6
    assert config.output_chunk_shift == 3
