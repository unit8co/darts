import pytest

from darts.utils.forecast_config import ForecastConfig
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
    config = ForecastConfig(
        lags=2,
        output_chunk_length=output_chunk_length,
        multi_models=multi_models,
    )

    assert isinstance(config.strategy, expected)


def test_resolve_happens_in_place():
    config = ForecastConfig(
        lags={"a": [-2, -1], "b": [-3]},
        lags_future_covariates={"f": [0, 1], "default_lags": [2]},
        output_chunk_length=4,
        output_chunk_shift=2,
        multi_models=False,
    )

    resolved = config.resolve(["a", "b"], future_components=["f", "g"])

    assert resolved is config
    assert isinstance(config.strategy, RecursiveOutputStrategy)
    assert config.output_chunk_length == 4
    assert config.output_chunk_shift == 2
    assert config.resolved_per_component("target") == {
        "a": (-2, -1),
        "b": (-3,),
    }
    assert config.resolved_per_component("future") == {
        "f": (2, 3),
        "g": (4,),
    }
    assert config.is_component_specific("target")
    assert config.is_component_specific("future")
    assert not config.is_component_specific("past")
    assert config.extreme_lags("target") == (-3, -1)
    assert config.extreme_lags("future") == (2, 4)


def test_output_chunk_shift_is_applied_exactly_once():
    config = ForecastConfig(
        lags=2,
        lags_future_covariates=[0, 1],
        output_chunk_shift=2,
    )

    assert config.lags_for("future").shared == (2, 3)

    config.resolve(["target"], future_components=["f1", "f2"])
    assert config.lags_for("future").shared == (2, 3)
    assert config.resolved_per_component("future") == {
        "f1": (2, 3),
        "f2": (2, 3),
    }


@pytest.mark.parametrize(
    "multi_models,expected_strategy,expected_pred_dim,expected_offset",
    [
        (True, DirectOutputStrategy, 4, 0),
        (False, RecursiveOutputStrategy, 1, 3),
    ],
)
def test_derived_properties(
    multi_models, expected_strategy, expected_pred_dim, expected_offset
):
    config = ForecastConfig(
        lags=2,
        output_chunk_length=4,
        multi_models=multi_models,
    )

    assert isinstance(config.strategy, expected_strategy)
    assert config.pred_dim == expected_pred_dim
    assert config.feature_lag_offset == expected_offset
    assert config.multi_models is multi_models


def test_output_chunk_properties_delegate_to_strategy():
    config = ForecastConfig(
        lags=2,
        output_chunk_length=6,
        output_chunk_shift=3,
    )

    assert config.output_chunk_length == 6
    assert config.output_chunk_shift == 3


def test_has_lags():
    config = ForecastConfig(
        lags=2,
        lags_past_covariates=2,
    )

    assert config.has_lags("target")
    assert config.has_lags("past")
    assert not config.has_lags("future")


def test_at_least_one_lags_argument():
    with pytest.raises(ValueError) as exc:
        ForecastConfig()

    assert str(exc.value) == (
        "At least one of `lags`, `lags_future_covariates` or "
        "`lags_past_covariates` must be not None."
    )


def test_unknown_lags_kind():
    config = ForecastConfig(lags=2)

    with pytest.raises(ValueError) as exc:
        config.lags_for("unknown")

    assert str(exc.value) == "Unknown lags kind: 'unknown'."


class TestResolve:
    def test_shared_lags(self):
        config = ForecastConfig(lags=2)
        config.resolve(["a", "b"])

        assert config.resolved_per_component("target") == {
            "a": (-2, -1),
            "b": (-2, -1),
        }
        assert config.resolved_per_component("past") is None
        assert config.resolved_per_component("future") is None
        assert config.is_component_specific("target") is False
        assert config.extreme_lags("target") == (-2, -1)

    def test_component_lags_are_ordered_like_the_series(self):
        config = ForecastConfig(lags={"b": [-1], "a": [-3, -2]})
        config.resolve(["a", "b"])

        target_lags = config.resolved_per_component("target")
        assert list(target_lags) == ["a", "b"]
        assert target_lags == {"a": (-3, -2), "b": (-1,)}
        assert config.extreme_lags("target") == (-3, -1)

    def test_default_lags_fill_the_missing_components(self):
        config = ForecastConfig(lags={"a": [-1], "default_lags": [-4, -3]})
        config.resolve(["a", "b", "c"])

        assert config.resolved_per_component("target") == {
            "a": (-1,),
            "b": (-4, -3),
            "c": (-4, -3),
        }
        assert config.extreme_lags("target") == (-4, -1)

    def test_extreme_lags_use_the_resolved_components(self):
        config = ForecastConfig(lags={"a": [-1], "b": [-2], "default_lags": [-5]})
        target_lags = config.lags_for("target")

        # Before resolution the legacy extremes still include the unused default.
        assert target_lags.extreme_lags == (-5, -1)

        config.resolve(["a", "b"])

        assert config.resolved_per_component("target") == {
            "a": (-1,),
            "b": (-2,),
        }
        assert config.extreme_lags("target") == (-2, -1)

    def test_unknown_component(self):
        config = ForecastConfig(lags={"unknown": [-1]})

        with pytest.raises(ValueError) as exc:
            config.resolve(["a"])

        assert str(exc.value) == (
            "The `lags` dictionary specifies lags for components that are not present in "
            "the series : ['unknown']. They must be removed to avoid any ambiguity."
        )

    def test_missing_component_without_default(self):
        config = ForecastConfig(lags={"a": [-1]})

        with pytest.raises(ValueError) as exc:
            config.resolve(["a", "b"])

        assert str(exc.value) == (
            "The lags dictionary is missing the lags for the following components present "
            "in the series: ['b']. The key 'default_lags' can be used to provide lags for "
            "all the non-explicitely defined components."
        )

    def test_errors_for_all_variates_are_aggregated(self):
        config = ForecastConfig(
            lags={"unknown": [-1]},
            lags_past_covariates={"p1": [-1]},
        )

        with pytest.raises(ValueError) as exc:
            config.resolve(["target"], ["p1", "p2"])

        message = str(exc.value)
        assert "\n" in message
        assert "`lags` dictionary specifies lags for components" in message
        assert "lags_past_covariates dictionary is missing the lags" in message

    def test_output_chunk_shift_only_applies_to_future_covariates(self):
        config = ForecastConfig(
            lags=2,
            lags_past_covariates=[-2, -1],
            lags_future_covariates={"f1": [0], "f2": [-1], "default_lags": [1]},
            output_chunk_shift=2,
        )
        config.resolve(["target"], ["past0", "past1"], ["f1", "f2", "f3"])

        assert config.resolved_per_component("target") == {
            "target": (-2, -1),
        }
        assert config.resolved_per_component("past") == {
            "past0": (-2, -1),
            "past1": (-2, -1),
        }
        assert config.resolved_per_component("future") == {
            "f1": (2,),
            "f2": (1,),
            "f3": (3,),
        }
        assert config.extreme_lags("target") == (-2, -1)
        assert config.extreme_lags("past") == (-2, -1)
        assert config.extreme_lags("future") == (1, 3)

    def test_output_chunk_shift_with_shared_future_lags(self):
        config = ForecastConfig(
            lags=2,
            lags_future_covariates=[0, 1],
            output_chunk_shift=2,
        )
        config.resolve(["target"], future_components=["f1", "f2"])

        assert config.resolved_per_component("future") == {
            "f1": (2, 3),
            "f2": (2, 3),
        }
        assert config.extreme_lags("future") == (2, 3)

    def test_resolve_is_idempotent_and_does_not_mutate_user_inputs(self):
        value = {"a": [-1], "default_lags": [-2]}
        config = ForecastConfig(lags=value)

        config.resolve(["a", "b"])
        first = config.resolved_per_component("target")
        config.resolve(["a", "b"])
        second = config.resolved_per_component("target")

        assert first == second == {"a": (-1,), "b": (-2,)}
        assert config.lags_for("target").component_lags == {"a": (-1,)}
        assert config.lags_for("target").default == (-2,)
        assert value == {"a": [-1], "default_lags": [-2]}

    def test_resolved_per_component_requires_resolution(self):
        config = ForecastConfig(lags=2)

        with pytest.raises(ValueError) as exc:
            config.resolved_per_component("target")

        assert "`Lags` have not been resolved yet" in str(exc.value)


class TestForecastConfigConstruction:
    def test_builds_all_variates(self):
        config = ForecastConfig(
            lags={"a": [-1]},
            lags_past_covariates=2,
            lags_future_covariates=(1, 1),
            output_chunk_length=6,
            output_chunk_shift=2,
        )

        assert config.target.kind == "target"
        assert config.target.is_component_specific
        assert config.past.shared == (-2, -1)
        assert not config.past.is_component_specific
        assert config.future.kind == "future"
        assert config.future.shared == (1, 2)
        assert config.output_chunk_length == 6
        assert config.output_chunk_shift == 2
