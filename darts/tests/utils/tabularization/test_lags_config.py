import pytest

from darts.utils.data.tabularization.lags import (
    ComponentLags,
    FutureLags,
    LagConfig,
    Lags,
    SharedLags,
)


class TestLagsFromUser:
    @pytest.mark.parametrize(
        "value,kind,expected",
        [
            (3, "target", (-3, -2, -1)),
            ([-1, -3, -2], "target", (-3, -2, -1)),
            (2, "past", (-2, -1)),
            ([-4, -1], "past", (-4, -1)),
            ((3, 5), "future", tuple(range(-3, 5))),
            ([-1, 0, 3], "future", (-1, 0, 3)),
            ({"default_lags": 2}, "target", (-2, -1)),
            ({"default_lags": [-1, 0]}, "future", (-1, 0)),
        ],
    )
    def test_shared_lags(self, value, kind, expected):
        lags = Lags.from_user(value, kind=kind)

        assert isinstance(lags, SharedLags)
        assert lags.lags == expected
        assert lags.legacy_lags == expected

    @pytest.mark.parametrize(
        "value,kind,per_component,default",
        [
            (
                {"a": [-4, -2], "b": [-5, -3]},
                "target",
                {"a": (-4, -2), "b": (-5, -3)},
                None,
            ),
            (
                {"a": 2, "b": 1},
                "past",
                {"a": (-2, -1), "b": (-1,)},
                None,
            ),
            (
                {"a": (1, 2), "b": [-1, 0, 2]},
                "future",
                {"a": (-1, 0, 1), "b": (-1, 0, 2)},
                None,
            ),
            (
                {"a": [-1], "default_lags": [-3, -2]},
                "target",
                {"a": (-1,)},
                (-3, -2),
            ),
            (
                {"a": [0], "default_lags": [-1]},
                "future",
                {"a": (0,)},
                (-1,),
            ),
        ],
    )
    def test_component_lags(self, value, kind, per_component, default):
        lags = Lags.from_user(value, kind=kind)

        assert isinstance(lags, ComponentLags)
        assert lags.per_component == per_component
        assert lags.default == default

    def test_input_dictionary_is_copied(self):
        value = {"a": [-2, -1]}
        lags = Lags.from_user(value, kind="target")

        value["a"].append(-3)
        value["b"] = [-4]

        assert lags.per_component == {"a": (-2, -1)}

    @pytest.mark.parametrize(
        "value,kind,message",
        [
            (
                "bad",
                "target",
                "`lags` - `default_lags`: must be either a strictly positive integer or a "
                "list. Given : <class 'str'>.",
            ),
            (
                [1],
                "target",
                "`lags` - `default_lags`: list must contain only strictly negative "
                "integers. Given: [1].",
            ),
            (
                [-1.5],
                "target",
                "`lags` - `default_lags`: list must contain only strictly negative "
                "integers. Given: [-1.5].",
            ),
            (
                [0],
                "target",
                "`lags` - `default_lags`: list must contain only strictly negative "
                "integers. Given: [0].",
            ),
            (
                False,
                "past",
                "`lags_past_covariates` - `default_lags`: integer must be strictly "
                "positive . Given: False.",
            ),
            (
                {},
                "target",
                "When passed as a dictionary, `lags` must contain at least one key.",
            ),
            (
                {"default_lags": "bad"},
                "target",
                "`lags` - `default_lags`: must be either a strictly positive integer or a "
                "list. Given : <class 'str'>.",
            ),
            (
                1,
                "future",
                "`lags_future_covariates` - `default_lags`: must be either a tuple or a "
                "list. Given : <class 'int'>.",
            ),
            (
                (1, -3),
                "future",
                "`lags_future_covariates` - `default_lags`: tuple must contain positive "
                "integers. Given: (1, -3).",
            ),
            (
                (1, 2, 3),
                "future",
                "`lags_future_covariates` - `default_lags`: tuple must be of length 2, and "
                "must contain two integers.",
            ),
            (
                (1, True),
                "future",
                "`lags_future_covariates` - `default_lags`: tuple must contain integers, "
                "not bool.",
            ),
            (
                (1, 1.0),
                "future",
                "`lags_future_covariates` - `default_lags`: tuple must be of length 2, and "
                "must contain two integers.",
            ),
            (
                (0, 0),
                "future",
                "`lags_future_covariates` - `default_lags`: tuple cannot be (0, 0) as it "
                "corresponds to an empty list of lags.",
            ),
            (
                [True],
                "future",
                "`lags_future_covariates` - `default_lags`: list must contain only "
                "integers. Given: [True].",
            ),
            (
                {"f": 1},
                "future",
                "`lags_future_covariates` - `f`: must be either a tuple or a list. "
                "Given : <class 'int'>.",
            ),
            (
                [],
                "target",
                "`lags` - `default_lags`: list must contain at least one lag.",
            ),
            (
                {"a": []},
                "target",
                "`lags` - `a`: list must contain at least one lag.",
            ),
            (
                [],
                "future",
                "`lags_future_covariates` - `default_lags`: list must contain at "
                "least one lag.",
            ),
            (
                {"f": []},
                "future",
                "`lags_future_covariates` - `f`: list must contain at least one lag.",
            ),
        ],
    )
    def test_invalid_lags(self, value, kind, message):
        with pytest.raises(ValueError) as exc:
            Lags.from_user(value, kind=kind)

        assert str(exc.value) == message

    def test_shifted(self):
        shared = SharedLags((-1, 0, 1), _kind="future")
        component = ComponentLags({"a": (-1, 0)}, default=(1,), _kind="future")

        assert shared.shifted(2).lags == (1, 2, 3)
        assert component.shifted(3).per_component == {"a": (2, 3)}
        assert component.shifted(3).default == (4,)
        assert shared.lags == (-1, 0, 1)
        assert component.per_component == {"a": (-1, 0)}


class TestResolve:
    def test_shared_lags(self):
        config = LagConfig.from_user(lags=2)
        resolved = config.resolve(["a", "b"])

        assert resolved.target == {"a": (-2, -1), "b": (-2, -1)}
        assert resolved.past is None
        assert resolved.future is None
        assert resolved.is_component_specific == {"target": False}
        assert resolved.legacy_lags_dict() == {"target": [-2, -1]}
        assert resolved.legacy_component_lags_dict() == {}
        assert not resolved.has_stepwise

    def test_component_lags_are_ordered_like_the_series(self):
        config = LagConfig.from_user(lags={"b": [-1], "a": [-3, -2]})
        resolved = config.resolve(["a", "b"])

        assert list(resolved.target) == ["a", "b"]
        assert resolved.target == {"a": (-3, -2), "b": (-1,)}
        assert resolved.legacy_lags_dict() == {"target": [-3, -1]}
        assert resolved.legacy_component_lags_dict() == {
            "target": {"a": [-3, -2], "b": [-1]}
        }

    def test_default_lags_fill_the_missing_components(self):
        config = LagConfig.from_user(lags={"a": [-1], "default_lags": [-4, -3]})
        resolved = config.resolve(["a", "b", "c"])

        assert resolved.target == {
            "a": (-1,),
            "b": (-4, -3),
            "c": (-4, -3),
        }
        assert resolved.legacy_lags_dict() == {"target": [-4, -1]}
        assert resolved.legacy_component_lags_dict() == {
            "target": {
                "a": [-1],
                "b": [-4, -3],
                "c": [-4, -3],
            }
        }

    def test_legacy_lags_include_unused_default_lags(self):
        # The legacy `model.lags` takes the extreme values over the user mapping, including
        # `default_lags`, even when all the components define their own lags.
        config = LagConfig.from_user(lags={"a": [-1], "b": [-2], "default_lags": [-5]})
        resolved = config.resolve(["a", "b"])

        assert resolved.target == {"a": (-1,), "b": (-2,)}
        assert resolved.legacy_lags_dict() == {"target": [-5, -1]}

    def test_unknown_component(self):
        config = LagConfig.from_user(lags={"unknown": [-1]})

        with pytest.raises(ValueError) as exc:
            config.resolve(["a"])

        assert str(exc.value) == (
            "The `lags` dictionary specifies lags for components that are not present in "
            "the series : ['unknown']. They must be removed to avoid any ambiguity."
        )

    def test_missing_component_without_default(self):
        config = LagConfig.from_user(lags={"a": [-1]})

        with pytest.raises(ValueError) as exc:
            config.resolve(["a", "b"])

        assert str(exc.value) == (
            "The lags dictionary is missing the lags for the following components present "
            "in the series: ['b']. The key 'default_lags' can be used to provide lags for "
            "all the non-explicitely defined components."
        )

    def test_errors_for_all_variates_are_aggregated(self):
        config = LagConfig.from_user(
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
        config = LagConfig.from_user(
            lags=2,
            lags_past_covariates=[-2, -1],
            lags_future_covariates={"f1": [0], "f2": [-1], "default_lags": [1]},
            output_chunk_shift=2,
        )
        resolved = config.resolve(["target"], ["past0", "past1"], ["f1", "f2", "f3"])

        assert resolved.target == {
            "target": (-2, -1),
        }
        assert resolved.past == {
            "past0": (-2, -1),
            "past1": (-2, -1),
        }
        assert resolved.future == {
            "f1": (2,),
            "f2": (1,),
            "f3": (3,),
        }
        assert resolved.legacy_lags_dict() == {
            "target": [-2, -1],
            "past": [-2, -1],
            "future": [1, 3],
        }

    def test_output_chunk_shift_with_shared_future_lags(self):
        config = LagConfig.from_user(
            lags=2,
            lags_future_covariates=[0, 1],
            output_chunk_shift=2,
        )
        resolved = config.resolve(["target"], future_components=["f1", "f2"])

        assert resolved.future == {"f1": (2, 3), "f2": (2, 3)}
        assert resolved.legacy_lags_dict()["future"] == [2, 3]

    def test_resolve_is_idempotent_and_does_not_mutate_inputs(self):
        value = {"a": [-1], "default_lags": [-2]}
        config = LagConfig.from_user(lags=value)

        first = config.resolve(["a", "b"])
        second = config.resolve(["a", "b"])

        assert first == second
        assert config.target.per_component == {"a": (-1,)}
        assert config.target.default == (-2,)
        assert value == {"a": [-1], "default_lags": [-2]}

    def test_legacy_dictionaries_are_copies(self):
        resolved = LagConfig.from_user(lags={"a": [-1]}).resolve(["a"])

        lags = resolved.legacy_lags_dict()
        component_lags = resolved.legacy_component_lags_dict()
        lags["target"].append(-99)
        component_lags["target"]["a"].append(-99)

        assert resolved.legacy_lags_dict() == {"target": [-1, -1]}
        assert resolved.legacy_component_lags_dict() == {"target": {"a": [-1]}}


class TestFutureLags:
    def test_stepwise_true_flags_every_component(self):
        config = LagConfig.from_user(
            lags=2,
            lags_future_covariates=[0, 1],
            lags_future_covariates_stepwise=True,
            output_chunk_length=4,
        )
        resolved = config.resolve(["target"], future_components=["f1", "f2"])

        assert resolved.future_stepwise == {"f1": True, "f2": True}
        assert resolved.has_stepwise

    def test_stepwise_false_is_inactive(self):
        config = LagConfig.from_user(
            lags=2,
            lags_future_covariates=[0, 1],
            output_chunk_length=4,
        )
        resolved = config.resolve(["target"], future_components=["f1", "f2"])

        assert resolved.future_stepwise == {}
        assert not resolved.has_stepwise

    def test_stepwise_mapping_with_default(self):
        config = LagConfig.from_user(
            lags=2,
            lags_future_covariates=[0, 1],
            lags_future_covariates_stepwise={"f1": True, "default_lags": False},
            output_chunk_length=4,
        )
        resolved = config.resolve(["target"], future_components=["f1", "f2"])

        assert resolved.future_stepwise == {"f1": True, "f2": False}
        assert resolved.has_stepwise

    def test_stepwise_is_ignored_with_output_chunk_length_one(self):
        config = LagConfig.from_user(
            lags=2,
            lags_future_covariates=[0, 1],
            lags_future_covariates_stepwise=True,
            output_chunk_length=1,
        )
        resolved = config.resolve(["target"], future_components=["f1", "f2"])

        assert resolved.future_stepwise == {}
        assert not resolved.has_stepwise

    def test_stepwise_requires_future_covariates_lags(self):
        with pytest.raises(ValueError) as exc:
            LagConfig.from_user(lags=2, lags_future_covariates_stepwise=True)

        assert str(exc.value) == (
            "`lags_future_covariates_stepwise` requires `lags_future_covariates` to be "
            "not None."
        )

    def test_stepwise_missing_component_flags(self):
        config = LagConfig.from_user(
            lags=2,
            lags_future_covariates={"f1": [0], "default_lags": [0]},
            lags_future_covariates_stepwise={"f1": True},
            output_chunk_length=4,
        )

        with pytest.raises(ValueError) as exc:
            config.resolve(["target"], future_components=["f1", "f2"])

        assert str(exc.value) == (
            "The lags_future_covariates_stepwise dictionary is missing the flags for the "
            "following components present in the series: ['f2']. The key 'default_lags' "
            "can be used to provide flags for all the non-explicitely defined components."
        )

    def test_stepwise_unknown_component_flag(self):
        config = LagConfig.from_user(
            lags=2,
            lags_future_covariates=[0],
            lags_future_covariates_stepwise={"unknown": True, "default_lags": False},
            output_chunk_length=4,
        )

        with pytest.raises(ValueError) as exc:
            config.resolve(["target"], future_components=["f1"])

        assert str(exc.value) == (
            "The `lags_future_covariates_stepwise` dictionary specifies flags for "
            "components that are not present in the series : ['unknown']. They must be "
            "removed to avoid any ambiguity."
        )

    def test_stepwise_component_without_lags(self):
        with pytest.raises(ValueError) as exc:
            LagConfig.from_user(
                lags=2,
                lags_future_covariates={"f1": [0]},
                lags_future_covariates_stepwise={"f2": True},
                output_chunk_length=4,
            )

        assert str(exc.value) == (
            "The `lags_future_covariates_stepwise` dictionary flags components for which "
            "`lags_future_covariates` does not define any lags: ['f2']. Either add them to "
            "`lags_future_covariates` or provide its 'default_lags' key."
        )

    @pytest.mark.parametrize(
        "stepwise,message",
        [
            (
                1,
                "`lags_future_covariates_stepwise` must be a boolean or a dictionary "
                "mapping `future_covariates` component names to booleans. Given: "
                "<class 'int'>.",
            ),
            (
                {},
                "When passed as a dictionary, `lags_future_covariates_stepwise` must "
                "contain at least one key.",
            ),
            (
                {"f1": 1},
                "`lags_future_covariates_stepwise` - `f1`: must be a boolean. Given: "
                "<class 'int'>.",
            ),
        ],
    )
    def test_stepwise_validation(self, stepwise, message):
        with pytest.raises(ValueError) as exc:
            LagConfig.from_user(
                lags=2,
                lags_future_covariates=[0],
                lags_future_covariates_stepwise=stepwise,
                output_chunk_length=4,
            )

        assert str(exc.value) == message

    def test_shifted_preserves_stepwise(self):
        future = FutureLags(
            SharedLags((-1, 0), _kind="future"),
            stepwise={"default_lags": True},
        )

        shifted = future.shifted(2)

        assert shifted.lags.lags == (1, 2)
        assert shifted.stepwise == {"default_lags": True}
        assert shifted.has_stepwise


class TestLagConfig:
    def test_legacy_views_before_resolution(self):
        config = LagConfig.from_user(
            lags={"a": [-2, -1], "b": [-3]},
            lags_past_covariates=2,
            lags_future_covariates={"f": [0, 1], "default_lags": [2]},
            output_chunk_shift=2,
        )

        assert config.legacy_lags_dict() == {
            "target": [-3, -1],
            "past": [-2, -1],
            "future": [2, 4],
        }
        assert config.legacy_component_lags_dict() == {
            "target": {"a": [-2, -1], "b": [-3]},
            "future": {"f": [2, 3], "default_lags": [4]},
        }

    def test_at_least_one_lags_argument(self):
        with pytest.raises(ValueError) as exc:
            LagConfig.from_user()

        assert str(exc.value) == (
            "At least one of `lags`, `lags_future_covariates` or "
            "`lags_past_covariates` must be not None."
        )

    def test_from_user_builds_all_variates(self):
        config = LagConfig.from_user(
            lags={"a": [-1]},
            lags_past_covariates=2,
            lags_future_covariates=(1, 1),
            output_chunk_length=6,
            output_chunk_shift=2,
        )

        assert isinstance(config.target, ComponentLags)
        assert isinstance(config.past, SharedLags)
        assert isinstance(config.future, FutureLags)
        assert config.output_chunk_length == 6
        assert config.output_chunk_shift == 2
