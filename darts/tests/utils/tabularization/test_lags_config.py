import pytest

from darts.utils.data.tabularization.lags import LagConfig, Lags


class TestLagsParse:
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
        lags = Lags.parse(value, kind=kind)

        assert lags.kind == kind
        assert not lags.is_component_specific
        assert lags.shared == expected
        assert lags.component_lags == {}
        assert lags.default is None
        assert lags.extreme_lags == (expected[0], expected[-1])

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
        lags = Lags.parse(value, kind=kind)

        assert lags.kind == kind
        assert lags.is_component_specific
        assert lags.shared is None
        assert lags.component_lags == per_component
        assert lags.default == default

    def test_input_dictionary_is_copied(self):
        value = {"a": [-2, -1]}
        lags = Lags.parse(value, kind="target")

        value["a"].append(-3)
        value["b"] = [-4]

        assert lags.component_lags == {"a": (-2, -1)}

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
            Lags.parse(value, kind=kind)

        assert str(exc.value) == message

    def test_invalid_kind(self):
        with pytest.raises(ValueError) as exc:
            Lags.parse(2, kind="unknown")

        assert str(exc.value) == "Unknown lags kind: 'unknown'."

    def test_direct_constructor_requires_a_representation(self):
        with pytest.raises(ValueError) as exc:
            Lags(kind="target")

        assert str(exc.value) == (
            "`Lags` must define shared or component-specific lags."
        )

    def test_shifted(self):
        shared = Lags(kind="future", shared=(-1, 0, 1))
        component = Lags(kind="future", component_lags={"a": (-1, 0)}, default=(1,))

        assert shared.shifted(2).shared == (1, 2, 3)
        assert component.shifted(3).component_lags == {"a": (2, 3)}
        assert component.shifted(3).default == (4,)
        assert shared.shared == (-1, 0, 1)
        assert component.component_lags == {"a": (-1, 0)}
        assert component.default == (1,)

    def test_extreme_lags_before_and_after_resolution(self):
        lags = Lags.parse(
            {"a": [-2, -1], "b": [-3], "default_lags": [-5]}, kind="target"
        )

        # Before resolution the legacy extremes include the unused default.
        assert lags.extreme_lags == (-5, -1)

        lags.resolve(["a", "b"])

        # After resolution they only cover the actual components.
        assert lags.extreme_lags == (-3, -1)

    def test_resolved_per_component_requires_resolution(self):
        lags = Lags.parse(2, kind="target")

        with pytest.raises(ValueError) as exc:
            lags.resolved_per_component

        assert "`Lags` have not been resolved yet" in str(exc.value)

    def test_resolve_is_idempotent_and_re_resolves(self):
        lags = Lags.parse({"a": [-1], "default_lags": [-2]}, kind="target")

        assert lags.resolve(["a", "b"]) is None
        assert lags.resolved_per_component == {"a": (-1,), "b": (-2,)}

        lags.resolve(["a"])
        assert lags.resolved_per_component == {"a": (-1,)}

        lags.resolve(["a", "b", "c"])
        assert lags.resolved_per_component == {
            "a": (-1,),
            "b": (-2,),
            "c": (-2,),
        }

    def test_failed_resolution_does_not_keep_stale_state(self):
        lags = Lags.parse({"a": [-1]}, kind="target")
        lags.resolve(["a"])

        with pytest.raises(ValueError):
            lags.resolve(["a", "b"])

        with pytest.raises(ValueError):
            lags.resolved_per_component

    def test_shifted_resets_the_resolved_state(self):
        lags = Lags.parse(2, kind="target")
        lags.resolve(["a"])

        shifted = lags.shifted(1)

        assert shifted.shared == (-1, 0)
        with pytest.raises(ValueError):
            shifted.resolved_per_component
        assert lags.resolved_per_component == {"a": (-2, -1)}


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
        assert config.target.component_lags == {"a": (-1,)}
        assert config.target.default == (-2,)
        assert value == {"a": [-1], "default_lags": [-2]}
        assert config.target.extreme_lags == (-2, -1)
        with pytest.raises(ValueError):
            config.target.resolved_per_component

    def test_legacy_dictionaries_are_copies(self):
        resolved = LagConfig.from_user(lags={"a": [-1]}).resolve(["a"])

        lags = resolved.legacy_lags_dict()
        component_lags = resolved.legacy_component_lags_dict()
        lags["target"].append(-99)
        component_lags["target"]["a"].append(-99)

        assert resolved.legacy_lags_dict() == {"target": [-1, -1]}
        assert resolved.legacy_component_lags_dict() == {"target": {"a": [-1]}}


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

        assert config.target.kind == "target"
        assert config.target.is_component_specific
        assert config.past.shared == (-2, -1)
        assert not config.past.is_component_specific
        assert config.future.kind == "future"
        assert config.future.shared == (-1, 0)
        assert config.output_chunk_length == 6
        assert config.output_chunk_shift == 2
