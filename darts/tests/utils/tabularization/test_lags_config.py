import pytest

from darts.utils.data.tabularization.lags import Lags


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
