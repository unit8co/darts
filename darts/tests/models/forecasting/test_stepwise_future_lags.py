"""
Tests for `lags_future_covariates_stepwise` at the model level: lags normalization, forcing of the multi-output
wrapper and per-horizon training (issue #2968).
"""

from itertools import product

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.models import (
    CatBoostModel,
    LightGBMModel,
    LinearRegressionModel,
    SKLearnClassifierModel,
    SKLearnModel,
)
from darts.tests.conftest import CB_AVAILABLE, LGBM_AVAILABLE
from darts.utils.data.tabularization import StepwiseLaggedFeatures
from darts.utils.multioutput import MultiOutputRegressor

N_TARGET = 60
OCL = 4


def _target(n: int = N_TARGET, n_comps: int = 1) -> TimeSeries:
    """Target with a linear trend per component, so that linear estimators fit it exactly."""
    values = np.column_stack([
        np.arange(n, dtype=float) + 100.0 * c for c in range(n_comps)
    ])
    return TimeSeries.from_times_and_values(
        pd.RangeIndex(n), values, columns=[f"tgt{c}" for c in range(n_comps)]
    )


def _horizon_params(model, horizon: int, target_dim: int, n_comps: int):
    """
    The `(coef_, intercept_)` of the estimator predicting `horizon` for `target_dim`. Step-wise models are always
    wrapped in a `MultiOutputRegressor`, while a natively multi-output estimator holds one row per output.
    """
    if isinstance(model.model, MultiOutputRegressor):
        estimator = model.get_estimator(horizon=horizon, target_dim=target_dim)
        return estimator.coef_, estimator.intercept_
    idx = horizon * n_comps + target_dim
    return model.model.coef_[idx], model.model.intercept_[idx]


def _covariates(n: int, comps: list[str], offset: float = 1000.0) -> TimeSeries:
    """Future covariates whose component `i` holds `offset * (i + 1) + t`."""
    values = np.column_stack([
        offset * (i + 1) + np.arange(n, dtype=float) for i in range(len(comps))
    ])
    return TimeSeries.from_times_and_values(pd.RangeIndex(n), values, columns=comps)


class TestStepwiseFutureLags:
    # ------------------------------------------------------------------ lags normalization
    @pytest.mark.parametrize(
        "flag,lags_fc,expected_stepwise",
        [
            # a boolean flag is normalized to the `default_lags` entry and resolved in `fit()`
            (True, [0], {"default_lags": True}),
            (True, {"fc0": [0], "fc1": [1]}, {"default_lags": True}),
            # a dictionary is kept as provided until `fit()`
            ({"fc0": True}, {"fc0": [0], "fc1": [1]}, {"fc0": True}),
            (
                {"fc0": True, "default_lags": False},
                {"default_lags": [0]},
                {"fc0": True, "default_lags": False},
            ),
        ],
    )
    def test_lags_normalized_to_component_wise(self, flag, lags_fc, expected_stepwise):
        """Step-wise lags always switch the future lags to their component-wise representation."""
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates=lags_fc,
            lags_future_covariates_stepwise=flag,
            output_chunk_length=OCL,
        )
        assert model.component_lags_stepwise == expected_stepwise
        assert model._uses_stepwise_future_lags
        assert "future" in model.component_lags
        # `lags` only holds the extreme values once the component-wise representation is used
        assert len(model.lags["future"]) == 2

    def test_inactive_flag_leaves_everything_unchanged(self):
        """With the flag disabled, the lags are identical to a model that doesn't know about the parameter."""
        kwargs = dict(lags=2, lags_future_covariates=(1, 3), output_chunk_length=OCL)
        reference = LinearRegressionModel(**kwargs)
        for flag in [False, {"fc0": False}]:
            model = LinearRegressionModel(
                **kwargs, lags_future_covariates_stepwise=flag
            )
            assert model.lags == reference.lags
            assert model.component_lags == reference.component_lags
            assert model.component_lags_stepwise is None
            assert not model._uses_stepwise_future_lags
            assert model._max_stepwise_future_lag is None

    def test_output_chunk_length_one_is_a_no_op(self):
        """With `output_chunk_length=1` every horizon is the first one, so the flag is ignored."""
        reference = LinearRegressionModel(
            lags=2, lags_future_covariates=[0], output_chunk_length=1
        )
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates=[0],
            lags_future_covariates_stepwise=True,
            output_chunk_length=1,
        )
        assert model.component_lags_stepwise is None
        assert not model._uses_stepwise_future_lags
        assert model.lags == reference.lags
        assert model.component_lags == reference.component_lags

        series, fc = _target(), _covariates(N_TARGET + 10, ["fc0"])
        model.fit(series, future_covariates=fc)
        assert model.component_lags_stepwise is None
        assert not isinstance(model.model, MultiOutputRegressor)

    @pytest.mark.parametrize("output_chunk_shift", [0, 2])
    def test_max_stepwise_future_lag(self, output_chunk_shift):
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"fc0": [0, 2], "fc1": [5]},
            lags_future_covariates_stepwise={"fc0": True, "fc1": False},
            output_chunk_length=OCL,
            output_chunk_shift=output_chunk_shift,
        )
        # only the step-wise components are considered, and the shift is already applied to the lags
        assert model._max_stepwise_future_lag == 2 + output_chunk_shift

    # ------------------------------------------------------------------ multi_models=False
    @pytest.mark.parametrize("lag", [-2, 0, 3])
    def test_multi_models_false_is_a_constant_shift(self, lag):
        """With `multi_models=False` step-wise lags are exactly a shift of `output_chunk_length - 1`."""
        series = _target()
        fc = _covariates(N_TARGET + 20, ["fc0"])
        common = dict(lags=2, output_chunk_length=OCL, multi_models=False)

        stepwise = LinearRegressionModel(
            lags_future_covariates={"fc0": [lag]},
            lags_future_covariates_stepwise=True,
            **common,
        )
        legacy = LinearRegressionModel(
            lags_future_covariates={"fc0": [lag + OCL - 1]}, **common
        )
        # the shift is pre-added to the lags, so nothing is step-wise downstream
        assert stepwise.component_lags_stepwise is None
        assert not stepwise._uses_stepwise_future_lags
        assert stepwise.lags == legacy.lags
        assert stepwise.component_lags == legacy.component_lags

        stepwise.fit(series, future_covariates=fc)
        legacy.fit(series, future_covariates=fc)
        assert stepwise.lagged_feature_names == legacy.lagged_feature_names
        np.testing.assert_array_equal(
            stepwise.predict(OCL, future_covariates=fc).values(),
            legacy.predict(OCL, future_covariates=fc).values(),
        )

    def test_multi_models_false_shifts_only_flagged_components(self):
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"fc0": [0], "fc1": [0]},
            lags_future_covariates_stepwise={"fc0": True, "fc1": False},
            output_chunk_length=OCL,
            multi_models=False,
        )
        assert model.component_lags["future"] == {"fc0": [OCL - 1], "fc1": [0]}
        assert model.lags["future"] == [0, OCL - 1]

    def test_multi_models_false_uses_default_lags_for_flagged_component(self):
        """A component that is flagged but has no dedicated lags inherits the default lags (and the shift)."""
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"default_lags": [1]},
            lags_future_covariates_stepwise={"fc0": True},
            output_chunk_length=OCL,
            multi_models=False,
        )
        assert model.component_lags["future"] == {
            "default_lags": [1],
            "fc0": [1 + OCL - 1],
        }

    # ------------------------------------------------------------------ training
    def test_estimators_are_trained_on_their_own_horizon(self):
        """
        Linear probe: with `y[t] = t` and `fc[t] = 1000 + t`, the estimator of horizon `h` sees `fc[t + h]` and
        must predict `y[t + h]`, i.e. fit `coef=1` and `intercept=-1000` for every horizon.
        """
        series = _target()
        fc = _covariates(N_TARGET + 20, ["fc0"])
        model = LinearRegressionModel(
            lags=None,
            lags_future_covariates={"fc0": [0]},
            lags_future_covariates_stepwise=True,
            output_chunk_length=OCL,
        )
        model.fit(series, future_covariates=fc)

        assert isinstance(model.model, MultiOutputRegressor)
        assert len(model.model.estimators_) == OCL
        for horizon in range(OCL):
            estimator = model.get_estimator(horizon=horizon, target_dim=0)
            np.testing.assert_allclose(estimator.coef_, [1.0], atol=1e-8)
            np.testing.assert_allclose(estimator.intercept_, -1000.0, atol=1e-6)

    @pytest.mark.parametrize("lag", [-2, 0, 3])
    def test_horizon_estimator_matches_legacy_shifted_lag(self, lag):
        """
        The estimator of horizon `h` of a step-wise model with lag `k` is trained on exactly the same data as the
        estimator of horizon `h` of a model with the absolute lag `k + h`. The `future_covariates` extend far
        beyond the target, so that both models share the same anchors (the target bounds them).
        """
        series = _target(n_comps=2)
        fc = _covariates(N_TARGET + 40, ["fc0"])
        stepwise = LinearRegressionModel(
            lags=3,
            lags_future_covariates={"fc0": [lag]},
            lags_future_covariates_stepwise=True,
            output_chunk_length=OCL,
        )
        stepwise.fit(series, future_covariates=fc)

        for horizon in range(OCL):
            legacy = LinearRegressionModel(
                lags=3,
                lags_future_covariates={"fc0": [lag + horizon]},
                output_chunk_length=OCL,
            )
            legacy.fit(series, future_covariates=fc)
            for target_dim in range(series.width):
                coef_sw, intercept_sw = _horizon_params(
                    stepwise, horizon, target_dim, series.width
                )
                coef_lg, intercept_lg = _horizon_params(
                    legacy, horizon, target_dim, series.width
                )
                np.testing.assert_allclose(coef_sw, coef_lg, atol=1e-8)
                np.testing.assert_allclose(intercept_sw, intercept_lg, atol=1e-6)

    def test_mixed_stepwise_and_absolute_components(self):
        """A step-wise and an absolute component with the same lag coexist in the same features block."""
        series = _target()
        fc = _covariates(N_TARGET + 40, ["fc0", "fc1"])
        stepwise = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"fc0": [0], "fc1": [0]},
            lags_future_covariates_stepwise={"fc0": True, "fc1": False},
            output_chunk_length=OCL,
        )
        absolute = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"fc0": [0], "fc1": [0]},
            output_chunk_length=OCL,
        )
        stepwise.fit(series, future_covariates=fc)
        absolute.fit(series, future_covariates=fc)
        # the feature names don't carry the horizon: it is given by the estimator
        assert stepwise.lagged_feature_names == absolute.lagged_feature_names
        # horizon 0 is the same for both models, the later horizons differ on the step-wise component only
        np.testing.assert_allclose(
            _horizon_params(stepwise, 0, 0, series.width)[0],
            _horizon_params(absolute, 0, 0, series.width)[0],
            atol=1e-8,
        )

    @pytest.mark.parametrize(
        "lags_fc,flag",
        [
            # component-wise lags of different lengths, mixing step-wise and absolute components
            (
                {"fc0": [-1, 2], "fc1": [0], "fc2": [-2, 0, 1]},
                {"fc0": True, "fc1": False, "fc2": True},
            ),
            # global lags, which the flag converts to the component-wise representation
            ([-1, 0, 2], True),
            ((1, 3), True),
        ],
    )
    def test_feature_names_match_the_columns_of_every_horizon(self, lags_fc, flag):
        """
        `create_lagged_component_names()` is unchanged by step-wise lags: the names still describe the columns of
        every horizon, with `{comp}_futcov_lag{k}` read at `t + h + k` for a step-wise component and at `t + k`
        for an absolute one.
        """
        n_fc = N_TARGET + 20
        offsets = {"fc0": 1000.0, "fc1": 2000.0, "fc2": 3000.0}
        # each value identifies the (component, time) pair it comes from
        fc = TimeSeries.from_times_and_values(
            pd.RangeIndex(n_fc),
            np.column_stack([
                offset + np.arange(n_fc, dtype=float) for offset in offsets.values()
            ]),
            columns=list(offsets),
        )
        series = _target()
        model = LinearRegressionModel(
            lags=None,
            lags_future_covariates=lags_fc,
            lags_future_covariates_stepwise=flag,
            output_chunk_length=OCL,
        )
        model.fit(series, future_covariates=fc)

        X, y, _ = model._create_lagged_data([series], None, [fc], None)
        # `y[0, 0]` is the label of horizon 0, i.e. `series` at the first anchor
        first_anchor = int(y[0, 0])
        for horizon in range(OCL):
            row = X.horizon(horizon)[0]
            assert len(row) == len(model.lagged_feature_names)
            for col, name in enumerate(model.lagged_feature_names):
                comp, _, lag = name.split("_")
                lag = int(lag.removeprefix("lag"))
                shift = horizon if model.component_lags_stepwise[comp] else 0
                assert row[col] == offsets[comp] + first_anchor + lag + shift, (
                    f"horizon {horizon}, column {col} ({name})"
                )

    @pytest.mark.parametrize("lags_fc", [[-1, 0, 2], (1, 3), {"fc0": [0], "fc1": [1]}])
    def test_feature_names_are_not_changed_by_the_flag(self, lags_fc):
        series = _target()
        fc = _covariates(N_TARGET + 20, ["fc0", "fc1"])
        common = dict(lags=2, lags_future_covariates=lags_fc, output_chunk_length=OCL)

        absolute = LinearRegressionModel(**common)
        absolute.fit(series, future_covariates=fc)
        stepwise = LinearRegressionModel(**common, lags_future_covariates_stepwise=True)
        stepwise.fit(series, future_covariates=fc)

        assert stepwise.lagged_feature_names == absolute.lagged_feature_names
        # the horizon 0 features are the absolute ones, on the anchors both models share
        X_abs, _, _ = absolute._create_lagged_data([series], None, [fc], None)
        X_sw, _, _ = stepwise._create_lagged_data([series], None, [fc], None)
        np.testing.assert_array_equal(X_sw.horizon(0), X_abs[: len(X_sw)])

    def test_stepwise_flags_are_resolved_against_the_series_components(self):
        series = _target()
        fc = _covariates(N_TARGET + 40, ["fc0", "fc1", "fc2"])
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"default_lags": [0]},
            lags_future_covariates_stepwise={"fc1": True, "default_lags": False},
            output_chunk_length=OCL,
        )
        model.fit(series, future_covariates=fc)
        assert model.component_lags_stepwise == {
            "fc0": False,
            "fc1": True,
            "fc2": False,
        }
        assert model.component_lags["future"] == {
            "fc0": [0],
            "fc1": [0],
            "fc2": [0],
        }

    def test_component_order_is_normalized_to_the_series(self):
        """
        The tabularization binds the keys of the component-wise lags to the components of the series by
        position (see `_get_lagged_indices()`), so `fit()` must reorder both dictionaries. Declaring the
        components in any order must give exactly the same model.
        """
        series = _target()
        fc = _covariates(N_TARGET + 40, ["fc0", "fc1", "fc2"])
        ordered_lags = {"fc0": [-1, 0], "fc1": [0], "fc2": [1]}
        ordered_flags = {"fc0": False, "fc1": True, "fc2": False}
        scrambled = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"fc2": [1], "fc0": [-1, 0], "fc1": [0]},
            lags_future_covariates_stepwise={"fc1": True, "fc2": False, "fc0": False},
            output_chunk_length=OCL,
        )
        ordered = LinearRegressionModel(
            lags=2,
            lags_future_covariates=ordered_lags,
            lags_future_covariates_stepwise=ordered_flags,
            output_chunk_length=OCL,
        )
        for model in [scrambled, ordered]:
            model.fit(series, future_covariates=fc)

        # both dictionaries follow the components of the series, not the declaration order
        assert list(scrambled.component_lags["future"]) == list(fc.components)
        assert list(scrambled.component_lags_stepwise) == list(fc.components)
        assert scrambled.component_lags["future"] == ordered_lags
        assert scrambled.component_lags_stepwise == ordered_flags

        # the step-wise columns are the ones of the flagged component, and not those of the component
        # sharing its position in the declaration order
        assert scrambled._lagged_feature_names == ordered._lagged_feature_names
        X, _, _ = scrambled._create_lagged_data([series], None, [fc], None)
        step_names = [scrambled._lagged_feature_names[col] for col in X.step_cols]
        assert step_names == ["fc1_futcov_lag0"]

        # ... and the estimators of both models are identical
        for horizon in range(OCL):
            coefs_scrambled, intercept_scrambled = _horizon_params(
                scrambled, horizon, 0, 1
            )
            coefs_ordered, intercept_ordered = _horizon_params(ordered, horizon, 0, 1)
            np.testing.assert_array_almost_equal(coefs_scrambled, coefs_ordered)
            np.testing.assert_almost_equal(intercept_scrambled, intercept_ordered)

    def test_stepwise_flags_from_encoders(self):
        """Components generated by the encoders are covered by the `default_lags` entry."""
        series = TimeSeries.from_times_and_values(
            pd.date_range("2000-01-01", periods=N_TARGET, freq="D"),
            np.arange(N_TARGET, dtype=float),
        )
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates=[0],
            lags_future_covariates_stepwise=True,
            output_chunk_length=OCL,
            add_encoders={"datetime_attribute": {"future": ["dayofweek", "month"]}},
        )
        model.fit(series)
        assert model.component_lags_stepwise == {
            "darts_enc_fc_dta_dayofweek": True,
            "darts_enc_fc_dta_month": True,
        }

    # ------------------------------------------------------------------ multi-output wrapper
    def test_wrapper_is_forced_for_natively_multioutput_estimators(self):
        """`LinearRegression` supports multi-output natively, but each horizon needs its own features array."""
        series = _target()
        fc = _covariates(N_TARGET + 40, ["fc0"])

        absolute = LinearRegressionModel(
            lags=2, lags_future_covariates=[0], output_chunk_length=OCL
        )
        absolute.fit(series, future_covariates=fc)
        assert absolute._supports_native_multioutput
        assert not isinstance(absolute.model, MultiOutputRegressor)

        stepwise = LinearRegressionModel(
            lags=2,
            lags_future_covariates=[0],
            lags_future_covariates_stepwise=True,
            output_chunk_length=OCL,
        )
        stepwise.fit(series, future_covariates=fc)
        assert isinstance(stepwise.model, MultiOutputRegressor)
        assert len(stepwise.model.estimators_) == OCL * series.width

    def test_wrapper_routes_predictions_per_horizon(self):
        """`MultiOutputMixin.predict()` materializes each horizon and keeps the `estimators_` output layout."""
        series = _target(n_comps=2)
        fc = _covariates(N_TARGET + 40, ["fc0"])
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"fc0": [0]},
            lags_future_covariates_stepwise=True,
            output_chunk_length=OCL,
        )
        model.fit(series, future_covariates=fc)

        X, _, _ = model._create_lagged_data(
            series=[series],
            past_covariates=None,
            future_covariates=[fc],
            max_samples_per_ts=None,
        )
        assert isinstance(X, StepwiseLaggedFeatures)
        assert X.n_horizons == OCL

        preds = model.model.predict(X)
        assert preds.shape == (len(X), OCL * series.width)
        expected = np.column_stack([
            estimator.predict(X.horizon(i // series.width))
            for i, estimator in enumerate(model.model.estimators_)
        ])
        np.testing.assert_array_equal(preds, expected)
        # a plain array is still routed to every estimator, as before
        np.testing.assert_array_equal(
            model.model.predict(X.horizon(0)),
            np.column_stack([
                estimator.predict(X.horizon(0)) for estimator in model.model.estimators_
            ]),
        )

    # ------------------------------------------------------------------ prediction
    @pytest.mark.parametrize("multi_models", [True, False])
    def test_extreme_lags_cover_the_forecasted_steps(self, multi_models):
        """A step-wise component is read up to `output_chunk_length - 1` steps further into the future."""
        common = dict(
            lags=2,
            lags_future_covariates={"fc0": [-2, 1], "fc1": [0]},
            output_chunk_length=OCL,
            multi_models=multi_models,
        )
        absolute = LinearRegressionModel(**common)
        stepwise = LinearRegressionModel(
            **common,
            lags_future_covariates_stepwise={"fc1": True, "default_lags": False},
        )
        # the minimum is unchanged (the extension only applies to the forecasted steps, `h >= 0`)
        assert stepwise.extreme_lags[4] == absolute.extreme_lags[4] == -2
        # `fc1` is read up to `0 + OCL - 1`, beyond the largest absolute lag. With `multi_models=False` the
        # lags of the flagged components are shifted instead, which gives the same requirement.
        assert absolute.extreme_lags[5] == 1
        assert stepwise.extreme_lags[5] == OCL - 1

    def test_encoders_generate_the_forecasted_steps(self):
        """The encoder settings must cover the last step the estimators read, or `predict()` would fail."""
        common = dict(lags=2, lags_future_covariates=[0], output_chunk_length=OCL)
        assert LinearRegressionModel(**common)._model_encoder_settings[-1] == [0, 0]
        assert LinearRegressionModel(
            **common, lags_future_covariates_stepwise=True
        )._model_encoder_settings[-1] == [0, OCL - 1]

        series = TimeSeries.from_times_and_values(
            pd.date_range("2000-01-01", periods=N_TARGET, freq="D"),
            np.arange(N_TARGET, dtype=float),
        )
        model = LinearRegressionModel(
            **common,
            lags_future_covariates_stepwise=True,
            add_encoders={"datetime_attribute": {"future": ["dayofweek"]}},
        )
        model.fit(series)
        assert len(model.predict(OCL)) == OCL

    @pytest.mark.parametrize("n", [1, OCL - 1, OCL, OCL + 1, 2 * OCL + 1])
    def test_predict_reads_the_covariates_of_the_forecasted_step(self, n):
        """With `y[t] = t` and `fc[t] = 1000 + t`, every estimator recovers `y` exactly from its own step."""
        series = _target()
        fc = _covariates(N_TARGET + 3 * OCL, ["fc0"])
        model = LinearRegressionModel(
            lags=None,
            lags_future_covariates={"fc0": [0]},
            lags_future_covariates_stepwise=True,
            output_chunk_length=OCL,
        )
        model.fit(series, future_covariates=fc)
        for horizon in range(OCL):
            coefs, intercept = _horizon_params(model, horizon, 0, 1)
            np.testing.assert_array_almost_equal(coefs, [1.0])
            np.testing.assert_almost_equal(intercept, -1000.0)

        pred = model.predict(n, future_covariates=fc)
        np.testing.assert_array_almost_equal(
            pred.values().ravel(), np.arange(N_TARGET, N_TARGET + n, dtype=float)
        )

    @pytest.mark.parametrize("lag", [-1, 0, 1])
    def test_predict_matches_the_legacy_shifted_lags_per_horizon(self, lag):
        """The forecast of the horizon `h` is the one of a legacy model whose lag is shifted by `h`."""
        rng = np.random.default_rng(42)
        series = TimeSeries.from_times_and_values(
            pd.RangeIndex(N_TARGET), rng.normal(size=N_TARGET)
        )
        fc = TimeSeries.from_times_and_values(
            pd.RangeIndex(N_TARGET + 40),
            rng.normal(size=(N_TARGET + 40, 1)),
            columns=["fc0"],
        )
        stepwise = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"fc0": [lag]},
            lags_future_covariates_stepwise=True,
            output_chunk_length=OCL,
        )
        stepwise.fit(series, future_covariates=fc)
        forecast = stepwise.predict(OCL, future_covariates=fc).values().ravel()

        for horizon in range(OCL):
            legacy = LinearRegressionModel(
                lags=2,
                lags_future_covariates={"fc0": [lag + horizon]},
                output_chunk_length=OCL,
            )
            # the step-wise model loses the last `OCL - 1` anchors, so the covariates of the legacy model
            # are trimmed to train both of them on the same samples
            legacy.fit(series, future_covariates=fc[: len(fc) - (OCL - 1 - horizon)])
            np.testing.assert_almost_equal(
                forecast[horizon],
                legacy.predict(OCL, future_covariates=fc).values().ravel()[horizon],
            )

    def test_predict_requires_the_forecasted_steps_of_the_covariates(self):
        """Covariates that are long enough for the absolute lags can be too short for the step-wise ones."""
        series = _target()
        fc = _covariates(N_TARGET + 3 * OCL, ["fc0"])
        common = dict(
            lags=2, lags_future_covariates={"fc0": [0]}, output_chunk_length=OCL
        )
        absolute = LinearRegressionModel(**common)
        absolute.fit(series, future_covariates=fc)
        stepwise = LinearRegressionModel(**common, lags_future_covariates_stepwise=True)
        stepwise.fit(series, future_covariates=fc)

        # exactly the single step the absolute lag `0` requires: it is read once per sample, at the first
        # step of the output chunk, whereas the step-wise model reads it at every one of the `OCL` steps
        short_fc = fc[: N_TARGET + 1]
        assert len(absolute.predict(OCL, future_covariates=short_fc)) == OCL
        with pytest.raises(ValueError, match="are not long enough"):
            stepwise.predict(OCL, future_covariates=short_fc)
        # the message reports the declared lag and the step it is actually read at
        with pytest.raises(ValueError, match=r"read up to " + str(OCL - 1)):
            stepwise.predict(OCL, future_covariates=short_fc)

    @pytest.mark.parametrize(
        ("lags_fc", "stepwise_flag"),
        [
            ({"fc0": [-1, 0], "fc1": [-1, 0]}, True),
            ({"fc0": [-1, 0], "fc1": [-1, 0]}, {"fc0": True, "default_lags": False}),
            # the absolute `fc1` is read further into the future than the step-wise `fc0`, so the
            # covariates window is the one of the absolute lags and no anchor is lost
            ({"fc0": [-1, 0], "fc1": [0, 2]}, {"fc0": True, "default_lags": False}),
        ],
        ids=["all-stepwise", "mixed-absolute", "absolute-reaches-further"],
    )
    def test_historical_forecasts_optimized_match_the_predict_loop(
        self, lags_fc, stepwise_flag
    ):
        """The optimized routine produces the same forecasts as the `predict()` loop."""
        rng = np.random.default_rng(0)
        series = TimeSeries.from_times_and_values(
            pd.RangeIndex(N_TARGET), rng.normal(size=N_TARGET)
        )
        fc = _covariates(N_TARGET + 40, ["fc0", "fc1"], offset=1000.0)
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates=lags_fc,
            lags_future_covariates_stepwise=stepwise_flag,
            output_chunk_length=OCL,
        )
        model.fit(series, future_covariates=fc)
        assert model._check_optimizable_historical_forecasts(retrain=False)

        # forecast horizon below, equal to and above `output_chunk_length` (auto-regression),
        # with and without stride, and both `last_points_only` modes
        for forecast_horizon, stride, last_points_only in product(
            (1, OCL - 1, OCL, OCL + 3), (1, 2), (True, False)
        ):
            kwargs = dict(
                series=series,
                future_covariates=fc,
                retrain=False,
                forecast_horizon=forecast_horizon,
                stride=stride,
                last_points_only=last_points_only,
                start=N_TARGET - 20,
                start_format="value",
            )
            optimized = model.historical_forecasts(enable_optimization=True, **kwargs)
            reference = model.historical_forecasts(enable_optimization=False, **kwargs)
            if last_points_only:
                optimized, reference = [optimized], [reference]
            assert len(optimized) == len(reference)
            for forecast, expected in zip(optimized, reference):
                np.testing.assert_array_almost_equal(
                    forecast.values(), expected.values()
                )
                assert forecast.time_index.equals(expected.time_index)

        # the last forecast is the one of a plain `predict()` on the truncated series
        reference = model.historical_forecasts(
            enable_optimization=False,
            series=series,
            future_covariates=fc,
            retrain=False,
            forecast_horizon=OCL,
            last_points_only=False,
            start=N_TARGET - 20,
            start_format="value",
        )
        np.testing.assert_array_almost_equal(
            reference[-1].values(),
            model.predict(
                OCL,
                series=series[: len(series) - OCL],
                future_covariates=fc,
            ).values(),
        )

        # with re-training, the non-optimized predict loop covers the same forecastable span
        retrained = model.historical_forecasts(
            series=series,
            future_covariates=fc,
            forecast_horizon=OCL,
            last_points_only=True,
            start=N_TARGET - 20,
            start_format="value",
        )
        reference_last_points = model.historical_forecasts(
            series=series,
            future_covariates=fc,
            retrain=False,
            forecast_horizon=OCL,
            last_points_only=True,
            start=N_TARGET - 20,
            start_format="value",
        )
        assert retrained.time_index.equals(reference_last_points.time_index)
        assert np.isfinite(retrained.values()).all()

    def test_historical_forecasts_with_quantile_likelihood(self):
        """The optimized routine routes the per-horizon features of every quantile model."""
        series = _target()
        fc = _covariates(N_TARGET + 40, ["fc0"])
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"fc0": [0]},
            lags_future_covariates_stepwise=True,
            output_chunk_length=OCL,
            likelihood="quantile",
            quantiles=[0.1, 0.5, 0.9],
        )
        model.fit(series, future_covariates=fc)
        for kwargs in (
            dict(forecast_horizon=OCL, num_samples=1),
            # auto-regression with sampled forecasts: exercises `repeat()` on the container
            dict(forecast_horizon=OCL + 3, num_samples=4),
        ):
            hf_kwargs = dict(
                series=series,
                future_covariates=fc,
                retrain=False,
                last_points_only=False,
                start=N_TARGET - 20,
                start_format="value",
                random_state=42,
                **kwargs,
            )
            optimized = model.historical_forecasts(
                enable_optimization=True, **hf_kwargs
            )
            reference = model.historical_forecasts(
                enable_optimization=False, **hf_kwargs
            )
            assert len(optimized) == len(reference)
            for forecast, expected in zip(optimized, reference):
                np.testing.assert_array_almost_equal(
                    forecast.all_values(), expected.all_values()
                )

    def test_predict_with_static_covariates_and_multiple_series(self):
        """The per-horizon container keeps its rows aligned across series and static covariates."""
        fc = _covariates(N_TARGET + 3 * OCL, ["fc0"])
        series = [
            _target().with_static_covariates(pd.DataFrame({"sc": [value]}))
            for value in (1.0, 2.0)
        ]
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"fc0": [0]},
            lags_future_covariates_stepwise=True,
            output_chunk_length=OCL,
        )
        model.fit(series, future_covariates=[fc, fc])
        assert model.uses_static_covariates

        forecasts = model.predict(OCL, series=series, future_covariates=[fc, fc])
        for idx, forecast in enumerate(forecasts):
            np.testing.assert_array_almost_equal(
                forecast.values(),
                model.predict(OCL, series=series[idx], future_covariates=fc).values(),
            )

    def test_predict_with_quantile_likelihood(self):
        """Sampling and likelihood parameters go through the per-horizon routing of every quantile model."""
        series = _target()
        fc = _covariates(N_TARGET + 3 * OCL, ["fc0"])
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"fc0": [0]},
            lags_future_covariates_stepwise=True,
            output_chunk_length=OCL,
            likelihood="quantile",
            quantiles=[0.1, 0.5, 0.9],
        )
        model.fit(series, future_covariates=fc)

        sampled = model.predict(OCL, future_covariates=fc, num_samples=50)
        assert sampled.n_samples == 50
        assert len(sampled) == OCL

        params = model.predict(
            OCL, future_covariates=fc, predict_likelihood_parameters=True
        )
        assert len(params) == OCL
        assert params.width == 3

    # ------------------------------------------------------------------ other estimators
    @pytest.mark.skipif(not LGBM_AVAILABLE, reason="requires lightgbm")
    def test_lgbm_with_validation_set(self):
        """Each horizon gets its own validation features array through the wrapper's `eval_set`."""
        series = _target()
        fc = _covariates(N_TARGET + 40, ["fc0"])
        model = LightGBMModel(
            lags=2,
            lags_future_covariates={"fc0": [0]},
            lags_future_covariates_stepwise=True,
            output_chunk_length=OCL,
            n_estimators=2,
            verbose=-1,
        )
        model.fit(
            series[:40],
            future_covariates=fc,
            val_series=series[40:],
            val_future_covariates=fc,
        )
        assert len(model.model.estimators_) == OCL

    @pytest.mark.skipif(not CB_AVAILABLE, reason="requires catboost")
    def test_catboost_with_categorical_features_and_validation_set(self):
        """CatBoost materializes one validation `Pool` per horizon, and formats each horizon's features."""
        series = _target()
        fc = TimeSeries.from_times_and_values(
            pd.RangeIndex(N_TARGET + 40),
            np.column_stack([
                (np.arange(N_TARGET + 40) % 3).astype(float),
                1000.0 + np.arange(N_TARGET + 40, dtype=float),
            ]),
            columns=["cat", "fc0"],
        )
        model = CatBoostModel(
            lags=2,
            lags_future_covariates={"cat": [0], "fc0": [0]},
            lags_future_covariates_stepwise={"cat": True, "default_lags": False},
            output_chunk_length=OCL,
            categorical_future_covariates=["cat"],
            iterations=2,
        )
        model.fit(
            series[:40],
            future_covariates=fc,
            val_series=series[40:],
            val_future_covariates=fc,
        )
        assert len(model.model.estimators_) == OCL

    def test_classifier_fit(self):
        from sklearn.linear_model import LogisticRegression

        labels = TimeSeries.from_times_and_values(
            pd.RangeIndex(N_TARGET), (np.arange(N_TARGET) % 2).astype(float)
        )
        fc = _covariates(N_TARGET + 40, ["fc0"])
        model = SKLearnClassifierModel(
            model=LogisticRegression(),
            lags=2,
            lags_future_covariates={"fc0": [0]},
            lags_future_covariates_stepwise=True,
            output_chunk_length=OCL,
        )
        model.fit(labels, future_covariates=fc)
        assert len(model.model.estimators_) == OCL
        assert len(model.model.classes_) == OCL

    # ------------------------------------------------------------------ serialization
    def test_model_saved_before_the_parameter_existed(self):
        """Models pickled before this parameter existed have no `component_lags_stepwise` attribute."""
        model = LinearRegressionModel(
            lags=2, lags_future_covariates=[0], output_chunk_length=OCL
        )
        del model.component_lags_stepwise
        assert model._stepwise_future_lags is None
        assert not model._uses_stepwise_future_lags
        assert model._max_stepwise_future_lag is None

    # ------------------------------------------------------------------ errors
    @pytest.mark.parametrize(
        "flag,lags_fc,match",
        [
            (
                "yes",
                [0],
                "must be a boolean or a dictionary",
            ),
            ({}, [0], "must contain at least one key"),
            ({"fc0": "yes"}, [0], "`fc0`: must be a boolean"),
            (True, None, "requires `lags_future_covariates` to be not None"),
            (
                {"fc0": True},
                {"fc1": [0]},
                "does not define any lags",
            ),
        ],
    )
    def test_invalid_flag(self, flag, lags_fc, match):
        with pytest.raises(ValueError, match=match):
            LinearRegressionModel(
                lags=2,
                lags_future_covariates=lags_fc,
                lags_future_covariates_stepwise=flag,
                output_chunk_length=OCL,
            )

    def test_unknown_component_raises_at_fit(self):
        series = _target()
        fc = _covariates(N_TARGET + 40, ["fc0"])
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"default_lags": [0]},
            lags_future_covariates_stepwise={"unknown": True},
            output_chunk_length=OCL,
        )
        with pytest.raises(
            ValueError, match="components that are not present in the series"
        ):
            model.fit(series, future_covariates=fc)

    def test_missing_component_raises_at_fit(self):
        series = _target()
        fc = _covariates(N_TARGET + 40, ["fc0", "fc1"])
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"default_lags": [0]},
            lags_future_covariates_stepwise={"fc0": True},
            output_chunk_length=OCL,
        )
        with pytest.raises(
            ValueError, match="is missing the flags for the following components"
        ):
            model.fit(series, future_covariates=fc)

    def test_refit_with_an_additional_component_requires_default_lags(self):
        """
        `fit()` expands both dictionaries onto the components of the series, which consumes their
        `default_lags` entry. Re-fitting with an additional component therefore raises, exactly as it
        already does for `lags_future_covariates` on its own.
        """
        series = _target()
        fc = _covariates(N_TARGET + 40, ["fc0", "fc1"])
        model = LinearRegressionModel(
            lags=2,
            lags_future_covariates={"default_lags": [0]},
            lags_future_covariates_stepwise={"default_lags": True},
            output_chunk_length=OCL,
        )
        model.fit(series, future_covariates=fc)
        # the `default_lags` entry is replaced by one entry per component of the series
        assert model.component_lags["future"] == {"fc0": [0], "fc1": [0]}
        assert model.component_lags_stepwise == {"fc0": True, "fc1": True}

        fc_extra = _covariates(N_TARGET + 40, ["fc0", "fc1", "fc2"])
        with pytest.raises(ValueError) as exc:
            model.fit(series, future_covariates=fc_extra)
        # both dictionaries report the missing component, in a single error message
        assert (
            "lags_future_covariates dictionary is missing the lags for the following "
            "components present in the series: ['fc2']" in str(exc.value)
        )
        assert (
            "lags_future_covariates_stepwise dictionary is missing the flags for the "
            "following components present in the series: ['fc2']" in str(exc.value)
        )

    def test_stepwise_drops_the_last_anchors(self):
        """Step-wise components need `output_chunk_length - 1` more covariates values than absolute ones."""
        series = _target()
        # bounded by the covariates rather than by the target
        fc = _covariates(N_TARGET - (OCL - 1), ["fc0"])
        common = dict(
            lags=2, lags_future_covariates={"fc0": [0]}, output_chunk_length=OCL
        )

        absolute = LinearRegressionModel(**common)
        absolute.fit(series, future_covariates=fc)
        stepwise = LinearRegressionModel(**common, lags_future_covariates_stepwise=True)
        stepwise.fit(series, future_covariates=fc)

        X_abs, _, _ = absolute._create_lagged_data([series], None, [fc], None)
        X_sw, _, _ = stepwise._create_lagged_data([series], None, [fc], None)
        assert len(X_sw) == len(X_abs) - (OCL - 1)

    def test_stepwise_requires_longer_future_covariates(self):
        """
        Step-wise components require `output_chunk_length - 1` more values, so covariates that are long enough
        for absolute lags can leave no valid sample at all.
        """
        series = _target()
        short_fc = _covariates(OCL, ["fc0"])
        common = dict(
            lags=2, lags_future_covariates={"fc0": [0]}, output_chunk_length=OCL
        )
        LinearRegressionModel(**common).fit(series, future_covariates=short_fc)

        model = LinearRegressionModel(**common, lags_future_covariates_stepwise=True)
        with pytest.raises(ValueError, match="do not share any common times"):
            model.fit(series, future_covariates=short_fc)

    def test_multi_output_wrapper_rejects_inconsistent_horizons(self):
        model = SKLearnModel(
            lags=2,
            lags_future_covariates={"fc0": [0]},
            lags_future_covariates_stepwise=True,
            output_chunk_length=OCL,
        )
        model.fit(_target(), future_covariates=_covariates(N_TARGET + 40, ["fc0"]))
        wrapper = model.model
        X, y, _ = model._create_lagged_data(
            series=[_target()],
            past_covariates=None,
            future_covariates=[_covariates(N_TARGET + 40, ["fc0"])],
            max_samples_per_ts=None,
        )
        with pytest.raises(
            ValueError, match="must be a multiple of the number of horizons"
        ):
            wrapper.fit(X, np.concatenate([y, y[:, :1]], axis=1))
