"""Selective SHAP construction with real tree models and additive references."""

from functools import wraps

import numpy as np
import pytest
import shap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from darts import TimeSeries
from darts.explainability import ShapExplainer
from darts.models import SKLearnModel


@pytest.fixture
def tree_constructions(monkeypatch):
    constructions = []
    original = shap.TreeExplainer.__init__

    @wraps(original)
    def counted(self, *args, **kwargs):
        constructions.append((args, kwargs))
        original(self, *args, **kwargs)

    monkeypatch.setattr(shap.TreeExplainer, "__init__", counted)
    return constructions


def series(multivariate=False):
    x = np.arange(64, dtype=float)
    values = (np.sin(x / 3) + x / 15)[:, None]
    columns = ["y"]
    if multivariate:
        values = np.column_stack([values[:, 0], np.cos(x / 5) - x / 30])
        columns.append("other")
    return TimeSeries.from_values(values, columns=columns)


def quantile_model(multivariate=False):
    pytest.importorskip("lightgbm")
    from darts.models import LightGBMModel

    target = series(multivariate)
    model = LightGBMModel(
        lags=4,
        output_chunk_length=3,
        likelihood="quantile",
        quantiles=[0.1, 0.5, 0.9],
        n_estimators=5,
        num_leaves=4,
        min_child_samples=3,
        n_jobs=1,
        verbosity=-1,
        random_state=13,
    )
    model.fit(target)
    return model, target


def test_requested_quantile_horizons_are_lazy_and_cached(tree_constructions):
    model, target = quantile_model()
    explainer = ShapExplainer(model, test_stationarity=False)
    assert len(tree_constructions) == 0
    first = explainer.explain(horizons=[1], target_components=["y_q0.500"])
    assert len(tree_constructions) == 1
    again = explainer.explain(horizons=[1], target_components=["y_q0.500"])
    assert len(tree_constructions) == 1
    np.testing.assert_array_equal(
        first.get_explanation(1, "y_q0.500").values(),
        again.get_explanation(1, "y_q0.500").values(),
    )
    explainer.explain(horizons=[3], target_components=["y_q0.100"])
    assert len(tree_constructions) == 2
    explainer.explain()
    assert len(tree_constructions) == 9


@pytest.mark.parametrize(
    "multivariate,component,target_dim,quantile",
    [(False, "y_q0.500", 0, 0.5), (True, "other_q0.900", 1, 0.9)],
)
def test_lazy_values_reconstruct_selected_estimator(
    tree_constructions, multivariate, component, target_dim, quantile
):
    model, target = quantile_model(multivariate)
    explainer = ShapExplainer(model, test_stationarity=False)
    assert len(tree_constructions) == 0
    result = explainer.explain(
        foreground_series=target[-20:], horizons=[2], target_components=[component]
    )
    assert len(tree_constructions) == 1
    obj = result.get_shap_explanation_object(2, component)
    estimate = model.get_estimator(horizon=1, target_dim=target_dim, quantile=quantile)
    expected = estimate.predict(obj.data)
    np.testing.assert_allclose(
        obj.base_values + obj.values.sum(axis=1), expected, rtol=1e-6, atol=1e-6
    )


def test_explain_single_uses_the_same_cache(tree_constructions):
    model, target = quantile_model()
    explainer = ShapExplainer(model, test_stationarity=False)
    assert len(tree_constructions) == 0
    explainer.explain(horizons=[1], target_components=["y_q0.500"])
    assert len(tree_constructions) == 1
    first = explainer.explain_single(target, target_components=["y_q0.500"])
    assert len(tree_constructions) == 3
    assert np.isfinite(first.get_explanation("y_q0.500").values()).all()
    assert first.get_explanation("y_q0.500").n_timesteps == 3
    explainer.explain_single(target, target_components=["y_q0.500"])
    assert len(tree_constructions) == 3


def test_deterministic_wrapped_estimators_are_lazy(tree_constructions):
    target = series()
    model = SKLearnModel(
        lags=4,
        output_chunk_length=3,
        model=GradientBoostingRegressor(n_estimators=5, random_state=13),
    )
    model.fit(target)
    explainer = ShapExplainer(model, test_stationarity=False)
    assert len(tree_constructions) == 0
    result = explainer.explain(horizons=[2], target_components=["y"])
    assert len(tree_constructions) == 1
    obj = result.get_shap_explanation_object(2, "y")
    expected = model.get_estimator(horizon=1, target_dim=0).predict(obj.data)
    np.testing.assert_allclose(obj.base_values + obj.values.sum(axis=1), expected)


def test_native_multioutput_remains_eager(tree_constructions):
    target = series()
    model = SKLearnModel(
        lags=4,
        output_chunk_length=3,
        model=DecisionTreeRegressor(max_depth=2, random_state=13),
    )
    model.fit(target)
    explainer = ShapExplainer(model, test_stationarity=False)
    assert len(tree_constructions) == 1
    result = explainer.explain(horizons=[2], target_components=["y"])
    assert len(tree_constructions) == 1
    assert np.isfinite(result.get_explanation(2, "y").values()).all()


def test_constructor_options_reach_lazy_tree(tree_constructions):
    model, target = quantile_model()
    explainer = ShapExplainer(
        model,
        test_stationarity=False,
        feature_perturbation="interventional",
        model_output="raw",
    )
    assert len(tree_constructions) == 0
    result = explainer.explain(horizons=[1], target_components=["y_q0.500"])
    assert len(tree_constructions) == 1
    assert tree_constructions[0][1]["feature_perturbation"] == "interventional"
    np.testing.assert_array_equal(
        tree_constructions[0][0][1], explainer.explainer.background_arr
    )
    obj = result.get_shap_explanation_object(1, "y_q0.500")
    expected = model.get_estimator(horizon=0, target_dim=0, quantile=0.5).predict(
        obj.data
    )
    np.testing.assert_allclose(
        obj.base_values + obj.values.sum(axis=1), expected, rtol=1e-5, atol=1e-6
    )


def test_invalid_method_still_fails_before_construction(tree_constructions):
    model, target = quantile_model()
    with pytest.raises(ValueError, match="Invalid `shap_method"):
        ShapExplainer(model, shap_method="invalid", test_stationarity=False)
    assert len(tree_constructions) == 0
