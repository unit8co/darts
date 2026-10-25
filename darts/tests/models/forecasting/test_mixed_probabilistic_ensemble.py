"""Mixed ensemble samples must retain deterministic forecasts as point masses."""

import copy

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.models import (
    LinearRegressionModel,
    NaiveEnsembleModel,
    NaiveSeasonal,
    RegressionEnsembleModel,
)
from darts.models.forecasting.forecasting_model import GlobalForecastingModel


def make_series(multivariate=False):
    t = np.arange(80)
    values = (
        2 + 0.03 * t + np.sin(t / 4) + np.random.default_rng(0).normal(0, 0.1, len(t))
    )
    if multivariate:
        values = np.column_stack([values, 3 + np.cos(t / 6)])
    names = ["load", "temperature"] if multivariate else ["load"]
    return TimeSeries.from_values(values, columns=names).with_static_covariates(
        pd.DataFrame({"region": [1] * len(names)})
    )


def probabilistic_model():
    return LinearRegressionModel(
        lags=3, likelihood="quantile", quantiles=[0.1, 0.5, 0.9], random_state=7
    )


@pytest.mark.parametrize("multiple_series", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_naive_forecasts(multiple_series, reverse):
    """Sample-wise means equal independently broadcast member forecasts in either order."""
    base = make_series(multivariate=multiple_series)
    series = (
        [
            base,
            (base * 1.2 + 0.3).with_static_covariates(
                pd.DataFrame({"region": [2] * base.n_components})
            ),
        ]
        if multiple_series
        else base
    )
    deterministic = (
        LinearRegressionModel(lags=2) if multiple_series else NaiveSeasonal(K=3)
    )
    members = [probabilistic_model(), deterministic]
    if reverse:
        members.reverse()
    ensemble = NaiveEnsembleModel(members).fit(series)
    reference_models = copy.deepcopy(ensemble.forecasting_models)
    forecasts = ensemble.predict(
        4, series=series if multiple_series else None, num_samples=7, random_state=9
    )
    expected_members = []
    for member in reference_models:
        kwargs = (
            {"series": series} if isinstance(member, GlobalForecastingModel) else {}
        )
        kwargs["num_samples"] = 7 if member.supports_probabilistic_prediction else 1
        if member.supports_probabilistic_prediction:
            kwargs["random_state"] = 9
        expected_members.append(member.predict(4, **kwargs))
    forecasts = forecasts if multiple_series else [forecasts]
    expected_members = (
        expected_members if multiple_series else [[m] for m in expected_members]
    )
    inputs = series if multiple_series else [series]
    for i, (forecast, original) in enumerate(zip(forecasts, inputs)):
        expected = sum(
            member[i].all_values(copy=False) for member in expected_members
        ) / len(members)
        assert forecast.n_samples == 7
        np.testing.assert_allclose(
            forecast.all_values(copy=False), expected, rtol=1e-12, atol=1e-12
        )
        assert forecast.components.equals(original.components)
        assert forecast.static_covariates.equals(original.static_covariates)
    assert ensemble.supports_probabilistic_prediction
    assert not ensemble._models_are_probabilistic
    assert not ensemble.supports_likelihood_parameter_prediction


@pytest.mark.parametrize("probabilistic_combiner", [False, True])
def test_mixed_regression_training_without_reduction(probabilistic_combiner):
    """Training may retain base samples; output sampling still depends on the combiner."""
    combiner = LinearRegressionModel(
        lags_future_covariates=[0],
        likelihood="quantile" if probabilistic_combiner else None,
        quantiles=[0.1, 0.5, 0.9] if probabilistic_combiner else None,
        use_static_covariates=False,
    )
    ensemble = RegressionEnsembleModel(
        [probabilistic_model(), LinearRegressionModel(lags=2)],
        regression_train_n_points=12,
        regression_model=combiner,
        regression_train_num_samples=7,
        regression_train_samples_reduction=None,
    ).fit(make_series())
    assert ensemble.supports_probabilistic_prediction == probabilistic_combiner
    # Metadata is preserved, but static features are not part of this regression test.
    count = 7 if probabilistic_combiner else 1
    forecast = ensemble.predict(4, num_samples=count, random_state=9)
    assert forecast.n_samples == count
    assert forecast.static_covariates.equals(make_series().static_covariates)
    assert np.isfinite(forecast.all_values(copy=False)).all()
    if not probabilistic_combiner:
        with pytest.raises(ValueError, match="num_samples"):
            ensemble.predict(4, num_samples=7)


def test_deterministic_ensemble_stays_deterministic():
    """An all-deterministic ensemble still rejects stochastic predictions."""
    ensemble = NaiveEnsembleModel([NaiveSeasonal(K=2), NaiveSeasonal(K=3)]).fit(
        make_series()
    )
    assert not ensemble.supports_probabilistic_prediction
    with pytest.raises(ValueError, match="num_samples"):
        ensemble.predict(4, num_samples=7)
