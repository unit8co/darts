import json
import logging
import random
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest

from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.metrics import mape
from darts.models import FourTheta, Theta
from darts.tests.parametrize_helpers import param_product
from darts.utils._modes import ADDITIVE, EXPONENTIAL, LINEAR, MULTIPLICATIVE
from darts.utils.timeseries_generation import linear_timeseries as lt
from darts.utils.timeseries_generation import random_walk_timeseries as rt
from darts.utils.timeseries_generation import sine_timeseries as st


class TestFourTheta:
    @pytest.mark.parametrize("model_cls", [Theta, FourTheta])
    @pytest.mark.parametrize("period", [0, 12])
    def test_inferred_seasonality(self, model_cls, period):
        values = 10 + np.random.default_rng(0).normal(scale=0.1, size=120)
        if period:
            values += np.cos(2 * np.pi * np.arange(len(values)) / period)
        model = model_cls().fit(TimeSeries.from_values(values))

        assert model.is_seasonal == (period > 0)
        assert model.season_period == period
        assert np.isfinite(model.predict(12).values()).all()
    series = AirPassengersDataset().load()

    def test_input(self):
        with pytest.raises(ValueError):
            FourTheta(model_mode=LINEAR)
        with pytest.raises(ValueError):
            FourTheta(season_mode=EXPONENTIAL)
        with pytest.raises(ValueError):
            FourTheta(trend_mode="invalid")

    @pytest.mark.parametrize(
        "model_cls,season_mode",
        param_product([Theta, FourTheta], [ADDITIVE, MULTIPLICATIVE, None]),
    )
    def test_season_modes(
        self, model_cls, season_mode: Literal["additive", "multiplicative"] | None
    ):
        model = model_cls(theta=2, seasonality_period=12, season_mode=season_mode)
        model.fit(self.series)
        predictions = model.predict(n=12)
        assert np.isfinite(predictions.values()).all()

    def test_model_params_json_serializable(self):
        model = Theta(2, 12, season_mode=ADDITIVE)
        params = json.loads(json.dumps(dict(model.model_params)))
        assert Theta(**params).model_params == model.model_params

    def test_negative_series(self):
        sine_series = st(length=50)
        model = FourTheta(
            model_mode=MULTIPLICATIVE,
            trend_mode=EXPONENTIAL,
            season_mode=ADDITIVE,
            normalization=False,
        )
        model.fit(sine_series)
        assert model.model_mode == ADDITIVE and model.trend_mode == LINEAR

    def test_fallback_when_drift_is_non_positive(self, caplog):
        # Positive values, but the fitted linear drift crosses zero.
        values = np.array([10.0] * 6 + [0.2] * 4)
        series = TimeSeries.from_values(values.reshape(-1, 1))
        model = FourTheta(
            theta=2,
            model_mode=MULTIPLICATIVE,
            trend_mode=LINEAR,
            season_mode=None,
            normalization=False,
            seasonality_period=1,
        )
        with caplog.at_level(logging.WARNING):
            model.fit(series)
            assert "Negative Theta line" in caplog.text
        caplog.clear()
        assert model.model_mode == ADDITIVE
        assert model.trend_mode == LINEAR
        assert np.isfinite(model.predict(3).values()).all()

    def test_fallback_when_ses_theta_line_is_non_positive(self):
        # A strictly positive series keeps the multiplicative theta line positive,
        # so force the first SES fit to return a non-positive line.
        import statsmodels.tsa.holtwinters as hw

        series = lt(length=24, start_value=10, end_value=20)
        calls = {"n": 0}

        class _NonPositiveFirstFit(hw.SimpleExpSmoothing):
            def fit(self, *args, **kwargs):
                result = super().fit(*args, **kwargs)
                calls["n"] += 1
                if calls["n"] == 1:
                    result.fittedvalues = np.full(len(result.fittedvalues), -1.0)
                return result

        with patch(
            "darts.models.forecasting.theta.hw.SimpleExpSmoothing",
            _NonPositiveFirstFit,
        ):
            model = FourTheta(
                theta=2,
                model_mode=MULTIPLICATIVE,
                trend_mode=LINEAR,
                season_mode=None,
                normalization=False,
                seasonality_period=1,
            )
            model.fit(series)

        assert calls["n"] == 2
        assert model.model_mode == ADDITIVE
        assert np.isfinite(model.predict(3).values()).all()

    def test_select_best_model_negative_series(self):
        series = st(length=24)
        best_model, _, _ = FourTheta.select_best_model(
            series,
            thetas=[2],
            m=1,
            normalization=False,
        )
        assert best_model.model_mode == ADDITIVE
        assert best_model.trend_mode == LINEAR
        assert best_model.season_mode == ADDITIVE

    def test_zero_mean(self):
        sine_series = st(length=50)
        with pytest.raises(ValueError):
            model = FourTheta(model_mode=MULTIPLICATIVE, trend_mode=EXPONENTIAL)
            model.fit(sine_series)

    def test_theta(self):
        np.random.seed(1)
        series = rt(length=50, mean=100)
        theta_param = np.random.randint(1, 5)
        theta = Theta(theta_param)
        fourtheta = FourTheta(theta_param, normalization=False)
        theta.fit(series)
        fourtheta.fit(series)
        forecast_theta = theta.predict(20)
        forecast_fourtheta = fourtheta.predict(20)
        weighted_delta = (forecast_theta - forecast_fourtheta) / forecast_theta
        assert (weighted_delta <= 3e-5).all().item()

    def test_best_model(self):
        random.seed(1)
        sine_series = st(length=50, value_y_offset=50)
        linear_series = lt(length=50)
        series = sine_series + linear_series
        train_series, val_series = series.split_before(series.time_index[-10])
        thetas = np.linspace(-3, 3, 30)
        best_model, _, _ = FourTheta.select_best_model(train_series, thetas)
        model = FourTheta(
            random.choice(thetas),
            model_mode=random.choice([ADDITIVE, MULTIPLICATIVE]),
            trend_mode=random.choice([LINEAR, EXPONENTIAL]),
            season_mode=random.choice([ADDITIVE, MULTIPLICATIVE, None]),
        )
        model.fit(train_series)
        best_model.fit(train_series)
        forecast_random = model.predict(10)
        forecast_best = best_model.predict(10)
        assert mape(val_series, forecast_best) <= mape(val_series, forecast_random)

    def test_min_train_series_length_with_seasonality(self):
        seasonality_period = 12
        fourtheta = FourTheta(
            model_mode=MULTIPLICATIVE,
            trend_mode=EXPONENTIAL,
            season_mode=ADDITIVE,
            seasonality_period=seasonality_period,
            normalization=False,
        )
        theta = Theta(
            season_mode=ADDITIVE,
            seasonality_period=seasonality_period,
        )
        assert fourtheta.min_train_series_length == 2 * seasonality_period
        assert theta.min_train_series_length == 2 * seasonality_period

    def test_min_train_series_length_without_seasonality(self):
        fourtheta = FourTheta(
            model_mode=MULTIPLICATIVE,
            trend_mode=EXPONENTIAL,
            season_mode=ADDITIVE,
            seasonality_period=None,
            normalization=False,
        )
        theta = Theta(
            season_mode=ADDITIVE,
            seasonality_period=None,
        )
        assert fourtheta.min_train_series_length == 3
        assert theta.min_train_series_length == 3

    def test_fit_insufficient_train_series_length(self):
        sine_series = st(length=21, freq="MS")
        with pytest.raises(ValueError):
            fourtheta = FourTheta(
                model_mode=MULTIPLICATIVE,
                trend_mode=EXPONENTIAL,
                season_mode=ADDITIVE,
                seasonality_period=12,
            )
            fourtheta.fit(sine_series)
        with pytest.raises(ValueError):
            theta = Theta(
                season_mode=ADDITIVE,
                seasonality_period=12,
            )
            theta.fit(sine_series)


class TestThetaInputValidation:
    def test_invalid_season_mode(self):
        with pytest.raises(ValueError, match="Unknown value for `season_mode`"):
            Theta(season_mode="invalid")

    def test_zero_theta(self):
        with pytest.raises(ValueError, match="theta cannot be equal to 0"):
            Theta(theta=0)
