import random

import numpy as np
import pytest

from darts.metrics import mape
from darts.models import FourTheta, Theta
from darts.utils.timeseries_generation import linear_timeseries as lt
from darts.utils.timeseries_generation import random_walk_timeseries as rt
from darts.utils.timeseries_generation import sine_timeseries as st
from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode


class TestFourTheta:
    def test_input(self):
        with pytest.raises(ValueError):
            FourTheta(model_mode=SeasonalityMode.ADDITIVE)
        with pytest.raises(ValueError):
            FourTheta(season_mode=ModelMode.ADDITIVE)
        with pytest.raises((ValueError, TypeError)):
            FourTheta(trend_mode="linear")

    def test_negative_series(self):
        sine_series = st(length=50)
        model = FourTheta(
            model_mode=ModelMode.MULTIPLICATIVE,
            trend_mode=TrendMode.EXPONENTIAL,
            season_mode=SeasonalityMode.ADDITIVE,
            normalization=False,
        )
        model.fit(sine_series)
        assert (
            model.model_mode is ModelMode.ADDITIVE
            and model.trend_mode is TrendMode.LINEAR
        )

    def test_zero_mean(self):
        sine_series = st(length=50)
        with pytest.raises(ValueError):
            model = FourTheta(
                model_mode=ModelMode.MULTIPLICATIVE, trend_mode=TrendMode.EXPONENTIAL
            )
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
            model_mode=random.choice(list(ModelMode)),
            trend_mode=random.choice(list(TrendMode)),
            season_mode=random.choice(list(SeasonalityMode)),
        )
        model.fit(train_series)
        best_model.fit(train_series)
        forecast_random = model.predict(10)
        forecast_best = best_model.predict(10)
        assert mape(val_series, forecast_best) <= mape(val_series, forecast_random)

    def test_min_train_series_length_with_seasonality(self):
        seasonality_period = 12
        fourtheta = FourTheta(
            model_mode=ModelMode.MULTIPLICATIVE,
            trend_mode=TrendMode.EXPONENTIAL,
            season_mode=SeasonalityMode.ADDITIVE,
            seasonality_period=seasonality_period,
            normalization=False,
        )
        theta = Theta(
            season_mode=SeasonalityMode.ADDITIVE,
            seasonality_period=seasonality_period,
        )
        assert fourtheta.min_train_series_length == 2 * seasonality_period
        assert theta.min_train_series_length == 2 * seasonality_period

    def test_min_train_series_length_without_seasonality(self):
        fourtheta = FourTheta(
            model_mode=ModelMode.MULTIPLICATIVE,
            trend_mode=TrendMode.EXPONENTIAL,
            season_mode=SeasonalityMode.ADDITIVE,
            seasonality_period=None,
            normalization=False,
        )
        theta = Theta(
            season_mode=SeasonalityMode.ADDITIVE,
            seasonality_period=None,
        )
        assert fourtheta.min_train_series_length == 3
        assert theta.min_train_series_length == 3

    def test_fit_insufficient_train_series_length(self):
        sine_series = st(length=21, freq="MS")
        with pytest.raises(ValueError):
            fourtheta = FourTheta(
                model_mode=ModelMode.MULTIPLICATIVE,
                trend_mode=TrendMode.EXPONENTIAL,
                season_mode=SeasonalityMode.ADDITIVE,
                seasonality_period=12,
            )
            fourtheta.fit(sine_series)
        with pytest.raises(ValueError):
            theta = Theta(
                season_mode=SeasonalityMode.ADDITIVE,
                seasonality_period=12,
            )
            theta.fit(sine_series)
