import copy

from darts import TimeSeries
from darts.logging import get_logger
from darts.models import (
    ARIMA,
    BATS,
    FFT,
    TBATS,
    AutoARIMA,
    Croston,
    ExponentialSmoothing,
    FourTheta,
    KalmanForecaster,
    MultivariateForecastingModelWrapper,
    NaiveMean,
    NaiveMovingAverage,
    NaiveSeasonal,
    Prophet,
    StatsForecastAutoCES,
    StatsForecastAutoTheta,
    Theta,
)
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

local_models = [
    NaiveMean(),
    NaiveMovingAverage(5),
    NaiveSeasonal(),
    ExponentialSmoothing(),
    StatsForecastAutoTheta(season_length=12),
    StatsForecastAutoCES(season_length=12, model="Z"),
    Theta(1),
    FourTheta(1),
    FFT(trend="poly"),
    TBATS(use_trend=True, use_arma_errors=True, use_box_cox=True),
    BATS(use_trend=True, use_arma_errors=True, use_box_cox=True),
]

future_covariates_models = [
    Prophet(),
    Croston(),
    AutoARIMA(),
    ARIMA(12, 1, 1),
    KalmanForecaster(),
]


class TestMultivariateForecastingModelWrapper:
    RANDOM_SEED = 42

    ts_length = 50
    n_pred = 5

    univariate = tg.gaussian_timeseries(length=ts_length, mean=50)
    multivariate = univariate.stack(univariate)

    future_covariates = tg.gaussian_timeseries(length=ts_length + n_pred, mean=50)

    def test_fit_predict_local_models(self):
        for model in local_models:
            self._test_predict_with_base_model(model, None)

    def test_fit_predict_local_future_covariates_models(self):
        for model in future_covariates_models:
            self._test_predict_with_base_model(model, self.future_covariates)

    def test_encoders_support(self):
        add_encoders = {
            "position": {"future": ["relative"]},
        }

        # test some models that support encoders
        for model_object in future_covariates_models:
            # test once with user supplied covariates, and once without
            for fc in [self.future_covariates, None]:
                model_params = {
                    k: vals
                    for k, vals in copy.deepcopy(model_object.model_params).items()
                }
                model_params["add_encoders"] = add_encoders
                model = model_object.__class__(**model_params)

                self._test_predict_with_base_model(model, fc)

    def _test_predict_with_base_model(self, model, future_covariates):
        series_univariate = self.univariate
        series_multivariate = self.multivariate

        combinations = [
            series_univariate,
            series_multivariate,
        ]

        for combination in combinations:
            preds = self.trained_model_predictions(
                model, self.n_pred, combination, future_covariates
            )
            assert isinstance(preds, TimeSeries)
            assert preds.n_components == combination.n_components

    def trained_model_predictions(self, base_model, n, series, future_covariates):
        model = MultivariateForecastingModelWrapper(base_model)
        model.fit(series, future_covariates=future_covariates)
        return model.predict(n=n, series=series, future_covariates=future_covariates)
