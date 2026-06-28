import pytest

from darts import TimeSeries
from darts.logging import get_logger
from darts.models import (
    ARIMA,
    FFT,
    TBATS,
    AutoARIMA,
    AutoCES,
    AutoTheta,
    Croston,
    ExponentialSmoothing,
    FourTheta,
    KalmanForecaster,
    MultivariateModelWrapper,
    NaiveMean,
    NaiveMovingAverage,
    NaiveSeasonal,
    Prophet,
    Theta,
)
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

local_models_kwargs = [
    (NaiveMean, {}),
    (NaiveMovingAverage, {"input_chunk_length": 5}),
    (NaiveSeasonal, {}),
    (ExponentialSmoothing, {}),
    (AutoTheta, {"season_length": 12}),
    (AutoCES, {"season_length": 12, "model": "Z"}),
    (Theta, {"theta": 1}),
    (FourTheta, {"theta": 1}),
    (FFT, {"trend": "poly"}),
    (
        TBATS,
        {
            "use_trend": True,
            "use_arma_errors": True,
            "use_boxcox": True,
            "season_length": [12],
        },
    ),
]

future_covariates_models_kwargs = [
    (Prophet, {}),
    (Croston, {}),
    (AutoARIMA, {}),
    (ARIMA, {"p": 12, "d": 1, "q": 1}),
    (KalmanForecaster, {}),
]


class TestMultivariateForecastingModelWrapper:
    RANDOM_SEED = 42

    ts_length = 50
    n_pred = 5

    univariate = tg.gaussian_timeseries(length=ts_length, mean=50)
    multivariate = univariate.stack(tg.gaussian_timeseries(length=ts_length, mean=20))

    future_covariates = tg.gaussian_timeseries(length=ts_length + n_pred, mean=50)

    @pytest.mark.parametrize("config", local_models_kwargs)
    @pytest.mark.parametrize("series", [univariate, multivariate])
    def test_fit_predict_local_models(self, config, series):
        model_cls, model_kwargs = config
        self._test_predict_with_base_model(model_cls(**model_kwargs), series)

    @pytest.mark.parametrize("config", future_covariates_models_kwargs)
    @pytest.mark.parametrize("series", [univariate, multivariate])
    @pytest.mark.parametrize("future_covariates", [future_covariates, None])
    def test_fit_predict_local_future_covariates_models(
        self, config, series, future_covariates
    ):
        model_cls, model_kwargs = config
        self._test_predict_with_base_model(
            model_cls(**model_kwargs), series, future_covariates
        )

    @pytest.mark.parametrize("config", future_covariates_models_kwargs)
    @pytest.mark.parametrize("series", [univariate, multivariate])
    @pytest.mark.parametrize("future_covariates", [future_covariates, None])
    def test_encoders_support(self, config, series, future_covariates):
        model_cls, model_kwargs = config
        add_encoders = {"position": {"future": ["relative"]}}
        model = model_cls(**{**model_kwargs, "add_encoders": add_encoders})
        self._test_predict_with_base_model(model, series, future_covariates)

    def _test_predict_with_base_model(
        self, model, series: TimeSeries, future_covariates=None
    ):
        preds = self.trained_model_predictions(
            model, self.n_pred, series, future_covariates
        )
        assert isinstance(preds, TimeSeries)
        assert preds.n_components == series.n_components

        # Make sure that the compound prediction is the same as the individual predictions
        individual_preds = self.trained_individual_model_predictions(
            model, self.n_pred, series, future_covariates
        )
        for component in range(series.n_components):
            assert preds.univariate_component(component) == individual_preds[component]

    def trained_model_predictions(self, base_model, n, series, future_covariates):
        model = MultivariateModelWrapper(base_model)
        model.fit(series, future_covariates=future_covariates)
        return model.predict(n=n, series=series, future_covariates=future_covariates)

    def trained_individual_model_predictions(
        self, base_model, n, series, future_covariates
    ):
        predictions = []
        for component in range(series.n_components):
            single_series = series.univariate_component(component)
            model = base_model.untrained_model()
            if model.supports_future_covariates:
                model.fit(single_series, future_covariates=future_covariates)
                predictions.append(
                    model.predict(n=n, future_covariates=future_covariates)
                )
            else:
                model.fit(single_series)
                predictions.append(model.predict(n=n))

        return predictions
