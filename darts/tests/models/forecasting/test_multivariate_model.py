import numpy as np
import pytest

from darts import TimeSeries, concatenate
from darts.logging import get_logger
from darts.models import (
    ARIMA,
    ExponentialSmoothing,
    LinearRegressionModel,
    MultivariateModel,
    NaiveSeasonal,
    Prophet,
    StatsForecastModel,
)
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.tests.conftest import PROPHET_AVAILABLE, SF_AVAILABLE
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

no_cov_models_kwargs = [
    (ExponentialSmoothing, {}),  # local, univariate
    (NaiveSeasonal, {}),  # local, uni- and multivariate
    (LinearRegressionModel, {"lags": 3}),  # global, multivariate
]

fut_cov_models_kwargs: list[tuple[type[ForecastingModel], dict]] = [
    (ARIMA, {"p": 12, "d": 1, "q": 1}),  # transferable future cov model
]
if PROPHET_AVAILABLE:
    fut_cov_models_kwargs.append(
        (Prophet, {}),  # simple future cov model
    )
if SF_AVAILABLE:
    fut_cov_models_kwargs.append(
        (StatsForecastModel, {"model": "AutoETS"}),  # transferable future cov model
    )

prob_models_kwargs = [
    (ExponentialSmoothing, {}),  # local, univariate
    (
        LinearRegressionModel,
        {
            "lags": 3,
            "output_chunk_length": 5,
            "likelihood": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
        },
    ),  # global, multivariate
]


class TestMultivariateModel:
    np.random.seed(42)

    ts_length = 50
    n_pred = 5

    univariate = tg.gaussian_timeseries(length=ts_length, mean=50)
    multivariate = univariate.stack(tg.gaussian_timeseries(length=ts_length, mean=20))

    future_covariates = tg.gaussian_timeseries(length=ts_length + n_pred, mean=50)

    @pytest.mark.parametrize("config", no_cov_models_kwargs)
    @pytest.mark.parametrize("series", [univariate, multivariate])
    def test_fit_predict_local_models(self, config, series):
        model_cls, model_kwargs = config
        model = self.helper_setup_model(model_cls, model_kwargs)
        self.helper_test_predict_with_base_model(model, series)

    @pytest.mark.parametrize("config", fut_cov_models_kwargs)
    @pytest.mark.parametrize("series", [univariate, multivariate])
    @pytest.mark.parametrize("future_covariates", [future_covariates, None])
    def test_fit_predict_local_future_covariates_models(
        self, config, series, future_covariates
    ):
        model_cls, model_kwargs = config
        model = self.helper_setup_model(model_cls, model_kwargs)
        self.helper_test_predict_with_base_model(model, series, future_covariates)

    @pytest.mark.parametrize("config", fut_cov_models_kwargs)
    @pytest.mark.parametrize("series", [univariate, multivariate])
    @pytest.mark.parametrize("future_covariates", [future_covariates, None])
    def test_encoders_support(self, config, series, future_covariates):
        model_cls, model_kwargs = config
        add_encoders = {"position": {"future": ["relative"]}}
        model = self.helper_setup_model(
            model_cls, {**model_kwargs, "add_encoders": add_encoders}
        )
        self.helper_test_predict_with_base_model(model, series, future_covariates)

    @pytest.mark.parametrize("config", prob_models_kwargs)
    def test_probabilistic_forecasts(self, config):
        model_cls, model_kwargs = config
        model = self.helper_setup_model(model=model_cls, model_kwargs=model_kwargs)

        n, n_comps = 5, self.multivariate.n_components
        params = {"model": model, "series": self.multivariate, "random_state": 42}
        preds = self.helper_test_predict_with_base_model(**params)
        assert preds.shape == (n, n_comps, 1)

        preds = self.helper_test_predict_with_base_model(num_samples=10, **params)
        assert preds.shape == (n, n_comps, 10)

        if model.supports_likelihood_parameter_prediction:
            model.fit(self.multivariate)
            predict_kwargs = {"n": n, "random_state": 42}
            if model.supports_transferable_series_prediction:
                predict_kwargs["series"] = self.multivariate
            preds = model.predict(predict_likelihood_parameters=True, **predict_kwargs)
            n_comps_params = n_comps * 3
            assert preds.shape == (n, n_comps_params, 1)

    @pytest.mark.parametrize("model_setup", ["class", "object", "string"])
    def test_model_init(self, model_setup):
        """Test that models can be created with class, object, or string."""
        model = self.helper_setup_model(
            model=ExponentialSmoothing,
            model_kwargs={},
            model_setup=model_setup,
        )
        assert isinstance(model, MultivariateModel)
        assert isinstance(model._base_model, ExponentialSmoothing)

    def test_invalid_model(self):
        with pytest.raises(
            ValueError, match="Could not find a `model` named 'invalid'"
        ):
            _ = self.helper_setup_model(
                model="invalid", model_kwargs=dict(), model_setup="string"
            )

        with pytest.raises(
            ValueError, match="`model` must be a valid Darts `ForecastingModel`"
        ):
            _ = MultivariateModel(model=1)

    def test_unsupported_future_covariates(self):
        """Test that passing future_covariates to a model that doesn't support them raises."""
        model = self.helper_setup_model(ExponentialSmoothing, {})
        assert not model.supports_future_covariates

        with pytest.raises(
            ValueError,
            match="The underlying model does not support `future_covariates`.",
        ):
            model.fit(self.univariate, future_covariates=self.future_covariates)

        model.fit(self.univariate)
        with pytest.raises(
            ValueError,
            match="The underlying model does not support `future_covariates`.",
        ):
            model.predict(n=self.n_pred, future_covariates=self.future_covariates)

    def test_unsupported_predict_likelihood_parameters(self):
        """Test that predict_likelihood_parameters=True raises for models without likelihood."""
        model = self.helper_setup_model(ExponentialSmoothing, {})
        assert not model.supports_likelihood_parameter_prediction

        model.fit(self.univariate)
        with pytest.raises(
            ValueError,
            match="The underlying model does not support `predict_likelihood_parameters`.",
        ):
            model.predict(n=self.n_pred, predict_likelihood_parameters=True)

    def test_unsupported_series_prediction(self):
        """Test that passing series to predict raises for non-transferable models."""
        model = self.helper_setup_model(ExponentialSmoothing, {})
        assert not model.supports_transferable_series_prediction

        model.fit(self.univariate)
        with pytest.raises(
            ValueError,
            match="The underlying model does not support `series` for prediction.",
        ):
            model.predict(n=self.n_pred, series=self.univariate)

    def test_property_delegation(self):
        """Test that wrapper properties correctly delegate to the base model."""
        model = self.helper_setup_model(ExponentialSmoothing, {})
        base = model._base_model

        assert model.supports_multivariate is True
        assert model.supports_past_covariates == base.supports_past_covariates
        assert model.supports_static_covariates == base.supports_static_covariates
        assert model.extreme_lags == base.extreme_lags
        assert model._model_encoder_settings == base._model_encoder_settings
        assert model._supports_range_index == base._supports_range_index
        assert (
            model._supports_non_retrainable_historical_forecasts
            == base._supports_non_retrainable_historical_forecasts
        )

    @staticmethod
    def helper_setup_model(model, model_kwargs=None, model_setup="class"):
        if model_setup == "object":
            model, model_kwargs = model(**(model_kwargs or {})), None
        elif model_setup == "string":
            model = model if isinstance(model, str) else model.__name__
        return MultivariateModel(model=model, model_kwargs=model_kwargs)

    def helper_test_predict_with_base_model(
        self,
        model,
        series: TimeSeries,
        future_covariates=None,
        **predict_kwargs,
    ):
        preds = self.helper_trained_model_predictions(
            model, self.n_pred, series, future_covariates, **predict_kwargs
        )
        assert isinstance(preds, TimeSeries)
        assert preds.n_components == series.n_components

        # Make sure that the compound prediction is the same as the individual predictions
        individual_preds = self.helper_trained_individual_model_predictions(
            model, self.n_pred, series, future_covariates, **predict_kwargs
        )

        assert preds.time_index.equals(individual_preds.time_index)
        assert preds.components.equals(individual_preds.components)
        np.testing.assert_array_almost_equal(
            preds.all_values(), individual_preds.all_values()
        )
        return preds

    @staticmethod
    def helper_trained_model_predictions(
        model, n, series, future_covariates, **predict_kwargs
    ):
        fit_kwargs = {}
        if model.supports_future_covariates:
            fit_kwargs["future_covariates"] = future_covariates
        model.fit(series, **fit_kwargs)

        if model._base_model.supports_multivariate:
            expected_n_models = 1
        else:
            expected_n_models = series.n_components
        assert len(model._models) == expected_n_models

        if model.supports_future_covariates:
            predict_kwargs["future_covariates"] = future_covariates
        if model.supports_transferable_series_prediction:
            predict_kwargs["series"] = series
        return model.predict(n=n, **predict_kwargs)

    @staticmethod
    def helper_trained_individual_model_predictions(
        model, n, series, future_covariates, **predict_kwargs
    ):
        base_model = model._base_model.untrained_model()

        fit_kwargs = (
            {"future_covariates": future_covariates}
            if base_model.supports_future_covariates
            else {}
        )
        if base_model.supports_multivariate:
            return (
                base_model.untrained_model()
                .fit(series, **fit_kwargs)
                .predict(n=n, **predict_kwargs)
            )

        predictions = []
        for component in range(series.n_components):
            single_series = series.univariate_component(component)
            model = base_model.untrained_model()
            model.fit(single_series, **fit_kwargs)
            predictions.append(model.predict(n=n, **predict_kwargs))
        return concatenate(predictions, axis=1)
