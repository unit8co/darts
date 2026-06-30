"""
Multivariate forecasting model wrapper
-------------------------

A wrapper around any base forecasting model to enable multivariate series training and forecasting. One independent
model is trained per component of the target series. Interactions between components are not covered.
"""

from typing import Any

import darts.models as darts_models
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.timeseries import TimeSeries, concatenate

logger = get_logger(__name__)


class MultivariateModel(TransferableFutureCovariatesLocalForecastingModel):
    def __init__(
        self,
        model: str | type[ForecastingModel] | ForecastingModel,
        model_kwargs: dict | None = None,
    ):
        """
        Wrapper for any base ForecastingModel to enable multivariate forecasting support.

        One independent model-copy is trained per component of the target series. Interactions between components are
        not covered. Bypasses the multimodel setup if the base model already supports multivariate forecasting.

        Parameters
        ----------
        model
            Name, class, or instance of the Darts
            :class:`~darts.models.forecasting.forecasting_model.ForecastingModel` to be used, e.g.,
            ``"ExponentialSmoothing"``, ``ExponentialSmoothing``, or ``ExponentialSmoothing()``.
            See all available models in :mod:`darts.models`.
        model_kwargs
            A dictionary of model parameters to initialize the model. Only effective when `model` is a string or
            class. Default: ``None``.
        """
        model_kwargs = model_kwargs or {}
        if isinstance(model, ForecastingModel):
            pass
        elif isinstance(model, str):
            try:
                model_class = getattr(darts_models, model)
            except AttributeError:
                raise_log(
                    ValueError(
                        f"Could not find a `model` named '{model}' in `darts.models`."
                    )
                )
            model = model_class(**model_kwargs)
        elif isinstance(model, type) and issubclass(model, ForecastingModel):
            model = model(**model_kwargs)
        else:
            raise_log(
                ValueError(
                    "`model` must be a valid Darts `ForecastingModel` name (str), class, or instance."
                )
            )

        super().__init__(add_encoders=None)
        self._models: list[ForecastingModel] = [model]

    def fit(
        self,
        series: TimeSeries,
        future_covariates: TimeSeries | None = None,
        verbose: bool | None = None,
        **kwargs,
    ):
        fit_kwargs: dict[str, Any] = {"verbose": verbose}

        ForecastingModel.fit(self, series=series, **fit_kwargs)

        fit_kwargs = {**fit_kwargs, **kwargs}
        if self.supports_future_covariates:
            fit_kwargs["future_covariates"] = future_covariates
        elif future_covariates is not None:
            raise_log(
                ValueError(
                    "The underlying model does not support `future_covariates`."
                ),
            )

        base_model = self._base_model

        models: list[ForecastingModel] = list()
        if self._base_model.supports_multivariate or series.n_components == 1:
            models.append(base_model.untrained_model().fit(series=series, **fit_kwargs))
        else:
            for comp in series.components:
                comp = series.univariate_component(comp)
                component_model = base_model.untrained_model().fit(
                    series=comp, **fit_kwargs
                )
                models.append(component_model)
        self._models = models

        return self

    def predict(
        self,
        n: int,
        series: TimeSeries | None = None,
        future_covariates: TimeSeries | None = None,
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
        verbose: bool | None = None,
        show_warnings: bool = True,
        random_state: int | None = None,
        **kwargs,
    ) -> TimeSeries:
        predict_kwargs: dict[str, Any] = {
            "n": n,
            "num_samples": num_samples,
            "verbose": verbose,
            "show_warnings": show_warnings,
            "random_state": random_state,
        }

        ForecastingModel.predict(self, **predict_kwargs)

        predict_kwargs = {**predict_kwargs, **kwargs}
        if self.supports_likelihood_parameter_prediction:
            predict_kwargs["predict_likelihood_parameters"] = (
                predict_likelihood_parameters
            )
        elif predict_likelihood_parameters:
            raise_log(
                ValueError(
                    "The underlying model does not support `predict_likelihood_parameters`."
                ),
            )
        if self.supports_future_covariates:
            predict_kwargs["future_covariates"] = future_covariates
        elif future_covariates is not None:
            raise_log(
                ValueError(
                    "The underlying model does not support `future_covariates`."
                ),
            )
        if not self.supports_transferable_series_prediction:
            if series is not None:
                raise_log(
                    ValueError(
                        "The underlying model does not support `series` for prediction."
                    ),
                )

        if len(self._models) == 1:
            if series is not None:
                predict_kwargs["series"] = series
            return self._base_model.predict(**predict_kwargs)

        predictions = []
        for comp_idx, model in enumerate(self._models):
            if series is not None:
                predict_kwargs["series"] = series.univariate_component(comp_idx)
            predictions.append(model.predict(**predict_kwargs))
        return concatenate(predictions, axis=1)

    def _fit(*args, **kwargs):
        raise_log(NotImplementedError("`_fit` not implemented."))  # pragma: no cover

    def _predict(*args, **kwargs):
        raise_log(
            NotImplementedError("`_predict` not implemented.")
        )  # pragma: no cover

    @property
    def _base_model(self) -> ForecastingModel:
        return self._models[0]

    @property
    def extreme_lags(
        self,
    ) -> tuple[
        int | None,
        int | None,
        int | None,
        int | None,
        int | None,
        int | None,
        int,
    ]:
        return self._base_model.extreme_lags

    @property
    def _target_window_lengths(self) -> tuple[int, int]:
        return self._base_model._target_window_lengths

    @property
    def _model_encoder_settings(
        self,
    ) -> tuple[
        int | None,
        int | None,
        bool,
        bool,
        list[int] | None,
        list[int] | None,
    ]:
        return self._base_model._model_encoder_settings

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def supports_past_covariates(self) -> bool:
        return self._base_model.supports_past_covariates

    @property
    def supports_future_covariates(self) -> bool:
        return self._base_model.supports_future_covariates

    @property
    def supports_static_covariates(self) -> bool:
        return self._base_model.supports_static_covariates

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return self._base_model.supports_probabilistic_prediction

    @property
    def supports_transferable_series_prediction(self) -> bool:
        return self._base_model.supports_transferable_series_prediction

    @property
    def likelihood(self):
        return self._base_model.likelihood

    @property
    def _supports_range_index(self) -> bool:
        return self._base_model._supports_range_index

    @property
    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        return self._base_model._supports_non_retrainable_historical_forecasts
