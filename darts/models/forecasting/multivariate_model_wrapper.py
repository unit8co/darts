"""
Multivariate forecasting model wrapper
-------------------------

A wrapper around local forecasting models to enable multivariate series training and forecasting. One model is trained
for each component of the target series, independently of the others hence ignoring the potential interactions between
its components.
"""

import darts.models as darts_models
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import (
    LocalForecastingModel,
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.timeseries import TimeSeries, concatenate

logger = get_logger(__name__)


class MultivariateModelWrapper(TransferableFutureCovariatesLocalForecastingModel):
    def __init__(
        self,
        model: str | type[LocalForecastingModel] | LocalForecastingModel,
        model_kwargs: dict | None = None,
    ):
        """
        Wrapper for univariate LocalForecastingModel to enable multivariate series training and forecasting.

        A copy of the provided model will be trained independently on each component of the target series, ignoring the
        potential interactions.

        Parameters
        ----------
        model
            Name, class, or instance of the Darts
            :class:`~darts.models.forecasting.forecasting_model.LocalForecastingModel` to be used, e.g.,
            ``"ExponentialSmoothing"``, ``ExponentialSmoothing``, or ``ExponentialSmoothing()``.
            See all available models in :mod:`darts.models`.
        model_kwargs
            A dictionary of model parameters to initialize the model. Only effective when `model` is a string or
            class. Default: ``None``.
        """
        model_kwargs = model_kwargs or {}
        if isinstance(model, LocalForecastingModel):
            pass
        elif isinstance(model, str):
            try:
                model_class = getattr(darts_models, model)
            except AttributeError:
                raise_log(
                    ValueError(
                        f"Could not find a Darts LocalForecastingModel named `{model}` in `darts.models`."
                    ),
                    logger,
                )
            model = model_class(**model_kwargs)
        elif isinstance(model, type) and issubclass(model, LocalForecastingModel):
            model = model(**model_kwargs)
        else:
            raise_log(
                ValueError(
                    "`model` must be a valid Darts LocalForecastingModel name (str), class, or instance."
                ),
                logger,
            )

        super().__init__()
        self._model: LocalForecastingModel = model
        self._trained_models: list[LocalForecastingModel] = []

    def _fit(
        self,
        series: TimeSeries,
        future_covariates: TimeSeries | None = None,
        verbose: bool | None = None,
    ):
        super()._fit(series, future_covariates)
        self._trained_models = []

        for comp in series.components:
            comp = series.univariate_component(comp)
            component_model = (
                self._model.untrained_model().fit(
                    series=comp, future_covariates=future_covariates
                )
                if self._model.supports_future_covariates
                else self._model.untrained_model().fit(series=comp)
            )
            self._trained_models.append(component_model)

        return self

    def predict(
        self,
        n: int,
        series: TimeSeries | None = None,
        future_covariates: TimeSeries | None = None,
        num_samples: int = 1,
        **kwargs,
    ) -> TimeSeries:
        return self._predict(n, series, future_covariates, num_samples, **kwargs)

    def _predict(
        self,
        n: int,
        series: TimeSeries | None = None,
        future_covariates: TimeSeries | None = None,
        num_samples: int = 1,
        verbose: bool = False,
        **kwargs,
    ) -> TimeSeries:
        prediction_kwargs = {"n", n}
        if self._model.supports_transferable_series_prediction:
            prediction_kwargs["series"] = series
        if self._model.supports_future_covariates:
            prediction_kwargs["future_covariates"] = future_covariates
        if self._model.supports_probabilistic_prediction:
            prediction_kwargs["num_samples"] = num_samples

        predictions = [
            model.predict(**prediction_kwargs) for model in self._trained_models
        ]

        return concatenate(predictions, axis=1)

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
        return self._model.extreme_lags

    @property
    def _target_window_lengths(self) -> tuple[int, int]:
        return self._model._target_window_lengths

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
        return self._model._model_encoder_settings

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def supports_past_covariates(self) -> bool:
        return self._model.supports_past_covariates

    @property
    def supports_future_covariates(self) -> bool:
        return self._model.supports_future_covariates

    @property
    def supports_static_covariates(self) -> bool:
        return self._model.supports_static_covariates

    @property
    def supports_sample_weight(self) -> bool:
        return self._model.supports_sample_weight

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return self._model.supports_probabilistic_prediction

    @property
    def supports_transferable_series_prediction(self) -> bool:
        return self._model.supports_transferable_series_prediction

    @property
    def likelihood(self):
        return self._model.likelihood

    @property
    def _supports_range_index(self) -> bool:
        return self._model._supports_range_index

    @property
    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        return isinstance(
            self._model, TransferableFutureCovariatesLocalForecastingModel
        )

    @property
    def _supress_generate_predict_encoding(self) -> bool:
        return isinstance(
            self._model, TransferableFutureCovariatesLocalForecastingModel
        )
