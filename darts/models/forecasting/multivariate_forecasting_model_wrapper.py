"""
Multivariate forecasting model wrapper
-------------------------

A wrapper around local forecasting models to enable multivariate series training and forecasting. One model is trained for each component of the target series, independently of the others hence ignoring the potential interactions between its components.
"""
from typing import List, Optional, Tuple

from darts.logging import get_logger
from darts.models.forecasting.forecasting_model import (
    FutureCovariatesLocalForecastingModel,
    LocalForecastingModel,
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.timeseries import TimeSeries, concatenate, split_multivariate
from darts.utils.utils import seq2series

logger = get_logger(__name__)


class MultivariateForecastingModelWrapper(FutureCovariatesLocalForecastingModel):
    def __init__(self, model: LocalForecastingModel):
        """
        Wrapper around LocalForecastingModel allowing it to predict multivariate TimeSeries.

        A copy of the provided model will be trained on each of the components of the provided series separately.
        Parameters
        ----------
        model
            Model used to predict individual components
        """
        super().__init__()

        self.model: LocalForecastingModel = model
        self._trained_models: List[LocalForecastingModel] = []

    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        super()._fit(series, future_covariates)
        self._trained_models = []

        series = seq2series(series)
        for comp in series.components:
            self._trained_models.append(
                self.model.untrained_model().fit(
                    series=series.univariate_component(comp),
                    future_covariates=future_covariates if self.supports_future_covariates else None
                )
            )

        return self

    def predict(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        **kwargs,
    ) -> TimeSeries:
        # we override `predict()` to pass a non-None `series`, so that historic_future_covariates
        # will be passed to `_predict()` (some future covariates local models require it ex. Kalman)
        return self._predict(n, future_covariates, num_samples, **kwargs)

    def _predict(
        self,
        n: int,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        verbose: bool = False,
        **kwargs,
    ) -> TimeSeries:
        predictions = [
            model.predict(n=n, future_covariates=future_covariates)
            if model.supports_future_covariates
            else model.predict(n=n)
            for model in self._trained_models
        ]

        multivariate_series = concatenate(predictions, axis=1)
        return multivariate_series

    @property
    def extreme_lags(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
    ]:
        return self.model.extreme_lags

    @property
    def _model_encoder_settings(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        bool,
        bool,
        Optional[List[int]],
        Optional[List[int]],
    ]:
        return None, None, False, self.supports_future_covariates, None, None

    @property
    def supports_multivariate(self) -> bool:
        return True

    @property
    def supports_past_covariates(self) -> bool:
        return self.model.supports_past_covariates

    @property
    def supports_future_covariates(self) -> bool:
        return self.model.supports_future_covariates

    @property
    def supports_static_covariates(self) -> bool:
        return self.model.supports_static_covariates

    @property
    def _is_probabilistic(self) -> bool:
        """
        A MultivariateForecastingModelWrapper is probabilistic if the base_model
        is probabilistic
        """
        return self.model._is_probabilistic

    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        return isinstance(self.model, TransferableFutureCovariatesLocalForecastingModel)

    @property
    def _supress_generate_predict_encoding(self) -> bool:
        return isinstance(self.model, TransferableFutureCovariatesLocalForecastingModel)
