"""
Baseline Models
---------------

A collection of simple benchmark models for univariate series.
"""

from typing import List, Optional, Sequence, Union

import numpy as np

from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.ensemble_model import EnsembleModel
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    LocalForecastingModel,
)
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class NaiveMean(LocalForecastingModel):
    def __init__(self):
        """Naive Mean Model

        This model has no parameter, and always predicts the
        mean value of the training series.
        """
        super().__init__()
        self.mean_val = None

    def fit(self, series: TimeSeries):
        super().fit(series)

        self.mean_val = np.mean(series.values(copy=False), axis=0)
        return self

    def predict(self, n: int, num_samples: int = 1, verbose: bool = False):
        super().predict(n, num_samples)
        forecast = np.tile(self.mean_val, (n, 1))
        return self._build_forecast_series(forecast)


class NaiveSeasonal(LocalForecastingModel):
    def __init__(self, K: int = 1):
        """Naive Seasonal Model

        This model always predicts the value of `K` time steps ago.
        When `K=1`, this model predicts the last value of the training set.
        When `K>1`, it repeats the last `K` values of the training set.

        Parameters
        ----------
        K
            the number of last time steps of the training set to repeat
        """
        super().__init__()
        self.last_k_vals = None
        self.K = K

    @property
    def min_train_series_length(self):
        return max(self.K, 3)

    def fit(self, series: TimeSeries):
        super().fit(series)

        raise_if_not(
            len(series) >= self.K,
            f"The time series requires at least K={self.K} points",
            logger,
        )
        self.last_k_vals = series.values(copy=False)[-self.K :, :]
        return self

    def predict(self, n: int, num_samples: int = 1, verbose: bool = False):
        super().predict(n, num_samples)
        forecast = np.array([self.last_k_vals[i % self.K, :] for i in range(n)])
        return self._build_forecast_series(forecast)


class NaiveDrift(LocalForecastingModel):
    def __init__(self):
        """Naive Drift Model

        This model fits a line between the first and last point of the training series,
        and extends it in the future. For a training series of length :math:`T`, we have:

        .. math:: \\hat{y}_{T+h} = y_T + h\\left( \\frac{y_T - y_1}{T - 1} \\right)
        """
        super().__init__()

    def fit(self, series: TimeSeries):
        super().fit(series)
        assert series.n_samples == 1, "This model expects deterministic time series"

        series = self.training_series
        return self

    def predict(self, n: int, num_samples: int = 1, verbose: bool = False):
        super().predict(n, num_samples)
        first, last = (
            self.training_series.first_values(),
            self.training_series.last_values(),
        )
        slope = (last - first) / (len(self.training_series) - 1)
        last_value = last + slope * n
        forecast = np.linspace(last, last_value, num=n + 1)[1:]
        return self._build_forecast_series(forecast)


class NaiveMovingAverage(LocalForecastingModel):
    def __init__(self, input_chunk_length: int = 1):
        """Naive Moving Average Model

        This model forecasts using an auto-regressive moving average (ARMA).

        Parameters
        ----------
        input_chunk_length
            The size of the sliding window used to calculate the moving average
        """
        super().__init__()
        self.input_chunk_length = input_chunk_length
        self.rolling_window = None

    @property
    def min_train_series_length(self):
        return self.input_chunk_length

    def __str__(self):
        return f"NaiveMovingAverage({self.input_chunk_length})"

    def fit(self, series: TimeSeries):
        super().fit(series)
        raise_if_not(
            series.is_deterministic,
            "This model expects deterministic time series",
            logger,
        )

        self.rolling_window = series[-self.input_chunk_length :].values(copy=False)
        return self

    def predict(self, n: int, num_samples: int = 1, verbose: bool = False):
        super().predict(n, num_samples)

        predictions_with_observations = np.concatenate(
            (self.rolling_window, np.zeros(shape=(n, self.rolling_window.shape[1]))),
            axis=0,
        )
        rolling_sum = sum(self.rolling_window)

        chunk_length = self.input_chunk_length
        for i in range(chunk_length, chunk_length + n):
            prediction = rolling_sum / chunk_length
            predictions_with_observations[i] = prediction
            lost_value = predictions_with_observations[i - chunk_length]
            rolling_sum += prediction - lost_value
        return self._build_forecast_series(predictions_with_observations[-n:])


class NaiveEnsembleModel(EnsembleModel):
    def __init__(
        self,
        forecasting_models: List[ForecastingModel],
        retrain_forecasting_models: bool = True,
        show_warnings: bool = True,
    ):
        """Naive combination model

        Naive implementation of `EnsembleModel`
        Returns the average of all predictions of the constituent models

        If `future_covariates` or `past_covariates` are provided at training or inference time,
        they will be passed only to the models supporting them.

        Parameters
        ----------
        forecasting_models
            List of forecasting models whose predictions to ensemble
        retrain_forecasting_models
            If set to `False`, the `forecasting_models` are not retrained when calling `fit()` (only supported
            if all the `forecasting_models` are pretrained `GlobalForecastingModels`). Default: ``True``.

            .. note::
                if `forecasting_models` are already fitted and `retrain_forecasting_models=False`, `predict()`
                can be called directly by the `NaiveEnsembleModel` (without calling `fit()`).
            ..
        show_warnings
            Whether to show warnings related to models covariates support.
        """
        super().__init__(
            forecasting_models=forecasting_models,
            train_num_samples=1,
            train_samples_reduction=None,
            retrain_forecasting_models=retrain_forecasting_models,
            show_warnings=show_warnings,
        )

        # ensemble model initialised with trained global models can directly call predict()
        if self.all_trained and not retrain_forecasting_models:
            self._fit_called = True

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ):
        super().fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        if self.retrain_forecasting_models:
            # Some models may need to be 'reset' to allow being retrained from scratch, especially torch-based models
            self.forecasting_models: List[ForecastingModel] = [
                model.untrained_model() for model in self.forecasting_models
            ]
            for model in self.forecasting_models:
                model._fit_wrapper(
                    series=series,
                    past_covariates=past_covariates
                    if model.supports_past_covariates
                    else None,
                    future_covariates=future_covariates
                    if model.supports_future_covariates
                    else None,
                )
        # update training_series attribute to make predict() behave as expected
        else:
            for model in self.forecasting_models:
                model.training_series = (
                    series if isinstance(series, TimeSeries) else None
                )

        return self

    def ensemble(
        self,
        predictions: Union[TimeSeries, Sequence[TimeSeries]],
        series: Optional[Sequence[TimeSeries]] = None,
        num_samples: int = 1,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        def take_average(prediction: TimeSeries) -> TimeSeries:
            # average across the components, keep n_samples, rename components
            return prediction.mean(axis=1).with_columns_renamed(
                "components_mean", prediction.components[0]
            )

        if isinstance(predictions, Sequence):
            return [take_average(p) for p in predictions]
        else:
            return take_average(predictions)
