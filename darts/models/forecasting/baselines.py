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
    GlobalForecastingModel,
)
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class NaiveMean(ForecastingModel):
    def __init__(self):
        """Naive Mean Model

        This model has no parameter, and always predicts the
        mean value of the training series.
        """
        super().__init__()
        self.mean_val = None

    def __str__(self):
        return "Naive mean predictor model"

    def fit(self, series: TimeSeries):
        super().fit(series)
        self.mean_val = np.mean(series.univariate_values())
        return self

    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        forecast = np.array([self.mean_val for _ in range(n)])
        return self._build_forecast_series(forecast)


class NaiveSeasonal(ForecastingModel):
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

    def __str__(self):
        return f"Naive seasonal model, with K={self.K}"

    def fit(self, series: TimeSeries):
        super().fit(series)
        raise_if_not(
            len(series) >= self.K,
            f"The time series requires at least K={self.K} points",
            logger,
        )
        self.last_k_vals = series.univariate_values()[-self.K :]
        return self

    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        forecast = np.array([self.last_k_vals[i % self.K] for i in range(n)])
        return self._build_forecast_series(forecast)


class NaiveDrift(ForecastingModel):
    def __init__(self):
        """Naive Drift Model

        This model fits a line between the first and last point of the training series,
        and extends it in the future. For a training series of length :math:`T`, we have:

        .. math:: \\hat{y}_{T+h} = y_T + h\\left( \\frac{y_T - y_1}{T - 1} \\right)
        """
        super().__init__()

    def __str__(self):
        return "Naive drift model"

    def fit(self, series: TimeSeries):
        super().fit(series)
        series = self.training_series
        return self

    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        first, last = (
            self.training_series.first_value(),
            self.training_series.last_value(),
        )
        slope = (last - first) / (len(self.training_series) - 1)
        last_value = last + slope * n
        forecast = np.linspace(last, last_value, num=n + 1)[1:]
        return self._build_forecast_series(forecast)


class NaiveEnsembleModel(EnsembleModel):
    def __init__(
        self, models: Union[List[ForecastingModel], List[GlobalForecastingModel]]
    ):
        """Naive combination model

        Naive implementation of `EnsembleModel`
        Returns the average of all predictions of the constituent models
        """
        super().__init__(models)

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
        for model in self.models:
            if self.is_global_ensemble:
                kwargs = dict(series=series)
                if model.uses_past_covariates:
                    kwargs["past_covariates"] = past_covariates
                if model.uses_future_covariates:
                    kwargs["future_covariates"] = future_covariates
                model.fit(**kwargs)
            else:
                model.fit(series=series)

        return self

    def ensemble(
        self,
        predictions: Union[TimeSeries, Sequence[TimeSeries]],
        series: Optional[Sequence[TimeSeries]] = None,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        if isinstance(predictions, Sequence):
            return [
                TimeSeries.from_series(p.pd_dataframe().sum(axis=1) / len(self.models))
                for p in predictions
            ]
        else:
            return TimeSeries.from_series(
                predictions.pd_dataframe().sum(axis=1) / len(self.models)
            )
