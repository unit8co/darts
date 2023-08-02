"""
Baseline Models
---------------

A collection of simple benchmark models for univariate series.
"""

from typing import List, Optional, Sequence, Union

import numpy as np

from darts.logging import get_logger, raise_if, raise_if_not
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

    @property
    def supports_multivariate(self) -> bool:
        return True

    def fit(self, series: TimeSeries):
        super().fit(series)

        self.mean_val = np.mean(series.values(copy=False), axis=0)
        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: bool = False,
    ):
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
    def supports_multivariate(self) -> bool:
        return True

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

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: bool = False,
    ):
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

    @property
    def supports_multivariate(self) -> bool:
        return True

    def fit(self, series: TimeSeries):
        super().fit(series)
        assert series.n_samples == 1, "This model expects deterministic time series"

        series = self.training_series
        return self

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: bool = False,
    ):
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
    def supports_multivariate(self) -> bool:
        return True

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

    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: bool = False,
    ):
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
        models: List[ForecastingModel],
        show_warnings: bool = True,
    ):
        """Naive combination model

        Naive implementation of `EnsembleModel`
        Returns the average of all predictions of the constituent models

        If `future_covariates` or `past_covariates` are provided at training or inference time,
        they will be passed only to the models supporting them.

        Parameters
        ----------
        models
            List of forecasting models whose predictions to ensemble
        show_warnings
            Whether to show warnings related to models covariates support.
        """
        super().__init__(
            models=models,
            train_num_samples=None,
            train_samples_reduction=None,
            show_warnings=show_warnings,
        )

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
            kwargs = dict(series=series)
            if model.supports_past_covariates:
                kwargs["past_covariates"] = past_covariates
            if model.supports_future_covariates:
                kwargs["future_covariates"] = future_covariates
            model.fit(**kwargs)

        return self

    def ensemble(
        self,
        predictions: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Average the `forecasting_models` predictions, component-wise"""
        raise_if(
            predict_likelihood_parameters
            and not self.supports_likelihood_parameter_prediction,
            "`predict_likelihood_parameters=True` is supported only if all the `forecasting_models` "
            "are probabilistic and fitting the same likelihood.",
            logger,
        )

        if series is None:
            series = self.training_series

        if isinstance(predictions, Sequence):
            return [
                self._target_average(p, ts)
                if not predict_likelihood_parameters
                else self._params_average(p, ts)
                for p, ts in zip(predictions, series)
            ]
        else:
            return (
                self._target_average(predictions, series)
                if not predict_likelihood_parameters
                else self._params_average(predictions, series)
            )

    def _target_average(self, prediction: TimeSeries, series: TimeSeries) -> TimeSeries:
        """Average across the components, keep n_samples, rename components"""
        n_forecasting_models = len(self.models)
        n_components = series.n_components
        prediction_values = prediction.all_values(copy=False)
        target_values = np.zeros(
            (prediction.n_timesteps, n_components, prediction.n_samples)
        )
        for idx_target in range(n_components):
            target_values[:, idx_target] = prediction_values[
                :,
                range(
                    idx_target,
                    n_forecasting_models * n_components,
                    n_components,
                ),
            ].mean(axis=1)

        return TimeSeries.from_times_and_values(
            times=prediction.time_index,
            values=target_values,
            freq=series.freq,
            columns=series.components,
            static_covariates=series.static_covariates,
            hierarchy=series.hierarchy,
        )

    def _params_average(self, prediction: TimeSeries, series: TimeSeries) -> TimeSeries:
        """Average across the components after grouping by likelihood parameter, rename components"""
        # str or torch Likelihood
        likelihood = getattr(self.models[0], "likelihood")
        if isinstance(likelihood, str):
            likelihood_n_params = self.models[0].num_parameters
        else:  # Likelihood
            likelihood_n_params = likelihood.num_parameters
        n_forecasting_models = len(self.models)
        n_components = series.n_components
        # aggregate across predictions [model1_param0, model1_param1, ..., modeln_param0, modeln_param1]
        prediction_values = prediction.values(copy=False)
        params_values = np.zeros(
            (prediction.n_timesteps, likelihood_n_params * n_components)
        )
        for idx_param in range(likelihood_n_params * n_components):
            params_values[:, idx_param] = prediction_values[
                :,
                range(
                    idx_param,
                    likelihood_n_params * n_forecasting_models * n_components,
                    likelihood_n_params * n_components,
                ),
            ].mean(axis=1)

        return TimeSeries.from_times_and_values(
            times=prediction.time_index,
            values=params_values,
            freq=series.freq,
            columns=prediction.components[: likelihood_n_params * n_components],
            static_covariates=None,
            hierarchy=None,
        )
