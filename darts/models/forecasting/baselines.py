"""
Baseline Models
---------------

A collection of simple benchmark models for univariate series.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from darts.logging import get_logger, raise_if, raise_log
from darts.models.forecasting.ensemble_model import EnsembleModel
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    GlobalForecastingModel,
)
from darts.timeseries import TimeSeries
from darts.utils.utils import seq2series, series2seq

logger = get_logger(__name__)


class BaselineModel(GlobalForecastingModel, ABC):
    def __init__(self):
        super().__init__(add_encoders=None)

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> "BaselineModel":
        """Fit/train the model on a (or potentially multiple) series.
        This method is only implemented for naive baseline models to provide a unified fit/predict API with other
        forecasting models.

        The models are not really trained on the input, but they store the training `series` in case only a single
        `TimeSeries` was passed. This allows to call `predict()` without having to pass the single `series`.

        All baseline models compute the forecasts for each series directly when calling `predict()`.

        Parameters
        ----------
        series
            One or several target time series. The model will be trained to forecast these time series.
            The series may or may not be multivariate, but if multiple series are provided
            they must have the same number of components.

        Returns
        -------
        self
            Fitted model.
        """
        series = seq2series(series)
        super().fit(series=series)
        self._fit_model(series=series)

    @abstractmethod
    def _fit_model(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> None:
        """Must implement the fit logic and checks for each sub model."""
        pass

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Forecasts values for `n` time steps after the end of the series.

        If :func:`fit()` has been called with only one ``TimeSeries`` as argument, then the `series` argument of
        this function is optional, and it will simply produce the next `horizon` time steps forecast.

        If :func:`fit()` has been called with `series` specified as a ``Sequence[TimeSeries]`` (i.e., the model
        has been trained on multiple time series), the `series` argument must be specified.

        When the `series` argument is specified, this function will compute the next `n` time steps forecasts
        for the simple series (or for each series in the sequence) given by `series`.

        Parameters
        ----------
        n
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        series
            The series whose future values will be predicted.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            If `series` is not specified, this function returns a single time series containing the `n`
            next points after then end of the training series.
            If `series` is given and is a simple ``TimeSeries``, this function returns the `n` next points
            after the end of `series`.
            If `series` is given and is a sequence of several time series, this function returns
            a sequence where each element contains the corresponding `n` points forecasts.
        """
        if series is None:
            # then there must be a single TS, and that was saved in super().fit as self.training_series
            if self.training_series is None:
                raise_log(
                    ValueError(
                        "Input `series` must be provided. This is the result either from fitting on multiple series, "
                        "or from not having fit the model yet."
                    ),
                    logger,
                )
            series = self.training_series

        called_with_single_series = True if isinstance(series, TimeSeries) else False

        series = series2seq(series)
        super().predict(n=n, series=series, num_samples=num_samples)
        predictions = self._predict(n=n, series=series, num_samples=num_samples)
        return predictions[0] if called_with_single_series else predictions

    @abstractmethod
    def _predict(
        self, n: int, series: Sequence[TimeSeries] = None, num_samples: int = 1
    ) -> Sequence[TimeSeries]:
        pass

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
        return -self.min_train_series_length, -1, None, None, None, None

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
        """Baseline models do not support covariates and therefore also no encoders."""
        return None, None, False, False, None, None

    def supports_multivariate(self) -> bool:
        return True


class NaiveMean(BaselineModel):
    def __init__(self):
        """Naive Mean Model

        This model has no parameter, and always predicts the
        mean value of the training series.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import NaiveMean
        >>> series = AirPassengersDataset().load()
        >>> model = NaiveMean()
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[280.29861111],
              [280.29861111],
              [280.29861111],
              [280.29861111],
              [280.29861111],
              [280.29861111]])
        """
        super().__init__()

    def _fit_model(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> None:
        super()._fit_model(series)

    def _predict(
        self, n: int, series: Sequence[TimeSeries] = None, num_samples: int = 1
    ) -> Sequence[TimeSeries]:
        predictions = []
        for series_ in series:
            mean_val = np.mean(series_.values(copy=False), axis=0)
            predictions.append(
                self._build_forecast_series(
                    points_preds=np.tile(mean_val, (n, 1)),
                    input_series=series_,
                )
            )
        return predictions


class NaiveSeasonal(BaselineModel):
    def __init__(self, K: int = 1):
        """Naive Seasonal Model

        This model always predicts the value of `K` time steps ago.
        When `K=1`, this model predicts the last value of the training set.
        When `K>1`, it repeats the last `K` values of the training set.

        Parameters
        ----------
        K
            the number of last time steps of the training set to repeat

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import NaiveSeasonal
        >>> series = AirPassengersDataset().load()
        # prior analysis suggested seasonality of 12
        >>> model = NaiveSeasonal(K=12)
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[417.],
               [391.],
               [419.],
               [461.],
               [472.],
               [535.]])
        """
        super().__init__()
        self.K = K

    def _fit_model(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> None:

        super()._fit_model(series)

    def _predict(
        self,
        n: int,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        num_samples: int = 1,
    ):
        predictions = []
        for series_ in series:
            last_k_vals = series_.values(copy=False)[-self.K :, :]
            forecast = np.array([last_k_vals[i % self.K, :] for i in range(n)])

        predictions.append(
            self._build_forecast_series(
                points_preds=forecast,
                input_series=series_,
            )
        )
        return predictions


class NaiveDrift(BaselineModel):
    def __init__(self):
        """
        Naive Drift Model

        This model fits a line between the first and last point of the training series,
        and extends it in the future. For a training series of length :math:`T`, we have:

        .. math:: \\hat{y}_{T+h} = y_T + h\\left( \\frac{y_T - y_1}{T - 1} \\right)

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import NaiveDrift
        >>> series = AirPassengersDataset().load()
        >>> model = NaiveDrift()
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[434.23776224],
               [436.47552448],
               [438.71328671],
               [440.95104895],
               [443.18881119],
               [445.42657343]])
        """
        super().__init__()

    def _fit_model(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> None:
        super()._fit_model(series)

    def _predict(
        self, n: int, series: Sequence[TimeSeries] = None, num_samples: int = 1
    ) -> Sequence[TimeSeries]:

        super()._predict(n, num_samples)

        predictions = []

        for series_ in series:
            first, last = (
                series_.first_values(),
                series_.last_values(),
            )
            slope = (last - first) / (len(series_) - 1)
            last_value = last + slope * n

            predictions.append(
                self._build_forecast_series(
                    points_preds=np.linspace(last, last_value, num=n + 1)[1:],
                    input_series=series_,
                )
            )
        return predictions


class NaiveMovingAverage(BaselineModel):
    def __init__(self, input_chunk_length: int = 1):
        """Naive Moving Average Model

        This model forecasts using an auto-regressive moving average (ARMA).

        Parameters
        ----------
        input_chunk_length
            The size of the sliding window used to calculate the moving average

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import NaiveMovingAverage
        >>> series = AirPassengersDataset().load()
        # using the average of the last 6 months
        >>> model = NaiveMovingAverage(input_chunk_length=6)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[503.16666667],
               [483.36111111],
               [462.9212963 ],
               [455.40817901],
               [454.47620885],
               [465.22224366]])
        """
        super().__init__()
        self.input_chunk_length = input_chunk_length

    def _fit_model(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> None:
        super()._fit_model(series)

    def _predict(
        self, n: int, series: Sequence[TimeSeries] = None, num_samples: int = 1
    ) -> Sequence[TimeSeries]:

        super()._predict(n, num_samples)

        series = series2seq(series)
        predictions = []

        for series_ in series:
            rolling_window = series_[-self.input_chunk_length :].values(copy=False)
            predictions_with_observations = np.concatenate(
                (
                    rolling_window,
                    np.zeros(shape=(n, rolling_window.shape[1])),
                ),
                axis=0,
            )
            rolling_sum = sum(rolling_window)

            chunk_length = self.input_chunk_length
            for i in range(chunk_length, chunk_length + n):
                prediction = rolling_sum / chunk_length
                predictions_with_observations[i] = prediction
                lost_value = predictions_with_observations[i - chunk_length]
                rolling_sum += prediction - lost_value

            predictions.append(
                self._build_forecast_series(
                    points_preds=predictions_with_observations[-n:],
                    input_series=series_,
                )
            )
        return predictions


class NaiveEnsembleModel(EnsembleModel):
    def __init__(
        self,
        forecasting_models: List[ForecastingModel],
        train_forecasting_models: bool = True,
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
        train_forecasting_models
            Whether to train the `forecasting_models` from scratch. If `False`, the models are not trained when calling
            `fit()` and `predict()` can be called directly (only supported if all the `forecasting_models` are
            pretrained `GlobalForecastingModels`). Default: ``True``.
        show_warnings
            Whether to show warnings related to models covariates support.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import NaiveEnsembleModel, NaiveSeasonal, LinearRegressionModel
        >>> series = AirPassengersDataset().load()
        >>> # defining the ensemble
        >>> model = NaiveEnsembleModel([NaiveSeasonal(K=12), LinearRegressionModel(lags=4)])
        >>> model.fit(series)
        >>> pred = model.predict(6)
        >>> pred.values()
        array([[439.23152974],
               [431.41161602],
               [439.72888401],
               [453.70180806],
               [454.96757177],
               [485.16604194]])
        """
        super().__init__(
            forecasting_models=forecasting_models,
            train_num_samples=1,
            train_samples_reduction=None,
            train_forecasting_models=train_forecasting_models,
            show_warnings=show_warnings,
        )

        # ensemble model initialised with trained global models can directly call predict()
        if self.all_trained and not train_forecasting_models:
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
        if self.train_forecasting_models:
            for model in self.forecasting_models:
                model._fit_wrapper(
                    series=series,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                )

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
        n_forecasting_models = len(self.forecasting_models)
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
        likelihood = getattr(self.forecasting_models[0], "likelihood")
        if isinstance(likelihood, str):
            likelihood_n_params = self.forecasting_models[0].num_parameters
        else:  # Likelihood
            likelihood_n_params = likelihood.num_parameters
        n_forecasting_models = len(self.forecasting_models)
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
