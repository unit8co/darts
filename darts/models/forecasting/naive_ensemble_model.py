"""
Naive Ensemble Model
--------------------
"""

from collections.abc import Sequence

import numpy as np

from darts import TimeSeries
from darts.logging import get_logger
from darts.models.forecasting.ensemble_model import EnsembleModel
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.typing import TimeSeriesLike

logger = get_logger(__name__)


class NaiveEnsembleModel(EnsembleModel):
    def __init__(
        self,
        forecasting_models: list[ForecastingModel],
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
        >>> print(pred.values())
        [[439.23152974]
         [431.41161602]
         [439.72888401]
         [453.70180806]
         [454.96757177]
         [485.16604194]]
        """
        super().__init__(
            forecasting_models=forecasting_models,
            ensemble_model=None,
            train_num_samples=1,
            train_samples_reduction=None,
            train_forecasting_models=train_forecasting_models,
            train_n_points=0,
            show_warnings=show_warnings,
        )

        # ensemble model initialised with trained global models can directly call predict()
        if self.all_trained and not train_forecasting_models:
            self._fit_called = True

    def fit(
        self,
        series: TimeSeriesLike,
        past_covariates: TimeSeriesLike | None = None,
        future_covariates: TimeSeriesLike | None = None,
        sample_weight: TimeSeriesLike | str | None = None,
        verbose: bool | None = None,
    ):
        super().fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            verbose=verbose,
        )
        if self.train_forecasting_models:
            for model in self.forecasting_models:
                model._fit_wrapper(
                    series=series,
                    past_covariates=(
                        past_covariates if model.supports_past_covariates else None
                    ),
                    future_covariates=(
                        future_covariates if model.supports_future_covariates else None
                    ),
                    sample_weight=sample_weight
                    if model.supports_sample_weight
                    else None,
                    verbose=verbose,
                )
        return self

    def ensemble(
        self,
        predictions: TimeSeriesLike,
        series: TimeSeriesLike,
        n: int,
        num_samples: int = 1,
        predict_likelihood_parameters: bool = False,
        random_state: int | None = None,
        verbose: bool | None = None,
    ) -> TimeSeriesLike:
        """Average the `forecasting_models` predictions, component-wise"""
        # at this point, if `predict_likelihood_parameters=True`, it's guaranteed
        # that all models use the same likelihood
        if isinstance(predictions, Sequence):
            return [
                (
                    self._target_average(p, ts)
                    if not predict_likelihood_parameters
                    else self._params_average(p, ts)
                )
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
        target_values = np.zeros((
            prediction.n_timesteps,
            n_components,
            prediction.n_samples,
        ))
        for idx_target in range(n_components):
            target_values[:, idx_target] = prediction_values[
                :,
                range(
                    idx_target,
                    n_forecasting_models * n_components,
                    n_components,
                ),
            ].mean(axis=1)

        return TimeSeries(
            times=prediction.time_index,
            values=target_values,
            components=series.components,
            copy=False,
            **series._attrs,
        )

    def _params_average(self, prediction: TimeSeries, series: TimeSeries) -> TimeSeries:
        """Average across the components after grouping by likelihood parameter, rename components"""
        likelihood = self.forecasting_models[0].likelihood
        likelihood_n_params = likelihood.num_parameters
        n_forecasting_models = len(self.forecasting_models)
        n_components = series.n_components
        # aggregate across predictions [model1_param0, model1_param1, ..., modeln_param0, modeln_param1]
        prediction_values = prediction.values(copy=False)
        params_values = np.zeros((
            prediction.n_timesteps,
            likelihood_n_params * n_components,
        ))
        for idx_param in range(likelihood_n_params * n_components):
            params_values[:, idx_param] = prediction_values[
                :,
                range(
                    idx_param,
                    likelihood_n_params * n_forecasting_models * n_components,
                    likelihood_n_params * n_components,
                ),
            ].mean(axis=1)

        return TimeSeries(
            times=prediction.time_index,
            values=params_values,
            components=prediction.components[: likelihood_n_params * n_components],
            static_covariates=None,
            hierarchy=None,
            metadata=prediction.metadata,
            copy=False,
        )
