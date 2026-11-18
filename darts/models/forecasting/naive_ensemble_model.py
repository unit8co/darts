"""
Naive Ensemble Model
--------------------
"""

from collections.abc import Sequence

import numpy as np

from darts import TimeSeries
from darts.models.forecasting.ensemble_model import EnsembleModel
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.typing import TimeSeriesLike
from darts.utils.utils import random_method


class NaiveEnsembleModel(EnsembleModel):
    @random_method
    def __init__(
        self,
        forecasting_models: list[ForecastingModel],
        train_forecasting_models: bool = True,
        show_warnings: bool = True,
        random_state: int | None = None,
    ):
        """Naive combination model

        Depending on how :func:`predict()` is called, there are differences in how the forecasts of all
        `forecasting_models` (`F`) are combined:

        - Deterministic predictions (`num_samples == 1`): gives the component-wise average of `F`.
        - Stochastic predictions (`num_samples > 1`): samples from the mixture of stochastic `F`. Each output
          sample is a complete sample trajectory (all forecast time steps and all target components) of one model,
          drawn uniformly at random for every output sample.
        - Likelihood parameter predictions (`predict_likelihood_parameters=True`): gives the
          component- and parameter-wise average of `F`.

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
        random_state
            Controls the randomness for reproducible forecasting.

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

    @random_method
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

    def _target_average(
        self,
        prediction: TimeSeries,
        series: TimeSeries,
    ) -> TimeSeries:
        """Combine across the models, keep n_samples, rename components.

        Deterministic predictions (`n_samples == 1`) are averaged. Stochastic predictions (`n_samples > 1`) are
        sampled from the mixture of the models: output sample `s` is sample `s` (all time steps and components) of a
        model that is drawn uniformly at random. The base samples are i.i.d., so this is a valid mixture sample, the
        joint dependence across time steps and components is preserved, and values of discrete predictions are kept.
        """
        n_forecasting_models = len(self.forecasting_models)
        n_components = series.n_components
        n_samples = prediction.n_samples
        prediction_values = prediction.all_values(copy=False).reshape(
            prediction.n_timesteps, n_forecasting_models, n_components, n_samples
        )

        if prediction.n_samples > 1:
            # (time, model * component, sample) -> (time, model, component, sample)
            # uniformly draw the model that generates each output sample
            model_idx = np.random.randint(0, n_forecasting_models, size=n_samples)
            # pair model `model_idx[s]` with sample index `s`; the advanced indices are separated by a slice,
            # so numpy puts the sample axis first: (sample, time, component) -> (time, component, sample)
            prediction_values = prediction_values[:, model_idx, :, np.arange(n_samples)]
            prediction_values = prediction_values.transpose(1, 2, 0)
        else:
            prediction_values = prediction_values.mean(axis=1)

        return TimeSeries(
            times=prediction.time_index,
            values=prediction_values,
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
