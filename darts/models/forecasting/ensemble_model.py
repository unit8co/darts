"""
Ensemble Model Base Class
"""

from abc import abstractmethod
from functools import reduce
from typing import List, Optional, Sequence, Union

from darts.logging import get_logger, raise_if, raise_if_not
from darts.models.forecasting.forecasting_model import (
    GlobalForecastingModel,
    LocalForecastingModel,
)
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class EnsembleModel(GlobalForecastingModel):
    """
    Abstract base class for ensemble models.
    Ensemble models take in a list of forecasting models and ensemble their predictions
    to make a single one according to the rule defined by their `ensemble()` method.

    Parameters
    ----------
    models
        List of forecasting models whose predictions to ensemble
    """

    def __init__(
        self, models: Union[List[LocalForecastingModel], List[GlobalForecastingModel]]
    ):
        raise_if_not(
            isinstance(models, list) and models,
            "Cannot instantiate EnsembleModel with an empty list of models",
            logger,
        )

        is_local_ensemble = all(
            isinstance(model, LocalForecastingModel) for model in models
        )
        self.is_global_ensemble = all(
            isinstance(model, GlobalForecastingModel) for model in models
        )

        raise_if_not(
            is_local_ensemble or self.is_global_ensemble,
            "All models must be of the same type: either GlobalForecastingModel, or LocalForecastingModel.",
            logger,
        )

        raise_if(
            any([m._fit_called for m in models]),
            "Cannot instantiate EnsembleModel with trained/fitted models. "
            "Consider resetting all models with `my_model.untrained_model()`",
            logger,
        )

        super().__init__()
        self.models = models

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ):
        """
        Fits the model on the provided series.
        Note that `EnsembleModel.fit()` does NOT call `fit()` on each of its constituent forecasting models.
        It is left to classes inheriting from EnsembleModel to do so appropriately when overriding `fit()`
        """
        raise_if(
            not self.is_global_ensemble and not isinstance(series, TimeSeries),
            "The models are of type LocalForecastingModel, which does not support training on multiple series.",
            logger,
        )
        raise_if(
            not self.is_global_ensemble and past_covariates is not None,
            "The models are of type LocalForecastingModel, which does not support past covariates.",
            logger,
        )

        is_single_series = isinstance(series, TimeSeries)

        # check that if timeseries is single series, than covariates are as well and vice versa
        error = False

        if past_covariates is not None:
            error = is_single_series != isinstance(past_covariates, TimeSeries)

        if future_covariates is not None:
            error = is_single_series != isinstance(future_covariates, TimeSeries)

        raise_if(
            error,
            "Both series and covariates have to be either univariate or multivariate.",
            logger,
        )

        super().fit(series, past_covariates, future_covariates)

        return self

    def _stack_ts_seq(self, predictions):
        # stacks list of predictions into one multivariate timeseries
        return reduce(lambda a, b: a.stack(b), predictions)

    def _stack_ts_multiseq(self, predictions_list):
        # stacks multiple sequences of timeseries elementwise
        return [self._stack_ts_seq(ts_list) for ts_list in zip(*predictions_list)]

    def _model_encoder_settings(self):
        raise NotImplementedError(
            "Encoders are not supported by EnsembleModels. Instead add encoder to the underlying `models`."
        )

    def _make_multiple_predictions(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
    ):
        is_single_series = isinstance(series, TimeSeries) or series is None
        predictions = [
            model._predict_wrapper(
                n=n,
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                num_samples=num_samples,
            )
            for model in self.models
        ]
        return (
            self._stack_ts_seq(predictions)
            if is_single_series
            else self._stack_ts_multiseq(predictions)
        )

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        verbose: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:

        super().predict(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=num_samples,
            verbose=verbose,
        )

        predictions = self._make_multiple_predictions(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=num_samples,
        )
        return self.ensemble(predictions, series=series)

    @abstractmethod
    def ensemble(
        self,
        predictions: Union[TimeSeries, Sequence[TimeSeries]],
        series: Optional[Sequence[TimeSeries]] = None,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """
        Defines how to ensemble the individual models' predictions to produce a single prediction.

        Parameters
        ----------
        predictions
            Individual predictions to ensemble
        series
            Sequence of timeseries to predict on. Optional, since it only makes sense for sequences of timeseries -
            local models retain timeseries for prediction.

        Returns
        -------
        TimeSeries or Sequence[TimeSeries]
            The predicted ``TimeSeries`` or sequence of ``TimeSeries`` obtained by ensembling the individual predictions
        """
        pass

    @property
    def min_train_series_length(self) -> int:
        return max(model.min_train_series_length for model in self.models)

    def _is_probabilistic(self) -> bool:
        return all([model._is_probabilistic() for model in self.models])
