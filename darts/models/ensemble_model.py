"""
Ensemble Model Base Class
-------------------------
"""

from abc import abstractmethod
from typing import List, Optional, Union, Sequence
from functools import reduce

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_if_not, raise_if
from ..models.forecasting_model import ForecastingModel, GlobalForecastingModel

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
    def __init__(self, models: Union[List[ForecastingModel], List[GlobalForecastingModel]]):
        raise_if_not(isinstance(models, list) and models,
                     "Cannot instantiate EnsembleModel with an empty list of models",
                     logger)

        is_local_ensemble = all(isinstance(model, ForecastingModel) and not isinstance(model, GlobalForecastingModel)
                                for model in models)
        self.is_global_ensemble = all(isinstance(model, GlobalForecastingModel) for model in models)

        raise_if_not(is_local_ensemble or self.is_global_ensemble,
                     "All models must be instances of the same type, either darts.models.ForecastingModel"
                     "or darts.models.GlobalForecastingModel",
                     logger)
        super().__init__()
        self.models: Union[List[ForecastingModel], List[GlobalForecastingModel]] = models

    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None) -> None:
        """
        Fits the model on the provided series.
        Note that `EnsembleModel.fit()` does NOT call `fit()` on each of its constituent forecasting models.
        It is left to classes inheriting from EnsembleModel to do so appropriately when overriding `fit()`
        """
        raise_if(not self.is_global_ensemble and not isinstance(series, TimeSeries),
                 "All models are of type darts.models.ForecastingModel which do not support covariates.",
                 logger
                 )
        raise_if(not self.is_global_ensemble and covariates is not None,
                 "All models are of type darts.models.ForecastingModel which do not support covariates.",
                 logger
                 )
        super().fit(series, covariates)

    def _ts_sequence_to_multivariate_ts(self, ts_sequence: Sequence[TimeSeries]) -> TimeSeries:
        if isinstance(ts_sequence, Sequence):
            return reduce(lambda a, b: a.stack(b), ts_sequence)
        else:
            return ts_sequence

    def predict(self,
                n: int,
                series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                num_samples: int = 1,
                ) -> Union[TimeSeries, Sequence[TimeSeries]]:

        super().predict(n, series, covariates, num_samples)

        if self.is_global_ensemble:
            predictions = self._ts_sequence_to_multivariate_ts(
                self.models[0].predict(n, series, covariates, num_samples))
        else:
            predictions = self.models[0].predict(n, num_samples)

        if len(self.models) > 1:
            for model in self.models[1:]:
                if self.is_global_ensemble:
                    prediction = self._ts_sequence_to_multivariate_ts(
                        model.predict(n, series, covariates, num_samples))
                else:
                    prediction = model.predict(n, num_samples)

                predictions = predictions.stack(prediction)

        return self.ensemble(predictions)

    @abstractmethod
    def ensemble(self, predictions: TimeSeries) -> TimeSeries:
        """
        Defines how to ensemble the individual models' predictions to produce a single prediction.

        Parameters
        ----------
        predictions
            Individual predictions to ensemble

        Returns
        -------
        TimeSeries
            The predicted `TimeSeries` obtained by ensembling the individual predictions
        """
        pass

    @property
    def min_train_series_length(self) -> int:
        return max(model.min_train_series_length for model in self.models)
