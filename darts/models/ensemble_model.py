"""
Ensemble Model Base Class
-------------------------
"""

from abc import abstractmethod
from typing import List, Optional, Union, Sequence

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_if_not
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
        self.models = models

    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None) -> None:
        """
        Fits the model on the provided series.
        Note that `EnsembleModel.fit()` does NOT call `fit()` on each of its constituent forecasting models.
        It is left to classes inheriting from EnsembleModel to do so appropriately when overriding `fit()`
        """
        super().fit(series, covariates)

    def predict(self,
                n: int,
                series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                num_samples: int = 1,
                ) -> Union[TimeSeries, Sequence[TimeSeries]]:

        super().predict(n, series, covariates, num_samples)

        if self.is_global_ensemble:
            predictions = self.models[0].predict(n, series, covariates, num_samples)
        else:
            predictions = self.models[0].predict(n, num_samples)

        if len(self.models) > 1:
            for model in self.models[1:]:
                if self.is_global_ensemble:
                    prediction = model.predict(n, series, covariates, num_samples)
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
