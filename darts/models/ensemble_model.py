"""
Ensemble Model Base Class
"""

from abc import abstractmethod
from typing import List, Optional

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_if_not
from ..models.forecasting_model import ForecastingModel

logger = get_logger(__name__)


class EnsembleModel(ForecastingModel):
    """
    Abstract base class for ensemble models.
    Ensemble models take in a list of forecasting models and ensemble their predictions
    to make a single one according to the rule defined by their `ensemble()` method.

    Parameters
    ----------
    models
        List of forecasting models whose predictions to ensemble
    """
    def __init__(self, models: List[ForecastingModel]):
        raise_if_not(isinstance(models, list) and models,
                     "Cannot instantiate EnsembleModel with an empty list of models",
                     logger)
        raise_if_not(all(isinstance(model, ForecastingModel) for model in models),
                     "All models must be instances of darts.models.ForecastingModel",
                     logger)
        super().__init__()
        self.models = models

    def fit(self, series: TimeSeries) -> None:
        """
        Fits the model on the provided series.
        Note that `EnsembleModel.fit()` does NOT call `fit()` on each of its constituent forecasting models.
        It is left to classes inheriting from EnsembleModel to do so appropriately when overriding `fit()`
        """
        super().fit(series)

    def predict(self,
                n: int,
                num_samples: int = 1) -> TimeSeries:
        super().predict(n, num_samples)

        predictions = self.models[0].predict(n=n, num_samples=num_samples)
        if len(self.models) > 1:
            for model in self.models[1:]:
                predictions = predictions.stack(model.predict(n=n, num_samples=num_samples))

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
