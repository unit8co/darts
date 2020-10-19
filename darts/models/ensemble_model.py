"""
Ensemble model
--------------
"""

from abc import ABC, abstractmethod
from typing import List

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not
from ..models.forecasting_model import ForecastingModel

logger = get_logger(__name__)


class EnsembleModel(ABC):
    """
    Abstract base class for ensemble models
    """
    def __init__(self, models: List[ForecastingModel]):
        raise_if_not(isinstance(models, list) and models, "Must give at least one model")
        raise_if_not(all(isinstance(model, ForecastingModel) for model in models),
                     "All models must be instances of darts.models.ForecastingModel")
        self.models = models
        self.train_ts = None
        self._fit_called = False

    def fit(self, train_ts: TimeSeries) -> None:
        self.train_ts = train_ts
        for model in self.models:
            model.fit(self.train_ts)
        self._fit_called = True

    def predict(self, n: int) -> TimeSeries:
        if not self._fit_called:
            raise_log(Exception('fit() must be called before predict()'), logger)
        predictions = []
        for model in self.models:
            predictions.append(model.predict(n))
        return self.ensemble(predictions)

    @abstractmethod
    def ensemble(self, predictions: List[TimeSeries]):
        """
        This function should be implemented to return the ensembled prediction of the `EnsembleModel` instance.

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
