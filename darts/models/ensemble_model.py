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
    Abstract base class for ensemble models.
    Ensemble models take in a list of models and ensemble their predictions to make a single
    one according to the rule defined by their `ensemble()` method.
    """
    def __init__(self, models: List[ForecastingModel]):
        raise_if_not(isinstance(models, list) and models,
                     "Cannot instantiate EnsembleModel with an empty list of models",
                     logger)
        raise_if_not(all(isinstance(model, ForecastingModel) for model in models),
                     "All models must be instances of darts.models.ForecastingModel",
                     logger)

        self.models = models
        self.training_series = None
        self._fit_called = False

    def fit(self, training_series: TimeSeries) -> None:
        self.training_series = training_series

        for model in self.models:
            model.fit(self.training_series)

        self._fit_called = True

    def predict(self, n: int) -> TimeSeries:
        if not self._fit_called:
            raise_log(Exception('fit() must be called before predict()'), logger)

        predictions = []
        for model in self.models:
            predictions.append(model.predict(n))

        return self.ensemble(predictions)

    @abstractmethod
    def ensemble(self, predictions: List[TimeSeries]) -> TimeSeries:
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
