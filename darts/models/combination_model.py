"""
Mean Combination model
-------------------------
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not
from ..models.forecasting_model import ForecastingModel

logger = get_logger(__name__)


class CombinationModel(ABC):
    """
    Base class for combination models.
    Acts as a constant weight combination model that averages all predictions from `models`
    """
    def __init__(self, models: List[ForecastingModel]):
        raise_if_not(isinstance(models, list) and models, "Must give at least one model")
        raise_if_not(all(isinstance(model, ForecastingModel) for model in models),
                     "All models must be instances of forecasting models from darts.models")
        self.models = models
        self.weights = np.ones(len(self.models)) / len(self.models)
        self.train_ts = None
        self.predictions = None
        self._fit_called = False

    def fit(self, train_ts: TimeSeries) -> None:
        self.train_ts = train_ts
        for model in self.models:
            model.fit(self.train_ts)
        self._fit_called = True

    def predict(self, n: int) -> TimeSeries:
        if not self._fit_called:
            raise_log(Exception('fit() must be called before predict()'), logger)
        self.predictions = []
        for model in self.models:
            self.predictions.append(model.predict(n))
        return self.combination_function()

    @abstractmethod
    def combination_function(self):
        """
        Given `self.predictions` obtained from calling `self.predict` and given other fields
        that a subclass of `CombinationModel` might contain, this function should be implemented
        to return the combined prediction of `CombinationModel` instance.
        This could involve a weighted sum of the individual predictions, computing the combined result
        using regression, or other means.

        Returns
        -------
        TimeSeries
            The predicted `TimeSeries` obtained by taking all the predictions from `self.models`
            into consideration.
        """
        pass
