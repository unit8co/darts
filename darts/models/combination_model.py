"""
Mean Combination model
-------------------------
"""

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not
from ..models.forecasting_model import ForecastingModel
from typing import List
import numpy as np

logger = get_logger(__name__)


class CombinationModel:
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
        self._fit_called = True

    def predict(self, n: int) -> TimeSeries:
        if not self._fit_called:
            raise_log(Exception('fit() must be called before predict()'), logger)
        self.predictions = []
        for model in self.models:
            model.fit(self.train_ts)
            self.predictions.append(model.predict(n))
        return sum(map(lambda ts, weight: ts * weight, self.predictions, self.weights))
