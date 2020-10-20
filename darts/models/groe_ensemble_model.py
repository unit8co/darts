"""
GROE ensemble model
-------------------------
"""

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_if
from ..models.forecasting_model import ForecastingModel
from ..models.ensemble_model import EnsembleModel
from ..metrics import smape
from typing import List, Callable
import numpy as np

from ..utils.cross_validation import generalized_rolling_origin_evaluation as groe

logger = get_logger(__name__)


class GROEEnsembleModel(EnsembleModel):
    def __init__(self,
                 models: List[ForecastingModel],
                 metric: Callable[[TimeSeries, TimeSeries], float] = smape,
                 n_prediction_steps: int = 6,
                 **groe_kwargs):
        """
        Implementation of an EnsembleModel using GROE to compute the weights.

        The weights are a function of the loss function of the GROE cross-validation scheme.
        The weights for each constituent model's output are computed as the inverse of the
        value of the loss function obtained by applying GROE on that model, normalized such
        that all weights add up to 1.

        Disclaimer: This model constitutes an experimental attempt at implementing ensembling using
        generalized rolling window evaluation.

        Parameters
        ----------
        models
            List of forecasting models, whose predictions to ensemble.
        metric
            Metric function used for the GROE cross-validation function.
        n_prediction_steps
            The maximum number of predictions (ie. max number of calls to predict())
            performed by the GROE function.
        groe_kwargs
            Any additional args passed to the GROE function
        """
        super().__init__(models)
        self.metric = metric
        self.n_prediction_steps = n_prediction_steps
        self.groe_kwargs = groe_kwargs
        self.criteria = None
        self.weights = None

    def fit(self, training_series: TimeSeries):
        super().fit(training_series)
        self.criteria = []
        for model in self.models:
            self.criteria.append(groe(self.training_series,
                                      model,
                                      self.metric,
                                      n_prediction_steps=self.n_prediction_steps,
                                      **self.groe_kwargs))

        raise_if(np.inf in self.criteria,
                 "Cannot evaluate one of the models on this TimeSeries. Choose another fallback method",
                 logger)

        if 0. in self.criteria:
            self.weights = np.zeros(len(self.criteria))
            self.weights[self.criteria.index(0.)] = 1
        else:
            scores = 1 / np.array(self.criteria)
            self.weights = scores / scores.sum()

    def ensemble(self, predictions: List[TimeSeries]):
        return sum(map(lambda ts, weight: ts * weight, predictions, self.weights))
