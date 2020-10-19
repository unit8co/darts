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
    def __init__(self, models: List[ForecastingModel], metrics: Callable[[TimeSeries, TimeSeries], float] = smape,
                 n_evaluations: int = 6, **groe_kwargs):
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
        metrics
            Metrics function used for the GROE cross-validation function.
        n_evaluations
            Number of evaluation performed by the GROE function.
        groe_kwargs
            Any additional args passed to the GROE function
        """
        super(GROEEnsembleModel, self).__init__(models)
        self.metrics = metrics
        self.n_evaluations = n_evaluations
        self.groe_kwargs = groe_kwargs
        self.criteria = None
        self.weights = None

    def update_groe_params(self, **groe_kwargs):
        if "n_evaluations" in groe_kwargs:
            self.n_evaluations = groe_kwargs.pop("n_evaluations")
        self.groe_kwargs = groe_kwargs

    def fit(self, train_ts: TimeSeries):
        super().fit(train_ts)
        self.criteria = []
        for model in self.models:
            self.criteria.append(groe(self.train_ts, model, self.metrics,
                                       n_evaluations=self.n_evaluations, **self.groe_kwargs))

        raise_if(np.inf in self.criteria,
                 "Cannot evaluate one of the models on this TimeSeries. Choose another fallback method",
                 logger)

        if 0. in self.criteria:
            self.weights = np.zeros(len(self.criteria))
            self.weights[self.criteria.index(0.)] = 1.
        else:
            scores = 1 / np.array(self.criteria)
            self.weights = scores / scores.sum()

    def ensemble(self, predictions: List[TimeSeries]):
        return sum(map(lambda ts, weight: ts * weight, predictions, self.weights))
