"""
GROE combination model
-------------------------
"""

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log
from ..models.forecasting_model import ForecastingModel
from ..models.combination_model import CombinationModel
from ..metrics import smape
from typing import List, Callable
import numpy as np

from ..utils.cross_validation import generalized_rolling_origin_evaluation as groe

logger = get_logger(__name__)


class GROECombinationModel(CombinationModel):
    def __init__(self, models: List[ForecastingModel], metrics: Callable[[TimeSeries, TimeSeries], float] = smape,
                 n_evaluation: int = 6, **groe_kwargs):
        """
        Implementation of a Combination Model using GROE to compute its weights.

        The weights are 1/loss function/sum

        Parameters
        ----------
        models
            List of forecasting models, whose predictions to combinate.
        metrics
            Metrics function used for the GROE cross-valisation function.
        n_evaluation
            Number of evaluation performed by the GROE function.
        groe_args
            Any additional args passed to the GROE function
        """
        super(GROECombinationModel, self).__init__(models)
        self.metrics = metrics
        self.n_evaluation = n_evaluation
        self.groe_kwargs = groe_kwargs
        self.criterion = None

    def update_groe_params(self, **groe_kwargs):
        if "n_evaluation" in groe_kwargs:
            self.n_evaluation = groe_kwargs.pop("n_evaluation")
        self.groe_kwargs = groe_kwargs

    def fit(self, train_ts: TimeSeries):
        super().fit(train_ts)
        self.criterion = []
        for model in self.models:
            self.criterion.append(groe(self.train_ts, model, self.metrics,
                                       n_evaluation=self.n_evaluation, **self.groe_kwargs))
        if np.inf in self.criterion:
            raise_log(ValueError("Impossible to evaluate one of the models on this TimeSeries. "
                                 "Choose another fallback method"), logger)
        if 0. in self.criterion:
            self.weights = np.zeros(len(self.criterion))
            self.weights[self.criterion.index(0.)] = 1.
        else:
            score = 1 / np.array(self.criterion)
            self.weights = score / score.sum()

    def predict(self, n: int):
        return super().predict(n)
