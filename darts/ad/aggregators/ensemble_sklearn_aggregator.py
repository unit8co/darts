"""
Ensemble scikit-learn aggregator
--------------------------------

Aggregator wrapped around the Ensemble model of sklearn.
`sklearn https://scikit-learn.org/stable/modules/ensemble.html`_.
"""

import numpy as np
from sklearn.ensemble import BaseEnsemble

from darts.ad.aggregators.aggregators import FittableAggregator
from darts.logging import raise_if_not


class EnsembleSklearnAggregator(FittableAggregator):
    def __init__(self, model) -> None:

        raise_if_not(
            isinstance(model, BaseEnsemble),
            f"Scorer is expecting a model of type BaseEnsemble (from sklearn ensemble), \
            found type {type(model)}.",
        )

        self.model = model
        super().__init__()

    def __str__(self):
        return "EnsembleSklearnAggregator: {}".format(
            self.model.__str__().split("(")[0]
        )

    def _fit_core(
        self, np_series: np.ndarray, np_actual_anomalies: np.ndarray
    ) -> np.ndarray:
        self.model.fit(np_series, np_actual_anomalies)

    def _predict_core(self, np_series: np.ndarray, width: int) -> np.ndarray:
        return self.models[width].predict(np_series)
