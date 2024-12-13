"""
Ensemble scikit-learn aggregator
--------------------------------
"""

from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import BaseEnsemble

from darts import TimeSeries
from darts.ad.aggregators.aggregators import FittableAggregator
from darts.logging import raise_if_not


class EnsembleSklearnAggregator(FittableAggregator):
    def __init__(self, model: BaseEnsemble) -> None:
        """Ensemble scikit-learn aggregator

        Aggregator wrapped around the sklearn ensemble model `sklearn ensemble model
        <https://scikit-learn.org/stable/modules/ensemble.html>`_.

        Parameters
        ----------
        model
            The sklearn ensemble model.
        """
        raise_if_not(
            isinstance(model, BaseEnsemble),
            f"Scorer is expecting a model of type BaseEnsemble (from sklearn ensemble), \
            found type {type(model)}.",
        )

        self.model = model
        super().__init__()

    def __str__(self) -> str:
        return "EnsembleSklearnAggregator: {}".format(
            self.model.__str__().split("(")[0]
        )

    def _fit_core(self, anomalies: Sequence[np.ndarray], series: Sequence[np.ndarray]):
        X = np.concatenate(series, axis=0)
        y = np.concatenate(
            [s.flatten() for s in anomalies],
            axis=0,
        )
        self.model.fit(y=y, X=X)

    def _predict_core(self, series: Sequence[TimeSeries]) -> Sequence[TimeSeries]:
        # assume that parallelization occurs at sklearn model level
        return [
            TimeSeries.from_times_and_values(
                s.time_index,
                self.model.predict(s.values(copy=False)),
            )
            for s in series
        ]
