"""
Ensemble scikit-learn aggregator
--------------------------------

Aggregator wrapped around the Ensemble model of sklearn.
`sklearn https://scikit-learn.org/stable/modules/ensemble.html`_.
"""

from typing import Sequence

import numpy as np
from sklearn.ensemble import BaseEnsemble

from darts import TimeSeries
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
        self,
        actual_anomalies: Sequence[TimeSeries],
        series: Sequence[TimeSeries],
    ):

        X = np.concatenate(
            [s.all_values(copy=False).reshape(len(s), -1) for s in series],
            axis=0,
        )

        y = np.concatenate(
            [s.all_values(copy=False).reshape(len(s)) for s in actual_anomalies],
            axis=0,
        )

        self.model.fit(y=y, X=X)
        return self

    def _predict_core(self, series: Sequence[TimeSeries]) -> Sequence[TimeSeries]:

        return [
            TimeSeries.from_times_and_values(
                s.time_index,
                self.model.predict((s).all_values(copy=False).reshape(len(s), -1)),
            )
            for s in series
        ]
