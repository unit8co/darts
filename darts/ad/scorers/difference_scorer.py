"""
Difference Scorer
-----------------

This scorer simply computes the elementwise difference
between two series. If the two series are multivariate, it
returns a multivariate series.
"""

from typing import Union

import numpy as np
import pandas as pd

from darts.ad.scorers.scorers import AnomalyScorer
from darts.timeseries import TimeSeries


class DifferenceScorer(AnomalyScorer):
    def __init__(self) -> None:
        super().__init__(univariate_scorer=False, window=1)

    def __str__(self):
        return "Difference"

    def _score_core_from_prediction(
        self,
        actual_series: np.ndarray,
        pred_series: np.ndarray,
        time_index: Union[pd.DatetimeIndex, pd.RangeIndex],
    ) -> TimeSeries:
        actual_series = self._extract_deterministic2(actual_series, "actual_series")
        pred_series = self._extract_deterministic2(pred_series, "pred_series")
        return TimeSeries.from_times_and_values(
            values=actual_series - pred_series,
            times=time_index,
        )
