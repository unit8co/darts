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
    """Difference Scorer"""

    def __init__(self) -> None:
        super().__init__(is_univariate=False, window=1)

    def __str__(self):
        return "Difference"

    def _score_core_from_prediction(
        self,
        actual_vals: np.ndarray,
        pred_vals: np.ndarray,
        time_index: Union[pd.DatetimeIndex, pd.RangeIndex],
    ) -> TimeSeries:
        actual_vals = self._extract_deterministic_values(actual_vals, "actual_series")
        pred_vals = self._extract_deterministic_values(pred_vals, "pred_series")
        return TimeSeries.from_times_and_values(
            values=actual_vals - pred_vals,
            times=time_index,
        )
