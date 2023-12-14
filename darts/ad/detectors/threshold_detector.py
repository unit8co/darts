"""
Threshold Detector
------------------

Detector that detects anomaly based on user-given threshold.
This detector compares time series values with user-given thresholds, and
identifies time points as anomalous when values are beyond the thresholds.
"""

from typing import Sequence, Union

import numpy as np

from darts.ad.detectors.detectors import Detector, _BoundedDetectorMixin
from darts.logging import get_logger, raise_if, raise_if_not
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class ThresholdDetector(Detector, _BoundedDetectorMixin):
    def __init__(
        self,
        low_threshold: Union[int, float, Sequence[float], None] = None,
        high_threshold: Union[int, float, Sequence[float], None] = None,
    ) -> None:
        """
        Flags values that are either below or above the `low_threshold` and `high_threshold`,
        respectively.

        If a single value is provided for `low_threshold` or `high_threshold`, this same
        value will be used across all components of the series.

        If sequences of values are given for the parameters `low_threshold` and/or `high_threshold`,
        they must be of the same length, matching the dimensionality of the series passed
        to ``detect()``, or have a length of 1. In the latter case, this single value will be used
        across all components of the series.

        If either `low_threshold` or `high_threshold` is None, the corresponding bound will not be used.
        However, at least one of the two must be set.

        Parameters
        ----------
        low_threshold
            (Sequence of) lower bounds.
            If a sequence, must match the dimensionality of the series
            this detector is applied to.
        high_threshold
            (Sequence of) upper bounds.
            If a sequence, must match the dimensionality of the series
            this detector is applied to.
        """
        super().__init__()

        low_threshold, high_threshold = self._prepare_boundaries(
            lower_bound=low_threshold,
            upper_bound=high_threshold,
            lower_bound_name="low_threshold",
            upper_bound_name="high_threshold",
        )
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def _detect_core(self, series: TimeSeries) -> TimeSeries:
        raise_if_not(
            series.is_deterministic, "This detector only works on deterministic series."
        )

        raise_if(
            len(self.low_threshold) > 1 and len(self.low_threshold) != series.width,
            "The number of components of input must be equal to the number"
            + " of threshold values. Found number of "
            + f"components equal to {series.width} and expected {len(self.low_threshold)}.",
        )

        # if length is 1, tile it to series width:
        low_threshold = (
            self.low_threshold * series.width
            if len(self.low_threshold) == 1
            else self.low_threshold
        )
        high_threshold = (
            self.high_threshold * series.width
            if len(self.high_threshold) == 1
            else self.high_threshold
        )

        # (time, components)
        np_series = series.all_values(copy=False).squeeze(-1)

        def _detect_fn(x, lo, hi):
            # x of shape (time,) for 1 component
            return (x < (np.NINF if lo is None else lo)) | (
                x > (np.Inf if hi is None else hi)
            )

        detected = np.zeros_like(np_series, dtype=int)

        for component_idx in range(series.width):
            detected[:, component_idx] = _detect_fn(
                np_series[:, component_idx],
                low_threshold[component_idx],
                high_threshold[component_idx],
            )

        return TimeSeries.from_times_and_values(series.time_index, detected)
