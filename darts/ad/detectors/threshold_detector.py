"""
Threshold Detector
------------------

Detector that detects anomaly based on user-given threshold.
This detector compares time series values with user-given thresholds, and
identifies time points as anomalous when values are beyond the thresholds.
"""

from collections.abc import Sequence
from typing import Union

import numpy as np

from darts.ad.detectors.detectors import Detector, _BoundedDetectorMixin
from darts.logging import get_logger, raise_log
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class ThresholdDetector(Detector, _BoundedDetectorMixin):
    def __init__(
        self,
        low_threshold: Union[int, float, Sequence[float], None] = None,
        high_threshold: Union[int, float, Sequence[float], None] = None,
    ) -> None:
        """Threshold Detector

        Flags values that are either below or above the `low_threshold` and `high_threshold`,
        respectively.

        If a single value is provided for `low_threshold` or `high_threshold`, this same
        value will be used across all components of the series.

        If sequences of values are given for the parameters `low_threshold` and/or `high_threshold`,
        they must be of the same length, matching the dimensionality of the series passed
        to `detect()`, or have a length of 1. In the latter case, this single value will be used
        across all components of the series.

        If either `low_threshold` or `high_threshold` is None, the corresponding bound will not be used.
        However, at least one of the two must be set.

        Parameters
        ----------
        low_threshold
            (Sequence of) lower bounds. If a sequence, must match the dimensionality of the series this
            detector is applied to.
        high_threshold
            (Sequence of) upper bounds. If a sequence, must match the dimensionality of the series this
            detector is applied to.
        """
        super().__init__()
        low_threshold, high_threshold = self._prepare_boundaries(
            lower_bound=low_threshold,
            upper_bound=high_threshold,
            lower_bound_name="low_threshold",
            upper_bound_name="high_threshold",
        )
        self._low_threshold = low_threshold
        self._high_threshold = high_threshold

    def _detect_core(self, series: TimeSeries, name: str = "series") -> TimeSeries:
        if len(self.low_threshold) > 1 and len(self.low_threshold) != series.width:
            raise_log(
                ValueError(
                    f"The number of components for each series in `{name}` must be "
                    f"equal to the number of threshold values. Found number of "
                    f"components equal to {series.width} and expected {len(self.low_threshold)}."
                ),
                logger=logger,
            )

        # if length is 1, tile it to series width:
        low_threshold = self._expand_threshold(series[0], self.low_threshold)
        high_threshold = self._expand_threshold(series[0], self.high_threshold)

        # (time, components)
        np_series = series.values(copy=False)

        def _detect_fn(x, lo, hi):
            # x of shape (time,) for 1 component
            return (x < (-np.inf if lo is None else lo)) | (
                x > (np.inf if hi is None else hi)
            )

        detected = np.zeros_like(np_series, dtype=int)

        for component_idx in range(series.width):
            detected[:, component_idx] = _detect_fn(
                np_series[:, component_idx],
                low_threshold[component_idx],
                high_threshold[component_idx],
            )
        return series.with_values(np.expand_dims(detected, -1).astype(series.dtype))

    @property
    def low_threshold(self):
        return self._low_threshold

    @property
    def high_threshold(self):
        return self._high_threshold
