"""
Quantile Detector
-----------------

Flags anomalies that are beyond some quantiles of historical data.
This is similar to a threshold-based detector, where the thresholds are
computed as quantiles of historical data when the detector is fitted.
"""

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from darts.ad.detectors.detectors import FittableDetector, _BoundedDetectorMixin
from darts.ad.detectors.threshold_detector import ThresholdDetector
from darts.logging import get_logger, raise_log
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class QuantileDetector(FittableDetector, _BoundedDetectorMixin):
    def __init__(
        self,
        low_quantile: Union[Sequence[float], float, None] = None,
        high_quantile: Union[Sequence[float], float, None] = None,
    ) -> None:
        """Quantile Detector

        Flags values that are either below or above the `low_quantile` and `high_quantile` quantiles
        of historical data, respectively.

        If a single value is provided for `low_quantile` or `high_quantile`, this same value will be
        used across all components of the series.

        If sequences of values are given for the parameters `low_quantile` and/or `high_quantile`,
        they must be of the same length, matching the dimensionality of the series passed
        to `fit()`, or have a length of 1. In the latter case, this single value will be used
        across all components of the series.

        If either `low_quantile` or `high_quantile` is None, the corresponding bound will not be used.
        However, at least one of the two must be set.

        Parameters
        ----------
        low_quantile
            (Sequence of) quantile of historical data below which a value is regarded as anomaly.
            Must be between 0 and 1. If a sequence, must match the dimensionality of the series
            this detector is applied to.
        high_quantile
            (Sequence of) quantile of historical data above which a value is regarded as anomaly.
            Must be between 0 and 1. If a sequence, must match the dimensionality of the series
            this detector is applied to.
        """

        super().__init__()
        low_quantile, high_quantile = self._prepare_boundaries(
            lower_bound=low_quantile,
            upper_bound=high_quantile,
            lower_bound_name="low_quantile",
            upper_bound_name="high_quantile",
        )
        for q in (low_quantile, high_quantile):
            if not all([x is None or 0 <= x <= 1 for x in q]):
                raise_log(
                    ValueError("All quantiles must be between 0 and 1, or None."),
                    logger=logger,
                )
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        # We'll use an inner Threshold detector once the quantiles are fitted
        self.detector: Optional[ThresholdDetector] = None

    def _fit_core(self, series: Sequence[TimeSeries]) -> None:
        # if len(low) > 1 and len(high) > 1, then check it matches input width:
        if len(self.low_quantile) > 1 and len(self.low_quantile) != series[0].width:
            raise_log(
                ValueError(
                    "The number of components of input must be equal to the number "
                    "of values given for `high_quantile` or/and `low_quantile`. Found number of "
                    f"components equal to {series[0].width} and expected {len(self.low_quantile)}."
                ),
                logger=logger,
            )

        # otherwise, make them the right length
        self.low_quantile = self._expand_threshold(series[0], self.low_quantile)
        self.high_quantile = self._expand_threshold(series[0], self.high_quantile)

        # concatenate everything along the time axis
        np_series = np.concatenate(
            [series.all_values(copy=False) for series in series], axis=0
        )

        # move sample dimension to position 1
        np_series = np.moveaxis(np_series, 2, 1)

        # flatten it in order to obtain an array of shape (time * samples, components)
        # where all samples of a given component are concatenated along time
        np_series = np_series.reshape(np_series.shape[0] * np_series.shape[1], -1)

        # Compute 2 thresholds (low, high) for each component:
        # TODO: we could make this more efficient when low_quantile or high_quantile contain a single value
        low_threshold = [
            np.quantile(np_series[:, i], q=lo, axis=0) if lo is not None else None
            for i, lo in enumerate(self.low_quantile)
        ]
        high_threshold = [
            np.quantile(np_series[:, i], q=hi, axis=0) if hi is not None else None
            for i, hi in enumerate(self.high_quantile)
        ]

        self.detector = ThresholdDetector(
            low_threshold=low_threshold, high_threshold=high_threshold
        )

    def _detect_core(self, series: TimeSeries, name: str = "series") -> TimeSeries:
        return self.detector.detect(series, name=name)

    @property
    def low_threshold(self):
        return self.detector.low_threshold if self.detector is not None else None

    @property
    def high_threshold(self):
        return self.detector.high_threshold if self.detector is not None else None
