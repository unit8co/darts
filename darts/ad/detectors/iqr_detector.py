"""
Interquartile Range (IQR) Detector
-----------------

Flags anomalies that are beyond the IQR (between the third and the first quartile)
of historical data by some factor of it's difference (typically 1.5).
This is similar to a threshold-based detector, but the thresholds are
computed as distances from the IQR of historical data when the detector is fitted.
"""

from collections.abc import Sequence
from typing import Union

import numpy as np

from darts.ad.detectors.quantile_detector import QuantileDetector
from darts.ad.detectors.threshold_detector import ThresholdDetector
from darts.logging import get_logger, raise_log
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class IQRDetector(QuantileDetector):
    def __init__(self, scale: Union[Sequence[float], float] = 1.5) -> None:
        """IQR Detector

        Flags values that lie outside of the interquartile range (IQR)
        by more than a certain factor of IQR's value as anomalies.
        The factor is passed in the `scale` parameter.

        If a single value is provided for `scale`,
        this same value will be used across all components of the series.

        If a sequences of values is given for the `scale` parameter,
        it's length must match the dimensionality of the series passed.

        Parameters
        ----------
        scale
            (Sequence of) scale(s) used to indicate what distance from the IQR constitutes an anomaly.
            Defaults to `1.5`. Must be non-negative. If a sequence, must match the dimensionality of the series
            this detector is applied to.
        """

        # Parent QuantileDetector will compute Q1 and Q3 thresholds
        super().__init__(low_quantile=0.25, high_quantile=0.75)

        self.scale = np.array(scale)
        if self.scale.ndim == 0:
            self.scale = np.expand_dims(self.scale, 0)

        if not np.issubdtype(self.scale.dtype, np.number) or (self.scale < 0.0).any():
            raise_log(
                ValueError("All values in `scale` must be non-negative numbers."),
                logger=logger,
            )

    def _fit_core(self, series: Sequence[TimeSeries]) -> None:
        super()._fit_core(series)

        if len(self.scale) > 1 and len(self.scale) != series[0].width:
            raise_log(
                ValueError(
                    "The number of components of input must be equal to the number "
                    "of values given for `scale`. Found number of components "
                    f"equal to {series[0].width} and expected {len(self.scale)}."
                ),
                logger=logger,
            )

        low_threshold = np.array(self.detector.low_threshold)
        high_threshold = np.array(self.detector.high_threshold)

        IQR = high_threshold - low_threshold

        low_threshold -= self.scale * IQR
        high_threshold += self.scale * IQR

        self.detector = ThresholdDetector(
            low_threshold=list(low_threshold), high_threshold=list(high_threshold)
        )
