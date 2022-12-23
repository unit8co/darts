"""
Quantile Detector
-----------------

Flags anomalies that are beyond some quantiles of historical data.
This is similar to a threshold-based detector, where the thresholds are
computed as quantiles of historical data when the detector is fitted.
"""

from typing import Sequence, Union

import numpy as np

from darts.ad.detectors.detectors import FittableDetector
from darts.ad.detectors.threshold_detector import ThresholdDetector
from darts.logging import raise_if, raise_if_not
from darts.timeseries import TimeSeries


class QuantileDetector(FittableDetector):
    def __init__(
        self,
        low_quantile: Union[Sequence[float], float, None] = None,
        high_quantile: Union[Sequence[float], float, None] = None,
    ) -> None:
        """
        Flags values that are either
        below or above the `low_quantile` and `high_quantile`
        quantiles of historical data, respectively.

        If a single value is provided for `low_quantile` or `high_quantile`, this same
        value will be used across all components of the series.

        If sequences of values are given for the parameters `low_quantile` and/or `high_quantile`,
        they must be of the same length, matching the dimensionality of the series passed
        to ``fit()``, or have a length of 1. In the latter case, this single value will be used
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

        Attributes
        ----------
        low_threshold
            The (sequence of) lower quantile values.
        high_threshold
            The (sequence of) upper quantile values.
        """

        super().__init__()

        raise_if(
            low_quantile is None and high_quantile is None,
            "At least one parameter must be not None (`low` and `high` are both None).",
        )

        def _prep_quantile(q):
            return (
                q.tolist()
                if isinstance(q, np.ndarray)
                else [q]
                if not isinstance(q, Sequence)
                else q
            )

        low = _prep_quantile(low_quantile)
        high = _prep_quantile(high_quantile)

        for q in (low, high):
            raise_if_not(
                all([x is None or 0 <= x <= 1 for x in q]),
                "Quantiles must be between 0 and 1, or None.",
            )

        self.low_quantile = low * len(high) if len(low) == 1 else low
        self.high_quantile = high * len(low) if len(high) == 1 else high

        # the quantiles parameters are now sequences of the same length,
        # possibly containing some None values, but at least one non-None value

        # We'll use an inner Threshold detector once the quantiles are fitted
        self.detector = None

        # A few more checks:
        raise_if_not(
            len(self.low_quantile) == len(self.high_quantile),
            "Parameters `low_quantile` and `high_quantile` must be of the same length,"
            + f" found `low`: {len(self.low_quantile)} and `high`: {len(self.high_quantile)}.",
        )

        raise_if(
            all([lo is None for lo in self.low_quantile])
            and all([hi is None for hi in self.high_quantile]),
            "All provided quantile values are None.",
        )

        raise_if_not(
            all(
                [
                    l < h
                    for (l, h) in zip(self.low_quantile, self.high_quantile)
                    if ((l is not None) and (h is not None))
                ]
            ),
            "all values in `low_quantile` must be lower than their corresponding value in `high_quantile`.",
        )

    def _fit_core(self, list_series: Sequence[TimeSeries]) -> None:

        # if len(low) > 1 and len(high) > 1, then check it matches input width:
        raise_if(
            len(self.low_quantile) > 1
            and len(self.low_quantile) != list_series[0].width,
            "The number of components of input must be equal to the number"
            + " of values given for `high_quantile` or/and `low_quantile`. Found number of "
            + f"components equal to {list_series[0].width} and expected {len(self.low_quantile)}.",
        )

        # otherwise, make them the right length
        self.low_quantile = (
            self.low_quantile * list_series[0].width
            if len(self.low_quantile) == 1
            else self.low_quantile
        )
        self.high_quantile = (
            self.high_quantile * list_series[0].width
            if len(self.high_quantile) == 1
            else self.high_quantile
        )

        # concatenate everything along time axis
        np_series = np.concatenate(
            [series.all_values(copy=False) for series in list_series], axis=0
        )

        # move sample dimension to position 1
        np_series = np.moveaxis(np_series, 2, 1)

        # flatten it in order to obtain an array of shape (time * samples, components)
        # where all samples of a given component are concatenated along time
        np_series = np_series.reshape(np_series.shape[0] * np_series.shape[1], -1)

        # Compute 2 thresholds (low, high) for each component:
        # TODO: we could make this more efficient when low_quantile or high_quantile contain a single value
        self.low_threshold = [
            np.quantile(np_series[:, i], q=lo, axis=0) if lo is not None else None
            for i, lo in enumerate(self.low_quantile)
        ]
        self.high_threshold = [
            np.quantile(np_series[:, i], q=hi, axis=0) if hi is not None else None
            for i, hi in enumerate(self.high_quantile)
        ]

        self.detector = ThresholdDetector(
            low_threshold=self.low_threshold, high_threshold=self.high_threshold
        )

        return self

    def _detect_core(self, series: TimeSeries) -> TimeSeries:
        return self.detector.detect(series)
