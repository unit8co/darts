"""
Quantile Detector
-----------------

Detector that detects anomalies based on quantiles of historical data.
This detector compares time series values with user-specified quantiles
of historical data, and identifies time points as anomalous when values
are beyond the thresholds.
"""

from typing import Sequence, Union

import numpy as np

from darts.ad.detectors.detectors import FittableDetector
from darts.logging import raise_if, raise_if_not
from darts.timeseries import TimeSeries


class QuantileDetector(FittableDetector):
    """
    Parameters
    ----------
    low: float, optional
        Quantile of historical data lower which a value is regarded as anomaly.
        Must be between 0 and 1.
    high: float, optional
        Quantile of historical data above which a value is regarded as anomaly.
        Must be between 0 and 1.

    Attributes
    ----------
    abs_low_: float
        The fitted lower bound of normal range.
    abs_high_: float
        The fitted upper bound of normal range.
    """

    def __init__(
        self, low: Union[int, float, None] = None, high: Union[int, float, None] = None
    ) -> None:
        super().__init__()

        raise_if(
            low is None and high is None,
            "At least one parameter must be not None (low and high both None)",
        )

        self._check_param(low, "low")
        self._check_param(high, "high")

        if low is not None and high is not None:
            raise_if_not(
                low < high,
                f"Parameter `low` must be lower than parameter `high`, found `low`: {low} and `high`: {high}.",
            )

        self.low = low
        self.high = high

    def _check_param(self, param: Union[int, float, None], name_param: str):
        "Checks if parameter `param` is of type float or int if not None"

        if param is not None:
            raise_if_not(
                isinstance(param, (float, int)),
                f"Parameter {name_param} must be of type float, found type {type(param)}",
            )

            raise_if_not(
                param >= 0 and param <= 1,
                f"Parameter {name_param} must be between 0 and 1, found value {param}",
            )

    def _fit_core(self, list_series: Sequence[TimeSeries]) -> None:

        np_series = np.concatenate(
            [series.all_values(copy=False) for series in list_series]
        )

        if self.high is not None:
            self.abs_high_ = np.quantile(np_series, q=self.high, axis=0)

        if self.low is not None:
            self.abs_low_ = np.quantile(np_series, q=self.low, axis=0)

        self._fit_called = True

    def _detect_core(self, series: TimeSeries) -> TimeSeries:

        np_series = series.all_values(copy=False)

        # TODO: vectorize

        detected = []
        for width in range(series.width):
            np_series_temp = np_series[:, width]

            detected.append(
                (
                    np_series_temp
                    > (
                        self.abs_high_[width][0]
                        if (self.high is not None)
                        else float("inf")
                    )
                )
                | (
                    np_series_temp
                    < (
                        self.abs_low_[width][0]
                        if (self.low is not None)
                        else -float("inf")
                    )
                ).astype(int)
            )

        return TimeSeries.from_times_and_values(
            series._time_index, list(zip(*detected))
        )
