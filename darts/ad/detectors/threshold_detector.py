"""
Threshold Detector
------------------

Detector that detects anomaly based on user-given threshold.
This detector compares time series values with user-given thresholds, and
identifies time points as anomalous when values are beyond the thresholds.
"""

from typing import Union

from darts.ad.detectors.detectors import NonFittableDetector
from darts.logging import raise_if, raise_if_not
from darts.timeseries import TimeSeries


class ThresholdDetector(NonFittableDetector):
    """
    Parameters
    ----------
    low: float, optional
        Threshold below which a value is regarded anomaly. Default: None, i.e.
        no threshold on lower side.
    high: float, optional
        Threshold above which a value is regarded anomaly. Default: None, i.e.
        no threshold on upper side.
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

    def _detect_core(self, series: TimeSeries) -> TimeSeries:

        np_series = series.all_values(copy=False)
        detected = (
            np_series > (self.high if (self.high is not None) else float("inf"))
        ) | (
            np_series < (self.low if (self.low is not None) else -float("inf"))
        ).astype(
            int
        )

        return TimeSeries.from_times_and_values(series._time_index, detected)
