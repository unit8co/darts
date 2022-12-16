"""
Threshold Detector
------------------

Detector that detects anomaly based on user-given threshold.
This detector compares time series values with user-given thresholds, and
identifies time points as anomalous when values are beyond the thresholds.
"""

from typing import Sequence, Union

import numpy as np

from darts.ad.detectors.detectors import NonFittableDetector
from darts.logging import raise_if, raise_if_not
from darts.timeseries import TimeSeries


class ThresholdDetector(NonFittableDetector):
    """

    If a sequence of values is given for the parameters `low` and/or `high`:
        - they must be of the same length
        - if the length of one parameter is equal to one, the value will be duplicated
          to have the same length as the other parameter
        - the functions ``fit()`` and ``score()``:
            * only accepts series that have the same number of components as the number of values
              in the given sequence. The thresholding algorithm will be computed for each component
              with the corresponding value.
            * if a series is multivariate, and the sequences for the parameters
              are equal to 1: the same threshold will be used for all the components.

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
        self,
        low: Union[int, float, Sequence[int], Sequence[float], None] = None,
        high: Union[int, float, Sequence[int], Sequence[float], None] = None,
    ) -> None:
        super().__init__()

        raise_if(
            low is None and high is None,
            "At least one parameter must be not None (`low` and `high` are both None).",
        )

        if low is not None:
            if isinstance(low, np.ndarray):
                low = low.tolist()
            elif not isinstance(low, Sequence):
                low = [low]
            self._check_param(low, "low")
            low = [-float("inf") if x is None else x for x in low]

        if high is not None:
            if isinstance(high, np.ndarray):
                high = high.tolist()
            elif not isinstance(high, Sequence):
                high = [high]
            self._check_param(high, "high")
            high = [float("inf") if x is None else x for x in high]

        if low is not None and high is not None:

            if len(low) == 1 and len(high) > 1:
                low = low * len(high)
            if len(high) == 1 and len(low) > 1:
                high = high * len(low)

            raise_if_not(
                len(low) == len(high),
                "Parameters `low` and `high` must be of the same length,"
                + f" found `low`: {len(low)} and `high`: {len(high)}.",
            )

            raise_if_not(
                all(
                    [
                        l < h
                        for (l, h) in zip(low, high)
                        if (l is not None) & (h is not None)
                    ]
                ),
                "all values in `low` must be lower than their corresponding value in `high`.",
            )

        self.low = low
        self.high = high

    def _check_param(
        self, param: Union[Sequence[float], Sequence[int]], name_param: str
    ):
        "Checks if parameter `param` is of type float or int if not None."

        raise_if_not(
            all([isinstance(p, (float, int)) for p in param if p is not None]),
            f"all values in parameter `{name_param}` must be of type float or int.",
        )

        raise_if(
            all([p is None for p in param]),
            f"all values in parameter `{name_param}` cannot be None.",
        )

    def _check_input_width(self, series: TimeSeries):
        """Checks if input widths is equal to the number of values
        contained in parameter `high` or/and `low`.
        """

        if self.low is not None:
            param = self.low
        else:
            param = self.high

        if len(param) > 1:
            raise_if_not(
                len(param) == series.width,
                "The number of components of input must be equal to the number"
                + " of values given for `high` or/and `low`, found number of "
                + f"components equal to {series.width} and expected {len(param)}.",
            )

    def _detect_core(self, series: TimeSeries) -> TimeSeries:
        self._check_input_width(series)
        np_series = series.all_values(copy=False)

        detected = self._detection_per_array(
            np_series.flatten(order="F").reshape(series[0].width, -1),
            lower=self.low if self.low is not None else None,
            upper=self.high if self.high is not None else None,
        )

        return TimeSeries.from_times_and_values(series.time_index, list(zip(*detected)))

    def _detection_per_array(self, np_data, lower, upper):
        """Identifies time points as anomalous when values
        are beyond the thresholds (lower and upper).
        """

        if lower is not None:
            if len(lower) != len(np_data):
                lower = lower * len(np_data)
        else:
            lower = [None] * len(np_data)

        if upper is not None:
            if len(upper) != len(np_data):
                upper = upper * len(np_data)
        else:
            upper = [None] * len(np_data)

        return np.apply_along_axis(
            lambda x: (x[2:] < (x[0] if (x[0] is not None) else -float("inf")))
            | (x[2:] > (x[1] if (x[1] is not None) else float("inf"))),
            1,
            np.concatenate(
                [
                    np.array(lower)[:, np.newaxis],
                    np.array(upper)[:, np.newaxis],
                    np_data,
                ],
                axis=1,
            ),
        ).astype(int)
