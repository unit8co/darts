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
    If a sequence of values is given for the parameters `low` and/or `high`:
        - they must be of the same length
        - if the length of one parameter is equal to one, the value will be duplicated
          to have the same length as the other parameter
        - the functions ``fit()`` and ``score()``:
            * only accepts series with the same width as the number of values in the given sequence.
              The quantile will be computed along each width with the corresponding value.
            * if a series has a width higher than 1, and the sequences for the parameters
              are equal to 1. The same quantile will be computed for all the widths.

    Parameters
    ----------
    low: float, optional
        (Sequence of) quantile of historical data lower which a value is regarded as anomaly.
        Must be between 0 and 1.
    high: float, optional
        (Sequence of) quantile of historical data above which a value is regarded as anomaly.
        Must be between 0 and 1.

    Attributes
    ----------
    abs_low_: float
        The (sequence of) fitted lower bound of normal range.
    abs_high_: float
        The (sequence of) fitted upper bound of normal range.
    """

    def __init__(
        self,
        low: Union[Sequence[float], float, None] = None,
        high: Union[Sequence[float], float, None] = None,
    ) -> None:
        super().__init__()

        raise_if(
            low is None and high is None,
            "At least one parameter must be not None (`low` and `high` are both None).",
        )

        if low is not None:
            low = [low] if not isinstance(low, Sequence) else low
            self._check_param(low, "low")
            low = [0.0 if x is None else x for x in low]

        if high is not None:
            high = [high] if not isinstance(high, Sequence) else high
            self._check_param(high, "high")
            high = [1.0 if x is None else x for x in high]

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
                        if ((l is not None) & (h is not None))
                    ]
                ),
                "all values in `low` must be lower than their corresponding value in `high`.",
            )

        self.low = low
        self.high = high

    def _check_param(self, param: Sequence[float], name_param: str):
        "Checks if parameter `param` is of type float or int if not None"

        raise_if_not(
            all([isinstance(p, float) for p in param if p is not None]),
            f"all values in parameter `{name_param}` must be of type float.",
        )

        raise_if_not(
            all([(p >= 0 and p <= 1) for p in param if p is not None]),
            f"all values in parameter `{name_param}` must be between 0 and 1.",
        )

        raise_if(
            all([p is None for p in param]),
            f"all values in parameter `{name_param}` cannot be None.",
        )

    def _quantile_per_array(self, np_data, quantiles):
        """Computes the quantile along each dimnension of np_data
        with a different quantile value.
        """
        if len(quantiles) != len(np_data):
            quantiles = quantiles * len(np_data)

        return np.apply_along_axis(
            lambda x: np.quantile(x[1:], x[0]),
            1,
            np.concatenate([np.array(quantiles)[:, np.newaxis], np_data], axis=1),
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
                "Input widths must be equal to the number of values given for `high`"
                + f"or/and `low`, found width {series.width} and expected {len(param)}.",
            )

    def _detection_per_array(self, np_data, lower, upper):
        """Identifies time points as anomalous when values
        are beyond the thresholds (lower and upper).
        """

        # peut etre enlever
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

    def _fit_core(self, list_series: Sequence[TimeSeries]) -> None:

        self._check_input_width(list_series[0])

        np_series = np.concatenate(
            [series.all_values(copy=False) for series in list_series]
        )

        if self.high is not None:
            self.abs_high_ = self._quantile_per_array(
                np_series.flatten(order="F").reshape(list_series[0].width, -1),
                self.high,
            )

        if self.low is not None:
            self.abs_low_ = self._quantile_per_array(
                np_series.flatten(order="F").reshape(list_series[0].width, -1), self.low
            )

        self._fit_called = True

    def _detect_core(self, series: TimeSeries) -> TimeSeries:

        self._check_input_width(series)
        np_series = series.all_values(copy=False)

        detected = self._detection_per_array(
            np_series.flatten(order="F").reshape(series[0].width, -1),
            lower=self.abs_low_ if self.low is not None else None,
            upper=self.abs_high_ if self.high is not None else None,
        )

        return TimeSeries.from_times_and_values(series.time_index, list(zip(*detected)))
