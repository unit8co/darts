"""
Detector
-------

Detectors can be trainable (FittableDetector) or not trainable (NonFittableDetector). The main functions are
`fit()` (only for the trainable scorer), `detect()` and `score()`.

`fit()` learns the function `f()`, over the history of one anomaly score time series. The function `detect()`
takes an anomaly score time series as input, and applies the function `f()` to obtain a binary prediction.
The function `score()` returns the metric score (accuracy/precision/recall/f1), between a binary prediction
time series and a binary ground truth time series indicating the presence of anomalies.

TODO:
    - check error message and add name of variable in the message error
    - rethink the positionning of fun _check_param()
    - add possibility to input a list of param rather than only one number
    - add more complex detectors
        - create an ensemble fittable detector
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

import numpy as np

from darts import TimeSeries
from darts.ad.utils import _check_timeseries_type, eval_accuracy_from_binary_prediction
from darts.logging import raise_if, raise_if_not


class Detector(ABC):
    "Base class for all detectors"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def detect(self, input: Any) -> Any:
        pass

    @abstractmethod
    def _detect_core(self, input: Any) -> Any:
        pass

    def eval_accuracy(
        self,
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        anomaly_score: Union[TimeSeries, Sequence[TimeSeries]],
        window: int = 1,
        metric: str = "recall",
    ) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
        """Score the results against true anomalies.

        Parameters
        ----------
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not).
        anomaly_score
            Series indicating how anomoulous each window of size w is.
        window
            Integer value indicating the number of past samples each point represents
            in the anomaly_score.
        metric
            Metric function to use. Must be one of "recall", "precision",
            "f1", and "iou".
            Default: "recall"

        Returns
        -------
        Union[float, Sequence[float], Sequence[Sequence[float]]]
            Metric results for each anomaly score
        """

        return eval_accuracy_from_binary_prediction(
            actual_anomalies, self.detect(anomaly_score), window, metric
        )


class NonFittableDetector(Detector):
    "Base class of Detectors that do not need training."

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = False

    def detect(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Detect anomalies from given time series.

        Parameters
        ----------
        series
            series to detect anomalies from.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            binary prediciton (1 if considered as an anomaly, 0 if not)
        """

        list_series = [series] if not isinstance(series, Sequence) else series

        detected_series = []
        for series in list_series:
            _check_timeseries_type(series)

            detected_series.append(self._detect_core(series))

        if len(detected_series) == 1 and not isinstance(series, Sequence):
            return detected_series[0]
        else:
            return detected_series


class FittableDetector(Detector):
    "Base class of Detectors that need training."

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fit_called = False
        self.trainable = True

    def detect(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Detect anomalies from given time series.

        Parameters
        ----------
        series
            series to detect anomalies from.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            binary prediciton (1 if considered as an anomaly, 0 if not)
        """

        list_series = [series] if not isinstance(series, Sequence) else series

        if self.trainable:
            raise_if_not(
                self._fit_called,
                "The Detector has not been fitted yet. Call `fit()` first",
            )

        detected_series = []
        for series in list_series:
            _check_timeseries_type(series)

            if self.trainable:
                raise_if_not(
                    self.width_trained_on == series.width,
                    f"Input must have the same width of the data used for training the detector model, \
                    found width: {self.width_trained_on} and {series.width}",
                )

            detected_series.append(self._detect_core(series))

        if len(detected_series) == 1 and not isinstance(series, Sequence):
            return detected_series[0]
        else:
            return detected_series

    @abstractmethod
    def _fit_core(self, input: Any) -> Any:
        pass

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> None:
        """Trains the detector on the given time series.

        Parameters
        ----------
        series
            Time series to be used to train the detector.

        Returns
        -------
        self
            Fitted Detector.
        """

        list_series = [series] if not isinstance(series, Sequence) else series

        self.width_trained_on = series

        for idx, series in enumerate(list_series):
            _check_timeseries_type(series)

            if idx == 0:
                self.width_trained_on = series.width
            else:
                raise_if_not(
                    series.width == self.width_trained_on,
                    f"Series must have same width, found width {self.width_trained_on} \
                    and {series.width} for index 0 and {idx}",
                )

        self._fit_core(list_series)

    def fit_detect(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Trains the detector and detects anomalies on the same series.

        Parameters
        ----------
        series
            Time series to be used for training and be detected for anomalies.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            Binary prediciton (1 if considered as an anomaly, 0 if not)
        """
        self.fit(series)
        return self.detect(series)


class ThresholdAD(NonFittableDetector):
    """Detector that detects anomaly based on user-given threshold.
    This detector compares time series values with user-given thresholds, and
    identifies time points as anomalous when values are beyond the thresholds.

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


class QuantileAD(FittableDetector):
    """Detector that detects anomaly based on quantiles of historical data.
    This detector compares time series values with user-specified quantiles
    of historical data, and identifies time points as anomalous when values
    are beyond the thresholds.

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

        detected = []
        for width in range(series.width):
            detected.append(
                (
                    (
                        np_series[:, width] > self.abs_high_[width]
                        if (self.high is not None)
                        else float("inf")
                    )
                    | (
                        np_series[:, width] < self.abs_low_[width]
                        if (self.low is not None)
                        else -float("inf")
                    )
                ).astype(int)
            )

        return TimeSeries.from_times_and_values(
            series._time_index, list(zip(*detected))
        )
