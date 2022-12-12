"""
Detector Base Classes
---------------------

Detectors can be trainable (FittableDetector) or not trainable (NonFittableDetector). The main functions are
``fit()`` (only for the trainable scorer), ``detect()`` and ``eval_accuracy()``.

``fit()`` learns a function `f()`, over the history of one anomaly score time series. The function ``detect()``
takes an anomaly score time series as input, and applies the function `f()` to obtain a binary prediction.
The function ``eval_accuracy()`` returns the metric score (accuracy/precision/recall/f1), between a binary prediction
time series and a binary ground truth time series indicating the presence of anomalies.
"""

# TODO:
#     - check error message and add name of variable in the message error
#     - rethink the positionning of fun _check_param()
#     - add possibility to input a list of param rather than only one number
#     - add more complex detectors
#         - create an ensemble fittable detector

from abc import ABC, abstractmethod
from typing import Any, Sequence, Union

from darts import TimeSeries
from darts.ad.utils import _check_timeseries_type, eval_accuracy_from_binary_prediction
from darts.logging import raise_if_not


class Detector(ABC):
    """Base class for all detectors"""

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
    """Base class of Detectors that do not need training."""

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
        for s in list_series:
            _check_timeseries_type(s)

            detected_series.append(self._detect_core(s))

        if len(detected_series) == 1 and not isinstance(series, Sequence):
            return detected_series[0]
        else:
            return detected_series


class FittableDetector(Detector):
    """Base class of Detectors that need training."""

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

        raise_if_not(
            self._fit_called,
            "The Detector has not been fitted yet. Call `fit()` first",
        )

        detected_series = []
        for s in list_series:
            _check_timeseries_type(s)

            raise_if_not(
                self.width_trained_on == s.width,
                f"Input must have the same width of the data used for training the detector model, \
                found training width {self.width_trained_on} and input width {s.width}",
            )

            detected_series.append(self._detect_core(s))

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
