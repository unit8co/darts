"""
Detector Base Classes
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
from darts.ad.utils import eval_accuracy_from_binary_prediction
from darts.logging import raise_if_not


class Detector(ABC):
    """Base class for all detectors"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def detect(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Detect anomalies on given time series.

        Parameters
        ----------
        series
            series on which to detect anomalies.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            binary prediciton (1 if considered as an anomaly, 0 if not)
        """

        list_series = [series] if not isinstance(series, Sequence) else series

        raise_if_not(
            all([isinstance(s, TimeSeries) for s in list_series]),
            "all series in `series` must be of type TimeSeries.",
        )

        raise_if_not(
            all([s.is_deterministic for s in list_series]),
            "all series in `series` must be deterministic (number of samples equal to 1).",
        )

        detected_series = []
        for s in list_series:
            detected_series.append(self._detect_core(s))

        if len(detected_series) == 1 and not isinstance(series, Sequence):
            return detected_series[0]
        else:
            return detected_series

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
            "f1", and "accuracy".
            Default: "recall"

        Returns
        -------
        Union[float, Sequence[float], Sequence[Sequence[float]]]
            Metric results for each anomaly score
        """

        if isinstance(anomaly_score, Sequence):
            raise_if_not(
                all([isinstance(s, TimeSeries) for s in anomaly_score]),
                "all series in `anomaly_score` must be of type TimeSeries.",
            )

            raise_if_not(
                all([s.is_deterministic for s in anomaly_score]),
                "all series in `anomaly_score` must be deterministic (number of samples equal to 1).",
            )
        else:
            raise_if_not(
                isinstance(anomaly_score, TimeSeries),
                f"Input `anomaly_score` must be of type TimeSeries, found {type(anomaly_score)}.",
            )

            raise_if_not(
                anomaly_score.is_deterministic,
                "Input `anomaly_score` must be deterministic (number of samples equal to 1).",
            )

        return eval_accuracy_from_binary_prediction(
            actual_anomalies, self.detect(anomaly_score), window, metric
        )


class FittableDetector(Detector):
    """Base class of Detectors that need training."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fit_called = False

    def detect(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Detect anomalies on given time series.

        Parameters
        ----------
        series
            series on which to detect anomalies.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            binary prediciton (1 if considered as an anomaly, 0 if not)
        """

        list_series = [series] if not isinstance(series, Sequence) else series

        raise_if_not(
            self._fit_called,
            "The Detector has not been fitted yet. Call `fit()` first.",
        )

        raise_if_not(
            all([self.width_trained_on == s.width for s in list_series]),
            "all series in `series` must have the same number of components as the data "
            + "used for training the detector model, number of components in training: "
            + f" {self.width_trained_on}.",
        )

        return super().detect(series)

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

        raise_if_not(
            all([isinstance(s, TimeSeries) for s in list_series]),
            "all series in `series` must be of type TimeSeries.",
        )

        raise_if_not(
            all([s.is_deterministic for s in list_series]),
            "all series in `series` must be deterministic (number of samples equal to 1).",
        )

        self.width_trained_on = list_series[0].width

        raise_if_not(
            all([s.width == self.width_trained_on for s in list_series]),
            "all series in `series` must have the same number of components.",
        )

        self._fit_called = True
        return self._fit_core(list_series)

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
