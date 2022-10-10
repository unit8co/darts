"""
Detector
-------

Detectors can be trainable (TrainableDetector) or not trainable (NonTrainableDetector). The main functions are 
`fit()` (only for the trainable scorer), `detect()` and `score()`.

`fit()` learns the function `f()`, over the history of one anomaly score time series. The function `detect()` 
takes an anomaly score time series as input, and applies the function `f()` to obtain a binary prediction. 
The function `score()` returns the metric score (accuracy/precision/recall/f1), between a binary prediction 
time series and a binary ground truth time series indicating the presence of anomalies.  
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence, Union

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from darts import TimeSeries
from darts.logging import raise_if, raise_if_not


class Detector(ABC):
    "Base class for all detectors (TS_anomaly_score -> TS_binary_prediction)"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def _detect_core(self, input: Any) -> Any:
        pass

    def score(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        scoring: str = "recall",
    ) -> Union[float, Sequence[float]]:
        """Score the results against true anomalies.

        Parameters
        ----------
        ts: Darts timeseries (uni/multivariate)
            Time series to detect anomalies from.
        actual_anomalies: Darts timeseries
            True anomalies.
        scoring: str, optional
            Scoring function to use. Must be one of "recall", "precision",
            "f1", and "iou".
            Default: "recall"

        Returns
        -------
        Darts timeseries (uni/multivariate)
            Score(s) for each timeseries
        """

        if scoring == "recall":
            scoring_fn = recall_score
        elif scoring == "precision":
            scoring_fn = precision_score
        elif scoring == "f1":
            scoring_fn = f1_score
        elif scoring == "accuracy":
            scoring_fn = accuracy_score
        else:
            raise ValueError(
                "Argument `scoring` must be one of 'recall', 'precision', "
                "'f1' and 'accuracy'."
            )

        return scoring_fn(
            y_true=actual_anomalies.all_values().flatten(),
            y_pred=self.detect(series).all_values().flatten(),
        )

    def detect(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Detect anomalies from given time series.

        Parameters
        ----------
        series: Darts.Series
            Time series to detect anomalies from.

        Returns
        -------
        Darts.Series
            Binary prediciton (1 if considered as an anomaly, 0 if not)
        """
        # check input (type, size, values)

        if self.trainable:
            raise_if_not(
                self._fit_called,
                "The Detector has not been fitted yet. Call `fit()` first",
            )

        return self._detect_core(series)


class NonTrainableDetector(Detector):
    "Base class of Detectors that do not need training."

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trainable = False


class TrainableDetector(Detector):
    "Base class of Detectors that need training."

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fit_called = False
        self.trainable = True

    @abstractmethod
    def _fit_core(self, input: Any) -> Any:
        pass

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> None:
        """Trains the detector on the given time series.

        Parameters
        ----------
        series: Darts.Series
            Time series to be used to train the detector.

        Returns
        -------
        self
            Fitted Detector.
        """
        self._fit_core(series)

    def fit_detect(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Trains the detector and detects anomalies on the time series used
        for training.

        Parameters
        ----------
        series: Darts.Series
            Time series to be used for training and be detected for anomalies.

        Returns
        -------
        Darts.Series
            Binary prediciton (1 if considered as an anomaly, 0 if not)
        """
        self.fit(series)
        return self.detect(series)


class ThresholdAD(NonTrainableDetector):
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
        self, low: Union[float, None] = None, high: Union[float, None] = None
    ) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def _detect_core(self, series: TimeSeries) -> TimeSeries:
        detected = (
            series.pd_series()
            > (self.high if (self.high is not None) else float("inf"))
        ) | (
            series.pd_series() < (self.low if (self.low is not None) else -float("inf"))
        ).astype(
            int
        )

        return TimeSeries.from_series(detected)


class QuantileAD(TrainableDetector):
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

    def __init__(self, low: Union[float, None], high: Union[float, None]) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def _fit_core(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> None:

        self.abs_high_ = series.pd_series().quantile(self.high)
        self.abs_low_ = series.pd_series().quantile(self.low)
        self._fit_called = True

    def _detect_core(self, series: TimeSeries) -> TimeSeries:
        series = series.pd_series()
        detected = (series > self.abs_high_) | (series < self.abs_low_)
        return TimeSeries.from_series(detected)
