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
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from darts import TimeSeries
from darts.ad.utils import eval_metric_from_binary_prediction
from darts.logging import get_logger, raise_if, raise_if_not

logger = get_logger(__name__)


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
            logger,
        )

        raise_if_not(
            all([s.is_deterministic for s in list_series]),
            "all series in `series` must be deterministic (number of samples equal to 1).",
            logger,
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

    def eval_metric(
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
                logger,
            )

            raise_if_not(
                all([s.is_deterministic for s in anomaly_score]),
                "all series in `anomaly_score` must be deterministic (number of samples equal to 1).",
                logger,
            )
        else:
            raise_if_not(
                isinstance(anomaly_score, TimeSeries),
                f"Input `anomaly_score` must be of type TimeSeries, found {type(anomaly_score)}.",
                logger,
            )

            raise_if_not(
                anomaly_score.is_deterministic,
                "Input `anomaly_score` must be deterministic (number of samples equal to 1).",
                logger,
            )

        return eval_metric_from_binary_prediction(
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
            logger,
        )

        raise_if_not(
            all([self.width_trained_on == s.width for s in list_series]),
            "all series in `series` must have the same number of components as the data "
            + "used for training the detector model, number of components in training: "
            + f" {self.width_trained_on}.",
            logger,
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
            logger,
        )

        raise_if_not(
            all([s.is_deterministic for s in list_series]),
            "all series in `series` must be deterministic (number of samples equal to 1).",
            logger,
        )

        self.width_trained_on = list_series[0].width

        raise_if_not(
            all([s.width == self.width_trained_on for s in list_series]),
            "all series in `series` must have the same number of components.",
            logger,
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


class _BoundedDetectorMixin:
    """
    A class containing functions supporting bounds-based detection, to be used as a mixin for some
    `Detector` subclasses.
    """

    @staticmethod
    def _prepare_boundaries(
        lower_bound_name: str,
        upper_bound_name: str,
        lower_bound: Optional[Union[Sequence[float], float]] = None,
        upper_bound: Optional[Union[Sequence[float], float]] = None,
    ) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """
        Process the boundaries argument and perform some sanity checks

        Parameters
        ----------
        lower_bound_name
            Name of the lower bound
        upper_bound_name
            Name of the upper bound
        lower_bound
            (Sequence of) numerical bound below which a value is regarded as anomaly.
            If a sequence, must match the dimensionality of the series
            this detector is applied to.
        upper_bound
            (Sequence of) numerical bound above which a value is regarded as anomaly.
            If a sequence, must match the dimensionality of the series
            this detector is applied to.

        Returns
        -------
        lower_bound
            Lower bounds, as a list of values (at least one not None value)
        upper_bound
            Upper bounds, as a list of values (at least one not None value)
        """

        raise_if(
            lower_bound is None and upper_bound is None,
            f"`{lower_bound_name} and `{upper_bound_name}` cannot both be `None`.",
            logger,
        )

        def _prep_boundaries(boundaries) -> List[Optional[float]]:
            """Convert boundaries to List"""
            return (
                boundaries.tolist()
                if isinstance(boundaries, np.ndarray)
                else (
                    [boundaries] if not isinstance(boundaries, Sequence) else boundaries
                )
            )

        # convert to list
        lower_bound = _prep_boundaries(lower_bound)
        upper_bound = _prep_boundaries(upper_bound)

        raise_if(
            all([lo is None for lo in lower_bound])
            and all([hi is None for hi in upper_bound]),
            "All provided upper and lower bounds values are None.",
            logger,
        )

        # match the lengths of the boundaries
        lower_bound = (
            lower_bound * len(upper_bound) if len(lower_bound) == 1 else lower_bound
        )
        upper_bound = (
            upper_bound * len(lower_bound) if len(upper_bound) == 1 else upper_bound
        )

        raise_if_not(
            len(lower_bound) == len(upper_bound),
            f"Parameters `{lower_bound_name}` and `{upper_bound_name}` must be of the same length,"
            f" found `{lower_bound_name}`: {len(lower_bound)} and `{upper_bound_name}`: {len(upper_bound)}.",
            logger,
        )

        raise_if_not(
            all(
                [
                    lb is None or ub is None or lb <= ub
                    for (lb, ub) in zip(lower_bound, upper_bound)
                ]
            ),
            f"All values in `{lower_bound_name}` must be lower or equal"
            f"to their corresponding value in `{upper_bound_name}`.",
            logger,
        )

        return lower_bound, upper_bound
