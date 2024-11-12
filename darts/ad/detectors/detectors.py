"""
Detector Base Classes
"""

# TODO:
#     - check error message and add name of variable in the message error
#     - add possibility to input a list of param rather than only one number
#     - add more complex detectors
#         - create an ensemble fittable detector

import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np

from darts import TimeSeries
from darts.ad.utils import (
    _assert_fit_called,
    _check_input,
    eval_metric_from_binary_prediction,
)
from darts.logging import get_logger, raise_log
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)


class Detector(ABC):
    """Base class for all detectors"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.width_trained_on: Optional[int] = None

    def detect(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        name: str = "series",
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Detect anomalies on given time series.

        Parameters
        ----------
        series
            The (sequence of) series on which to detect anomalies.
        name
            The name of `series`.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            binary prediction (1 if considered as an anomaly, 0 if not)
        """
        called_with_single_series = isinstance(series, TimeSeries)
        series = _check_input(
            series,
            name=name,
            width_expected=self.width_trained_on,
            check_deterministic=True,
        )
        detected_series = []
        for s in series:
            detected_series.append(self._detect_core(s, name=name))
        return detected_series[0] if called_with_single_series else detected_series

    def eval_metric(
        self,
        anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        pred_scores: Union[TimeSeries, Sequence[TimeSeries]],
        window: int = 1,
        metric: Literal["recall", "precision", "f1", "accuracy"] = "recall",
    ) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
        """Score the results against true anomalies.

        Parameters
        ----------
        anomalies
            The (sequence of) ground truth binary anomaly series (`1` if it is an anomaly and `0` if not).
        pred_scores
            The (sequence of) of estimated anomaly score series indicating how anomalous each window of size w is.
        window
            Integer value indicating the number of past samples each point represents in the `pred_scores`.
        metric
            The name of the metric function to use. Must be one of "recall", "precision", "f1", and "accuracy".
            Default: "recall".

        Returns
        -------
        Union[float, Sequence[float], Sequence[Sequence[float]]]
            Metric results for each anomaly score
        """
        return eval_metric_from_binary_prediction(
            anomalies=anomalies,
            pred_anomalies=self.detect(pred_scores),
            window=window,
            metric=metric,
        )

    @abstractmethod
    def _detect_core(self, series: TimeSeries, name: str = "series") -> TimeSeries:
        pass


class FittableDetector(Detector):
    """Base class of Detectors that require training."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fit_called = False

    def detect(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        name: str = "series",
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        _assert_fit_called(self._fit_called, name="Detector")
        return super().detect(series, name=name)

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> Self:
        """Trains the detector on the given time series.

        Parameters
        ----------
        series
            Time (sequence of) series to be used to train the detector.

        Returns
        -------
        self
            Fitted Detector.
        """
        width = series2seq(series)[0].width
        series = _check_input(
            series,
            name="series",
            width_expected=width,
            check_deterministic=True,
            check_binary=False,
            check_multivariate=False,
        )
        self.width_trained_on = width
        self._fit_core(series)
        self._fit_called = True
        return self

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
            Binary prediction (1 if considered as an anomaly, 0 if not)
        """
        self.fit(series)
        return self.detect(series, name="series")

    @abstractmethod
    def _fit_core(self, series: Sequence[TimeSeries]) -> None:
        pass


class _BoundedDetectorMixin(ABC):
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
    ) -> tuple[list[Optional[float]], list[Optional[float]]]:
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
        if lower_bound is None and upper_bound is None:
            raise_log(
                ValueError(
                    f"`{lower_bound_name} and `{upper_bound_name}` cannot both be `None`."
                ),
                logger=logger,
            )

        def _prep_boundaries(boundaries) -> list[Optional[float]]:
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

        if all([lo is None for lo in lower_bound]) and all([
            hi is None for hi in upper_bound
        ]):
            raise_log(
                ValueError("All provided upper and lower bounds values are None."),
                logger=logger,
            )

        # match the lengths of the boundaries
        lower_bound = (
            lower_bound * len(upper_bound) if len(lower_bound) == 1 else lower_bound
        )
        upper_bound = (
            upper_bound * len(lower_bound) if len(upper_bound) == 1 else upper_bound
        )

        if not len(lower_bound) == len(upper_bound):
            raise_log(
                ValueError(
                    f"Parameters `{lower_bound_name}` and `{upper_bound_name}` "
                    f"must be of the same length `n`, found "
                    f"`{lower_bound_name}`: n={len(lower_bound)} and "
                    f"`{upper_bound_name}`: n={len(upper_bound)}."
                ),
                logger=logger,
            )
        if not all([
            lb is None or ub is None or lb <= ub
            for (lb, ub) in zip(lower_bound, upper_bound)
        ]):
            raise_log(
                ValueError(
                    f"All values in `{lower_bound_name}` must be lower or equal"
                    f"to their corresponding value in `{upper_bound_name}`."
                ),
                logger=logger,
            )
        return lower_bound, upper_bound

    @staticmethod
    def _expand_threshold(series: TimeSeries, threshold: list[float]) -> list[float]:
        return threshold * series[0].width if len(threshold) == 1 else threshold

    @property
    @abstractmethod
    def low_threshold(self):
        pass

    @property
    @abstractmethod
    def high_threshold(self):
        pass
