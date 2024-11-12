"""
Anomaly aggregators base classes
"""

# TODO:
# - add customize aggregators
# - add in trainable aggregators
#     - log regression
#     - decision tree
# - create show_all_combined (info about correlation, and from what path did
#   the anomaly alarm came from)
import sys

import numpy as np

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

from darts import TimeSeries
from darts.ad.utils import (
    _assert_fit_called,
    _check_input,
    eval_metric_from_binary_prediction,
    series2seq,
)
from darts.logging import get_logger, raise_log

logger = get_logger(__name__)


class Aggregator(ABC):
    """Base class for Aggregators."""

    def __init__(self):
        self.width_trained_on: Optional[int] = None

    @abstractmethod
    def __str__(self):
        """returns the name of the aggregator"""
        pass

    @abstractmethod
    def _predict_core(self, series: Sequence[TimeSeries]) -> Sequence[TimeSeries]:
        """Aggregates the sequence of multivariate binary series given as
        input into a sequence of univariate binary series. assuming the input is
        in the correct shape.

        Parameters
        ----------
        series
            The sequence of multivariate binary series to aggregate

        Returns
        -------
        TimeSeries
            Sequence of aggregated results
        """
        pass

    def predict(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        name: str = "series",
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Aggregates the (sequence of) multivariate binary series given as
        input into a (sequence of) univariate binary series.

        Parameters
        ----------
        series
            The (sequence of) multivariate binary series to aggregate.
        name
            The name of `series`.

        Returns
        -------
        TimeSeries
            (Sequence of) aggregated results.
        """
        called_with_single_series = isinstance(series, TimeSeries)
        series = _check_input(
            series,
            name=name,
            width_expected=self.width_trained_on,
            check_deterministic=True,
            check_binary=True,
            check_multivariate=True,
        )
        pred = self._predict_core(series)
        return pred[0] if called_with_single_series else pred

    def eval_metric(
        self,
        anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
        window: int = 1,
        metric: Literal["recall", "precision", "f1", "accuracy"] = "recall",
    ) -> Union[float, Sequence[float]]:
        """Aggregates the (sequence of) multivariate series given as input into one (sequence of)
        series and evaluates the results against the ground truth anomaly labels.

        Parameters
        ----------
        anomalies
            The (sequence of) binary ground truth anomaly labels (1 if it is an anomaly and 0 if not).
        series
            The (sequence of) predicted multivariate binary series to aggregate.
        window
            (Sequence of) integer value indicating the number of past samples each point
            represents in the (sequence of) series. The parameter will be used by the
            function `_window_adjustment_anomalies()` in darts.ad.utils to transform
            anomalies.
        metric
            The name of the metric function to use. Must be one of "recall", "precision", "f1", and "accuracy".
            Default: "recall".

        Returns
        -------
        Union[float, Sequence[float]]
            (Sequence of) score for the (sequence of) series.
        """
        pred_anomalies = self.predict(series)
        return eval_metric_from_binary_prediction(
            anomalies=anomalies,
            pred_anomalies=pred_anomalies,
            window=window,
            metric=metric,
        )


class FittableAggregator(Aggregator):
    """Base class for Aggregators that require training."""

    def __init__(self):
        super().__init__()
        self._fit_called = False

    @abstractmethod
    def _fit_core(self, anomalies: Sequence[np.ndarray], series: Sequence[np.ndarray]):
        """Fits the aggregator, assuming the input is in the correct shape.

        Parameters
        ----------
        anomalies
            The (sequence of) binary ground truth anomaly labels (1 if it is an anomaly and 0 if not).
        series
            The (sequence of) multivariate binary anomalies (predicted labels) to aggregate.
        """
        pass

    def fit(
        self,
        anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Self:
        """Fit the aggregators on the (sequence of) multivariate binary anomaly series.

        If a list of series is given, they must have the same number of components.

        Parameters
        ----------
        anomalies
            The (sequence of) binary ground truth anomaly labels (1 if it is an anomaly and 0 if not).
        series
            The (sequence of) multivariate binary series (predicted labels) to aggregate.
        """
        pred_width = series2seq(series)[0].width
        series = _check_input(
            series,
            name="series",
            width_expected=pred_width,
            check_deterministic=True,
            check_binary=True,
            check_multivariate=True,
        )
        self.width_trained_on = pred_width

        anomalies = _check_input(
            anomalies,
            name="anomalies",
            width_expected=1,
            check_deterministic=True,
            check_binary=True,
            check_multivariate=False,
        )
        if len(anomalies) != len(series):
            raise_log(
                ValueError(
                    "`anomalies` and `series` must contain the same number of series."
                ),
                logger=logger,
            )
        anomalies_vals, series_vals = [], []
        for anom, pred_anom in zip(anomalies, series):
            anomalies_vals.append(anom.slice_intersect_values(pred_anom)[:, :, 0])
            series_vals.append(pred_anom.slice_intersect_values(anom)[:, :, 0])
        self._fit_core(anomalies_vals, series_vals)
        self._fit_called = True
        return self

    def predict(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        name: str = "series",
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        _assert_fit_called(self._fit_called, name="Aggregator")
        return super().predict(series=series, name=name)
