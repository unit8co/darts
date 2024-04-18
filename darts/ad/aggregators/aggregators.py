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

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union

from darts import TimeSeries
from darts.ad.utils import (
    _assert_binary,
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
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Aggregates the (sequence of) multivariate binary series given as
        input into a (sequence of) univariate binary series.

        Parameters
        ----------
        series
            The (sequence of) multivariate binary series to aggregate.

        Returns
        -------
        TimeSeries
            (Sequence of) aggregated results.
        """
        list_series = self._check_input(
            series,
            name="series",
            width_expected=self.width_trained_on,
            check_multivariate=True,
        )
        if isinstance(series, TimeSeries):
            return self._predict_core(list_series)[0]
        else:
            return self._predict_core(list_series)

    @staticmethod
    def _check_input(
        series: Union[TimeSeries, Sequence[TimeSeries]],
        name: str,
        width_expected: Optional[int],
        check_multivariate: bool,
    ):
        """
        Checks for input if:
            - it is a (sequence of) multivariate series (width>1)
            - (sequence of) series must be:
                * a deterministic TimeSeries
                * binary (only values equal to 0 or 1)
                * must have width `width_expected` if it is not `None`.

        Parameters
        ----------
        series
            A (sequence of) multivariate series.
        name
            The name of the series.
        width_expected
            Optionally, the expected number of components/width of each series.
        check_multivariate
            Whether to check if all series are multivariate.
        """

        list_series = series2seq(series)
        for s in list_series:
            if not isinstance(s, TimeSeries):
                raise_log(
                    ValueError(f"all series in `{name}` must be of type TimeSeries."),
                    logger=logger,
                )
            if not s.is_deterministic:
                raise_log(
                    ValueError(
                        f"all series in `{name}` must be deterministic (number of samples=1)."
                    ),
                    logger=logger,
                )
            if check_multivariate and s.width <= 1:
                raise_log(
                    ValueError(
                        f"all series in `{name}` must be multivariate (width>1)."
                    ),
                    logger=logger,
                )
            if width_expected is not None and s.width != width_expected:
                raise_log(
                    ValueError(
                        f"all series in `{name}` must have `{width_expected}` component(s) (width={width_expected})."
                    ),
                    logger=logger,
                )

            _assert_binary(s, name_series="`series`")
        return list_series

    def eval_metric(
        self,
        actual_series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
        window: int = 1,
        metric: str = "recall",
    ) -> Union[float, Sequence[float]]:
        """Aggregates the (sequence of) multivariate series given as input into one (sequence of)
        series and evaluates the results against the ground truth anomaly labels.

        Parameters
        ----------
        actual_series
            The (sequence of) binary ground truth anomaly labels (1 if it is an anomaly and 0 if not).
        pred_series
            The (sequence of) multivariate binary series (predicted labels) to aggregate.
        window
            (Sequence of) integer value indicating the number of past samples each point
            represents in the (sequence of) series. The parameter will be used by the
            function ``_window_adjustment_anomalies()`` in darts.ad.utils to transform
            actual_series.
        metric
            Metric function to use. Must be one of "recall", "precision", "f1", and "accuracy".
            Default: "recall".

        Returns
        -------
        Union[float, Sequence[float]]
            (Sequence of) score for the (sequence of) series.
        """
        preds = self.predict(pred_series)
        return eval_metric_from_binary_prediction(
            series2seq(actual_series), preds, window, metric
        )


class FittableAggregator(Aggregator):
    """Base class for Aggregators that require training."""

    def __init__(self):
        super().__init__()
        # flag whether Aggregator was fitted
        self._fit_called = False

    def _assert_fit_called(self):
        """Checks if the Aggregator has been fitted before calling its `score()` function."""

        if not self._fit_called:
            raise_log(
                ValueError(
                    f"The Aggregator {self.__str__()} has not been fitted yet. Call `fit()` first."
                ),
                logger=logger,
            )

    @abstractmethod
    def _fit_core(
        self, actual_series: Sequence[TimeSeries], pred_series: Sequence[TimeSeries]
    ) -> "FittableAggregator":
        """Fits the aggregator, assuming the input is in the correct shape.

        Parameters
        ----------
        actual_series
            The (sequence of) binary ground truth anomaly labels (1 if it is an anomaly and 0 if not).
        pred_series
            The (sequence of) multivariate binary series (predicted labels) to aggregate.

        Returns
        -------
        FittableAggregator
            The fitted model
        """
        pass

    def fit(
        self,
        actual_series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> "FittableAggregator":
        """Fit the aggregators on the (sequence of) multivariate binary series.

        If a list of series is given, they must have the same number of components.

        Parameters
        ----------
        actual_series
            The (sequence of) binary ground truth anomaly labels (1 if it is an anomaly and 0 if not).
        pred_series
            The (sequence of) multivariate binary series (predicted labels) to aggregate.
        """
        pred_width = series2seq(pred_series)[0].width
        pred_series = self._check_input(
            pred_series,
            name="pred_series",
            width_expected=pred_width,
            check_multivariate=True,
        )
        self.width_trained_on = pred_width

        actual_series = self._check_input(
            actual_series,
            name="actual_series",
            width_expected=1,
            check_multivariate=False,
        )
        if len(actual_series) != len(pred_series):
            raise_log(
                ValueError(
                    "`actual_series` and `pred_series` must contain the same number of series."
                ),
                logger=logger,
            )
        same_intersection = list(
            zip(
                *[
                    [anomalies.slice_intersect(series), series.slice_intersect(series)]
                    for (anomalies, series) in zip(actual_series, pred_series)
                ]
            )
        )
        actual_series = list(same_intersection[0])
        pred_series = list(same_intersection[1])

        ret = self._fit_core(actual_series, pred_series)
        self._fit_called = True
        return ret

    def predict(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        self._assert_fit_called()
        return super().predict(series=series)
