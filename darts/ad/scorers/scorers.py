"""
Scorers Base Classes
"""

# TODO:
#     - add stride for Scorers like Kmeans and Wasserstein
#     - add option to normalize the windows for kmeans? capture only the form and not the values.

import copy
import sys
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Union

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from darts import TimeSeries, metrics
from darts.ad.utils import (
    _assert_same_length,
    _assert_timeseries,
    _check_input,
    _intersect,
    _sanity_check_two_series,
    eval_metric_from_scores,
    show_anomalies_from_scores,
)
from darts.logging import get_logger, raise_log
from darts.metrics.metrics import METRIC_TYPE
from darts.utils.ts_utils import series2seq
from darts.utils.utils import _build_tqdm_iterator, _parallel_apply

logger = get_logger(__name__)


class AnomalyScorer(ABC):
    """Base class for all anomaly scorers"""

    def __init__(
        self, univariate_scorer: bool, window: int, trainable: bool = False
    ) -> None:
        """
        Parameters
        ----------
        univariate_scorer
            Whether the scorer is a univariate scorer.
        window
            Integer value indicating the size of the window W used by the scorer to transform the series into an
            anomaly score. A scorer will slice the given series into subsequences of size W and returns a value
            indicating how anomalous these subset of W values are. A post-processing step will convert this anomaly
            score into a point-wise anomaly score (see definition of `window_transform`). The window size should be
            commensurate to the expected durations of the anomalies one is looking for.
        trainable
            Whether the scorer is trainable.
        """
        if type(window) is not int:  # noqa: E721
            raise_log(
                ValueError(
                    f"Parameter `window` must be an integer, found type {type(window)}."
                ),
                logger=logger,
            )
        if not window > 0:
            raise_log(
                ValueError(
                    f"Parameter `window` must be strictly greater than 0, found `{window}`."
                ),
                logger=logger,
            )
        self.window = window
        self.univariate_scorer = univariate_scorer
        self.trainable = trainable

    def score_from_prediction(
        self,
        actual_series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Computes the anomaly score on the two (sequence of) series.

        If a pair of sequences is given, they must contain the same number
        of series. The scorer will score each pair of series independently
        and return an anomaly score for each pair.

        Parameters
        ----------
        actual_series:
            The (sequence of) actual series.
        pred_series
            The (sequence of) predicted series.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            (Sequence of) anomaly score time series
        """
        called_with_single_series = isinstance(actual_series, TimeSeries)
        actual_series, pred_series = series2seq(actual_series), series2seq(pred_series)
        actual_name, pred_name = "actual_series", "pred_series"
        _assert_same_length(actual_series, pred_series, actual_name, pred_name)

        anomaly_scores = []
        for actual, pred in zip(actual_series, pred_series):
            _sanity_check_two_series(actual, pred, actual_name, pred_name)
            index = actual.slice_intersect_times(pred, copy=False)
            self._check_window_size(index)
            anomaly_scores.append(
                self._score_core_from_prediction(
                    actual.slice_intersect_values(pred),
                    pred.slice_intersect_values(actual),
                    index,
                )
            )
        return anomaly_scores[0] if called_with_single_series else anomaly_scores

    def eval_metric_from_prediction(
        self,
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        actual_series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
        metric: str = "AUC_ROC",
    ) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
        """Computes the anomaly score between `actual_series` and `pred_series`, and returns the score
        of an agnostic threshold metric.

        Parameters
        ----------
        actual_anomalies
            The (sequence of) ground truth binary anomaly series (`1` if it is an anomaly and `0` if not).
        actual_series
            The (sequence of) actual series.
        pred_series
            The (sequence of) predicted series.
        metric
            The name of the metric function to use. Must be one of "AUC_ROC" (Area Under the
            Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
            Default: "AUC_ROC"

        Returns
        -------
        float
            A single metric value for a single univariate `actual_series`.
        Sequence[float]
            A sequence of metric values for:

            - a single multivariate `actual_series`.
            - a sequence of univariate `actual_series`.
        Sequence[Sequence[float]]
            A sequence of sequences of metric values for a sequence of multivariate `actual_series`.
            The outer sequence is over the series, and inner sequence is over the series' components/columns.
        """
        self._check_univariate_scorer(actual_anomalies)
        anomaly_score = self.score_from_prediction(actual_series, pred_series)
        return eval_metric_from_scores(
            actual_anomalies, anomaly_score, self.window, metric
        )

    def show_anomalies_from_prediction(
        self,
        actual_series: TimeSeries,
        pred_series: TimeSeries,
        scorer_name: str = None,
        actual_anomalies: TimeSeries = None,
        title: str = None,
        metric: str = None,
    ):
        """Plot the results of the scorer.

        Computes the anomaly score on the two series. And plots the results.

        The plot will be composed of the following:
            - the actual_series and the pred_series.
            - the anomaly score of the scorer.
            - the actual anomalies, if given.

        It is possible to:
            - add a title to the figure with the parameter `title`
            - give personalized name to the scorer with `scorer_name`
            - show the results of a metric for the anomaly score (AUC_ROC or AUC_PR),
              if the actual anomalies is provided.

        Parameters
        ----------
        actual_series
            The actual series to visualize anomalies from.
        pred_series
            The predicted series of `actual_series`.
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        scorer_name
            Name of the scorer.
        title
            Title of the figure
        metric
            The name of the metric function to use. Must be one of "AUC_ROC" (Area Under the
            Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
            Default: "AUC_ROC"
        """
        actual_series = _check_input(
            actual_series, name="series", num_series_expected=1
        )[0]
        pred_series = _check_input(pred_series, name="series", num_series_expected=1)[0]
        anomaly_score = self.score_from_prediction(actual_series, pred_series)

        if title is None:
            title = f"Anomaly results by scorer {self.__str__()}"

        if scorer_name is None:
            scorer_name = [f"anomaly score by {self.__str__()}"]

        return show_anomalies_from_scores(
            actual_series,
            model_output=pred_series,
            anomaly_scores=anomaly_score,
            window=self.window,
            names_of_scorers=scorer_name,
            actual_anomalies=actual_anomalies,
            title=title,
            metric=metric,
        )

    @property
    def is_probabilistic(self) -> bool:
        """Whether the scorer expects a probabilistic prediction as the first input."""
        return False

    @abstractmethod
    def __str__(self):
        """returns the name of the scorer"""
        pass

    @abstractmethod
    def _score_core_from_prediction(
        self,
        actual_series: np.ndarray,
        pred_series: np.ndarray,
        time_index: Union[pd.DatetimeIndex, pd.RangeIndex],
    ) -> TimeSeries:
        pass

    def _check_univariate_scorer(
        self, actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]]
    ):
        """Checks if `actual_anomalies` contains only univariate series when the scorer has the
        parameter 'univariate_scorer' set to True.

        'univariate_scorer' is:
            True -> when the function of the scorer ``score(series)`` (or, if applicable,
                ``score_from_prediction(actual_series, pred_series)``) returns a univariate
                anomaly score regardless of the input `series` (or, if applicable, `actual_series`
                and `pred_series`).
            False -> when the scorer will return a series that has the
                same number of components as the input (can be univariate or multivariate).
        """

        def _check_univariate(s: TimeSeries):
            """Checks if `actual_anomalies` contains only univariate series, which
            is required if any of the scorers returns a univariate score.
            """
            if self.univariate_scorer and not s.width == 1:
                raise_log(
                    ValueError(
                        f"Scorer {self.__str__()} will return a univariate anomaly score series (width=1). "
                        f"Found a multivariate `actual_anomalies`. "
                        f"The evaluation of the accuracy cannot be computed between the two series."
                    ),
                    logger=logger,
                )

        _ = _check_input(
            actual_anomalies, name="actual_anomalies", extra_checks=_check_univariate
        )

    def _check_window_size(self, series: Sequence):
        """Checks if the parameter window is less or equal than the length of the given series"""
        if not self.window <= len(series):
            raise_log(
                ValueError(
                    f"Window size {self.window} is greater than the targeted series length {len(series)}, "
                    f"must be lower or equal. Decrease the window size or increase the length series "
                    f"input to score on."
                ),
                logger=logger,
            )

    def _assert_stochastic(self, series: np.ndarray, name_series: str):
        """Checks if the series is stochastic (number of samples is larger than one)."""
        if not series.shape[2] > 1:
            raise_log(
                ValueError(
                    f"Scorer {self.__str__()} is expecting `{name_series}` to be a stochastic "
                    f"timeseries (number of samples must be higher than 1, found: {series.shape[2]}).",
                ),
                logger=logger,
            )

    def _extract_deterministic(self, series: TimeSeries, name_series: str):
        """Extract a deterministic series from `series` (quantile=0.5 if `series` is probabilistic)."""
        if not series.is_deterministic:
            logger.warning(
                f"Scorer {self.__str__()} is expecting `{name_series}` to be a (sequence of) deterministic "
                f"timeseries (number of samples must be equal to 1, found: {series.n_samples}). The series "
                f"will be converted to a deterministic series by taking the median of the samples.",
            )
            series = series.quantile_timeseries(quantile=0.5)
        return series

    def _extract_deterministic2(self, series: np.ndarray, name_series: str):
        """Extract a deterministic series from `series` (quantile=0.5 if `series` is probabilistic)."""
        if series.shape[2] == 1:
            return series

        logger.warning(
            f"Scorer {self.__str__()} is expecting `{name_series}` to be a (sequence of) deterministic "
            f"timeseries (number of samples must be equal to 1, found: {series.shape[2]}). The series "
            f"will be converted to a deterministic series by taking the median of the samples.",
        )
        return np.expand_dims(np.quantile(series, q=0.5, axis=2), -1)


class FittableAnomalyScorer(AnomalyScorer):
    """Base class of scorers that require training."""

    def __init__(
        self,
        univariate_scorer: bool,
        window: int,
        window_agg: bool,
        diff_fn: METRIC_TYPE = metrics.ae,
        n_jobs: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        univariate_scorer
            Whether the scorer is a univariate scorer.
        window
            Integer value indicating the size of the window W used by the scorer to transform the series into an
            anomaly score. A scorer will slice the given series into subsequences of size W and returns a value
            indicating how anomalous these subset of W values are. A post-processing step will convert this anomaly
            score into a point-wise anomaly score (see definition of `window_transform`). The window size should be
            commensurate to the expected durations of the anomalies one is looking for.
        window_agg
            Whether to transform/aggregate window-wise anomaly scores into a point-wise anomaly scores.
        diff_fn
            The differencing function to use to transform the predicted and actual series into one series.
            The scorer is then applied to this series. Must be one of Darts per-time-step metrics (e.g.,
            :func:`~darts.metrics.metrics.ae` for the absolute difference, :func:`~darts.metrics.metrics.err` for the
            difference, :func:`~darts.metrics.metrics.se` for the squared difference, ...).
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is
            passed as input, parallelising operations regarding different ``TimeSeries``. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
        """
        super().__init__(
            univariate_scorer=univariate_scorer, window=window, trainable=True
        )
        if diff_fn not in metrics.TIME_DEPENDENT_METRICS:
            valid_metrics = [m.__name__ for m in metrics.TIME_DEPENDENT_METRICS]
            raise_log(
                ValueError(
                    f"`diff_fn` must be one of Darts 'per time step' metrics "
                    f"{valid_metrics}. Found `{diff_fn}`"
                ),
                logger=logger,
            )
        self.diff_fn = diff_fn
        self.window_agg = window_agg
        self._n_jobs = n_jobs

        # indicates if the scorer has been trained yet
        self._fit_called = False
        self.width_trained_on: Optional[int] = None

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Self:
        """Fits the scorer on the given time series.

        If a sequence of series, the scorer is fitted on the concatenation of the sequence.

        The assumption is that `series` is generally anomaly-free.

        Parameters
        ----------
        series
            The (sequence of) series with no anomalies.

        Returns
        -------
        self
            Fitted Scorer.
        """
        width = series2seq(series)[0].width
        series = _check_input(
            series,
            name="series",
            width_expected=width,
            extra_checks=self._check_window_size,
        )
        self.width_trained_on = width
        self._fit_core(series)
        self._fit_called = True
        return self

    def fit_from_prediction(
        self,
        actual_series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    ):
        """Fits the scorer on the two (sequences of) series.

        The function ``diff_fn`` passed as a parameter to the scorer, will transform `pred_series` and `actual_series`
        into one series. By default, ``diff_fn`` will compute the absolute difference (Default:
        :func:`~darts.metrics.metrics.ae`). If `pred_series` and `actual_series` are sequences, ``diff_fn`` will be
        applied to all pairwise elements of the sequences.

        The scorer will then be fitted on this (sequence of) series. If a sequence of series is given,
        the scorer will be fitted on the concatenation of the sequence.

        The scorer assumes that the (sequence of) actual_series is anomaly-free.

        Parameters
        ----------
        actual_series
            The (sequence of) actual series.
        pred_series
            The (sequence of) predicted series.

        Returns
        -------
        self
            Fitted Scorer.
        """
        actual_series = _check_input(actual_series, "actual_series")
        pred_series = _check_input(pred_series, "pred_series")
        series = self._diff_series(actual_series, pred_series)
        self.fit(series)
        self._fit_called = True

    def score(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Computes the anomaly score on the given series.

        If a sequence of series is given, the scorer will score each series independently
        and return an anomaly score for each series in the sequence.

        Parameters
        ----------
        series
            The (sequence of) series to detect anomalies from.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            (Sequence of) anomaly score time series
        """

        self._check_fit_called()

        list_series = series2seq(series)

        for s in list_series:
            _assert_timeseries(s)
            self._check_window_size(s)

        list_series = [self._extract_deterministic(s, "series") for s in list_series]

        anomaly_scores = self._score_core(list_series)

        if len(anomaly_scores) == 1 and not isinstance(series, Sequence):
            return anomaly_scores[0]
        else:
            return anomaly_scores

    def score_from_prediction(
        self,
        actual_series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Computes the anomaly score on the two (sequence of) series.

        The function ``diff_fn`` passed as a parameter to the scorer, will transform `pred_series` and `actual_series`
        into one "difference" series. By default, ``diff_fn`` will compute the absolute difference
        (Default: :func:`~darts.metrics.metrics.ae`).
        If actual_series and pred_series are sequences, ``diff_fn`` will be applied to all pairwise elements
        of the sequences.

        The scorer will then transform this series into an anomaly score. If a sequence of series is given,
        the scorer will score each series independently and return an anomaly score for each series in the sequence.

        Parameters
        ----------
        actual_series
            The (sequence of) actual series.
        pred_series
            The (sequence of) predicted series.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            (Sequence of) anomaly score time series
        """
        self._check_fit_called()
        called_with_single_series = isinstance(actual_series, TimeSeries)
        actual_series = _check_input(actual_series, "actual_series")
        pred_series = _check_input(pred_series, "pred_series")
        diff = self._diff_series(actual_series, pred_series)
        anomaly_scores = self.score(diff)
        return anomaly_scores[0] if called_with_single_series else anomaly_scores

    def eval_metric(
        self,
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
        metric: str = "AUC_ROC",
    ) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
        """Computes the anomaly score of the given time series, and returns the score
        of an agnostic threshold metric.

        Parameters
        ----------
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        series
            The (sequence of) series to detect anomalies from.
        metric
            Optionally, metric function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"

        Returns
        -------
        Union[float, Sequence[float], Sequence[Sequence[float]]]
            Score of an agnostic threshold metric for the computed anomaly score
                - ``float`` if `series` is a univariate series (dimension=1).
                - ``Sequence[float]``

                    * if `series` is a multivariate series (dimension>1), returns one
                    value per dimension, or
                    * if `series` is a sequence of univariate series, returns one value
                    per series
                - ``Sequence[Sequence[float]]]`` if `series` is a sequence of multivariate
                series. Outer Sequence is over the sequence input and the inner Sequence
                is over the dimensions of each element in the sequence input.
        """
        actual_anomalies = series2seq(actual_anomalies)
        self._check_univariate_scorer(actual_anomalies)
        anomaly_score = self.score(series)
        window = 1 if self.window_agg else self.window
        return eval_metric_from_scores(actual_anomalies, anomaly_score, window, metric)

    def show_anomalies(
        self,
        series: TimeSeries,
        actual_anomalies: TimeSeries = None,
        scorer_name: str = None,
        title: str = None,
        metric: str = None,
    ):
        """Plot the results of the scorer.

        Computes the score on the given series input. And plots the results.

        The plot will be composed of the following:
            - the series itself.
            - the anomaly score of the score.
            - the actual anomalies, if given.

        It is possible to:
            - add a title to the figure with the parameter `title`
            - give personalized name to the scorer with `scorer_name`
            - show the results of a metric for the anomaly score (AUC_ROC or AUC_PR),
            if the actual anomalies is provided.

        Parameters
        ----------
        series
            The series to visualize anomalies from.
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        scorer_name
            Name of the scorer.
        title
            Title of the figure
        metric
            The name of the metric function to use. Must be one of "AUC_ROC" (Area Under the
            Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
            Default: "AUC_ROC"
        """

        if isinstance(series, Sequence):
            if not len(series) == 1:
                raise_log(
                    ValueError(
                        f"``show_anomalies`` expects one series for `series`, found a list "
                        f"of length {len(series)} as input."
                    ),
                    logger=logger,
                )
            series = series[0]

        if not isinstance(series, TimeSeries):
            raise_log(
                ValueError(
                    f"``show_anomalies`` expects an input of type TimeSeries, "
                    f"found type {type(series)} for `series`."
                ),
                logger=logger,
            )

        anomaly_score = self.score(series)

        if title is None:
            title = f"Anomaly results by scorer {self.__str__()}"

        if scorer_name is None:
            scorer_name = f"anomaly score by {self.__str__()}"

        if self.window_agg:
            window = 1
        else:
            window = self.window

        return show_anomalies_from_scores(
            series,
            anomaly_scores=anomaly_score,
            window=window,
            names_of_scorers=scorer_name,
            actual_anomalies=actual_anomalies,
            title=title,
            metric=metric,
        )

    @abstractmethod
    def _fit_core(self, series: Any) -> Any:
        pass

    @abstractmethod
    def _score_core(self, series: Any) -> Any:
        pass

    def _score_core_from_prediction(
        self,
        actual_series: np.ndarray,
        pred_series: np.ndarray,
        time_index: Union[pd.DatetimeIndex, pd.RangeIndex],
    ) -> TimeSeries:
        pass

    def _diff_series(
        self,
        series_1: Sequence[TimeSeries],
        series_2: Sequence[TimeSeries],
    ) -> Sequence[TimeSeries]:
        """Applies the ``diff_fn`` to two sequences of time series. Converts two time series into 1.

        Each series-pair in series_1 and series_2 must:
            - have a non-empty time intersection
            - be of the same width W

        Parameters
        ----------
        series_1
            A sequence of time series
        series_2:
            A sequence of other time series to compute ``diff_fn`` on.

        Returns
        -------
        Sequence[TimeSeries]
            A sequence of series of width W from the difference between `series_1` and `series_2`.
        """
        series_diffs = self.diff_fn(series_1, series_2, component_reduction=None)
        out = []
        for s1, s2, s_diff in zip(series_1, series_2, series_diffs):
            time_index = s2.slice_intersect_times(s1, copy=False)
            out.append(s2.with_times_and_values(times=time_index, values=s_diff))
        return out

    def _fun_window_agg(
        self, list_scores: Sequence[TimeSeries], window: int
    ) -> Sequence[TimeSeries]:
        """
        Transforms a window-wise anomaly score into a point-wise anomaly score.

        When using a window of size `W`, a scorer will return an anomaly score
        with values that represent how anomalous each past `W` is. If the parameter
        `window_agg` is set to True (default value), the scores for each point
        can be assigned by aggregating the anomaly scores for each window the point
        is included in.

        This post-processing step is equivalent to a rolling average of length window
        over the anomaly score series. The return anomaly score represents the abnormality
        of each timestamp.
        """
        list_scores_point_wise = []
        for score in list_scores:
            mean_score = np.empty(score.all_values().shape)
            for idx_point in range(len(score)):
                # "look ahead window" to account for the "look behind window" of the scorer
                mean_score[idx_point] = score.all_values(copy=False)[
                    idx_point : idx_point + window
                ].mean(axis=0)

            score_point_wise = TimeSeries.from_times_and_values(
                score.time_index, mean_score, columns=score.components
            )

            list_scores_point_wise.append(score_point_wise)

        return list_scores_point_wise

    def _check_fit_called(self):
        """Checks if the scorer has been fitted before calling its `score()` function."""
        if not self._fit_called:
            raise_log(
                ValueError(
                    f"The Scorer {self.__str__()} has not been fitted yet. Call ``fit()`` first."
                ),
                logger=logger,
            )


class WindowedAnomalyScorer(FittableAnomalyScorer):
    """Base class for anomaly scorers that rely on windows to detect anomalies"""

    def __init__(
        self,
        window: int,
        univariate_scorer: bool,
        window_agg: bool,
        diff_fn: METRIC_TYPE = metrics.ae,
    ) -> None:
        """
        Parameters
        ----------
        window
            Integer value indicating the size of the window W used by the scorer to transform the series into an
            anomaly score. A scorer will slice the given series into subsequences of size W and returns a value
            indicating how anomalous these subset of W values are. A post-processing step will convert this anomaly
            score into a point-wise anomaly score (see definition of `window_transform`). The window size should be
            commensurate to the expected durations of the anomalies one is looking for.
        univariate_scorer
            Whether the scorer is a univariate scorer.
        window_agg
            Whether to transform/aggregate window-wise anomaly scores into a point-wise anomaly scores.
        diff_fn
            The differencing function to use to transform the predicted and actual series into one series.
            The scorer is then applied to this series. Must be one of Darts per-time-step metrics (e.g.,
            :func:`~darts.metrics.metrics.ae` for the absolute difference, :func:`~darts.metrics.metrics.err` for the
            difference, :func:`~darts.metrics.metrics.se` for the squared difference, ...).
        """

        super().__init__(
            window=window,
            univariate_scorer=univariate_scorer,
            window_agg=window_agg,
            diff_fn=diff_fn,
        )

    @abstractmethod
    def _model_score_method(self, model, data: np.ndarray) -> np.ndarray:
        """Wrapper around model inference method"""
        pass

    def _fit_core(
        self,
        list_series: Sequence[TimeSeries],
        *args,
        **kwargs,
    ):
        """Train one sub-model for each component when self.component_wise=True and series is multivariate"""

        if (not self.component_wise) | (list_series[0].width == 1):
            self.model.fit(
                self._tabularize_series(
                    list_series, concatenate=True, component_wise=False
                )
            )
        else:
            tabular_data = self._tabularize_series(
                list_series, concatenate=True, component_wise=True
            )
            # parallelize fitting of the component-wise models
            fit_iterator = zip(tabular_data, [None] * len(tabular_data))
            input_iterator = _build_tqdm_iterator(
                fit_iterator, verbose=False, desc=None, total=tabular_data.shape[1]
            )
            self.model = _parallel_apply(
                input_iterator,
                copy.deepcopy(self.model).fit,
                n_jobs=self._n_jobs,
                fn_args=args,
                fn_kwargs=kwargs,
            )

    def _score_core(
        self, list_series: Sequence[TimeSeries], *args, **kwargs
    ) -> Sequence[TimeSeries]:
        """Apply the scorer (sub) model scoring method on the series components"""
        if not all([(self.width_trained_on == series.width) for series in list_series]):
            raise_log(
                ValueError(
                    f"All series in 'series' must have the same number of components as "
                    f"the data used for training the model, expected "
                    f"{self.width_trained_on} components."
                ),
                logger=logger,
            )
        if (not self.component_wise) | (list_series[0].width == 1):
            list_np_anomaly_score = [
                self._model_score_method(model=self.model, data=tabular_data)
                for tabular_data in self._tabularize_series(
                    list_series, concatenate=False, component_wise=False
                )
            ]
            list_anomaly_score = [
                TimeSeries.from_times_and_values(
                    series.time_index[self.window - 1 :], np_anomaly_score
                )
                for series, np_anomaly_score in zip(list_series, list_np_anomaly_score)
            ]

        else:
            # parallelize scoring of components by the corresponding sub-model
            score_iterator = zip(
                self.model,
                self._tabularize_series(
                    list_series, concatenate=True, component_wise=True
                ),
            )
            input_iterator = _build_tqdm_iterator(
                score_iterator, verbose=False, desc=None, total=len(self.model)
            )

            list_np_anomaly_score = np.array(
                _parallel_apply(
                    input_iterator,
                    self._model_score_method,
                    n_jobs=self._n_jobs,
                    fn_args=args,
                    fn_kwargs=kwargs,
                )
            )

            list_anomaly_score = self._convert_tabular_to_series(
                list_series, list_np_anomaly_score
            )

        if self.window > 1 and self.window_agg:
            return self._fun_window_agg(list_anomaly_score, self.window)
        else:
            return list_anomaly_score

    def _tabularize_series(
        self, list_series: Sequence[TimeSeries], concatenate: bool, component_wise: bool
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        """Internal function called by WindowedAnomalyScorer ``fit()`` and ``score()`` functions.

        Transforms a (sequence of) series into tabular data of size window `W`. The parameter `component_wise`
        indicates how the rolling window must treat the different components if the series is multivariate.
        If set to False, the rolling window will be done on each component independently. If set to True,
        the `N` components will be concatenated to create windows of size `W` * `N`. The resulting tabular
        data of each series are concatenated if the parameter `concatenate` is set to True.
        """

        list_np_series = [series.all_values(copy=False) for series in list_series]

        if component_wise:

            tabular_data = [
                np.stack(
                    sliding_window_view(arr, window_shape=self.window, axis=0), axis=1
                ).reshape(len(arr[0]), -1, self.window)
                for arr in list_np_series
            ]

        else:

            tabular_data = [
                sliding_window_view(arr, window_shape=self.window, axis=0).reshape(
                    -1, self.window * len(arr[0])
                )
                for arr in list_np_series
            ]

        if concatenate:
            return np.concatenate(tabular_data, axis=1 if component_wise else 0)

        return tabular_data

    def _convert_tabular_to_series(
        self, list_series: Sequence[TimeSeries], list_np_anomaly_score: np.ndarray
    ) -> Sequence[TimeSeries]:
        """Internal function called by WindowedAnomalyScorer  ``score()`` functions when the parameter
        `component_wise` is set to True and the (sequence of) series has more than 1 component.

        Returns the resulting anomaly score as a (sequence of) series, from the np.array `list_np_anomaly_score`
        containing the anomaly score of the (sequence of) series. For efficiency reasons, the anomaly scores were
        computed in one go for each component (component-wise is set to True, so each component has its own fitted
        model). If a list of series is given, each series will be concatenated by its components. The function
        aims to split the anomaly score at the proper indexes to create an anomaly score for each series.
        """
        indice = 0
        result = []
        for series in list_series:
            result.append(
                TimeSeries.from_times_and_values(
                    series.time_index[self.window - 1 :],
                    list(
                        zip(
                            *list_np_anomaly_score[
                                :, indice : indice + len(series) - self.window + 1
                            ]
                        )
                    ),
                )
            )
            indice += len(series) - self.window + 1

        return result


class NLLScorer(AnomalyScorer):
    """Parent class for all LikelihoodScorer"""

    def __init__(self, window) -> None:
        """
        Parameters
        ----------
        window
            Integer value indicating the size of the window W used by the scorer to transform the series into an
            anomaly score. A scorer will slice the given series into subsequences of size W and returns a value
            indicating how anomalous these subset of W values are. A post-processing step will convert this anomaly
            score into a point-wise anomaly score (see definition of `window_transform`). The window size should be
            commensurate to the expected durations of the anomalies one is looking for.
        """
        super().__init__(univariate_scorer=False, window=window)

    @property
    def is_probabilistic(self) -> bool:
        return True

    def _score_core_from_prediction(
        self,
        actual_series: np.ndarray,
        pred_series: np.ndarray,
        time_index: Union[pd.DatetimeIndex, pd.RangeIndex],
    ) -> TimeSeries:
        """For each timestamp of the inputs:
            - the parameters of the considered distribution are fitted on the samples of the probabilistic time series
            - the negative log-likelihood of the deterministic time series values are computed

        If the series is multivariate, the score will be computed on each component independently.

        Parameters
        ----------
        actual_series:
            The values of a deterministic time series (number of samples per timestamp must be equal to 1)
        pred_series
            The values of a probabilistic time series (number of samples per timestamp must be higher than 1)
        time_index
            The time index intersection between `actual_series` and `pred_series`.

        Returns
        -------
        TimeSeries
        """
        actual_series = self._extract_deterministic2(actual_series, "actual_series")
        self._assert_stochastic(pred_series, "pred_series")

        np_anomaly_scores = []
        for component_idx in range(pred_series.shape[1]):
            np_anomaly_scores.append(
                self._score_core_nllikelihood(
                    # shape actual: (time_steps,)
                    # shape pred: (time_steps, samples)
                    actual_series[:, component_idx].squeeze(-1),
                    pred_series[:, component_idx],
                )
            )

        anomaly_scores = TimeSeries.from_times_and_values(
            values=list(zip(*np_anomaly_scores)),
            times=time_index,
        )

        if self.window == 1:
            # the process results in replacing every value by itself -> return directly the series
            return anomaly_scores

        # apply a moving average with window size `self.window` to the anomaly scores starting at `self.window`;
        # series of length `n` will be transformed into a series of length `n-self.window+1`.
        return anomaly_scores.window_transform(
            transforms={
                "window": self.window,
                "function": "mean",
                "mode": "rolling",
                "min_periods": self.window,
            },
            treat_na="dropna",
        )

    @abstractmethod
    def _score_core_nllikelihood(self, input_1: Any, input_2: Any) -> Any:
        """For each timestamp, the corresponding distribution is fitted on the probabilistic time-series
        input_2, and returns the negative log-likelihood of the deterministic time-series input_1
        given the distribution.
        """
        pass
