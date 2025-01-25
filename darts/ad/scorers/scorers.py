"""
Scorers Base Classes
"""

# TODO:
#     - add stride for Scorers like Kmeans and Wasserstein
#     - add option to normalize the windows for kmeans? capture only the form and not the values.

import copy
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np

from darts import TimeSeries, metrics
from darts.ad.utils import (
    _assert_same_length,
    _check_input,
    _sanity_check_two_series,
    eval_metric_from_scores,
    show_anomalies_from_scores,
)
from darts.logging import get_logger, raise_log
from darts.metrics.metrics import METRIC_TYPE
from darts.utils.data.tabularization import create_lagged_data
from darts.utils.ts_utils import series2seq
from darts.utils.utils import _build_tqdm_iterator, _parallel_apply

logger = get_logger(__name__)


class AnomalyScorer(ABC):
    """Base class for all anomaly scorers"""

    def __init__(self, is_univariate: bool, window: int) -> None:
        """
        Parameters
        ----------
        is_univariate
            Whether the scorer is a univariate scorer.
        window
            Integer value indicating the size of the window W used by the scorer to transform the series into an
            anomaly score. A scorer will slice the given series into subsequences of size W and returns a value
            indicating how anomalous these subset of W values are. A post-processing step will convert this anomaly
            score into a point-wise anomaly score (see definition of `window_transform`). The window size should be
            commensurate to the expected durations of the anomalies one is looking for.
        """
        if window <= 0:
            raise_log(
                ValueError(
                    f"Parameter `window` must be strictly greater than 0, found `{window}`."
                ),
                logger=logger,
            )
        self.window = window
        self._is_univariate = is_univariate

    def score_from_prediction(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Computes the anomaly score on the two (sequence of) series.

        If a pair of sequences is given, they must contain the same number
        of series. The scorer will score each pair of series independently
        and return an anomaly score for each pair.

        Parameters
        ----------
        series:
            The (sequence of) actual series.
        pred_series
            The (sequence of) predicted series.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            (Sequence of) anomaly score time series
        """
        called_with_single_series = isinstance(series, TimeSeries)
        series, pred_series = series2seq(series), series2seq(pred_series)
        name, pred_name = "series", "pred_series"
        _assert_same_length(series, pred_series, name, pred_name)

        pred_scores = []
        for actual, pred in zip(series, pred_series):
            _sanity_check_two_series(actual, pred, name, pred_name)
            index = actual.slice_intersect_times(pred, copy=False)
            self._check_window_size(index)
            scores = self._score_core_from_prediction(
                vals=actual.slice_intersect_values(pred),
                pred_vals=pred.slice_intersect_values(actual),
            )
            scores = TimeSeries.from_times_and_values(
                values=scores,
                times=index,
            )

            if self.window > 1:
                # apply a moving average with window size `self.window` to the anomaly scores starting at `self.window`;
                # series of length `n` will be transformed into a series of length `n-self.window+1`.
                scores = scores.window_transform(
                    transforms={
                        "window": self.window,
                        "function": "mean",
                        "mode": "rolling",
                        "min_periods": self.window,
                    },
                    treat_na="dropna",
                )
            pred_scores.append(scores)
        return pred_scores[0] if called_with_single_series else pred_scores

    def eval_metric_from_prediction(
        self,
        anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
        metric: Literal["AUC_ROC", "AUC_PR"] = "AUC_ROC",
    ) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
        """Computes the anomaly score between `series` and `pred_series`, and returns the score
        of an agnostic threshold metric.

        Parameters
        ----------
        anomalies
            The (sequence of) ground truth binary anomaly series (`1` if it is an anomaly and `0` if not).
        series
            The (sequence of) actual series.
        pred_series
            The (sequence of) predicted series.
        metric
            The name of the metric function to use. Must be one of "AUC_ROC" (Area Under the
            Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
            Default: "AUC_ROC".

        Returns
        -------
        float
            A single metric value for a single univariate `series`.
        Sequence[float]
            A sequence of metric values for:

            - a single multivariate `series`.
            - a sequence of univariate `series`.
        Sequence[Sequence[float]]
            A sequence of sequences of metric values for a sequence of multivariate `series`.
            The outer sequence is over the series, and inner sequence is over the series' components/columns.
        """
        self._check_univariate_scorer(anomalies)
        pred_scores = self.score_from_prediction(series, pred_series)
        return eval_metric_from_scores(
            anomalies=anomalies,
            pred_scores=pred_scores,
            window=self.window,
            metric=metric,
        )

    def show_anomalies_from_prediction(
        self,
        series: TimeSeries,
        pred_series: TimeSeries,
        scorer_name: str = None,
        anomalies: TimeSeries = None,
        title: str = None,
        metric: Optional[Literal["AUC_ROC", "AUC_PR"]] = None,
        component_wise: bool = False,
    ):
        """Plot the results of the scorer.

        Computes the anomaly score on the two series. And plots the results.

        The plot will be composed of the following:
            - the series and the pred_series.
            - the anomaly score of the scorer.
            - the actual anomalies, if given.

        It is possible to:
            - add a title to the figure with the parameter `title`
            - give personalized name to the scorer with `scorer_name`
            - show the results of a metric for the anomaly score (AUC_ROC or AUC_PR),
              if the actual anomalies is provided.

        Parameters
        ----------
        series
            The actual series to visualize anomalies from.
        pred_series
            The predicted series of `series`.
        anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        scorer_name
            Name of the scorer.
        title
            Title of the figure
        metric
            Optionally, the name of the metric function to use. Must be one of "AUC_ROC" (Area Under the
            Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
            Default: "AUC_ROC".
        component_wise
            If True, will separately plot each component in case of multivariate anomaly detection.
        """
        series = _check_input(series, name="series", num_series_expected=1)[0]
        pred_series = _check_input(
            pred_series, name="pred_series", num_series_expected=1
        )[0]
        pred_scores = self.score_from_prediction(series, pred_series)

        if title is None:
            title = f"Anomaly results by scorer {str(self)}"

        if scorer_name is None:
            scorer_name = [f"anomaly score by {str(self)}"]

        return show_anomalies_from_scores(
            series=series,
            anomalies=anomalies,
            pred_series=pred_series,
            pred_scores=pred_scores,
            window=self.window,
            names_of_scorers=scorer_name,
            title=title,
            metric=metric,
            component_wise=component_wise,
        )

    @property
    def is_probabilistic(self) -> bool:
        """Whether the scorer expects a probabilistic prediction as the first input."""
        return False

    @property
    def is_univariate(self) -> bool:
        """Whether the Scorer is a univariate scorer."""
        return self._is_univariate

    @property
    def is_trainable(self) -> bool:
        """Whether the scorer is trainable."""
        return False

    @abstractmethod
    def __str__(self):
        """returns the name of the scorer"""
        pass

    @abstractmethod
    def _score_core_from_prediction(
        self,
        vals: np.ndarray,
        pred_vals: np.ndarray,
    ) -> np.ndarray:
        pass

    def _check_univariate_scorer(
        self, anomalies: Union[TimeSeries, Sequence[TimeSeries]]
    ):
        """Checks if `anomalies` contains only univariate series when the scorer has the
        parameter 'is_univariate' set to True.

        'is_univariate' is:
            True -> when the function of the scorer `score(series)` (or, if applicable,
                `score_from_prediction(series, pred_series)`) returns a univariate
                anomaly score regardless of the input `series` (or, if applicable, `series`
                and `pred_series`).
            False -> when the scorer will return a series that has the
                same number of components as the input (can be univariate or multivariate).
        """

        def _check_univariate(s: TimeSeries):
            """Checks if `anomalies` contains only univariate series, which
            is required if any of the scorers returns a univariate score.
            """
            if self.is_univariate and not s.width == 1:
                raise_log(
                    ValueError(
                        f"Scorer {str(self)} will return a univariate anomaly score series (width=1). "
                        f"Found a multivariate `anomalies`. "
                        f"The evaluation of the accuracy cannot be computed between the two series."
                    ),
                    logger=logger,
                )

        _ = _check_input(anomalies, name="anomalies", extra_checks=_check_univariate)

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
                    f"Scorer {str(self)} is expecting `{name_series}` to be a stochastic "
                    f"timeseries (number of samples must be higher than 1, found: {series.shape[2]}).",
                ),
                logger=logger,
            )

    def _extract_deterministic_series(self, series: TimeSeries, name_series: str):
        """Extract a deterministic series from `series` (quantile=0.5 if `series` is probabilistic)."""
        if series.is_deterministic:
            return series

        logger.warning(
            f"Scorer {str(self)} is expecting `{name_series}` to be a (sequence of) deterministic "
            f"timeseries (number of samples must be equal to 1, found: {series.n_samples}). The series "
            f"will be converted to a deterministic series by taking the median of the samples.",
        )
        return series.quantile_timeseries(quantile=0.5)

    def _extract_deterministic_values(self, series: np.ndarray, name_series: str):
        """Extract deterministic values from `series` (quantile=0.5 if `series` is probabilistic)."""
        if series.shape[2] == 1:
            return series

        logger.warning(
            f"Scorer {str(self)} is expecting `{name_series}` to be a (sequence of) deterministic "
            f"timeseries (number of samples must be equal to 1, found: {series.shape[2]}). The series "
            f"will be converted to a deterministic series by taking the median of the samples.",
        )
        return np.expand_dims(np.quantile(series, q=0.5, axis=2), -1)


class FittableAnomalyScorer(AnomalyScorer):
    """Base class of scorers that require training."""

    def __init__(
        self,
        is_univariate: bool,
        window: int,
        window_agg: bool,
        diff_fn: METRIC_TYPE = metrics.ae,
        n_jobs: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        is_univariate
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
            By default, uses the absolute difference (:func:`~darts.metrics.metrics.ae`).
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when a `Sequence[TimeSeries]` is
            passed as input, parallelising operations regarding different `TimeSeries`. Defaults to `1`
            (sequential). Setting the parameter to `-1` means using all the available processors.
        """
        super().__init__(is_univariate=is_univariate, window=window)
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
        series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    ):
        """Fits the scorer on the two (sequences of) series.

        The function `diff_fn` passed as a parameter to the scorer, will transform `pred_series` and `series`
        into one series. By default, `diff_fn` will compute the absolute difference (Default:
        :func:`~darts.metrics.metrics.ae`). If `pred_series` and `series` are sequences, `diff_fn` will be
        applied to all pairwise elements of the sequences.

        The scorer will then be fitted on this (sequence of) series. If a sequence of series is given,
        the scorer will be fitted on the concatenation of the sequence.

        The scorer assumes that the (sequence of) series is anomaly-free.

        If any of the series is stochastic (with `n_samples>1`), `diff_fn` is computed on quantile `0.5`.

        Parameters
        ----------
        series
            The (sequence of) actual series.
        pred_series
            The (sequence of) predicted series.

        Returns
        -------
        self
            Fitted Scorer.
        """
        series = _check_input(series, "series")
        pred_series = _check_input(pred_series, "pred_series")
        diff_series = self._diff_series(series, pred_series)
        self.fit(diff_series)
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

        called_with_single_series = isinstance(series, TimeSeries)
        series = _check_input(
            series, name="series", extra_checks=self._check_window_size
        )
        series = [self._extract_deterministic_series(s, "series") for s in series]

        pred_scores = self._score_core(series)
        return pred_scores[0] if called_with_single_series else pred_scores

    def score_from_prediction(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Computes the anomaly score on the two (sequence of) series.

        The function `diff_fn` passed as a parameter to the scorer, will transform `pred_series` and `series`
        into one "difference" series. By default, `diff_fn` will compute the absolute difference
        (Default: :func:`~darts.metrics.metrics.ae`).
        If series and pred_series are sequences, `diff_fn` will be applied to all pairwise elements
        of the sequences.

        The scorer will then transform this series into an anomaly score. If a sequence of series is given,
        the scorer will score each series independently and return an anomaly score for each series in the sequence.

        Parameters
        ----------
        series
            The (sequence of) actual series.
        pred_series
            The (sequence of) predicted series.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            (Sequence of) anomaly score time series
        """
        self._check_fit_called()

        called_with_single_series = isinstance(series, TimeSeries)
        series = _check_input(series, "series")
        pred_series = _check_input(pred_series, "pred_series")

        diff = self._diff_series(series, pred_series)
        pred_scores = self.score(diff)
        return pred_scores[0] if called_with_single_series else pred_scores

    def eval_metric(
        self,
        anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
        metric: Literal["AUC_ROC", "AUC_PR"] = "AUC_ROC",
    ) -> Union[float, Sequence[float], Sequence[Sequence[float]]]:
        """Computes the anomaly score of the given time series, and returns the score
        of an agnostic threshold metric.

        Parameters
        ----------
        anomalies
            The (sequence of) ground truth binary anomaly series (`1` if it is an anomaly and `0` if not).
        series
            The (sequence of) series to detect anomalies from.
        metric
            The name of the metric function to use. Must be one of "AUC_ROC" (Area Under the
            Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
            Default: "AUC_ROC".

        Returns
        -------
        float
            A single score/metric for univariate `series` series (with only one component/column).
        Sequence[float]
            A sequence (list) of scores for:

            - multivariate `series` series (multiple components). Gives a score for each component.
            - a sequence (list) of univariate `series` series. Gives a score for each series.
        Sequence[Sequence[float]]
            A sequence of sequences of scores for a sequence of multivariate `series` series.
            Gives a score for each series (outer sequence) and component (inner sequence).
        """
        anomalies = series2seq(anomalies)
        self._check_univariate_scorer(anomalies)
        pred_scores = self.score(series)
        window = 1 if self.window_agg else self.window
        return eval_metric_from_scores(
            anomalies=anomalies,
            pred_scores=pred_scores,
            window=window,
            metric=metric,
        )

    def show_anomalies(
        self,
        series: TimeSeries,
        anomalies: TimeSeries = None,
        scorer_name: str = None,
        title: str = None,
        metric: Optional[Literal["AUC_ROC", "AUC_PR"]] = None,
        component_wise: bool = False,
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
        anomalies
            The (sequence of) ground truth binary anomaly series (`1` if it is an anomaly and `0` if not).
        scorer_name
            Name of the scorer.
        title
            Title of the figure
        metric
            Optionally, the name of the metric function to use. Must be one of "AUC_ROC" (Area Under the
            Receiver Operating Characteristic Curve) and "AUC_PR" (Average Precision from scores).
            Default: "AUC_ROC".
        component_wise
            If True, will separately plot each component in case of multivariate anomaly detection.
        """
        series = _check_input(series, name="series", num_series_expected=1)[0]
        pred_scores = self.score(series)

        if title is None:
            title = f"Anomaly results by scorer {str(self)}"

        if scorer_name is None:
            scorer_name = f"anomaly score by {str(self)}"

        if self.window_agg:
            window = 1
        else:
            window = self.window

        return show_anomalies_from_scores(
            series=series,
            anomalies=anomalies,
            pred_scores=pred_scores,
            window=window,
            names_of_scorers=scorer_name,
            title=title,
            metric=metric,
            component_wise=component_wise,
        )

    @property
    def is_trainable(self) -> bool:
        """Whether the Scorer is trainable."""
        return True

    @abstractmethod
    def _fit_core(self, series: Sequence[TimeSeries], *args, **kwargs):
        pass

    @abstractmethod
    def _score_core(
        self, series: Sequence[TimeSeries], *args, **kwargs
    ) -> Sequence[TimeSeries]:
        pass

    def _score_core_from_prediction(
        self,
        vals: np.ndarray,
        pred_vals: np.ndarray,
    ) -> np.ndarray:
        pass

    def _diff_series(
        self,
        series: Sequence[TimeSeries],
        pred_series: Sequence[TimeSeries],
    ) -> Sequence[TimeSeries]:
        """Applies the `diff_fn` to two sequences of time series. Converts two time series into 1.

        Each series-pair in series and pred_series must:
            - have a non-empty time intersection
            - be of the same width W

        Parameters
        ----------
        series
            A sequence of time series
        pred_series
            A sequence of predicted time series to compute `diff_fn` on.

        Returns
        -------
        Sequence[TimeSeries]
            A sequence of series of width W from the difference between `series` and `pred_series`.
        """
        residuals = self.diff_fn(series, pred_series, component_reduction=None)
        out = []
        for s1, s2, res in zip(series, pred_series, residuals):
            time_index = s2.slice_intersect_times(s1, copy=False)
            out.append(s2.with_times_and_values(times=time_index, values=res))
        return out

    def _fun_window_agg(
        self, scores: Sequence[TimeSeries], window: int
    ) -> Sequence[TimeSeries]:
        """
        Transforms a window-wise anomaly score into a point-wise anomaly score.

        When using a window of size `W`, a scorer will return an anomaly score
        with values that represent how anomalous each past `W` is. If the parameter
        `window_agg` is set to `True` (default value), the scores for each point
        can be assigned by aggregating the anomaly scores for each window the point
        is included in.

        This post-processing step is equivalent to a rolling average of length window
        over the anomaly score series. The return anomaly score represents the abnormality
        of each timestamp.
        """
        # TODO: can we use window_transform here?
        scores_point_wise = []
        for score in scores:
            score_vals = score.all_values(copy=False)
            mean_score = np.empty(score_vals.shape)
            for idx_point in range(len(score)):
                # "look ahead window" to account for the "look behind window" of the scorer
                mean_score[idx_point] = score_vals[idx_point : idx_point + window].mean(
                    axis=0
                )
            score_point_wise = score.with_times_and_values(score.time_index, mean_score)
            scores_point_wise.append(score_point_wise)
        return scores_point_wise

    def _check_fit_called(self):
        """Checks if the scorer has been fitted before calling its `score()` function."""
        if not self._fit_called:
            raise_log(
                ValueError(
                    f"The Scorer {str(self)} has not been fitted yet. Call `fit()` first."
                ),
                logger=logger,
            )


class WindowedAnomalyScorer(FittableAnomalyScorer):
    """Base class for anomaly scorers that rely on windows to detect anomalies"""

    def __init__(
        self,
        is_univariate: bool,
        window: int,
        window_agg: bool,
        diff_fn: METRIC_TYPE,
    ) -> None:
        """
        Parameters
        ----------
        is_univariate
            Whether the scorer is a univariate scorer. If `True` and when using multivariate series, the scores are
            computed on the concatenated components/columns in the considered window to compute one score.
        window
            Integer value indicating the size of the window W used by the scorer to transform the series into an
            anomaly score. A scorer slices the given series into subsequences of size W and returns a value
            indicating how anomalous these subsets of W values are. A post-processing step will convert the anomaly
            scores into point-wise anomaly scores (see definition of `window_transform`). The window size should be
            commensurate to the expected durations of the anomalies one is looking for.
        window_agg
            Whether to transform/aggregate window-wise anomaly scores into point-wise anomaly scores.
        diff_fn
            The differencing function to use to transform the predicted and actual series into one series.
            The scorer is then applied to this series. Must be one of Darts per-time-step metrics (e.g.,
            :func:`~darts.metrics.metrics.ae` for the absolute difference, :func:`~darts.metrics.metrics.err` for the
            difference, :func:`~darts.metrics.metrics.se` for the squared difference, ...).
            By default, uses the absolute difference (:func:`~darts.metrics.metrics.ae`).
        """
        super().__init__(
            is_univariate=is_univariate,
            window=window,
            window_agg=window_agg,
            diff_fn=diff_fn,
        )

    @abstractmethod
    def _model_score_method(self, model, data: np.ndarray) -> np.ndarray:
        """Wrapper around model inference method"""
        pass

    def _fit_core(self, series: Sequence[TimeSeries], *args, **kwargs):
        """Train one sub-model for each component when self.is_univariate=False and series is multivariate"""
        if self.is_univariate or series[0].width == 1:
            self.model.fit(self._tabularize_series(series, component_wise=False))
            return

        tabular_data = self._tabularize_series(series, component_wise=True)
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
        self, series: Sequence[TimeSeries], *args, **kwargs
    ) -> Sequence[TimeSeries]:
        """Apply the scorer (sub) model scoring method on the series components"""
        _ = _check_input(series, "series", width_expected=self.width_trained_on)
        if self.is_univariate or series[0].width == 1:
            # n series * (time, components, samples) -> (n series * (time - (window - 1)),)
            score_vals = self._model_score_method(
                model=self.model,
                data=self._tabularize_series(series, component_wise=False),
            )
            # (n series * (time - (window - 1)),) -> (components=1, n series * (time - (window - 1)))
            score_vals = np.expand_dims(score_vals, 0)
        else:
            # parallelize scoring of components by the corresponding sub-model
            score_iterator = zip(
                self.model,
                self._tabularize_series(series, component_wise=True),
            )
            input_iterator = _build_tqdm_iterator(
                score_iterator, verbose=False, desc=None, total=len(self.model)
            )
            # n series * (time, components, samples) -> (components, n series * (time - (window - 1)))
            score_vals = np.array(
                _parallel_apply(
                    input_iterator,
                    self._model_score_method,
                    n_jobs=self._n_jobs,
                    fn_args=args,
                    fn_kwargs=kwargs,
                )
            )
        # (components, n series * (time - (window - 1))) -> n series * (time - (window - 1), components)
        score_series = self._convert_tabular_to_series(series, score_vals)
        if self.window > 1 and self.window_agg:
            return self._fun_window_agg(score_series, self.window)
        else:
            return score_series

    def _tabularize_series(
        self, series: Sequence[TimeSeries], component_wise: bool
    ) -> np.ndarray:
        """Internal function called by WindowedAnomalyScorer `fit()` and `score()` functions.

        Transforms a sequence of series into tabular data of size window `W`. The parameter `component_wise`
        indicates how the rolling window must treat the different components if the series is multivariate.
        If set to `False`, the rolling window will be done on each component independently. If set to `True`,
        the `N` components will be concatenated to create windows of size `W` * `N`. The resulting tabular
        data of each series are concatenated.

        Returns
        -------
        np.ndarray
            For `component_wise=True`, an array of shape (components, time - (window - 1), window).
            The component dimension is in first place for easy parallelization over all component-wise models.
            For `component_wise=False`, an array of shape (time - (window - 1), window * components).
        """
        # n series * (time, components, sample) -> (time - (window - 1), window * components)
        data = create_lagged_data(
            target_series=series,
            lags=[i for i in range(-self.window, 0)],
            uses_static_covariates=False,
            is_training=False,
            concatenate=True,
        )[0].squeeze(-1)

        # bring into required model input shape
        if component_wise:
            # (time - (window - 1), window * components) -> (time - (window - 1), window, components)
            data = data.reshape((-1, self.window, series[0].width))
            # (time - (window - 1), window, components) -> (components, time - (window - 1), window)
            d_time, d_wind, d_comp = (0, 1, 2)
            data = np.moveaxis(data, [d_time, d_comp], [d_wind, d_time])
        return data

    def _convert_tabular_to_series(
        self, series: Sequence[TimeSeries], score_vals: np.ndarray
    ) -> Sequence[TimeSeries]:
        """Converts generated anomaly score from `np.ndarray` into a sequence of series. For efficiency reasons,
        the anomaly scores were computed in one go (for each component if `component_wise=True`). If a list of series
        is given, each series will be concatenated by its components. The function aims to split the anomaly score at
        the proper indexes to create an anomaly score for each series.
        """
        if not self.is_univariate or self.is_univariate and series[0].width == 1:
            # number of input components matches output components, we can generate a new series
            # with the same attrs, and component names
            create_fn = "with_times_and_values"
        else:
            # otherwise, create a clean new series
            create_fn = "from_times_and_values"

        # (components, n series * (time - (window - 1))) -> (n series * (time - (window - 1)), components)
        score_vals = score_vals.T
        result = []
        idx = 0
        # (n series * (time - (window - 1)), components) -> n series * (time - (window - 1), components)
        for s in series:
            result.append(
                getattr(s, create_fn)(
                    times=s._time_index[self.window - 1 :],
                    values=score_vals[idx : idx + len(s) - self.window + 1, :],
                )
            )
            idx += len(s) - self.window + 1
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
        super().__init__(is_univariate=False, window=window)

    @property
    def is_probabilistic(self) -> bool:
        return True

    def _score_core_from_prediction(
        self,
        vals: np.ndarray,
        pred_vals: np.ndarray,
    ) -> np.ndarray:
        """For each timestamp of the inputs:

        - the parameters of the considered distribution are fitted on the samples of the probabilistic time series
        - the negative log-likelihood of the deterministic time series values are computed

        If the series is multivariate, the score will be computed on each component independently.

        Parameters
        ----------
        vals
            The values of a deterministic time series (number of samples per timestamp must be equal to 1)
        pred_vals
            The values of a probabilistic time series (number of samples per timestamp must be higher than 1)
        time_index
            The time index intersection between `series` and `pred_series`.

        Returns
        -------
        TimeSeries
        """
        vals = self._extract_deterministic_values(vals, "series")
        self._assert_stochastic(pred_vals, "pred_series")

        np_anomaly_scores = []
        for component_idx in range(pred_vals.shape[1]):
            np_anomaly_scores.append(
                self._score_core_nllikelihood(
                    vals[:, component_idx].squeeze(-1),
                    pred_vals[:, component_idx],
                )
            )
        return np.array(np_anomaly_scores).T

    @abstractmethod
    def _score_core_nllikelihood(
        self, vals: np.ndarray, pred_vals: np.ndarray
    ) -> np.ndarray:
        """For each timestamp, the corresponding distribution is fitted on the probabilistic time-series
        input_2, and returns the negative log-likelihood of the deterministic time-series input_1
        given the distribution.
        """
        pass
