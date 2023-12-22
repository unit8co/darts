"""
Anomaly models base classes
"""

from abc import ABC, abstractmethod
from typing import Dict, Sequence, Union

from darts.ad.scorers.scorers import AnomalyScorer
from darts.ad.utils import eval_metric_from_scores, show_anomalies_from_scores
from darts.logging import get_logger, raise_if_not
from darts.timeseries import TimeSeries

logger = get_logger(__name__)


class AnomalyModel(ABC):
    """Base class for all anomaly models."""

    def __init__(self, model, scorer):

        self.scorers = [scorer] if not isinstance(scorer, Sequence) else scorer

        raise_if_not(
            all([isinstance(s, AnomalyScorer) for s in self.scorers]),
            "all scorers must be of instance darts.ad.scorers.AnomalyScorer.",
        )

        self.scorers_are_trainable = any(s.trainable for s in self.scorers)
        self.univariate_scoring = any(s.univariate_scorer for s in self.scorers)

        self.model = model

    def _check_univariate(self, actual_anomalies):
        """Checks if `actual_anomalies` contains only univariate series, which
        is required if any of the scorers returns a univariate score.
        """

        if self.univariate_scoring:
            raise_if_not(
                all([s.width == 1 for s in actual_anomalies]),
                "Anomaly model contains scorer {} that will return".format(
                    [s.__str__() for s in self.scorers if s.univariate_scorer]
                )
                + " a univariate anomaly score series (width=1). Found a"
                + " multivariate `actual_anomalies`. The evaluation of the"
                + " accuracy cannot be computed. If applicable, think about"
                + " setting the scorer parameter `componenet_wise` to True.",
            )

    @abstractmethod
    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        allow_model_training: bool,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        raise_if_not(
            type(allow_model_training) is bool,
            f"`allow_filter_training` must be Boolean, found type: {type(allow_model_training)}.",
        )

        if not allow_model_training and not self.scorers_are_trainable:
            logger.warning(
                f"The model {self.model.__class__.__name__} won't be trained"
                + " because the parameter `allow_filter_training` is set to False, and no scorer"
                + " is fittable. The ``.fit()`` function has no effect."
            )
            return

    @abstractmethod
    def score(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        pass

    @abstractmethod
    def eval_metric(
        self,
        actual_anomalies: Union[TimeSeries, Sequence[TimeSeries]],
        series: Union[TimeSeries, Sequence[TimeSeries]],
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        pass

    @abstractmethod
    def show_anomalies(self, series: TimeSeries):
        pass

    def _show_anomalies(
        self,
        series: TimeSeries,
        model_output: TimeSeries = None,
        anomaly_scores: Union[TimeSeries, Sequence[TimeSeries]] = None,
        names_of_scorers: Union[str, Sequence[str]] = None,
        actual_anomalies: TimeSeries = None,
        title: str = None,
        metric: str = None,
    ):
        """Internal function that plots the results of the anomaly model.
        Called by the function show_anomalies().
        """

        if title is None:
            title = f"Anomaly results ({self.model.__class__.__name__})"

        if names_of_scorers is None:
            names_of_scorers = [s.__str__() for s in self.scorers]

        list_window = [s.window for s in self.scorers]

        return show_anomalies_from_scores(
            series,
            model_output=model_output,
            anomaly_scores=anomaly_scores,
            window=list_window,
            names_of_scorers=names_of_scorers,
            actual_anomalies=actual_anomalies,
            title=title,
            metric=metric,
        )

    def _eval_metric_from_scores(
        self,
        list_actual_anomalies: Sequence[TimeSeries],
        list_anomaly_scores: Sequence[TimeSeries],
        metric: str,
    ) -> Union[Sequence[Dict[str, float]], Sequence[Dict[str, Sequence[float]]]]:
        """Internal function that computes the metric over the anomaly scores
        computed by the model. Called by the function eval_metric().
        """
        windows = [s.window for s in self.scorers]

        # create a list of unique names for each scorer that
        # will be used as keys for the dictionary containing
        # the accuracy of each scorer.
        name_scorers = []
        for scorer in self.scorers:
            name = scorer.__str__() + "_w=" + str(scorer.window)

            if name in name_scorers:
                i = 1
                new_name = name + "_" + str(i)
                while new_name in name_scorers:
                    i = i + 1
                    new_name = name + "_" + str(i)
                name = new_name

            name_scorers.append(name)

        acc = []
        for anomalies, scores in zip(list_actual_anomalies, list_anomaly_scores):
            acc.append(
                eval_metric_from_scores(
                    actual_anomalies=anomalies,
                    anomaly_score=scores,
                    window=windows,
                    metric=metric,
                )
            )

        return [dict(zip(name_scorers, scorer_values)) for scorer_values in acc]
