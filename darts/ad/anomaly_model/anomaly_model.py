"""
Anomaly models base classes
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Union

import pandas as pd

from darts.ad.scorers.scorers import AnomalyScorer
from darts.ad.utils import eval_metric_from_scores, show_anomalies_from_scores
from darts.logging import get_logger, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts.utils.utils import series2seq

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

        raise_if_not(
            all([isinstance(s, TimeSeries) for s in series2seq(series)]),
            "all input `series` must be of type Timeseries.",
        )

    def _fit_scorers(
        self, list_series: Sequence[TimeSeries], list_pred: Sequence[TimeSeries]
    ):
        """Train the fittable scorers using model forecasts"""
        for scorer in self.scorers:
            if hasattr(scorer, "fit"):
                scorer.fit_from_prediction(list_series, list_pred)

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

    def show_anomalies(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        forecast_horizon: int = 1,
        start: Union[pd.Timestamp, float, int] = 0.5,
        num_samples: int = 1,
        actual_anomalies: TimeSeries = None,
        names_of_scorers: Union[str, Sequence[str]] = None,
        title: str = None,
        metric: str = None,
        **score_kwargs,
    ):
        """Plot the results of the anomaly model.

        Computes the score on the given series input and shows the different anomaly scores with respect to time.

        The plot will be composed of the following:

        - the series itself with the output of the forecasting model.
        - the anomaly score for each scorer. The scorers with different windows will be separated.
        - the actual anomalies, if given.

        It is possible to:

        - add a title to the figure with the parameter `title`
        - give personalized names for the scorers with `names_of_scorers`
        - show the results of a metric for each anomaly score (AUC_ROC or AUC_PR),
            if the actual anomalies are provided.

        Parameters
        ----------
        series
            The series to visualize anomalies from.
        past_covariates
            An optional past-observed covariate series or sequence of series. This applies only if the model
            supports past covariates.
        future_covariates
            An optional future-known covariate series or sequence of series. This applies only if the model
            supports future covariates.
        forecast_horizon
            The forecast horizon for the predictions.
        start
            The first point of time at which a prediction is computed for a future time.
            This parameter supports 3 different data types: ``float``, ``int`` and ``pandas.Timestamp``.
            In the case of ``float``, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of ``int``, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of ``pandas.Timestamp``, this time stamp will be used to determine the first prediction time
            directly.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1 for
            deterministic models.
        actual_anomalies
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        names_of_scorers
            Name of the scores. Must be a list of length equal to the number of scorers in the anomaly_model.
        title
            Title of the figure
        metric
            Optionally, Scoring function to use. Must be one of "AUC_ROC" and "AUC_PR".
            Default: "AUC_ROC"
        score_kwargs
            parameters for the `.score()` function
        """
        if not isinstance(series, TimeSeries):
            raise_log(
                ValueError("`series` must be a single `TimeSeries`."),
                logger=logger,
            )

        raise_if_not(
            isinstance(series, TimeSeries),
            f"`show_anomalies` expects an input of type TimeSeries, found type: {type(series)}.",
        )

        # at the moment, only forecasting_am support these
        if hasattr(self, "filter"):
            score_opt_kwargs = {}
        else:
            score_opt_kwargs = {
                "past_covariates": past_covariates,
                "future_covariates": future_covariates,
                "forecast_horizon": forecast_horizon,
                "start": start,
                "num_samples": num_samples,
            }

        anomaly_scores, model_output = self.score(
            series, return_model_prediction=True, **score_kwargs, **score_opt_kwargs
        )

        return self._show_anomalies(
            series,
            model_output=model_output,
            anomaly_scores=anomaly_scores,
            names_of_scorers=names_of_scorers,
            actual_anomalies=actual_anomalies,
            title=title,
            metric=metric,
        )

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
