"""
Anomaly Model
------------

An anomaly model expects a model and one or multiple scorers.
It offers a :func:`score()` function, which takes
as input one or multiple series, and returns their anomaly score(s) as ``TimeSeries``.

The model can be a forecasting method (:class:`ForecastingAnomalyModel`) or a filtering method
(:class:`FilteringAnomalyModel`).
The main functions are :func:`fit()` (only for the trainable model and/or scorer), :func:`score()` and
:func:`eval_accuracy()`. :func:`fit()` trains the model and/or the scorer(s) over the history of one or multiple series.
:func:`score()` applies the model on the time series input and and calls the scorer(s) to return an anomaly score of
the input series. The main role of the scorer objects is to compare the output of the model (e.g., the forecast(s))
and emit a score describing how anomalous they are. If several scorer objects are provided, several scores
are computed.

The function :func:`eval_accuracy()` is the same as :func:`score()`, but outputs the score of an agnostic
threshold metric (AUC-ROC or AUC-PR), between the predicted anomaly score time series, and some known binary
ground-truth time series indicating the presence of actual anomalies.

TODO:
    - check if warning is the right way to do: with import warnings
    - put start default value to its minimal value
    - check problem with component wise is False/True fit with a uni/multi and score on a uni/multi
"""

from abc import ABC, abstractmethod
from typing import Sequence, Union

from darts.ad.scorers.scorers import AnomalyScorer
from darts.ad.utils import _to_list, show_anomalies_from_scores
from darts.logging import raise_if_not
from darts.timeseries import TimeSeries


class AnomalyModel(ABC):
    "Base class for all anomaly models"

    def __init__(self, model, scorer):

        self.scorers = _to_list(scorer)

        raise_if_not(
            all([isinstance(s, AnomalyScorer) for s in self.scorers]),
            "all scorers must be of instance darts.ad.scorers.AnomalyScorer",
        )

        self.scorers_are_trainable = any(s.trainable for s in self.scorers)
        self.scorers_are_returns_UTS = any(s.returns_UTS for s in self.scorers)

        self.model = model

    def check_returns_UTS(self, actual_anomalies):
        """Checks if 'actual_anomalies' contains only univariate series when the
        parameter 'scorers_are_returns_UTS' is set to True.

        'scorers_are_returns_UTS', an anomaly model parameter, is:
            True -> at least one of the scorer has its parameter set to True
            False -> all scorers have their parameter set to False

        'returns_UTS', a scorer parameter, is:
            True -> when the function of the scorer ``score(series)`` (or, if applicable,
                ``score_from_prediction(actual_series, pred_series)``) returns a univariate
                anomaly score regardless of the input 'series' (or, if applicable, 'actual_series'
                and 'pred_series').
            False -> when the scorer will return a series that has the
                same width as the input (can be univariate or multivariate).
        """

        if self.scorers_are_returns_UTS:
            raise_if_not(
                all([s.width == 1 for s in actual_anomalies]),
                "Anomaly model contains scorer {} that will return a univariate anomaly score series (width=1). \
                Found a multivariate 'actual_anomalies'. The evaluation of the accuracy cannot be computed.".format(
                    [s.__str__() for s in self.scorers if s.returns_UTS]
                ),
            )

    @abstractmethod
    def fit(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        pass

    @abstractmethod
    def score(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        pass

    @abstractmethod
    def eval_accuracy(
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
            title = f"Anomaly results by model {self.model.__class__.__name__}"

        if names_of_scorers is None:
            names_of_scorers = []
            for scorer in self.scorers:
                names_of_scorers.append(scorer.__str__())

        list_window = []
        for scorer in self.scorers:
            list_window.append(scorer.window)

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
