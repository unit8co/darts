"""
AnomalyModel
------------

An anomaly model expect a model and a scorer. It offers a ``score()``function, which takes
as input a time series and returns its anomaly score as a ``TimeSeries``.

The model can be a forecasting method (:class:`ForecastingAnomalyModel`) or a filtering method
(:class:`FilteringAnomalyModel`).
The main functions are ``fit()`` (only for the trainable model/scorer), ``score()`` and
``score_metric()``. ``fit()`` trains the model and/or the scorer over the history of one or multiple series.
``score()`` applies the model on the time series input and and calls the scorer to return an anomaly score of
the input series.

The function ``score_metric()` is the same as ``score()``, but outputs the score of an agnostic
threshold metric (AUC-ROC or AUC-PR), between the predicted anomaly score time series, and some known binary
ground-truth time series indicating the presence of actual anomalies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Union

from darts.anomaly_detection.score import Scorer
from darts.logging import raise_if_not
from darts.models.filtering.filtering_model import FilteringModel
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries


class AnomalyModel(ABC):
    "Base class for all anomaly models"

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
    def score_metric(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        pass


class ForecastingAnomalyModel(AnomalyModel):
    def __init__(
        self, model: ForecastingModel, scorer: Union[Scorer, Sequence[Scorer]]
    ):
        """Forecasting-based Anomaly Model

        Wraps around a Darts forecasting model and an anomaly scorer to compute anomaly scores
        by comparing how actuals deviate from the model's predictions.

        Parameters
        ----------
        model : ForecastingModel
            A Darts forecasting model
        scorer : Scorer
            A scorer that will be used to convert the actual and predicted time series to
            an anomaly score ``TimeSeries``. If a list of `N` scorer is given, the anomaly model will call each
            one of the scorers and output a list of `N` anomaly scores ``TimeSeries``.
        """
        super().__init__()

        raise_if_not(
            isinstance(model, ForecastingModel),
            f"Model must be a darts.models.forecasting not a {type(model)}",
        )
        self.model = model

        if isinstance(scorer, Sequence):
            self.scorers = scorer
        else:
            self.scorers = [scorer]

        for scorer in self.scorers:
            raise_if_not(
                isinstance(scorer, Scorer),
                f"Scorer must be a darts.anomaly_detection.score not a {type(scorer)}",
            )

    def fit(
        self,
        series: TimeSeries,
        model_fit_params: Optional[Dict[str, Any]] = None,
        hist_forecasts_params: Optional[Dict[str, Any]] = None,
    ):
        """Train the model and the scorer(s) on the given time series.

        Parameters
        ----------
        series : Darts TimeSeries
        model_fit_params: dict, optional
            parameters of the Darts `.fit()` forecasting model
        hist_forecasts_params: dict, optional
            parameters of the Darts `.historical_forecasts()` forecasting model

        Returns
        -------
        self
            Fitted Anomaly model (forecasting model and scorer(s))
        """

        if model_fit_params is None:
            model_fit_params = {}

        if hist_forecasts_params is None:
            hist_forecasts_params = {}

        # fit forecasting model
        if hasattr(self.model, "fit"):
            if not self.model._fit_called:
                self.model.fit(series, **model_fit_params)

        # fit scorer model
        for scorer in self.scorers:
            if hasattr(scorer, "fit"):
                pred = self.model.historical_forecasts(
                    series, retrain=False, **hist_forecasts_params
                )

                scorer.fit(pred, series)

    def score(
        self, series: TimeSeries, hist_forecasts_params: Optional[Dict[str, Any]] = None
    ):
        """Predicts the given input time series with the forecasting model, and applies the scorer(s)
        on the prediction and the given input time series. Outputs the anomaly score of the given
        input time series.

        Parameters
        ----------
        series : Darts TimeSeries
        hist_forecasts_params: dict, optional
            parameters of the Darts `.historical_forecasts()` forecasting model

        Returns
        -------
        Darts TimeSeries
            Anomaly score time series
        """

        if hist_forecasts_params is None:
            hist_forecasts_params = {}

        raise_if_not(
            self.model._fit_called,
            f"Model {self.model} has not been trained. Please call .fit()",
        )

        pred = self.model.historical_forecasts(
            series, retrain=False, **hist_forecasts_params
        )

        anomaly_scores = []

        for i, scorer in enumerate(self.scorers):
            anomaly_scores.append(scorer.compute(pred, series))

        if i == 0:
            return anomaly_scores[0]
        else:
            return anomaly_scores

    def score_metric(
        self,
        series: TimeSeries,
        true_anomalies: TimeSeries,
        hist_forecasts_params: Optional[Dict[str, Any]] = None,
        metric="AUC_ROC",
    ):
        """Predicts the given input time series with the forecasting model, and applies the
        scorer(s) on the filtered time series and the given input time series. Returns the
        score(s) of an agnostic threshold metric, based on the anomaly score given by the scorer(s).

        Parameters
        ----------
        series : Darts TimeSeries
        actual_anomalies: Binary Darts TimeSeries
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        hist_forecasts_params: dict, optional
            parameters of the Darts `.historical_forecasts()` forecasting model
        metric: str
            The selected metric to use. Can be 'AUC_ROC' (default value) or 'AUC_PR'

        Returns
        -------
        float
            Score for the time series
        """

        if hist_forecasts_params is None:
            hist_forecasts_params = {}

        raise_if_not(
            self.model._fit_called,
            f"Model {self.model} has not been trained. Please call .fit()",
        )

        pred = self.model.historical_forecasts(
            series, retrain=False, **hist_forecasts_params
        )

        anomaly_scores = []

        for i, scorer in enumerate(self.scorers):
            anomaly_scores.append(
                scorer.compute_score(true_anomalies, pred, series, metric)
            )

        if i == 0:
            return anomaly_scores[0]
        else:
            return anomaly_scores


class FilteringAnomalyModel(AnomalyModel):
    def __init__(self, filter: FilteringModel, scorer: Union[Scorer, Sequence[Scorer]]):
        """Filtering Anomaly Model

        Parameters
        ----------
        model : Filtering
            A filtering model from Darts that will be used to filter the actual time series
        scorer : Scorer
            A scorer that will be used to convert the actual and filtered time series to
            an anomaly score time series
        """

        super().__init__()

        raise_if_not(
            isinstance(filter, FilteringModel),
            f"Model must be a darts.models.filtering not a {type(filter)}",
        )
        self.filter = filter

        if isinstance(scorer, Sequence):
            self.scorers = scorer
        else:
            self.scorers = [scorer]

        for scorer in self.scorers:
            raise_if_not(
                isinstance(scorer, Scorer),
                f"Scorer must be a darts.anomaly_detection.score not a {type(scorer)}",
            )

    def fit(
        self, series: TimeSeries, filter_fit_params: Optional[Dict[str, Any]] = None
    ):
        """Train the filter and the scorer(s) on the given time series.

        Parameters
        ----------
        series : Darts TimeSeries
        filter_fit_params: dict, optional
            parameters of the Darts `.fit()` filtering model

        Returns
        -------
        self
            Fitted Anomaly model (filtering model and scorer(s))
        """

        if filter_fit_params is None:
            filter_fit_params = {}

        # fit filtering model
        if hasattr(self.filter, "fit"):
            # TODO: check if filter is already fitted (for now fit it regardless -> only Kallman)
            self.filter.fit(series, **filter_fit_params)

        # fit scorer model
        for scorer in self.scorers:
            if hasattr(scorer, "fit"):
                pred = self.filter.filter(series)
                scorer.fit(pred, series)

    def score(self, series: TimeSeries, filter_params: Optional[Dict[str, Any]] = None):
        """Filters the given input time series with the filtering model, and applies the scorer(s)
        on the filtered time series and the given input time series. Outputs the anomaly score of
        the given input time series.

        Parameters
        ----------
        series : Darts TimeSeries
        filter_params: dict, optional
            parameters of the Darts `.filter()` filtering model

        Returns
        -------
        Darts TimeSeries
            Anomaly score time series
        """

        if filter_params is None:
            filter_params = {}

        pred = self.filter.filter(series, **filter_params)

        anomaly_scores = []

        for i, scorer in enumerate(self.scorers):
            anomaly_scores.append(scorer.compute(pred, series))

        if i == 0:
            return anomaly_scores[0]
        else:
            return anomaly_scores

    def score_metric(
        self,
        series: TimeSeries,
        true_anomalies: TimeSeries,
        filter_params: Optional[Dict[str, Any]] = None,
        metric="AUC_ROC",
    ):
        """Filters the given input time series with the filtering model, and applies the scorer(s)
        on the filtered time series and the given input time series. Returns the score(s)
        of an agnostic threshold metric, based on the anomaly score given by the scorer(s).

        Parameters
        ----------
        series : Darts TimeSeries
        actual_anomalies: Binary Darts TimeSeries
            The ground truth of the anomalies (1 if it is an anomaly and 0 if not)
        filter_params: dict, optional
            parameters of the Darts `.filter()` filtering model
        metric: str
            The selected metric to use. Can be 'AUC_ROC' (default value) or 'AUC_PR'

        Returns
        -------
        float
            Score for the time series
        """
        if filter_params is None:
            filter_params = {}

        pred = self.filter.filter(series, **filter_params)

        anomaly_scores = []

        for i, scorer in enumerate(self.scorers):
            anomaly_scores.append(
                scorer.compute_score(true_anomalies, pred, series, metric)
            )

        if i == 0:
            return anomaly_scores[0]
        else:
            return anomaly_scores
