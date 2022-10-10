"""
AnomalyModel
-------
Anomaly models expect a model and a scorer, and will take as input a time series and returns its anomaly score
as a time series. 

The model can be a forecasting method (ForecastingAnomalyModel) or a filtering method (FilteringAnomalyModel). 
The main functions are `fit()` (only for the trainable model/scorer), `score()` and `score_metric()`. `fit()` 
will train the model and/or the scorer, over the history of one time series. `score()` will apply the model on
the time series input, and the scorer on the prediction of the model and the time series input. The `score()` 
will output the anomaly score of the time series input. The function `score_metric()` is the same as `score()`, 
but outputs the score of an agnostic threshold metric (AUC-ROC or AUC-PR), between the predicted anomaly score 
time series and a binary ground truth time series indicating the presence of anomalies.  

"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Union

from darts.anomaly_detection.score import Scorer
from darts.logging import raise_if_not
from darts.models.filtering.filtering_model import FilteringModel
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries


class AnomalyModel(ABC):
    "Base class for all Anomaly Model"

    @abstractmethod
    def score(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        pass

    @abstractmethod
    def fit(
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
        """Forecasting Anomaly Model

        Parameters
        ----------
        model : ForecastingModel
            A forecasting model from Darts that will be used to predict the actual time series
        scorer : Scorer
            A scorer that will be used to convert the actual and predicted time series to
            an anomaly score time series. If a list of n scorer is given, the anomaly model will test each
            one of the scorers and output n anomaly score.
        """
        super().__init__()

        raise_if_not(
            isinstance(model, ForecastingModel),
            "Model must be a darts.models.forecasting not a {}".format(type(model)),
        )
        self.model = model

        if isinstance(scorer, Sequence):
            self.scorers = scorer
        else:
            self.scorers = [scorer]

        for scorer in self.scorers:
            raise_if_not(
                isinstance(scorer, Scorer),
                "Scorer must be a darts.anomaly_detection.score not a {}".format(
                    type(scorer)
                ),
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
            "Model {} has not been trained. Please call .fit()".format(self.model),
        )

        pred = self.model.historical_forecasts(
            series, retrain=False, **hist_forecasts_params
        )

        anomaly_scores = []

        for i, scorer in enumerate(self.scorers):
            anomaly_scores.append(scorer.compute(pred, series.slice_intersect(pred)))

        if i == 0:
            return anomaly_scores[0]
        else:
            return anomaly_scores

    def score_metric(
        self,
        series: TimeSeries,
        true_anomalies: TimeSeries,
        hist_forecasts_params: Optional[Dict[str, Any]] = None,
        return_metric="AUC_ROC",
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

        Returns
        -------
        float
            Score for the time series
        """

        if hist_forecasts_params is None:
            hist_forecasts_params = {}

        raise_if_not(
            self.model._fit_called,
            "Model {} has not been trained. Please call .fit()".format(self.model),
        )

        pred = self.model.historical_forecasts(
            series, retrain=False, **hist_forecasts_params
        )

        anomaly_scores = []

        for i, scorer in enumerate(self.scorers):
            anomaly_scores.append(
                scorer.compute_score(
                    true_anomalies, pred, series.slice_intersect(pred), return_metric
                )
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
            "Model must be a darts.models.filtering not a {}".format(type(filter)),
        )
        self.filter = filter

        if isinstance(scorer, Sequence):
            self.scorers = scorer
        else:
            self.scorers = [scorer]

        for scorer in self.scorers:
            raise_if_not(
                isinstance(scorer, Scorer),
                "Scorer must be a darts.anomaly_detection.score not a {}".format(
                    type(scorer)
                ),
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
            anomaly_scores.append(scorer.compute(pred, series.slice_intersect(pred)))

        if i == 0:
            return anomaly_scores[0]
        else:
            return anomaly_scores

    def score_metric(
        self,
        series: TimeSeries,
        true_anomalies: TimeSeries,
        filter_params: Optional[Dict[str, Any]] = None,
        return_metric="AUC_ROC",
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
                scorer.compute_score(
                    true_anomalies, pred, series.slice_intersect(pred), return_metric
                )
            )

        if i == 0:
            return anomaly_scores[0]
        else:
            return anomaly_scores
