"""
AnomalyModel
-------

describe
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from darts.anomaly_detection.score import Scorer, TrainableScorer
from darts.logging import raise_if, raise_if_not
from darts.models.filtering.filtering_model import FilteringModel
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.timeseries import TimeSeries


class AnomalyModel(ABC):

    @abstractmethod
    def score(
              self, series: Union[TimeSeries, Sequence[TimeSeries]]
             ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        pass
        # returns a scalar TimeSeries indicating presence of anomalies in series

class ForecastingAnomalyModel(AnomalyModel):
   def __init__(
        self, 
        model: ForecastingModel, 
        scorer: Union[Scorer, Sequence[Scorer]]
    ):
        """
        Description

        Parameters
        ----------
        series : TimeSeries
        Returns
        -------
        """
        super().__init__()
        # init with this model and scorer

        raise_if_not(
            isinstance(model, ForecastingModel),
            "Model must be a darts.models.forecasting not a {}".format(type(model))
            )
        self.model = model

        if isinstance(scorer, Sequence):
            self.scorers = scorer
        else :
            self.scorers = [scorer]
        
        for scorer in self.scorers:
            raise_if_not(
                isinstance(scorer, Scorer),
                "Scorer must be a darts.anomaly_detection.score not a {}".format(type(scorer))
                )
        
   def fit(
        self, 
        series: TimeSeries, 
        model_fit_params: Optional[Dict[str, Any]] = None,
        hist_forecasts_params: Optional[Dict[str, Any]] = None
    ):
        """
        Description

        Parameters
        ----------
        series : TimeSeries
        Returns
        -------
        """

        if model_fit_params == None:
            model_fit_params = {}

        if hist_forecasts_params == None:
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
        self, 
        series: TimeSeries, 
        hist_forecasts_params: Optional[Dict[str, Any]] = None
    ):
        """
        Description

        Parameters
        ----------
        series : TimeSeries
        Returns
        -------
        """

        if hist_forecasts_params == None:
            hist_forecasts_params = {}
        
        raise_if_not(
            self.model._fit_called,
            "Model {} has not been trained. Please call .fit()".format(self.model))
    
        pred = self.model.historical_forecasts(
                series, retrain=False, **hist_forecasts_params
                )

        anomaly_scores = []

        for i,scorer in enumerate(self.scorers):
            anomaly_scores.append(scorer.compute(pred, series.slice_intersect(pred)))

        if i == 0:
            return anomaly_scores[0]
        else: return anomaly_scores

   def score_metric(
        self, 
        series: TimeSeries, 
        true_anomalies: TimeSeries, 
        hist_forecasts_params: Optional[Dict[str, Any]] = None, 
        return_metric="AUC_ROC"
    ):
        """
        Description

        Parameters
        ----------
        series : TimeSeries
        Returns
        -------
        """

        if hist_forecasts_params == None:
            hist_forecasts_params = {}
        
        raise_if_not(
            self.model._fit_called,
            "Model {} has not been trained. Please call .fit()".format(self.model))
    
        pred = self.model.historical_forecasts(
                series, retrain=False, **hist_forecasts_params
                )

        anomaly_scores = []

        for i,scorer in enumerate(self.scorers):
            anomaly_scores.append(scorer.compute_score(true_anomalies, pred, series.slice_intersect(pred), return_metric))

        if i == 0:
            return anomaly_scores[0]
        else: return anomaly_scores

class FilteringAnomalyModel(AnomalyModel):

    def __init__(
        self, 
        filter: FilteringModel, 
        scorer: Union[Scorer, Sequence[Scorer]]
    ):
        """
        Description

        Parameters
        ----------
        series : TimeSeries
        Returns
        -------
        """

        super().__init__()

        raise_if_not(
            isinstance(filter, FilteringModel),
            "Model must be a darts.models.filtering not a {}".format(type(filter))
            )
        self.filter = filter

        if isinstance(scorer, Sequence):
            self.scorers = scorer
        else :
            self.scorers = [scorer]
        
        for scorer in self.scorers:
            raise_if_not(
                isinstance(scorer, Scorer),
                "Scorer must be a darts.anomaly_detection.score not a {}".format(type(scorer))
                )

        
    def fit(
        self, 
        series: TimeSeries, 
        filter_fit_params: Optional[Dict[str, Any]] = None
    ):
        """
        Description

        Parameters
        ----------
        series : TimeSeries
        Returns
        -------
        """

        if filter_fit_params == None:
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

                
    def score(
        self, 
        series: TimeSeries, 
        filter_params: Optional[Dict[str, Any]] = None
    ):
        """
        Description

        Parameters
        ----------
        series : TimeSeries
        Returns
        -------
        """

        if filter_params == None:
            filter_params = {}

        pred = self.filter.filter(series, **filter_params)

        anomaly_scores = []

        for i,scorer in enumerate(self.scorers):
            anomaly_scores.append(self.scorer.compute(pred, series.slice_intersect(pred)))

        if i == 0:
            return anomaly_scores[0]
        else: return anomaly_scores

    
    def score_metric(
        self, 
        series: TimeSeries, 
        true_anomalies: TimeSeries, 
        filter_params: Optional[Dict[str, Any]] = None, 
        return_metric= "AUC_ROC"
    ):        
        """
        Description

        Parameters
        ----------
        series : TimeSeries
        Returns
        -------
        """

        if filter_params == None:
            filter_params = {}
        
        pred = self.filter.filter(series, **filter_params)
    
        anomaly_scores = []

        for i,scorer in enumerate(self.scorers):
            anomaly_scores.append(scorer.compute_score(true_anomalies, pred, series.slice_intersect(pred), return_metric))

        if i == 0:
            return anomaly_scores[0]
        else: return anomaly_scores