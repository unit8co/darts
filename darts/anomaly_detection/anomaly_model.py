from cProfile import label
from darts import TimeSeries
import numpy as np
from typing import Union, Any, Dict, Sequence, Tuple
from abc import ABC, abstractmethod
from darts.datasets import AirPassengersDataset
import pandas as pd
import darts
import matplotlib.pyplot as plt
from darts.logging import raise_if, raise_if_not

from darts.anomaly_detection.score import _TrainableScorer
from darts.anomaly_detection import score

from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.filtering.filtering_model import FilteringModel

from darts.models import NaiveSeasonal, MovingAverage, NBEATSModel, KalmanFilter, GaussianProcessFilter, ARIMA, RegressionModel

class _AnomalyModel(ABC):

    @abstractmethod
    def score(
              self, series: Union[TimeSeries, Sequence[TimeSeries]]
             ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        pass
        # returns a scalar TimeSeries indicating presence of anomalies in series
        
        
class ForecastingAnomalyModel(_AnomalyModel):

   def __init__(self, model: ForecastingModel, scorer: score):
        super().__init__()
        # init with this model and scorer
        if isinstance(model, ForecastingModel):
            self.model = model 
        else :
            raise ValueError(
            "Model must be a darts.models.forecasting not a {}".format(type(model))
            )

        self.scorer = scorer
        
   def fit(self, series, model_fit_kwargs={}, scorer_fit_kwargs={}):

        # fit filtering model 
        if hasattr(self.model, "fit"):
            if not self.model._fit_called:
                self.model.fit(series)  # <- add model_fit_kwargs

        # fit scorer model 
        if hasattr(self.scorer, "fit"):
            if not self.scorer._fit_called:  # should we check is scorer has already been trained? Or we overwrite everytime?
                # start should be the smallest value possible
                pred = self.model.historical_forecasts(
                    series, start=0.4, forecast_horizon=1, verbose=True
                    )

                residuals = (pred - series.slice_intersect(pred)).map(lambda x: np.abs(x))

                self.scorer.fit(residuals) # <- add scorer_fit_kwargs
       
   def score(self, series):
        if self.model._fit_called :
            pred = self.model.historical_forecasts(
                    series, start=0.4, forecast_horizon=1, verbose=True
                    )

            if issubclass(type(self.scorer), _TrainableScorer):
                residuals = (pred - series.slice_intersect(pred)).map(lambda x: np.abs(x))
                return self.scorer.compute(residuals)
            else:
                return self.scorer.compute(pred, series.slice_intersect(pred))

        else : 
            raise ValueError(
            "Model {} has not been trained. Please call .fit()".format(self.model)
            )
       

class FilteringAnomalyModel(_AnomalyModel):

   def __init__(self, filter: FilteringModel, scorer: score):
        super().__init__()
        # init with this model and scorer
        if isinstance(filter, FilteringModel):
            self.filter = filter 
        else :
            raise ValueError(
            "Filter must be a darts.models.filtering not a {}".format(type(filter))
            )

        self.scorer = scorer
        
   def fit(self, series, model_fit_kwargs={}, scorer_fit_kwargs={}):
        
        # fit filtering model 
        if hasattr(self.filter, "fit"):
            # TODO: check if filter is already fitted (for now fit it regardless -> only Kallman)
            self.filter.fit(series)  # <- add model_fit_kwargs

        # fit scorer model 
        if hasattr(self.scorer, "fit"):
            if not self.scorer._fit_called:
                pred = self.filter.filter(series)
                self.scorer.fit(pred) # <- add scorer_fit_kwargs

                
   def score(self, series):

       pred = self.filter.filter(series)
       
       if issubclass(type(self.scorer), _TrainableScorer):
            return self.scorer.compute(pred)
       else:
            return self.scorer.compute(pred, series.slice_intersect(pred))
    
       



"""

series = AirPassengersDataset().load()
series_train, series_test = series.split_before(pd.Timestamp("19580101"))
np_anomalies = np.random.choice(a = [0,1], size = len(series_test), p = [0.5, 0.5])
anomalies = TimeSeries.from_times_and_values(series_test._time_index, np_anomalies)

AS_model1 = ForecastingAnomalyModel(
    model=NaiveSeasonal(K=12),
    scorer=score.KmeansAnomaly(k=4)
)

AS_model1.fit(series_train)
anomaly_score = AS_model1.score(series_test)

print(anomaly_score)





class AnomalyModel2(ABC):
    "Base class for models+scorer" 

    def __init__(self, 
        model_to_train: Union[ForecastingModel,FilteringModel],
        scorer_fn: score,
        return_score_train: str = "no"
        ) -> Union[TimeSeries, Sequence[TimeSeries]]:

        super().__init__()
        self.return_score_train = return_score_train
        self.model = model_to_train
        self.scorer_fn = scorer_fn

        if isinstance(model_to_train, FilteringModel):
            self.type_model = "filter"
        elif isinstance(model_to_train, ForecastingModel):
            self.type_model = "forecast"
        else :
            raise ValueError(
            "Model must be a darts.models.forecasting or a darts.models.filtering not a {}".format(type(model_to_train))
            )

        if hasattr(model_to_train, "fit"):
            self.to_fit = True 
        else :
            self.to_fit = False


    def run_compute(self,  
        series: Union[TimeSeries, Sequence[TimeSeries]],
        split_train_test: Union[pd.Timestamp, float, int]):

        TS_train, TS_test = series.split_before(split_train_test)

        if self.type_model=="filter":
            # filtering case 
            if self.to_fit:
                self.model.fit(TS_train)
            pred = self.model.filter(TS_test)
        else : 
            # forecasting case 
            if self.to_fit:
                pred = self.model.historical_forecasts(
                    series, start=split_train_test, forecast_horizon=1, verbose=True
                    )
            else:
                pred = self.model.predict(TS_test)

        anomaly_score = self.scorer_fn.compute(TS_test, pred)

        return anomaly_score

series = AirPassengersDataset().load()
series_train, series_test = series.split_before(pd.Timestamp("19580101"))
np_anomalies = np.random.choice(a = [0,1], size = len(series_test), p = [0.7, 0.3])
anomalies = TimeSeries.from_times_and_values(series_test._time_index, np_anomalies)
print(anomalies)

model_NS = ForecastingAnomalyModel(
    model_to_train= NaiveSeasonal(K=12),
    scorer_fn= S.L1()
)
anomaly_score_NS = model_NS.run_compute(series, pd.Timestamp("19580101"))


# Assumptions V0
# # models filtering/forescasting are fitted (to be modified)
# # no kwargs 
# # score needs to be fit 

"""