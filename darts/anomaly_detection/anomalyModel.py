from cProfile import label
from darts import TimeSeries
import numpy as np
from typing import Union, Any, Dict, Sequence, Tuple
from abc import ABC, abstractmethod
from darts.datasets import AirPassengersDataset
import pandas as pd
import darts
import matplotlib.pyplot as plt

import darts.anomaly_detection.score as score

from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.filtering.filtering_model import FilteringModel

from darts.models import NaiveSeasonal, MovingAverage, NBEATSModel, KalmanFilter, GaussianProcessFilter, ARIMA, RegressionModel

class AnomalyModel(ABC):
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

