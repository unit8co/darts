from abc import ABC, abstractmethod
from darts import TimeSeries

from sklearn.metrics import roc_auc_score

import numpy as np
from typing import Union, Any, Dict, Sequence, Tuple
from darts.datasets import AirPassengersDataset
from darts.logging import raise_if, raise_if_not

from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

class _Scorer(ABC):
    "Base class for all scores ([TS, TS] -> TS_anomaly_score)" 

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def _compute_core(self, input_1: Any, input_2: Any) -> Any:
        pass

    def score(self, 
            series: Union[TimeSeries, Sequence[TimeSeries]],
            true_anomaly: Union[TimeSeries, Sequence[TimeSeries]],
            scoring: str = "AUC_ROC",
            ) -> Union[float, Sequence[float]]:
        
        if scoring == "AUC_ROC":
            scoring_fn = roc_auc_score  
        else:
            raise ValueError(
                "Argument `scoring` must be one of 'AUC_ROC'"
            )

        return scoring_fn(
            y_true = true_anomaly.all_values().flatten(),
            y_score = series.all_values().flatten()
            )


class _NonTrainableScorer(_Scorer):
    "Base class of Detectors that do not need training."

    def compute(
        self, 
        series_1: Union[TimeSeries, Sequence[TimeSeries]], 
        series_2: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """
        Parameters
        ----------
        series_1: Darts.Series
        series_2: Darts.Series
     
        Returns
        -------
        Darts.Series
            anomaly score
        """
        # check inputs (type, size, values)
        # check if same timestamp!
            
        return self._compute_core(series_1, series_2)

    def compute_score(
                self, 
                series_1: Union[TimeSeries, Sequence[TimeSeries]], 
                series_2: Union[TimeSeries, Sequence[TimeSeries]],
                true_anomaly: Union[TimeSeries, Sequence[TimeSeries]],
                scoring: str = "AUC_ROC"
            ) -> Union[float, Sequence[float]]:
        """
        Parameters
        ----------
        series_1: Darts.Series
        series_2: Darts.Series
        true_anomalies: Binary Darts.Series
     
        Returns
        -------
        Darts.Series
            anomaly score
        """
        # check inputs (type, size, values) -> of anomaly_score
        # check if same timestamp -> of anomaly_score

        anomaly_score = self.compute(series_1, series_2)
        
        return  self.score(anomaly_score, true_anomaly, scoring)


class _TrainableScorer(_Scorer):
    "Base class of Detectors that do need training."
    # need to output an error if score is called and it was not trained

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fit_called = False  

    def compute(
            self, 
            series: Union[TimeSeries, Sequence[TimeSeries]], 
        ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """
        Parameters
        ----------
        series: Darts.Series
     
        Returns
        -------
        Darts.Series
            anomaly score
        """
        # check inputs (type, size, values)
        # check if same timestamp!
        
        raise_if(
            not self._fit_called,
            "The Scorer has not been fitted yet. Call `fit()` first",
        )
        
        return self._compute_core(series)

    def compute_score(
                self, 
                series: Union[TimeSeries, Sequence[TimeSeries]], 
                true_anomaly: Union[TimeSeries, Sequence[TimeSeries]],
                scoring: str = "AUC_ROC",
            ) -> Union[float, Sequence[float]]:
        """
        Parameters
        ----------
        series: Darts.Series
        true_anomalies: Binary Darts.Series
     
        Returns
        -------
        Darts.Series
            anomaly score
        """
        # check inputs (type, size, values)
        # check if same timestamp!

        anomaly_score = self.compute(series)
        return  self.score(anomaly_score, true_anomaly, scoring)

    def fit(
            self, 
            series: Union[TimeSeries, Sequence[TimeSeries]], 
            ) -> Union[float, Sequence[float]]:
        """
        Parameters
        ----------
        series_1: Darts.Series
     
        Returns
        -------
        Darts.Series
            anomaly score
        """
        # check inputs (type, size, values)
        # check if same timestamp!

        self._fit_core(series)

    def fit_compute(
                self, 
                series_train: Union[TimeSeries, Sequence[TimeSeries]], 
                series_test: Union[TimeSeries, Sequence[TimeSeries]],
            ) -> Union[float, Sequence[float]]:
        """
        Parameters
        ----------
        series_1: Darts.Series
        series_2: Darts.Series
     
        Returns
        -------
        Darts.Series
            anomaly score
        """
        # check inputs (type, size, values)
        # check if same timestamp!

        self._fit_core(series_train)
        return self._compute_core(series_test)

    def fit_compute_score(
                self, 
                series_train: Union[TimeSeries, Sequence[TimeSeries]], 
                series_test: Union[TimeSeries, Sequence[TimeSeries]],
                true_anomaly: Union[TimeSeries, Sequence[TimeSeries]],
                scoring: str = "AUC_ROC"
            ) -> Union[float, Sequence[float]]:
        """
        Parameters
        ----------
        series_1: Darts.Series
        series_2: Darts.Series
        true_anomalies: Binary Darts.Series
     
        Returns
        -------
        Darts.Series
            anomaly score
        """
        # check inputs (type, size, values)
        # check if same timestamp!

        self._fit_core(series_train)
        anomaly_score = self._compute_core(series_test)
        return  self.score(anomaly_score, true_anomaly, scoring)



class KmeansAnomaly(_TrainableScorer):
    """ Likelihood anomaly score
    """
    def __init__(self, k: Union[int, list[int]]) -> None:
        super().__init__()
        self.k = k 
        self.model = KMeans(n_clusters=k)

    def _fit_core(self, series: Union[TimeSeries, Sequence[TimeSeries]]):
        self._fit_called = True  
        self.model.fit(series.all_values().flatten().reshape(-1, 1))

    def _compute_core(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> Union[TimeSeries, Sequence[TimeSeries]]:
        # return distance to the clostest centroid  
        np_anomaly_score= self.model.transform(series.all_values().flatten().reshape(-1, 1)).min(axis=1)
        return TimeSeries.from_times_and_values(series._time_index , np_anomaly_score)

class LocalOutlierFactorAnomaly(_TrainableScorer):
    """ LocalOutlierFactor anomaly score
    """
    def __init__(self, n_neighbors: int) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors 
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)

    def _fit_core(self, series: Union[TimeSeries, Sequence[TimeSeries]]):
        self._fit_called = True  
        self.model.fit(series.all_values().flatten().reshape(-1, 1))

    def _compute_core(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> Union[TimeSeries, Sequence[TimeSeries]]:
        # return distance to the clostest centroid  
        np_anomaly_score= np.abs(self.model.score_samples(series.all_values().flatten().reshape(-1, 1)))
        return TimeSeries.from_times_and_values(series._time_index , np_anomaly_score)


class L2(_NonTrainableScorer):
    """ L2 distance metric
    """

    def __init__(self) -> None:
        super().__init__()


    def _compute_core(self, series_1: Union[TimeSeries, Sequence[TimeSeries]], series_2: Union[TimeSeries, Sequence[TimeSeries]]) -> Union[TimeSeries, Sequence[TimeSeries]]:
        return (series_1 - series_2)**2


class L1(_NonTrainableScorer):
    """ L1 distance metric
    """

    def __init__(self) -> None:
        super().__init__()

    def _compute_core(self, series_1: Union[TimeSeries, Sequence[TimeSeries]], series_2: Union[TimeSeries, Sequence[TimeSeries]]) -> Union[TimeSeries, Sequence[TimeSeries]]:
        return (series_1 - series_2).map(lambda x: np.abs(x))

class difference(_NonTrainableScorer):
    """ difference distance metric
    """

    def __init__(self) -> None:
        super().__init__()

    def _compute_core(self, series_1: Union[TimeSeries, Sequence[TimeSeries]], series_2: Union[TimeSeries, Sequence[TimeSeries]]) -> Union[TimeSeries, Sequence[TimeSeries]]:
        return series_1 - series_2



# To implement!! 
class Likelihood(_TrainableScorer):
    """ Likelihood anomaly score
    """









"""
import pandas as pd

series = AirPassengersDataset().load()
series_train, series_test = series.split_before(pd.Timestamp("19580101"))
np_anomalies = np.random.choice(a = [0,1], size = len(series_test), p = [0.5, 0.5])
anomalies = TimeSeries.from_times_and_values(series_test._time_index, np_anomalies)

series_test_pred = series_test * anomalies

dif = L1().compute(series_test, series_test_pred)
score = L1().compute_score(series_test, series_test_pred, anomalies)
    
model = KmeansAnomaly(4)
modelA = KmeansAnomaly(4)

model.fit(series_train)
pred2 = modelA.fit_compute_score(series_train, series_test, anomalies)

pred = model.compute(series_test)
pred1 = model.compute_score(series_test, anomalies)

print(pred)
print(pred1)
print(pred2)

model1 = LocalOutlierFactorAnomaly(20)
model2 = LocalOutlierFactorAnomaly(30)

model1.fit(series_train)
pred3 = model2.fit_compute_score(series_train, series_test, anomalies)

pred4 = model1.compute(series_test)
pred5 = model1.compute_score(series_test, anomalies)

print(pred3)
print(pred4)
print(pred5)
"""