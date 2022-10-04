from abc import ABC, abstractmethod
from darts import TimeSeries

from sklearn.metrics import roc_auc_score

import numpy as np
from typing import Union, Any, Dict, Sequence, Tuple
from darts.datasets import AirPassengersDataset


class _Scorer(ABC):
    "Base class for all scores ([TS, TS] -> TS_anomaly_score)" 

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
                y_true = true_anomaly.to_numpy().flatten(),
                y_score = series.pd_series().to_numpy().flatten()
            )


class _NonTrainableScorer(_Scorer):
    "Base class of Detectors that do not need training."

    @abstractmethod
    def _compute_core(self, input_1: Any, input_2: Any) -> Any:
        pass

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
                scoring: str = "AUC_ROC",
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

        anomaly_score = self._compute_core(series_1, series_2)
            
        return  self.score(anomaly_score, true_anomaly, scoring)



class _TrainableScorer(_Scorer):
    "Base class of Detectors that do need training."


class L2(_NonTrainableScorer):
    """ L2 distance metric
    """

    def __init__(self) -> None:
        super().__init__()


    def _compute_core(self, series_1: TimeSeries, series_2: TimeSeries) -> TimeSeries:
        L2_array = np.linalg.norm(series_1.pd_series() - series_2.pd_series(), axis=0)
        print(L2_array)
        L2_series = TimeSeries.from_values(L2_array)

        return L2_series


class L1(_NonTrainableScorer):
    """ L1 distance metric
    """

    def __init__(self) -> None:
        super().__init__()

    def _compute_core(self, series_1: TimeSeries, series_2: TimeSeries) -> TimeSeries:
        return (series_1 - series_2).map(lambda x: np.abs(x))

class difference(_NonTrainableScorer):
    """ difference distance metric
    """

    def __init__(self) -> None:
        super().__init__()

    def _compute_core(self, series_1: TimeSeries, series_2: TimeSeries) -> TimeSeries:
        return series_1 - series_2


class Likelihood(_TrainableScorer):
    """ Likelihood anomaly score
    """


