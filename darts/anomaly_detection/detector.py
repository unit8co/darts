from abc import ABC, abstractmethod
from darts import TimeSeries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from typing import Union, Any, Dict, Sequence, Tuple
from darts.datasets import AirPassengersDataset

class _Detector(ABC):
    "Base class for all detectors (TS_anomaly_score -> TS_binary_prediction)" 

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def score(
            self, 
            series: Union[TimeSeries, Sequence[TimeSeries]],
            anomaly_true: Union[TimeSeries, Sequence[TimeSeries]],
            scoring: str = "recall",
             ) -> Union[float, Sequence[float]]:

        """Detect anomalies and score the results against true anomalies.

        Parameters
        ----------
        ts: Darts timeseries (uni/multivariate)
            Time series to detect anomalies from.
        anomaly_true: Darts timeseries
            True anomalies.
        scoring: str, optional
            Scoring function to use. Must be one of "recall", "precision",
            "f1", and "iou".
            Default: "recall"

        Returns
        -------
        Darts timeseries (uni/multivariate)
            Score(s) for each timeseries
        """

        if scoring == "recall":
            scoring_fn = recall_score  
        elif scoring == "precision":
            scoring_fn = precision_score
        elif scoring == "f1":
            scoring_fn = f1_score
        elif scoring == "accuracy":
            scoring_fn = accuracy_score
        else:
            raise ValueError(
                "Argument `scoring` must be one of 'recall', 'precision', "
                "'f1' and 'accuracy'."
            )

        return scoring_fn(
                y_true = anomaly_true.all_values().flatten(),
                y_pred = self.detect(series).all_values().flatten()
                )

 
class _NonTrainableDetector(_Detector):
    "Base class of Detectors that do not need training."

    @abstractmethod
    def _detect_core(self, input: Any) -> Any:
        pass

    def detect(
        self, 
        series: Union[TimeSeries, Sequence[TimeSeries]], 
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Detect anomalies from given time series.
        Parameters
        ----------
        series: Darts.Series
            Time series to detect anomalies from. 
        Returns
        -------
        Darts.Series
            Binary prediciton
        """
        # check input (type, size, values)
            
        return self._detect_core(series)



class ThresholdAD(_NonTrainableDetector):
    """Detector that detects anomaly based on user-given threshold.
    This detector compares time series values with user-given thresholds, and
    identifies time points as anomalous when values are beyond the thresholds.
    Parameters
    ----------
    low: float, optional
        Threshold below which a value is regarded anomaly. Default: None, i.e.
        no threshold on lower side.
    high: float, optional
        Threshold above which a value is regarded anomaly. Default: None, i.e.
        no threshold on upper side.
    """

    def __init__(
        self, low: Union[float, None] = None, high: Union[float, None] = None
    ) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def _detect_core(self, series: TimeSeries) -> TimeSeries:
        detected = (
            series.pd_series() > (self.high if (self.high is not None) else float("inf"))
        ) | (series.pd_series() < (self.low if (self.low is not None) else -float("inf"))).astype(int)

        return TimeSeries.from_series(detected)




class _TrainableDetector(_Detector):
    "Base class of Detectors that need training."

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fit_called = False  

    def fit(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> None:
        """Train the detector with given time series.
        Parameters
        ----------
        series: Darts.Series
            Time series to be used to train the detector.
        """
        self._fit_core(series)

    def detect(
        self, 
        series: Union[TimeSeries, Sequence[TimeSeries]], 
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:

        """Detect anomalies from given time series.
        Parameters
        ----------
        series: Darts.Series
            Time series to detect anomalies from. 
        Returns
        -------
        Darts.Series
            Binary prediciton
        """

        # check input (type, size, values)

        if not self._fit_called:
            raise ValueError(
                "Model needs to be trained first. Call .fit() or .fit_detect() "
            )
            
        return self._detect_core(series)

    def fit_detect(
        self, series: Union[TimeSeries, Sequence[TimeSeries]]
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Train the detector and detect anomalies from the time series used
        for training.
        Parameters
        ----------
        series: Darts.Series
            Time series to be used for training and be detected for anomalies.
        Returns
        -------
        Darts.Series
            Binary prediciton
        """
        self.fit(series)
        return self.detect(series)



class QuantileAD(_TrainableDetector):
    """Detector that detects anomaly based on quantiles of historical data.
    This detector compares time series values with user-specified quantiles
    of historical data, and identifies time points as anomalous when values
    are beyond the thresholds.
    Parameters
    ----------
    low: float, optional
        Quantile of historical data lower which a value is regarded as anomaly.
        Must be between 0 and 1. 
    high: float, optional
        Quantile of historical data above which a value is regarded as anomaly.
        Must be between 0 and 1. 
    Attributes
    ----------
    abs_low_: float
        The fitted lower bound of normal range.
    abs_high_: float
        The fitted upper bound of normal range.
    """

    def __init__(
        self, low: Union[float, None], high: Union[float, None]
    ) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def _fit_core(self, series: Union[TimeSeries, Sequence[TimeSeries]]) -> None:

        self.abs_high_ = series.pd_series().quantile(self.high)
        self.abs_low_ = series.pd_series().quantile(self.low)
        self._fit_called = True

    def _detect_core(self, series: TimeSeries) -> TimeSeries:
        series = series.pd_series()
        detected = (series > self.abs_high_) | (series < self.abs_low_)
        return TimeSeries.from_series(detected)



















"""

TS_noise = series
TS_train = series
TS_test = series

L1 = 1
KalmanFilter =1 
NBEATSModel = 1
Kmeans_reconstruction = 1
Threshold= 1
true_anomalies = 1
OrAggregator = 1

# filtering/forecasting/reconstruction done by the user 

# example filtering with kf
kf = KalmanFilter("< parameters >")
kf.fit(TS_noise)
filtered_TS = kf.filter(TS_noise)

# example forecasting with NBEATS
model_nbeats = NBEATSModel("< parameters >")
pred_TS_train = model_nbeats.fit_predict(TS_train)
pred_TS_test = model_nbeats.predict(TS_test)

# example reconstruction with Kmeans
model_Kmeans_recon = Kmeans_reconstruction("< parameters >")
recon_TS_train = model_Kmeans_recon.fit_predict(TS_train)
recon_TS_test = model_nbeats.predict(TS_test)


# Scorer: output-> anomaly score 
l1_diff = L1()

diff_filtering = L1().compute(TS_noise, filtered_TS)
diff_forecast = L1().compute(TS_test, pred_TS_test)
diff_recon = L1().compute(TS_test, recon_TS_test)

ROC_AUC_filt = L1().compute_score(TS_noise, filtered_TS)
ROC_AUC_forecast = L1().compute_score(TS_test, pred_TS_test)
ROC_AUC_recon = L1().compute_score(TS_test, recon_TS_test)

# Detector: output-> binary prediction 
detector = Threshold(low=10, high=40)

anomaly_filtering = detector.detect(diff_filtering)
anomaly_forecast = detector.detect(diff_forecast)
anomaly_recon = detector.detect(diff_recon)

f1, recall, prec = detector.detect_score(anomaly_filtering, true_anomalies)
f1, recall, prec = detector.detect_score(anomaly_forecast, true_anomalies)
f1, recall, prec = detector.detect_score(anomaly_recon, true_anomalies)

# Aggregator: output-> binary prediction 

list_prediction = [anomaly_filtering,anomaly_forecast,anomaly_recon]

aggregator = OrAggregator()
prediction = aggregator.predict(list_prediction)
f1, recall, prec = aggregator.predict_score(list_prediction, true_anomalies)

"""