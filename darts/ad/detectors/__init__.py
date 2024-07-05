"""
Anomaly Detectors
-----------------

Detectors provide binary anomaly classification on time series. They can typically be used to transform anomaly scores
time series into binary anomaly time series.

Some detectors are trainable. For instance, :class:`~darts.ad.detectors.quantile_detector.QuantileDetector` emits a
binary anomaly for every time step where the observed value(s) are beyond the quantile(s) observed on the training
series.

The main functions are `fit()` (for the trainable detectors), `detect()` and `eval_metric()`.

`fit()` trains the detector over the history of one or multiple time series. It can for instance be called on series
containing anomaly scores (or even raw values) during normal times. The function `detect()` takes an anomaly score
time series as input, and applies the detector to obtain binary predictions. The function `eval_metric()` returns
the accuracy metric ("accuracy", "precision", "recall" or "f1") between a binary prediction time series and some known
binary ground truth time series indicating the presence of anomalies.
"""

from darts.ad.detectors.iqr_detector import IQRDetector
from darts.ad.detectors.quantile_detector import QuantileDetector
from darts.ad.detectors.threshold_detector import ThresholdDetector

__all__ = [
    "IQRDetector",
    "QuantileDetector",
    "ThresholdDetector",
]
