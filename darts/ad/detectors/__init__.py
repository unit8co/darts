"""
Anomaly Detectors
-----------------

Detectors provide binary anomaly classification on time series.
They can typically be used to transform anomaly scores time series into binary anomaly time series.

Some detectors are trainable. For instance, ``QuantileDetector`` emits a binary anomaly for
every time step where the observed value(s) are beyond the quantile(s) observed
on the training series.

The main functions are ``fit()`` (for the trainable detectors), ``detect()`` and ``eval_accuracy()``.

``fit()`` trains the detector over the history of one or multiple time series. It can
for instance be called on series containing anomaly scores (or even raw values) during normal times.
The function ``detect()`` takes an anomaly score time series as input, and applies the detector
to obtain binary predictions. The function ``eval_accuracy()`` returns the accuracy metric
("accuracy", "precision", "recall" or "f1") between a binary prediction time series and some known
binary ground truth time series indicating the presence of anomalies.
"""

from .quantile_detector import QuantileDetector
from .threshold_detector import ThresholdDetector
