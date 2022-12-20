"""
Anomaly Detectors
-----------------

Detectors are used to transform an anomaly score time series into a binary anomaly time series.

Some detectors are trainable. For instance, ``QuantileDetector`` emits a binary anomaly for
every time step where the observed value(s) are beyond the quantile(s) observed
on the training score series.

The main functions are ``fit()`` (for the trainable detectors), ``detect()`` and ``eval_accuracy()``.

``fit()`` trains the detector over the history of one or multiple time series containing anomaly scores.
The function ``detect()`` takes an anomaly score time series as input, and applies the detector
to obtain a binary predictions. The function ``eval_accuracy()`` returns the accuracy metric
("accuracy", "precision", "recall" or "f1") between a binary prediction time series and some known
binary ground truth time series indicating the presence of anomalies.
"""

from .quantile_detector import QuantileDetector
from .threshold_detector import ThresholdDetector
