"""
Anomaly Aggregators
-------------------

An anomaly aggregator can take multiple detected anomalies
(in the form of binary TimeSeries, as coming from an anomaly detector)
and combine them into one. It can typically be used to combine
the detections of multiple models into one final detection.

The key method is ``predict()``, which takes as input one (or multiple)
multivariate binary TimeSeries where each component represents the
detection of a single model, and returns one (or multiple) univariate
binary TimeSeries representing the final detection.
"""

from .and_aggregator import AndAggregator
from .ensemble_sklearn_aggregator import EnsembleSklearnAggregator
from .or_aggregator import OrAggregator
