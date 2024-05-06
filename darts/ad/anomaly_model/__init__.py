"""
Anomaly Models
--------------

Anomaly models make it possible to use any of Darts' forecasting or filtering models to detect anomalies in time series.

The basic idea is to compare the predictions produced by a fitted model (the forecasts or the filtered series) with the
actual observations, and to emit an anomaly score describing how "different" the observations are from the predictions.

An anomaly model takes as parameters a model and one or multiple scorer objects. The key method is `score()`, which
takes as input one (or multiple) time series and produces one or multiple anomaly scores time series, for each provided
series.

:class:`~darts.ad.anomaly_model.forecasting_am.ForecastingAnomalyModel` works with Darts forecasting models, and
:class:`~darts.ad.anomaly_model.filtering_am.FilteringAnomalyModel` works with Darts filtering models. The anomaly
models can also be fitted by calling :func:`fit()`, which trains the scorer(s) (in case some are trainable), and
potentially the model as well.

The function :func:`eval_metric()` is the same as :func:`score()`, but outputs the score of an agnostic threshold
metric ("AUC-ROC" or "AUC-PR"), between the predicted anomaly score time series, and some known binary ground-truth
time series indicating the presence of actual anomalies. Finally, the function :func:`show_anomalies()` can also be
used to visualize the predictions (in-sample predictions and anomaly scores) of the anomaly model.
"""

from darts.ad.anomaly_model.filtering_am import FilteringAnomalyModel
from darts.ad.anomaly_model.forecasting_am import ForecastingAnomalyModel

__all__ = [
    "FilteringAnomalyModel",
    "ForecastingAnomalyModel",
]
