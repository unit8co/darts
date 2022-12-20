"""
Anomaly Detection
-----------------

A suite of tools for performing anomaly detection and classification
on time series.

* Anomaly Scorers: produce anomaly scores time series, either for single series (``score()``),
  or for series accompanied by some predictions (``score_from_prediction()``).
  The scorers can be trainable (e.g., ``KMeansScorer``, which learns clusters) or not 
  (e.g., ``NormScorer``) and they apply windowing to the time series.
* Anomaly Models: offering a single class to produce anomaly scores from any of Darts
  forecasting models (``ForecastingAnomalyModel``) or filtering models (``FilteringAnomalyModel``),
  by comparing models' predictions with actual observations.
  These classes take as parameters one Darts model, and one or multiple scorers, and can be readily
  used to produce anomaly scores with the ``score()`` method.
* Anomaly Detectors: transform anomaly scores time series into binary anomaly time series.
* Anomaly Aggregators: combine multiple binary anomaly time series (in the form of multivariate time series)
  into a single binary anomaly time series.
"""

# anomaly aggregators
from .aggregators import AndAggregator, EnsembleSklearnAggregator, OrAggregator

# anomaly models
from .anomaly_model.filtering_am import FilteringAnomalyModel
from .anomaly_model.forecasting_am import ForecastingAnomalyModel

# anomaly detectors
from .detectors import QuantileDetector, ThresholdDetector

# anomaly scorers
from .scorers import (
    CauchyNLLScorer,
    DifferenceScorer,
    ExponentialNLLScorer,
    GammaNLLScorer,
    GaussianNLLScorer,
    KMeansScorer,
    LaplaceNLLScorer,
    NormScorer,
    PoissonNLLScorer,
    PyODScorer,
    WassersteinScorer,
)
