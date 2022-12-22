"""
Anomaly Detection
-----------------

A suite of tools for performing anomaly detection and classification
on time series.

* `Anomaly Scorers <https://unit8co.github.io/darts/generated_api/darts.ad.scorers.html>`_
  are at the core of the anomaly detection module. They
  produce anomaly scores time series, either for single series (``score()``),
  or for series accompanied by some predictions (``score_from_prediction()``).
  Scorers can be trainable (e.g., ``KMeansScorer``) or not (e.g., ``NormScorer``).

* `Anomaly Models <https://unit8co.github.io/darts/generated_api/darts.ad.anomaly_model.html>`_ 
  offer a convenient way to produce anomaly scores from any of Darts
  forecasting models (``ForecastingAnomalyModel``) or filtering models (``FilteringAnomalyModel``),
  by comparing models' predictions with actual observations.
  These classes take as parameters one Darts model, and one or multiple scorers, and can be readily
  used to produce anomaly scores with the ``score()`` method.

* `Anomaly Detectors <https://unit8co.github.io/darts/generated_api/darts.ad.detectors.html>`_: 
  transform raw time series (such as anaomly scores) into binary anomaly time series.

* `Anomaly Aggregators <https://unit8co.github.io/darts/generated_api/darts.ad.aggregators.html>`_:
  combine multiple binary anomaly time series (in the form of multivariate time series)
  into a single binary anomaly time series applying boolean logic.
"""

# anomaly aggregators
from .aggregators import AndAggregator, EnsembleSklearnAggregator, OrAggregator

# anomaly models
from .anomaly_model import FilteringAnomalyModel, ForecastingAnomalyModel

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
