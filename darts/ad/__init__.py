"""
Anomaly Detection
-----------------

A suite of tools for performing anomaly detection and classification
on time series.

- `Anomaly Scorers <https://unit8co.github.io/darts/generated_api/darts.ad.scorers.html>`__ are at the core of the
  anomaly detection module. They produce anomaly scores time series, either for single series (`score()`),
  or for series accompanied by some predictions (`score_from_prediction()`). Scorers can be trainable
  (e.g., :class:`~darts.ad.scorers.kmeans_scorer.KMeansScorer`) or not
  (e.g., :class:`~darts.ad.scorers.norm_scorer.NormScorer`).

- `Anomaly Models <https://unit8co.github.io/darts/generated_api/darts.ad.anomaly_model.html>`__ offer a convenient way
  to produce anomaly scores from any of Darts forecasting models
  (:class:`~darts.ad.anomaly_model.forecasting_am.ForecastingAnomalyModel`) or filtering models
  (:class:`~darts.ad.anomaly_model.filtering_am.FilteringAnomalyModel`), by comparing models' predictions with actual
  observations. These classes take as parameters one Darts model, and one or multiple scorers, and can be readily used
  to produce anomaly scores with the `score()` method.

- `Anomaly Detectors <https://unit8co.github.io/darts/generated_api/darts.ad.detectors.html>`__: transform raw time
  series (such as anomaly scores) into binary anomaly time series.

- `Anomaly Aggregators <https://unit8co.github.io/darts/generated_api/darts.ad.aggregators.html>`__: combine multiple
  binary anomaly time series (in the form of multivariate time series) into a single binary anomaly time series
  applying boolean logic.
"""

from darts.utils._lazy import setup_lazy_imports

_LAZY_IMPORTS: dict[str, str] = {
    # anomaly aggregators
    "AndAggregator": "darts.ad.aggregators",
    "EnsembleSklearnAggregator": "darts.ad.aggregators",
    "OrAggregator": "darts.ad.aggregators",
    # anomaly models
    "FilteringAnomalyModel": "darts.ad.anomaly_model",
    "ForecastingAnomalyModel": "darts.ad.anomaly_model",
    # anomaly detectors
    "QuantileDetector": "darts.ad.detectors",
    "ThresholdDetector": "darts.ad.detectors",
    # anomaly scorers
    "CauchyNLLScorer": "darts.ad.scorers",
    "DifferenceScorer": "darts.ad.scorers",
    "ExponentialNLLScorer": "darts.ad.scorers",
    "GammaNLLScorer": "darts.ad.scorers",
    "GaussianNLLScorer": "darts.ad.scorers",
    "KMeansScorer": "darts.ad.scorers",
    "LaplaceNLLScorer": "darts.ad.scorers",
    "NormScorer": "darts.ad.scorers",
    "PoissonNLLScorer": "darts.ad.scorers",
    "PyODScorer": "darts.ad.scorers",
    "WassersteinScorer": "darts.ad.scorers",
}

__all__, __getattr__, __dir__ = setup_lazy_imports(_LAZY_IMPORTS, __name__, globals())
