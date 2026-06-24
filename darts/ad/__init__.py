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

from typing import TYPE_CHECKING

from darts.utils._lazy import setup_lazy_imports

if TYPE_CHECKING:
    from darts.ad.aggregators import AndAggregator as AndAggregator
    from darts.ad.aggregators import (
        EnsembleSklearnAggregator as EnsembleSklearnAggregator,
    )
    from darts.ad.aggregators import OrAggregator as OrAggregator
    from darts.ad.anomaly_model import FilteringAnomalyModel as FilteringAnomalyModel
    from darts.ad.anomaly_model import (
        ForecastingAnomalyModel as ForecastingAnomalyModel,
    )
    from darts.ad.detectors import QuantileDetector as QuantileDetector
    from darts.ad.detectors import ThresholdDetector as ThresholdDetector
    from darts.ad.scorers import CauchyNLLScorer as CauchyNLLScorer
    from darts.ad.scorers import DifferenceScorer as DifferenceScorer
    from darts.ad.scorers import ExponentialNLLScorer as ExponentialNLLScorer
    from darts.ad.scorers import GammaNLLScorer as GammaNLLScorer
    from darts.ad.scorers import GaussianNLLScorer as GaussianNLLScorer
    from darts.ad.scorers import KMeansScorer as KMeansScorer
    from darts.ad.scorers import LaplaceNLLScorer as LaplaceNLLScorer
    from darts.ad.scorers import NormScorer as NormScorer
    from darts.ad.scorers import PoissonNLLScorer as PoissonNLLScorer
    from darts.ad.scorers import PyODScorer as PyODScorer
    from darts.ad.scorers import WassersteinScorer as WassersteinScorer

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
