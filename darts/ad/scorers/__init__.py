"""
Anomaly Scorers
---------------

# TODO

Anomaly scorers can be trainable (FittableAnomalyScorer) or not trainable (NonFittableAnomalyScorer).

The scorers have the following main functions:
    - ``score_from_prediction()``
        Takes as input two (sequence of) series and returns the anomaly score of each pairwise element.
        An anomaly score is a series that represents how anomalous the considered point (if window = 1)
        or past W points are (if window is equal to W). The higher the value, the more anomalous the sample.
        The interpretability of the score is dependent on the scorer.

    - ``eval_accuracy_from_prediction()``
        Takes as input two (sequence of) series, computes the anomaly score of each pairwise element, and
        returns the score of an agnostic threshold metric (AUC-ROC or AUC-PR) compared to the ground truth
        of anomalies. The returned value is between 0 and 1. 1 indicates that the scorer could perfectly
        separate the anomalous point from the normal ones.

The trainable scorers have the following additional functions:
    - ``fit_from_prediction()``
        Takes two (sequence of) series as input and fits the scorer. This task is dependent on the scorer,
        but as a general case the scorer will calibrate its scoring function based on the training series that is
        considered to be anomaly-free. This training phase will allow the scorer to detect anomalies during
        the scoring phase, by comparing the series to score with the anomaly-free series seen during training.

For the trainable scorers, the previous three functions expect a tuple of (sequence of) series as input. A
function is used to compute a "difference" between the prediction series and the observation series,
in order to obtain a single "difference" series. The trainable scorer is then
applied on this series. The function is by default the absolute difference, but it can be changed thanks to
the parameter named ``diff_fn``.

It is possible to apply the trainable scorer directly on a series. This is allowed by the three following
functions. They are equivalent to the ones described previously but take as input only one (sequence of) series:
    - ``score()``
    - ``eval_accuracy()``
    - ``fit()``

As an example, the ``KMeansScorer``, which is a ``FittableAnomalyScorer``, can be applied thanks to the functions:
    - ``fit()`` and ``score()``: directly on a series to uncover the relationship between the different
    dimensions (over timesteps within windows and/or over dimensions of multivariate series).
    - ``fit_from_prediction`` and ``score_from_prediction``: which will compute a difference (residuals)
    between some prediction (coming e.g., from a forecasting model) and the series itself.
    The scorer will then flag residuals that are distant from the clusters found during the training phase.

Most of the scorers have the following main parameters:
    - `window`:
        Integer value indicating the size of the window W used by the scorer to transform the series into
        an anomaly score. A scorer will slice the given series into subsequences of size W and returns
        a value indicating how anomalous these subset of W values are. The window size must be commensurate
        to the expected durations of the anomalies one is looking for.
    - `component_wise`
        boolean parameter indicating how the scorer should behave with multivariate inputs series. If set to
        True, the model will treat each series dimension independently. If set to False, the model will
        consider the dimensions jointly in the considered `window` W to compute the score.

More details can be found in the API documentation of each scorer.
"""

from .cauchy_nll_scorer import CauchyNLLScorer
from .difference_scorer import DifferenceScorer
from .exponential_nll_scorer import ExponentialNLLScorer
from .gamma_nll_scorer import GammaNLLScorer
from .gaussian_nll_scorer import GaussianNLLScorer
from .kmeans_scorer import KMeansScorer
from .laplace_nll_scorer import LaplaceNLLScorer
from .norm_scorer import NormScorer
from .poisson_nll_scorer import PoissonNLLScorer
from .pyod_scorer import PyODScorer
from .scorers import FittableAnomalyScorer, NonFittableAnomalyScorer
from .wasserstein_scorer import WassersteinScorer
