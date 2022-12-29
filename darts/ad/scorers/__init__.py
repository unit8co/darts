"""
Anomaly Scorers
---------------

Scorers are at the core of the anomaly detection module. They
produce anomaly scores time series, either for series directly (``score()``),
or for series accompanied by some predictions (``score_from_prediction()``).

The higher an anomaly score is, the more "anomalous" the corresponding
time period is. Scorers can work over time windows, and the length of the window is related
to the time scale over which anomalies are expected to occur.
The interpretability of the anomaly score is dependent on the scorer.

The function ``score_from_prediction()`` works by taking some "difference" (or "residual")
between the prediction and the actual series (captured by the ``"diff_fn"`` parameter).
Some scorers are trainable (e.g., ``KMeansScorer``, which learns clusters over historical data),
in which case the ``score()`` function can be used to score new series.
Other scorers are not trainable (e.g., ``NormScorer``, which simply takes the Lp-norm between
predicted values and actual values over windows). In this latter case ``score()`` cannot be
used and scoring is only possible using ``score_from_prediction()``.

Some scorers can handle probabilistic predictions from models (at the moment all the "NLL" scorers),
while others handle deterministic predictions (e.g., ``KMeansScorer``).

As an example, the ``KMeansScorer``, which is trainable, can be applied using the functions:

- ``fit()`` and ``score()``: directly on a series to uncover the relationships between the different
  dimensions (over timesteps within windows and/or over dimensions of multivariate series).
- ``fit_from_prediction`` and ``score_from_prediction``: which will compute a difference (residuals)
  between the prediction (coming e.g., from a forecasting model) and the series itself.
  When scoring, the scorer will attribute a higher score to residuals that are distant
  from the clusters found during the training phase.
    
Note that `Anomaly Models <https://unit8co.github.io/darts/generated_api/darts.ad.anomaly_model.html>`_
can be used to conveniently combine any of Darts forecasting and filtering models with one or multiple scorers.

Most of the scorers have the following main parameters:

- `window`:
  Integer value indicating the size of the window W used by the scorer to transform the series into
  an anomaly score. A scorer will slice the given series into subsequences of size W and returns
  a value indicating how anomalous these subset of W values are. The window size should be commensurate
  to the expected durations of the anomalies one is looking for.
- `component_wise`:
  boolean parameter indicating how the scorer should behave with multivariate series. If set to
  True, the model will treat each series dimension independently. If set to False, the model will
  consider the dimensions jointly in the considered `window` W to compute the score.


Other useful functions are:

- ``eval_accuracy_from_prediction()``
  Takes as input two (sequence of) series, computes all the anomaly scores, and
  returns the value of an agnostic threshold metric (AUC-ROC or AUC-PR) based on some known ground truth
  of anomalies. The returned value is between 0 and 1, with 1 indicating that the scorer could perfectly
  separate the anomalous point from the normal ones.

- ``fit_from_prediction()``
  Takes two (sequence of) series as input and fits the scorer. This task is dependent on the scorer,
  but as a general case the scorer will calibrate its scoring function based on the training series that is
  considered to be anomaly-free. This training phase will allow the scorer to detect anomalies during
  the scoring phase, by comparing the series to score with the anomaly-free series seen during training.


More details can be found in the API documentation of each scorer.
"""

from .difference_scorer import DifferenceScorer
from .kmeans_scorer import KMeansScorer
from .nll_cauchy_scorer import CauchyNLLScorer
from .nll_exponential_scorer import ExponentialNLLScorer
from .nll_gamma_scorer import GammaNLLScorer
from .nll_gaussian_scorer import GaussianNLLScorer
from .nll_laplace_scorer import LaplaceNLLScorer
from .nll_poisson_scorer import PoissonNLLScorer
from .norm_scorer import NormScorer
from .pyod_scorer import PyODScorer
from .scorers import FittableAnomalyScorer, NonFittableAnomalyScorer
from .wasserstein_scorer import WassersteinScorer
