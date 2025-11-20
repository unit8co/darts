"""
Likelihoods for `StatsForecast` Models
--------------------------------------
"""

from abc import ABC

import numpy as np

from darts.logging import get_logger
from darts.utils.likelihood_models.base import (
    Likelihood,
    LikelihoodType,
    quantile_names,
)
from darts.utils.utils import _check_quantiles, sample_from_quantiles

logger = get_logger(__name__)


class QuantilePrediction(Likelihood, ABC):
    def __init__(self, quantiles: list[float]):
        """Quantile Prediction Likelihood

        Can be used to generate quantile predictions for any Darts model that wraps a `statsforecast` model.

        Parameters
        ----------
        quantiles
            A list of quantiles. Default: `[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]`.
        """
        if quantiles is None:
            quantiles = [
                0.01,
                0.05,
                0.1,
                0.25,
                0.5,
                0.75,
                0.9,
                0.95,
                0.99,
            ]
        else:
            quantiles = sorted(quantiles)
            _check_quantiles(quantiles)
        self.quantiles = quantiles
        self._median_idx = quantiles.index(0.5)
        self.levels = [
            round(100 * (q_h - q_l), 2)
            for q_l, q_h in zip(
                quantiles[: self._median_idx], quantiles[self._median_idx + 1 :][::-1]
            )
        ]

        super().__init__(
            likelihood_type=LikelihoodType.Quantile,
            parameter_names=quantile_names(self.quantiles),
        )
        # ignore additional attrs for equality tests
        self.ignore_attrs_equality += ["_median_idx", "levels"]

    def predict(self, model_output: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Generates sampled or direct likelihood parameter predictions.

        Parameters
        ----------
        model_output
            The output of the StatsForecast model.
        num_samples
            Number of times a prediction is sampled from the likelihood model / distribution.
            If `1` and `predict_likelihood_parameters=False`, returns median / mean predictions.
        """
        if num_samples > 1:
            return self.sample(model_output, num_samples=num_samples)

        # likelihood parameters or median prediction are handled by the `levels` parameter in
        # `StatsForecastModel._predict()`
        return model_output

    def sample(self, model_output: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Samples a prediction from the likelihood distribution and the predicted parameters.
        """
        return sample_from_quantiles(
            vals=model_output,
            quantiles=np.array(self.quantiles),
            num_samples=num_samples,
        )
