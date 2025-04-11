"""
Likelihoods for `StatsForecast` Models
--------------------------------------
"""

from abc import ABC
from typing import Optional

import numpy as np

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.likelihood_models.base import (
    Likelihood,
    LikelihoodType,
    quantile_names,
)
from darts.utils.utils import _check_quantiles, sample_from_quantiles

logger = get_logger(__name__)


class StatsForecastLikelihood(Likelihood, ABC):
    def __init__(self, quantiles: list[float]):
        """Base likelihood class for any Darts model that wraps a `statsforecast` model.

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

    def predict(
        self,
        model,
        n: int,
        future_covariates: Optional[TimeSeries],
        num_samples: int,
        predict_likelihood_parameters: bool,
        **kwargs,
    ) -> np.ndarray:
        """
        Generates sampled or direct likelihood parameter predictions.

        Parameters
        ----------
        model
            One of Darts' statsforecast models.
        n
            The number of time steps after the end of the training time series for which to produce predictions
        future_covariates
            Optionally, the future-known covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension.
        num_samples
            Number of times a prediction is sampled from the likelihood model / distribution.
            If `1` and `predict_likelihood_parameters=False`, returns median / mean predictions.
        predict_likelihood_parameters
            If set to `True`, generates likelihood parameter predictions instead of sampling from the
            likelihood model / distribution. Only supported with `num_samples = 1` and
            `n<=output_chunk_length`.
        kwargs
            Some kwargs passed to the underlying estimator's `predict()` method.
        """
        levels = (
            self.levels if num_samples > 1 or predict_likelihood_parameters else None
        )
        model_output = self._estimator_predict(
            model, n=n, future_covariates=future_covariates, levels=levels, **kwargs
        )
        if predict_likelihood_parameters:
            return self.predict_likelihood_parameters(model_output)
        elif num_samples == 1:
            return self._get_median_prediction(model_output)
        else:
            return self.sample(model_output, num_samples=num_samples)

    def sample(self, model_output: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Samples a prediction from the likelihood distribution and the predicted parameters.
        """
        return sample_from_quantiles(
            vals=model_output,
            quantiles=np.array(self.quantiles),
            num_samples=num_samples,
        )

    def predict_likelihood_parameters(self, model_output: np.ndarray) -> np.ndarray:
        """
        Returns the distribution parameters as an array, extracted from the raw model outputs.
        """
        return model_output

    def _estimator_predict(
        self,
        model,
        n: int,
        future_covariates: Optional[TimeSeries],
        levels: Optional[list[float]],
        **kwargs,
    ) -> np.ndarray:
        """
        Computes the model output.

        Parameters
        ----------
        model
            One of Darts' statsforecast models.
        n
            The number of time steps after the end of the training time series for which to produce predictions
        future_covariates
            Optionally, the future-known covariates series needed as inputs for the model.
            They must match the covariates used for training in terms of dimension.
        levels
            The confidence levels (0. - 100.) for the prediction intervals.
        kwargs
            Some kwargs passed to the underlying estimator's `predict()` method.
        """
        forecast_dict = model.model.predict(
            h=n,
            X=(
                future_covariates.values(copy=False)
                if future_covariates is not None
                and model._supports_native_future_covariates
                else None
            ),
            level=levels,  # ask one std for the confidence interval.
        )
        vals = _unpack_sf_dict(forecast_dict, levels=levels)
        if (
            future_covariates is not None
            and not model._supports_native_future_covariates
        ):
            mu_linreg = model._linreg.predict(n, future_covariates=future_covariates)
            mu_linreg_values = mu_linreg.values(copy=False).reshape(n, 1)
            vals += mu_linreg_values
        return vals

    def _get_median_prediction(self, model_output: np.ndarray) -> np.ndarray:
        """
        Gets the median prediction per component extracted from the model output.
        """
        return model_output[:, self._median_idx]


class QuantileRegression(StatsForecastLikelihood):
    def __init__(self, quantiles: Optional[list[float]] = None):
        """
        Quantile Regression [1]_.

        Parameters
        ----------
        quantiles
            A list of quantiles. Default: `[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]`.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Quantile_regression
        """
        super().__init__(quantiles=quantiles)


def _unpack_sf_dict(
    forecast_dict: dict,
    levels: Optional[list[float]],
) -> np.ndarray:
    """Unpack the dictionary that is returned by the StatsForecast 'predict()' method.

    Into an array of quantile predictions with shape (n (horizon), n quantiles) ordered by increasing quantile.
    """
    mu = np.expand_dims(forecast_dict["mean"], -1)
    if levels is None:
        return mu

    lows = np.concatenate(
        [np.expand_dims(forecast_dict[f"lo-{level}"], -1) for level in levels], axis=1
    )
    highs = np.concatenate(
        [np.expand_dims(forecast_dict[f"hi-{level}"], -1) for level in levels[::-1]],
        axis=1,
    )
    return np.concatenate([lows, mu, highs], axis=1)


def _check_likelihood(likelihood: str, available_likelihoods: list[str]):
    """Check whether the likelihood is supported.

    Parameters
    ----------
    likelihood
        The likelihood name. Must be one of ('quantile').
    available_likelihoods
        A list of supported likelihood names.
    """
    if likelihood not in available_likelihoods:
        raise_log(
            ValueError(
                f"Invalid `likelihood='{likelihood}'`. Must be one of {available_likelihoods}"
            ),
            logger=logger,
        )


def _get_likelihood(
    likelihood: Optional[str],
    quantiles: Optional[list[float]],
) -> Optional[StatsForecastLikelihood]:
    """Get the `Likelihood` object for `RegressionModel`.

    Parameters
    ----------
    likelihood
        The likelihood name. Must be one of ('quantile').
    quantiles
        Optionally, a list of quantiles. Only effective for `likelihood='quantile'`.
    """
    if likelihood == "quantile":
        return QuantileRegression(quantiles=quantiles)
    else:
        raise_log(
            ValueError(
                f"Invalid `likelihood='{likelihood}'`. Must be one of ('quantile')"
            ),
            logger=logger,
        )
