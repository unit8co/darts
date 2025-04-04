"""
Likelihoods for `RegressionModel`
---------------------------------
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from darts.logging import get_logger, raise_log
from darts.utils.likelihood_models.base import (
    Likelihood,
    LikelihoodType,
    quantile_names,
)
from darts.utils.utils import _check_quantiles

logger = get_logger(__name__)


class SKLearnLikelihood(Likelihood, ABC):
    def __init__(
        self,
        likelihood_type: LikelihoodType,
        parameter_names: list[str],
        n_outputs: int,
        random_state: Optional[int] = None,
    ):
        """Base class for sklearn wrapper (e.g. `RegressionModel`) likelihoods.

        Parameters
        ----------
        likelihood_type
            A pre-defined `LikelihoodType`.
        parameter_names
            The likelihood (distribution) parameter names.
        n_outputs
            The number of predicted outputs per model call. `1` if `multi_models=False`, otherwise
            `output_chunk_length`.
        random_state
            Optionally, control the randomness of the sampling.
        """
        self._n_outputs = n_outputs
        self._rng = np.random.default_rng(seed=random_state)
        super().__init__(
            likelihood_type=likelihood_type,
            parameter_names=parameter_names,
        )
        # ignore additional attrs for equality tests
        self.ignore_attrs_equality += ["_n_outputs", "_rng"]

    def predict(
        self,
        model,
        x: np.ndarray,
        num_samples: int,
        predict_likelihood_parameters: bool,
        **kwargs,
    ) -> np.ndarray:
        """
        Generates sampled or direct likelihood parameter predictions.

        Parameters
        ----------
        model
            The Darts `RegressionModel`.
        x
            The input feature array passed to the underlying estimator's `predict()` method.
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
        model_output = self._estimator_predict(model, x, **kwargs)
        if predict_likelihood_parameters:
            return self.predict_likelihood_parameters(model_output)
        elif num_samples == 1:
            return self._get_median_prediction(model_output)
        else:
            return self.sample(model_output)

    @abstractmethod
    def sample(self, model_output: np.ndarray) -> np.ndarray:
        """
        Samples a prediction from the likelihood distribution and the predicted parameters.
        """

    @abstractmethod
    def predict_likelihood_parameters(self, model_output: np.ndarray) -> np.ndarray:
        """
        Returns the distribution parameters as a array, extracted from the raw model outputs.
        """

    @abstractmethod
    def _estimator_predict(
        self,
        model,
        x: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Computes the model output.

        Parameters
        ----------
        model
            The Darts `RegressionModel`.
        x
            The input feature array passed to the underlying estimator's `predict()` method.
        kwargs
            Some kwargs passed to the underlying estimator's `predict()` method.
        """

    @abstractmethod
    def _get_median_prediction(self, model_output: np.ndarray) -> np.ndarray:
        """
        Gets the median prediction per component extracted from the model output.
        """


class GaussianLikelihood(SKLearnLikelihood):
    def __init__(
        self,
        n_outputs: int,
        random_state: Optional[int] = None,
    ):
        """
        Gaussian distribution [1]_.

        Parameters
        ----------
        n_outputs
            The number of predicted outputs per model call. `1` if `multi_models=False`, otherwise
            `output_chunk_length`.
        random_state
            Optionally, control the randomness of the sampling.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Normal_distribution
        """
        super().__init__(
            likelihood_type=LikelihoodType.Gaussian,
            parameter_names=["mu", "sigma"],
            n_outputs=n_outputs,
            random_state=random_state,
        )

    def sample(self, model_output: np.ndarray) -> np.ndarray:
        # shape (n_components * output_chunk_length, n_series * n_samples, 2)
        # [mu, sigma] on the last dimension, grouped by component
        n_entries, n_samples, n_params = model_output.shape

        # get samples (n_components * output_chunk_length, n_series * n_samples)
        samples = self._rng.normal(
            model_output[:, :, 0],  # mean
            model_output[:, :, 1],  # variance
        )
        # reshape to (n_series * n_samples, n_components * output_chunk_length)
        samples = samples.transpose()
        # reshape to (n_series * n_samples, output_chunk_length, n_components)
        samples = samples.reshape(n_samples, self._n_outputs, -1)
        return samples

    def predict_likelihood_parameters(self, model_output: np.ndarray) -> np.ndarray:
        # shape (n_components * output_chunk_length, n_series * n_samples, 2)
        # [mu, sigma] on the last dimension, grouped by component
        n_samples = model_output.shape[1]

        # reshape to (n_series * n_samples, output_chunk_length, n_components)
        params_reshaped = model_output.transpose(1, 0, 2).reshape(
            n_samples, self._n_outputs, -1
        )
        return params_reshaped

    def _estimator_predict(
        self,
        model,
        x: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        # returns samples computed from double-valued inputs [mean, variance].
        # `x` is of shape (n_series * n_samples, n_regression_features)
        # `model_output` is of shape:
        #  - (n_series * n_samples, 2): if univariate & output_chunk_length == 1
        #  - (2, n_series * n_samples, n_components * output_chunk_length): otherwise
        # where the axis with 2 dims is mu, sigma
        model_output = model.model.predict(x, **kwargs)
        output_dim = len(model_output.shape)

        # univariate & single-chunk output
        if output_dim <= 2:
            # embedding well shaped 2D output into 3D
            model_output = np.expand_dims(model_output, axis=0)
        else:
            # we transpose to get mu, sigma couples on last axis
            # shape becomes: (n_components * output_chunk_length, n_series * n_samples, 2)
            model_output = model_output.transpose()

        # shape (n_components * output_chunk_length, n_series * n_samples, 2)
        return model_output

    def _get_median_prediction(self, model_output: np.ndarray) -> np.ndarray:
        # shape (n_components * output_chunk_length, n_series * n_samples, 2)
        # [mu, sigma] on the last dimension, grouped by component
        k = model_output.shape[1]
        # extract mu (mean) per component
        component_medians = slice(0, None, self.num_parameters)
        # shape (n_series * n_samples, output_chunk_length, n_components)
        return model_output[:, :, component_medians].reshape(k, self._n_outputs, -1)


class PoissonLikelihood(SKLearnLikelihood):
    def __init__(
        self,
        n_outputs: int,
        random_state: Optional[int] = None,
    ):
        """
        Poisson distribution [1]_.

        Parameters
        ----------
        n_outputs
            The number of predicted outputs per model call. `1` if `multi_models=False`, otherwise
            `output_chunk_length`.
        random_state
            Optionally, control the randomness of the sampling.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Poisson_distribution
        """
        super().__init__(
            likelihood_type=LikelihoodType.Poisson,
            parameter_names=["lambda"],
            n_outputs=n_outputs,
            random_state=random_state,
        )

    def sample(self, model_output: np.ndarray) -> np.ndarray:
        # shape (n_series * n_samples, output_chunk_length, n_components)
        return self._rng.poisson(lam=model_output).astype(float)

    def predict_likelihood_parameters(self, model_output: np.ndarray) -> np.ndarray:
        # lambdas on the last dimension, grouped by component
        return model_output

    def _estimator_predict(
        self,
        model,
        x: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        k = x.shape[0]
        # returns shape (n_series * n_samples, output_chunk_length, n_components)
        return model.model.predict(x, **kwargs).reshape(k, self._n_outputs, -1)

    def _get_median_prediction(self, model_output: np.ndarray) -> np.ndarray:
        # shape (n_series * n_samples, output_chunk_length, n_components)
        # lambda is already the median prediction
        return model_output


class QuantileRegression(SKLearnLikelihood):
    def __init__(
        self,
        n_outputs: int,
        random_state: Optional[int] = None,
        quantiles: Optional[list[float]] = None,
    ):
        """
        Quantile Regression [1]_.

        Parameters
        ----------
        n_outputs
            The number of predicted outputs per model call. `1` if `multi_models=False`, otherwise
            `output_chunk_length`.
        random_state
            Optionally, control the randomness of the sampling.
        quantiles
            A list of quantiles. Default: `[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]`.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Quantile_regression
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

        super().__init__(
            likelihood_type=LikelihoodType.Quantile,
            parameter_names=quantile_names(self.quantiles),
            n_outputs=n_outputs,
            random_state=random_state,
        )
        self.ignore_attrs_equality += ["_median_idx"]

    def sample(self, model_output: np.ndarray) -> np.ndarray:
        # model_output is of shape (n_series * n_samples, output_chunk_length, n_components, n_quantiles)
        # sample uniformly between [0, 1] (for each batch example) and return the
        # linear interpolation between the fitted quantiles closest to the sampled value.
        k, n_times, n_components, n_quantiles = model_output.shape

        # obtain samples
        probs = self._rng.uniform(
            size=(
                k,
                n_times,
                n_components,
                1,
            )
        )

        # add dummy dim
        probas = np.expand_dims(probs, axis=-2)

        # tile and transpose
        p = np.tile(probas, (1, 1, 1, n_quantiles, 1)).transpose((0, 1, 2, 4, 3))

        # prepare quantiles
        tquantiles = np.array(self.quantiles).reshape((1, 1, 1, -1))

        # calculate index of the largest quantile smaller than the sampled value
        left_idx = np.sum(p > tquantiles, axis=-1)

        # obtain index of the smallest quantile larger than the sampled value
        right_idx = left_idx + 1

        # repeat the model output on the edges
        repeat_count = [1] * n_quantiles
        repeat_count[0] = 2
        repeat_count[-1] = 2
        repeat_count = np.array(repeat_count)
        shifted_output = np.repeat(model_output, repeat_count, axis=-1)

        # obtain model output values corresponding to the quantiles left and right of the sampled value
        left_value = np.take_along_axis(shifted_output, left_idx, axis=-1)
        right_value = np.take_along_axis(shifted_output, right_idx, axis=-1)

        # add 0 and 1 to quantiles
        ext_quantiles = [0.0] + self.quantiles + [1.0]
        expanded_q = np.tile(np.array(ext_quantiles), left_idx.shape)

        # calculate closest quantiles to the sampled value
        left_q = np.take_along_axis(expanded_q, left_idx, axis=-1)
        right_q = np.take_along_axis(expanded_q, right_idx, axis=-1)

        # linear interpolation
        weights = (probs - left_q) / (right_q - left_q)
        inter = left_value + weights * (right_value - left_value)

        # shape (n_series * n_samples, output_chunk_length, n_components * n_quantiles)
        return inter.squeeze(-1)

    def predict_likelihood_parameters(self, model_output: np.ndarray) -> np.ndarray:
        # shape (n_series * n_samples, output_chunk_length, n_components, n_quantiles)
        # quantiles on the last dimension, grouped by component
        k, n_times, n_components, n_quantiles = model_output.shape
        # last dim : [comp_1_q_1, ..., comp_1_q_n, ..., comp_n_q_1, ..., comp_n_q_n]
        # shape (n_series * n_samples, output_chunk_length, n_components * n_quantiles)
        return model_output.reshape(k, n_times, n_components * n_quantiles)

    def _estimator_predict(
        self,
        model,
        x: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        # `x` is of shape (n_series * n_samples, n_regression_features)
        k = x.shape[0]
        model_outputs = []
        for quantile, fitted in model._model_container.items():
            model.model = fitted
            # model output has shape (n_series * n_samples, output_chunk_length, n_components)
            model_output = fitted.predict(x, **kwargs).reshape(k, self._n_outputs, -1)
            model_outputs.append(model_output)
        model_outputs = np.stack(model_outputs, axis=-1)
        # shape (n_series * n_samples, output_chunk_length, n_components, n_quantiles)
        return model_outputs

    def _get_median_prediction(self, model_output: np.ndarray) -> np.ndarray:
        # shape (n_series * n_samples, output_chunk_length, n_components, n_quantiles)
        k, n_times, n_components, n_quantiles = model_output.shape
        # extract the median quantiles per component
        component_medians = slice(self._median_idx, None, n_quantiles)
        # shape (n_series * n_samples, output_chunk_length, n_components)
        return model_output[:, :, :, component_medians].reshape(
            k, n_times, n_components
        )


def _check_likelihood(likelihood: str, available_likelihoods: list[str]):
    """Check whether the likelihood is supported.

    Parameters
    ----------
    likelihood
        The likelihood name. Must be one of ('gaussian', 'poisson', 'quantile').
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
    n_outputs: int,
    random_state: Optional[int],
    quantiles: Optional[list[float]],
) -> Optional[SKLearnLikelihood]:
    """Get the `Likelihood` object for `RegressionModel`.

    Parameters
    ----------
    likelihood
        The likelihood name. Must be one of ('gaussian', 'poisson', 'quantile').
    n_outputs
        The number of predicted outputs per model call. `1` if `multi_models=False`, otherwise
        `output_chunk_length`.
    random_state
        Optionally, control the randomness of the sampling.
    quantiles
        Optionally, a list of quantiles. Only effective for `likelihood='quantile'`.
    """
    if likelihood is None:
        return None
    elif likelihood == "gaussian":
        return GaussianLikelihood(n_outputs=n_outputs, random_state=random_state)
    elif likelihood == "poisson":
        return PoissonLikelihood(n_outputs=n_outputs, random_state=random_state)
    elif likelihood == "quantile":
        return QuantileRegression(
            n_outputs=n_outputs, random_state=random_state, quantiles=quantiles
        )
    else:
        raise_log(
            ValueError(
                f"Invalid `likelihood='{likelihood}'`. Must be one of ('gaussian', 'poisson', 'quantile')"
            ),
            logger=logger,
        )
