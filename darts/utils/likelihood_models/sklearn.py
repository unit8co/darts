"""
Likelihoods for `SKLearnModel`
------------------------------
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from darts.logging import get_logger, raise_log
from darts.timeseries import TimeSeries
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
        """Base class for sklearn wrapper (e.g. `SKLearnModel`) likelihoods.

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
            The Darts `SKLearnModel`.
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
        model_output = self._estimator_predict(model, x=x, **kwargs)
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
            The Darts `SKLearnModel`.
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


class ClassProbabilityLikelihood(SKLearnLikelihood):
    def __init__(
        self,
        n_outputs: int,
        random_state: Optional[int] = None,
    ):
        """
        Class probability likelihood.
        Likelihood to predict the probability of each class for a forecasting classification task.

        Parameters
        ----------
        n_outputs
            The number of predicted outputs per model call. `1` if `multi_models=False`, otherwise
            `output_chunk_length`.
        random_state
            Optionally, control the randomness of the sampling.
        """
        super().__init__(
            likelihood_type=LikelihoodType.ClassProbability,
            n_outputs=n_outputs,
            random_state=random_state,
            parameter_names=[],
        )
        self._index_first_param_per_component: Optional[np.ndarray] = None
        self._classes: list[np.ndarray] = None

    def fit(self, model):
        """
        Fits the likelihood to the model.

        Parameters
        ----------
        model
            The model to fit the likelihood to. The model is expected to be fitted and have a `class_labels` attribute.
        """
        if not hasattr(model, "class_labels"):
            raise_log(
                ValueError(
                    "The model must have a `class_labels` attribute to fit the likelihood."
                ),
                logger,
            )

        self._classes = model.class_labels

        # estimators/classes are ordered by chunk then by component: [classes_comp0_chunk0, classes_comp1_chunk0, ...]
        # Since classes are same across chunk for same component, we can just take the first chunk for each component
        num_components = len(self._classes) // self._n_outputs

        # index of the first parameter of each component in the likelihood parameters
        # e.g. [0, 2, 4, 6] for 3 components with 2 parameters each
        self._index_first_param_per_component = np.insert(
            np.cumsum([
                len(estimator_classes)
                for estimator_classes in self._classes[:num_components]
            ]),
            0,
            0,
        )

        self._parameter_names = [
            f"p_{int(label)}"
            for i in range(num_components)
            for label in self._classes[i]
        ]
        return self

    def component_names(self, input_series: TimeSeries) -> list[str]:
        """Generates names for the parameters of the Likelihood."""
        if self._index_first_param_per_component is None:
            raise_log(
                ValueError("`component_names` requires the likelihood to be fitted.")
            )
        # format: <component_name>_p_<label>
        return [
            f"{component_name}_{parameter_name}"
            for i, component_name in enumerate(input_series.components)
            for parameter_name in self.parameter_names[
                self._index_first_param_per_component[
                    i
                ] : self._index_first_param_per_component[i + 1]
            ]
        ]

    def _estimator_predict(
        self,
        model,
        x: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        # list of length n_components * output_chunk_length of numpy arrays of shape (n_samples*n_series, n_classes)
        model_output = model.model.predict_proba(x, **kwargs)
        if not isinstance(model_output, list):
            model_output = [model_output]

        # shape: (output_chunk_length, n_series * n_samples, n_likelihood_parameters)
        # n_likelihood_parameters is the sum of the number of classes for each component
        class_proba = np.zeros((
            self._n_outputs,
            model_output[0].shape[0],
            self.num_parameters,
        ))

        # ordered first by _n_outputs then by component: [comp0_chunk0, comp1_chunk0, ..]
        n_component = len(model_output) // self._n_outputs
        # filling the class_proba array with the predicted probabilities
        for i, params_proba in enumerate(model_output):
            component_index = i % n_component
            output_index = i // n_component
            class_proba[
                output_index,
                :,
                self._index_first_param_per_component[
                    component_index
                ] : self._index_first_param_per_component[component_index + 1],
            ] = params_proba

        # shape (output_chunk_length, n_series * n_samples, n_likelihood_parameters)
        # n_likelihood_parameters is the sum of the number of classes for each component
        # n_likelihood_parameters are ordered as in component_names:
        # [comp0_l1, comp0_l2, comp1_l1, comp1_l2, comp1_l3,...]
        # should be accessed through _index_first_param_per_component (for comp1 => 2)
        return class_proba

    def sample(self, model_output: np.ndarray) -> np.ndarray:
        """
        Samples a prediction from the likelihood distribution and the predicted parameters.
        """
        # shape (output_chunk_length, n_series * n_samples, n_likelihood_parameters)
        n_output, n_samples, _ = model_output.shape

        n_component = len(self._classes) // self._n_outputs

        # shape (output_chunk_length, n_series * n_samples, n_components)
        preds = np.empty((n_output, n_samples, n_component), dtype=int)

        # Some models have an approximation error, the probabilities are adjusted
        # if their total is below the 1e-7 tolerance threshold around 1.
        for component_idx, (component_start, component_end) in enumerate(
            zip(
                self._index_first_param_per_component[:-1],
                self._index_first_param_per_component[1:],
            )
        ):
            component_probabilities = model_output[:, :, component_start:component_end]
            total_proba = component_probabilities.sum(
                axis=2
            )  # shape  (n_output, n_samples)
            difference = np.ones((n_output, n_samples)) - total_proba
            tolerance = 1e-7
            if np.any(np.abs(difference) > tolerance):
                raise_log(
                    ValueError(
                        "The class probabilities returned by the model do not sum to one"
                    )
                )

            component_probabilities += (
                difference[:, :, np.newaxis] / component_probabilities.shape[2]
            )

            # argmax stops at the first True
            # first time the random sample is greater than cumulative probability
            # [0.2, 0.1, 0.5, 0.2] -> [0.2, 0.3, 0.8, 1]
            # random: 0.7 -> idx = 2, this outcomes has 0.5 probability of happening
            sampled_idx = np.argmax(
                self._rng.uniform(0, 1, size=(difference.shape))[:, :, np.newaxis]
                < np.cumsum(component_probabilities, axis=2),
                axis=2,
            )
            preds[:, :, component_idx] = np.take(self._classes, sampled_idx)

        return preds.transpose(1, 0, 2)

    def predict_likelihood_parameters(self, model_output: np.ndarray) -> np.ndarray:
        """
        Returns the distribution parameters as a array, extracted from the raw model outputs.
        """
        # reshape to (n_series * n_samples, output_chunk_length, n_likelihood_parameters)
        return model_output.transpose(1, 0, 2)

    def _get_median_prediction(self, model_output: np.ndarray) -> np.ndarray:
        """
        Gets the class label with highest predicted probability per component extracted
         from the model output.
        """
        # shape (output_chunk_length, n_series * n_samples, n_likelihood_parameters)
        n_output, n_samples, n_params = model_output.shape
        # shape (output_chunk_length, n_series * n_samples, n_components)
        n_component = len(self._classes) // self._n_outputs
        preds = np.empty((n_output, n_samples, n_component), dtype=int)

        for component_idx, (component_start, component_end) in enumerate(
            zip(
                self._index_first_param_per_component[:-1],
                self._index_first_param_per_component[1:],
            )
        ):
            # shape (output_chunk_length, n_series * n_samples)
            indices = np.argmax(
                model_output[:, :, component_start:component_end], axis=2
            )

            preds[:, :, component_idx] = self._classes[component_idx][indices]

        # reshape to (n_series * n_samples, output_chunk_length, n_components)
        return preds.transpose(1, 0, 2)


def _get_likelihood(
    likelihood: Optional[str],
    n_outputs: int,
    available_likelihoods: list[LikelihoodType],
    random_state: Optional[int] = None,
    quantiles: Optional[list[float]] = None,
) -> Optional[SKLearnLikelihood]:
    """Get the `Likelihood` object for `SKLearnModel`.

    Parameters
    ----------
    likelihood
        The likelihood name. Must be one of `available_likelihoods` type value or `None`.
    n_outputs
        The number of predicted outputs per model call. `1` if `multi_models=False`, otherwise
        `output_chunk_length`.
    available_likelihoods
        The list of available likelihood types for the model.
    random_state
        Optionally, control the randomness of the sampling.
    quantiles
        Optionally, a list of quantiles. Only effective for `likelihood='quantile'`.

    """

    if likelihood is None:
        return None

    # Convert LikelihoodType to string
    available_likelihoods = [
        likelihood.value if isinstance(likelihood, LikelihoodType) else likelihood
        for likelihood in available_likelihoods
    ]

    if likelihood not in available_likelihoods:
        raise_log(
            ValueError(
                f"Invalid `likelihood='{likelihood}'`. Must be one of {available_likelihoods}"
            ),
            logger=logger,
        )

    if likelihood == LikelihoodType.Gaussian.value:
        return GaussianLikelihood(n_outputs=n_outputs, random_state=random_state)
    elif likelihood == LikelihoodType.Poisson.value:
        return PoissonLikelihood(n_outputs=n_outputs, random_state=random_state)
    elif likelihood == LikelihoodType.Quantile.value:
        return QuantileRegression(
            n_outputs=n_outputs, random_state=random_state, quantiles=quantiles
        )
    elif likelihood == LikelihoodType.ClassProbability.value:
        return ClassProbabilityLikelihood(
            n_outputs=n_outputs, random_state=random_state
        )
