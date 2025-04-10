from typing import Optional

import numpy as np

from darts.logging import get_logger, raise_log
from darts.utils.likelihood_models.base import LikelihoodType
from darts.utils.likelihood_models.sklearn import SKLearnLikelihood

logger = get_logger(__name__)


class ClassProbabilityLikelihood(SKLearnLikelihood):
    def __init__(
        self,
        n_outputs: int,
        random_state: Optional[int] = None,
    ):
        """
        Classes probability likelihood.
        This likelihood is used for classification tasks where the model predicts classes probability.

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

    def fit(self, model):
        """
        Fits the likelihood to the model.

        Parameters
        ----------
        model
            The model to fit the likelihood to. The model is expected to be fitted and have a `classes_` attribute.
        """
        if not hasattr(model, "classes_"):
            raise_log(
                ValueError(
                    "The model must have a `classes_` attribute to fit the likelihood."
                ),
                logger,
            )

        classes = model.classes_
        if type(classes) is not list:
            classes = [classes]
        self._classes = classes

        unqiue_classes = {label for classes in classes for label in classes}
        self._parameter_names = [
            f"probability_class_{int(label)}" for label in unqiue_classes
        ]
        return self

    def _estimator_predict(
        self,
        model,
        x: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        # TODO support multi-output with inequal number of classes
        model_output = np.array(model.model.predict_proba(x, **kwargs))

        output_dim = len(model_output.shape)

        # univariate & single-chunk output
        if output_dim <= 2:
            # embedding well shaped 2D output into 3D
            model_output = np.expand_dims(model_output, axis=0)
        # shape (n_components * output_chunk_length, n_series * n_samples, n_classes)
        return model_output

    def sample(self, model_output: np.ndarray) -> np.ndarray:
        """
        Samples a prediction from the likelihood distribution and the predicted parameters.
        """
        # model_output is a 3D array of shape (n_components * output_chunk_length, n_series * n_samples, n_classes)

        n_entries, n_samples, n_params = model_output.shape

        preds = np.empty((model_output.shape[0], model_output.shape[1]), dtype=int)

        for i in range(model_output.shape[0]):
            preds[i] = self._rng.choice(
                self._classes[i],
                size=(model_output.shape[1]),
                p=model_output[i],
                replace=True,
            )
        return preds.reshape(n_samples, self._n_outputs, -1)

    def predict_likelihood_parameters(self, model_output: np.ndarray) -> np.ndarray:
        """
        Returns the distribution parameters as a array, extracted from the raw model outputs.
        """
        n_samples = model_output.shape[1]

        # reshape to (n_series * n_samples, output_chunk_length, n_components)
        params_reshaped = model_output.transpose(1, 0, 2).reshape(
            n_samples, self._n_outputs, -1
        )
        return params_reshaped

    def _get_median_prediction(self, model_output: np.ndarray) -> np.ndarray:
        """
        Gets the median prediction per component extracted from the model output.
        """
        # model_output is a 3D array of shape (n_components * output_chunk_length, n_series * n_samples, n_classes)
        preds = np.empty((model_output.shape[0], model_output.shape[1]), dtype=int)

        # get the class with the highest probability
        indices = np.argmax(model_output, axis=2)
        for i in range(indices.shape[0]):
            preds[i] = self._classes[i][indices[i]]

        # reshape to (n_series * n_samples, output_chunk_length, n_components)
        return preds.reshape(model_output.shape[1], self._n_outputs, -1)


def _get_categorical_likelihood(
    likelihood: Optional[str],
    n_outputs: int,
    random_state: Optional[int],
) -> Optional[SKLearnLikelihood]:
    """Get the `Likelihood` object for `CategoricalModel`.

    Parameters
    ----------
    likelihood
        The likelihood name. Must be one of ('classprobability').
    n_outputs
        The number of predicted outputs per model call. `1` if `multi_models=False`, otherwise
        `output_chunk_length`.
    random_state
        Optionally, control the randomness of the sampling.
    """
    if likelihood is None:
        return None
    elif likelihood == LikelihoodType.ClassProbability.value:
        return ClassProbabilityLikelihood(
            n_outputs=n_outputs, random_state=random_state
        )
    else:
        raise_log(
            ValueError(
                f"Invalid `likelihood='{likelihood}'`. Must be one of ('{LikelihoodType.ClassProbability.value}')"
            ),
            logger=logger,
        )
