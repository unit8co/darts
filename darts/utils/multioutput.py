"""
Multi-Output Models for SKLearnModel
------------------------------------
"""

import inspect
from typing import Optional

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.multioutput import MultiOutputClassifier as sk_MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor as sk_MultiOutputRegressor
from sklearn.multioutput import _fit_estimator
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_method_params,
    has_fit_parameter,
    validate_data,
)

from darts.logging import get_logger, raise_log
from darts.utils.utils import ModelType

logger = get_logger(__name__)


class MultiOutputMixin:
    """
    Mixin for :class:`sklearn.utils.multioutput._MultiOutputEstimator` with a modified ``fit()`` method that also slices
    validation data correctly. The validation data has to be passed as parameter ``eval_set`` in ``**fit_params``.
    """

    def __init__(
        self,
        estimator,
        eval_set_name: Optional[str] = None,
        eval_weight_name: Optional[str] = None,
        output_chunk_length: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(estimator=estimator, **kwargs)
        # according to sklearn, set only attributes in `__init__` that are known before fitting;
        # all other params at fitting time must have the suffix `"_"`
        self.eval_set_name = eval_set_name
        self.eval_weight_name = eval_weight_name
        self.output_chunk_length = output_chunk_length

    def _validate_fit_inputs(self, y, sample_weight):
        """Validate inputs for fitting. Shared by MultiOutputMixin and RecurrentMultiOutputMixin.

        Parameters
        ----------
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets.
        sample_weight : array-like of shape (n_samples, n_outputs), default=None
            Sample weights.

        Returns
        -------
        y : validated y array
        """
        if not hasattr(self.estimator, "fit"):
            raise_log(
                ValueError("The base estimator should implement a fit method"),
                logger=logger,
            )
        y = validate_data(self.estimator, X="no_validation", y=y, multi_output=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise_log(
                ValueError(
                    "`y` must have at least two dimensions for multi-output but has only one."
                ),
                logger=logger,
            )
        if sample_weight is not None and (
            sample_weight.ndim == 1 or sample_weight.shape[1] != y.shape[1]
        ):
            raise_log(
                ValueError("`sample_weight` must have the same dimensions as `y`."),
                logger=logger,
            )

        if sample_weight is not None and not self.supports_sample_weight:
            raise_log(
                ValueError("Underlying estimator does not support sample weights."),
                logger=logger,
            )

        return y

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the model to data, separately for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like of shape (n_samples, n_outputs), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        y = self._validate_fit_inputs(y, sample_weight)
        
        if (
            fit_params.get("verbose") is not None
            and "verbose" not in inspect.signature(self.estimator.fit).parameters
        ):
            fit_params.pop("verbose")

        fit_params_validated = _check_method_params(X, fit_params)
        eval_set = fit_params_validated.pop(self.eval_set_name, None)
        eval_weight = fit_params_validated.pop(self.eval_weight_name, None)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator,
                X,
                y[:, i],
                sample_weight=sample_weight[:, i]
                if sample_weight is not None
                else None,
                **({self.eval_set_name: [eval_set[i]]} if eval_set is not None else {}),
                **(
                    {self.eval_weight_name: [eval_weight[i]]}
                    if eval_weight is not None
                    else {}
                ),
                **fit_params_validated,
            )
            for i in range(y.shape[1])
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    @property
    def supports_sample_weight(self) -> bool:
        """
        Whether model supports sample weight for training.
        """
        return has_fit_parameter(self.estimator, "sample_weight")


class MultiOutputRegressor(MultiOutputMixin, sk_MultiOutputRegressor):
    """
    :class:`sklearn.utils.multioutput.MultiOutputRegressor` with a modified ``fit()`` method that also slices
    validation data correctly. The validation data has to be passed as parameter ``eval_set`` in ``**fit_params``.
    """


class MultiOutputClassifier(MultiOutputMixin, sk_MultiOutputClassifier):
    """
    :class:`sklearn.utils.multioutput.MultiOutputClassifier` with a modified ``fit()`` method that also slices
    validation data correctly. The validation data has to be passed as parameter ``eval_set`` in ``**fit_params``.
    """

    def fit(self, X, y, sample_weight=None, **fit_params):
        super().fit(X=X, y=y, sample_weight=sample_weight, **fit_params)
        self.classes_ = [estimator.classes_ for estimator in self.estimators_]
        return self


class RecurrentMultiOutputMixin(MultiOutputMixin):
    """
    Mixin for :class:`sklearn.utils.multioutput._MultiOutputEstimator` that implements direct-recursive multi-output
    prediction. Each output estimator is trained sequentially, using predictions from all previous estimators as
    additional features.

    This approach differs from :class:`MultiOutputRegressor` where all estimators are trained independently
    in parallel. Here, estimator i uses predictions from estimators 0, 1, ..., i-1 as additional input features,
    creating a chain of dependencies.
    """

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the model to data sequentially for each output variable, where each estimator
        uses predictions from previous estimators as additional features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like of shape (n_samples, n_outputs), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.

        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

            .. versionadded:: 0.23

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        y = self._validate_fit_inputs(y, sample_weight)

        fit_params_validated = _check_method_params(X, fit_params)
        eval_set = fit_params_validated.pop(self.eval_set_name, None)
        eval_weight = fit_params_validated.pop(self.eval_weight_name, None)

        self.estimators_ = []
        predictions = []  # Store predictions from previous estimators

        # Sequential training - each estimator depends on predictions from previous ones
        for i in range(y.shape[1]):
            # Clone the base estimator for this output
            estimator = clone(self.estimator)

            # Augment features with predictions from all previous estimators
            if predictions:
                X_augmented = np.concatenate([X] + predictions, axis=1)
            else:
                X_augmented = X

            # Prepare evaluation set with augmented features if provided
            # not 100% sure about this
            eval_set_augmented = None
            if eval_set is not None:
                eval_X, eval_y = eval_set[i]
                if predictions:
                    # Augment eval_X with predictions from already-trained estimators
                    eval_predictions = [
                        est.predict(eval_X).reshape(-1, 1) for est in self.estimators_
                    ]
                    eval_X_augmented = np.concatenate(
                        [eval_X] + eval_predictions, axis=1
                    )
                else:
                    eval_X_augmented = eval_X
                eval_set_augmented = [(eval_X_augmented, eval_y)]

            # Fit the estimator for this output
            estimator.fit(
                X_augmented,
                y[:, i],
                sample_weight=sample_weight[:, i]
                if sample_weight is not None
                else None,
                **(
                    {self.eval_set_name: eval_set_augmented}
                    if eval_set_augmented is not None
                    else {}
                ),
                **(
                    {self.eval_weight_name: [eval_weight[i]]}
                    if eval_weight is not None
                    else {}
                ),
                **fit_params_validated,
            )

            self.estimators_.append(estimator)

            # Generate predictions to use as features for the next estimator
            y_pred = estimator.predict(X_augmented).reshape(-1, 1)
            predictions.append(y_pred)

        # Set sklearn-standard attributes from first estimator
        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def predict(self, X):
        """Predict multi-output variable sequentially, where each estimator uses predictions
        from previous estimators as additional features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : array of shape (n_samples, n_outputs)
            Multi-output targets predicted across multiple predictors.
        """
        predictions = []

        for estimator in self.estimators_:
            # Augment features with predictions from previous estimators
            if predictions:
                X_augmented = np.concatenate([X] + predictions, axis=1)
            else:
                X_augmented = X

            # Predict for this output
            y_pred = estimator.predict(X_augmented).reshape(-1, 1)
            predictions.append(y_pred)

        # Stack all predictions horizontally to form (n_samples, n_outputs)
        return np.concatenate(predictions, axis=1)


class RecurrentMultiOutputRegressor(RecurrentMultiOutputMixin, sk_MultiOutputRegressor):
    "wrapper class for recurrent multioutput regressor"


class RecurrentMultiOutputClassifier(
    RecurrentMultiOutputMixin, sk_MultiOutputRegressor
):
    "wrapper class for recurrent multioutput classifier"

    def fit(self, X, y, sample_weight=None, **fit_params):
        super().fit(X=X, y=y, sample_weight=sample_weight, **fit_params)
        self.classes_ = [estimator.classes_ for estimator in self.estimators_]
        return self


def get_multioutput_estimator_cls(model_type: ModelType) -> type[MultiOutputMixin]:
    if model_type == ModelType.FORECASTING_REGRESSOR:
        return MultiOutputRegressor
    elif model_type == ModelType.FORECASTING_CLASSIFIER:
        return MultiOutputClassifier
    else:
        raise_log(
            ValueError(
                "Model type must be one of `[ModelType.FORECASTING_REGRESSOR, ModelType.FORECASTING_CLASSIFIER]`. "
                f"Received: `{model_type}`."
            )
        )


def get_recurrent_multioutput_estimator_cls(
    model_type: ModelType,
) -> type[RecurrentMultiOutputMixin]:
    if model_type == ModelType.FORECASTING_REGRESSOR:
        return RecurrentMultiOutputRegressor
    elif model_type == ModelType.FORECASTING_CLASSIFIER:
        return RecurrentMultiOutputClassifier
    else:
        raise_log(
            ValueError(
                "Model type must be one of `[ModelType.FORECASTING_REGRESSOR, ModelType.FORECASTING_CLASSIFIER]`. "
                f"Received: `{model_type}`."
            )
        )
