"""
Multi-Output Models for SKLearnModel
------------------------------------
"""

import inspect
from typing import Any

import numpy as np
from sklearn.base import is_classifier
from sklearn.multioutput import MultiOutputClassifier as sk_MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor as sk_MultiOutputRegressor
from sklearn.multioutput import _fit_estimator
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_method_params,
    check_is_fitted,
    has_fit_parameter,
    validate_data,
)

from darts.logging import raise_log
from darts.utils.utils import ModelType


class MultiOutputMixin:
    """
    Mixin for :class:`sklearn.utils.multioutput._MultiOutputEstimator` with:

    - a modified ``fit()`` method that also slices validation data correctly. The validation data has to be passed as
      parameter ``eval_set`` in ``**fit_params``.
    - a modified ``predict()`` method which also accepts ``**kwargs`` that are passed along.
    """

    estimator: Any
    n_jobs: int | None

    def __init__(
        self,
        estimator,
        eval_set_name: str | tuple[str, str] | None = None,
        eval_weight_name: str | None = None,
        output_chunk_length: int | None = None,
        **kwargs,
    ):
        """Multi output / target mixin.

        This strategy consists of fitting one estimator per horizon, target component and optional quantile /
        parameter-specific model. It is a simple strategy for extending regressors that do not natively support
        multi-target regression.

        Parameters
        ----------
        estimator
            An estimator object or list of estimator objects, each implementing :term:`fit` and :term:`predict`.
            If a list (e.g. for multi-quantile regression), all estimators are expected to produce identical
            `y` / output shape based on the same input `X`.
        eval_set_name
            Optionally (if available), the name of the evaluation set parameter passed to :term:`fit`. If a tuple,
            gives the X and y evaluation set parameter names.
        eval_weight_name
            Optionally (if available), the name of the sample weights parameter passed to :term:`fit`.
        output_chunk_length
            The forecasting model's output chunk length.
        **kwargs
            Optional keyword arguments used to instantiate the SKLearn multioutput estimator.
        """
        if isinstance(estimator, list):
            # parameter-specific models (e.g. quantile regression), each producing one output per target in `y`;
            # results in `y.shape[1] * len(estimator)` fitted estimators
            param_estimators = estimator
        else:
            # one model producing one output per target in `y`;
            # results in `y.shape[1]` fitted estimators
            param_estimators = [estimator]
        super().__init__(estimator=param_estimators[0], **kwargs)
        # according to sklearn, set only attributes in `__init__` that are known before fitting;
        # all other params at fitting time must have the suffix `"_"`
        self.eval_set_name = eval_set_name
        self.eval_weight_name = eval_weight_name
        # dedicated X and y eval parameters
        self.eval_samples_name, self.eval_labels_name = None, None
        if isinstance(self.eval_set_name, tuple):
            self.eval_samples_name, self.eval_labels_name = self.eval_set_name
        self.output_chunk_length = output_chunk_length
        self.param_estimators = param_estimators

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the model to data, separately for each output variable.

        Parameters
        ----------
        X
            Array-like, sparse matrix of shape (n_samples, n_features).
            The input data.
        y
            Array-like, sparse matrix of shape (n_samples, n_outputs).
            Multi-output targets. An indicator matrix turns on multilabel estimation.
        sample_weight
            Array-like of shape (n_samples, n_outputs), default=None
            Sample weights. If `None`, then samples are equally weighted.
            Only supported if the underlying regressor supports sample
            weights.
        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.
        """
        if not hasattr(self.estimator, "fit"):
            raise_log(
                ValueError("The base estimator should implement a fit method"),
            )
        y = validate_data(self.estimator, X="no_validation", y=y, multi_output=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            if len(self.param_estimators) > 1:
                # multiple quantiles / parameter predictions for a single target
                y = y.reshape(-1, 1)
            else:
                raise_log(
                    ValueError(
                        "`y` must have at least two dimensions for multi-output but has only one."
                    ),
                )

        n_targets = y.shape[1]
        n_params = len(self.param_estimators)
        n_outputs = n_targets * n_params
        if sample_weight is not None and (
            sample_weight.ndim == 1 or sample_weight.shape[1] != n_targets
        ):
            raise_log(
                ValueError("`sample_weight` must have the same dimensions as `y`."),
            )

        if sample_weight is not None and not self.supports_sample_weight:
            raise_log(
                ValueError("Underlying estimator does not support sample weights."),
            )

        if (
            fit_params.get("verbose") is not None
            and "verbose" not in inspect.signature(self.estimator.fit).parameters
        ):
            fit_params.pop("verbose")

        fit_params_validated = _check_method_params(X, fit_params)
        eval_set, eval_samples, eval_labels = None, None, None
        if self.eval_set_name is not None and self.eval_labels_name is not None:
            eval_samples = fit_params_validated.pop(self.eval_samples_name, None)
            eval_labels = fit_params_validated.pop(self.eval_labels_name, None)
        else:
            eval_set = fit_params_validated.pop(self.eval_set_name, None)
        eval_weight = fit_params_validated.pop(self.eval_weight_name, None)

        def _estimator_idx(i: int) -> int:
            return i % n_params

        def _target_idx(i: int) -> int:
            return i // n_params

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.param_estimators[_estimator_idx(i)],
                X,
                y[:, _target_idx(i)],
                sample_weight=sample_weight[:, _target_idx(i)]
                if sample_weight is not None
                else None,
                **(
                    {self.eval_set_name: [eval_set[_target_idx(i)]]}
                    if eval_set is not None
                    else {}
                ),
                **(
                    {
                        self.eval_samples_name: eval_samples[_target_idx(i)],
                        self.eval_labels_name: eval_labels[_target_idx(i)],
                    }
                    if eval_samples is not None and eval_labels is not None
                    else {}
                ),
                **(
                    {self.eval_weight_name: [eval_weight[_target_idx(i)]]}
                    if eval_weight is not None
                    else {}
                ),
                **fit_params_validated,
            )
            for i in range(n_outputs)
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def predict(self, X, **kwargs):
        """Predict multi-output variable using model for each target variable.

        Parameters
        ----------
        X
            Array-like, sparse matrix of shape (n_samples, n_features).
            The input data.
        **kwargs
            Parameters passed to the ``estimator.predict`` method of each step.

        Returns
        -------
        y
            Array-like, sparse matrix of shape (n_samples, n_outputs) or
            (n_samples, n_outputs, n_params) if `estimator` was a list (of
            length `n_params`) of estimators at model creation.
            Multi-output targets predicted across multiple predictors.
        """
        check_is_fitted(self)
        if not hasattr(self.estimators_[0], "predict"):
            raise ValueError("The base estimator should implement a predict method")

        y = Parallel(n_jobs=self.n_jobs)(
            delayed(e.predict)(X, **kwargs) for e in self.estimators_
        )

        if len(self.param_estimators) == 1:
            # return shape (n series * n samples, output_chunk_length * n_targets)
            return np.asarray(y).T

        # return shape (n series * n samples, output_chunk_length * n_targets, n_parameters)
        n_params = len(self.param_estimators)
        n_targets = len(self.estimators_) // n_params
        return np.asarray(y).T.reshape(-1, n_targets, n_params)

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
