"""
Multi-Output Models for SKLearnModel
------------------------------------
"""

import inspect

import numpy as np
from sklearn.base import is_classifier
from sklearn.multioutput import MultiOutputClassifier as sk_MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor as sk_MultiOutputRegressor
from sklearn.multioutput import _fit_estimator
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_method_params,
    check_is_fitted,
    has_fit_parameter,
    validate_data,
)

from darts.logging import raise_log
from darts.utils.output_chunk import OutputChunkStrategy
from darts.utils.utils import ModelType


class MultiOutputMixin:
    """
    Mixin for :class:`sklearn.utils.multioutput._MultiOutputEstimator` with a modified ``fit()`` method that also slices
    validation data correctly. The validation data has to be passed as parameter ``eval_set`` in ``**fit_params``.

    When an :class:`OutputChunkStrategy` is passed, it controls how outputs are grouped, how features
    are materialized per horizon, and the order in which estimators are fitted and predicted.
    """

    def __init__(
        self,
        estimator,
        eval_set_name: str | tuple[str, str] | None = None,
        eval_weight_name: str | None = None,
        strategy: OutputChunkStrategy | None = None,
        **kwargs,
    ):
        super().__init__(estimator=estimator, **kwargs)
        # according to sklearn, set only attributes in `__init__` that are known before fitting;
        # all other params at fitting time must have the suffix `"_"`
        self.eval_set_name = eval_set_name
        self.eval_weight_name = eval_weight_name
        # dedicated X and y eval parameters
        self.eval_samples_name, self.eval_labels_name = None, None
        if isinstance(self.eval_set_name, tuple):
            self.eval_samples_name, self.eval_labels_name = self.eval_set_name
        self.strategy = strategy

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

        if not hasattr(self.estimator, "fit"):
            raise_log(
                ValueError("The base estimator should implement a fit method"),
            )
        y = validate_data(self.estimator, X="no_validation", y=y, multi_output=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise_log(
                ValueError(
                    "`y` must have at least two dimensions for multi-output but has only one."
                ),
            )
        if sample_weight is not None and (
            sample_weight.ndim == 1 or sample_weight.shape[1] != y.shape[1]
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

        strategy, n_components, fit_groups = self._resolve_strategy(y.shape[1])

        def fit_params_for_output(output_idx: int) -> dict:
            params = dict(fit_params_validated)
            if eval_set is not None:
                eval_entry = eval_set[output_idx]
                # some estimators (e.g. CatBoost) receive ready-made evaluation
                # objects instead of ``(samples, labels)`` pairs.
                if isinstance(eval_entry, tuple | list) and len(eval_entry) == 2:
                    eval_samples_i, eval_labels_i = eval_entry
                    eval_samples_i = self._features_for_output(
                        eval_samples_i, output_idx, strategy, n_components
                    )
                    eval_entry = (eval_samples_i, eval_labels_i)
                params[self.eval_set_name] = [eval_entry]
            if eval_samples is not None and eval_labels is not None:
                params[self.eval_samples_name] = self._features_for_output(
                    eval_samples[output_idx],
                    output_idx,
                    strategy,
                    n_components,
                )
                params[self.eval_labels_name] = eval_labels[output_idx]
            if eval_weight is not None:
                params[self.eval_weight_name] = [eval_weight[output_idx]]
            return params

        self.estimators_ = [None] * y.shape[1]
        for group in fit_groups:
            fitted_estimators = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator,
                    self._features_for_output(X, output_idx, strategy, n_components),
                    y[:, output_idx],
                    sample_weight=sample_weight[:, output_idx]
                    if sample_weight is not None
                    else None,
                    **fit_params_for_output(output_idx),
                )
                for output_idx in group
            )
            for output_idx, estimator in zip(group, fitted_estimators):
                self.estimators_[output_idx] = estimator

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def _resolve_strategy(
        self, n_outputs: int
    ) -> tuple[OutputChunkStrategy | None, int | None, list[list[int]]]:
        strategy = self.strategy
        if strategy is not None and not isinstance(strategy, OutputChunkStrategy):
            raise_log(
                ValueError(
                    "`strategy` must be an `OutputChunkStrategy` instance or None."
                ),
            )
        self.strategy_ = strategy
        if strategy is None:
            self.n_components_ = None
            return None, None, [list(range(n_outputs))]

        if n_outputs % strategy.n_horizons != 0:
            raise_log(
                ValueError(
                    f"The number of outputs ({n_outputs}) must be divisible by the "
                    f"number of horizons ({strategy.n_horizons})."
                ),
            )
        n_components = n_outputs // strategy.n_horizons
        self.n_components_ = n_components
        return strategy, n_components, strategy.fit_order(n_outputs, n_components)

    @staticmethod
    def _features_for_output(
        features,
        output_idx: int,
        strategy: OutputChunkStrategy | None,
        n_components: int | None,
    ):
        if strategy is None:
            return features
        horizon = strategy.horizon_of(output_idx, n_components)
        horizon_fn = getattr(features, "horizon", None)
        return horizon_fn(horizon) if callable(horizon_fn) else features

    def _predict(self, X):
        """Predict multi-output targets using the estimator of each horizon."""
        check_is_fitted(self)
        if not hasattr(self.estimators_[0], "predict"):
            raise_log(
                ValueError("The base estimator should implement a predict method"),
            )

        strategy = getattr(self, "strategy_", None)
        if strategy is None:
            return super().predict(X)
        return self._predict_per_output(X, method="predict", stack=True)

    def _predict_per_output(self, X, *, method: str, stack: bool):
        strategy = self.strategy_
        n_components = self.n_components_
        n_outputs = len(self.estimators_)
        results = [None] * n_outputs
        for group in strategy.predict_order(n_outputs, n_components):
            group_results = Parallel(n_jobs=self.n_jobs)(
                delayed(getattr(self.estimators_[output_idx], method))(
                    self._features_for_output(X, output_idx, strategy, n_components)
                )
                for output_idx in group
            )
            for output_idx, result in zip(group, group_results):
                results[output_idx] = result
        return np.asarray(results).T if stack else results

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

    def predict(self, X):
        """Predict multi-output targets using the estimator of each horizon."""
        return self._predict(X)


class MultiOutputClassifier(MultiOutputMixin, sk_MultiOutputClassifier):
    """
    :class:`sklearn.utils.multioutput.MultiOutputClassifier` with a modified ``fit()`` method that also slices
    validation data correctly. The validation data has to be passed as parameter ``eval_set`` in ``**fit_params``.
    """

    def fit(self, X, y, sample_weight=None, **fit_params):
        super().fit(X=X, y=y, sample_weight=sample_weight, **fit_params)
        self.classes_ = [estimator.classes_ for estimator in self.estimators_]
        return self

    def predict(self, X):
        """Predict multi-output targets using the estimator of each horizon."""
        return self._predict(X)

    @available_if(lambda self: self._check_predict_proba())
    def predict_proba(self, X):
        """Return prediction probabilities for each output, using the correct horizon features."""
        check_is_fitted(self)
        strategy = getattr(self, "strategy_", None)
        if strategy is None:
            return super().predict_proba(X)
        return self._predict_per_output(X, method="predict_proba", stack=False)


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
