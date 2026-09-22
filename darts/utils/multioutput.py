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
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_method_params,
    check_is_fitted,
    has_fit_parameter,
    validate_data,
)

from darts.logging import raise_log
from darts.utils.data.tabularization.stepwise import StepwiseLaggedFeatures
from darts.utils.utils import ModelType


class MultiOutputMixin:
    """
    Mixin for :class:`sklearn.utils.multioutput._MultiOutputEstimator` with a modified ``fit()`` method that also slices
    validation data correctly. The validation data has to be passed as parameter ``eval_set`` in ``**fit_params``.
    """

    def __init__(
        self,
        estimator,
        eval_set_name: str | tuple[str, str] | None = None,
        eval_weight_name: str | None = None,
        output_chunk_length: int | None = None,
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
        self.output_chunk_length = output_chunk_length

    def _n_estimators_per_horizon(self, n_outputs: int, n_horizons: int) -> int:
        """
        The number of estimators (i.e. target components) per horizon, when each horizon is trained on its own
        features array.
        """
        if n_horizons < 1 or n_outputs % n_horizons:
            raise_log(
                ValueError(
                    f"The number of outputs (`{n_outputs}`) must be a multiple of the number of horizons "
                    f"(`{n_horizons}`) of the step-wise features."
                ),
            )
        if (
            self.output_chunk_length is not None
            and self.output_chunk_length != n_horizons
        ):
            raise_log(
                ValueError(
                    f"The step-wise features hold `{n_horizons}` horizons but the model was configured with "
                    f"`output_chunk_length={self.output_chunk_length}`."
                ),
            )
        return n_outputs // n_horizons

    @staticmethod
    def _horizon_samples(X, horizon: int | None):
        """Materializes the features array of `horizon` when `X` holds one array per horizon."""
        if horizon is None or not isinstance(X, StepwiseLaggedFeatures):
            return X
        return X.horizon(horizon)

    @classmethod
    def _horizon_eval_set(cls, eval_set, horizon: int | None):
        """Same as `_horizon_samples()` for a validation set entry, i.e. a `(samples, labels)` tuple."""
        if horizon is None or not isinstance(eval_set, tuple | list) or not eval_set:
            return eval_set
        return (cls._horizon_samples(eval_set[0], horizon), *eval_set[1:])

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

        # with step-wise future covariates lags, each horizon is fit on its own features array; the
        # observations (and hence the fit params) are shared by all of them
        if isinstance(X, StepwiseLaggedFeatures):
            n_estimators_per_horizon = self._n_estimators_per_horizon(
                n_outputs=y.shape[1], n_horizons=X.n_horizons
            )
        else:
            n_estimators_per_horizon = None

        fit_params_validated = _check_method_params(
            X.base if isinstance(X, StepwiseLaggedFeatures) else X, fit_params
        )
        eval_set, eval_samples, eval_labels = None, None, None
        if self.eval_set_name is not None and self.eval_labels_name is not None:
            eval_samples = fit_params_validated.pop(self.eval_samples_name, None)
            eval_labels = fit_params_validated.pop(self.eval_labels_name, None)
        else:
            eval_set = fit_params_validated.pop(self.eval_set_name, None)
        eval_weight = fit_params_validated.pop(self.eval_weight_name, None)

        def horizon_of(output_idx: int) -> int | None:
            """The horizon that estimator `output_idx` is fit on (`None` when features are shared)."""
            if n_estimators_per_horizon is None:
                return None
            return output_idx // n_estimators_per_horizon

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_estimator)(
                self.estimator,
                self._horizon_samples(X, horizon_of(i)),
                y[:, i],
                sample_weight=sample_weight[:, i]
                if sample_weight is not None
                else None,
                **(
                    {
                        self.eval_set_name: [
                            self._horizon_eval_set(eval_set[i], horizon_of(i))
                        ]
                    }
                    if eval_set is not None
                    else {}
                ),
                **(
                    {
                        self.eval_samples_name: self._horizon_samples(
                            eval_samples[i], horizon_of(i)
                        ),
                        self.eval_labels_name: eval_labels[i],
                    }
                    if eval_samples is not None and eval_labels is not None
                    else {}
                ),
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

    def _predict_per_horizon(self, X, method: str) -> list:
        """
        Calls `method` of each estimator on the features array of the horizon it was fit on, materializing each
        horizon only once. Returns one entry per output, in the `[hrz0_comp0, ..., hrz1_comp0, ...]` order of
        `estimators_`.
        """
        check_is_fitted(self)
        n_per_horizon = self._n_estimators_per_horizon(
            n_outputs=len(self.estimators_), n_horizons=X.n_horizons
        )
        results = []
        for horizon in range(X.n_horizons):
            X_horizon = X.horizon(horizon)
            estimators = self.estimators_[
                horizon * n_per_horizon : (horizon + 1) * n_per_horizon
            ]
            results.extend(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(getattr(estimator, method))(X_horizon)
                    for estimator in estimators
                )
            )
        return results

    def predict(self, X):
        """Predicts multi-output targets, routing each horizon to the estimators fit on its features array."""
        if not isinstance(X, StepwiseLaggedFeatures):
            return super().predict(X)
        return np.asarray(self._predict_per_horizon(X, "predict")).T

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

    def predict_proba(self, X):
        """Predicts class probabilities, routing each horizon to the estimators fit on its features array."""
        if not isinstance(X, StepwiseLaggedFeatures):
            return super().predict_proba(X)
        return self._predict_per_horizon(X, "predict_proba")


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
