"""
Multi-Output Models for SKLearnModel
------------------------------------
"""

import inspect
from collections import OrderedDict

from sklearn.base import is_classifier
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

from darts.logging import raise_log
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
                    {
                        self.eval_samples_name: eval_samples[i],
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


def _fit_quantile_estimator(estimator, X, y, fit_params):
    """Fit a single quantile estimator. Module-level function for joblib pickling."""
    # Strip verbose if the estimator (or its base estimator inside MultiOutputMixin)
    # does not support it — mirroring MultiOutputMixin behaviour.
    if fit_params.get("verbose") is not None:
        base = (
            estimator.estimator
            if isinstance(estimator, MultiOutputMixin)
            else estimator
        )
        if "verbose" not in inspect.signature(base.fit).parameters:
            fit_params = {k: v for k, v in fit_params.items() if k != "verbose"}
    estimator.fit(X, y, **fit_params)
    return estimator


class MultiQuantileRegressor:
    """Meta-estimator that fits one estimator per quantile level.

    Each sub-estimator may itself be a :class:`MultiOutputRegressor` when the
    task requires multi-output support (multivariate or ``multi_models``).
    Quantile fits can run in parallel via ``n_jobs``.

    Parameters
    ----------
    quantile_estimators : OrderedDict[float, estimator]
        Mapping from quantile level to an (unfitted) sklearn-compatible estimator.
        Each estimator should already be wrapped in ``MultiOutputRegressor`` if
        multi-output handling is needed.
    n_jobs : int or None, default=None
        Number of parallel jobs for fitting across quantiles. ``None`` or ``1``
        means sequential fitting. When set > 1, inner ``MultiOutputRegressor``
        instances should use ``n_jobs=1`` to avoid CPU oversubscription.
    """

    def __init__(
        self,
        quantile_estimators: OrderedDict,
        n_jobs: int | None = None,
    ):
        self.quantile_estimators = quantile_estimators
        self.n_jobs = n_jobs
        self.estimators_: OrderedDict = quantile_estimators

    def fit(self, X, y, **fit_params):
        """Fit every quantile estimator on the same training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        **fit_params
            Forwarded to each sub-estimator's ``fit()`` (e.g. ``sample_weight``,
            ``verbose``, validation-set kwargs).

        Returns
        -------
        self
        """
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_quantile_estimator)(est, X, y, fit_params)
            for est in self.estimators_.values()
        )
        self.estimators_ = OrderedDict(zip(self.quantile_estimators.keys(), results))
        return self

    def predict(self, X, **kwargs):
        """Predict using the median quantile estimator (convenience only)."""
        median_est = next(iter(self.estimators_.values()))
        return median_est.predict(X, **kwargs)

    @property
    def uses_multioutput(self) -> bool:
        """Whether the sub-estimators are wrapped in ``MultiOutputMixin``."""
        first = next(iter(self.estimators_.values()))
        return isinstance(first, MultiOutputMixin)

    @property
    def supports_sample_weight(self) -> bool:
        """Whether the underlying estimators support sample weights."""
        first = next(iter(self.estimators_.values()))
        if isinstance(first, MultiOutputMixin):
            return first.supports_sample_weight
        return has_fit_parameter(first, "sample_weight")

    def __str__(self):
        n_q = len(self.estimators_)
        first = next(iter(self.estimators_.values()))
        return f"MultiQuantileRegressor(n_quantiles={n_q}, estimator={first})"

    def __repr__(self):
        return self.__str__()


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
