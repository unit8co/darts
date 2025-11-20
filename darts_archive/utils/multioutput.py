"""
Multi-Output Models for `SKLearnModel`
--------------------------------------
"""

import inspect
from typing import Optional

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
