from joblib import Parallel
from sklearn.base import is_classifier
from sklearn.multioutput import MultiOutputRegressor as sk_MultiOutputRegressor
from sklearn.multioutput import _fit_estimator
from sklearn.utils.fixes import delayed
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_fit_params, has_fit_parameter


class MultiOutputRegressor(sk_MultiOutputRegressor):
    """
    :class:`sklearn.utils.multioutput.MultiOutputRegressor` with a modified ``fit()`` method that also slices
    validation data correctly. The validation data has to be passed as parameter ``eval_set`` in ``**fit_params``.
    """

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the model to data, separately for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.

        sample_weight : array-like of shape (n_samples,), default=None
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
            raise ValueError("The base estimator should implement a fit method")

        y = self._validate_data(X="no_validation", y=y, multi_output=True)

        if is_classifier(self):
            check_classification_targets(y)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        if sample_weight is not None and not has_fit_parameter(
            self.estimator, "sample_weight"
        ):
            raise ValueError("Underlying estimator does not support sample weights.")

        fit_params_validated = _check_fit_params(X, fit_params)

        if "eval_set" in fit_params_validated.keys():
            # with validation set
            eval_set = fit_params_validated.pop("eval_set")
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator,
                    X,
                    y[:, i],
                    sample_weight,
                    # eval set may be a list (for XGBRegressor), in which case we have to keep it as a list
                    eval_set=[(eval_set[0][0], eval_set[0][1][:, i])]
                    if isinstance(eval_set, list)
                    else (eval_set[0], eval_set[1][:, i]),
                    **fit_params_validated
                )
                for i in range(y.shape[1])
            )
        else:
            # without validation set
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator, X, y[:, i], sample_weight, **fit_params_validated
                )
                for i in range(y.shape[1])
            )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self
