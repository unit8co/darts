from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from darts.utils.multioutput import MultiOutputRegressor


class TestMultiOutputParameterSpecific:
    def test_univariate_target_with_multiple_param_estimators(self):
        mor = MultiOutputRegressor(
            estimator=[LinearRegression(), LinearRegression()],
            n_jobs=1,
        )
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = 3.0 * X[:, 0]
        mor.fit(X, y)
        assert mor.is_parameter_specific
        assert len(mor.estimators_) == 2
        pred = mor.predict(X)
        assert pred.shape == (6, 1, 2)

    def test_multivariate_predict_shape_and_values(self):
        mor = MultiOutputRegressor(
            estimator=[LinearRegression(), LinearRegression(), LinearRegression()],
            n_jobs=1,
        )
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.column_stack([X[:, 0] + 1.0, X[:, 0] + 2.0])
        mor.fit(X, y)
        pred = mor.predict(X)
        assert pred.shape == (10, 2, 3)
        # estimators are interleaved: target major, parameter minor
        for t in range(2):
            for q in range(3):
                idx = t * 3 + q
                np.testing.assert_allclose(
                    pred[:, t, q], mor.estimators_[idx].predict(X)
                )

    def test_non_parameter_specific_still_2d(self):
        mor = MultiOutputRegressor(estimator=LinearRegression(), n_jobs=1)
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.column_stack([X[:, 0], X[:, 1]])
        mor.fit(X, y)
        assert not mor.is_parameter_specific
        pred = mor.predict(X)
        assert pred.ndim == 2
        assert pred.shape == (6, 2)

    def test_invalid_model(self):
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.column_stack([X[:, 0], X[:, 1]])

        mor = MultiOutputRegressor(estimator=Mock(spec=[]))
        with pytest.raises(ValueError, match="implement a fit method"):
            mor.fit(X, y)

        class _EstimatorWithoutPredict(BaseEstimator):
            def fit(self, X, y):
                self.n_features_in_ = X.shape[1]
                return self

        mor = MultiOutputRegressor(estimator=_EstimatorWithoutPredict(), n_jobs=1)
        mor.fit(X, y)
        with pytest.raises(ValueError, match="implement a predict method"):
            mor.predict(X)

    def test_invalid_y1d(self):
        mor = MultiOutputRegressor(estimator=LinearRegression(), n_jobs=1)
        X = np.arange(12, dtype=float).reshape(6, 2)
        y_1d = X[:, 0]
        with pytest.raises(ValueError, match="must have at least two dimensions"):
            mor.fit(X, y_1d)

        # but allowed with parameter-specific models
        mor = MultiOutputRegressor(
            estimator=[LinearRegression(), LinearRegression()], n_jobs=1
        )
        mor.fit(X, y_1d)

    def test_invalid_sample_weight(self):
        mor = MultiOutputRegressor(estimator=LinearRegression(), n_jobs=1)
        X = np.arange(12, dtype=float).reshape(6, 2)
        y = np.column_stack([X[:, 0], X[:, 1]])
        with pytest.raises(
            ValueError, match="`sample_weight` must have the same dimensions as `y`"
        ):
            mor.fit(X, y, sample_weight=y[:, :1])
