"""
Standard Regression model
-------------------------
"""

from .regression_model import RegressionModel
from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log
from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = get_logger(__name__)


class StandardRegressionModel(RegressionModel):

    def __init__(self,
                 train_n_points: int,
                 model=LinearRegression(n_jobs=-1, fit_intercept=False)):
        """
        Simple wrapper for regression models implementing a fit() predict() functions models
        (e.g., scikit-learn regression).

        Parameters
        ----------
        train_n_points
            The number of most recent points from the training (features and target) time series that
            will be used to train the regression model. If this is `None`, or if the provided training series
            contain fewer points, the maximum possible number of time steps will be used for training.
        model
            A regression model that implements `fit()` and `predict()` methods.
            Default: `sklearn.linear_model.LinearRegression(n_jobs=-1, fit_intercept=False)`
        """

        super(StandardRegressionModel, self).__init__()
        if (not callable(getattr(model, "fit", None))):
            raise_log(Exception('Provided model object must have a fit() method', logger))
        if (not callable(getattr(model, "predict", None))):
            raise_log(Exception('Provided model object must have a predict() method', logger))

        self.train_n_points = train_n_points
        self.model = model

    @staticmethod
    def _get_features_matrix_from_series(features: List[TimeSeries]):
        return np.concatenate([s.values() for s in features], axis=1)  # (n_samples x n_features)

    def fit(self,
            train_features: List[TimeSeries],
            train_target: TimeSeries):

        if self.train_n_points is None:
            train_n_points = min([len(s) for s in train_features] + [len(train_target)])
        else:
            train_n_points = self.train_n_points

        # Get (at most) the last [train_n_points] of each series
        last_train_ts = train_features[0].end_time()
        last_n_points_features = [s.slice_n_points_before(last_train_ts, train_n_points) for s in train_features]
        last_n_points_target = train_target.slice_n_points_before(last_train_ts, train_n_points)

        super().fit(last_n_points_features, last_n_points_target)

        self.model.fit(self._get_features_matrix_from_series(last_n_points_features),
                       last_n_points_target.values())

    def predict(self, features: List[TimeSeries]):
        super().predict(features)
        y = self.model.predict(self._get_features_matrix_from_series(features))
        return TimeSeries(pd.DataFrame(y, index=features[0].time_index()))
