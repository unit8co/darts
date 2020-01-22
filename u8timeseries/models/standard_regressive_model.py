from .regressive_model import RegressiveModel
from ..timeseries import TimeSeries
from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class StandardRegressiveModel(RegressiveModel):

    def __init__(self, train_n_points, model=LinearRegression(n_jobs=-1, fit_intercept=False)):
        """
        Simple regression based on other fit() predict() models (e.g., from sklearn).

        :param train_n_points: The number of most recent points from the training time series that
                               will be used to train the regressive model. If the provided training series
                               contain fewer points, they will all be used for training.
        :param model: The actual regressive model. It must contain fit() and predict() methods.
        """
        super(StandardRegressiveModel, self).__init__()
        assert callable(getattr(model, "fit", None)), 'Provided model object must have a fit() method'
        assert callable(getattr(model, "predict", None)), 'Provided model object must have a predict() method'

        self.train_n_points = train_n_points
        self.model = model

    @staticmethod
    def _get_features_matrix_from_series(features: List[TimeSeries]):
        return np.array([s.values() for s in features]).T  # (n_samples x n_features)

    def fit(self, train_features: List[TimeSeries], train_target: TimeSeries):
        # Get (at most) the last [train_n_points] of each series
        last_train_ts = train_features[0].end_time()
        last_n_points_features = [s.slice_n_points_before(last_train_ts, self.train_n_points) for s in train_features]
        last_n_points_target = train_target.slice_n_points_before(last_train_ts, self.train_n_points)

        super().fit(last_n_points_features, last_n_points_target)

        self.model.fit(self._get_features_matrix_from_series(last_n_points_features), last_n_points_target.values())

    def predict(self, features: List[TimeSeries]):
        super().predict(features)
        y = self.model.predict(self._get_features_matrix_from_series(features))
        return TimeSeries(pd.Series(y, index=features[0].time_index()))
