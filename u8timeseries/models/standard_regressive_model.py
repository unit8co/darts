from .regressive_model import RegressiveModel
from ..timeseries import TimeSeries
from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class StandardRegressiveModel(RegressiveModel):
    """
    Simple regression based on other fit() predict() models (e.g., from sklearn)
    """
    def __init__(self, model=LinearRegression(n_jobs=-1, fit_intercept=False)):
        super(StandardRegressiveModel, self).__init__()
        assert callable(getattr(model, "fit", None)), 'Provided model object must have a fit() method'
        assert callable(getattr(model, "predict", None)), 'Provided model object must have a predict() method'

        self.model = model

    @staticmethod
    def _get_features_matrix_from_series(features: List[TimeSeries]):
        return np.array([s.values() for s in features]).T  # (n_samples x n_features)

    def fit(self, train_features: List[TimeSeries], train_target: TimeSeries):
        super().fit(train_features, train_target)
        self.model.fit(self._get_features_matrix_from_series(train_features), train_target.values())

    def predict(self, features: List[TimeSeries]):
        super().predict(features)
        y = self.model.predict(self._get_features_matrix_from_series(features))
        return TimeSeries(pd.Series(y, index=features[0].time_index()))
