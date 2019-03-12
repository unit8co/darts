from .regressive_model import RegressiveModel
from ..timeseries import TimeSeries
from typing import List


class StandardRegressiveModel(RegressiveModel):
    def __init__(self):
        pass

    def fit(self, train_features: List[TimeSeries]):