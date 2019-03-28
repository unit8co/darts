from .regressive_model import RegressiveModel
from ..timeseries import TimeSeries
from typing import List


class SequentialAggregation(RegressiveModel):
    """
    TODO
    https://s3.amazonaws.com/academia.edu.documents/45700981/Forecasting_electricity_consumption_by_a20160517-4457-1a8m354.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1552341840&Signature=x7vegA6suvqbvDuyLiaK%2BKCt23s%3D&response-content-disposition=inline%3B%20filename%3DForecasting_electricity_consumption_by_a.pdf

    https://link.springer.com/article/10.1007/s10994-012-5314-7

    https://hal.inria.fr/tel-01697133v2/document (Section 1.2)
    """
    def __init__(self):
        super().__init__()

    def fit(self, train_features: List[TimeSeries], train_target: TimeSeries):
        super().fit(train_features, train_target)

    def predict(self, features: List[TimeSeries]):
        super().predict(features)