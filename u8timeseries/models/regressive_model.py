from abc import ABC, abstractmethod
from ..timeseries import TimeSeries
from typing import List


class RegressiveModel(ABC):
    """
    This is a base class for various implementations of multi-variate models - models predicting time series
    from one or several time series. It also allows to do ensembling.
    """

    @abstractmethod
    def __init__(self):
        # Stores training date information:
        self.train_features: List[TimeSeries] = None
        self.target_series: TimeSeries = None

        # state
        self.fit_called = False

    @abstractmethod
    def fit(self, train_features: List[TimeSeries], target_series: TimeSeries) -> None:
        assert len(train_features) > 0, 'Need at least one feature series'
        assert all([s.has_same_time_as(target_series) for s in train_features]), 'All provided time series must ' \
                                                                                 'have the same time index'
        self.train_features = train_features
        self.target_series = target_series
        self.fit_called = True

    @abstractmethod
    def predict(self, features: List[TimeSeries]) -> TimeSeries:
        """
        :return: A TimeSeries containing the prediction obtained from [features], of same length as [features]
        """
        assert self.fit_called, 'fit() must be called before predict()'
        assert len(features) == len(self.train_features), 'Provided features must have same dimensionality as ' \
                                                          'training features. There were {} training features and ' \
                                                          'the function has been called with {} features' \
                                                          .format(len(self.train_features), len(features))
