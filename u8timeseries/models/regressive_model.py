"""
Implementation of an Regressive model.
--------------------------------------
"""

from abc import ABC, abstractmethod
from ..timeseries import TimeSeries
from ..custom_logging import raise_if_not, get_logger, raise_log
from typing import List

logger = get_logger(__name__)

class RegressiveModel(ABC):
    """
    This is a base class for various implementations of multi-variate models - models predicting time series
    from one or several time series. It also allows to do ensembling.

    TODO: Extend this to a "DynamicRegressiveModel" class, which acts on List[List[TimeSeries]].
    TODO: The first List[] would contain time-sliding lists of time series, letting the model
    TODO: be able to learn how to change weights over time. When len() of outer List[] is 0 it's a particular case
    """

    @abstractmethod
    def __init__(self):
        # Stores training date information:
        self.train_features: List[TimeSeries] = None
        self.train_target: TimeSeries = None

        # state
        self._fit_called = False

    @abstractmethod
    def fit(self, train_features: List[TimeSeries], train_target: TimeSeries) -> None:
        raise_if_not(len(train_features) > 0, 'Need at least one feature series', logger)
        raise_if_not(all([s.has_same_time_as(train_target) for s in train_features]), 'All provided time series must ' \
                                                                                'have the same time index', logger)
        self.train_features = train_features
        self.train_target = train_target
        self._fit_called = True

    @abstractmethod
    def predict(self, features: List[TimeSeries]) -> TimeSeries:
        """
        :return: A TimeSeries containing the prediction obtained from [features], of same length as [features]
        """
        if (not self._fit_called):
            raise_log(Exception('fit() must be called before predict()'), logger)
        raise_if_not(len(features) == len(self.train_features), 'Provided features must have same dimensionality as ' \
                                                          'training features. There were {} training features and ' \
                                                          'the function has been called with {} features' \
                                                          .format(len(self.train_features), len(features)), logger)

    def residuals(self) -> TimeSeries:
        """
        :return: a time series of residuals (absolute errors of the model on the training set)
        """
        if (not self._fit_called):
            raise_log(Exception('fit() must be called before predict()'), logger)

        train_pred = self.predict(self.train_features)
        return abs(train_pred - self.train_target)
