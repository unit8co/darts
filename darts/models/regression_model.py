"""
Regression Model Base Class
---------------------------

A regression model predicts values for a time series :math:`Y_t` as a function
of :math:`N` "features" time series :math:`X^i_t`:

.. math:: Y_t = f(X^1_t, ..., X^N_t),

where :math:`t` denotes the time step. Here, the function :math:`f()` is not necessarily linear.
"""

from abc import ABC, abstractmethod
from ..timeseries import TimeSeries
from ..logging import raise_if_not, get_logger, raise_log
from typing import List

logger = get_logger(__name__)


# TODO: Extend this to a "DynamicRegressiveModel" class, which acts on List[List[TimeSeries]].
# TODO: The first List[] would contain time-sliding lists of time series, letting the model
# TODO: be able to learn how to change weights over time. When len() of outer List[] is 0 it's a particular case
class RegressionModel(ABC):
    @abstractmethod
    def __init__(self):
        """ Regression Model.

            This is the base class for all regression models.
        """

        # Stores training date information:
        self.train_features: List[TimeSeries] = None
        self.train_target: TimeSeries = None

        # state
        self._fit_called = False

    @abstractmethod
    def fit(self, train_features: List[TimeSeries], train_target: TimeSeries) -> None:
        """ Fits/trains the model using the provided list of features time series and the target time series.

        Parameters
        ----------
        train_features
            A list of features time series, all of the same length as the target series
        train_target
            A target time series, of the same length as the features series
        """

        raise_if_not(len(train_features) > 0, 'Need at least one feature series', logger)
        raise_if_not(all([s.has_same_time_as(train_target) for s in train_features]),
                     'All provided time series must have the same time index', logger)
        self.train_features = train_features
        self.train_target = train_target
        self._fit_called = True

    @abstractmethod
    def predict(self, features: List[TimeSeries]) -> TimeSeries:
        """ Predicts values of the target time series, given a list of features time series

        Parameters
        ----------
        features
            The list of features time series, of the same length

        Returns
        -------
        TimeSeries
            A series containing the predicted targets, of the same length as the features series
        """

        if (not self._fit_called):
            raise_log(Exception('fit() must be called before predict()'), logger)

        length_ok = len(features) == len(self.train_features)
        dimensions_ok = all(features[i].width == self.train_features[i].width for i in range(len(features)))
        raise_if_not(length_ok and dimensions_ok,
                     'The number and dimensionalities of all given features must correspond to those used for'
                     ' training.', logger)

    def residuals(self) -> TimeSeries:
        """ Computes the time series of residuals of this model on the training time series

        The residuals are computed as

        .. math:: z_t := y_t - \\hat{y}_t,

        where :math:`y_t` is the actual target time series over the training set,
        and :math:`\\hat{y}_t` is the time series of predicted targets, over the training set.

        Returns
        -------
        TimeSeries
            The time series containing the residuals
        """

        if (not self._fit_called):
            raise_log(Exception('fit() must be called before predict()'), logger)

        train_pred = self.predict(self.train_features)
        return self.train_target - train_pred
