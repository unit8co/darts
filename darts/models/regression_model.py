"""
Regression Model Base Class
---------------------------

A regression model predicts values for a time series :math:`Y_t` as a function
of :math:`N` "features" time series :math:`X^i_t`:

.. math:: Y_t = f(X^1_t, ..., X^N_t),

where :math:`t` denotes the time step. Here, the function :math:`f()` is not necessarily linear.
"""

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import List, Iterable, Union, Any

from ..timeseries import TimeSeries
from ..logging import raise_if_not, get_logger, raise_log
from ..utils import _build_tqdm_iterator, _with_sanity_checks, _get_timestamp_at_point, _backtest_general_checks

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

    def _backtest_sanity_checks(self, *args: Any, **kwargs: Any) -> None:
        """Sanity checks for the backtest function

        Parameters
        ----------
        args
            The args parameter(s) provided to the backtest function.
        kwargs
            The kwargs paramter(s) provided to the backtest function.

        Raises
        ------
        ValueError
            when a check on the parameter does not pass.
        """

        # parse args
        feature_series = args[0]
        target_series = args[1]

        raise_if_not(all([s.has_same_time_as(target_series) for s in feature_series]), 'All provided time series must '
                     'have the same time index', logger)

        _backtest_general_checks(target_series, kwargs)

    def _backtest_model_specific_sanity_checks(self, *args: Any, **kwargs: Any) -> None:
        """Method to be overriden in subclass for model specific sanity checks"""
        pass

    @_with_sanity_checks("_backtest_sanity_checks", "_backtest_model_specific_sanity_checks")
    def backtest(self,
                 feature_series: Iterable[TimeSeries],
                 target_series: TimeSeries,
                 start: Union[pd.Timestamp, float, int] = 0.7,
                 forecast_horizon: int = 1,
                 stride: int = 1,
                 trim_to_series: bool = True,
                 verbose=False) -> TimeSeries:
        """ A function for backtesting `RegressionModel`'s.

        This function computes the time series of historical predictions
        that would have been obtained, if the current model had been used to predict `target_series`
        using the `feature_series`, with a certain time horizon.

        To this end, it repeatedly builds a training set composed of both features and targets,
        from `feature_series` and `target_series`, respectively.
        It trains the current model on the training set, emits a (point) prediction for a fixed
        forecast horizon, and then moves the end of the training set forward by `stride`
        time steps. The resulting predictions are then returned.

        This always re-trains the models on the entire available history,
        corresponding an expending window strategy.

        Parameters
        ----------
        feature_series
            A list of time series representing the features for the regression model (independent variables)
        target_series
            The univariate target time series for the regression model (dependent variable)
        start
            The first prediction time, at which a prediction is computed for a future time
        forecast_horizon
            The forecast horizon for the point predictions
        stride
            The number of time steps (the unit being the frequency of `series`) between two consecutive predictions.
        trim_to_series
            Whether the predicted series has the end trimmed to match the end of the main series
        verbose
            Whether to print progress

        Returns
        -------
        TimeSeries
            A time series containing the forecast values when successively applying
            the current model with the specified forecast horizon.
        """
        start = _get_timestamp_at_point(start, target_series)

        # build the prediction times in advance (to be able to use tqdm)
        if trim_to_series:
            last_pred_time = target_series.time_index()[-forecast_horizon - stride]
        else:
            last_pred_time = target_series.time_index()[-stride - 1]

        # build the prediction times in advance (to be able to use tqdm)
        pred_times = [start]
        while pred_times[-1] <= last_pred_time:
            pred_times.append(pred_times[-1] + target_series.freq() * stride)

        # what we'll return
        values = []
        times = []

        iterator = _build_tqdm_iterator(pred_times, verbose)

        for pred_time in iterator:
            # build train/val series
            train_features = [s.drop_after(pred_time) for s in feature_series]
            train_target = target_series.drop_after(pred_time)
            val_features = [s.slice_n_points_after(pred_time, forecast_horizon) for s in feature_series]

            self.fit(train_features, train_target)
            pred = self.predict(val_features)
            values.append(pred.values()[-1])  # store the N-th point
            times.append(pred.end_time())  # store the N-th timestamp

        return TimeSeries.from_times_and_values(pd.DatetimeIndex(times), np.array(values))

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
