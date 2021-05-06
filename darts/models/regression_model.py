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
from typing import List, Iterable, Union, Any, Callable

from ..timeseries import TimeSeries
from ..logging import raise_if_not, get_logger, raise_log
from ..utils import (
    _build_tqdm_iterator,
    _with_sanity_checks,
    _historical_forecasts_general_checks
)
from .. import metrics

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

    def _historical_forecasts_sanity_checks(self, *args: Any, **kwargs: Any) -> None:
        """Sanity checks for the historical_forecasts function

        Parameters
        ----------
        args
            The args parameter(s) provided to the historical_forecasts function.
        kwargs
            The kwargs paramter(s) provided to the historical_forecasts function.

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

        _historical_forecasts_general_checks(target_series, kwargs)

    @_with_sanity_checks("_historical_forecasts_sanity_checks")
    def historical_forecasts(self,
                             feature_series: Iterable[TimeSeries],
                             target_series: TimeSeries,
                             start: Union[pd.Timestamp, float, int] = 0.5,
                             forecast_horizon: int = 1,
                             stride: int = 1,
                             overlap_end: bool = False,
                             last_points_only: bool = True,
                             verbose: bool = False) -> Union[List[TimeSeries], TimeSeries]:
        """ Computes the historical forecasts the model would have produced with an expanding training window
        and (by default) returns a time series created from the last point of each of these individual forecasts

        To this end, it repeatedly builds a training set composed of both features and targets,
        from `feature_series` and `target_series`, respectively.
        It trains the current model on the training set, emits a forecast of length equal to forecast_horizon,
        and then moves the end of the training set forward by `stride` time steps.

        By default, this method will return a single time series made up of the last point of each
        historical forecast. This time series will thus have a frequency of training_series.freq() * stride
        If `last_points_only` is set to False, it will instead return a list of the historical forecasts.

        This always re-trains the models on the entire available history,
        corresponding an expanding window strategy.

        Parameters
        ----------
        feature_series
            An iterable of time series representing the features for the regression model (independent variables)
        target_series
            The univariate target time series for the regression model (dependent variable)
        start
            The first prediction time, at which a prediction is computed for a future time
        forecast_horizon
            The forecast horizon for the point predictions
        stride
            The number of time steps between two consecutive predictions.
        overlap_end
            Whether the returned forecasts can go beyond the series' end or not
        last_points_only
            Whether to retain only the last point of each historical forecast.
            If set to True, the method returns a single `TimeSeries` of the point forecasts.
            Otherwise returns a list of historical `TimeSeries` forecasts.
        verbose
            Whether to print progress

        Returns
        -------
        TimeSeries or List[TimeSeries]
            By default, a single TimeSeries instance created from the last point of each individual forecast.
            If `last_points_only` is set to False, a list of the historical forecasts
        """
        start = target_series.get_timestamp_at_point(start)

        # build the prediction times in advance (to be able to use tqdm)
        if not overlap_end:
            last_valid_pred_time = target_series.time_index()[-1 - forecast_horizon]
        else:
            last_valid_pred_time = target_series.time_index()[-2]

        pred_times = [start]
        while pred_times[-1] < last_valid_pred_time:
            # compute the next prediction time and add it to pred times
            pred_times.append(pred_times[-1] + target_series.freq() * stride)

        # the last prediction time computed might have overshot last_valid_pred_time
        if pred_times[-1] > last_valid_pred_time:
            pred_times.pop(-1)

        iterator = _build_tqdm_iterator(pred_times, verbose)

        # Either store the whole forecasts or only the last points of each forecast, depending on last_points_only
        forecasts = []

        last_points_times = []
        last_points_values = []

        for pred_time in iterator:
            # build train/val series
            train_features = [s.drop_after(pred_time) for s in feature_series]
            train_target = target_series.drop_after(pred_time)
            val_features = [s.slice_n_points_after(pred_time, forecast_horizon) for s in feature_series]

            self.fit(train_features, train_target)
            forecast = self.predict(val_features)

            if last_points_only:
                last_points_values.append(forecast.values()[-1])
                last_points_times.append(forecast.end_time())
            else:
                forecasts.append(forecast)

        if last_points_only:
            return TimeSeries.from_times_and_values(pd.DatetimeIndex(last_points_times),
                                                    np.array(last_points_values),
                                                    freq=target_series.freq() * stride)

        return forecasts

    def backtest(self,
                 feature_series: Iterable[TimeSeries],
                 target_series: TimeSeries,
                 start: Union[pd.Timestamp, float, int] = 0.5,
                 forecast_horizon: int = 1,
                 stride: int = 1,
                 overlap_end: bool = False,
                 last_points_only: bool = False,
                 metric: Callable[[TimeSeries, TimeSeries], float] = metrics.mape,
                 reduction: Union[Callable[[np.ndarray], float], None] = np.mean,
                 verbose: bool = False) -> Union[float, List[float]]:
        """Computes an error score between the historical forecasts the model would have produced
        with an expanding training window over `series` and the actual series.

        To this end, it repeatedly builds a training set composed of both features and targets,
        from `feature_series` and `target_series`, respectively.
        It trains the current model on the training set, emits a forecast of length equal to forecast_horizon,
        and then moves the end of the training set forward by `stride` time steps.

        By default, this method will use each historical forecast (whole) to compute error scores.
        If `last_points_only` is set to True, it will use only the last point of each historical forecast.

        This always re-trains the models on the entire available history,
        corresponding an expanding window strategy.

        Parameters
        ----------
        feature_series
            An iterable of time series representing the features for the regression model (independent variables)
        target_series
            The univariate target time series for the regression model (dependent variable)
        start
            The first prediction time, at which a prediction is computed for a future time
        forecast_horizon
            The forecast horizon for the point predictions
        stride
            The number of time steps between two consecutive predictions.
        overlap_end
            Whether the returned forecasts can go beyond the series' end or not
        last_points_only
            Whether to keep the whole historical forecasts or only the last point of each forecast
        metric
            A function that takes two TimeSeries instances as inputs and returns a float error value.
        reduction
            A function used to combine the individual error scores obtained when `last_points_only` is set to False.
            If explicitely set to `None`, the method will return a list of the individual error scores instead.
            Set to np.mean by default.
        verbose
            Whether to print progress

        Returns
        -------
        float or List[float]
            The error score, or the list of individual error scores if `reduction` is `None`
        """
        forecasts = self.historical_forecasts(feature_series,
                                              target_series,
                                              start,
                                              forecast_horizon,
                                              stride,
                                              overlap_end,
                                              last_points_only,
                                              verbose)

        if last_points_only:
            return metric(target_series, forecasts)

        errors = []
        for forecast in forecasts:
            errors.append(metric(target_series, forecast))

        if reduction is None:
            return errors

        return reduction(errors)

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
