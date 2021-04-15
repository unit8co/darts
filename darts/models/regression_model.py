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


from .. import metrics
from ..timeseries import TimeSeries
from sklearn.linear_model import LinearRegression
from .forecasting_model import ExtendedForecastingModel
from typing import List, Iterable, Union, Any, Callable, Optional
from ..logging import raise_if, raise_if_not, get_logger, raise_log
from ..utils import (
    _build_tqdm_iterator,
    _with_sanity_checks,
    _historical_forecasts_general_checks
)

logger = get_logger(__name__)


# TODO: Extend this to a "DynamicRegressiveModel" class, which acts on List[List[TimeSeries]].
# TODO: The first List[] would contain time-sliding lists of time series, letting the model
# TODO: be able to learn how to change weights over time. When len() of outer List[] is 0 it's a particular case
class RegressionModel(ExtendedForecastingModel):
    def __init__(self,
                 lags: Union[int, list],
                 lags_exog: Union[int, list, bool] = True,
                 model=LinearRegression(n_jobs=-1, fit_intercept=False)):
        """ Regression Model

        Can be used to fit any scikit-learn-like regressor class to predict the target
        time series with lagged values.

        Parameters
        ----------
        lags : Union[int, list]
            Number of lagged target values used to predict the next time step. If an integer is given
            the last `lags` lags are used (inclusive). Otherwise a list of integers with lags.
        lags_exog : Union[int, list, bool]
            Number of lagged exogenous values used to predict the next time step. If an integer is given
            the last `lags_exog` lags are used (inclusive). Otherwise a list of integers with lags. If False,
            the value at time `t` is used which might lead to leakage. If True the same lags as for the
            target variable are used.
        model
            A regression model that implements `fit()` and `predict()` methods.
            Default: `sklearn.linear_model.LinearRegression(n_jobs=-1, fit_intercept=False)`
        """
        raise_if(isinstance(lags, int) and (lags < 0), "Lags must be positive integer or list.")
        raise_if(isinstance(lags, float), "Lags must be integer, not float.")

        if (not callable(getattr(model, "fit", None))):
            raise_log(Exception('Provided model object must have a fit() method', logger))
        if (not callable(getattr(model, "predict", None))):
            raise_log(Exception('Provided model object must have a predict() method', logger))

        self.lags = lags
        self.lags_exog = lags_exog

        self.model = model

        if isinstance(self.lags, int):
            self.lags = list(range(1, self.lags+1))

        if self.lags_exog is True:
            self.lags_exog = self.lags[:]
        elif self.lags_exog is False:
            self.lags_exog = [0]
        elif isinstance(self.lags_exog, int):
            self.lags_exog = list(range(1, self.lags_exog+1))

        self.max_lag = int(np.max([np.max(self.lags), np.max(self.lags_exog)]))
        self._fit_called = False

    def fit(self, series: TimeSeries, exog: Optional[TimeSeries] = None, **kwargs) -> None:
        """ Fits/trains the model using the provided list of features time series and the target time series.

        Parameters
        ----------
        series : TimeSeries
            TimeSeries object containing the target values.
        exog : TimeSeries, optional
            TimeSeries object containing the exogenous values.
        """
        super().fit(series, exog)

        if exog is not None:
            print(series.stack(exog))
        else:
            print(series)
        self.target_column = series.columns()[0]
        self.exog_columns = exog.columns().values.tolist() if exog is not None else None
        self.training_data = self._create_training_data(series, exog)
        print(self.training_data)
        self.train_x =  self.training_data.pd_dataframe().drop(self.target_column, axis=1)
        self.train_y =  self.training_data.pd_dataframe()[self.target_column]
        self.nr_exog = self.train_x.shape[1] - len(self.lags)

        self.model.fit(
            X=self.train_x,
            y=self.train_y,
            **kwargs
        )
        if exog is not None:
            self.prediction_data = series.stack(other=exog)[-self.max_lag:]
        else:
            self.prediction_data = series[-self.max_lag:]

        self._fit_called = True

    def _create_training_data(self, series: TimeSeries, exog: TimeSeries = None):
        """ Create dataframe of exogenous and endogenous variables containing lagged values.

        Parameters
        ----------
        series : TimeSeries
            Target series.
        exog : TimeSeries, optional
            Exogenous variables.

        Returns
        -------
        pd.dataframe
            Data frame with lagged values.
        """
        raise_if(series.width > 1,
            "Series must not be multivariate. Pass exogenous variables to 'exog' parameter.",
            logger
        )
        training_data = self._create_lagged_data(series=series, lags=self.lags, keep_current=True)
        if exog is not None:
            for col in exog.pd_dataframe().columns:
                col_lagged_data = self._create_lagged_data(
                    series=exog[col], lags=self.lags_exog, keep_current=False
                )
                training_data = pd.concat([training_data, col_lagged_data], axis=1)

        return TimeSeries(training_data.dropna(), freq=series.freq())

    def _create_lagged_data(self, series: TimeSeries, lags: list, keep_current: bool):
        """ Creates a data frame where every lag is a new column.

        After creating this input it can be used in many regression models to make predictions
        based on previous available values for both the exogenous as well as endogenous variables.

        Parameters
        ----------
        series : TimeSeries
            Time series to be lagged.
        lags : list
            List if indexes.
        keep_current : bool
            If False, the current value (not-lagged) is dropped.

        Returns
        -------
        TYPE
            Returns a data frame that contains lagged values.
        """
        lagged_series = series.pd_dataframe(copy=True)
        target_name = lagged_series.columns[0]
        for lag in lags:
            new_column_name = target_name + "_lag{}".format(lag)
            lagged_series[new_column_name] = lagged_series[target_name].shift(lag)
        lagged_series.dropna(inplace=True)
        if not keep_current:
            lagged_series.drop(target_name, axis=1, inplace=True)
        return lagged_series

    def predict(self, n: int, exog: Optional[TimeSeries] = None, **kwargs):
        super().predict(n, exog)
        if isinstance(exog, TimeSeries):
            exog = exog.pd_dataframe()

        prediction_data = self.prediction_data.copy()
        dummy_row = np.zeros(shape=(1, prediction_data.width))
        prediction_data = prediction_data.append_values(dummy_row)
        forecasts = list(range(n))

        for i in range(n):
            target_data = prediction_data[[self.target_column]]
            if self.exog_columns is not None:
                exog_data = prediction_data[self.exog_columns]
            else:
                exog_data = None
            forecasting_data = self._create_training_data(target_data, exog=exog_data)
            forecasting_data = forecasting_data.pd_dataframe().iloc[:, 1:]
            forecast = self.model.predict(
                X=forecasting_data,
                **kwargs
            )
            prediction_data = prediction_data[1:-1]
            if self.exog_columns is not None:
                append_row = [[forecast[0], *exog.iloc[i, :].values.tolist()]]
                prediction_data = prediction_data.append_values(append_row)
            else:
                prediction_data = prediction_data.append_values(forecast)
            prediction_data = prediction_data.append_values(dummy_row)
            forecasts[i] = forecast[0]
        return self._build_forecast_series(forecasts)

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

        train_pred = self.predict(self.train_x)
        return self.train_y - train_pred

    def __str__(self):
        return model.__str__()