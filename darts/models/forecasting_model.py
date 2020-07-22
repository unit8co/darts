"""
Forecasting Model Base Class
----------------------------

A forecasting model captures the future values of a time series as a function of the past as follows:

.. math:: y_{t+1} = f(y_t, y_{t-1}, ..., y_1),

where :math:`y_t` represents the time series' value(s) at time :math:`t`.
"""

from typing import Optional, Tuple, Union, Dict, Any
from types import SimpleNamespace
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not
from ..utils import _build_tqdm_iterator, _with_sanity_checks

logger = get_logger(__name__)


class ForecastingModel(ABC):
    """ The base class for all forecasting models.

    All implementations of forecasting have to implement the `fit()` and `predict()` methods defined below.
    """

    @abstractmethod
    def __init__(self):
        # Stores training date information:
        self.training_series: TimeSeries = None

        # state
        self._fit_called = False

    @abstractmethod
    def fit(self, series: TimeSeries) -> None:
        """ Fits/trains the model on the provided series

        Parameters
        ----------
        series
            the training time series on which to fit the model
        """
        raise_if_not(len(series) >= self.min_train_series_length,
                     "Train series only contains {} elements but {} model requires at least {} entries"
                     .format(len(series), str(self), self.min_train_series_length))
        self.training_series = series
        self._fit_called = True

    @abstractmethod
    def predict(self, n: int) -> TimeSeries:
        """ Predicts values for a certain number of time steps after the end of the training series

        Parameters
        ----------
        n
            The number of time steps after the end of the training time series for which to produce predictions

        Returns
        -------
        TimeSeries
            A time series containing the `n` next points, starting after the end of the training time series
        """

        if (not self._fit_called):
            raise_log(Exception('fit() must be called before predict()'), logger)

    @property
    def min_train_series_length(self) -> int:
        """
        Class property defining the minimum required length for the training series.
        This function/property should be overridden if a value higher than 3 is required.
        """
        return 3

    def _generate_new_dates(self, n: int) -> pd.DatetimeIndex:
        """
        Generates `n` new dates after the end of the training set
        """
        new_dates = [
            (self.training_series.time_index()[-1] + (i * self.training_series.freq())) for i in range(1, n + 1)
        ]
        return pd.DatetimeIndex(new_dates, freq=self.training_series.freq_str())

    def _build_forecast_series(self,
                               points_preds: np.ndarray) -> TimeSeries:
        """
        Builds a forecast time series starting after the end of the training time series, with the
        correct time index.
        """

        time_index = self._generate_new_dates(len(points_preds))

        return TimeSeries.from_times_and_values(time_index, points_preds, freq=self.training_series.freq())

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
        covariate_series = args[0]
        n = SimpleNamespace(**kwargs)
        if n.target_series is None:
            target_series = covariate_series
        raise_if_not(covariate_series.time_index == target_series.time_index, "the target and covariate series must "
                     "have the same time indices.")
        raise_if_not(n.start in covariate_series, '`start` timestamp is not in the `series`.', logger)
        raise_if_not(n.start != covariate_series.end_time(), '`start` timestamp is the last timestamp of `series`',
                     logger)
        raise_if_not(n.forecast_horizon > 0, 'The provided forecasting horizon must be a positive integer.', logger)

    def _backtest_model_specific_sanity_checks(self, *args: Any, **kwargs: Any) -> None:
        """Method to be overriden in subclass for model specific sanity checks"""
        pass

    @_with_sanity_checks("_backtest_sanity_checks", "_backtest_model_specific_sanity_checks")
    def backtest(self,
                 covariate_series: TimeSeries,
                 target_series: Optional[TimeSeries] = None,
                 start: Union[pd.Timestamp, float, int] = 0.5,
                 forecast_horizon: int = 1,
                 stride: int = 1,
                 retrain: bool = True,
                 trim_to_series: bool = True,
                 verbose: bool = False,
                 fit_kwargs: Optional[Dict[str, Any]] = None,
                 predict_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[TimeSeries, TimeSeries]:
        """ Retrain and forecast values pointwise with an expanding training window over `series`.

        To this end, it repeatedly builds a training set from the beginning of `series`. It trains `model` on the
        training set, emits a (point) prediction for a fixed forecast horizon, and then moves the end of the training
        set forward by one time step. The resulting predictions are then returned.

        Unless `retrain` is set to False, this always re-trains the models on the entire available history,
        corresponding an expending window strategy.

        If `retrain` is set to False (useful for models with many parameter such as `TorchForecastingModel` instances),
        the model will only be trained only on the initial training window (up to `start` time stamp), and only if it
        has not been trained before. Then, at every iteration, the newly expanded 'training sequence' will be fed to the
        model to produce the new output.

        Parameters
        ----------
        covariate_series
            The covariate time series on which to backtest
        target_series
            The target time series on which to backtest
        start
            The first prediction time, at which a prediction is computed for a future time
        forecast_horizon
            The forecast horizon for the point prediction
        stride
            The number of time steps (the unit being the frequency of `series`) between two consecutive predictions.
        retrain
            Whether to retrain the model for every prediction or not. Currently only `TorchForecastingModel`
            instances as `model` argument support setting `retrain` to `False`.
        trim_to_series
            Whether the predicted series has the end trimmed to match the end of the main series
        verbose
            Whether to print progress
        fit_kwargs
            Additional arguments passed to the fit method during backtesting
        predict_kwargs
            Additional arguments passed to the predict method during backtesting

        Returns
        -------
        forecast
            A time series containing the forecast values for `target_series`, when successively applying the specified
            model with the specified forecast horizon.
        residuals
            Difference between the `forecast` and the actual `target_series` provided.
        """
        # build the prediction times in advance (to be able to use tqdm)
        if trim_to_series:
            last_pred_time = covariate_series.time_index()[-forecast_horizon - stride]
        else:
            last_pred_time = covariate_series.time_index()[-stride - 1]

        pred_times = [start]
        while pred_times[-1] <= last_pred_time:
            pred_times.append(pred_times[-1] + covariate_series.freq() * stride)

        # iterate and predict pointwise
        values = []
        times = []

        iterator = _build_tqdm_iterator(pred_times, verbose)

        if not retrain:
            self.fit(covariate_series.drop_after(start), target_series.drop_after(start), verbose=verbose)

        for pred_time in iterator:
            train = covariate_series.drop_after(pred_time)  # build the training series
            target = target_series.drop_after(pred_time)  # build the target series
            if (retrain):
                self.fit(train, target)
                pred = self.predict(forecast_horizon, **predict_kwargs)
            else:
                pred = self.predict(forecast_horizon, input_series=train, **predict_kwargs)
            values.append(pred.values()[-1])  # store the N-th point
            times.append(pred.end_time())  # store the N-th timestamp

        forecast = TimeSeries.from_times_and_values(pd.DatetimeIndex(times), np.array(values))

        # compute the residuals between forecast and series
        target_series_trimmed = target_series.slice_intersect(forecast)
        residuals = target_series_trimmed - forecast

        return forecast, residuals


class UnivariateForecastingModel(ForecastingModel):
    """ The base class for univariate forecasting models.
    """

    @abstractmethod
    def fit(self, series: TimeSeries) -> None:
        """ Fits/trains the univariate model on selected univariate series.

        Parameters
        ----------
        series
            A **univariate** training time series on which to fit the model.
        """
        series._assert_univariate()
        super().fit(series)


class MultivariateForecastingModel(ForecastingModel):
    """ The base class for multivariate forecasting models.
    """

    @abstractmethod
    def fit(self, covariate_series: TimeSeries, target_series: TimeSeries) -> None:
        """ Fits/trains the multivariate model on the provided series with selected target components.

        Parameters
        ----------
        covariate_series
            The training time series on which to fit the model (can be multivariate or univariate).
        target_series
            The target values used ad dependent variables when training the model
        """
        raise_if_not(len(covariate_series) == len(target_series), "covariate_series and target_series musth have same "
                     "length.")
        super().fit(covariate_series)
