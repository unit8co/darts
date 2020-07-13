"""
Forecasting Model Base Class
----------------------------

A forecasting model captures the future values of a time series as a function of the past as follows:

.. math:: y_{t+1} = f(y_t, y_{t-1}, ..., y_1),

where :math:`y_t` represents the time series' value(s) at time :math:`t`.
"""

from typing import Optional, List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not, raise_if
from ..utils import _build_tqdm_iterator

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

        return TimeSeries.from_times_and_values(time_index, points_preds, self.training_series.freq())

    def _backtest_sanity_checks(self, series: TimeSeries, start: pd.Timestamp, forcast_horizon: int):
        """Perform sanity checks on backtest inputs.

        Parameters
        ----------
        series
            The univariate time series on which to backtest
        start
            The first prediction time, at which a prediction is computed for a future time
        forcast_horizon
            The forecast horizon for the point predictions

        Raises
        ------
        ValueError
            if an input doesn't pass sanity checks.
        """
        raise_if_not(start in series, '`start` timestamp is not in the `series`.', logger)
        raise_if_not(start != series.end_time(), '`start` timestamp is the last timestamp of `series`', logger)
        raise_if_not(forcast_horizon > 0, 'The provided forecasting horizon must be a positive integer.', logger)

    def _backtest_model_specfic_sanity_checks(self, retrain: bool):
        """Add model specific sanity check(s) on the backtest inputs.

        Parameters
        ----------
        retrain
            Whether to retrain the model for every prediction or not. Currently only `TorchForecastingModel`
            instances as `model` argument support setting `retrain` to `False`.

        Raises
        ------
        ValueError
            if an input doesn't pass sanity checks.
        """
        raise_if(not retrain, "Only 'TorchForecastingModel' instances support the option 'retrain=False'.", logger)

    def _backtest_build_fit_and_predict_kwargs(self,
                                               target_indices: Optional[List[int]],
                                               component_index: Optional[int],
                                               use_full_output_length: bool):
        """ Adapt fit and predict kwargs depending on the model used for backtesting

        Parameters
        ----------
        target_indices
            In case `series` is multivariate and `model` is a subclass of `MultivariateForecastingModel`,
            a list of indices of components of `series` to be predicted by `model`.
        component_index
            In case `series` is multivariate and `model` is a subclass of `UnivariateForecastingModel`,
            an integer index of the component of `series` to be predicted by `model`.
        use_full_output_length
            In case `model` is a subclass of `TorchForecastingModel`, this argument will be passed along
            as argument to the predict method of `model`.

        Returns
        -------
        fit_kwargs
            kwargs passed to the fit method during backtesting.
        predict kwargs
            kwargs passed to the predict method during backtesting.
        """
        fit_kwargs = {}
        predict_kwargs = {}
        fit_kwargs['target_indices'] = target_indices
        return fit_kwargs, predict_kwargs

    def backtest(self,
                 series: TimeSeries,
                 start: pd.Timestamp,
                 forcast_horizon: int,
                 target_indices: Optional[List[int]] = None,
                 component_index: Optional[int] = None,
                 use_full_output_length: bool = True,
                 stride: int = 1,
                 retrain: bool = True,
                 trim_to_series: bool = True,
                 verbose: bool = False) -> TimeSeries:
        """ Retrain and forcast values pointwise with an expanding training window over `series`.
        
        To this end, it repeatedly builds a training set from the beginning of `series`. It trains `model` on the
        training set, emits a (point) prediction for a fixed forecast horizon, and then moves the end of the training
        set forward by one time step. The resulting predictions are then returned.

        Unless `retrain` is set to False, this always re-trains the models on the entire available history,
        corresponding an expending window strategy.

        If `retrain` is set to False (useful for models with many parameter such as `TorchForecastingModel` instances),
        the model will only be trained only on the initial training window (up to `start` time stamp), and only if it
        has not been trained before. Then, at every iteration, the newly expanded 'training sequence' will be fed to the
        model to produce the new output.

        iteration 1:

             x            o
           /   \          ^
         x       x - x ...|
        |_________|_______|
         training  forcast
          window   horizon

        iteration 2:

             x            o
           /   \            \  
         x       x - x ...     -  - o
        |_____________|_______|_____^
           training    forcast  stride
            window     horizon
        
        ...

        legend:
            - x: point from provided `series`.
            - o: point forcasted by the model trained on the `training_window` data.

        Parameters
        ----------
        series
            The univariate time series on which to backtest
        start
            The first prediction time, at which a prediction is computed for a future time
        forcast_horizon
            The forecast horizon for the point predictions
        target_indices
            In case `series` is multivariate and `model` is a subclass of `MultivariateForecastingModel`,
            a list of indices of components of `series` to be predicted by `model`.
        component_index
            In case `series` is multivariate and `model` is a subclass of `UnivariateForecastingModel`,
            an integer index of the component of `series` to be predicted by `model`.
        use_full_output_length
            In case `model` is a subclass of `TorchForecastingModel`, this argument will be passed along
            as argument to the predict method of `model`.
        stride
            The number of time steps (the unit being the frequency of `series`) between two consecutive predictions.
        retrain
            Whether to retrain the model for every prediction or not. Currently only `TorchForecastingModel`
            instances as `model` argument support setting `retrain` to `False`.
        trim_to_series
            Whether the predicted series has the end trimmed to match the end of the main series
        verbose
            Whether to print progress
        
        Returns
        -------
        TimeSeries
            A time series containing the forecast values for `series`, when successively applying the specified model
            with the specified forecast horizon.
        """ # noqa : W605

        # sanity checks
        self._backtest_sanity_checks(series, start, forcast_horizon)
        self._backtest_model_specfic_sanity_checks(retrain)

        # specify the correct fit and predict keyword arguments depending on the model
        fit_kwargs, predict_kwargs = self._backtest_build_fit_and_predict_kwargs(target_indices,
                                                                                 component_index,
                                                                                 use_full_output_length)

        # build the prediction times in advance (to be able to use tqdm)
        last_pred_time = (
            series.time_index()[-forcast_horizon - stride] if trim_to_series else series.time_index()[-stride - 1]
        )
        pred_times = [start]
        while pred_times[-1] <= last_pred_time:
            pred_times.append(pred_times[-1] + series.freq() * stride)

        # iterate and predict pointwise
        values = []
        times = []

        iterator = _build_tqdm_iterator(pred_times, verbose)

        if ((not retrain) and (not self._fit_called)):
            self.fit(series.drop_after(start), verbose=verbose, **fit_kwargs)

        for pred_time in iterator:
            train = series.drop_after(pred_time)  # build the training series
            if (retrain):
                self.fit(train, **fit_kwargs)
                pred = self.predict(forcast_horizon, **predict_kwargs)
            else:
                pred = self.predict(forcast_horizon, input_series=train, **predict_kwargs)
            values.append(pred.values()[-1])  # store the N-th point
            times.append(pred.end_time())  # store the N-th timestamp
        return TimeSeries.from_times_and_values(pd.DatetimeIndex(times), np.array(values))


class UnivariateForecastingModel(ForecastingModel):
    """ The base class for univariate forecasting models.
    """

    @abstractmethod
    def fit(self, series: TimeSeries, component_index: Optional[int] = None) -> None:
        """ Fits/trains the univariate model on selected univariate series.

        Parameters
        ----------
        series
            The training time series on which to fit the model.
        component_index
            Optionally, a zero-indexed integer indicating the component to use if a multivariate
            time series is passed.
        """

        raise_if_not(series.width == 1 or (component_index is not None), "If a multivariate series is given"
                     "as input to this univariate model, please provide a `component_index` integer indicating"
                     " which component to use.", logger)

        if series.width == 1:
            super().fit(series)
        else:
            super().fit(series.univariate_component(component_index))

    def _backtest_build_fit_and_predict_kwargs(self,
                                               target_indices: Optional[List[int]],
                                               component_index: Optional[int],
                                               use_full_output_length: bool):
        """ Adapt fit and predict kwargs depending on the model used for backtesting

        Parameters
        ----------
        target_indices
            In case `series` is multivariate and `model` is a subclass of `MultivariateForecastingModel`,
            a list of indices of components of `series` to be predicted by `model`.
        component_index
            In case `series` is multivariate and `model` is a subclass of `UnivariateForecastingModel`,
            an integer index of the component of `series` to be predicted by `model`.
        use_full_output_length
            In case `model` is a subclass of `TorchForecastingModel`, this argument will be passed along
            as argument to the predict method of `model`.

        Returns
        -------
        fit_kwargs
            kwargs passed to the fit method during backtesting.
        predict kwargs
            kwargs passed to the predict method during backtesting.
        """
        fit_kwargs = {}
        predict_kwargs = {}
        fit_kwargs['component_index'] = component_index
        return fit_kwargs, predict_kwargs


class MultivariateForecastingModel(ForecastingModel):
    """ The base class for multivariate forecasting models.
    """

    @abstractmethod
    def fit(self, series: TimeSeries, target_indices: Optional[List[int]] = None) -> None:
        """ Fits/trains the multivariate model on the provided series with selected target components.

        Parameters
        ----------
        series
            The training time series on which to fit the model.
        target_indices
            A list of integers indicating which component(s) of the time series should be used
            as targets for forecasting.
        """

        if series.width == 1:
            target_indices = [0]

        raise_if_not(target_indices is not None and len(target_indices) > 0,
                     "If a multivariate series is given as input to this multivariate model, please"
                     " provide a list of integer indices `target_indices`"
                     " that indicate which components of this series should be predicted", logger)

        raise_if_not(all(idx >= 0 and idx < series.width for idx in target_indices), "The target indices "
                     "must all be between 0 and the width of the TimeSeries instance used for fitting - 1.", logger)

        self.target_indices = target_indices
        super().fit(series)
