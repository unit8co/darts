"""
Forecasting Model Base Class
----------------------------

A forecasting model captures the future values of a time series as a function of the past as follows:

.. math:: y_{t+1} = f(y_t, y_{t-1}, ..., y_1),

where :math:`y_t` represents the time series' value(s) at time :math:`t`.

The main functions are `fit()` and `predict()`. `fit()` learns the function `f()`, over the history of
one or several time series. The function `predict()` applies `f()` in order to obtain forecasts for a
desired number of time stamps into the future.
"""

from typing import Optional, Tuple, Union, Any, Callable
from types import SimpleNamespace
from itertools import product
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..utils.data.timeseries_dataset import TimeSeriesDataset
from ..logging import get_logger, raise_log, raise_if_not
from ..utils import _build_tqdm_iterator, _with_sanity_checks, _get_timestamp_at_point, _backtest_general_checks
from .. import metrics

logger = get_logger(__name__)


class ForecastingModel(ABC):
    """ The base class for all forecasting models.

    All implementations of forecasting have to implement the `fit()` and `predict()` methods defined below.
    The `fit()` method is meant to train the model on one training time series.

    In addition, models can optionally implement the `multi_fit()` method. This method is meant to train the model
    on several time series, or to handle cases where the "features" and "target" dimensions are not the same.
    `fit()` will typically be the entry point for quickly fitting reasonably simple models to one series,
     while `multi_fit()` will typically be the path taken by the more sophisticated machine learning models.

    Depending whether a model has been trained using `fit()` or `multi_fit()` defines two possible behaviors:
    * `fit()`: store the training series, to predict its future "automatically" when `predict()` is called.
    * `multi_fit()` does not store the dataset. The caller of `predict()` has to provide the input series for
    which the forecast has to be made.

    Attributes
    ----------
    training_series
        Reference to the `TimeSeries` used for training the model through the `fit()` function.
    """
    @abstractmethod
    def __init__(self):
        # Stores training date information:
        self.training_series: TimeSeries = None

        # state; whether the model has been fit or not
        self._fit_called = False

    @abstractmethod
    def fit(self, training_series: TimeSeries) -> None:
        """ Fits/trains the model on the provided series

        Defines behavior that should happen when calling the `fit()` method of every forecasting model.

        This is the entry point for training the model on one series.

        Some models support training on several time series, or differentiating between "input" and "target"
        dimensions. At the moment such functionalities can be used for PyTorch-based models,
        using some `TimeSeriesDataset`, and calling the `multi_fit()` method.

        Parameters
        ----------
        training_series
            The time series on which to train the model, and the future of which can then be forecast using `predict()`.
        """
        raise_if_not(len(training_series) >= self.min_train_series_length,
                     "Train series only contains {} elements but {} model requires at least {} entries"
                     .format(len(training_series), str(self), self.min_train_series_length))

        self.training_series = training_series
        self._fit_called = True

    def multi_fit(self, time_series_dataset: TimeSeriesDataset) -> None:
        """

        Parameters
        ----------
        time_series_dataset
            The dataset of `TimeSeries` over which to fit the model. This dataset can either emmit tuples of
            (input, target) series, in which case the model will be trained to predict "target" from "input"; or
            it can emmit simple `TimeSeries`, in which case the model will take all the series' components as
            inputs and targets.
        """
        raise NotImplemented('This model does not support multi_fit(). Currently only PyTorch-based models'
                             '(neural nets) support this functionality.')

    @abstractmethod
    def predict(self, n: int) -> TimeSeries:
        """ Forecasts values for a certain number of time steps after the end of the training series.

        Parameters
        ----------
        n
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.

        Returns
        -------
        TimeSeries
            A time series containing the `n` next points, starting after the end of the series to forecast.
        """

        if not self._fit_called:
            raise_log(Exception('The model must be fit before calling predict()'), logger)

    def multi_predict(self, n: int, input_series_dataset: TimeSeriesDataset) -> TimeSeriesDataset:
        """

        Parameters
        ----------
        n
            Forecast horizon - the number of time steps after the end of each series for which to produce predictions.
        input_series_dataset
            A dataset of input time series. The model will predict the `n` time stamps following the end
            of each of these time series. This dataset must emit simple `TimeSeries`. If it emits tuples
            of (input, target) series instead, then only the inputs will be considered. The dimensions of
            the time series must matched what has been used for training.
        """
        raise NotImplemented('This model does not support multi_predict(). Currently only PyTorch-based models'
                             '(neural nets) support this functionality.')

    @property
    def min_train_series_length(self) -> int:
        """
        Class property defining the minimum required length for the training series.
        This function/property should be overridden if a value higher than 3 is required.
        """
        return 3

    def _generate_new_dates(self,
                            n: int,
                            input_series: Optional[TimeSeries] = None) -> pd.DatetimeIndex:
        """
        Generates `n` new dates after the end of the training set (or after the end of the input series, if specified)
        """
        input_series = input_series if input_series is not None else self.training_series
        new_dates = [
            (input_series.time_index()[-1] + (i * input_series.freq())) for i in range(1, n + 1)
        ]
        return pd.DatetimeIndex(new_dates, freq=input_series.freq_str())

    def _build_forecast_series(self,
                               points_preds: np.ndarray,
                               input_series: Optional[TimeSeries] = None) -> TimeSeries:
        """
        Builds a forecast time series starting after the end of the training time series, with the
        correct time index (or after the end of the input series, if specified).
        """
        input_series = input_series if input_series is not None else self.training_series
        time_index = self._generate_new_dates(len(points_preds), input_series=input_series)
        return TimeSeries.from_times_and_values(time_index, points_preds, freq=input_series.freq_str())

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
        # parse args and kwargs
        training_series = args[0]
        n = SimpleNamespace(**kwargs)

        # check target and training series
        if n.target_series is None:
            target_series = training_series
        else:
            target_series = n.target_series

        raise_if_not(all(training_series.time_index() == target_series.time_index()), "the target and training series"
                     " must have the same time indices.")

        _backtest_general_checks(training_series, kwargs)

    def _backtest_model_specific_sanity_checks(self, *args: Any, **kwargs: Any) -> None:
        """Method to be overriden in subclass for model specific sanity checks"""
        pass

    @_with_sanity_checks("_backtest_sanity_checks", "_backtest_model_specific_sanity_checks")
    def backtest(self,
                 series: TimeSeries,
                 start: Union[pd.Timestamp, float, int] = 0.7,
                 forecast_horizon: int = 1,
                 stride: int = 1,
                 retrain: bool = True,
                 trim_to_series: bool = True,
                 verbose: bool = False,
                 use_full_output_length: Optional[bool] = None) -> TimeSeries:
        """ Retrain and forecast values pointwise with an expanding training window over `series`.

        To this end, it repeatedly builds a training set from the beginning of `series`. It trains the current model on
        the training set, emits a (point) prediction for a fixed forecast horizon, and then moves the end of the
        training set forward by `stride` time steps. The resulting predictions are then returned.

        Unless `retrain` is set to False, this always re-trains the models on the entire available history,
        corresponding an expending window strategy.

        If `retrain` is set to False (useful for models with many parameter such as `TorchForecastingModel` instances
        like `RNNModel` and `TCNModel`), the model will only be trained on the initial training window
        (up to `start` time stamp), and only if it has not been trained before. Then, at every iteration, the
        newly expanded input sequence will be fed to the model to produce the new output.

        This backtesting method is meant to backtest the model on one time series.

        Parameters
        ----------
        series
            The training time series on which to backtest
        start
            The first prediction time, at which a prediction is computed for a future time.
            This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
            In the case of `float`, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of `int`, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of `pandas.Timestamp`, this time stamp will be used to determine the first prediction time
            directly.
        forecast_horizon
            The forecast horizon for the point prediction
        stride
            The number of time steps (the unit being the frequency of `series`) between two consecutive predictions.
        retrain
            Whether to retrain the model for every prediction or not. Currently only `TorchForecastingModel`
            instances such as `RNNModel` and `TCNModel` support setting `retrain` to `False`.
        use_full_output_length
            Optionally, if the model is an instance of `TorchForecastingModel`, this argument will be passed along
            as argument to the `predict` method of the model. Otherwise, if this value is set and the model is not an
            instance of `TorchForecastingModel`, this will cause an error.
        trim_to_series
            Whether the predicted series has the end trimmed to match the end of the main series
        verbose
            Whether to print progress

        Returns
        -------
        TimeSeries
            A time series containing the forecast values for `target_series`, when successively applying the specified
            model with the specified forecast horizon.
        """

        # construct predict kwargs dictionary
        predict_kwargs = {}
        if use_full_output_length is not None:
            predict_kwargs['use_full_output_length'] = use_full_output_length

        # prepare the start parameter -> pd.Timestamp
        start = _get_timestamp_at_point(start, series)

        # build the prediction times in advance (to be able to use tqdm)
        if trim_to_series:
            last_pred_time = series.time_index()[-forecast_horizon - stride]
        else:
            last_pred_time = series.time_index()[-stride - 1]

        pred_times = [start]
        while pred_times[-1] <= last_pred_time:
            pred_times.append(pred_times[-1] + series.freq() * stride)

        # iterate and predict pointwise
        values = []
        times = []

        iterator = _build_tqdm_iterator(pred_times, verbose)

        if not retrain and not self._fit_called:
            self.fit(series.drop_after(start), verbose=verbose)

        for pred_time in iterator:
            train = series.drop_after(pred_time)  # build the training series
            if retrain:
                self.fit(train)
                pred = self.predict(forecast_horizon, **predict_kwargs)
            else:
                # TODO: remove this case (which can fail for non-torch models)
                # TODO: and implement dedicated backtest() method for torch/ML models
                pred = self.predict(forecast_horizon, input_series=train, **predict_kwargs)
            values.append(pred.values()[-1])  # store the N-th point
            times.append(pred.end_time())  # store the N-th timestamp

        forecast = TimeSeries.from_times_and_values(pd.DatetimeIndex(times), np.array(values),
                                                    freq=series.freq() * stride)

        return forecast

    @classmethod
    def gridsearch(model_class,
                   parameters: dict,
                   series: TimeSeries,
                   forecast_horizon: Optional[int] = None,
                   start: Union[pd.Timestamp, float, int] = 0.5,
                   use_full_output_length: Optional[bool] = None,
                   val_series: Optional[TimeSeries] = None,
                   use_fitted_values: bool = False,
                   metric: Callable[[TimeSeries, TimeSeries], float] = metrics.mape,
                   verbose=False) -> Tuple['ForecastingModel', Dict]:
        """ A function for finding the best hyperparameters.

        This function has 3 modes of operation: Expanding window mode, split mode and fitted value mode.
        The three modes of operation evaluate every possible combination of hyperparameter values
        provided in the `parameters` dictionary by instantiating the `model_class` subclass
        of ForecastingModel with each combination, and returning the best-performing model with regards
        to the `metric` function. The `metric` function is expected to return an error value,
        thus the model resulting in the smallest `metric` output will be chosen.
        The relationship of the training data and test data depends on the mode of operation.

        Expanding window mode (activated when `forecast_horizon` is passed):
        For every hyperparameter combination, the model is repeatedly trained and evaluated on different
        splits of `series`. This process is accomplished by using `ForecastingModel.backtest`
        as a subroutine to produce historic forecasts starting from `start`
        that are compared against the ground truth values of `series`.
        Note that the model is retrained for every single prediction, thus this mode is slower.

        Split window mode (activated when `val_series` is passed):
        This mode will be used when the `val_series` argument is passed.
        For every hyperparameter combination, the model is trained on `series` and
        evaluated on `val_series`.

        Fitted value mode (activated when `use_fitted_values` is set to `True`):
        For every hyperparameter combination, the model is trained on `series`
        and evaluated on the resulting fitted values.
        Not all models have fitted values, and this method raises an error if `model.fitted_values` does not exist.
        The fitted values are the result of the fit of the model on the training series. Comparing with the
        fitted values can be a quick way to assess the model, but one cannot see if the model overfits or underfits.

        Note that this method is meant to gridsearch and backtest over one training series (along with a possible
        validation series).

        Parameters
        ----------
        model_class
            The ForecastingModel subclass to be tuned for 'series'.
        parameters
            A dictionary containing as keys hyperparameter names, and as values lists of values for the
            respective hyperparameter.
        series
            The TimeSeries instance used as input for training.
        forecast_horizon
            The integer value of the forecasting horizon used in expanding window mode.
        start
            The `int`, `float` or `pandas.Timestamp` that represents the starting point in the time index
            of `series` from which predictions will be made to evaluate the model.
            For a detailed description of how the different data types are interpreted, please see the documentation
            for `ForecastingModel.backtest`.
        use_full_output_length
            This should only be set if `model_class` is equal to `TorchForecastingModel`.
            This argument will be passed along to the predict method of `TorchForecastingModel`.
        val_series
            The TimeSeries instance used for validation in split mode.
        use_fitted_values
            If `True`, uses the comparison with the fitted values.
            Raises an error if `fitted_values` is not an attribute of `model_class`.
        metric:
            A function that takes two TimeSeries instances as inputs and returns a float error value.
        verbose:
            Whether to print progress.

        Returns
        -------
        Tuple[ForecastingModel, Dict]
            An untrained 'model_class' instance, created with the best-performing hyperparameters,
            along with a dictionary containing these hyperparameters.
        """
        raise_if_not((forecast_horizon is not None) + (val_series is not None) + use_fitted_values == 1,
                     "Please pass exactly one of the arguments 'forecast_horizon', "
                     "'val_series' or 'use_fitted_values'.", logger)

        # construct predict kwargs dictionary
        predict_kwargs = {}
        if use_full_output_length is not None:
            predict_kwargs['use_full_output_length'] = use_full_output_length

        if use_fitted_values:
            raise_if_not(hasattr(model_class(), "fitted_values"), "The model must have a fitted_values attribute"
                         " to compare with the train TimeSeries", logger)

        elif val_series is not None:
            raise_if_not(series.width == val_series.width, "Training and validation series require the"
                         " same number of components.", logger)

        min_error = float('inf')
        best_param_combination = {}

        # compute all hyperparameter combinations from selection
        params_cross_product = list(product(*parameters.values()))

        # iterate through all combinations of the provided parameters and choose the best one
        iterator = _build_tqdm_iterator(params_cross_product, verbose)
        for param_combination in iterator:
            param_combination_dict = dict(list(zip(parameters.keys(), param_combination)))
            model = model_class(**param_combination_dict)
            if use_fitted_values:  # fitted value mode
                model.fit(series)
                fitted_values = TimeSeries.from_times_and_values(series.time_index(), model.fitted_values)
                error = metric(fitted_values, series)
            elif val_series is None:  # expanding window mode
                backtest_forecast = model.backtest(series,
                                                   start,
                                                   forecast_horizon,
                                                   use_full_output_length=use_full_output_length)
                error = metric(backtest_forecast, series)
            else:  # split mode
                model.fit(series)
                error = metric(model.predict(len(val_series), **predict_kwargs), val_series)
            if error < min_error:
                min_error = error
                best_param_combination = param_combination_dict
        logger.info('Best performing hyper-parameters: ' + str(best_param_combination))

        return model_class(**best_param_combination), best_param_combination

    def residuals(self,
                  series: TimeSeries,
                  forecast_horizon: int = 1,
                  verbose: bool = False) -> TimeSeries:
        """ A function for computing the residuals produced by the current model on a univariate time series.

        This function computes the difference between the actual observations from `series`
        and the fitted values vector p obtained by training the model on `series`.
        For every index i in `series`, p[i] is computed by training the model on
        series[:(i - `forecast_horizon`)] and forecasting `forecast_horizon` into the future.
        (p[i] will be set to the last value of the predicted vector.)
        The vector of residuals will be shorter than `series` due to the minimum
        training series length required by the model and the gap introduced by `forecast_horizon`.
        Note that the common usage of the term residuals implies a value for `forecast_horizon` of 1.

        Parameters
        ----------
        series
            The univariate TimeSeries instance which the residuals will be computed for.
        forecast_horizon
            The forecasting horizon used to predict each fitted value.
        verbose
            Whether to print progress.
        Returns
        -------
        TimeSeries
            The vector of residuals.
        """
        series._assert_univariate()

        # get first index not contained in the first training set
        first_index = series.time_index()[self.min_train_series_length]

        # compute fitted values
        p = self.backtest(series, first_index, forecast_horizon, 1, True, verbose=verbose)

        # compute residuals
        series_trimmed = series.slice_intersect(p)
        residuals = series_trimmed - p

        return residuals
