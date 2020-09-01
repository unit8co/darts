"""
Forecasting Model Base Class
----------------------------

A forecasting model captures the future values of a time series as a function of the past as follows:

.. math:: y_{t+1} = f(y_t, y_{t-1}, ..., y_1),

where :math:`y_t` represents the time series' value(s) at time :math:`t`.
"""

from typing import Optional, Tuple, Union, Any, Callable
from types import SimpleNamespace
from itertools import product
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not, raise_if
from ..utils import _build_tqdm_iterator, _with_sanity_checks
from .. import metrics

logger = get_logger(__name__)


class ForecastingModel(ABC):
    """ The base class for all forecasting models.

    All implementations of forecasting have to implement the `fit()` and `predict()` methods defined below.

    Attributes
    ----------
    training_series
        reference to the `TimeSeries` used for training the model.
    target_series
        reference to the `TimeSeries` used as target to train the model.
    """
    @abstractmethod
    def __init__(self):
        # Stores training date information:
        self.training_series: TimeSeries = None
        self.target_series: TimeSeries = None

        # state
        self._fit_called = False

    @abstractmethod
    def fit(self) -> None:
        """ Fits/trains the model on the provided series

        Implements behavior that should happen when calling the `fit` method of every forcasting model regardless of
        wether they are univariate or multivariate.
        """
        for series in (self.training_series, self.target_series):
            if series is not None:
                raise_if_not(len(series) >= self.min_train_series_length,
                             "Train series only contains {} elements but {} model requires at least {} entries"
                             .format(len(series), str(self), self.min_train_series_length))
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
        # parse args and kwargs
        if len(args) > 0:
            training_series = args[0]
        else:
            training_series = kwargs['training_series']
        n = SimpleNamespace(**kwargs)

        # check target and training series
        if not hasattr(n, 'target_series'):
            target_series = training_series
        else:
            target_series = n.target_series
        raise_if_not(all(training_series.time_index() == target_series.time_index()), "the target and training series"
                     " must have the same time indices.")

        # check forecast horizon‚
        if hasattr(n, 'forecast_horizon'):
            raise_if_not(n.forecast_horizon > 0, 'The provided forecasting horizon must be a positive integer.', logger)

        # check start parameter
        if hasattr(n, 'start'):
            if isinstance(n.start, float):
                raise_if_not(n.start >= 0.0 and n.start < 1.0, '`start` should be between 0.0 and 1.0.', logger)
            elif isinstance(n.start, pd.Timestamp):
                raise_if_not(n.start in training_series, '`start` timestamp is not in the `series`.', logger)
                raise_if(n.start == training_series.end_time(), '`start` timestamp is the last timestamp of `series`',
                         logger)
            else:
                raise_if(training_series[n.start].time_index()[0] == training_series.end_time(), '`start` timestamp '
                         'is the last timestamp of `series`', logger)

    def _backtest_model_specific_sanity_checks(self, *args: Any, **kwargs: Any) -> None:
        """Method to be overriden in subclass for model specific sanity checks"""
        pass

    @_with_sanity_checks("_backtest_sanity_checks", "_backtest_model_specific_sanity_checks")
    def backtest(self,
                 training_series: TimeSeries,
                 target_series: Optional[TimeSeries] = None,
                 start: Union[pd.Timestamp, float, int] = 0.5,
                 forecast_horizon: int = 1,
                 stride: int = 1,
                 retrain: bool = True,
                 trim_to_series: bool = True,
                 verbose: bool = False,
                 use_full_output_length: Optional[bool] = None) -> Tuple[TimeSeries, TimeSeries]:
        """ Retrain and forecast values pointwise with an expanding training window over `series`.

        To this end, it repeatedly builds a training set from the beginning of `series`. It trains the current model on
        the training set, emits a (point) prediction for a fixed forecast horizon, and then moves the end of the
        training set forward by one time step. The resulting predictions are then returned.

        Unless `retrain` is set to False, this always re-trains the models on the entire available history,
        corresponding an expending window strategy.

        If `retrain` is set to False (useful for models with many parameter such as `TorchForecastingModel` instances),
        the model will only be trained only on the initial training window (up to `start` time stamp), and only if it
        has not been trained before. Then, at every iteration, the newly expanded 'training sequence' will be fed to the
        model to produce the new output.

        Parameters
        ----------
        training_series
            The training time series on which to backtest
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
            instances as `self` argument support setting `retrain` to `False`.
        use_full_output_length
            Optionally, if `self` is an instance of `TorchForecastingModel`, this argument will be passed along
            as argument to the predict method of `self`. Otherwise, if this value is set and `self` is not an
            instance of `TorchForecastingModel`, this will cause an error.
        trim_to_series
            Whether the predicted series has the end trimmed to match the end of the main series
        verbose
            Whether to print progress

        Returns
        -------
        forecast
            A time series containing the forecast values for `target_series`, when successively applying the specified
            model with the specified forecast horizon.
        """
        # handle case where target_series not specified
        if target_series is None:
            target_series = training_series

        # construct predict kwargs dictionary
        predict_kwargs = {}
        if use_full_output_length is not None:
            predict_kwargs['use_full_output_length'] = use_full_output_length

        # construct fit function (used to ignore target series for univariate models)
        if isinstance(self, MultivariateForecastingModel):
            fit_function = self.fit
        else:
            fit_function = lambda train, target, **kwargs: self.fit(train, **kwargs)  # noqa: E731

        # prepare the start parameter -> pd.Timestamp
        if isinstance(start, float):
            start_index = int((len(training_series.time_index()) - 1) * start)
            start = training_series.time_index()[start_index]
        elif isinstance(start, int):
            start = training_series[start].time_index()[0]

        # build the prediction times in advance (to be able to use tqdm)
        if trim_to_series:
            last_pred_time = training_series.time_index()[-forecast_horizon - stride]
        else:
            last_pred_time = training_series.time_index()[-stride - 1]

        pred_times = [start]
        while pred_times[-1] <= last_pred_time:
            pred_times.append(pred_times[-1] + training_series.freq() * stride)

        # iterate and predict pointwise
        values = []
        times = []

        iterator = _build_tqdm_iterator(pred_times, verbose)

        if not retrain and not self._fit_called:
            fit_function(training_series.drop_after(start), target_series.drop_after(start), verbose=verbose)

        for pred_time in iterator:
            train = training_series.drop_after(pred_time)  # build the training series
            target = target_series.drop_after(pred_time)  # build the target series
            if (retrain):
                fit_function(train, target)
                pred = self.predict(forecast_horizon, **predict_kwargs)
            else:
                pred = self.predict(forecast_horizon, input_series=train, **predict_kwargs)
            values.append(pred.values()[-1])  # store the N-th point
            times.append(pred.end_time())  # store the N-th timestamp

        forecast = TimeSeries.from_times_and_values(pd.DatetimeIndex(times), np.array(values))

        return forecast

    @classmethod
    def gridsearch(model_class,
                   parameters: dict,
                   training_series: TimeSeries,
                   target_series: TimeSeries = None,
                   fcast_horizon_n: Optional[int] = None,
                   use_full_output_length: Optional[bool] = None,
                   val_target_series: Optional[TimeSeries] = None,
                   num_predictions: int = 10,
                   use_fitted_values: bool = False,
                   metric: Callable[[TimeSeries, TimeSeries], float] = metrics.mape,
                   verbose=False):
        """ A function for finding the best hyperparameters.

        This function has 3 modes of operation: Expanding window mode, split mode and fitted value mode.
        The three modes of operation evaluate every possible combination of hyperparameter values
        provided in the `parameters` dictionary by instantiating the `model_class` subclass
        of ForecastingModel with each combination, and returning the best-performing model with regards
        to the `metric` function. The `metric` function is expected to return an error value,
        thus the model resulting in the smallest `metric` output will be chosen.
        The relationship of the training data and test data depends on the mode of operation.

        Expanding window mode (activated when `fcast_horizon_n` is passed):
        For every hyperparameter combination, the model is repeatedly trained and evaluated on different
        splits of `training_series` and `target_series`. The number of splits is equal to `num_predictions`, and the
        forecasting horizon used when making a prediction is `fcast_horizon_n`.
        Note that the model is retrained for every single prediction, thus this mode is slower.

        Split window mode (activated when `val_series` is passed):
        This mode will be used when the `val_series` argument is passed.
        For every hyperparameter combination, the model is trained on `training_series` + `target_series` and
        evaluated on `val_series`.

        Fitted value mode (activated when `use_fitted_values` is set to `True`):
        For every hyperparameter combination, the model is trained on `training_series` + `target_series`
        and evaluated on the resulting fitted values.
        Not all models have fitted values, and this method raises an error if `model.fitted_values` does not exist.
        The fitted values are the result of the fit of the model on the training series. Comparing with the
        fitted values can be a quick way to assess the model, but one cannot see if the model overfits or underfits.


        Parameters
        ----------
        model_class
            The ForecastingModel subclass to be tuned for 'series'.
        parameters
            A dictionary containing as keys hyperparameter names, and as values lists of values for the
            respective hyperparameter.
        training_series
            The TimeSeries instance used as input for training.
        target_series
            The TimeSeries instance used as target for training (and also validation in expanding window mode).
        fcast_horizon_n
            The integer value of the forecasting horizon used in expanding window mode.
        use_full_output_length
            In case `model` is a subclass of `TorchForecastingModel`, this argument will be passed along
            as argument to the predict method of `model`.
        val_target_series
            The TimeSeries instance used for validation in split mode.
        num_predictions:
            The number of train/prediction cycles performed in one iteration of expanding window mode.
        use_fitted_values
            If `True`, uses the comparison with the fitted values.
            Raises an error if `fitted_values` is not an attribute of `model_class`.
        metric:
            A function that takes two TimeSeries instances as inputs and returns a float error value.
        verbose:
            Whether to print progress.

        Returns
        -------
        ForecastingModel
            An untrained 'model_class' instance with the best-performing hyperparameters from the given selection.
        """
        raise_if_not((fcast_horizon_n is not None) + (val_target_series is not None) + use_fitted_values == 1,
                     "Please pass exactly one of the arguments 'forecast_horizon_n', "
                     "'val_target_series' or 'use_fitted_values'.", logger)

        # check target and training series
        if target_series is None:
            target_series = training_series
        raise_if_not(all(training_series.time_index() == target_series.time_index()), "the target and training series"
                     " must have the same time indices.")

        # construct predict kwargs dictionary
        predict_kwargs = {}
        if use_full_output_length is not None:
            predict_kwargs['use_full_output_length'] = use_full_output_length

        if use_fitted_values:
            model = model_class()
            raise_if_not(hasattr(model, "fitted_values"), "The model must have a fitted_values attribute"
                         " to compare with the train TimeSeries", logger)

        elif val_target_series is not None:
            raise_if_not(training_series.width == val_target_series.width, "Training and validation series require the"
                         " same number of components.", logger)

        if (val_target_series is None) and (not use_fitted_values):
            backtest_start_time = (
                training_series.end_time() - (num_predictions + fcast_horizon_n) * training_series.freq()
            )
        min_error = float('inf')
        best_param_combination = {}

        # compute all hyperparameter combinations from selection
        params_cross_product = list(product(*parameters.values()))

        # iterate through all combinations of the provided parameters and choose the best one
        iterator = _build_tqdm_iterator(params_cross_product, verbose)
        for param_combination in iterator:
            param_combination_dict = dict(list(zip(parameters.keys(), param_combination)))
            model = model_class(**param_combination_dict)
            if use_fitted_values:
                model.fit(training_series)
                # Takes too much time to create a TimeSeries
                # Overhead: 2-10 ms in average
                fitted_values = TimeSeries.from_times_and_values(training_series.time_index(), model.fitted_values)
                error = metric(fitted_values, target_series)
            elif val_target_series is None:  # expanding window mode
                backtest_forecast = model.backtest(training_series, target_series, backtest_start_time,
                                                   fcast_horizon_n, use_full_output_length=use_full_output_length)
                error = metric(backtest_forecast, target_series)
            else:  # split mode
                if isinstance(model, MultivariateForecastingModel):
                    model.fit(training_series, target_series)
                else:
                    model.fit(training_series)
                error = metric(model.predict(len(val_target_series), **predict_kwargs), val_target_series)
            if error < min_error:
                min_error = error
                best_param_combination = param_combination_dict
        logger.info('Chosen parameters: ' + str(best_param_combination))
        return model_class(**best_param_combination)

    def residuals(self,
                  series: TimeSeries,
                  fcast_horizon_n: int = 1,
                  verbose: bool = False) -> TimeSeries:
        """ A function for computing the residuals produced by the current model and a univariate time series.

        This function computes the difference between the actual observations from `series`
        and the fitted values vector p obtained by training `self` on `series`.
        For every index i in `series`, p[i] is computed by training `self` on
        series[:(i - `fcast_horizon_n`)] and forecasting `fcast_horizon_n` into the future.
        (p[i] will be set to the last value of the predicted vector.)
        The vector of residuals will be shorter than `series` due to the minimum
        training series length required by `self` and the gap introduced by `fcast_horizon_n`.
        Note that the common usage of the term residuals implies a value for `fcast_horizon_n` of 1.

        Parameters
        ----------
        series
            The univariate TimeSeries instance which the residuals will be computed for.
        fcast_horizon_n
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
        p = self.backtest(series, None, first_index, fcast_horizon_n, 1, True, verbose=verbose)

        # compute residuals
        series_trimmed = series.slice_intersect(p)
        residuals = series_trimmed - p

        return residuals


class UnivariateForecastingModel(ForecastingModel):
    """The base class for univariate forecasting models."""
    @abstractmethod
    def fit(self, series: TimeSeries) -> None:
        """ Fits/trains the univariate model on selected univariate series.

        Implements behavior specific to calling the `fit` method on `UnivariateForecastingModel`.

        Parameters
        ----------
        series
            A **univariate** timeseries on which to fit the model.
        """
        series._assert_univariate()
        self.training_series = series
        self.target_series = series
        super().fit()


class MultivariateForecastingModel(ForecastingModel):
    """ The base class for multivariate forecasting models.
    """
    def _make_fitable_series(self,
                             training_series: TimeSeries,
                             target_series: Optional[TimeSeries] = None) -> Tuple[TimeSeries, TimeSeries]:
        """Perform checks and returns ready to be used training and target series"""
        if target_series is None:
            target_series = training_series

        # general checks on training / target series
        raise_if_not(all(training_series.time_index() == target_series.time_index()), "training and target "
                     "timeseries must have same time indices.")

        return training_series, target_series

    @abstractmethod
    def fit(self, training_series: TimeSeries, target_series: Optional[TimeSeries] = None) -> None:
        """ Fits/trains the multivariate model on the provided series with selected target components.

        Parameters
        ----------
        training_series
            The training time series on which to fit the model (can be multivariate or univariate).
        target_series
            The target values used as dependent variables when training the model
        """
        training_series, target_series = self._make_fitable_series(training_series, target_series)

        self.training_series = training_series
        self.target_series = target_series
        super().fit()
