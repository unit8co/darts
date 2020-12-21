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


from typing import Optional, Tuple, Union, Any, Callable, Dict, List, Sequence
from itertools import product
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not
from ..utils import (
    _build_tqdm_iterator,
    _with_sanity_checks,
    get_timestamp_at_point,
    _historical_forecasts_general_checks
)
from .. import metrics

logger = get_logger(__name__)


class ForecastingModel(ABC):
    """ The base class for forecasting models. It defines the *minimal* behavior that all
        forecasting models have to support. Therefore the signatures are for "local" models
        handling only one series and not covariates. Sub-classes can handle more complex cases.

    Attributes
    ----------
    training_series
        Reference to the `TimeSeries` used for training the model through the `fit()` function.
        This is only used if the model has been fit on one time series.
    """
    @abstractmethod
    def __init__(self):
        # Stores training date information:
        self.training_series: Optional[TimeSeries] = None

        # state; whether the model has been fit (on a single time series)
        self._fit_called = False

    @abstractmethod
    def fit(self, series: TimeSeries) -> None:
        """ Fits/trains the model on the provided series

        Parameters
        ----------
        series
            A target time series. The model will be trained to forecast this time series.
        """
        raise_if_not(len(series) >= self.min_train_series_length,
                     "Train series only contains {} elements but {} model requires at least {} entries"
                     .format(len(series), str(self), self.min_train_series_length))
        self.training_series = series
        self._fit_called = True

    @abstractmethod
    def predict(self, n: int) -> TimeSeries:
        """ Forecasts values for `n` time steps after the end of the series.

        Parameters
        ----------
        n
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.

        Returns
        -------
        TimeSeries
            A time series containing the `n` next points after then end of the training series.
        """
        if not self._fit_called:
            raise_log(Exception('The model must be fit before calling `predict()`.'
                                'For global models, if `predict()` is called without specifying a series,'
                                'the model must have been fit on a single training series.'), logger)

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
        Generates `n` new dates after the end of the specified series
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
        # parse args and kwargs
        series = args[0]
        _historical_forecasts_general_checks(series, kwargs)

    @_with_sanity_checks("_historical_forecasts_sanity_checks")
    def historical_forecasts(self,
                             series: TimeSeries,
                             start: Union[pd.Timestamp, float, int] = 0.5,
                             forecast_horizon: int = 1,
                             stride: int = 1,
                             retrain: bool = True,
                             overlap_end: bool = False,
                             last_points_only: bool = True,
                             verbose: bool = False) -> Union[TimeSeries, List[TimeSeries]]:

        """
        Computes the historical forecasts the model would have produced with an expanding training window
        and (by default) returns a time series created from the last point of each of these individual forecasts
        To this end, it repeatedly builds a training set from the beginning of `series`. It trains the
        current model on the training set, emits a forecast of length equal to forecast_horizon, and then moves
        the end of the training set forward by `stride` time steps.
        By default, this method will return a single time series made up of the last point of each
        historical forecast. This time series will thus have a frequency of `series.freq() * stride`.
        If `last_points_only` is set to False, it will instead return a list of the historical forecasts.
        By default, this method always re-trains the models on the entire available history,
        corresponding to an expanding window strategy.
        If `retrain` is set to False (useful for models with many parameter such as `TorchForecastingModel` instances
        like `RNNModel` and `TCNModel`), the model will only be trained on the initial training window
        (up to `start` time stamp), and only if it has not been trained before. Then, at every iteration, the
        newly expanded input sequence will be fed to the model to produce the new output.

        This method does not currently support covariates.

        Parameters
        ----------
        series
            The training time series to use to compute and evaluate the historical forecasts
        start
            The first point at which a prediction is computed for a future time.
            This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
            In the case of `float`, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of `int`, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of `pandas.Timestamp`, this time stamp will be used to determine the first prediction time
            directly.
        forecast_horizon
            The forecast horizon for the point predictions
        stride
            The number of time steps between two consecutive predictions.
        retrain
            Whether to retrain the model for every prediction or not. Currently only `TorchForecastingModel`
            instances such as `RNNModel` and `TCNModel` support setting `retrain` to `False`.
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
            If `last_points_only` is set to False, a list of the historical forecasts.
        """

        # construct predict kwargs dictionary
        predict_kwargs = {}

        # prepare the start parameter -> pd.Timestamp
        start = get_timestamp_at_point(start, series)

        # build the prediction times in advance (to be able to use tqdm)
        if not overlap_end:
            last_valid_pred_time = series.time_index()[-1 - forecast_horizon]
        else:
            last_valid_pred_time = series.time_index()[-2]

        pred_times = [start]
        while pred_times[-1] < last_valid_pred_time:
            # compute the next prediction time and add it to pred times
            pred_times.append(pred_times[-1] + series.freq() * stride)

        # the last prediction time computed might have overshot last_valid_pred_time
        if pred_times[-1] > last_valid_pred_time:
            pred_times.pop(-1)

        iterator = _build_tqdm_iterator(pred_times, verbose)

        # Either store the whole forecasts or only the last points of each forecast, depending on last_points_only
        forecasts = []

        last_points_times = []
        last_points_values = []

        # iterate and forecast
        for pred_time in iterator:
            train = series.drop_after(pred_time)  # build the training series

            if retrain:
                self.fit(train)
                forecast = self.predict(forecast_horizon, **predict_kwargs)
            else:
                # TODO: remove this case (which can fail for non-torch models)
                # TODO: and implement dedicated backtest() method for torch/ML models
                forecast = self.predict(forecast_horizon, series=train, **predict_kwargs)

            if last_points_only:
                last_points_values.append(forecast.values()[-1])
                last_points_times.append(forecast.end_time())
            else:
                forecasts.append(forecast)

        if last_points_only:
            return TimeSeries.from_times_and_values(pd.DatetimeIndex(last_points_times),
                                                    np.array(last_points_values),
                                                    freq=series.freq() * stride)
        return forecasts

    def backtest(self,
                 series: TimeSeries,
                 start: Union[pd.Timestamp, float, int] = 0.5,
                 forecast_horizon: int = 1,
                 stride: int = 1,
                 retrain: bool = True,
                 overlap_end: bool = False,
                 last_points_only: bool = False,
                 metric: Callable[[TimeSeries, TimeSeries], float] = metrics.mape,
                 reduction: Union[Callable[[np.ndarray], float], None] = np.mean,
                 use_full_target_length: Optional[bool] = None,
                 verbose: bool = False) -> Union[float, List[float]]:

        """
        Computes an error score between the historical forecasts the model would have produced
        with an expanding training window over `series` and the actual series.
        To this end, it repeatedly builds a training set from the beginning of `series`. It trains the current model on
        the training set, emits a forecast of length equal to forecast_horizon, and then moves the end of the
        training set forward by `stride` time steps.
        By default, this method will use each historical forecast (whole) to compute error scores.
        If `last_points_only` is set to True, it will use only the last point of each historical forecast.
        By default, this method always re-trains the models on the entire available history,
        corresponding to an expanding window strategy.
        If `retrain` is set to False (useful for models with many parameter such as `TorchForecastingModel` instances
        like `RNNModel` and `TCNModel`), the model will only be trained on the initial training window
        (up to `start` time stamp), and only if it has not been trained before. Then, at every iteration, the
        newly expanded input sequence will be fed to the model to produce the new output.

        This method does not currently support covariates.

        Parameters
        ----------
        series
            The training time series to use to compute and evaluate the historical forecasts
        start
            The first prediction time, at which a prediction is computed for a future time.
            This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
            In the case of `float`, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of `int`, the parameter will be treated as an integer index to the time index of
            `training_series` that will be used as first prediction time.
            In case of `pandas.Timestamp`, this time stamp will be used to determine the first prediction time
            directly.
        forecast_horizon
            The forecast horizon for the point prediction
        stride
            The number of time steps (the unit being the frequency of `series`) between two consecutive predictions.
        retrain
            Whether to retrain the model for every prediction or not. Currently only `TorchForecastingModel`
            instances such as `RNNModel` and `TCNModel` support setting `retrain` to `False`.
        overlap_end
            Whether the returned forecasts can go beyond the series' end or not
        last_points_only
            Whether to use the whole historical forecasts or only the last point of each forecast to compute the error
        metric
            A function that takes two TimeSeries instances as inputs and returns a float error value.
        reduction
            A function used to combine the individual error scores obtained when `last_points_only` is set to False.
            If explicitely set to `None`, the method will return a list of the individual error scores instead.
            Set to np.mean by default.
        use_full_target_length
            Optionally, if the model is an instance of `TorchForecastingModel`, this argument will be passed along
            as argument to the `predict` method of the model. Otherwise, if this value is set and the model is not an
            instance of `TorchForecastingModel`, this will cause an error.
        verbose
            Whether to print progress
        Returns
        -------
        float or List[float]
            The error score, or the list of individual error scores if `reduction` is `None`
        """
        forecasts = self.historical_forecasts(series,
                                              start,
                                              forecast_horizon,
                                              stride,
                                              retrain,
                                              overlap_end,
                                              last_points_only,
                                              verbose,
                                              use_full_target_length)

        if last_points_only:
            return metric(series, forecasts)

        errors = [metric(series, forecast) for forecast in forecasts]

        if reduction is None:
            return errors

        return reduction(errors)

    @classmethod
    def gridsearch(model_class,
                   parameters: dict,
                   series: TimeSeries,
                   forecast_horizon: Optional[int] = None,
                   start: Union[pd.Timestamp, float, int] = 0.5,
                   last_points_only: bool = False,
                   use_full_target_length: Optional[bool] = None,
                   val_series: Optional[TimeSeries] = None,
                   use_fitted_values: bool = False,
                   metric: Callable[[TimeSeries, TimeSeries], float] = metrics.mape,
                   reduction: Callable[[np.ndarray], float] = np.mean,
                   verbose=False) -> Tuple['ForecastingModel', Dict]:
        """
        A function for finding the best hyperparameters.
        This function has 3 modes of operation: Expanding window mode, split mode and fitted value mode.
        The three modes of operation evaluate every possible combination of hyperparameter values
        provided in the `parameters` dictionary by instantiating the `model_class` subclass
        of ForecastingModel with each combination, and returning the best-performing model with regards
        to the `metric` function. The `metric` function is expected to return an error value,
        thus the model resulting in the smallest `metric` output will be chosen.
        The relationship of the training data and test data depends on the mode of operation.
        Expanding window mode (activated when `forecast_horizon` is passed):
        For every hyperparameter combination, the model is repeatedly trained and evaluated on different
        splits of `training_series` and `target_series`. This process is accomplished by using
        `ForecastingModel.backtest` as a subroutine to produce historic forecasts starting from `start`
        that are compared against the ground truth values of `training_series` or `target_series`, if
        specifed.
        Note that the model is retrained for every single prediction, thus this mode is slower.
        Split window mode (activated when `val_series` is passed):
        This mode will be used when the `val_series` argument is passed.
        For every hyperparameter combination, the model is trained on `series` and
        evaluated on `val_series`.
        Fitted value mode (activated when `use_fitted_values` is set to `True`):
        For every hyperparameter combination, the model is trained on `series`
        and evaluated on the resulting fitted values.
        Not all models have fitted values, and this method raises an error if `model.fitted_values` does not exist.
        The fitted values are the result of the fit of the model on `series`. Comparing with the
        fitted values can be a quick way to assess the model, but one cannot see if the model overfits or underfits.

        This method does not currently support covariates.

        Parameters
        ----------
        model_class
            The ForecastingModel subclass to be tuned for 'series'.
        parameters
            A dictionary containing as keys hyperparameter names, and as values lists of values for the
            respective hyperparameter.
        series
            The TimeSeries instance used as input and target for training.
        forecast_horizon
            The integer value of the forecasting horizon used in expanding window mode.
        start
            The `int`, `float` or `pandas.Timestamp` that represents the starting point in the time index
            of `training_series` from which predictions will be made to evaluate the model.
            For a detailed description of how the different data types are interpreted, please see the documentation
            for `ForecastingModel.backtest`.
        last_points_only
            Whether to use the whole forecasts or only the last point of each forecast to compute the error
        use_full_target_length
            This should only be set if `model_class` is equal to `TorchForecastingModel`.
            This argument will be passed along to the predict method of `TorchForecastingModel`.
        val_series
            The TimeSeries instance used for validation in split mode. If provided, this series must start right after
            the end of `series`; so that a proper comparison of the forecast can be made.
        use_fitted_values
            If `True`, uses the comparison with the fitted values.
            Raises an error if `fitted_values` is not an attribute of `model_class`.
        metric
            A function that takes two TimeSeries instances as inputs and returns a float error value.
        reduction
            A reduction function (mapping array to float) describing how to aggregate the errors obtained
            on the different validation series when backtesting. By default it'll compute the mean of errors.
        verbose
            Whether to print progress.

        Returns
        -------
        ForecastingModel, Dict
            A tuple containing an untrained 'model_class' instance created from the best-performing hyper-parameters,
            along with a dictionary containing these best hyper-parameters.
        """
        raise_if_not((forecast_horizon is not None) + (val_series is not None) + use_fitted_values == 1,
                     "Please pass exactly one of the arguments 'forecast_horizon', "
                     "'val_target_series' or 'use_fitted_values'.", logger)

        # construct predict kwargs dictionary
        predict_kwargs = {}
        if use_full_target_length is not None:
            predict_kwargs['use_full_target_length'] = use_full_target_length

        if use_fitted_values:
            raise_if_not(hasattr(model_class(), "fitted_values"),
                         "The model must have a fitted_values attribute to compare with the train TimeSeries",
                         logger)

        elif val_series is not None:
            raise_if_not(series.width == val_series.width,
                         "Training and validation series require the same number of components.",
                         logger)

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
                error = model.backtest(series,
                                       start,
                                       forecast_horizon,
                                       metric=metric,
                                       reduction=reduction,
                                       last_points_only=last_points_only,
                                       use_full_target_length=use_full_target_length)
            else:  # split mode
                model.fit(series)
                error = metric(model.predict(len(val_series), **predict_kwargs), val_series)
            if error < min_error:
                min_error = error
                best_param_combination = param_combination_dict
        logger.info('Chosen parameters: ' + str(best_param_combination))

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

        This method does not currently support covariates.

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
        p = self.historical_forecasts(series,
                                      first_index,
                                      forecast_horizon,
                                      1,
                                      True,
                                      last_points_only=True,
                                      verbose=verbose)

        # compute residuals
        series_trimmed = series.slice_intersect(p)
        residuals = series_trimmed - p

        return residuals


class GlobalForecastingModel(ForecastingModel, ABC):
    """ The base class for "global" forecasting models, handling several time series and optional covariates.

    All implementations have to implement the `fit()` and `predict()` methods defined below.
    The `fit()` method is meant to train the model on one or several training time series, along with optional
    covariates. Note that not all global models support covariates.

    If `fit()` has been called with only one training series as argument, then calling `predict()` will
    forecast the future of this series. Otherwise, the user has to provide to `predict()` the series they want
    to forecast, as well as covariates if needed.
    """

    _expect_covariates = False

    @abstractmethod
    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None
            ) -> None:
        """ Fits/trains the model on the provided series

        Defines behavior that should happen when calling the `fit()` method of every forecasting model.

        Some models support training only on one time series, while others can handle a sequence.
        Similarly, some models can handle covariates.

        Some covariates are know in the future, and others aren't. This is a property of the `TimeSeries`, which
        may or may not be exploited by the models.

        Parameters
        ----------
        series
            One or several target time series. The model will be trained to forecast these time series.
        covariates
            One or several covariate time series. These time series will not be forecast, but can be used by
            some models as an input. Some of these covariates may represent forecasts known in advance. This knowledge
            is a property of the `TimeSeries`.
        """
        if isinstance(series, TimeSeries) and covariates is None:
            super().fit(series)  # handle the single series case
        if covariates is not None:
            self._expect_covariates = True


    @abstractmethod
    def predict(self,
                n: int,
                series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None
                ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """ Forecasts values for a certain number of time steps after the end of the series.

        If `fit()` has been called with only one `TimeSeries` as argument, then the `series` argument of this function
        is optional, and it will simply produce the next `horizon` time steps forecast.

        If `fit()` has been called with `series` specified as a `Sequence[TimeSeries]`, the `series` argument must
        be specified.

        When the `series` argument is specified, this function will compute the next `n` time steps forecasts
        for the simple series (or for each series in the sequence) given by `series`.

        If covariates were specified during the training, they must also be specified here.

        Parameters
        ----------
        n
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        series
            The series whose future we want to predict
        covariates
            One or several covariate time series which can be fed as inputs to the model. They must match the
            covariates that have been used with the `fit()` function for training.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            If `series` is not specified, this function returns a single time series containing the `n`
            next points after then end of the training series.
            If `series` is specified and is a simple `TimeSeries`, this function returns the `n` next points
            after the end of `series`.
            If `series` is a sequence of several time series, this function returns a sequence where each element
            contains the corresponding `n` points forecasts.
        """
        if series is None and covariates is None:
            super().predict(n)
        if self._expect_covariates and covariates is None:
            raise_log(ValueError('The model has been trained with covariates. Some matching covariates '
                                 'have to be provided to `predict()`.'))
