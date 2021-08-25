"""
Forecasting Model Base Classes

A forecasting model captures the future values of a time series as a function of the past as follows:

.. math:: y_{t+1} = f(y_t, y_{t-1}, ..., y_1),

where :math:`y_t` represents the time series' value(s) at time :math:`t`.

The main functions are `fit()` and `predict()`. `fit()` learns the function `f()`, over the history of
one or several time series. The function `predict()` applies `f()` on one or several time series in order
to obtain forecasts for a desired number of time stamps into the future.
"""


from typing import Optional, Tuple, Union, Any, Callable, Dict, List, Sequence
from itertools import product
from abc import ABC, abstractmethod
from inspect import signature
import numpy as np
import pandas as pd
import xarray as xr

from ..timeseries import TimeSeries
from ..logging import get_logger, raise_log, raise_if_not, raise_if
from ..utils import (
    _build_tqdm_iterator,
    _with_sanity_checks,
    _historical_forecasts_general_checks,
    _parallel_apply
)

from .. import metrics

logger = get_logger(__name__)


class ForecastingModel(ABC):
    """ The base class for forecasting models. It defines the *minimal* behavior that all forecasting models have to support.
        The signatures in this base class are for "local" models handling only one univariate series and no covariates.
        Sub-classes can handle more complex cases.
    """

    @abstractmethod
    def __init__(self):
        # The series used for training the model through the `fit()` function.
        # This is only used if the model has been fit on one time series.
        self.training_series: Optional[TimeSeries] = None

        # state; whether the model has been fit (on a single time series)
        self._fit_called = False

    @abstractmethod
    def fit(self, series: TimeSeries) -> None:
        """ Trains the model on the provided univariate series.

        Parameters
        ----------
        series
            A target time series. The model will be trained to forecast this time series.
        """
        if not isinstance(self, DualCovariatesForecastingModel):
            series._assert_univariate()
        raise_if_not(len(series) >= self.min_train_series_length,
                     "Train series only contains {} elements but {} model requires at least {} entries"
                     .format(len(series), str(self), self.min_train_series_length))
        self.training_series = series
        self._fit_called = True

        if series.has_range_index:
            self._supports_range_index()

    def _supports_range_index(self) -> bool:
        """ Checks if the forecasting model supports a range index.
        Some models may not support this, if for instance the rely on underlying dates.

        By default, returns True. Needs to be overwritten by models that do not support
        range indexing and raise meaningful exception.
        """
        return True

    def _is_probabilistic(self) -> bool:
        """
        Checks if the forecasting model supports probabilistic predictions.
        By default, returns False. Needs to be overwritten by models that do support
        probabilistic predictions.
        """
        return False

    @abstractmethod
    def predict(self,
                n: int,
                num_samples: int = 1) -> TimeSeries:
        """ Forecasts values for `n` time steps after the end of the series.

        Parameters
        ----------
        n
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.

        Returns
        -------
        TimeSeries
            A time series containing the `n` next points after then end of the training series.
        """
        if not self._fit_called:
            raise_log(ValueError('The model must be fit before calling `predict()`.'
                                 'For global models, if `predict()` is called without specifying a series,'
                                 'the model must have been fit on a single training series.'), logger)

        if not self._is_probabilistic() and num_samples > 1:
            raise_log(ValueError('`num_samples > 1` is only supported for probabilistic models.'), logger)

    def _fit_wrapper(self, series: TimeSeries,
                     past_covariates: Optional[TimeSeries],
                     future_covariates: Optional[TimeSeries]):
        self.fit(series)

    def _predict_wrapper(self, n: int, series: TimeSeries,
                         past_covariates: Optional[TimeSeries],
                         future_covariates: Optional[TimeSeries],
                         num_samples: int) -> TimeSeries:
        return self.predict(n, num_samples=num_samples)

    @property
    def min_train_series_length(self) -> int:
        """
        Class property defining the minimum required length for the training series.
        This function/property should be overridden if a value higher than 3 is required.
        """
        return 3

    def _generate_new_dates(self,
                            n: int,
                            input_series: Optional[TimeSeries] = None) -> Union[pd.DatetimeIndex, pd.RangeIndex]:
        """
        Generates `n` new dates after the end of the specified series
        """
        input_series = input_series if input_series is not None else self.training_series

        if input_series.has_datetime_index:
            time_index = input_series.time_index
            new_dates = [
                (time_index[-1] + (i * input_series.freq)) for i in range(1, n + 1)
            ]
            return pd.DatetimeIndex(new_dates, freq=input_series.freq_str)
        else:
            return pd.RangeIndex(start=input_series.end_time() + 1, stop=input_series.end_time() + n + 1, step=1)

    def _build_forecast_series(self,
                               points_preds: Union[np.ndarray, Sequence[np.ndarray]],
                               input_series: Optional[TimeSeries] = None) -> TimeSeries:
        """
        Builds a forecast time series starting after the end of the training time series, with the
        correct time index (or after the end of the input series, if specified).
        """
        input_series = input_series if input_series is not None else self.training_series
        time_index_length = len(points_preds) if isinstance(points_preds, np.ndarray) else len(points_preds[0])
        time_index = self._generate_new_dates(time_index_length, input_series=input_series)
        if isinstance(points_preds, np.ndarray):
            return TimeSeries.from_times_and_values(time_index, points_preds, freq=input_series.freq_str, columns=input_series.columns)

        return TimeSeries.from_times_and_values(time_index, np.stack(points_preds, axis=2),
                                                freq=input_series.freq_str, columns=input_series.columns)

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

    def _get_last_prediction_time(self, series, forecast_horizon, overlap_end):
        if overlap_end:
            last_valid_pred_time = series.time_index[-1]
        else:
            last_valid_pred_time = series.time_index[-forecast_horizon]

        return last_valid_pred_time

    @_with_sanity_checks("_historical_forecasts_sanity_checks")
    def historical_forecasts(self,
                             series: TimeSeries,
                             past_covariates: Optional[TimeSeries] = None,
                             future_covariates: Optional[TimeSeries] = None,
                             num_samples: int = 1,
                             start: Union[pd.Timestamp, float, int] = 0.5,
                             forecast_horizon: int = 1,
                             stride: int = 1,
                             retrain: bool = True,
                             overlap_end: bool = False,
                             last_points_only: bool = True,
                             verbose: bool = False) -> Union[TimeSeries, List[TimeSeries]]:

        """
        Computes the historical forecasts the model would have produced with an expanding training window
        and (by default) returns a time series created from the last point of each of these individual forecasts.
        To this end, it repeatedly builds a training set from the beginning of `series`. It trains the
        current model on the training set, emits a forecast of length equal to forecast_horizon, and then moves
        the end of the training set forward by `stride` time steps.

        By default, this method will return a single time series made up of the last point of each
        historical forecast. This time series will thus have a frequency of `series.freq * stride`.
        If `last_points_only` is set to False, it will instead return a list of the historical forecasts.

        By default, this method always re-trains the models on the entire available history,
        corresponding to an expanding window strategy.
        If `retrain` is set to False (useful for models for which training might be time-consuming, such as
        deep learning models), the model will only be trained on the initial training window
        (up to `start` time stamp), and only if it has not been trained before. Then, at every iteration, the
        newly expanded input sequence will be fed to the model to produce the new output.

        Parameters
        ----------
        series
            The target time series to use to successively train and evaluate the historical forecasts.
        past_covariates
            An optional past-observed covariate series. This applies only if the model supports past covariates.
        future_covariates
            An optional future-known covariate series. This applies only if the model supports future covariates.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.
        start
            The first point of time at which a prediction is computed for a future time.
            This parameter supports 3 different data types: `float`, `int` and `pandas.Timestamp`.
            In the case of `float`, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of `int`, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of `pandas.Timestamp`, this time stamp will be used to determine the first prediction time
            directly.
        forecast_horizon
            The forecast horizon for the predictions
        stride
            The number of time steps between two consecutive predictions.
        retrain
            Whether to retrain the model for every prediction or not. Currently only `TorchForecastingModel`
            instances such as `BlockRNNModel`, `RNNModel`, `TCNModel`, `NBEATSModel` and `TransformerModel` support
            setting `retrain` to `False`.
        overlap_end
            Whether the returned forecasts can go beyond the series' end or not
        last_points_only
            Whether to retain only the last point of each historical forecast.
            If set to True, the method returns a single `TimeSeries` containing the successive point forecasts.
            Otherwise returns a list of historical `TimeSeries` forecasts.
        verbose
            Whether to print progress
        Returns
        -------
        TimeSeries or List[TimeSeries]
            By default, a single TimeSeries instance created from the last point of each individual forecast.
            If `last_points_only` is set to False, a list of the historical forecasts.
        """

        # TODO: do we need a check here? I'd rather leave these checks to the models/datasets.
        # if covariates:
        #     raise_if_not(series.end_time() <= covariates.end_time() and covariates.start_time() <= series.start_time(),
        #                  'The provided covariates must be at least as long as the target series.')

        # prepare the start parameter -> pd.Timestamp
        start = series.get_timestamp_at_point(start)

        # build the prediction times in advance (to be able to use tqdm)
        last_valid_pred_time = self._get_last_prediction_time(series, forecast_horizon, overlap_end)

        pred_times = [start]
        while pred_times[-1] < last_valid_pred_time:
            # compute the next prediction time and add it to pred times
            pred_times.append(pred_times[-1] + series.freq * stride)

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

            # train_cov = covariates.drop_after(pred_time) if covariates else None

            if retrain:
                self._fit_wrapper(series=train, past_covariates=past_covariates, future_covariates=future_covariates)

            forecast = self._predict_wrapper(n=forecast_horizon,
                                             series=train,
                                             past_covariates=past_covariates,
                                             future_covariates=future_covariates,
                                             num_samples=num_samples)

            if last_points_only:
                last_points_values.append(forecast.all_values()[-1])
                last_points_times.append(forecast.end_time())
            else:
                forecasts.append(forecast)

        if last_points_only:
            if series.has_datetime_index:
                return TimeSeries.from_times_and_values(pd.DatetimeIndex(last_points_times, freq=series.freq * stride),
                                                        np.array(last_points_values))
            else:
                return TimeSeries.from_times_and_values(pd.RangeIndex(start=last_points_times[0],
                                                                      stop=last_points_times[-1] + 1,
                                                                      step=1),
                                                        np.array(last_points_values))


        return forecasts

    def backtest(self,
                 series: TimeSeries,
                 past_covariates: Optional[TimeSeries] = None,
                 future_covariates: Optional[TimeSeries] = None,
                 num_samples: int = 1,
                 start: Union[pd.Timestamp, float, int] = 0.5,
                 forecast_horizon: int = 1,
                 stride: int = 1,
                 retrain: bool = True,
                 overlap_end: bool = False,
                 last_points_only: bool = False,
                 metric: Callable[[TimeSeries, TimeSeries], float] = metrics.mape,
                 reduction: Union[Callable[[np.ndarray], float], None] = np.mean,
                 verbose: bool = False) -> Union[float, List[float]]:

        """
        Computes error scores between the historical forecasts the model would have produced
        with an expanding training window over `series` and the actual values.
        To this end, it repeatedly builds a training set from the beginning of `series`. It trains the current model on
        the training set, emits a forecast of length equal to forecast_horizon, and then moves the end of the
        training set forward by `stride` time steps. A metric (given by the `metric` function) is then evaluated
        on the forecast and the actual values. Finally, the method returns a `reduction` (the mean by default)
        of all these metric scores.

        By default, this method will use each historical forecast (whole) to compute error scores.
        If `last_points_only` is set to True, it will use only the last point of each historical
        forecast. In this case, no reduction is used.

        By default, this method always re-trains the models on the entire available history,
        corresponding to an expanding window strategy.
        If `retrain` is set to False (useful for models for which training might be time-consuming, such as
        deep learning models), the model will only be trained on the initial training window
        (up to `start` time stamp), and only if it has not been trained before. Then, at every iteration, the
        newly expanded input sequence will be fed to the model to produce the new output.

        Parameters
        ----------
        series
            The target time series to use to successively train and evaluate the historical forecasts
        past_covariates
            An optional past-observed covariate series. This applies only if the model supports past covariates.
        future_covariates
            An optional future-known covariate series. This applies only if the model supports future covariates.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.
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
            The number of time steps between two consecutive predictions.
        retrain
            Whether to retrain the model for every prediction or not. Currently only `TorchForecastingModel`
            instances such as `BlockRNNModel`, `RNNModel`, `TCNModel`, `NBEATSModel` and `TransformerModel` support
            setting `retrain` to `False`.
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
        verbose
            Whether to print progress
        Returns
        -------
        float or List[float]
            The error score, or the list of individual error scores if `reduction` is `None`
        """
        forecasts = self.historical_forecasts(series=series,
                                              past_covariates=past_covariates,
                                              future_covariates=future_covariates,
                                              num_samples=num_samples,
                                              start=start,
                                              forecast_horizon=forecast_horizon,
                                              stride=stride,
                                              retrain=retrain,
                                              overlap_end=overlap_end,
                                              last_points_only=last_points_only,
                                              verbose=verbose)

        if last_points_only:
            return metric(series, forecasts)

        errors = [metric(series, forecast) for forecast in forecasts]
        if reduction is None:
            return errors

        return reduction(np.array(errors))

    @classmethod
    def gridsearch(model_class,
                   parameters: dict,
                   series: TimeSeries,
                   past_covariates: Optional[TimeSeries] = None,
                   future_covariates: Optional[TimeSeries] = None,
                   forecast_horizon: Optional[int] = None,
                   start: Union[pd.Timestamp, float, int] = 0.5,
                   last_points_only: bool = False,
                   val_series: Optional[TimeSeries] = None,
                   use_fitted_values: bool = False,
                   metric: Callable[[TimeSeries, TimeSeries], float] = metrics.mape,
                   reduction: Callable[[np.ndarray], float] = np.mean,
                   verbose=False,
                   n_jobs: int = 1) -> Tuple['ForecastingModel', Dict]:
        """
        A function for finding the best hyper-parameters among a given set.
        This function has 3 modes of operation: Expanding window mode, split mode and fitted value mode.
        The three modes of operation evaluate every possible combination of hyper-parameter values
        provided in the `parameters` dictionary by instantiating the `model_class` subclass
        of ForecastingModel with each combination, and returning the best-performing model with regards
        to the `metric` function. The `metric` function is expected to return an error value,
        thus the model resulting in the smallest `metric` output will be chosen.

        The relationship of the training data and test data depends on the mode of operation.

        Expanding window mode (activated when `forecast_horizon` is passed):
        For every hyperparameter combination, the model is repeatedly trained and evaluated on different
        splits of `training_series` and `target_series`. This process is accomplished by using
        the `backtest` function as a subroutine to produce historic forecasts starting from `start`
        that are compared against the ground truth values of `training_series` or `target_series`, if
        specified. Note that the model is retrained for every single prediction, thus this mode is slower.

        Split window mode (activated when `val_series` is passed):
        This mode will be used when the `val_series` argument is passed.
        For every hyper-parameter combination, the model is trained on `series` and
        evaluated on `val_series`.

        Fitted value mode (activated when `use_fitted_values` is set to `True`):
        For every hyper-parameter combination, the model is trained on `series`
        and evaluated on the resulting fitted values.
        Not all models have fitted values, and this method raises an error if the model doesn't have a `fitted_values`
        member. The fitted values are the result of the fit of the model on `series`. Comparing with the
        fitted values can be a quick way to assess the model, but one cannot see if the model is overfitting the series.

        Derived classes must ensure that a single instance of a model will not share parameters with the other
        instances, e.g., saving models in the same path. Otherwise, an unexpected behavior can arise while running
        several models in parallel (when `n_jobs != 1`). If this cannot be avoided, then gridsearch should be redefined,
        forcing `n_jobs = 1`.

        Currently this method only supports deterministic predictions (i.e. `num_samples == 1`).

        Parameters
        ----------
        model_class
            The ForecastingModel subclass to be tuned for 'series'.
        parameters
            A dictionary containing as keys hyperparameter names, and as values lists of values for the
            respective hyperparameter.
        series
            The TimeSeries instance used as input and target for training.
        past_covariates
            An optional past-observed covariate series. This applies only if the model supports past covariates.
        future_covariates
            An optional future-known covariate series. This applies only if the model supports future covariates.
        forecast_horizon
            The integer value of the forecasting horizon used in expanding window mode.
        start
            The `int`, `float` or `pandas.Timestamp` that represents the starting point in the time index
            of `training_series` from which predictions will be made to evaluate the model.
            For a detailed description of how the different data types are interpreted, please see the documentation
            for `ForecastingModel.backtest`.
        last_points_only
            Whether to use the whole forecasts or only the last point of each forecast to compute the error
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
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when there are two or more parameters
            combinations to evaluate. Each job will instantiate, train, and evaluate a different instance of the model.
            Defaults to `1` (sequential). Setting the parameter to `-1` means using all the available cores.

        Returns
        -------
        ForecastingModel, Dict
            A tuple containing an untrained 'model_class' instance created from the best-performing hyper-parameters,
            along with a dictionary containing these best hyper-parameters.
        """
        raise_if_not((forecast_horizon is not None) + (val_series is not None) + use_fitted_values == 1,
                     "Please pass exactly one of the arguments 'forecast_horizon', "
                     "'val_target_series' or 'use_fitted_values'.", logger)

        if use_fitted_values:
            raise_if_not(hasattr(model_class(), "fitted_values"),
                         "The model must have a fitted_values attribute to compare with the train TimeSeries",
                         logger)

        elif val_series is not None:
            raise_if_not(series.width == val_series.width,
                         "Training and validation series require the same number of components.",
                         logger)

        # TODO: here too I'd say we can leave these checks to the models
        # if covariates is not None:
        #     raise_if_not(series.has_same_time_as(covariates), 'The provided series and covariates must have the '
        #                                                       'same time axes.')

        # compute all hyperparameter combinations from selection
        params_cross_product = list(product(*parameters.values()))

        # iterate through all combinations of the provided parameters and choose the best one
        iterator = _build_tqdm_iterator(zip(params_cross_product), verbose, total=len(params_cross_product))

        def _evaluate_combination(param_combination):
            param_combination_dict = dict(list(zip(parameters.keys(), param_combination)))
            model = model_class(**param_combination_dict)
            if use_fitted_values:  # fitted value mode
                model._fit_wrapper(series, past_covariates, future_covariates)
                fitted_values = TimeSeries.from_times_and_values(series.time_index, model.fitted_values)
                error = metric(fitted_values, series)
            elif val_series is None:  # expanding window mode
                error = model.backtest(series=series,
                                       past_covariates=past_covariates,
                                       future_covariates=future_covariates,
                                       num_samples=1,
                                       start=start,
                                       forecast_horizon=forecast_horizon,
                                       metric=metric,
                                       reduction=reduction,
                                       last_points_only=last_points_only)
            else:  # split mode
                model._fit_wrapper(series, past_covariates, future_covariates)
                pred = model._predict_wrapper(len(val_series), series, past_covariates, future_covariates, num_samples=1)
                error = metric(pred, val_series)

            return error

        errors = _parallel_apply(iterator, _evaluate_combination, n_jobs, {}, {})

        min_error = min(errors)

        best_param_combination = dict(list(zip(parameters.keys(), params_cross_product[errors.index(min_error)])))

        logger.info('Chosen parameters: ' + str(best_param_combination))

        return model_class(**best_param_combination), best_param_combination

    def residuals(self,
                  series: TimeSeries,
                  forecast_horizon: int = 1,
                  verbose: bool = False) -> TimeSeries:
        """ A function for computing the residuals produced by the current model on a univariate time series.

        This function computes the difference between the actual observations from `series`
        and the fitted values vector `p` obtained by training the model on `series`.
        For every index `i` in `series`, `p[i]` is computed by training the model on
        `series[:(i - forecast_horizon)]` and forecasting `forecast_horizon` into the future.
        (`p[i]` will be set to the last value of the predicted series.)
        The vector of residuals will be shorter than `series` due to the minimum
        training series length required by the model and the gap introduced by `forecast_horizon`.
        Most commonly, unless otherwise specified, the term "residuals" implies a value for `forecast_horizon` of 1.

        This method works only on univariate series and does not currently support covariates. It uses the median
        prediction (when dealing with stochastic forecasts having num_samples > 1).

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
        first_index = series.time_index[self.min_train_series_length]

        # compute fitted values
        p = self.historical_forecasts(series=series,
                                      start=first_index,
                                      forecast_horizon=forecast_horizon,
                                      stride=1,
                                      retrain=True,
                                      last_points_only=True,
                                      verbose=verbose)

        # compute residuals
        series_trimmed = series.slice_intersect(p)
        residuals = series_trimmed - (p.quantile_timeseries(quantile=0.5) if p.is_stochastic else p)

        return residuals


class GlobalForecastingModel(ForecastingModel, ABC):
    """ The base class for "global" forecasting models, handling several time series and optional covariates.

    Global forecasting models expand upon the functionality of `ForecastingModel` in 4 ways:
    1. Models can be fitted on many series (multivariate or univariate) with different indices.
    2. The input series used by `predict()` can be different from the series used to fit the model.
    3. Covariates can be supported (multivariate or univariate).
    4. They can allow for multivariate target series and covariates.

    The name "global" stems from the fact that a training set of a forecasting model of this class is not constrained
    to a temporally contiguous, "local", time series.

    All implementations have to implement the `fit()` and `predict()` methods defined below.
    The `fit()` method is meant to train the model on one or several training time series, along with optional
    covariates.

    If `fit()` has been called with only one training and covariate series as argument, then calling `predict()` will
    forecast the future of this series. Otherwise, the user has to provide to `predict()` the series they want
    to forecast, as well as covariates, if needed.
    """

    _expect_past_covariates, _expect_future_covariates = False, False
    past_covariate_series, future_covariate_series = None, None

    @abstractmethod
    def fit(self,
            series: Union[TimeSeries, Sequence[TimeSeries]],
            past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
            future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None
            ) -> None:
        """ Fits/trains the model on the provided series

        Defines behavior that should happen when calling the `fit()` method of every global forecasting model.

        Some models support training only on one time series, while others can handle a sequence.
        Similarly, some models can handle past covariates, future covariates, or both.

        Parameters
        ----------
        series
            One or several target time series. The model will be trained to forecast these time series.
            The series may or may not be multivariate, but if multiple series are provided they must have the same number of components.
        past_covariates
            One or several past-observed covariate time series. These time series will not be forecast, but can
            be used by some models as an input. The covariate(s) may or may not be multivariate, but if multiple
            covariates are provided they must have the same number of components.
        future_covariates
            One or several future-known covariate time series. These time series will not be forecast, but can
            be used by some models as an input. The covariate(s) may or may not be multivariate, but if multiple
            covariates are provided they must have the same number of components.
        """

        if isinstance(series, TimeSeries):
            # if only one series is provided, save it for prediction time (including covariates, if available)
            self.training_series = series
            if past_covariates is not None:
                self.past_covariate_series = past_covariates
            if future_covariates is not None:
                self.future_covariate_series = future_covariates
        else:
            if past_covariates is not None:
                self._expect_past_covariates = True
            if future_covariates is not None:
                self._expect_future_covariates = True
        self._fit_called = True

    @abstractmethod
    def predict(self,
                n: int,
                series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                num_samples: int = 1,
                ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """ Forecasts values for a certain number of time steps after the end of the series.

        If `fit()` has been called with only one `TimeSeries` as argument, then the `series` argument of this function
        is optional, and it will simply produce the next `horizon` time steps forecast.
        The `past_covariates` and `future_covariates` arguments also don't have to be provided again in this case.

        If `fit()` has been called with `series` specified as a `Sequence[TimeSeries]`, the `series` argument must
        be specified.

        When the `series` argument is specified, this function will compute the next `n` time steps forecasts
        for the simple series (or for each series in the sequence) given by `series`.

        If multiple past or future covariates were specified during the training, some corresponding covariates must
        also be specified here. For every input in `series` a matching (past and/or future) covariate time series
        has to be provided.

        Parameters
        ----------
        n
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        series
            The series whose future(s) we want to predict.
        past_covariates
            One past-observed covariate time series for every input time series in `series`. They must match the
            past covariates that have been used with the `fit()` function for training in terms of dimension and type.
        future_covariates
            One future-known covariate time series for every input time series in `series`. They must match the
            past covariates that have been used with the `fit()` function for training in terms of dimension and type.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            If `series` is not specified, this function returns a single time series containing the `n`
            next points after then end of the training series.
            If `series` is given and is a simple `TimeSeries`, this function returns the `n` next points
            after the end of `series`.
            If `series` is given and is a sequence of several time series, this function returns
            a sequence where each element contains the corresponding `n` points forecasts.
        """
        if series is None and past_covariates is None and future_covariates is None:
            super().predict(n, num_samples)
        if self._expect_past_covariates and past_covariates is None:
            raise_log(ValueError('The model has been trained with past covariates. Some matching past_covariates '
                                 'have to be provided to `predict()`.'))
        if self._expect_future_covariates and future_covariates is None:
            raise_log(ValueError('The model has been trained with future covariates. Some matching future_covariates '
                                 'have to be provided to `predict()`.'))

    def _predict_wrapper(self, n: int, series: TimeSeries, past_covariates: Optional[TimeSeries],
                         future_covariates: Optional[TimeSeries], num_samples: int) -> TimeSeries:
        return self.predict(n, series, past_covariates=past_covariates,
                            future_covariates=future_covariates, num_samples=num_samples)

    def _fit_wrapper(self, series: TimeSeries, past_covariates: Optional[TimeSeries],
                     future_covariates: Optional[TimeSeries]):
        self.fit(series, past_covariates=past_covariates, future_covariates=future_covariates)


class DualCovariatesForecastingModel(ForecastingModel, ABC):
    """ The base class for the forecasting models that are not global, but support future covariates.
    Among other things, it lets Darts forecasting models wrap around statsmodels models
    having a `future_covariates` parameter, which corresponds to future-known covariates.

    All implementations have to implement the `fit()` and `predict()` methods defined below.
    """

    _expect_covariate = False

    @abstractmethod
    def fit(self,
            series: TimeSeries,
            future_covariates: Optional[TimeSeries] = None
            ) -> None:
        """ Fits/trains the model on the provided series

        Defines behavior that should happen when calling the `fit()` method for the forecasting models handling
        optional future covariates (exogenous variables).

        Parameters
        ----------
        series
            The model will be trained to forecast this time series. Can be multivariate if the model supports it.
        future_covariates
            A time series of future-known covariates. This time series will not be forecasted, but can be used by
            some models as an input.
        """

        # TODO: is this really needed or could we find a workaround?
        if future_covariates is not None:
            raise_if_not(series.has_same_time_as(future_covariates),
                         'The target series and the future_covariates series must have the same time index.')
            self._expect_covariate = True
        super().fit(series)

    @abstractmethod
    def predict(self,
                n: int,
                future_covariates: Optional[TimeSeries] = None,
                num_samples: int = 1
                ) -> TimeSeries:
        """ Forecasts values for a certain number of time steps after the end of the series.

        If some future covariates were specified during the training, they must also be specified here.

        Parameters
        ----------
        n
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        future_covariates
            The time series of future-known covariates which can be fed as input to the model. It must correspond to the
            covariate time series that has been used with the `fit()` method for training, and it must be of length `n`.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.

        Returns
        -------
        TimeSeries, a single time series containing the `n` next points after then end of the training series.
        """
        if future_covariates is None:
            super().predict(n, num_samples)
        if self._expect_covariate and future_covariates is None:
            raise_log(ValueError('The model has been trained with future_covariates variables. Some matching '
                                 'future_covariates variables have to be provided to `predict()`.'))

        # TODO Here we could maybe slice the series and then check
        if self._expect_covariate and len(future_covariates) != n:
            raise_log(ValueError(f'Expecting future_covariates variables with the same length as the'
                                 f' forecasting horizon ({n}).'))

    def _fit_wrapper(self, series: TimeSeries, past_covariates: Optional[TimeSeries],
                     future_covariates: Optional[TimeSeries]):
        if future_covariates is not None and not series.has_same_time_as(future_covariates):
            # fit() expects future_covariates to have same time as the target, so we intersect it here
            # in case it's longer.
            future_covariates = future_covariates.slice_intersect(series)
        self.fit(series, future_covariates=future_covariates)

    def _predict_wrapper(self, n: int, series: TimeSeries,
                         past_covariates: Optional[TimeSeries],
                         future_covariates: Optional[TimeSeries],
                         num_samples: int) -> TimeSeries:
        if future_covariates is not None:
            start = series.end_time() + series.freq
            future_covariates = future_covariates[start:start + (n - 1) * series.freq]
        return self.predict(n, future_covariates=future_covariates, num_samples=num_samples)
