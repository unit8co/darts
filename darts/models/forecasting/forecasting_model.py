"""
Forecasting Model Base Classes

A forecasting model captures the future values of a time series as a function of the past as follows:

.. math:: y_{t+1} = f(y_t, y_{t-1}, ..., y_1),

where :math:`y_t` represents the time series' value(s) at time :math:`t`.

The main functions are `fit()` and `predict()`. `fit()` learns the function `f()`, over the history of
one or several time series. The function `predict()` applies `f()` on one or several time series in order
to obtain forecasts for a desired number of time stamps into the future.
"""
import copy
import datetime
import inspect
import os
import pickle
import time
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict
from itertools import product
from random import sample
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from darts import metrics
from darts.dataprocessing.encoders import SequentialEncoder
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.timeseries import TimeSeries
from darts.utils import (
    _build_tqdm_iterator,
    _historical_forecasts_general_checks,
    _parallel_apply,
    _retrain_wrapper,
    _with_sanity_checks,
)
from darts.utils.timeseries_generation import (
    _build_forecast_series,
    _generate_new_dates,
    generate_index,
)
from darts.utils.utils import drop_after_index, drop_before_index, series2seq

logger = get_logger(__name__)


class ModelMeta(ABCMeta):
    """Meta class to store parameters used at model creation.

    When creating a model instance, the parameters are extracted as follows:

        1)  Get the model's __init__ signature and store all arg and kwarg
            names as well as default values (empty for args) in an ordered
            dict `all_params`.
        2)  Replace the arg values from `all_params` with the positional
            args used at model creation.
        3)  Remove args from `all_params` that were not passed as positional
            args at model creation. This will enforce that an error is raised
            if not all positional args were passed. If all positional args
            were passed, no parameter will be removed.
        4)  Update `all_params` kwargs with optional kwargs from model creation.
        5)  Save `all_params` to the model.
        6)  Call (create) the model with `all_params`.
    """

    def __call__(cls, *args, **kwargs):
        # 1) get all default values from class' __init__ signature
        sig = inspect.signature(cls.__init__)
        all_params = OrderedDict(
            [
                (p.name, p.default)
                for p in sig.parameters.values()
                if not p.name == "self"
            ]
        )

        # 2) fill params with positional args
        for param, arg in zip(all_params, args):
            all_params[param] = arg

        # 3) remove args which were not set (and are per default empty)
        remove_params = []
        for param, val in all_params.items():
            if val is sig.parameters[param].empty:
                remove_params.append(param)
        for param in remove_params:
            all_params.pop(param)

        # 4) update defaults with actual model call parameters and store
        all_params.update(kwargs)

        # 5) save parameters in model
        cls._model_call = all_params

        # 6) call model
        return super().__call__(**all_params)


class ForecastingModel(ABC, metaclass=ModelMeta):
    """The base class for forecasting models. It defines the *minimal* behavior that all forecasting models have to
    support. The signatures in this base class are for "local" models handling only one univariate series and no
    covariates. Sub-classes can handle more complex cases.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        # The series, and covariates used for training the model through the `fit()` function.
        # This is only used if the model has been fit on one time series.
        self.training_series: Optional[TimeSeries] = None
        self.past_covariate_series: Optional[TimeSeries] = None
        self.future_covariate_series: Optional[TimeSeries] = None
        self.static_covariates: Optional[pd.DataFrame] = None

        self._expect_past_covariates, self._expect_future_covariates = False, False
        self._uses_past_covariates, self._uses_future_covariates = False, False

        # state; whether the model has been fit (on a single time series)
        self._fit_called = False

        # extract and store sub class model creation parameters
        self._model_params = self._extract_model_creation_params()

        if "add_encoders" not in kwargs:
            raise_log(
                NotImplementedError(
                    "Model subclass must pass the `add_encoders` parameter to base class."
                ),
                logger=logger,
            )

        # by default models do not use encoders
        self.add_encoders = kwargs["add_encoders"]
        self.encoders: Optional[SequentialEncoder] = None

    @abstractmethod
    def fit(self, series: TimeSeries) -> "ForecastingModel":
        """Fit/train the model on the provided series.

        Parameters
        ----------
        series
            A target time series. The model will be trained to forecast this time series.

        Returns
        -------
        self
            Fitted model.
        """
        raise_if_not(
            len(series) >= self.min_train_series_length,
            "Train series only contains {} elements but {} model requires at least {} entries".format(
                len(series), str(self), self.min_train_series_length
            ),
        )
        self.training_series = series
        self._fit_called = True

        if series.has_range_index:
            self._supports_range_index()

    def _supports_range_index(self) -> bool:
        """Checks if the forecasting model supports a range index.
        Some models may not support this, if for instance they rely on underlying dates.

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

    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        """
        Checks if the forecasting model supports historical forecasts without retraining
        the model. By default, returns False. Needs to be overwritten by models that do
        support historical forecasts without retraining.
        """
        return False

    @property
    def supports_past_covariates(self):
        return "past_covariates" in inspect.signature(self.fit).parameters.keys()

    @property
    def supports_future_covariates(self):
        return "future_covariates" in inspect.signature(self.fit).parameters.keys()

    @property
    def uses_past_covariates(self):
        """
        Whether the model uses past covariates, once fitted.
        """
        return self._uses_past_covariates

    @property
    def uses_future_covariates(self):
        """
        Whether the model uses future covariates, once fitted.
        """
        return self._uses_future_covariates

    @abstractmethod
    def predict(self, n: int, num_samples: int = 1) -> TimeSeries:
        """Forecasts values for `n` time steps after the end of the training series.

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
            raise_log(
                ValueError(
                    "The model must be fit before calling predict(). "
                    "For global models, if predict() is called without specifying a series, "
                    "the model must have been fit on a single training series."
                ),
                logger,
            )

        if not self._is_probabilistic() and num_samples > 1:
            raise_log(
                ValueError(
                    "`num_samples > 1` is only supported for probabilistic models."
                ),
                logger,
            )

    def _fit_wrapper(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
    ):
        self.fit(series)

    def _predict_wrapper(
        self,
        n: int,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
        num_samples: int,
        verbose: bool = False,
    ) -> TimeSeries:
        return self.predict(n, num_samples=num_samples, verbose=verbose)

    @property
    def min_train_series_length(self) -> int:
        """
        Class property defining the minimum required length for the training series.
        This function/property should be overridden if a value higher than 3 is required.
        """
        return 3

    @property
    def extreme_lags(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
    ]:
        """
        A 5-tuple containing in order:
        (minimum target lag, maximum target lag, min past covariate lag, min future covariate lag, max future covariate
        lag). If 0 is the index of the first prediction, then all lags are relative to this index, except for the
        maximum target lag, which is relative to the last element of the time series before the first prediction.
        See examples below.

        If the model wasn't fitted with:
            - target lag (concerning RegressionModels only) the first element should be `None`.

            - past covariates, the third element should be `None`.

            - future covariates, the fourth and fifth elements should be `None`.

        Should be overridden by models that use past or future covariates, and/or for model that have minimum target
        lag and maximum target lags potentially different from -1 and 1.

        Notes
        -----
        maximum target lag (second value) cannot be `None` and is always larger than 1.
        Examples
        --------
        >>> model = LinearRegressionModel(lags=3, output_chunk_length=2)
        >>> model.fit(train_series)
        >>> model.extreme_lags
        (-3, 2, None, None, None)
        >>> model = LinearRegressionModel(lags=[3, 5], past_covariates_lags = 4, output_chunk_length=7)
        >>> model.fit(train_series, past_covariates=past_covariates)
        >>> model.extreme_lags
        (-5, 7, -4, None, None)
        >>> model = LinearRegressionModel(lags=[3, 5], future_covariates_lags = [4, 6], output_chunk_length=7)
        >>> model.fit(train_series, future_covariates=future_covariates)
        >>> model.extreme_lags
        (-5, 7, None, 4, 6)
        >>> model = NBEATSModel(input_chunk_length=10, output_chunk_length=7)
        >>> model.fit(train_series)
        >>> model.extreme_lags
        (-10, 7, None, None, None)
        >>> model = NBEATSModel(input_chunk_length=10, output_chunk_length=7, future_covariates_lags=[4, 6])
        >>> model.fit(train_series, future_covariates)
        >>> model.extreme_lags
        (-10, 7, None, 4, 6)
        """

        return (-1, 1, None, None, None)

    @property
    def _training_sample_time_index_length(self) -> int:
        """
        Required time_index length for one training sample, for any model.
        """
        (
            min_target_lag,
            max_target_lag,
            min_past_cov_lag,
            min_future_cov_lag,
            max_future_cov_lag,
        ) = self.extreme_lags

        return max(
            max_target_lag,
            max_future_cov_lag if max_future_cov_lag else 0,
        ) - min(
            min_target_lag if min_target_lag else 0,
            min_past_cov_lag if min_past_cov_lag else 0,
            min_future_cov_lag if min_future_cov_lag else 0,
        )

    @property
    def _predict_sample_time_index_length(self) -> int:
        """
        Required time_index length for one predict sample, for any model.
         A predict sample is the minimum required set of series and covariates chunks to be able to predict
         a single point.
        """
        (
            min_target_lag,
            max_target_lag,
            min_past_cov_lag,
            min_future_cov_lag,
            max_future_cov_lag,
        ) = self.extreme_lags

        return (
            max_future_cov_lag
            if max_future_cov_lag
            else 0
            - min(
                min_target_lag if min_target_lag else 0,
                min_past_cov_lag if min_past_cov_lag else 0,
                min_future_cov_lag if min_future_cov_lag else 0,
            )
        )

    def _get_historical_forecastable_time_index(
        self,
        series: Optional[TimeSeries] = None,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        is_training: Optional[bool] = False,
    ) -> Union[pd.DatetimeIndex, pd.RangeIndex, None]:
        """
        Private function that returns the largest time_index representing the subset of each timestamps
        for which historical forecasts can be made, given the model's properties, the training series
        and the covariates.
            - If ``None`` is returned, there is no point where a forecast can be made.

            - If ``is_training=False``, it returns the time_index subset of predictable timestamps.

            - If ``is_training=True``, it returns the time_index subset of trainable timestamps. A trainable
            timestamp is a timestamp that has a training sample of length at least ``self.training_sample_length``
             preceding it.


        Parameters
        ----------
        series
            Optionally, a target series.
        past_covariates
            Optionally, a past covariates.
        future_covariates
            Optionally, a future covariates.
        is_training
            Whether the returned time_index should be taking into account the training.

        Returns
        -------
        Union[pd.DatetimeIndex, pd.RangeIndex, None]
            The longest time_index that can be used for historical forecasting.

        Examples
        --------
        >>> model = LinearRegressionModel(lags=3, output_chunk_length=2)
        >>> model.fit(train_series)
        >>> series = TimeSeries.from_times_and_values(pd.date_range('2000-01-01', '2000-01-10'), np.arange(10))
        >>> model._get_historical_forecastable_time_index(series=series, is_training=False)
        DatetimeIndex(['2000-01-04', '2000-01-05', '2000-01-06', '2000-01-07',
               '2000-01-08', '2000-01-09', '2000-01-10'],
              dtype='datetime64[ns]', freq='D')
        >>> model._get_historical_forecastable_time_index(series=series, is_training=True)
        DatetimeIndex(['2000-01-06', '2000-01-08', '2000-01-09', '2000-01-10'],
                dtype='datetime64[ns]', freq='D')

        >>> model = NBEATSModel(input_chunk_length=3, output_chunk_length=3)
        >>> model.fit(train_series, train_past_covariates)
        >>> series = TimeSeries.from_times_and_values(pd.date_range('2000-10-01', '2000-10-09'), np.arange(8))
        >>> past_covariates = TimeSeries.from_times_and_values(pd.date_range('2000-10-03', '2000-10-20'),
        np.arange(18))
        >>> model._get_historical_forecastable_time_index(series=series, past_covariates=past_covariates,
        is_training=False)
        DatetimeIndex(['2000-10-06', '2000-10-07', '2000-10-08', '2000-10-09'], dtype='datetime64[ns]', freq='D')
        >>> model._get_historical_forecastable_time_index(series=series, past_covariates=past_covariates,
        is_training=True)
        DatetimeIndex(['2000-10-09'], dtype='datetime64[ns]', freq='D') # Only one point is trainable, and
        # corresponds to the first point after we reach a common subset of timestamps of training_sample_length length.
        """

        (
            min_target_lag,
            max_target_lag,
            min_past_cov_lag,
            min_future_cov_lag,
            max_future_cov_lag,
        ) = self.extreme_lags

        intersect_ = None

        # target longest possible time index
        if (min_target_lag is not None) and (series is not None):
            intersect_ = generate_index(
                start=series.start_time()
                - (min_target_lag - max_target_lag) * series.freq
                if is_training
                else series.start_time() - min_target_lag * series.freq,
                end=series.end_time(),
                freq=series.freq,
            )

        # past covariates longest possible time index
        if (min_past_cov_lag is not None) and (past_covariates is not None):
            tmp_ = generate_index(
                start=past_covariates.start_time()
                - (min_past_cov_lag - max_target_lag) * past_covariates.freq
                if is_training
                else past_covariates.start_time()
                - min_past_cov_lag * past_covariates.freq,
                end=past_covariates.end_time(),
                freq=past_covariates.freq,
            )
            if intersect_ is not None:
                intersect_ = intersect_.intersection(tmp_)
            else:
                intersect_ = tmp_

        # future covariates longest possible time index
        if (min_future_cov_lag is not None) and (future_covariates is not None):
            tmp_ = generate_index(
                start=future_covariates.start_time()
                - (min_future_cov_lag - max_target_lag) * future_covariates.freq
                if is_training
                else future_covariates.start_time()
                - min_future_cov_lag * future_covariates.freq,
                end=future_covariates.end_time()
                - max_future_cov_lag * future_covariates.freq,
                freq=future_covariates.freq,
            )

            if intersect_ is not None:
                intersect_ = intersect_.intersection(tmp_)
            else:
                intersect_ = tmp_

        return intersect_ if len(intersect_) > 0 else None

    def _generate_new_dates(
        self, n: int, input_series: Optional[TimeSeries] = None
    ) -> Union[pd.DatetimeIndex, pd.RangeIndex]:
        """
        Generates `n` new dates after the end of the specified series
        """
        input_series = (
            input_series if input_series is not None else self.training_series
        )
        return _generate_new_dates(n=n, input_series=input_series)

    def _build_forecast_series(
        self,
        points_preds: Union[np.ndarray, Sequence[np.ndarray]],
        input_series: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """
        Builds a forecast time series starting after the end of the training time series, with the
        correct time index (or after the end of the input series, if specified).
        """
        input_series = (
            input_series if input_series is not None else self.training_series
        )
        return _build_forecast_series(points_preds, input_series)

    def _historical_forecasts_sanity_checks(self, *args: Any, **kwargs: Any) -> None:
        """Sanity checks for the historical_forecasts function

        Parameters
        ----------
        args
            The args parameter(s) provided to the historical_forecasts function.
        kwargs
            The kwargs parameter(s) provided to the historical_forecasts function.

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
    def historical_forecasts(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        train_length: Optional[int] = None,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        forecast_horizon: int = 1,
        stride: int = 1,
        retrain: Union[bool, int, Callable[..., bool]] = True,
        overlap_end: bool = False,
        last_points_only: bool = True,
        verbose: bool = False,
    ) -> Union[
        TimeSeries, List[TimeSeries], Sequence[TimeSeries], Sequence[List[TimeSeries]]
    ]:

        """Compute the historical forecasts that would have been obtained by this model on
        (potentially multiple) `series`.

        This method uses an expanding training window;
        it repeatedly builds a training set from the beginning of `series`. It trains the
        model on the training set, emits a forecast of length equal to forecast_horizon, and then moves
        the end of the training set forward by `stride` time steps.

        By default, this method will return one (or a sequence of) single time series made up of
        the last point of each historical forecast.
        This time series will thus have a frequency of ``series.freq * stride``.
        If `last_points_only` is set to False, it will instead return one (or a sequence of) list of the
        historical forecasts series.

        By default, this method always re-trains the models on the entire available history,
        corresponding to an expanding window strategy.
        If `retrain` is set to False, the model will only be trained on the initial training window
        (up to `start` time stamp), and only if it has not been trained before. This is not
        supported by all models.

        Parameters
        ----------
        series
            One or multiple target time series to use to successively train and evaluate
            the historical forecasts.
        past_covariates
            Optionally, one (or a sequence of) past-observed covariate series.
            This applies only if the model supports past covariates.
        future_covariates
            Optionally, one (or a sequence of) of future-known covariate series.
            This applies only if the model supports future covariates.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.
        train_length
            Number of time steps in our training set (size of backtesting window to train on).
            Default is set to train_length=None where it takes all available time steps up until prediction time,
            otherwise the moving window strategy is used. If larger than the number of time steps available, all steps
            up until prediction time are used, as in default case. Needs to be at least min_train_series_length.
        start
            The first point of time at which a prediction is computed for a future time.
            This parameter supports 3 different data types: ``float``, ``int`` and ``pandas.Timestamp``.
            In the case of ``float``, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of ``int``, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of ``pandas.Timestamp``, this time stamp will be used to determine the first prediction time
            directly.
            If `start` is not specified, the first prediction time will automatically be set to :
                 - the first predictable point if `retrain` is False

                 - the first trainable point if `retrain` is True and `train_length` is None

                 - the first trainable point + `train_length` otherwise
        forecast_horizon
            The forecast horizon for the predictions
        stride
            The number of time steps between two consecutive predictions.
        retrain
            Whether and/or on which condition to retrain the model before predicting.
            This parameter supports 3 different datatypes: ``bool``, (positive) ``int``, and
            ``Callable`` (returning a ``bool``).
            In the case of ``bool``: retrain the model at each step (`True`), or never retrains the model (`False`).
            In the case of ``int``: the model is retrained every `retrain` iterations.
            In the case of ``Callable``: the model is retrained whenever callable returns `True`.
            Arguments passed to the callable are as follows:

                - `pred_time (pd.Timestamp or int)`: timestamp of forecast time (end of the training series)
                - `train_series (TimeSeries)`: train series up to `pred_time`
                - `past_covariates (TimeSeries)`: past_covariates series up to `pred_time`
                - `future_covariates (TimeSeries)`: future_covariates series up
                  to `min(pred_time + series.freq * forecast_horizon, series.end_time())`

            Note: some models do require being retrained every time
            and do not support anything else than `retrain=True`.
        overlap_end
            Whether the returned forecasts can go beyond the series' end or not
        last_points_only
            Whether to retain only the last point of each historical forecast.
            If set to True, the method returns a single ``TimeSeries`` containing the successive point forecasts.
            Otherwise returns a list of historical ``TimeSeries`` forecasts.
        verbose
            Whether to print progress
        Returns
        -------
        TimeSeries or List[TimeSeries] or List[List[TimeSeries]]
            If `last_points_only` is set to True and a single series is provided in input,
            a single ``TimeSeries`` is returned, which contains the the historical forecast
            at the desired horizon.

            A ``List[TimeSeries]`` is returned if either `series` is a ``Sequence`` of ``TimeSeries``,
            or if `last_points_only` is set to False. A list of lists is returned if both conditions are met.
            In this last case, the outer list is over the series provided in the input sequence,
            and the inner lists contain the different historical forecasts.
        """

        # only GlobalForecastingModels support historical forecastings without retraining the model
        base_class_name = self.__class__.__base__.__name__
        raise_if(
            (isinstance(retrain, Callable) or int(retrain) != 1)
            and (not self._supports_non_retrainable_historical_forecasts()),
            f"{base_class_name} does not support historical forecastings with `retrain` set to `False`. "
            f"For now, this is only supported with GlobalForecastingModels such as TorchForecastingModels. "
            f"For more information, read the documentation for `retrain` in `historical_forecasts()`",
            logger,
        )

        if train_length and not isinstance(train_length, int):
            raise_log(
                TypeError("If not None, train_length needs to be an integer."),
                logger,
            )
        elif (train_length is not None) and train_length < 1:
            raise_log(
                ValueError("If not None, train_length needs to be positive."),
                logger,
            )
        elif (
            train_length is not None
        ) and train_length < self._training_sample_time_index_length:
            raise_log(
                ValueError(
                    "train_length is too small for the training requirements of this model"
                ),
                logger,
            )

        if isinstance(retrain, bool) or (isinstance(retrain, int) and retrain >= 0):
            retrain_func = _retrain_wrapper(
                lambda counter: counter % int(retrain) == 0 if retrain else False
            )

        elif isinstance(retrain, Callable):
            retrain_func = _retrain_wrapper(retrain)

        else:
            raise_log(
                ValueError(
                    "`retrain` argument must be either `bool`, positive `int` or `Callable` (returning `bool`)"
                ),
                logger,
            )
        retrain_func_signature = tuple(
            inspect.signature(retrain_func).parameters.keys()
        )

        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)

        # If the model has never been fitted before using historical_forecasts,
        # we need to know if it uses past or future covariates. The only possible assumption is that
        # the user is using the same covariates as they would in the fit method.
        if self._fit_called is False:
            if past_covariates is not None:
                self._uses_past_covariates = True
            if future_covariates is not None:
                self._uses_future_covariates = True

        if len(series) == 1:
            # Use tqdm on the outer loop only if there's more than one series to iterate over
            # (otherwise use tqdm on the inner loop).
            outer_iterator = series
        else:
            outer_iterator = _build_tqdm_iterator(series, verbose)

        forecasts_list = []
        for idx, series_ in enumerate(outer_iterator):

            past_covariates_ = past_covariates[idx] if past_covariates else None
            future_covariates_ = future_covariates[idx] if future_covariates else None

            # Determines the time index the historical forecasts will be made on
            historical_forecasts_time_index = (
                self._get_historical_forecastable_time_index(
                    series_,
                    past_covariates_,
                    future_covariates_,
                    (retrain is not False) or (not self._fit_called),
                )
            )

            # We also need the first value timestamp to be used for prediction or training
            min_timestamp = (
                historical_forecasts_time_index[0]
                - (
                    self._training_sample_time_index_length
                    if retrain
                    else self._predict_sample_time_index_length
                )
                * series_.freq
            )

            if historical_forecasts_time_index is None:
                raise_log(
                    ValueError(
                        "Given the provided model, series and covariates, there is no timestamps "
                        f" where we can make a prediction or train the model (series index: {idx}). "
                        "Please check the time indexes of the series and covariates."
                    ),
                    logger,
                )

            # prepare the start parameter -> pd.Timestamp
            if start is not None:

                historical_forecasts_time_index = drop_before_index(
                    historical_forecasts_time_index,
                    series_.get_timestamp_at_point(start),
                )

            else:
                if (retrain is not False) or (not self._fit_called):

                    if train_length:
                        historical_forecasts_time_index = drop_before_index(
                            historical_forecasts_time_index,
                            historical_forecasts_time_index[0]
                            + max(
                                (
                                    train_length
                                    - self._training_sample_time_index_length
                                ),
                                0,
                            )
                            * series_.freq,
                        )
                    # if not we start training right away, but with 2 minimum points, so we start
                    # 1 time step after the first trainable point.
                    # (sklearn check that there are at least 2 points in the training set and it seems
                    # rather reasonable)
                    else:
                        historical_forecasts_time_index = drop_before_index(
                            historical_forecasts_time_index,
                            historical_forecasts_time_index[0] + 1 * series_.freq,
                        )

                    # if retrain is False, fit hasn't been called yet and train_length None,
                    # it means that the entire backtesting will be based on a set of two training samples
                    # at the first step, so we warn the user.
                    raise_if(
                        (not self._fit_called)
                        and (retrain is False)
                        and (not train_length),
                        " The model has not been fitted yet, and `start` and train_length are not specified. "
                        " The model is not retraining during the historical forecasts. Hence the "
                        "the first and only training would be done on 2 samples.",
                        logger,
                    )

            # build the prediction times in advance (to be able to use tqdm)
            last_valid_pred_time = self._get_last_prediction_time(
                series_,
                forecast_horizon,
                overlap_end,
            )

            historical_forecasts_time_index = drop_after_index(
                historical_forecasts_time_index, last_valid_pred_time
            )

            if len(series) == 1:
                # Only use tqdm if there's no outer loop
                iterator = _build_tqdm_iterator(
                    historical_forecasts_time_index[::stride], verbose
                )
            else:
                iterator = historical_forecasts_time_index[::stride]

            # Either store the whole forecasts or only the last points of each forecast, depending on last_points_only
            forecasts = []

            last_points_times = []
            last_points_values = []

            # iterate and forecast
            for _counter, pred_time in enumerate(iterator):

                # build the training series
                if min_timestamp > series_.time_index[0]:
                    train_series = series_.drop_before(
                        min_timestamp - 1 * series_.freq
                    ).drop_after(pred_time)
                else:
                    train_series = series_.drop_after(pred_time)

                if train_length and len(train_series) > train_length:
                    train_series = train_series[-train_length:]

                if (not self._fit_called) or retrain_func(
                    counter=_counter,
                    pred_time=pred_time,
                    train_series=train_series,
                    past_covariates=past_covariates_.drop_after(pred_time)
                    if past_covariates_
                    and ("past_covariates" in retrain_func_signature)
                    else None,
                    future_covariates=future_covariates_.drop_after(
                        min(
                            pred_time + train_series.freq * forecast_horizon,
                            series_.end_time(),
                        )
                    )
                    if future_covariates_
                    else None,
                ):
                    self._fit_wrapper(
                        series=train_series,
                        past_covariates=past_covariates_,
                        future_covariates=future_covariates_,
                    )

                forecast = self._predict_wrapper(
                    n=forecast_horizon,
                    series=train_series,
                    past_covariates=past_covariates_,
                    future_covariates=future_covariates_,
                    num_samples=num_samples,
                    verbose=verbose,
                )

                if last_points_only:
                    last_points_values.append(forecast.all_values(copy=False)[-1])
                    last_points_times.append(forecast.end_time())
                else:
                    forecasts.append(forecast)

            if last_points_only:
                forecasts_list.append(
                    TimeSeries.from_times_and_values(
                        generate_index(
                            start=last_points_times[0],
                            end=last_points_times[-1],
                            freq=series_.freq,
                        )[::stride],
                        np.array(last_points_values),
                        columns=series_.columns,
                        static_covariates=series_.static_covariates,
                        hierarchy=series_.hierarchy,
                    )
                )

            else:
                forecasts_list.append(forecasts)

        return forecasts_list if len(series) > 1 else forecasts_list[0]

    def backtest(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        train_length: Optional[int] = None,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        forecast_horizon: int = 1,
        stride: int = 1,
        retrain: Union[bool, int, Callable[..., bool]] = True,
        overlap_end: bool = False,
        last_points_only: bool = False,
        metric: Union[
            Callable[[TimeSeries, TimeSeries], float],
            List[Callable[[TimeSeries, TimeSeries], float]],
        ] = metrics.mape,
        reduction: Union[Callable[[np.ndarray], float], None] = np.mean,
        verbose: bool = False,
    ) -> Union[float, List[float], Sequence[float], List[Sequence[float]]]:

        """Compute error values that the model would have produced when
        used on (potentially multiple) `series`.

        It repeatedly builds a training set from the beginning of `series`. It trains the
        current model on the training set, emits a forecast of length equal to forecast_horizon, and then moves
        the end of the
        training set forward by `stride` time steps. A metric (given by the `metric` function) is then evaluated
        on the forecast and the actual values. Finally, the method returns a `reduction` (the mean by default)
        of all these metric scores.

        By default, this method uses each historical forecast (whole) to compute error scores.
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
            The (or a sequence of) target time series to use to successively train and evaluate the historical forecasts
        past_covariates
            Optionally, one (or a sequence of) past-observed covariate series.
            This applies only if the model supports past covariates.
        future_covariates
            Optionally, one (or a sequence of) future-known covariate series.
            This applies only if the model supports future covariates.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.
        train_length
            Number of time steps in our training set (size of backtesting window to train on).
            Default is set to train_length=None where it takes all available time steps up until prediction time,
            otherwise the moving window strategy is used. If larger than the number of time steps available, all steps
            up until prediction time are used, as in default case. Needs to be at least min_train_series_length.
        start
            The first prediction time, at which a prediction is computed for a future time.
            This parameter supports 3 different types: ``float``, ``int`` and ``pandas.Timestamp``.
            In the case of ``float``, the parameter will be treated as the proportion of the time series
            that should lie before the first prediction point.
            In the case of ``int``, the parameter will be treated as an integer index to the time index of
            `series` that will be used as first prediction time.
            In case of ``pandas.Timestamp``, this time stamp will be used to determine the first prediction time
            directly.
            If `start` is not specified, the first prediction time will automatically be set to :
                 - the first predictable point if `retrain` is False

                 - the first trainable point if `retrain` is True and `train_length` is None

                 - the first trainable point + `train_length` otherwise
        forecast_horizon
            The forecast horizon for the point prediction.
        stride
            The number of time steps between two consecutive training sets.
        retrain
            Whether and/or on which condition to retrain the model before predicting.
            This parameter supports 3 different datatypes: ``bool``, (positive) ``int``, and
            ``Callable`` (returning a ``bool``).
            In the case of ``bool``: retrain the model at each step (`True`), or never retrains the model (`False`).
            In the case of ``int``: the model is retrained every `retrain` iterations.
            In the case of ``Callable``: the model is retrained whenever callable returns `True`.
            Arguments passed to the callable are as follows:

                - `pred_time (pd.Timestamp or int)`: timestamp of forecast time (end of the training series)
                - `train_series (TimeSeries)`: train series up to `pred_time`
                - `past_covariates (TimeSeries)`: past_covariates series up to `pred_time`
                - `future_covariates (TimeSeries)`: future_covariates series up
                  to `min(pred_time + series.freq * forecast_horizon, series.end_time())`

            Note: some models do require being retrained every time
            and do not support anything else than `retrain=True`.
        overlap_end
            Whether the returned forecasts can go beyond the series' end or not
        last_points_only
            Whether to use the whole historical forecasts or only the last point of each forecast to compute the error
        metric
            A function or a list of function that takes two ``TimeSeries`` instances as inputs and returns an
            error value.
        reduction
            A function used to combine the individual error scores obtained when `last_points_only` is set to False.
            When providing several metric functions, the function will receive the argument `axis = 0` to obtain single
            value for each metric function.
            If explicitly set to `None`, the method will return a list of the individual error scores instead.
            Set to ``np.mean`` by default.
        verbose
            Whether to print progress
        Returns
        -------
        float or List[float] or List[List[float]]
            The (sequence of) error score on a series, or list of list containing error scores for each
            provided series and each sample.
        """

        forecasts = self.historical_forecasts(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=num_samples,
            train_length=train_length,
            start=start,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=retrain,
            overlap_end=overlap_end,
            last_points_only=last_points_only,
            verbose=verbose,
        )

        series = series2seq(series)
        if len(series) == 1:
            forecasts = [forecasts]

        if not isinstance(metric, list):
            metric = [metric]

        backtest_list = []
        for idx, target_ts in enumerate(series):
            if last_points_only:
                errors = [metric_f(target_ts, forecasts[idx]) for metric_f in metric]
                if len(errors) == 1:
                    errors = errors[0]
                backtest_list.append(errors)
            else:

                errors = [
                    [metric_f(series, f) for metric_f in metric]
                    if len(metric) > 1
                    else metric[0](series, f)
                    for f in forecasts[idx]
                ]

                if reduction is None:
                    backtest_list.append(errors)
                else:
                    backtest_list.append(reduction(np.array(errors), axis=0))

        return backtest_list if len(backtest_list) > 1 else backtest_list[0]

    @classmethod
    def gridsearch(
        model_class,
        parameters: dict,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        forecast_horizon: Optional[int] = None,
        stride: int = 1,
        start: Union[pd.Timestamp, float, int] = 0.5,
        last_points_only: bool = False,
        val_series: Optional[TimeSeries] = None,
        use_fitted_values: bool = False,
        metric: Callable[[TimeSeries, TimeSeries], float] = metrics.mape,
        reduction: Callable[[np.ndarray], float] = np.mean,
        verbose=False,
        n_jobs: int = 1,
        n_random_samples: Optional[Union[int, float]] = None,
    ) -> Tuple["ForecastingModel", Dict[str, Any], float]:
        """
        Find the best hyper-parameters among a given set using a grid search.

        This function has 3 modes of operation: Expanding window mode, split mode and fitted value mode.
        The three modes of operation evaluate every possible combination of hyper-parameter values
        provided in the `parameters` dictionary by instantiating the `model_class` subclass
        of ForecastingModel with each combination, and returning the best-performing model with regard
        to the `metric` function. The `metric` function is expected to return an error value,
        thus the model resulting in the smallest `metric` output will be chosen.

        The relationship of the training data and test data depends on the mode of operation.

        Expanding window mode (activated when `forecast_horizon` is passed):
        For every hyperparameter combination, the model is repeatedly trained and evaluated on different
        splits of `series`. This process is accomplished by using
        the :func:`backtest` function as a subroutine to produce historic forecasts starting from `start`
        that are compared against the ground truth values of `series`.
        Note that the model is retrained for every single prediction, thus this mode is slower.

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
        several models in parallel (when ``n_jobs != 1``). If this cannot be avoided, then gridsearch
        should be redefined, forcing ``n_jobs = 1``.

        Currently this method only supports deterministic predictions (i.e. when models' predictions
        have only 1 sample).

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
            The integer value of the forecasting horizon. Activates expanding window mode.
        stride
            The number of time steps between two consecutive predictions. Only used in expanding window mode.
        start
            The ``int``, ``float`` or ``pandas.Timestamp`` that represents the starting point in the time index
            of `series` from which predictions will be made to evaluate the model.
            For a detailed description of how the different data types are interpreted, please see the documentation
            for `ForecastingModel.backtest`.
        last_points_only
            Whether to use the whole forecasts or only the last point of each forecast to compute the error
        val_series
            The TimeSeries instance used for validation in split mode. If provided, this series must start right after
            the end of `series`; so that a proper comparison of the forecast can be made.
        use_fitted_values
            If `True`, uses the comparison with the fitted values.
            Raises an error if ``fitted_values`` is not an attribute of `model_class`.
        metric
            A function that takes two TimeSeries instances as inputs (actual and prediction, in this order),
            and returns a float error value.
        reduction
            A reduction function (mapping array to float) describing how to aggregate the errors obtained
            on the different validation series when backtesting. By default it'll compute the mean of errors.
        verbose
            Whether to print progress.
        n_jobs
            The number of jobs to run in parallel. Parallel jobs are created only when there are two or more parameters
            combinations to evaluate. Each job will instantiate, train, and evaluate a different instance of the model.
            Defaults to `1` (sequential). Setting the parameter to `-1` means using all the available cores.
        n_random_samples
            The number/ratio of hyperparameter combinations to select from the full parameter grid. This will perform
            a random search instead of using the full grid.
            If an integer, `n_random_samples` is the number of parameter combinations selected from the full grid and
            must be between `0` and the total number of parameter combinations.
            If a float, `n_random_samples` is the ratio of parameter combinations selected from the full grid and must
            be between `0` and `1`. Defaults to `None`, for which random selection will be ignored.

        Returns
        -------
        ForecastingModel, Dict, float
            A tuple containing an untrained `model_class` instance created from the best-performing hyper-parameters,
            along with a dictionary containing these best hyper-parameters,
            and metric score for the best hyper-parameters.
        """
        raise_if_not(
            (forecast_horizon is not None)
            + (val_series is not None)
            + use_fitted_values
            == 1,
            "Please pass exactly one of the arguments 'forecast_horizon', "
            "'val_target_series' or 'use_fitted_values'.",
            logger,
        )

        if use_fitted_values:
            raise_if_not(
                hasattr(model_class(), "fitted_values"),
                "The model must have a fitted_values attribute to compare with the train TimeSeries",
                logger,
            )

        elif val_series is not None:
            raise_if_not(
                series.width == val_series.width,
                "Training and validation series require the same number of components.",
                logger,
            )

        # TODO: here too I'd say we can leave these checks to the models
        # if covariates is not None:
        #     raise_if_not(series.has_same_time_as(covariates), 'The provided series and covariates must have the '
        #                                                       'same time axes.')

        # compute all hyperparameter combinations from selection
        params_cross_product = list(product(*parameters.values()))

        # If n_random_samples has been set, randomly select a subset of the full parameter cross product to search with
        if n_random_samples is not None:
            params_cross_product = model_class._sample_params(
                params_cross_product, n_random_samples
            )

        # iterate through all combinations of the provided parameters and choose the best one
        iterator = _build_tqdm_iterator(
            zip(params_cross_product), verbose, total=len(params_cross_product)
        )

        def _evaluate_combination(param_combination) -> float:
            param_combination_dict = dict(
                list(zip(parameters.keys(), param_combination))
            )
            if param_combination_dict.get("model_name", None):
                current_time = time.strftime("%Y-%m-%d_%H.%M.%S.%f", time.localtime())
                param_combination_dict[
                    "model_name"
                ] = f"{current_time}_{param_combination_dict['model_name']}"

            model = model_class(**param_combination_dict)
            if use_fitted_values:  # fitted value mode
                model._fit_wrapper(series, past_covariates, future_covariates)
                fitted_values = TimeSeries.from_times_and_values(
                    series.time_index, model.fitted_values
                )
                error = metric(series, fitted_values)
            elif val_series is None:  # expanding window mode
                error = model.backtest(
                    series=series,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                    num_samples=1,
                    start=start,
                    forecast_horizon=forecast_horizon,
                    stride=stride,
                    metric=metric,
                    reduction=reduction,
                    last_points_only=last_points_only,
                    verbose=verbose,
                )
            else:  # split mode
                model._fit_wrapper(series, past_covariates, future_covariates)
                pred = model._predict_wrapper(
                    len(val_series),
                    series,
                    past_covariates,
                    future_covariates,
                    num_samples=1,
                    verbose=verbose,
                )
                error = metric(val_series, pred)

            return float(error)

        errors: List[float] = _parallel_apply(
            iterator, _evaluate_combination, n_jobs, {}, {}
        )

        min_error = min(errors)

        best_param_combination = dict(
            list(zip(parameters.keys(), params_cross_product[errors.index(min_error)]))
        )

        logger.info("Chosen parameters: " + str(best_param_combination))

        return model_class(**best_param_combination), best_param_combination, min_error

    def residuals(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        forecast_horizon: int = 1,
        retrain: bool = True,
        verbose: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Compute the residuals produced by this model on a (or sequence of) univariate  time series.

        This function computes the difference between the actual observations from `series`
        and the fitted values vector `p` obtained by training the model on `series`.
        For every index `i` in `series`, `p[i]` is computed by training the model on
        ``series[:(i - forecast_horizon)]`` and forecasting `forecast_horizon` into the future.
        (`p[i]` will be set to the last value of the predicted series.)
        The vector of residuals will be shorter than `series` due to the minimum
        training series length required by the model and the gap introduced by `forecast_horizon`.
        Most commonly, the term "residuals" implies a value for `forecast_horizon` of 1; but
        this can be configured.

        This method works only on univariate series. It uses the median
        prediction (when dealing with stochastic forecasts).

        Parameters
        ----------
        series
            The univariate TimeSeries instance which the residuals will be computed for.
        past_covariates
            One or several past-observed covariate time series.
        future_covariates
            One or several future-known covariate time series.
        forecast_horizon
            The forecasting horizon used to predict each fitted value.
        retrain
            Whether to train the model at each iteration, for models that support it.
            If False, the model is not trained at all. Default: True
        verbose
            Whether to print progress.
        Returns
        -------
        TimeSeries (or Sequence[TimeSeries])
            The vector of residuals.
        """

        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)

        raise_if_not(
            all([serie.is_univariate for serie in series]),
            "Each series in the sequence must be univariate.",
            logger,
        )

        residuals_list = []
        # compute residuals
        for idx, target_ts in enumerate(series):
            # get first index not contained in the first training set
            first_index = target_ts.time_index[self.min_train_series_length]

            # compute fitted values
            forecasts = self.historical_forecasts(
                series=target_ts,
                past_covariates=past_covariates[idx] if past_covariates else None,
                future_covariates=future_covariates[idx] if future_covariates else None,
                start=first_index,
                forecast_horizon=forecast_horizon,
                stride=1,
                retrain=retrain,
                last_points_only=True,
                verbose=verbose,
            )
            series_trimmed = target_ts.slice_intersect(forecasts)
            residuals_list.append(
                series_trimmed
                - (
                    forecasts.quantile_timeseries(quantile=0.5)
                    if forecasts.is_stochastic
                    else forecasts
                )
            )

        return residuals_list if len(residuals_list) > 1 else residuals_list[0]

    def initialize_encoders(self) -> SequentialEncoder:
        """instantiates the SequentialEncoder object based on self._model_encoder_settings and parameter
        ``add_encoders`` used at model creation"""
        (
            input_chunk_length,
            output_chunk_length,
            takes_past_covariates,
            takes_future_covariates,
            lags_past_covariates,
            lags_future_covariates,
        ) = self._model_encoder_settings

        return SequentialEncoder(
            add_encoders=self.add_encoders,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            lags_past_covariates=lags_past_covariates,
            lags_future_covariates=lags_future_covariates,
            takes_past_covariates=takes_past_covariates,
            takes_future_covariates=takes_future_covariates,
        )

    def generate_fit_encodings(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> Tuple[
        Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]
    ]:
        """Generates the covariate encodings that were used/generated for fitting the model and returns a tuple of
        past, and future covariates series with the original and encoded covariates stacked together. The encodings are
        generated by the encoders defined at model creation with parameter `add_encoders`. Pass the same `series`,
        `past_covariates`, and  `future_covariates` that you used to train/fit the model.

        Parameters
        ----------
        series
            The series or sequence of series with the target values used when fitting the model.
        past_covariates
            Optionally, the series or sequence of series with the past-observed covariates used when fitting the model.
        future_covariates
            Optionally, the series or sequence of series with the future-known covariates used when fitting the model.

        Returns
        -------
        Tuple[Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]]
            A tuple of (past covariates, future covariates). Each covariate contains the original as well as the
            encoded covariates.
        """
        raise_if(
            self.encoders is None or not self.encoders.encoding_available,
            "Encodings are not available. Consider adding parameter `add_encoders` at model creation and fitting the "
            "model with `model.fit()` before.",
            logger=logger,
        )
        return self.encoders.encode_train(
            target=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

    def generate_predict_encodings(
        self,
        n: int,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> Tuple[
        Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]
    ]:
        """Generates covariate encodings for the inference/prediction set and returns a tuple of past, and future
        covariates series with the original and encoded covariates stacked together. The encodings are generated by the
        encoders defined at model creation with parameter `add_encoders`. Pass the same `series`, `past_covariates`,
        and `future_covariates` that you intend to use for prediction.

        Parameters
        ----------
        n
            The number of prediction time steps after the end of `series` intended to be used for prediction.
        series
            The series or sequence of series with target values intended to be used for prediction.
        past_covariates
            Optionally, the past-observed covariates series intended to be used for prediction. The dimensions must
            match those of the covariates used for training.
        future_covariates
            Optionally, the future-known covariates series intended to be used for prediction. The dimensions must
            match those of the covariates used for training.

        Returns
        -------
        Tuple[Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]]
            A tuple of (past covariates, future covariates). Each covariate contains the original as well as the
            encoded covariates.
        """
        raise_if(
            self.encoders is None or not self.encoders.encoding_available,
            "Encodings are not available. Consider adding parameter `add_encoders` at model creation and fitting the "
            "model with `model.fit()` before.",
            logger=logger,
        )
        return self.encoders.encode_inference(
            n=n,
            target=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

    @property
    @abstractmethod
    def _model_encoder_settings(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        bool,
        bool,
        Optional[List[int]],
        Optional[List[int]],
    ]:
        """Abstract property that returns model specific encoder settings that are used to initialize the encoders.

        Must return Tuple (input_chunk_length, output_chunk_length, takes_past_covariates, takes_future_covariates,
        lags_past_covariates, lags_future_covariates).
        """
        pass

    @classmethod
    def _sample_params(model_class, params, n_random_samples):
        """Select the absolute number of samples randomly if an integer has been supplied. If a float has been
        supplied, select a fraction"""

        if isinstance(n_random_samples, int):
            raise_if_not(
                (n_random_samples > 0) and (n_random_samples <= len(params)),
                "If supplied as an integer, n_random_samples must be greater than 0 and less"
                "than or equal to the size of the cartesian product of the hyperparameters.",
            )
            return sample(params, n_random_samples)

        if isinstance(n_random_samples, float):
            raise_if_not(
                (n_random_samples > 0.0) and (n_random_samples <= 1.0),
                "If supplied as a float, n_random_samples must be greater than 0.0 and less than 1.0.",
            )
            return sample(params, int(n_random_samples * len(params)))

    def _extract_model_creation_params(self):
        """extracts immutable model creation parameters from `ModelMeta` and deletes reference."""
        model_params = copy.deepcopy(self._model_call)
        del self.__class__._model_call
        return model_params

    def untrained_model(self):
        return self.__class__(**self.model_params)

    @property
    def model_params(self) -> dict:
        return (
            self._model_params if hasattr(self, "_model_params") else self._model_call
        )

    @classmethod
    def _default_save_path(cls) -> str:
        return f"{cls.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"

    def save(self, path: Optional[Union[str, BinaryIO]] = None, **pkl_kwargs) -> None:
        """
        Saves the model under a given path or file handle.

        Example for saving and loading a :class:`RegressionModel`:

            .. highlight:: python
            .. code-block:: python

                from darts.models import RegressionModel

                model = RegressionModel(lags=4)

                model.save("my_model.pkl")
                model_loaded = RegressionModel.load("my_model.pkl")
            ..

        Parameters
        ----------
        path
            Path or file handle under which to save the model at its current state. If no path is specified, the model
            is automatically saved under ``"{ModelClass}_{YYYY-mm-dd_HH:MM:SS}.pkl"``.
            E.g., ``"RegressionModel_2020-01-01_12:00:00.pkl"``.
        pkl_kwargs
            Keyword arguments passed to `pickle.dump()`
        """

        if path is None:
            # default path
            path = self._default_save_path() + ".pkl"

        if isinstance(path, str):
            # save the whole object using pickle
            with open(path, "wb") as handle:
                pickle.dump(obj=self, file=handle, **pkl_kwargs)
        else:
            # save the whole object using pickle
            pickle.dump(obj=self, file=path, **pkl_kwargs)

    @staticmethod
    def load(path: Union[str, BinaryIO]) -> "ForecastingModel":
        """
        Loads the model from a given path or file handle.

        Parameters
        ----------
        path
            Path or file handle from which to load the model.
        """

        if isinstance(path, str):
            raise_if_not(
                os.path.exists(path),
                f"The file {path} doesn't exist",
                logger,
            )

            with open(path, "rb") as handle:
                model = pickle.load(file=handle)
        else:

            model = pickle.load(file=path)

        return model

    def _assert_univariate(self, series: TimeSeries):
        if not series.is_univariate:
            raise_log(
                ValueError("This model only supports univariate TimeSeries instances")
            )


class LocalForecastingModel(ForecastingModel, ABC):
    """The base class for "local" forecasting models, handling only single univariate time series.

    Local Forecasting Models (LFM) are models that can be trained on a single univariate target series only. In Darts,
    most models in this category tend to be simpler statistical models (such as ETS or FFT). LFMs usually train on
    the entire target series supplied when calling :func:`fit()` at once. They can also predict in one go with
    :func:`predict()` for any number of predictions `n` after the end of the training series.

    All implementations must implement the `fit()` and `predict()` methods.
    """

    def __init__(self, add_encoders: Optional[dict] = None):
        super().__init__(add_encoders=add_encoders)

    @property
    def _model_encoder_settings(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        bool,
        bool,
        Optional[List[int]],
        Optional[List[int]],
    ]:
        return None, None, False, False, None, None

    @abstractmethod
    def fit(self, series: TimeSeries) -> "LocalForecastingModel":
        super().fit(series)
        series._assert_deterministic()


class GlobalForecastingModel(ForecastingModel, ABC):
    """The base class for "global" forecasting models, handling several time series and optional covariates.

    Global forecasting models expand upon the functionality of `ForecastingModel` in 4 ways:
    1. Models can be fitted on many series (multivariate or univariate) with different indices.
    2. The input series used by :func:`predict()` can be different from the series used to fit the model.
    3. Covariates can be supported (multivariate or univariate).
    4. They can allow for multivariate target series and covariates.

    The name "global" stems from the fact that a training set of a forecasting model of this class is not constrained
    to a temporally contiguous, "local", time series.

    All implementations must implement the :func:`fit()` and :func:`predict()` methods.
    The :func:`fit()` method is meant to train the model on one or several training time series, along with optional
    covariates.

    If :func:`fit()` has been called with only one training and covariate series as argument, then
    calling :func:`predict()` will forecast the future of this series. Otherwise, the user has to
    provide to :func:`predict()` the series they want to forecast, as well as covariates, if needed.
    """

    def __init__(self, add_encoders: Optional[dict] = None):
        super().__init__(add_encoders=add_encoders)

    @abstractmethod
    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> "GlobalForecastingModel":
        """Fit/train the model on (potentially multiple) series.

        Optionally, one or multiple past and/or future covariates series can be provided as well.
        The number of covariates series must match the number of target series.

        Parameters
        ----------
        series
            One or several target time series. The model will be trained to forecast these time series.
            The series may or may not be multivariate, but if multiple series are provided
            they must have the same number of components.
        past_covariates
            One or several past-observed covariate time series. These time series will not be forecast, but can
            be used by some models as an input. The covariate(s) may or may not be multivariate, but if multiple
            covariates are provided they must have the same number of components. If `past_covariates` is provided,
            it must contain the same number of series as `series`.
        future_covariates
            One or several future-known covariate time series. These time series will not be forecast, but can
            be used by some models as an input. The covariate(s) may or may not be multivariate, but if multiple
            covariates are provided they must have the same number of components. If `future_covariates` is provided,
            it must contain the same number of series as `series`.

        Returns
        -------
        self
            Fitted model.
        """

        if isinstance(series, TimeSeries):
            # if only one series is provided, save it for prediction time (including covariates, if available)
            self.training_series = series
            self.static_covariates = series.static_covariates
            if past_covariates is not None:
                self.past_covariate_series = past_covariates
            if future_covariates is not None:
                self.future_covariate_series = future_covariates
        else:
            self.static_covariates = series[0].static_covariates

            if past_covariates is not None:
                self._expect_past_covariates = True
            if future_covariates is not None:
                self._expect_future_covariates = True

        if past_covariates is not None:
            self._uses_past_covariates = True
        if future_covariates is not None:
            self._uses_future_covariates = True

        self._fit_called = True

    @abstractmethod
    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        verbose: bool = False,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Forecasts values for `n` time steps after the end of the series.

        If :func:`fit()` has been called with only one ``TimeSeries`` as argument, then the `series` argument of
        this function is optional, and it will simply produce the next `horizon` time steps forecast.
        The `past_covariates` and `future_covariates` arguments also don't have to be provided again in this case.

        If :func:`fit()` has been called with `series` specified as a ``Sequence[TimeSeries]`` (i.e., the model
        has been trained on multiple time series), the `series` argument must be specified.

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
            The series whose future values will be predicted.
        past_covariates
            One past-observed covariate time series for every input time series in `series`. They must match the
            past covariates that have been used with the :func:`fit()` function for training in terms of dimension.
        future_covariates
            One future-known covariate time series for every input time series in `series`. They must match the
            past covariates that have been used with the :func:`fit()` function for training in terms of dimension.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.

        Returns
        -------
        Union[TimeSeries, Sequence[TimeSeries]]
            If `series` is not specified, this function returns a single time series containing the `n`
            next points after then end of the training series.
            If `series` is given and is a simple ``TimeSeries``, this function returns the `n` next points
            after the end of `series`.
            If `series` is given and is a sequence of several time series, this function returns
            a sequence where each element contains the corresponding `n` points forecasts.
        """
        super().predict(n, num_samples)
        if self._expect_past_covariates and past_covariates is None:
            raise_log(
                ValueError(
                    "The model has been trained with past covariates. Some matching past_covariates "
                    "have to be provided to `predict()`."
                )
            )
        if self._expect_future_covariates and future_covariates is None:
            raise_log(
                ValueError(
                    "The model has been trained with future covariates. Some matching future_covariates "
                    "have to be provided to `predict()`."
                )
            )

    def _predict_wrapper(
        self,
        n: int,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
        num_samples: int,
        verbose: bool = False,
    ) -> TimeSeries:
        return self.predict(
            n,
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=num_samples,
            verbose=verbose,
        )

    def _fit_wrapper(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
    ):
        self.fit(
            series=series,
            past_covariates=past_covariates if self.supports_past_covariates else None,
            future_covariates=future_covariates
            if self.supports_future_covariates
            else None,
        )

    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        """GlobalForecastingModel supports historical forecasts without retraining the model"""
        return True


class FutureCovariatesLocalForecastingModel(LocalForecastingModel, ABC):
    """The base class for future covariates "local" forecasting models, handling single uni- or multivariate target
    and optional future covariates time series.

    Future Covariates Local Forecasting Models (FC-LFM) are models that can be trained on a single uni- or multivariate
    target and optional future covariates series. In Darts, most models in this category tend to be simpler statistical
    models (such as ARIMA). FC-LFMs usually train on the entire target and future covariates series supplied when
    calling :func:`fit()` at once. They can also predict in one go with :func:`predict()` for any number of predictions
    `n` after the end of the training series. When using future covariates, the values for the future `n` prediction
    steps must be given in the covariate series.

    All implementations must implement the :func:`_fit()` and :func:`_predict()` methods.
    """

    def fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        """Fit/train the model on the (single) provided series.

        Optionally, a future covariates series can be provided as well.

        Parameters
        ----------
        series
            The model will be trained to forecast this time series. Can be multivariate if the model supports it.
        future_covariates
            A time series of future-known covariates. This time series will not be forecasted, but can be used by
            some models as an input. It must contain at least the same time steps/indices as the target `series`.
            If it is longer than necessary, it will be automatically trimmed.

        Returns
        -------
        self
            Fitted model.
        """

        if future_covariates is not None:
            if not series.has_same_time_as(future_covariates):
                # fit() expects future_covariates to have same time as the target, so we intersect it here
                future_covariates = future_covariates.slice_intersect(series)

            raise_if_not(
                series.has_same_time_as(future_covariates),
                "The provided `future_covariates` series must contain at least the same time steps/"
                "indices as the target `series`.",
                logger=logger,
            )
            self._expect_future_covariates = True

        self.encoders = self.initialize_encoders()
        if self.encoders.encoding_available:
            _, future_covariates = self.generate_fit_encodings(
                series=series,
                past_covariates=None,
                future_covariates=future_covariates,
            )

        super().fit(series)

        return self._fit(series, future_covariates=future_covariates)

    @abstractmethod
    def _fit(self, series: TimeSeries, future_covariates: Optional[TimeSeries] = None):
        """Fits/trains the model on the provided series.
        DualCovariatesModels must implement the fit logic in this method.
        """
        pass

    def predict(
        self,
        n: int,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        **kwargs,
    ) -> TimeSeries:
        """Forecasts values for `n` time steps after the end of the training series.

        If some future covariates were specified during the training, they must also be specified here.

        Parameters
        ----------
        n
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        future_covariates
            The time series of future-known covariates which can be fed as input to the model. It must correspond to
            the covariate time series that has been used with the :func:`fit()` method for training, and it must
            contain at least the next `n` time steps/indices after the end of the training target series.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.

        Returns
        -------
        TimeSeries, a single time series containing the `n` next points after then end of the training series.
        """

        super().predict(n, num_samples)

        # avoid generating encodings again if subclass has already generated them
        if not self._supress_generate_predict_encoding:
            self._verify_passed_predict_covariates(future_covariates)
            if self.encoders is not None and self.encoders.encoding_available:
                _, future_covariates = self.generate_predict_encodings(
                    n=n,
                    series=self.training_series,
                    past_covariates=None,
                    future_covariates=future_covariates,
                )

        if future_covariates is not None:
            start = self.training_series.end_time() + self.training_series.freq

            invalid_time_span_error = (
                f"For the given forecasting horizon `n={n}`, the provided `future_covariates` "
                f"series must contain at least the next `n={n}` time steps/indices after the "
                f"end of the target `series` that was used to train the model."
            )

            # we raise an error here already to avoid getting error from empty TimeSeries creation
            raise_if_not(
                future_covariates.end_time() >= start, invalid_time_span_error, logger
            )

            offset = (
                n - 1
                if isinstance(future_covariates.time_index, pd.DatetimeIndex)
                else n
            )
            future_covariates = future_covariates.slice(
                start, start + offset * self.training_series.freq
            )

            raise_if_not(
                len(future_covariates) == n,
                invalid_time_span_error,
                logger,
            )

        return self._predict(
            n, future_covariates=future_covariates, num_samples=num_samples, **kwargs
        )

    @abstractmethod
    def _predict(
        self,
        n: int,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        verbose: bool = False,
        **kwargs,
    ) -> TimeSeries:
        """Forecasts values for a certain number of time steps after the end of the series.
        DualCovariatesModels must implement the predict logic in this method.
        """
        pass

    def _fit_wrapper(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
    ):
        self.fit(series, future_covariates=future_covariates)

    def _predict_wrapper(
        self,
        n: int,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
        num_samples: int,
        verbose: bool = False,
    ) -> TimeSeries:
        return self.predict(
            n,
            future_covariates=future_covariates,
            num_samples=num_samples,
            verbose=verbose,
        )

    @property
    def _model_encoder_settings(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        bool,
        bool,
        Optional[List[int]],
        Optional[List[int]],
    ]:
        return None, None, False, True, None, None

    def _verify_passed_predict_covariates(self, future_covariates):
        """Simple check if user supplied/did not supply covariates as done at fitting time."""
        if self._expect_future_covariates and future_covariates is None:
            raise_log(
                ValueError(
                    "The model has been trained with `future_covariates` variable. Some matching "
                    "`future_covariates` variables have to be provided to `predict()`."
                )
            )
        if not self._expect_future_covariates and future_covariates is not None:
            raise_log(
                ValueError(
                    "The model has been trained without `future_covariates` variable, but the "
                    "`future_covariates` parameter provided to `predict()` is not None.",
                )
            )

    @property
    def _supress_generate_predict_encoding(self) -> bool:
        """Controls wether encodings should be generated in :func:`FutureCovariatesLocalForecastingModel.predict()``"""
        return False


class TransferableFutureCovariatesLocalForecastingModel(
    FutureCovariatesLocalForecastingModel, ABC
):
    """The base class for transferable future covariates "local" forecasting models, handling single uni- or
    multivariate target and optional future covariates time series. Additionally, at prediction time, it can be
    applied to new data unrelated to the original series used for fitting the model.

    Transferable Future Covariates Local Forecasting Models (TFC-LFM) are models that can be trained on a single uni-
    or multivariate target and optional future covariates series. Additionally, at prediction time, it can be applied
    to new data unrelated to the original series used for fitting the model. Currently in Darts, all models in this
    category wrap to statsmodel models such as VARIMA. TFC-LFMs usually train on the entire target and future covariates
    series supplied when calling :func:`fit()` at once. They can also predict in one go with :func:`predict()`
    for any number of predictions `n` after the end of the training series. When using future covariates, the values
    for the future `n` prediction steps must be given in the covariate series.

    All implementations must implement the :func:`_fit()` and :func:`_predict()` methods.
    """

    def predict(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        **kwargs,
    ) -> TimeSeries:
        """If the `series` parameter is not set, forecasts values for `n` time steps after the end of the training
        series. If some future covariates were specified during the training, they must also be specified here.

        If the `series` parameter is set, forecasts values for `n` time steps after the end of the new target
        series. If some future covariates were specified during the training, they must also be specified here.

        Parameters
        ----------
        n
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        series
            Optionally, a new target series whose future values will be predicted. Defaults to `None`, meaning that the
            model will forecast the future value of the training series.
        future_covariates
            The time series of future-known covariates which can be fed as input to the model. It must correspond to
            the covariate time series that has been used with the :func:`fit()` method for training.

            If `series` is not set, it must contain at least the next `n` time steps/indices after the end of the
            training target series. If `series` is set, it must contain at least the time steps/indices corresponding
            to the new target series (historic future covariates), plus the next `n` time steps/indices after the end.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Should be left set to 1
            for deterministic models.

        Returns
        -------
        TimeSeries, a single time series containing the `n` next points after then end of the training series.
        """
        self._verify_passed_predict_covariates(future_covariates)
        if self.encoders is not None and self.encoders.encoding_available:
            _, future_covariates = self.generate_predict_encodings(
                n=n,
                series=series if series is not None else self.training_series,
                past_covariates=None,
                future_covariates=future_covariates,
            )

        historic_future_covariates = None

        if series is not None and future_covariates:
            raise_if_not(
                future_covariates.start_time() <= series.start_time()
                and future_covariates.end_time() >= series.end_time() + n * series.freq,
                "The provided `future_covariates` related to the new target series must contain at least the same time"
                "steps/indices as the target `series` + `n`.",
                logger,
            )
            # splitting the future covariates
            (
                historic_future_covariates,
                future_covariates,
            ) = future_covariates.split_after(series.end_time())

            # in case future covariates have more values on the left end side that we don't need
            if not series.has_same_time_as(historic_future_covariates):
                historic_future_covariates = historic_future_covariates.slice_intersect(
                    series
                )

        # FutureCovariatesLocalForecastingModel performs some checks on self.training_series. We temporary replace
        # that with the new ts
        if series is not None:
            self._orig_training_series = self.training_series
            self.training_series = series

        result = super().predict(
            n=n,
            series=series,
            historic_future_covariates=historic_future_covariates,
            future_covariates=future_covariates,
            num_samples=num_samples,
            **kwargs,
        )

        # restoring the original training ts
        if series is not None:
            self.training_series = self._orig_training_series

        return result

    def generate_predict_encodings(
        self,
        n: int,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> Tuple[
        Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]
    ]:
        raise_if(
            self.encoders is None or not self.encoders.encoding_available,
            "Encodings are not available. Consider adding parameter `add_encoders` at model creation and fitting the "
            "model with `model.fit()` before.",
            logger=logger,
        )
        _, future_covariates_future = self.encoders.encode_inference(
            n=n,
            target=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        if future_covariates is not None:
            return _, future_covariates_future

        _, future_covariates_historic = self.encoders.encode_train(
            target=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        return _, future_covariates_historic.append(future_covariates_future)

    @abstractmethod
    def _predict(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        historic_future_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        verbose: bool = False,
    ) -> TimeSeries:
        """Forecasts values for a certain number of time steps after the end of the series.
        TransferableFutureCovariatesLocalForecastingModel must implement the predict logic in this method.
        """
        pass

    def _predict_wrapper(
        self,
        n: int,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
        num_samples: int,
        verbose: bool = False,
    ) -> TimeSeries:
        return self.predict(
            n=n,
            series=series,
            future_covariates=future_covariates,
            num_samples=num_samples,
            verbose=verbose,
        )

    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        return True

    @property
    def _supress_generate_predict_encoding(self) -> bool:
        return True

    @property
    def extreme_lags(self):
        return (-1, 1, None, 0, 0)
