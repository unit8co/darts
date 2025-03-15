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
import io
import os
import pickle
import sys
import time
from abc import ABC, ABCMeta, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from itertools import product
from random import sample
from typing import Any, BinaryIO, Callable, Literal, Optional, Union

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
import pandas as pd

from darts import metrics
from darts.dataprocessing.encoders import SequentialEncoder
from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import BaseDataTransformer
from darts.logging import get_logger, raise_if, raise_if_not, raise_log
from darts.metrics.metrics import METRIC_TYPE
from darts.timeseries import TimeSeries
from darts.utils import _build_tqdm_iterator, _parallel_apply, _with_sanity_checks
from darts.utils.historical_forecasts.utils import (
    _adjust_historical_forecasts_time_index,
    _apply_data_transformers,
    _apply_inverse_data_transformers,
    _convert_data_transformers,
    _extend_series_for_overlap_end,
    _get_historical_forecast_predict_index,
    _get_historical_forecast_train_index,
    _historical_forecasts_general_checks,
    _historical_forecasts_sanitize_kwargs,
    _process_historical_forecast_for_backtest,
    _reconciliate_historical_time_indices,
)
from darts.utils.timeseries_generation import (
    _build_forecast_series,
    _generate_new_dates,
)
from darts.utils.ts_utils import (
    SeriesType,
    get_series_seq_type,
    get_single_series,
    series2seq,
)
from darts.utils.utils import (
    generate_index,
    likelihood_component_names,
    quantile_interval_names,
    quantile_names,
)

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
        all_params = OrderedDict([
            (p.name, p.default) for p in sig.parameters.values() if not p.name == "self"
        ])

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

        self._expect_past_covariates, self._uses_past_covariates = False, False
        self._expect_future_covariates, self._uses_future_covariates = False, False
        # for static covariates there is the option to consider static covariates or ignore them
        self._considers_static_covariates = False
        self._expect_static_covariates, self._uses_static_covariates = False, False

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
        self.encoders = self.initialize_encoders(default=True)

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
        if not isinstance(series, TimeSeries):
            raise_log(
                ValueError("Train `series` must be a single `TimeSeries`."),
                logger=logger,
            )
        if not len(series) >= self.min_train_series_length:
            raise_log(
                ValueError(
                    f"Train series only contains {len(series)} elements"
                    f" but {str(self)} model requires at least {self.min_train_series_length} entries"
                ),
                logger=logger,
            )
        self.training_series = series
        self._fit_called = True

        if series.has_range_index:
            self._supports_range_index

    @property
    def _supports_range_index(self) -> bool:
        """Checks if the forecasting model supports a range index.
        Some models may not support this, if for instance they rely on underlying dates.

        By default, returns True. Needs to be overwritten by models that do not support
        range indexing and raise meaningful exception.
        """
        return True

    @property
    def supports_probabilistic_prediction(self) -> bool:
        """
        Checks if the forecasting model with this configuration supports probabilistic predictions.

        By default, returns False. Needs to be overwritten by models that do support
        probabilistic predictions.
        """
        return False

    @property
    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        """
        Checks if the forecasting model supports historical forecasts without retraining
        the model.

        By default, returns False. Needs to be overwritten by models that do
        support historical forecasts without retraining.
        """
        return False

    @property
    @abstractmethod
    def supports_multivariate(self) -> bool:
        """
        Whether the model considers more than one variate in the time series.
        """

    @property
    def supports_past_covariates(self) -> bool:
        """
        Whether model supports past covariates
        """
        return "past_covariates" in inspect.signature(self.fit).parameters.keys()

    @property
    def supports_future_covariates(self) -> bool:
        """
        Whether model supports future covariates
        """
        return "future_covariates" in inspect.signature(self.fit).parameters.keys()

    @property
    def supports_static_covariates(self) -> bool:
        """
        Whether model supports static covariates
        """
        return False

    @property
    def supports_sample_weight(self) -> bool:
        """
        Whether model supports sample weight for training.
        """
        return False

    @property
    def supports_likelihood_parameter_prediction(self) -> bool:
        """
        Whether model instance supports direct prediction of likelihood parameters
        """
        return getattr(self, "likelihood", None) is not None

    @property
    @abstractmethod
    def supports_transferrable_series_prediction(self) -> bool:
        """
        Whether the model supports prediction for any input `series`.
        """

    @property
    def uses_past_covariates(self) -> bool:
        """
        Whether the model uses past covariates, once fitted.
        """
        return self._uses_past_covariates

    @property
    def uses_future_covariates(self) -> bool:
        """
        Whether the model uses future covariates, once fitted.
        """
        return self._uses_future_covariates

    @property
    def uses_static_covariates(self) -> bool:
        """
        Whether the model uses static covariates, once fitted.
        """
        return self._uses_static_covariates

    @property
    def considers_static_covariates(self) -> bool:
        """
        Whether the model considers static covariates, if there are any.
        """
        return self._considers_static_covariates

    @property
    def supports_optimized_historical_forecasts(self) -> bool:
        """
        Whether the model supports optimized historical forecasts
        """
        return False

    @property
    def output_chunk_length(self) -> Optional[int]:
        """
        Number of time steps predicted at once by the model, not defined for statistical models.
        """
        return None

    @property
    def output_chunk_shift(self) -> int:
        """
        Number of time steps that the output/prediction starts after the end of the input.
        """
        return 0

    @abstractmethod
    def predict(
        self,
        n: int,
        num_samples: int = 1,
        verbose: bool = False,
        show_warnings: bool = True,
    ) -> TimeSeries:
        """Forecasts values for `n` time steps after the end of the training series.

        Parameters
        ----------
        n
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Must be `1` for deterministic models.
        verbose
            Optionally, set the prediction verbosity. Not effective for all models.
        show_warnings
            Optionally, control whether warnings are shown. Not effective for all models.

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
        is_autoregression = (
            False
            if self.output_chunk_length is None
            else (n > self.output_chunk_length)
        )
        if self.output_chunk_shift and is_autoregression:
            raise_log(
                ValueError(
                    "Cannot perform auto-regression `(n > output_chunk_length)` with a model that uses a "
                    "shifted output chunk `(output_chunk_shift > 0)`."
                ),
                logger=logger,
            )

        if not self.supports_probabilistic_prediction and num_samples > 1:
            raise_log(
                ValueError(
                    "`num_samples > 1` is only supported for probabilistic models."
                ),
                logger,
            )

    def _fit_wrapper(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        **kwargs,
    ):
        add_kwargs = {}
        # handle past and future covariates based on model support
        for series_, series_name in zip(
            [past_covariates, future_covariates, sample_weight],
            ["past_covariates", "future_covariates", "sample_weight"],
        ):
            if getattr(self, f"supports_{series_name}"):
                add_kwargs[series_name] = series_
            elif series_ is not None:
                raise_log(
                    ValueError(f"Model cannot be fit/trained with `{series_name}`."),
                    logger,
                )
        self.fit(series=series, **add_kwargs, **kwargs)

    def _predict_wrapper(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        predict_likelihood_parameters: bool = False,
        **kwargs,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        add_kwargs = {}
        # not all models supports input `series` at inference
        if self.supports_transferrable_series_prediction:
            add_kwargs["series"] = series

        # even if predict() accepts covariates, the model might not support them at inference
        for covs, name in zip(
            [past_covariates, future_covariates],
            ["past_covariates", "future_covariates"],
        ):
            if getattr(self, f"supports_{name}"):
                add_kwargs[name] = covs
            elif covs is not None:
                raise_log(
                    ValueError(
                        f"Model prediction does not support `{name}`, either because it "
                        f"does not support `{name}` in general, or because it was fit/trained "
                        f"without using `{name}`."
                    ),
                    logger,
                )

        if self.supports_likelihood_parameter_prediction:
            add_kwargs["predict_likelihood_parameters"] = predict_likelihood_parameters
        return self.predict(n=n, **add_kwargs, **kwargs)

    @property
    def min_train_series_length(self) -> int:
        """
        The minimum required length for the training series.
        """
        return 3

    @property
    def min_train_samples(self) -> int:
        """
        The minimum number of samples for training the model.
        """
        return 1

    @property
    @abstractmethod
    def extreme_lags(
        self,
    ) -> tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        int,
        Optional[int],
    ]:
        """
        A 8-tuple containing in order:
        (min target lag, max target lag, min past covariate lag, max past covariate lag, min future covariate
        lag, max future covariate lag, output shift, max target lag train (only for RNNModel)). If 0 is the index of the
        first prediction, then all lags are relative to this index.

        See examples below.

        If the model wasn't fitted with:
            - target (concerning RegressionModels only): then the first element should be `None`.

            - past covariates: then the third and fourth elements should be `None`.

            - future covariates: then the fifth and sixth elements should be `None`.

        Should be overridden by models that use past or future covariates, and/or for model that have minimum target
        lag and maximum target lags potentially different from -1 and 0.

        Notes
        -----
        maximum target lag (second value) cannot be `None` and is always larger than or equal to 0.

        Examples
        --------
        >>> model = LinearRegressionModel(lags=3, output_chunk_length=2)
        >>> model.fit(train_series)
        >>> model.extreme_lags
        (-3, 1, None, None, None, None, 0, None)
        >>> model = LinearRegressionModel(lags=3, output_chunk_length=2, output_chunk_shift=2)
        >>> model.fit(train_series)
        >>> model.extreme_lags
        (-3, 1, None, None, None, None, 2, None)
        >>> model = LinearRegressionModel(lags=[-3, -5], lags_past_covariates = 4, output_chunk_length=7)
        >>> model.fit(train_series, past_covariates=past_covariates)
        >>> model.extreme_lags
        (-5, 6, -4, -1,  None, None, 0, None)
        >>> model = LinearRegressionModel(lags=[3, 5], lags_future_covariates = [4, 6], output_chunk_length=7)
        >>> model.fit(train_series, future_covariates=future_covariates)
        >>> model.extreme_lags
        (-5, 6, None, None, 4, 6, 0, None)
        >>> model = NBEATSModel(input_chunk_length=10, output_chunk_length=7)
        >>> model.fit(train_series)
        >>> model.extreme_lags
        (-10, 6, None, None, None, None, 0, None)
        >>> model = NBEATSModel(input_chunk_length=10, output_chunk_length=7, lags_future_covariates=[4, 6])
        >>> model.fit(train_series, future_covariates)
        >>> model.extreme_lags
        (-10, 6, None, None, 4, 6, 0, None)
        """

    @property
    def _training_sample_time_index_length(self) -> int:
        """
        Required time_index length for one training sample, for any model.
        """
        (
            min_target_lag,
            max_target_lag,
            min_past_cov_lag,
            max_past_cov_lag,
            min_future_cov_lag,
            max_future_cov_lag,
            output_chunk_shift,
            max_target_lag_train,
        ) = self.extreme_lags

        # some models can have different output chunks for training and prediction (e.g. `RNNModel`)
        output_lag = max_target_lag_train or max_target_lag
        return max(
            output_lag + 1,
            max_future_cov_lag + 1 if max_future_cov_lag else 0,
        ) - min(
            min_target_lag if min_target_lag else 0,
            min_past_cov_lag if min_past_cov_lag else 0,
            min_future_cov_lag if min_future_cov_lag else 0,
        )

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
        custom_components: Union[list[str], None] = None,
        with_static_covs: bool = True,
        with_hierarchy: bool = True,
        pred_start: Optional[Union[pd.Timestamp, int]] = None,
    ) -> TimeSeries:
        """
        Builds a forecast time series starting after the end of the training time series, with the
        correct time index (or after the end of the input series, if specified).

        Parameters
        ----------
        points_preds
            Forecasted values, can be either the target(s) or parameters of the likelihood model
        input_series
            TimeSeries used as input for the prediction
        custom_components
            New names for the forecast TimeSeries components, used when the number of components changes
        with_static_covs
            If set to `False`, do not copy the input_series `static_covariates` attribute
        with_hierarchy
            If set to `False`, do not copy the input_series `hierarchy` attribute
        pred_start
            Optionally, give a custom prediction start point.

        Returns
        -------
        TimeSeries
            New TimeSeries instance starting after the input series
        """
        input_series = (
            input_series if input_series is not None else self.training_series
        )
        return _build_forecast_series(
            points_preds=points_preds,
            input_series=input_series,
            custom_columns=custom_components,
            with_static_covs=with_static_covs,
            with_hierarchy=with_hierarchy,
            pred_start=pred_start,
        )

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
        is_conformal = kwargs.get("is_conformal", False)
        _historical_forecasts_general_checks(
            self, series, kwargs, is_conformal=is_conformal
        )

    def _get_last_prediction_time(
        self,
        series,
        forecast_horizon,
        overlap_end,
        latest_possible_prediction_start,
    ):
        # if `overlap_end` is True, we can use the pre-computed latest possible first prediction point
        if overlap_end:
            return latest_possible_prediction_start

        # otherwise, the upper bound for the last time step of the last prediction is the end of the target series
        return series.time_index[-forecast_horizon]

    def _check_optimizable_historical_forecasts(
        self,
        forecast_horizon: int,
        retrain: Union[bool, int, Callable[..., bool]],
        show_warnings: bool,
    ) -> bool:
        """By default, historical forecasts cannot be optimized"""
        return False

    @_with_sanity_checks("_historical_forecasts_sanity_checks")
    def historical_forecasts(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        forecast_horizon: int = 1,
        num_samples: int = 1,
        train_length: Optional[int] = None,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        start_format: Literal["position", "value"] = "value",
        stride: int = 1,
        retrain: Union[bool, int, Callable[..., bool]] = True,
        overlap_end: bool = False,
        last_points_only: bool = True,
        verbose: bool = False,
        show_warnings: bool = True,
        predict_likelihood_parameters: bool = False,
        enable_optimization: bool = True,
        data_transformers: Optional[
            dict[str, Union[BaseDataTransformer, Pipeline]]
        ] = None,
        fit_kwargs: Optional[dict[str, Any]] = None,
        predict_kwargs: Optional[dict[str, Any]] = None,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ) -> Union[TimeSeries, list[TimeSeries], list[list[TimeSeries]]]:
        """Generates historical forecasts by simulating predictions at various points in time throughout the history of
        the provided (potentially multiple) `series`. This process involves retrospectively applying the model to
        different time steps, as if the forecasts were made in real-time at those specific moments. This allows for an
        evaluation of the model's performance over the entire duration of the series, providing insights into its
        predictive accuracy and robustness across different historical periods.

        There are two main modes for this method:

        - Re-training Mode (Default, `retrain=True`): The model is re-trained at each step of the simulation, and
          generates a forecast using the updated model. In case of multiple series, the model is re-trained on each
          series independently (global training is not yet supported).
        - Pre-trained Mode (`retrain=False`): The forecasts are generated at each step of the simulation without
          re-training. It is only supported for pre-trained global forecasting models. This mode is significantly
          faster as it skips the re-training step.

        By choosing the appropriate mode, you can balance between computational efficiency and the need for up-to-date
        model training.

        **Re-training Mode:** This mode repeatedly builds a training set by either expanding from the beginning of
        the `series` or by using a fixed-length `train_length` (the start point can also be configured with `start`
        and `start_format`). The model is then trained on this training set, and a forecast of length `forecast_horizon`
        is generated. Subsequently, the end of the training set is moved forward by `stride` time steps, and the process
        is repeated.

        **Pre-trained Mode:** This mode is only supported for pre-trained global forecasting models. It uses the same
        simulation steps as in the *Re-training Mode* (ignoring `train_length`), but generates the forecasts directly
        without re-training.

        By default, with `last_points_only=True`, this method returns a single time series (or a sequence of time
        series) composed of the last point from each historical forecast. This time series will thus have a frequency of
        `series.freq * stride`.
        If `last_points_only=False`, it will instead return a list (or a sequence of lists) of the full historical
        forecast series each with frequency `series.freq`.

        Parameters
        ----------
        series
            A (sequence of) target time series used to successively train (if `retrain` is not ``False``) and compute
            the historical forecasts.
        past_covariates
            Optionally, a (sequence of) past-observed covariate time series for every input time series in `series`.
            This applies only if the model supports past covariates.
        future_covariates
            Optionally, a (sequence of) future-known covariate time series for every input time series in `series`.
            This applies only if the model supports future covariates.
        forecast_horizon
            The forecast horizon for the predictions.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Use values ``>1`` only for probabilistic
            models.
        train_length
            Optionally, use a fixed length / number of time steps for every constructed training set (rolling window
            mode). Only effective when `retrain` is not ``False``. The default is ``None``, where it uses all time
            steps up until the prediction time (expanding window mode). If larger than the number of available time
            steps, uses the expanding mode. Needs to be at least `min_train_series_length`.
        start
            Optionally, the first point in time at which a prediction is computed. This parameter supports:
            ``float``, ``int``, ``pandas.Timestamp``, and ``None``.
            If a ``float``, it is the proportion of the time series that should lie before the first prediction point.
            If an ``int``, it is either the index position of the first prediction point for `series` with a
            `pd.DatetimeIndex`, or the index value for `series` with a `pd.RangeIndex`. The latter can be changed to
            the index position with `start_format="position"`.
            If a ``pandas.Timestamp``, it is the time stamp of the first prediction point.
            If ``None``, the first prediction point will automatically be set to:

            - the first predictable point if `retrain` is ``False``, or `retrain` is a Callable and the first
              predictable point is earlier than the first trainable point.
            - the first trainable point if `retrain` is ``True`` or ``int`` (given `train_length`),
              or `retrain` is a ``Callable`` and the first trainable point is earlier than the first predictable point.
            - the first trainable point (given `train_length`) otherwise

            Note: If `start` is not within the trainable / forecastable points, uses the closest valid start point that
              is a round multiple of `stride` ahead of `start`. Raises a `ValueError`, if no valid start point exists.
            Note: If the model uses a shifted output (`output_chunk_shift > 0`), then the first predicted point is also
              shifted by `output_chunk_shift` points into the future.
            Note: If `start` is outside the possible historical forecasting times, will ignore the parameter
              (default behavior with ``None``) and start at the first trainable/predictable point.
        start_format
            Defines the `start` format.
            If set to ``'position'``, `start` corresponds to the index position of the first predicted point and can
            range from `(-len(series), len(series) - 1)`.
            If set to ``'value'``, `start` corresponds to the index value/label of the first predicted point. Will raise
            an error if the value is not in `series`' index. Default: ``'value'``.
        stride
            The number of time steps between two consecutive predictions.
        retrain
            Whether and/or on which condition to retrain the model before predicting.
            This parameter supports 3 different types: ``bool``, (positive) ``int``, and ``Callable`` (returning a
            ``bool``).
            In the case of ``bool``: retrain the model at each step (`True`), or never retrain the model (`False`).
            In the case of ``int``: the model is retrained every `retrain` iterations.
            In the case of ``Callable``: the model is retrained whenever callable returns `True`.
            The callable must have the following positional arguments:

            - `counter` (int): current `retrain` iteration
            - `pred_time` (pd.Timestamp or int): timestamp of forecast time (end of the training series)
            - `train_series` (TimeSeries): train series up to `pred_time`
            - `past_covariates` (TimeSeries): past_covariates series up to `pred_time`
            - `future_covariates` (TimeSeries): future_covariates series up to `min(pred_time + series.freq *
              forecast_horizon, series.end_time())`

            Note: if any optional `*_covariates` are not passed to `historical_forecast`, ``None`` will be passed
            to the corresponding retrain function argument.
            Note: some models require being retrained every time and do not support anything other than
            `retrain=True`.
            Note: also controls the retraining of the `data_transformers`.
        overlap_end
            Whether the returned forecasts can go beyond the series' end or not.
        last_points_only
            Whether to return only the last point of each historical forecast. If set to ``True``, the method returns a
            single ``TimeSeries`` (for each time series in `series`) containing the successive point forecasts.
            Otherwise, returns a list of historical ``TimeSeries`` forecasts.
        verbose
            Whether to print the progress.
        show_warnings
            Whether to show warnings related to historical forecasts optimization, or parameters `start` and
            `train_length`.
        predict_likelihood_parameters
            If set to `True`, the model predicts the parameters of its `likelihood` instead of the target. Only
            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.
            Default: ``False``.
        enable_optimization
            Whether to use the optimized version of `historical_forecasts` when supported and available.
            Default: ``True``.
        data_transformers
            Optionally, a dictionary of `BaseDataTransformer` or `Pipeline` to apply to the corresponding series
            (possibles keys; "series", "past_covariates", "future_covariates"). If provided, all input series must be
            in the un-transformed space. For fittable transformer / pipeline:

            - if `retrain=True`, the data transformer re-fit on the training data at each historical forecast step.
            - if `retrain=False`, the data transformer transforms the series once before all the forecasts.

            The fitted transformer is used to transform the input during both training and prediction.
            If the transformation is invertible, the forecasts will be inverse-transformed.
        fit_kwargs
            Optionally, some additional arguments passed to the model `fit()` method.
        predict_kwargs
            Optionally, some additional arguments passed to the model `predict()` method.
        sample_weight
            Optionally, some sample weights to apply to the target `series` labels for training. Only effective when
            `retrain` is not ``False``. They are applied per observation, per label (each step in
            `output_chunk_length`), and per component.
            If a series or sequence of series, then those weights are used. If the weight series only have a single
            component / column, then the weights are applied globally to all components in `series`. Otherwise, for
            component-specific weights, the number of components must match those of `series`.
            If a string, then the weights are generated using built-in weighting functions. The available options are
            `"linear"` or `"exponential"` decay - the further in the past, the lower the weight. The weights are
            computed per time `series`.

        Returns
        -------
        TimeSeries
            A single historical forecast for a single `series` and `last_points_only=True`: it contains only the
            predictions at step `forecast_horizon` from all historical forecasts.
        List[TimeSeries]
            A list of historical forecasts for:

            - a sequence (list) of `series` and `last_points_only=True`: for each series, it contains only the
              predictions at step `forecast_horizon` from all historical forecasts.
            - a single `series` and `last_points_only=False`: for each historical forecast, it contains the entire
              horizon `forecast_horizon`.
        List[List[TimeSeries]]
            A list of lists of historical forecasts for a sequence of `series` and `last_points_only=False`. For each
            series, and historical forecast, it contains the entire horizon `forecast_horizon`. The outer list
            is over the series provided in the input sequence, and the inner lists contain the historical forecasts for
            each series.
        """
        model: ForecastingModel = self
        # only GlobalForecastingModels support historical forecasting without retraining the model
        base_class_name = model.__class__.__base__.__name__

        # we can directly abort if retrain is False and fit hasn't been called before
        raise_if(
            not model._fit_called and retrain is False,
            "The model has not been fitted yet, and `retrain` is ``False``. "
            "Either call `fit()` before `historical_forecasts()`, or set `retrain` "
            "to something different than ``False``.",
            logger,
        )

        raise_if(
            (isinstance(retrain, Callable) or int(retrain) != 1)
            and (not model._supports_non_retrainable_historical_forecasts),
            f"{base_class_name} does not support historical forecasting with `retrain` set to `False`. "
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
        ) and train_length < model._training_sample_time_index_length + (
            model.min_train_samples - 1
        ):
            raise_log(
                ValueError(
                    "train_length is too small for the training requirements of this model. "
                    f"Must be `>={model._training_sample_time_index_length + (model.min_train_samples - 1)}`."
                ),
                logger,
            )
        if train_length is not None and retrain is False:
            raise_log(
                ValueError("cannot use `train_length` when `retrain=False`."),
                logger,
            )

        if isinstance(retrain, bool) or (isinstance(retrain, int) and retrain >= 0):

            def retrain_func(
                counter, pred_time, train_series, past_covariates, future_covariates
            ):
                return counter % int(retrain) == 0 if retrain else False

        elif isinstance(retrain, Callable):
            retrain_func = retrain

            # check that the signature matches the documentation
            expected_arguments = [
                "counter",
                "pred_time",
                "train_series",
                "past_covariates",
                "future_covariates",
            ]
            passed_arguments = list(inspect.signature(retrain_func).parameters.keys())
            raise_if(
                expected_arguments != passed_arguments,
                f"the Callable `retrain` must have a signature/arguments matching the following positional arguments: "
                f"{expected_arguments}.",
                logger,
            )

            # passing dummy values to check the type of the output
            result = retrain_func(
                counter=0,
                pred_time=get_single_series(series).time_index[-1],
                train_series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )
            raise_if_not(
                isinstance(result, bool),
                f"Return value of `retrain` must be bool, received {type(result)}",
                logger,
            )
        else:
            retrain_func = None
            raise_log(
                ValueError(
                    "`retrain` argument must be either `bool`, positive `int` or `Callable` (returning `bool`)"
                ),
                logger,
            )

        data_transformers = _convert_data_transformers(
            data_transformers=data_transformers, copy=True
        )

        using_prefitted_transformers = False
        # data transformer already fitted and can be directly applied to all the series
        if data_transformers and not retrain:
            using_prefitted_transformers = True
            series, past_covariates, future_covariates = _apply_data_transformers(
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                data_transformers=data_transformers,
                max_future_cov_lag=model.extreme_lags[5],
                fit_transformers=False,
            )

        # remove unsupported arguments, raise exception if interference with historical forecasts logic
        fit_kwargs, predict_kwargs = _historical_forecasts_sanitize_kwargs(
            model=model,
            fit_kwargs=fit_kwargs,
            predict_kwargs=predict_kwargs,
            retrain=retrain is not False and retrain != 0,
            show_warnings=show_warnings,
        )

        if (
            enable_optimization
            and model.supports_optimized_historical_forecasts
            and model._check_optimizable_historical_forecasts(
                forecast_horizon=forecast_horizon,
                retrain=retrain,
                show_warnings=show_warnings,
            )
        ):
            forecasts = model._optimized_historical_forecasts(
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                num_samples=num_samples,
                start=start,
                start_format=start_format,
                forecast_horizon=forecast_horizon,
                stride=stride,
                overlap_end=overlap_end,
                last_points_only=last_points_only,
                verbose=verbose,
                show_warnings=show_warnings,
                predict_likelihood_parameters=predict_likelihood_parameters,
                **predict_kwargs,
            )

            return _apply_inverse_data_transformers(
                series=series, forecasts=forecasts, data_transformers=data_transformers
            )

        sequence_type_in = get_series_seq_type(series)
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)
        sample_weight = (
            sample_weight
            if isinstance(sample_weight, str)
            else series2seq(sample_weight)
        )

        if len(series) == 1:
            # Use tqdm on the outer loop only if there's more than one series to iterate over
            # (otherwise use tqdm on the inner loop).
            outer_iterator = series
        else:
            outer_iterator = _build_tqdm_iterator(
                series, verbose, total=len(series), desc="historical forecasts"
            )

        # deactivate the warning after displaying it once if show_warnings is True
        show_predict_warnings = show_warnings

        forecasts_list = []
        for idx, series_ in enumerate(outer_iterator):
            past_covariates_ = past_covariates[idx] if past_covariates else None
            future_covariates_ = future_covariates[idx] if future_covariates else None
            if isinstance(sample_weight, str):
                sample_weight_ = sample_weight
            else:
                sample_weight_ = sample_weight[idx] if sample_weight else None

            # predictable time indexes (assuming model is already trained)
            historical_forecasts_time_index_predict = (
                _get_historical_forecast_predict_index(
                    model,
                    series_,
                    idx,
                    past_covariates_,
                    future_covariates_,
                    forecast_horizon,
                    overlap_end,
                )
            )

            if retrain:
                # trainable time indexes (considering lags and available covariates)
                historical_forecasts_time_index_train = (
                    _get_historical_forecast_train_index(
                        model,
                        series_,
                        idx,
                        past_covariates_,
                        future_covariates_,
                        forecast_horizon,
                        overlap_end,
                    )
                )

                # We need the first value timestamp to be used in order to properly shift the series
                # Look at both past and future, since the target lags must be taken in consideration
                min_timestamp_series = (
                    historical_forecasts_time_index_train[0]
                    - model._training_sample_time_index_length * series_.freq
                )

                # based on `retrain`, historical_forecasts_time_index is based either on train or predict
                (
                    historical_forecasts_time_index,
                    train_length_,
                ) = _reconciliate_historical_time_indices(
                    model=model,
                    historical_forecasts_time_index_predict=historical_forecasts_time_index_predict,
                    historical_forecasts_time_index_train=historical_forecasts_time_index_train,
                    series=series_,
                    series_idx=idx,
                    retrain=retrain,
                    train_length=train_length,
                    show_warnings=show_warnings,
                )
            else:
                # we are only predicting: start of the series does not have to change
                min_timestamp_series = series_.time_index[0]
                historical_forecasts_time_index = (
                    historical_forecasts_time_index_predict
                )
                train_length_ = None

            # based on `forecast_horizon` and `overlap_end`, historical_forecasts_time_index is shortened
            historical_forecasts_time_index = _adjust_historical_forecasts_time_index(
                series=series_,
                series_idx=idx,
                historical_forecasts_time_index=historical_forecasts_time_index,
                start=start,
                start_format=start_format,
                stride=stride,
                show_warnings=show_warnings,
            )

            # adjust the start of the series depending on whether we train (at some point), or predict only
            # must be performed after the operation on historical_foreacsts_time_index
            if min_timestamp_series > series_.time_index[0]:
                series_ = series_.drop_before(min_timestamp_series - 1 * series_.freq)

            # generate time index for the iteration
            historical_forecasts_time_index = generate_index(
                start=historical_forecasts_time_index[0],
                end=historical_forecasts_time_index[-1],
                freq=series_.freq,
            )

            if len(series) == 1:
                # Only use tqdm if there's no outer loop
                iterator = _build_tqdm_iterator(
                    historical_forecasts_time_index[::stride],
                    verbose,
                    total=(len(historical_forecasts_time_index) - 1) // stride + 1,
                    desc="historical forecasts",
                )
            else:
                iterator = historical_forecasts_time_index[::stride]

            # Either store the whole forecasts or only the last points of each forecast, depending on last_points_only
            forecasts = []
            last_points_times = []
            last_points_values = []
            _counter_train = 0
            forecast_components = None
            # iterate and forecast
            for _counter, pred_time in enumerate(iterator):
                # drop everything after `pred_time` to train on / predict with shifting input
                if pred_time <= series_.end_time():
                    train_series = series_.drop_after(pred_time)
                else:
                    train_series = series_

                # optionally, apply moving window (instead of expanding window)
                if train_length_ and len(train_series) > train_length_:
                    train_series = train_series[-train_length_:]

                # when `retrain=True`, data transformers are also retrained between iterations to avoid data-leakage
                # using a single series
                if data_transformers and retrain:
                    train_series, past_covariates_, future_covariates_ = (
                        _apply_data_transformers(
                            series=train_series,
                            past_covariates=past_covariates_,
                            future_covariates=future_covariates_,
                            data_transformers=data_transformers,
                            max_future_cov_lag=model.extreme_lags[5],
                            fit_transformers=True,
                        )
                    )

                # testing `retrain` to exclude `False` and `0`
                if (
                    retrain
                    and historical_forecasts_time_index_train is not None
                    and historical_forecasts_time_index_train[0]
                    <= pred_time
                    <= historical_forecasts_time_index_train[-1]
                ):
                    # retrain_func processes the series that would be used for training
                    if retrain_func(
                        counter=_counter_train,
                        pred_time=pred_time,
                        train_series=train_series,
                        past_covariates=past_covariates_,
                        future_covariates=future_covariates_,
                    ):
                        # avoid fitting the same model multiple times
                        model = model.untrained_model()
                        model._fit_wrapper(
                            series=train_series,
                            past_covariates=past_covariates_,
                            future_covariates=future_covariates_,
                            sample_weight=sample_weight_,
                            **fit_kwargs,
                        )
                    else:
                        # untrained model was not trained on the first trainable timestamp
                        if not _counter_train and not model._fit_called:
                            raise_log(
                                ValueError(
                                    f"`retrain` is `False` in the first train iteration at prediction point (in time) "
                                    f"`{pred_time}` and the model has not been fit before. Either call `fit()` before "
                                    f"`historical_forecasts()`, use a different `retrain` value or modify the function "
                                    f"to return `True` at or before this timestamp."
                                ),
                                logger,
                            )
                    _counter_train += 1
                elif not _counter and not model._fit_called:
                    # model must be fit before the first prediction
                    # `historical_forecasts_time_index_train` is known to be not None
                    raise_log(
                        ValueError(
                            f"Model has not been fit before the first predict iteration at prediction point "
                            f"(in time) `{pred_time}`. Either call `fit()` before `historical_forecasts()`, "
                            f"set `retrain=True`, modify the function to return `True` at least once before "
                            f"`{pred_time}`, or use a different `start` value. The first 'predictable' "
                            f"timestamp with re-training inside `historical_forecasts` is: "
                            f"{historical_forecasts_time_index_train[0]} (potential `start` value)."
                        ),
                        logger,
                    )

                # for regression models with lags=None, lags_past_covariates=None and min(lags_future_covariates)>=0,
                # the first predictable timestamp is the first timestamp of the series, a dummy ts must be created
                # to support `predict()`
                if len(train_series) == 0:
                    train_series = TimeSeries.from_times_and_values(
                        times=generate_index(
                            start=pred_time - 1 * series_.freq,
                            length=1,
                            freq=series_.freq,
                        ),
                        values=np.array([np.nan]),
                    )

                forecast = model._predict_wrapper(
                    n=forecast_horizon,
                    series=train_series,
                    past_covariates=past_covariates_,
                    future_covariates=future_covariates_,
                    num_samples=num_samples,
                    verbose=verbose,
                    predict_likelihood_parameters=predict_likelihood_parameters,
                    show_warnings=show_predict_warnings,
                    **predict_kwargs,
                )

                forecast = _apply_inverse_data_transformers(
                    series=train_series,
                    forecasts=forecast,
                    data_transformers=data_transformers,
                    series_idx=idx if using_prefitted_transformers else None,
                )

                show_predict_warnings = False

                if forecast_components is None:
                    forecast_components = forecast.columns

                if last_points_only:
                    last_points_values.append(forecast.all_values(copy=False)[-1])
                    last_points_times.append(forecast.end_time())
                else:
                    forecasts.append(forecast)

            if last_points_only and last_points_values:
                forecasts_list.append(
                    TimeSeries.from_times_and_values(
                        generate_index(
                            start=last_points_times[0],
                            end=last_points_times[-1],
                            freq=series_.freq * stride,
                        ),
                        np.array(last_points_values),
                        columns=(
                            forecast_components
                            if forecast_components is not None
                            else series_.columns
                        ),
                        static_covariates=(
                            series_.static_covariates
                            if not predict_likelihood_parameters
                            else None
                        ),
                        hierarchy=(
                            series_.hierarchy
                            if not predict_likelihood_parameters
                            else None
                        ),
                        metadata=series_.metadata,
                    )
                )
            else:
                forecasts_list.append(forecasts)

        return series2seq(forecasts_list, seq_type_out=sequence_type_in)

    def backtest(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        historical_forecasts: Optional[
            Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]
        ] = None,
        forecast_horizon: int = 1,
        num_samples: int = 1,
        train_length: Optional[int] = None,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        start_format: Literal["position", "value"] = "value",
        stride: int = 1,
        retrain: Union[bool, int, Callable[..., bool]] = True,
        overlap_end: bool = False,
        last_points_only: bool = False,
        metric: Union[METRIC_TYPE, list[METRIC_TYPE]] = metrics.mape,
        reduction: Union[Callable[..., float], None] = np.mean,
        verbose: bool = False,
        show_warnings: bool = True,
        predict_likelihood_parameters: bool = False,
        enable_optimization: bool = True,
        data_transformers: Optional[
            dict[str, Union[BaseDataTransformer, Pipeline]]
        ] = None,
        metric_kwargs: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None,
        fit_kwargs: Optional[dict[str, Any]] = None,
        predict_kwargs: Optional[dict[str, Any]] = None,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ) -> Union[float, np.ndarray, list[float], list[np.ndarray]]:
        """Compute error values that the model produced for historical forecasts on (potentially multiple) `series`.

        If `historical_forecasts` are provided, the metric(s) (given by the `metric` function) is evaluated directly on
        all forecasts and actual values. The same `series` and `last_points_only` value must be passed that were used
        to generate the historical forecasts. Finally, the method returns an optional `reduction` (the mean by default)
        of all these metric scores.

        If `historical_forecasts` is ``None``, it first generates the historical forecasts with the parameters given
        below (see :meth:`ForecastingModel.historical_forecasts()
        <darts.models.forecasting.forecasting_model.ForecastingModel.historical_forecasts>` for more info) and then
        evaluates as described above.

        The metric(s) can be further customized `metric_kwargs` (e.g. control the aggregation over components, time
        steps, multiple series, other required arguments such as `q` for quantile metrics, ...).

        Parameters
        ----------
        series
            A (sequence of) target time series used to successively train (if `retrain` is not ``False``) and compute
            the historical forecasts.
        past_covariates
            Optionally, a (sequence of) past-observed covariate time series for every input time series in `series`.
            This applies only if the model supports past covariates.
        future_covariates
            Optionally, a (sequence of) future-known covariate time series for every input time series in `series`.
            This applies only if the model supports future covariates.
        historical_forecasts
            Optionally, the (or a sequence of / a sequence of sequences of) historical forecasts time series to be
            evaluated. Corresponds to the output of :meth:`historical_forecasts()
            <darts.models.forecasting.forecasting_model.ForecastingModel.historical_forecasts>`. The same `series` and
            `last_points_only` values must be passed that were used to generate the historical forecasts. If provided,
            will skip historical forecasting and ignore all parameters except `series`, `last_points_only`, `metric`,
            and `reduction`.
        forecast_horizon
            The forecast horizon for the predictions.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Use values ``>1`` only for probabilistic
            models.
        train_length
            Optionally, use a fixed length / number of time steps for every constructed training set (rolling window
            mode). Only effective when `retrain` is not ``False``. The default is ``None``, where it uses all time
            steps up until the prediction time (expanding window mode). If larger than the number of available time
            steps, uses the expanding mode. Needs to be at least `min_train_series_length`.
        start
            Optionally, the first point in time at which a prediction is computed. This parameter supports:
            ``float``, ``int``, ``pandas.Timestamp``, and ``None``.
            If a ``float``, it is the proportion of the time series that should lie before the first prediction point.
            If an ``int``, it is either the index position of the first prediction point for `series` with a
            `pd.DatetimeIndex`, or the index value for `series` with a `pd.RangeIndex`. The latter can be changed to
            the index position with `start_format="position"`.
            If a ``pandas.Timestamp``, it is the time stamp of the first prediction point.
            If ``None``, the first prediction point will automatically be set to:

            - the first predictable point if `retrain` is ``False``, or `retrain` is a Callable and the first
              predictable point is earlier than the first trainable point.
            - the first trainable point if `retrain` is ``True`` or ``int`` (given `train_length`),
              or `retrain` is a ``Callable`` and the first trainable point is earlier than the first predictable point.
            - the first trainable point (given `train_length`) otherwise

            Note: If `start` is not within the trainable / forecastable points, uses the closest valid start point that
              is a round multiple of `stride` ahead of `start`. Raises a `ValueError`, if no valid start point exists.
            Note: If the model uses a shifted output (`output_chunk_shift > 0`), then the first predicted point is also
              shifted by `output_chunk_shift` points into the future.
            Note: If `start` is outside the possible historical forecasting times, will ignore the parameter
              (default behavior with ``None``) and start at the first trainable/predictable point.
        start_format
            Defines the `start` format.
            If set to ``'position'``, `start` corresponds to the index position of the first predicted point and can
            range from `(-len(series), len(series) - 1)`.
            If set to ``'value'``, `start` corresponds to the index value/label of the first predicted point. Will raise
            an error if the value is not in `series`' index. Default: ``'value'``.
        stride
            The number of time steps between two consecutive predictions.
        retrain
            Whether and/or on which condition to retrain the model before predicting.
            This parameter supports 3 different types: ``bool``, (positive) ``int``, and ``Callable`` (returning a
            ``bool``).
            In the case of ``bool``: retrain the model at each step (`True`), or never retrain the model (`False`).
            In the case of ``int``: the model is retrained every `retrain` iterations.
            In the case of ``Callable``: the model is retrained whenever callable returns `True`.
            The callable must have the following positional arguments:

            - `counter` (int): current `retrain` iteration
            - `pred_time` (pd.Timestamp or int): timestamp of forecast time (end of the training series)
            - `train_series` (TimeSeries): train series up to `pred_time`
            - `past_covariates` (TimeSeries): past_covariates series up to `pred_time`
            - `future_covariates` (TimeSeries): future_covariates series up to `min(pred_time + series.freq *
              forecast_horizon, series.end_time())`

            Note: if any optional `*_covariates` are not passed to `historical_forecast`, ``None`` will be passed
            to the corresponding retrain function argument.
            Note: some models require being retrained every time and do not support anything other than
            `retrain=True`.
            Note: also controls the retraining of the `data_transformers`.
        overlap_end
            Whether the returned forecasts can go beyond the series' end or not.
        last_points_only
            Whether to return only the last point of each historical forecast. If set to ``True``, the method returns a
            single ``TimeSeries`` (for each time series in `series`) containing the successive point forecasts.
            Otherwise, returns a list of historical ``TimeSeries`` forecasts.
        metric
            A metric function or a list of metric functions. Each metric must either be a Darts metric (see `here
            <https://unit8co.github.io/darts/generated_api/darts.metrics.html>`_), or a custom metric that has an
            identical signature as Darts' metrics, uses decorators :func:`~darts.metrics.metrics.multi_ts_support` and
            :func:`~darts.metrics.metrics.multi_ts_support`, and returns the metric score.
        reduction
            A function used to combine the individual error scores obtained when `last_points_only` is set to `False`.
            When providing several metric functions, the function will receive the argument `axis = 1` to obtain single
            value for each metric function.
            If explicitly set to `None`, the method will return a list of the individual error scores instead.
            Set to ``np.mean`` by default.
        verbose
            Whether to print the progress.
        show_warnings
            Whether to show warnings related to historical forecasts optimization, or parameters `start` and
            `train_length`.
        predict_likelihood_parameters
            If set to `True`, the model predicts the parameters of its `likelihood` instead of the target. Only
            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.
            Default: ``False``.
        enable_optimization
            Whether to use the optimized version of `historical_forecasts` when supported and available.
            Default: ``True``.
        data_transformers
            Optionally, a dictionary of `BaseDataTransformer` or `Pipeline` to apply to the corresponding series
            (possibles keys; "series", "past_covariates", "future_covariates"). If provided, all input series must be
            in the un-transformed space. For fittable transformer / pipeline:

            - if `retrain=True`, the data transformer re-fit on the training data at each historical forecast step.
            - if `retrain=False`, the data transformer transforms the series once before all the forecasts.

            The fitted transformer is used to transform the input during both training and prediction.
            If the transformation is invertible, the forecasts will be inverse-transformed.
            Only effective when `historical_forecasts=None`.
        metric_kwargs
            Additional arguments passed to `metric()`, such as `'n_jobs'` for parallelization, `'component_reduction'`
            for reducing the component wise metrics, seasonality `'m'` for scaled metrics, etc. Will pass arguments to
            each metric separately and only if they are present in the corresponding metric signature. Parameter
            `'insample'` for scaled metrics (e.g. mase`, `rmsse`, ...) is ignored, as it is handled internally.
        fit_kwargs
            Optionally, some additional arguments passed to the model `fit()` method.
        predict_kwargs
            Optionally, some additional arguments passed to the model `predict()` method.
        sample_weight
            Optionally, some sample weights to apply to the target `series` labels for training. Only effective when
            `retrain` is not ``False``. They are applied per observation, per label (each step in
            `output_chunk_length`), and per component.
            If a series or sequence of series, then those weights are used. If the weight series only have a single
            component / column, then the weights are applied globally to all components in `series`. Otherwise, for
            component-specific weights, the number of components must match those of `series`.
            If a string, then the weights are generated using built-in weighting functions. The available options are
            `"linear"` or `"exponential"` decay - the further in the past, the lower the weight. The weights are
            computed per time `series`.

        Returns
        -------
        float
            A single backtest score for single uni/multivariate series, a single `metric` function and:

            - `historical_forecasts` generated with `last_points_only=True`
            - `historical_forecasts` generated with `last_points_only=False` and using a backtest `reduction`
        np.ndarray
            An numpy array of backtest scores. For single series and one of:

            - a single `metric` function, `historical_forecasts` generated with `last_points_only=False`
              and backtest `reduction=None`. The output has shape (n forecasts, *).
            - multiple `metric` functions and `historical_forecasts` generated with `last_points_only=False`.
              The output has shape (*, n metrics) when using a backtest `reduction`, and (n forecasts, *, n metrics)
              when `reduction=None`
            - multiple uni/multivariate series including `series_reduction` and at least one of
              `component_reduction=None` or `time_reduction=None` for "per time step metrics"
        List[float]
            Same as for type `float` but for a sequence of series. The returned metric list has length
            `len(series)` with the `float` metric for each input `series`.
        List[np.ndarray]
            Same as for type `np.ndarray` but for a sequence of series. The returned metric list has length
            `len(series)` with the `np.ndarray` metrics for each input `series`.
        """
        metric_kwargs = metric_kwargs or dict()
        if not isinstance(metric_kwargs, list):
            metric_kwargs = [metric_kwargs]

        if not isinstance(metric, list):
            metric = [metric]

        if len(metric_kwargs) > 1 and len(metric_kwargs) != len(metric):
            raise_log(
                ValueError(
                    f"Mismatch between number of metric-specific `metric_kwargs` "
                    f"({len(metric_kwargs)}) and number of metrics in `metric` ({len(metric)}). "
                    f"For `metric_kwargs`, either give a list of dicts of length `{len(metric)}` "
                    f"with metric-specific kwargs, or a single dict that is applied to all metrics."
                ),
                logger=logger,
            )
        if len(metric_kwargs) != len(metric):
            metric_kwargs = [metric_kwargs[0] for _ in range(len(metric))]

        historical_forecasts = historical_forecasts or self.historical_forecasts(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=num_samples,
            train_length=train_length,
            start=start,
            start_format=start_format,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=retrain,
            overlap_end=overlap_end,
            last_points_only=last_points_only,
            verbose=verbose,
            show_warnings=show_warnings,
            predict_likelihood_parameters=predict_likelihood_parameters,
            enable_optimization=enable_optimization,
            data_transformers=data_transformers,
            fit_kwargs=fit_kwargs,
            predict_kwargs=predict_kwargs,
            sample_weight=sample_weight,
        )

        # remember input series type
        series_seq_type = get_series_seq_type(series)
        # validate historical forecasts and convert to multiple series with multiple forecasts case
        series, historical_forecasts = _process_historical_forecast_for_backtest(
            series=series,
            historical_forecasts=historical_forecasts,
            last_points_only=last_points_only,
        )

        # we have multiple forecasts per series: rearrange forecasts to call each metric only once;
        # flatten historical forecasts, get matching target series index, remember cumulative target lengths
        # for later reshaping back to original
        series_idx = []
        cum_len = [0]
        forecasts_list = []
        for idx, fc_list in enumerate(historical_forecasts):
            series_idx += [idx] * len(fc_list)
            cum_len.append(cum_len[-1] + len(fc_list))
            forecasts_list.extend(fc_list)

        class SeriesGenerator(Sequence):
            """Yields the target `series` corresponding the historical forecast at index `i`.
            Allows lazy loading of target `series` in case it is a Sequence.
            """

            def __len__(self):
                return len(forecasts_list)

            def __getitem__(self, index) -> TimeSeries:
                return series[series_idx[index]]

        # extract metrics per metric and series, and optionally reduce
        # errors shape `(n metrics, n total historical forecasts)`
        series_gen = SeriesGenerator()
        errors = []
        for metric_f, metric_f_kwargs in zip(metric, metric_kwargs):
            # add user supplied metric kwargs
            kwargs = {k: v for k, v in metric_f_kwargs.items()}
            metric_params = inspect.signature(metric_f).parameters

            # scaled metrics require `insample` series
            if "insample" in metric_params:
                kwargs["insample"] = series_gen

            errors.append(metric_f(series_gen, forecasts_list, **kwargs))
        try:
            # multiple series can result in different number of forecasts; try if we can run it efficiently
            errors = np.array(errors)
            is_arr = True
        except ValueError:
            # otherwise, compute array later
            is_arr = False

        # get errors for each input `series`
        backtest_list = []
        for i in range(len(cum_len) - 1):
            # errors_series with shape `(n metrics, n series specific historical forecasts, *)`
            if is_arr:
                errors_series = errors[:, cum_len[i] : cum_len[i + 1]]
            else:
                errors_series = np.array([
                    errors_[cum_len[i] : cum_len[i + 1]] for errors_ in errors
                ])

            if reduction is not None:
                # shape `(n metrics, n forecasts, *)` -> `(n metrics, *)`
                errors_series = reduction(errors_series, axis=1)
            elif last_points_only:
                # shape `(n metrics, n forecasts = 1, *)` -> `(n metrics, *)`
                errors_series = errors_series[:, 0]

            if len(metric) == 1:
                # shape `(n metrics, *)` -> `(*,)`
                errors_series = errors_series[0]
            else:
                # shape `(n metrics, *)` -> `(*, n metrics)`
                errors_series = errors_series.transpose(
                    tuple(i for i in range(1, errors_series.ndim)) + (0,)
                )

            backtest_list.append(errors_series)
        return (
            backtest_list if series_seq_type > SeriesType.SINGLE else backtest_list[0]
        )

    @classmethod
    def gridsearch(
        model_class,
        parameters: dict,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        forecast_horizon: Optional[int] = None,
        stride: int = 1,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        start_format: Literal["position", "value"] = "value",
        last_points_only: bool = False,
        show_warnings: bool = True,
        val_series: Optional[TimeSeries] = None,
        use_fitted_values: bool = False,
        metric: Callable[[TimeSeries, TimeSeries], float] = metrics.mape,
        reduction: Callable[[np.ndarray], float] = np.mean,
        verbose=False,
        n_jobs: int = 1,
        n_random_samples: Optional[Union[int, float]] = None,
        data_transformers: Optional[
            dict[str, Union[BaseDataTransformer, Pipeline]]
        ] = None,
        fit_kwargs: Optional[dict[str, Any]] = None,
        predict_kwargs: Optional[dict[str, Any]] = None,
        sample_weight: Optional[Union[TimeSeries, str]] = None,
    ) -> tuple["ForecastingModel", dict[str, Any], float]:
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
            The target series used as input and target for training.
        past_covariates
            Optionally, a past-observed covariate series. This applies only if the model supports past covariates.
        future_covariates
            Optionally, a future-known covariate series. This applies only if the model supports future covariates.
        forecast_horizon
            The integer value of the forecasting horizon. Activates expanding window mode.
        stride
            Only used in expanding window mode. The number of time steps between two consecutive predictions.
        start
            Only used in expanding window mode. Optionally, the first point in time at which a prediction is computed.
            This parameter supports: ``float``, ``int``, ``pandas.Timestamp``, and ``None``.
            If a ``float``, it is the proportion of the time series that should lie before the first prediction point.
            If an ``int``, it is either the index position of the first prediction point for `series` with a
            `pd.DatetimeIndex`, or the index value for `series` with a `pd.RangeIndex`. The latter can be changed to
            the index position with `start_format="position"`.
            If a ``pandas.Timestamp``, it is the time stamp of the first prediction point.
            If ``None``, the first prediction point will automatically be set to:

            - the first predictable point if `retrain` is ``False``, or `retrain` is a Callable and the first
              predictable point is earlier than the first trainable point.
            - the first trainable point if `retrain` is ``True`` or ``int`` (given `train_length`),
              or `retrain` is a Callable and the first trainable point is earlier than the first predictable point.
            - the first trainable point (given `train_length`) otherwise

            Note: If `start` is not within the trainable / forecastable points, uses the closest valid start point that
              is a round multiple of `stride` ahead of `start`. Raises a `ValueError`, if no valid start point exists.
            Note: If the model uses a shifted output (`output_chunk_shift > 0`), then the first predicted point is also
              shifted by `output_chunk_shift` points into the future.
            Note: If `start` is outside the possible historical forecasting times, will ignore the parameter
              (default behavior with ``None``) and start at the first trainable/predictable point.
        start_format
            Only used in expanding window mode. Defines the `start` format. Only effective when `start` is an integer
            and `series` is indexed with a `pd.RangeIndex`.
            If set to 'position', `start` corresponds to the index position of the first predicted point and can range
            from `(-len(series), len(series) - 1)`.
            If set to 'value', `start` corresponds to the index value/label of the first predicted point. Will raise
            an error if the value is not in `series`' index. Default: ``'value'``
        last_points_only
            Only used in expanding window mode. Whether to use the whole forecasts or only the last point of each
            forecast to compute the error.
        show_warnings
            Only used in expanding window mode. Whether to show warnings related to the `start` parameter.
        val_series
            The TimeSeries instance used for validation in split mode. If provided, this series must start right after
            the end of `series`; so that a proper comparison of the forecast can be made.
        use_fitted_values
            If `True`, uses the comparison with the fitted values.
            Raises an error if ``fitted_values`` is not an attribute of `model_class`.
        metric
            A metric function that returns the error between two `TimeSeries` as a float value . Must either be one of
            Darts' "aggregated over time" metrics (see `here
            <https://unit8co.github.io/darts/generated_api/darts.metrics.html>`_), or a custom metric that as input two
            `TimeSeries` and returns the error
        reduction
            A reduction function (mapping array to float) describing how to aggregate the errors obtained
            on the different validation series when backtesting. By default it'll compute the mean of errors.
        verbose
            Whether to print the progress.
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
        data_transformers
            Optionally, a dictionary of `BaseDataTransformer` or `Pipeline` to apply to the corresponding series
            (possibles keys; "series", "past_covariates", "future_covariates"). If provided, all input series must be
            in the un-transformed space. For fittable transformer / pipeline:

            - if `retrain=True`, the data transformer re-fit on the training data at each historical forecast step.
            - if `retrain=False`, the data transformer transforms the series once before all the forecasts.

            The fitted transformer is used to transform the input during both training and prediction.
            If the transformation is invertible, the forecasts will be inverse-transformed.
        fit_kwargs
            Additional arguments passed to the model `fit()` method.
        predict_kwargs
            Additional arguments passed to the model `predict()` method.
        sample_weight
            Optionally, some sample weights to apply to the target `series` labels for training. Only effective when
            `retrain` is not ``False``. They are applied per observation, per label (each step in
            `output_chunk_length`), and per component.
            If a series, then those weights are used. If the weight series only have a single component / column, then
            the weights are applied globally to all components in `series`. Otherwise, for component-specific weights,
            the number of components must match those of `series`.
            If a string, then the weights are generated using built-in weighting functions. The available options are
            `"linear"` or `"exponential"` decay - the further in the past, the lower the weight.

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
            "'val_series' or 'use_fitted_values'.",
            logger,
        )

        if not isinstance(parameters, dict):
            raise_log(
                ValueError(
                    f"`parameters` should be a dictionary, received a: {type(parameters)}."
                )
            )

        if not all(
            isinstance(params, (list, np.ndarray)) for params in parameters.values()
        ):
            raise_log(
                ValueError(
                    "Every value in the `parameters` dictionary should be a list or a np.ndarray."
                ),
                logger,
            )

        if use_fitted_values:
            raise_if_not(
                hasattr(
                    model_class(**{k: v[0] for k, v in parameters.items()}),
                    "fitted_values",
                ),
                "The model must have a fitted_values attribute to compare with the train TimeSeries (local models)",
                logger,
            )

        elif val_series is not None:
            raise_if_not(
                series.width == val_series.width,
                "Training and validation series require the same number of components.",
                logger,
            )

        data_transformers = _convert_data_transformers(
            data_transformers=data_transformers, copy=True
        )

        if fit_kwargs is None:
            fit_kwargs = dict()
        if predict_kwargs is None:
            predict_kwargs = dict()

        # compute all hyperparameter combinations from selection
        params_cross_product = list(product(*parameters.values()))

        # If n_random_samples has been set, randomly select a subset of the full parameter cross product to search with
        if n_random_samples is not None:
            params_cross_product = model_class._sample_params(
                params_cross_product, n_random_samples
            )

        # iterate through all combinations of the provided parameters and choose the best one
        iterator = _build_tqdm_iterator(
            zip(params_cross_product),
            verbose,
            total=len(params_cross_product),
            desc="gridsearch",
        )

        def _evaluate_combination(param_combination) -> float:
            param_combination_dict = dict(
                list(zip(parameters.keys(), param_combination))
            )
            if param_combination_dict.get("model_name", None):
                current_time = time.strftime("%Y-%m-%d_%H.%M.%S.%f", time.localtime())
                param_combination_dict["model_name"] = (
                    f"{current_time}_{param_combination_dict['model_name']}"
                )

            model = model_class(**param_combination_dict)
            if use_fitted_values:  # fitted value mode
                if data_transformers:
                    series_, past_covariates_, future_covariates_ = (
                        _apply_data_transformers(
                            series=series,
                            past_covariates=past_covariates,
                            future_covariates=future_covariates,
                            data_transformers=data_transformers,
                            max_future_cov_lag=model.extreme_lags[5],
                            fit_transformers=True,
                        )
                    )
                else:
                    series_ = series
                    past_covariates_ = past_covariates
                    future_covariates_ = future_covariates

                model._fit_wrapper(
                    series=series_,
                    past_covariates=past_covariates_,
                    future_covariates=future_covariates_,
                    sample_weight=sample_weight,
                    **fit_kwargs,
                )
                fitted_values = TimeSeries.from_times_and_values(
                    series.time_index, model.fitted_values
                )
                if data_transformers and "series" in data_transformers:
                    fitted_values = _apply_inverse_data_transformers(
                        series=series_,
                        forecasts=fitted_values,
                        data_transformers=data_transformers,
                        series_idx=None,
                    )
                error = metric(series, fitted_values)
            elif val_series is None:  # expanding window mode
                error = model.backtest(
                    series=series,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                    num_samples=1,
                    start=start,
                    start_format=start_format,
                    forecast_horizon=forecast_horizon,
                    stride=stride,
                    metric=metric,
                    reduction=reduction,
                    last_points_only=last_points_only,
                    verbose=verbose,
                    show_warnings=show_warnings,
                    data_transformers=data_transformers,
                    fit_kwargs=fit_kwargs,
                    predict_kwargs=predict_kwargs,
                    sample_weight=sample_weight,
                )
            else:  # split mode
                if data_transformers:
                    series_, past_covariates_, future_covariates_ = (
                        _apply_data_transformers(
                            series=series,
                            past_covariates=past_covariates,
                            future_covariates=future_covariates,
                            data_transformers=data_transformers,
                            max_future_cov_lag=model.extreme_lags[5],
                            fit_transformers=True,
                        )
                    )
                else:
                    series_ = series
                    past_covariates_ = past_covariates
                    future_covariates_ = future_covariates

                model._fit_wrapper(
                    series=series_,
                    past_covariates=past_covariates_,
                    future_covariates=future_covariates_,
                    sample_weight=sample_weight,
                    **fit_kwargs,
                )
                pred = model._predict_wrapper(
                    n=len(val_series),
                    series=series_,
                    past_covariates=past_covariates_,
                    future_covariates=future_covariates_,
                    num_samples=1,
                    verbose=verbose,
                    **predict_kwargs,
                )
                pred = _apply_inverse_data_transformers(
                    series=series_,
                    forecasts=pred,
                    data_transformers=data_transformers,
                )
                error = metric(val_series, pred)

            return float(error)

        errors: list[float] = _parallel_apply(
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
        historical_forecasts: Optional[
            Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]
        ] = None,
        forecast_horizon: int = 1,
        num_samples: int = 1,
        train_length: Optional[int] = None,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        start_format: Literal["position", "value"] = "value",
        stride: int = 1,
        retrain: Union[bool, int, Callable[..., bool]] = True,
        overlap_end: bool = False,
        last_points_only: bool = True,
        metric: METRIC_TYPE = metrics.err,
        verbose: bool = False,
        show_warnings: bool = True,
        predict_likelihood_parameters: bool = False,
        enable_optimization: bool = True,
        data_transformers: Optional[
            dict[str, Union[BaseDataTransformer, Pipeline]]
        ] = None,
        metric_kwargs: Optional[dict[str, Any]] = None,
        fit_kwargs: Optional[dict[str, Any]] = None,
        predict_kwargs: Optional[dict[str, Any]] = None,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
        values_only: bool = False,
    ) -> Union[TimeSeries, list[TimeSeries], list[list[TimeSeries]]]:
        """Compute the residuals that the model produced for historical forecasts on (potentially multiple) `series`.

        This function computes the difference (or one of Darts' "per time step" metrics) between the actual
        observations from `series` and the fitted values obtained by training the model on `series` (or using a
        pre-trained model with `retrain=False`). Not all models support fitted values, so we use historical forecasts
        as an approximation for them.

        In sequence this method performs:

        - use pre-computed `historical_forecasts` or compute historical forecasts for each series (see
          :meth:`~darts.models.forecasting.forecasting_model.ForecastingModel.historical_forecasts` for more details).
          How the historical forecasts are generated can be configured with parameters `num_samples`, `train_length`,
          `start`, `start_format`, `forecast_horizon`, `stride`, `retrain`, `last_points_only`, `fit_kwargs`, and
          `predict_kwargs`.
        - compute a backtest using a "per time step" `metric` between the historical forecasts and `series` per
          component/column and time step (see
          :meth:`~darts.models.forecasting.forecasting_model.ForecastingModel.backtest` for more details). By default,
          uses the residuals :func:`~darts.metrics.metrics.err` (error) as a `metric`.
        - create and return `TimeSeries` (or simply a np.ndarray with `values_only=True`) with the time index from
          historical forecasts, and values from the metrics per component and time step.

        This method works for single or multiple univariate or multivariate series.
        It uses the median prediction (when dealing with stochastic forecasts).

        Parameters
        ----------
        series
            A (sequence of) target time series used to successively train (if `retrain` is not ``False``) and compute
            the historical forecasts.
        past_covariates
            Optionally, a (sequence of) past-observed covariate time series for every input time series in `series`.
            This applies only if the model supports past covariates.
        future_covariates
            Optionally, a (sequence of) future-known covariate time series for every input time series in `series`.
            This applies only if the model supports future covariates.
        historical_forecasts
            Optionally, the (or a sequence of / a sequence of sequences of) historical forecasts time series to be
            evaluated. Corresponds to the output of :meth:`historical_forecasts()
            <darts.models.forecasting.forecasting_model.ForecastingModel.historical_forecasts>`. The same `series` and
            `last_points_only` values must be passed that were used to generate the historical forecasts. If provided,
            will skip historical forecasting and ignore all parameters except `series`, `last_points_only`, `metric`,
            and `reduction`.
        forecast_horizon
            The forecast horizon for the predictions.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Use values ``>1`` only for probabilistic
            models.
        train_length
            Optionally, use a fixed length / number of time steps for every constructed training set (rolling window
            mode). Only effective when `retrain` is not ``False``. The default is ``None``, where it uses all time
            steps up until the prediction time (expanding window mode). If larger than the number of available time
            steps, uses the expanding mode. Needs to be at least `min_train_series_length`.
        start
            Optionally, the first point in time at which a prediction is computed. This parameter supports:
            ``float``, ``int``, ``pandas.Timestamp``, and ``None``.
            If a ``float``, it is the proportion of the time series that should lie before the first prediction point.
            If an ``int``, it is either the index position of the first prediction point for `series` with a
            `pd.DatetimeIndex`, or the index value for `series` with a `pd.RangeIndex`. The latter can be changed to
            the index position with `start_format="position"`.
            If a ``pandas.Timestamp``, it is the time stamp of the first prediction point.
            If ``None``, the first prediction point will automatically be set to:

            - the first predictable point if `retrain` is ``False``, or `retrain` is a Callable and the first
              predictable point is earlier than the first trainable point.
            - the first trainable point if `retrain` is ``True`` or ``int`` (given `train_length`),
              or `retrain` is a ``Callable`` and the first trainable point is earlier than the first predictable point.
            - the first trainable point (given `train_length`) otherwise

            Note: If `start` is not within the trainable / forecastable points, uses the closest valid start point that
              is a round multiple of `stride` ahead of `start`. Raises a `ValueError`, if no valid start point exists.
            Note: If the model uses a shifted output (`output_chunk_shift > 0`), then the first predicted point is also
              shifted by `output_chunk_shift` points into the future.
            Note: If `start` is outside the possible historical forecasting times, will ignore the parameter
              (default behavior with ``None``) and start at the first trainable/predictable point.
        start_format
            Defines the `start` format.
            If set to ``'position'``, `start` corresponds to the index position of the first predicted point and can
            range from `(-len(series), len(series) - 1)`.
            If set to ``'value'``, `start` corresponds to the index value/label of the first predicted point. Will raise
            an error if the value is not in `series`' index. Default: ``'value'``.
        stride
            The number of time steps between two consecutive predictions.
        retrain
            Whether and/or on which condition to retrain the model before predicting.
            This parameter supports 3 different types: ``bool``, (positive) ``int``, and ``Callable`` (returning a
            ``bool``).
            In the case of ``bool``: retrain the model at each step (`True`), or never retrain the model (`False`).
            In the case of ``int``: the model is retrained every `retrain` iterations.
            In the case of ``Callable``: the model is retrained whenever callable returns `True`.
            The callable must have the following positional arguments:

            - `counter` (int): current `retrain` iteration
            - `pred_time` (pd.Timestamp or int): timestamp of forecast time (end of the training series)
            - `train_series` (TimeSeries): train series up to `pred_time`
            - `past_covariates` (TimeSeries): past_covariates series up to `pred_time`
            - `future_covariates` (TimeSeries): future_covariates series up to `min(pred_time + series.freq *
              forecast_horizon, series.end_time())`

            Note: if any optional `*_covariates` are not passed to `historical_forecast`, ``None`` will be passed
            to the corresponding retrain function argument.
            Note: some models require being retrained every time and do not support anything other than
            `retrain=True`.
            Note: also controls the retraining of the `data_transformers`.
        overlap_end
            Whether the returned forecasts can go beyond the series' end or not.
        last_points_only
            Whether to return only the last point of each historical forecast. If set to ``True``, the method returns a
            single ``TimeSeries`` (for each time series in `series`) containing the successive point forecasts.
            Otherwise, returns a list of historical ``TimeSeries`` forecasts.
        metric
            Either one of Darts' "per time step" metrics (see `here
            <https://unit8co.github.io/darts/generated_api/darts.metrics.html>`_), or a custom metric that has an
            identical signature as Darts' "per time step" metrics, uses decorators
            :func:`~darts.metrics.metrics.multi_ts_support` and :func:`~darts.metrics.metrics.multi_ts_support`,
            and returns one value per time step.
        verbose
            Whether to print the progress.
        show_warnings
            Whether to show warnings related to historical forecasts optimization, or parameters `start` and
            `train_length`.
        predict_likelihood_parameters
            If set to `True`, the model predicts the parameters of its `likelihood` instead of the target. Only
            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.
            Default: ``False``.
        enable_optimization
            Whether to use the optimized version of `historical_forecasts` when supported and available.
            Default: ``True``.
        data_transformers
            Optionally, a dictionary of `BaseDataTransformer` or `Pipeline` to apply to the corresponding series
            (possibles keys; "series", "past_covariates", "future_covariates"). If provided, all input series must be
            in the un-transformed space. For fittable transformer / pipeline:

            - if `retrain=True`, the data transformer re-fit on the training data at each historical forecast step.
            - if `retrain=False`, the data transformer transforms the series once before all the forecasts.

            The fitted transformer is used to transform the input during both training and prediction.
            If the transformation is invertible, the forecasts will be inverse-transformed.
            Only effective when `historical_forecasts=None`.
        metric_kwargs
            Additional arguments passed to `metric()`, such as `'n_jobs'` for parallelization, `'m'` for scaled
            metrics, etc. Will pass arguments only if they are present in the corresponding metric signature. Ignores
            reduction arguments `"series_reduction", "component_reduction", "time_reduction"`, and parameter
            `'insample'` for scaled metrics (e.g. mase`, `rmsse`, ...), as they are handled internally.
        fit_kwargs
            Optionally, some additional arguments passed to the model `fit()` method.
        predict_kwargs
            Optionally, some additional arguments passed to the model `predict()` method.
        sample_weight
            Optionally, some sample weights to apply to the target `series` labels for training. Only effective when
            `retrain` is not ``False``. They are applied per observation, per label (each step in
            `output_chunk_length`), and per component.
            If a series or sequence of series, then those weights are used. If the weight series only have a single
            component / column, then the weights are applied globally to all components in `series`. Otherwise, for
            component-specific weights, the number of components must match those of `series`.
            If a string, then the weights are generated using built-in weighting functions. The available options are
            `"linear"` or `"exponential"` decay - the further in the past, the lower the weight. The weights are
            computed per time `series`.
        values_only
            Whether to return the residuals as `np.ndarray`. If `False`, returns residuals as `TimeSeries`.

        Returns
        -------
        TimeSeries
            Residual `TimeSeries` for a single `series` and `historical_forecasts` generated with
            `last_points_only=True`.
        List[TimeSeries]
            A list of residual `TimeSeries` for a sequence (list) of `series` with `last_points_only=True`.
            The residual list has length `len(series)`.
        List[List[TimeSeries]]
            A list of lists of residual `TimeSeries` for a sequence of `series` with `last_points_only=False`.
            The outer residual list has length `len(series)`. The inner lists consist of the residuals from
            all possible series-specific historical forecasts.
        """
        # `residuals()` should return metrics per series, component and time step (no reduction)
        metric_kwargs = copy.deepcopy(metric_kwargs) or {}
        metric_kwargs["series_reduction"] = None
        metric_kwargs["component_reduction"] = None
        metric_kwargs["time_reduction"] = None

        historical_forecasts = historical_forecasts or self.historical_forecasts(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=num_samples,
            train_length=train_length,
            start=start,
            start_format=start_format,
            forecast_horizon=forecast_horizon,
            stride=stride,
            retrain=retrain,
            last_points_only=last_points_only,
            verbose=verbose,
            show_warnings=show_warnings,
            predict_likelihood_parameters=predict_likelihood_parameters,
            enable_optimization=enable_optimization,
            data_transformers=data_transformers,
            fit_kwargs=fit_kwargs,
            predict_kwargs=predict_kwargs,
            overlap_end=overlap_end,
            sample_weight=sample_weight,
        )

        # remember input series type
        series_seq_type = get_series_seq_type(series)
        # validate historical forecasts and convert to multiple series with multiple forecasts case
        series, historical_forecasts = _process_historical_forecast_for_backtest(
            series=series,
            historical_forecasts=historical_forecasts,
            last_points_only=last_points_only,
        )

        # optionally, add nans to end of series to get residuals of same shape for each forecast
        if overlap_end:
            series = _extend_series_for_overlap_end(
                series=series, historical_forecasts=historical_forecasts
            )

        residuals = self.backtest(
            series=series,
            historical_forecasts=historical_forecasts,
            last_points_only=False,
            metric=metric,
            reduction=None,
            data_transformers=data_transformers,
            metric_kwargs=metric_kwargs,
        )

        # sanity check residual output
        q, q_interval = metric_kwargs.get("q"), metric_kwargs.get("q_interval")
        try:
            series_, res, fc = series[0], residuals[0][0], historical_forecasts[0][0]
            _ = np.reshape(res, (len(fc), -1, 1))
        except Exception as err:
            raise_log(
                ValueError(
                    f"`metric` function did not yield expected output. Make sure "
                    f"to use one of Darts 'per time step' metrics, or a similar "
                    f"custom metric. The following exception was raised: "
                    f"{type(err).__name__}('{err}')"
                ),
                logger=logger,
            )

        # process residuals
        residuals_out = []
        for series_, fc_list, res_list in zip(series, historical_forecasts, residuals):
            res_list_out = []
            if q is not None:
                q = [q] if isinstance(q, float) else q
                # multi-quantile metrics yield more components
                comp_names = likelihood_component_names(
                    components=series_.components,
                    parameter_names=quantile_names(q),
                )
            # `q` and `q_interval` are mutually exclusive
            elif q_interval is not None:
                # multi-quantile metrics yield more components
                q_interval = (
                    [q_interval] if isinstance(q_interval, tuple) else q_interval
                )
                comp_names = likelihood_component_names(
                    components=series_.components,
                    parameter_names=quantile_interval_names(q_interval),
                )
            else:
                comp_names = None
            for fc, res in zip(fc_list, res_list):
                # make sure all residuals have shape (n time steps, n components * n quantiles, n samples=1)
                if len(res.shape) != 3:
                    res = np.reshape(res, (len(fc), -1, 1))
                if values_only:
                    res = res
                elif (q is None and q_interval is None) and res.shape[
                    1
                ] == fc.n_components:
                    res = fc.with_values(res)
                else:
                    # quantile (interval) metrics created different number of components;
                    # create new series with unknown components
                    res = TimeSeries.from_times_and_values(
                        times=fc._time_index,
                        values=res,
                        columns=comp_names,
                    )
                res_list_out.append(res)

            residuals_out.append(res_list_out)

        # if required, reduce to `series` input type
        if series_seq_type == SeriesType.SINGLE:
            return residuals_out[0][0] if last_points_only else residuals_out[0]

        return (
            [res for res_list in residuals_out for res in res_list]
            if last_points_only
            else residuals_out
        )

    def initialize_encoders(self, default=False) -> SequentialEncoder:
        """instantiates the SequentialEncoder object based on self._model_encoder_settings and parameter
        ``add_encoders`` used at model creation"""
        if default:
            return SequentialEncoder(add_encoders={})

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
    ) -> tuple[
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
    ) -> tuple[
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

    def generate_fit_predict_encodings(
        self,
        n: int,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> tuple[
        Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]
    ]:
        """Generates covariate encodings for training and inference/prediction and returns a tuple of past, and future
        covariates series with the original and encoded covariates stacked together. The encodings are generated by the
        encoders defined at model creation with parameter `add_encoders`. Pass the same `series`, `past_covariates`,
        and `future_covariates` that you intend to use for training and prediction.

        Parameters
        ----------
        n
            The number of prediction time steps after the end of `series` intended to be used for prediction.
        series
            The series or sequence of series with target values intended to be used for training and prediction.
        past_covariates
            Optionally, the past-observed covariates series intended to be used for training and prediction. The
            dimensions must match those of the covariates used for training.
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
        return self.encoders.encode_train_inference(
            n=n,
            target=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

    def _process_validation_set(
        self,
        series: Sequence[TimeSeries],
        past_covariates: Optional[Sequence[TimeSeries]],
        future_covariates: Optional[Sequence[TimeSeries]],
        val_series: Optional[Sequence[TimeSeries]],
        val_past_covariates: Optional[Sequence[TimeSeries]],
        val_future_covariates: Optional[Sequence[TimeSeries]],
    ) -> tuple[
        Optional[Sequence[TimeSeries]],
        Optional[Sequence[TimeSeries]],
        Optional[Sequence[TimeSeries]],
    ]:
        """Validates the validation set and generates/adds the required encodings."""
        if val_series is None:
            return None, None, None

        # generate encodings for the val set covariates
        if self.encoders.encoding_available:
            (
                val_past_covariates,
                val_future_covariates,
            ) = self.generate_fit_encodings(
                series=val_series,
                past_covariates=val_past_covariates,
                future_covariates=val_future_covariates,
            )

        for idx in range(len(val_series)):
            val_s = val_series[idx]
            val_pc = (
                val_past_covariates[idx] if val_past_covariates is not None else None
            )
            val_fc = (
                val_future_covariates[idx]
                if val_future_covariates is not None
                else None
            )

            # check that val set has same number of features as train set
            match_series = series[0].width == val_s.width
            match_past_covariates = (
                past_covariates[0].width if past_covariates is not None else None
            ) == (val_pc.width if val_pc is not None else None)
            match_future_covariates = (
                future_covariates[0].width if future_covariates is not None else None
            ) == (val_fc.width if val_fc is not None else None)

            if self.uses_static_covariates:
                self._verify_static_covariates(val_s.static_covariates)
                match_static_covariates = (
                    series[0].static_covariates.shape
                    if series[0].static_covariates is not None
                    else None
                ) == (
                    val_s.static_covariates.shape
                    if val_s.static_covariates is not None
                    else None
                )
            else:
                match_static_covariates = True

            matches = [
                match_series,
                match_past_covariates,
                match_future_covariates,
                match_static_covariates,
            ]
            if not all(matches):
                invalid_series = [
                    name
                    for match, name in zip(
                        matches,
                        [
                            "`series`",
                            "`past_covariates`",
                            "`future_covariates`",
                            "`static_covariates`",
                        ],
                    )
                    if not match
                ]
                raise_log(
                    ValueError(
                        f"The dimensions of the ({', '.join(invalid_series)}) between the training and "
                        f"validation set "
                        f"{'' if len(val_series) == 1 else 'at sequence/list index `' + str(idx) + '` '}"
                        f"do not match."
                    ),
                    logger=logger,
                )
        return val_series, val_past_covariates, val_future_covariates

    @property
    @abstractmethod
    def _model_encoder_settings(
        self,
    ) -> tuple[
        Optional[int],
        Optional[int],
        bool,
        bool,
        Optional[list[int]],
        Optional[list[int]],
    ]:
        """Abstract property that returns model specific encoder settings that are used to initialize the encoders.

        Must return Tuple (input_chunk_length, output_chunk_length, takes_past_covariates, takes_future_covariates,
        lags_past_covariates, lags_future_covariates).
        """

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
        """Returns a new (untrained) model instance created with the same parameters."""
        return self.__class__(**copy.deepcopy(self.model_params))

    @property
    def model_params(self) -> dict:
        return (
            self._model_params if hasattr(self, "_model_params") else self._model_call
        )

    @classmethod
    def _default_save_path(cls) -> str:
        return f"{cls.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"

    def _clean(self) -> Self:
        """Returns a cleaned instance of the model. Has no effect for local forecasting models."""
        return self

    def save(
        self,
        path: Optional[Union[str, os.PathLike, BinaryIO]] = None,
        clean: bool = False,
        **pkl_kwargs,
    ) -> None:
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
            is automatically saved under ``"{ModelClass}_{YYYY-mm-dd_HH_MM_SS}.pkl"``.
            E.g., ``"RegressionModel_2020-01-01_12_00_00.pkl"``.
        clean
            Whether to store a cleaned version of the model. Only effective for global forecasting models.
            If `True`, the training series and covariates are removed.

            Note: After loading a global forecasting model stored with `clean=True`, a `series` must be passed
            'predict()', `historical_forecasts()` and other forecasting methods.
        pkl_kwargs
            Keyword arguments passed to `pickle.dump()`
        """

        if path is None:
            # default path
            path = self._default_save_path() + ".pkl"

        model_to_save = self._clean() if clean else self
        if isinstance(path, (str, os.PathLike)):
            # save the whole object using pickle
            with open(path, "wb") as handle:
                pickle.dump(obj=model_to_save, file=handle, **pkl_kwargs)
        elif isinstance(path, io.BufferedWriter):
            # save the whole object using pickle
            pickle.dump(obj=model_to_save, file=path, **pkl_kwargs)
        else:
            raise_log(
                ValueError(
                    "Argument 'path' has to be either 'str' or 'PathLike' (for a filepath) "
                    f"or 'BufferedWriter' (for an already opened file), but was '{path.__class__}'."
                ),
                logger=logger,
            )

    @staticmethod
    def load(path: Union[str, os.PathLike, BinaryIO]) -> "ForecastingModel":
        """
        Loads a model from a given path or file handle.

        Parameters
        ----------
        path
            Path or file handle from which to load the model.
        """

        if isinstance(path, (str, os.PathLike)):
            raise_if_not(
                os.path.exists(path),
                f"The file {path} doesn't exist",
                logger,
            )

            with open(path, "rb") as handle:
                model = pickle.load(file=handle)
        elif isinstance(path, io.BufferedReader):
            model = pickle.load(file=path)
        else:
            raise_log(
                ValueError(
                    "Argument 'path' has to be either 'str' or 'PathLike' (for a filepath) "
                    f"or 'BufferedReader' (for an already opened file), but was '{path.__class__}'."
                ),
                logger=logger,
            )

        return model

    def _assert_univariate(self, series: TimeSeries):
        if not series.is_univariate:
            raise_log(
                ValueError(
                    f"Model `{self.__class__.__name__}` only supports univariate TimeSeries instances"
                ),
                logger=logger,
            )

    def _assert_multivariate(self, series: TimeSeries):
        if series.is_univariate:
            raise_log(
                ValueError(
                    f"Model `{self.__class__.__name__}` only supports multivariate TimeSeries instances"
                ),
                logger=logger,
            )

    def __repr__(self):
        """
        Get full description for this estimator (includes all params).
        """
        return self._get_model_description_string(True)

    def __str__(self):
        """
        Get short description for this estimator (only includes params with non-default values).
        """
        return self._get_model_description_string(False)

    def _get_model_description_string(self, include_default_params):
        """
        Get model description string of structure `model_name`(`model_param_key_value_pairs`).

        Parameters
        ----------
        include_default_params : bool,
            If True, will include params with default values in the description.

        Returns
        -------
        description : String
            Model description.
        """
        default_model_params = self._get_default_model_params()
        changed_model_params = [
            (k, v)
            for k, v in self.model_params.items()
            if include_default_params or np.any(v != default_model_params.get(k, None))
        ]

        model_name = self.__class__.__name__
        params_string = ", ".join([f"{k}={str(v)}" for k, v in changed_model_params])
        return f"{model_name}({params_string})"

    @classmethod
    def _get_default_model_params(cls):
        """Get parameter key : default_value pairs for the estimator"""
        init_signature = inspect.signature(cls.__init__)
        # Consider the constructor parameters excluding 'self'
        return {
            p.name: p.default
            for p in init_signature.parameters.values()
            if p.name != "self"
        }

    def _verify_static_covariates(self, static_covariates: Optional[pd.DataFrame]):
        """
        Verify that all static covariates are numeric.
        """
        if static_covariates is not None:
            numeric_mask = static_covariates.columns.isin(
                static_covariates.select_dtypes(include=np.number)
            )
            if sum(~numeric_mask):
                raise_log(
                    ValueError(
                        f"{self.__class__.__name__} can only interpret numeric static covariate data. Consider "
                        "encoding/transforming categorical static covariates with "
                        "`darts.dataprocessing.transformers.static_covariates_transformer.StaticCovariatesTransformer` "
                        "or set `use_static_covariates=False` at model creation to ignore static covariates."
                    ),
                    logger,
                )

    def _optimized_historical_forecasts(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        start_format: Literal["position", "value"] = "value",
        forecast_horizon: int = 1,
        stride: int = 1,
        overlap_end: bool = False,
        last_points_only: bool = True,
        verbose: bool = False,
        show_warnings: bool = True,
        predict_likelihood_parameters: bool = False,
        data_transformers: Optional[dict[str, BaseDataTransformer]] = None,
        **kwargs,
    ) -> Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]:
        logger.warning(
            "`optimized historical forecasts is not available for this model, use `historical_forecasts` instead."
        )
        return []


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
    ) -> tuple[
        Optional[int],
        Optional[int],
        bool,
        bool,
        Optional[list[int]],
        Optional[list[int]],
    ]:
        return None, None, False, False, None, None

    @abstractmethod
    def fit(self, series: TimeSeries) -> "LocalForecastingModel":
        super().fit(series)
        series._assert_deterministic()

    @property
    def extreme_lags(
        self,
    ) -> tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        int,
        Optional[int],
    ]:
        # TODO: LocalForecastingModels do not yet handle extreme lags properly. Especially
        #  TransferableFutureCovariatesLocalForecastingModel, where there is a difference between fit and predict mode)
        #  do not yet. In general, Local models train on the entire series (input=output), different to Global models
        #  that use an input to predict an output.
        return -self.min_train_series_length, -1, None, None, None, None, 0, None

    @property
    def supports_transferrable_series_prediction(self) -> bool:
        """
        Whether the model supports prediction for any input `series`.
        """
        return False


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
            if past_covariates is not None:
                self.past_covariate_series = past_covariates
            if future_covariates is not None:
                self.future_covariate_series = future_covariates
            if (
                series.static_covariates is not None
                and self.supports_static_covariates
                and self.considers_static_covariates
            ):
                self.static_covariates = series.static_covariates
        else:
            if past_covariates is not None:
                self._expect_past_covariates = True
            if future_covariates is not None:
                self._expect_future_covariates = True
            if (
                get_single_series(series).static_covariates is not None
                and self.supports_static_covariates
                and self.considers_static_covariates
            ):
                self.static_covariates = series[0].static_covariates
                self._expect_static_covariates = True

        if past_covariates is not None:
            self._uses_past_covariates = True
        if future_covariates is not None:
            self._uses_future_covariates = True
        if (
            get_single_series(series).static_covariates is not None
            and self.supports_static_covariates
            and self.considers_static_covariates
        ):
            self._uses_static_covariates = True
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
        predict_likelihood_parameters: bool = False,
        show_warnings: bool = True,
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
            Number of times a prediction is sampled from a probabilistic model. Must be `1` for deterministic models.
        verbose
            Whether to print the progress.
        predict_likelihood_parameters
            If set to `True`, the model predicts the parameters of its `likelihood` instead of the target. Only
            supported for probabilistic models with a likelihood, `num_samples = 1` and `n<=output_chunk_length`.
            Default: ``False``
        show_warnings
            Whether to show warnings related auto-regression and past covariates usage.

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
        if predict_likelihood_parameters:
            self._sanity_check_predict_likelihood_parameters(
                n, self.output_chunk_length, num_samples
            )

        if self.uses_past_covariates and past_covariates is None:
            raise_log(
                ValueError(
                    "The model has been trained with past covariates. Some matching past_covariates "
                    "have to be provided to `predict()`."
                )
            )
        if self.uses_future_covariates and future_covariates is None:
            raise_log(
                ValueError(
                    "The model has been trained with future covariates. Some matching future_covariates "
                    "have to be provided to `predict()`."
                )
            )
        if (
            self.uses_static_covariates
            and get_single_series(series).static_covariates is None
        ):
            raise_log(
                ValueError(
                    "The model has been trained with static covariates. Some matching static covariates "
                    "must be embedded in the target `series` passed to `predict()`."
                )
            )
        if (
            show_warnings
            and self.uses_past_covariates
            and self.output_chunk_length is not None
            and n > self.output_chunk_length
        ):
            logger.warning(
                "`predict()` was called with `n > output_chunk_length`: using auto-regression to forecast "
                "the values after `output_chunk_length` points. The model will access `(n - output_chunk_length)` "
                "future values of your `past_covariates` (relative to the first predicted time step). "
                "To hide this warning, set `show_warnings=False`."
            )

    def _clean(self) -> Self:
        """Returns a cleaned instance of the model by removing the training series and covariates."""

        # a shallow copy is enough since we are only interested in removing pointers to the training data
        cleaned_model = copy.copy(self)
        cleaned_model.training_series = None
        cleaned_model.past_covariate_series = None
        cleaned_model.future_covariate_series = None
        cleaned_model.static_covariates = None
        return cleaned_model

    @property
    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        """GlobalForecastingModel supports historical forecasts without retraining the model"""
        return True

    @property
    def supports_optimized_historical_forecasts(self) -> bool:
        """
        Whether the model supports optimized historical forecasts
        """
        return True

    @property
    def supports_transferrable_series_prediction(self) -> bool:
        """
        Whether the model supports prediction for any input `series`.
        """
        return True

    @property
    def supports_sample_weight(self) -> bool:
        """
        Whether model supports sample weight for training.
        """
        return True

    def _sanity_check_predict_likelihood_parameters(
        self, n: int, output_chunk_length: Union[int, None], num_samples: int
    ):
        """Verify that the assumptions for likelihood parameters prediction are verified:
        - Probabilistic models fitted with a likelihood
        - `num_samples=1`
        - `n <= output_chunk_length`
        """
        if not self.supports_likelihood_parameter_prediction:
            raise_log(
                ValueError(
                    "`predict_likelihood_parameters=True` is only supported for probabilistic models fitted with "
                    "a likelihood."
                ),
                logger,
            )
        if num_samples != 1:
            raise_log(
                ValueError(
                    f"`predict_likelihood_parameters=True` is only supported for `num_samples=1`, "
                    f"received {num_samples}."
                ),
                logger,
            )
        if output_chunk_length is not None and n > output_chunk_length:
            raise_log(
                ValueError(
                    "`predict_likelihood_parameters=True` is only supported for `n` smaller than or equal to "
                    "`output_chunk_length`."
                ),
                logger,
            )


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
            self._uses_future_covariates = True

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

    def predict(
        self,
        n: int,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        verbose: bool = False,
        show_warnings: bool = True,
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
            Number of times a prediction is sampled from a probabilistic model. Must be `1` for deterministic models.
        verbose
            Optionally, set the prediction verbosity. Not effective for all models.
        show_warnings
            Optionally, control whether warnings are shown. Not effective for all models.

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

    @property
    def _model_encoder_settings(
        self,
    ) -> tuple[
        Optional[int],
        Optional[int],
        bool,
        bool,
        Optional[list[int]],
        Optional[list[int]],
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
        """Controls whether encodings should be generated in :func:`FutureCovariatesLocalForecastingModel.predict()``"""
        return False

    @property
    def extreme_lags(
        self,
    ) -> tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        int,
        Optional[int],
    ]:
        # TODO: LocalForecastingModels do not yet handle extreme lags properly. Especially
        #  TransferableFutureCovariatesLocalForecastingModel, where there is a difference between fit and predict mode)
        #  do not yet. In general, Local models train on the entire series (input=output), different to Global models
        #  that use an input to predict an output.
        return -self.min_train_series_length, -1, None, None, 0, 0, 0, None


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
        verbose: bool = False,
        show_warnings: bool = True,
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
            Number of times a prediction is sampled from a probabilistic model. Must be `1` for deterministic models.
        verbose
            Optionally, set the prediction verbosity. Not effective for all models.
        show_warnings
            Optionally, control whether warnings are shown. Not effective for all models.

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
    ) -> tuple[
        Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]
    ]:
        raise_if(
            self.encoders is None or not self.encoders.encoding_available,
            "Encodings are not available. Consider adding parameter `add_encoders` at model creation and fitting the "
            "model with `model.fit()` before.",
            logger=logger,
        )
        return self.generate_fit_predict_encodings(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

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

    @property
    def supports_transferrable_series_prediction(self) -> bool:
        """
        Whether the model supports prediction for any input `series`.
        """
        return True

    @property
    def _supports_non_retrainable_historical_forecasts(self) -> bool:
        return True

    @property
    def _supress_generate_predict_encoding(self) -> bool:
        return True
