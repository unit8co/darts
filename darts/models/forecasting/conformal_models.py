"""
Conformal Models
---------------

A collection of conformal prediction models for pre-trained global forecasting models.
"""

import copy
import os
from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd

from darts import TimeSeries, metrics
from darts.logging import get_logger, raise_log
from darts.metrics.metrics import METRIC_TYPE
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.models.utils import TORCH_AVAILABLE
from darts.utils import _build_tqdm_iterator, _with_sanity_checks
from darts.utils.historical_forecasts.utils import (
    _adjust_historical_forecasts_time_index,
)
from darts.utils.timeseries_generation import _build_forecast_series
from darts.utils.ts_utils import (
    SeriesType,
    get_series_seq_type,
    series2seq,
)
from darts.utils.utils import (
    _check_quantiles,
    generate_index,
    likelihood_component_names,
    n_steps_between,
    quantile_names,
    random_method,
    sample_from_quantiles,
)

if TORCH_AVAILABLE:
    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
else:
    TorchForecastingModel = None

logger = get_logger(__name__)


class ConformalModel(GlobalForecastingModel, ABC):
    @random_method
    def __init__(
        self,
        model: GlobalForecastingModel,
        quantiles: List[float],
        symmetric: bool = True,
        cal_length: Optional[int] = None,
        num_samples: int = 500,
        random_state: Optional[int] = None,
        stride_cal: bool = False,
    ):
        """Base Conformal Prediction Model.

        Base class for any probabilistic conformal model. A conformal model calibrates the predictions from any
        pre-trained global forecasting model. It does not have to be trained, and can generated calibrated forecasts
        directly using the underlying trained forecasting model. Since it is a probabilistic model, you can generate
        forecasts in two ways (when calling `predict()`, `historical_forecasts()`, ...):

        - Predict the calibrated quantile intervals directly: Pass parameters `predict_likelihood_parameters=True`, and
          `num_samples=1` to the forecast method.
        - Predict stochastic samples from the calibrated quantile intervals: Pass parameters
          `predict_likelihood_parameters=False`, and `num_samples>>1` to the forecast method.

        Conformal models can be applied to any of Darts' global forecasting model, as long as the model has been
        fitted before. In general the workflow of the models to produce one calibrated forecast/prediction is as
        follows:

        - Extract a calibration set: The number of calibration examples from the most recent past to use for one
          conformal prediction can be defined at model creation with parameter `cal_length`. If `stride_cal` is `True`,
          then the same `stride` from the forecasting methods is applied to the calibration set, and more calibration
          examples are required (`cal_length * stride` historical forecasts that were generated with `stride=1`).
          To make your life simpler, we support two modes:
            - Automatic extraction of the calibration set from the past of your input series (`series`,
              `past_covariates`, ...). This is the default mode and our predict/forecasting/backtest/.... API is
              identical to any other forecasting model
            - Supply a fixed calibration set with parameters `cal_series`, `cal_past_covariates`, ... .
        - Generate historical forecasts on the calibration set (using the forecasting model)
        - Compute the errors/non-conformity scores (specific to each conformal model) on these historical forecasts
        - Compute the quantile values from the errors / non-conformity scores (using our desired quantiles set at model
          creation with parameter `quantiles`).
        - Compute the conformal prediction: Add the calibrated intervals to (or adjust the existing intervals of) the
          forecasting model's predictions.

        Some notes:

        - When computing historical_forecasts(), backtest(), residuals(), ... the above is applied for each forecast
          (the forecasting model's historical forecasts are only generated once for efficiency).
        - For multi-horizon forecasts, the above is applied for each step in the horizon separately

        Parameters
        ----------
        model
            A pre-trained global forecasting model.
        quantiles
            A list of quantiles centered around the median `q=0.5` to use. For example quantiles
            [0.1, 0.2, 0.5, 0.8 0.9] correspond to two intervals with (0.9 - 0.1) = 80%, and (0.8 - 0.2) 60% coverage
            around the median (model forecast).
        symmetric
            Whether to use symmetric non-conformity scores. If `False`, uses asymmetric scores (individual scores
            for lower- and upper quantile interval bounds).
        cal_length
            The number of past forecast residuals/errors to consider as calibration input for each conformal forecast.
            If `None`, considers all past residuals.
        num_samples
            Number of times a prediction is sampled from the underlying `model` if it is probabilistic. Uses `1` for
            deterministic models. This is different to the `num_samples` produced by the conformal model which can be
            set in downstream forecasting tasks.
        random_state
            Control the randomness of probabilistic conformal forecasts (sample generation) across different runs.
        stride_cal
            Whether to apply the same historical forecast `stride` to the non-conformity scores of the calibration set.
        """
        if not isinstance(model, GlobalForecastingModel) or not model._fit_called:
            raise_log(
                ValueError("`model` must be a pre-trained `GlobalForecastingModel`."),
                logger=logger,
            )
        _check_quantiles(quantiles)
        super().__init__(add_encoders=None)

        # quantiles and interval setup
        self.quantiles = np.array(quantiles)
        self.idx_median = quantiles.index(0.5)
        self.q_interval = [
            (q_l, q_h)
            for q_l, q_h in zip(
                quantiles[: self.idx_median], quantiles[self.idx_median + 1 :][::-1]
            )
        ]
        self.interval_range = np.array([
            q_high - q_low
            for q_high, q_low in zip(
                self.quantiles[self.idx_median + 1 :][::-1],
                self.quantiles[: self.idx_median],
            )
        ])
        if symmetric:
            # symmetric considers both tails together
            self.interval_range_sym = copy.deepcopy(self.interval_range)
        else:
            # asymmetric considers tails separately
            self.interval_range_sym = 1 - (1 - self.interval_range) / 2
        self.symmetric = symmetric

        # model setup
        self.model = model
        self.cal_length = cal_length
        self.stride_cal = stride_cal
        self.num_samples = num_samples if model.supports_probabilistic_prediction else 1
        self._likelihood = "quantile"
        self._fit_called = True

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        **kwargs,
    ) -> "ConformalModel":
        """Fit/train the underlying forecasting model on (potentially multiple) series.

        Optionally, one or multiple past and/or future covariates series can be provided as well, depending on the
        forecasting model used. The number of covariates series must match the number of target series.

        Notes
        -----
        Conformal Models do not required calling `fit()`, since they use pre-trained global forecasting models.
        You can call `predict()` directly. Also, make sure that the input series used in `predict()` corresponds to
        a calibration set, and not the same as used during training with `fit()`.

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
        # does not have to be trained, but we allow it for unified API
        self.model.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            **kwargs,
        )
        return self

    def predict(
        self,
        n: int,
        series: Union[TimeSeries, Sequence[TimeSeries]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        stride: int = 1,
        num_samples: int = 1,
        verbose: bool = False,
        predict_likelihood_parameters: bool = False,
        show_warnings: bool = True,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        """Forecasts calibrated quantile intervals (or samples from calibrated intervals) for `n` time steps after the
        end of the `series`.

        It is important that the input series for prediction correspond to a calibration set - a set different to the
        series that the underlying forecasting `model` was trained one.

        Since it is a probabilistic model, you can generate forecasts in two ways:

        - Predict the calibrated quantile intervals directly: Pass parameters `predict_likelihood_parameters=True`, and
          `num_samples=1` to the forecast method.
        - Predict stochastic samples from the calibrated quantile intervals: Pass parameters
          `predict_likelihood_parameters=False`, and `num_samples>>1` to the forecast method.

        Under the hood, the simplified workflow to produce one calibrated forecast/prediction for every step in the
        horizon `n` is as follows:

        - Extract a calibration set: The number of calibration examples from the most recent past to use for one
          conformal prediction can be defined at model creation with parameter `cal_length`. To make your life simpler,
          we support two modes:
            - Automatic extraction of the calibration set from the past of your input series (`series`,
              `past_covariates`, ...). This is the default mode.
            - Supply a fixed calibration set with parameters `cal_series`, `cal_past_covariates`, ... .
        - Generate historical forecasts on the calibration set (using the forecasting model)
        - Compute the errors/non-conformity scores (specific to each conformal model) on these historical forecasts
        - Compute the quantile values from the errors / non-conformity scores (using our desired quantiles set at model
          creation with parameter `quantiles`).
        - Compute the conformal prediction: Add the calibrated intervals to (or adjust the existing intervals of) the
          forecasting model's predictions.

        Parameters
        ----------
        n
            Forecast horizon - the number of time steps after the end of the series for which to produce predictions.
        series
            A series or sequence of series, representing the history of the target series whose future is to be
            predicted. If `cal_series` is `None`, will use the past of this series for calibration.
        past_covariates
            Optionally, a (sequence of) past-observed covariate time series for every input time series in `series`.
            Their dimension must match that of the past covariates used for training. If `cal_series` is `None`, will
            use this series for calibration.
        future_covariates
            Optionally, a (sequence of) future-known covariate time series for every input time series in `series`.
            Their dimension must match that of the past covariates used for training. If `cal_series` is `None`, will
            use this series for calibration.
        cal_series
            Optionally, a (sequence of) target series for every input time series in `series` to use for calibration
            instead of `series`.
        cal_past_covariates
            Optionally, a (sequence of) past covariates series for every input time series in `series` to use for
            calibration instead of `past_covariates`.
        cal_future_covariates
            Optionally, a future covariates series for every input time series in `series` to use for calibration
            instead of `future_covariates`.
        stride
            The number of time steps between two consecutive predictions (and non-conformity scores) of the
            calibration set. Right-bound by the first time step of the generated forecast.
        num_samples
            Number of times a prediction is sampled from the calibrated quantile predictions using linear
            interpolation in-between the quantiles. For larger values, the sample distribution approximates the
            calibrated quantile predictions.
        verbose
            Whether to print the progress.
        predict_likelihood_parameters
            If set to `True`, generates the quantile predictions directly. Only supported with `num_samples = 1`.
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
        if series is None:
            # then there must be a single TS, and that was saved in super().fit as self.training_series
            if self.model.training_series is None:
                raise_log(
                    ValueError(
                        "Input `series` must be provided. This is the result either from fitting on multiple series, "
                        "or from not having fit the model yet."
                    ),
                    logger,
                )
            series = self.model.training_series

        called_with_single_series = get_series_seq_type(series) == SeriesType.SINGLE

        # guarantee that all inputs are either list of TimeSeries or None
        series = series2seq(series)
        if past_covariates is None and self.model.past_covariate_series is not None:
            past_covariates = [self.model.past_covariate_series] * len(series)
        if future_covariates is None and self.model.future_covariate_series is not None:
            future_covariates = [self.model.future_covariate_series] * len(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)

        super().predict(
            n,
            series,
            past_covariates,
            future_covariates,
            num_samples,
            verbose,
            predict_likelihood_parameters,
            show_warnings,
        )

        # if a calibration set is given, use it. Otherwise, use past of input as calibration
        if cal_series is None:
            cal_series = series
            cal_past_covariates = past_covariates
            cal_future_covariates = future_covariates

        cal_series = series2seq(cal_series)
        if len(cal_series) != len(series):
            raise_log(
                ValueError(
                    f"Mismatch between number of `cal_series` ({len(cal_series)}) "
                    f"and number of `series` ({len(series)})."
                ),
                logger=logger,
            )
        cal_past_covariates = series2seq(cal_past_covariates)
        cal_future_covariates = series2seq(cal_future_covariates)

        # generate model forecast to calibrate
        preds = self.model.predict(
            n=n,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=self.num_samples,
            verbose=verbose,
            predict_likelihood_parameters=False,
            show_warnings=show_warnings,
        )
        # convert to multi series case with `last_points_only=False`
        preds = [[pred] for pred in preds]

        # generate all possible forecasts for calibration
        cal_hfcs = self.model.historical_forecasts(
            series=cal_series,
            past_covariates=cal_past_covariates,
            future_covariates=cal_future_covariates,
            num_samples=self.num_samples,
            forecast_horizon=n,
            retrain=False,
            overlap_end=True,
            last_points_only=False,
            verbose=verbose,
            show_warnings=show_warnings,
            predict_likelihood_parameters=False,
        )
        cal_preds = self._calibrate_forecasts(
            series=series,
            forecasts=preds,
            cal_series=cal_series,
            cal_forecasts=cal_hfcs,
            num_samples=num_samples,
            forecast_horizon=n,
            stride=stride,
            overlap_end=True,
            last_points_only=False,
            verbose=verbose,
            show_warnings=show_warnings,
            predict_likelihood_parameters=predict_likelihood_parameters,
        )
        # convert historical forecasts output to simple forecast / prediction
        if called_with_single_series:
            return cal_preds[0][0]
        else:
            return [cp[0] for cp in cal_preds]

    @_with_sanity_checks("_historical_forecasts_sanity_checks")
    def historical_forecasts(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
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
        fit_kwargs: Optional[Dict[str, Any]] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ) -> Union[TimeSeries, List[TimeSeries], List[List[TimeSeries]]]:
        """Generates calibrated historical forecasts by simulating predictions at various points in time throughout the
        history of the provided (potentially multiple) `series`. This process involves retrospectively applying the
        model to different time steps, as if the forecasts were made in real-time at those specific moments. This
        allows for an evaluation of the model's performance over the entire duration of the series, providing insights
        into its predictive accuracy and robustness across different historical periods.

        Currently, conformal models only support the pre-trained historical forecasts mode (`retrain=False`).
        Parameters `retrain` and `train_length` are ignored.

        **Pre-trained Mode:** First, all historical forecasts are generated using the underlying pre-trained global
        forecasting model (see :meth:`ForecastingModel.historical_forecasts()
        <darts.models.forecasting.forecasting_model.ForecastingModel.historical_forecasts>` for more info). Then it
        repeatedly builds a calibration set by either expanding from the beginning of the historical forecasts or by
        using a fixed-length `cal_length` (the start point can also be configured with `start` and `start_format`).
        The next forecast of length `forecast_horizon` is then calibrated on this calibration set. Subsequently, the
        end of the calibration set is moved forward by `stride` time steps, and the process is repeated.
        You can also use a fixed calibration set to calibrate all forecasts equally by passing `cal_series`, and
        optional `cal_past_covariates` and `cal_future_covariates`.

        By default, with `last_points_only=True`, this method returns a single time series (or a sequence of time
        series) composed of the last point from each calibrated historical forecast. This time series will thus have a
        frequency of `series.freq * stride`.
        If `last_points_only=False`, it will instead return a list (or a sequence of lists) of the full calibrate
        historical forecast series each with frequency `series.freq`.

        Parameters
        ----------
        series
            A (sequence of) target time series used to successively compute the historical forecasts. If `cal_series`
            is `None`, will use the past of this series for calibration.
        past_covariates
            Optionally, a (sequence of) past-observed covariate time series for every input time series in `series`.
            Their dimension must match that of the past covariates used for training. If `cal_series` is `None`, will
            use this series for calibration.
        future_covariates
            Optionally, a (sequence of) future-known covariate time series for every input time series in `series`.
            Their dimension must match that of the past covariates used for training. If `cal_series` is `None`, will
            use this series for calibration.
        cal_series
            Optionally, a (sequence of) target series for every input time series in `series` to use as a fixed
            calibration set instead of `series`.
        cal_past_covariates
            Optionally, a (sequence of) past covariates series for every input time series in `series` to use as a fixed
            calibration set instead of `past_covariates`.
        cal_future_covariates
            Optionally, a future covariates series for every input time series in `series` to use as a fixed
            calibration set instead of `future_covariates`.
        forecast_horizon
            The forecast horizon for the predictions.
        num_samples
            Number of times a prediction is sampled from the calibrated quantile predictions using linear
            interpolation in-between the quantiles. For larger values, the sample distribution approximates the
            calibrated quantile predictions.
        train_length
            Currently ignored by conformal models.
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

            Note: If the model uses a shifted output (`output_chunk_shift > 0`), then the first predicted point is also
            shifted by `output_chunk_shift` points into the future.
            Note: Raises a ValueError if `start` yields a time outside the time index of `series`.
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
            Currently ignored by conformal models.
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
            If set to `True`, generates the quantile predictions directly. Only supported with `num_samples = 1`.
        enable_optimization
            Whether to use the optimized version of `historical_forecasts` when supported and available.
            Default: ``True``.
        fit_kwargs
            Currently ignored by conformal models.
        predict_kwargs
            Optionally, some additional arguments passed to the model `predict()` method.
        sample_weight
            Currently ignored by conformal models.

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
        called_with_single_series = get_series_seq_type(series) == SeriesType.SINGLE
        series = series2seq(series)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)

        if cal_series is not None:
            cal_series = series2seq(cal_series)
            if len(cal_series) != len(series):
                raise_log(
                    ValueError(
                        f"Mismatch between number of `cal_series` ({len(cal_series)}) "
                        f"and number of `series` ({len(series)})."
                    ),
                    logger=logger,
                )
            cal_past_covariates = series2seq(cal_past_covariates)
            cal_future_covariates = series2seq(cal_future_covariates)

        # generate all possible forecasts (overlap_end=True) to have enough residuals
        hfcs = self.model.historical_forecasts(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            num_samples=self.num_samples,
            forecast_horizon=forecast_horizon,
            retrain=False,
            overlap_end=overlap_end,
            last_points_only=last_points_only,
            verbose=verbose,
            show_warnings=show_warnings,
            predict_likelihood_parameters=False,
            enable_optimization=enable_optimization,
            fit_kwargs=fit_kwargs,
            predict_kwargs=predict_kwargs,
        )
        # optionally, generate calibration forecasts
        if cal_series is None:
            cal_hfcs = None
        else:
            cal_hfcs = self.model.historical_forecasts(
                series=cal_series,
                past_covariates=cal_past_covariates,
                future_covariates=cal_future_covariates,
                num_samples=self.num_samples,
                forecast_horizon=forecast_horizon,
                retrain=False,
                overlap_end=True,
                last_points_only=last_points_only,
                verbose=verbose,
                show_warnings=show_warnings,
                predict_likelihood_parameters=False,
                enable_optimization=enable_optimization,
                fit_kwargs=fit_kwargs,
                predict_kwargs=predict_kwargs,
            )
        calibrated_forecasts = self._calibrate_forecasts(
            series=series,
            forecasts=hfcs,
            cal_series=cal_series,
            cal_forecasts=cal_hfcs,
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
        )
        return (
            calibrated_forecasts[0]
            if called_with_single_series
            else calibrated_forecasts
        )

    def backtest(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
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
        metric: Union[METRIC_TYPE, List[METRIC_TYPE]] = metrics.mape,
        reduction: Union[Callable[..., float], None] = np.mean,
        verbose: bool = False,
        show_warnings: bool = True,
        predict_likelihood_parameters: bool = False,
        enable_optimization: bool = True,
        metric_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ) -> Union[float, np.ndarray, List[float], List[np.ndarray]]:
        """Compute error values that the model produced for historical forecasts on (potentially multiple) `series`.

        If `historical_forecasts` are provided, the metric(s) (given by the `metric` function) is evaluated directly on
        all forecasts and actual values. The same `series` and `last_points_only` value must be passed that were used
        to generate the historical forecasts. Finally, the method returns an optional `reduction` (the mean by default)
        of all these metric scores.

        If `historical_forecasts` is ``None``, it first generates the historical forecasts with the parameters given
        below (see :meth:`ConformalModel.historical_forecasts()
        <darts.models.forecasting.conformal_models.ConformalModel.historical_forecasts>` for more info) and then
        evaluates as described above.

        The metric(s) can be further customized `metric_kwargs` (e.g. control the aggregation over components, time
        steps, multiple series, other required arguments such as `q` for quantile metrics, ...).

        Notes
        -----
        Darts has several metrics to evaluate probabilistic forecasts. For conformal models, we recommend using
        quantile interval metrics (see `here <https://unit8co.github.io/darts/generated_api/darts.metrics.html>`_).
        You can specify which intervals to evaluate by setting `metric_kwargs={'q_interval': my_intervals}`. To check
        all intervals used by your conformal model `my_model`, you can set ``{'q_interval': my_model.q_interval}``.

        Parameters
        ----------
        series
            A (sequence of) target time series used to successively compute the historical forecasts. If `cal_series`
            is `None`, will use the past of this series for calibration.
        past_covariates
            Optionally, a (sequence of) past-observed covariate time series for every input time series in `series`.
            Their dimension must match that of the past covariates used for training. If `cal_series` is `None`, will
            use this series for calibration.
        future_covariates
            Optionally, a (sequence of) future-known covariate time series for every input time series in `series`.
            Their dimension must match that of the past covariates used for training. If `cal_series` is `None`, will
            use this series for calibration.
        cal_series
            Optionally, a (sequence of) target series for every input time series in `series` to use as a fixed
            calibration set instead of `series`.
        cal_past_covariates
            Optionally, a (sequence of) past covariates series for every input time series in `series` to use as a fixed
            calibration set instead of `past_covariates`.
        cal_future_covariates
            Optionally, a future covariates series for every input time series in `series` to use as a fixed
            calibration set instead of `future_covariates`.
        historical_forecasts
            Optionally, the (or a sequence of / a sequence of sequences of) historical forecasts time series to be
            evaluated. Corresponds to the output of :meth:`historical_forecasts()
            <darts.models.forecasting.conformal_models.ConformalModel.historical_forecasts>`. The same `series` and
            `last_points_only` values must be passed that were used to generate the historical forecasts. If provided,
            will skip historical forecasting and ignore all parameters except `series`, `last_points_only`, `metric`,
            and `reduction`.
        forecast_horizon
            The forecast horizon for the predictions.
        num_samples
            Number of times a prediction is sampled from the calibrated quantile predictions using linear
            interpolation in-between the quantiles. For larger values, the sample distribution approximates the
            calibrated quantile predictions.
        train_length
            Currently ignored by conformal models.
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

            Note: If the model uses a shifted output (`output_chunk_shift > 0`), then the first predicted point is also
            shifted by `output_chunk_shift` points into the future.
            Note: Raises a ValueError if `start` yields a time outside the time index of `series`.
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
            Currently ignored by conformal models.
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
            If set to `True`, generates the quantile predictions directly. Only supported with `num_samples = 1`.
        enable_optimization
            Whether to use the optimized version of `historical_forecasts` when supported and available.
            Default: ``True``.
        metric_kwargs
            Additional arguments passed to `metric()`, such as `'n_jobs'` for parallelization, `'component_reduction'`
            for reducing the component wise metrics, seasonality `'m'` for scaled metrics, etc. Will pass arguments to
            each metric separately and only if they are present in the corresponding metric signature. Parameter
            `'insample'` for scaled metrics (e.g. mase`, `rmsse`, ...) is ignored, as it is handled internally.
        fit_kwargs
            Currently ignored by conformal models.
        predict_kwargs
            Optionally, some additional arguments passed to the model `predict()` method.
        sample_weight
            Currently ignored by conformal models.

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
        historical_forecasts = historical_forecasts or self.historical_forecasts(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            cal_series=cal_series,
            cal_past_covariates=cal_past_covariates,
            cal_future_covariates=cal_future_covariates,
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
            fit_kwargs=fit_kwargs,
            predict_kwargs=predict_kwargs,
            overlap_end=overlap_end,
            sample_weight=sample_weight,
        )
        return super().backtest(
            series=series,
            historical_forecasts=historical_forecasts,
            forecast_horizon=forecast_horizon,
            num_samples=num_samples,
            train_length=train_length,
            start=start,
            start_format=start_format,
            stride=stride,
            retrain=retrain,
            overlap_end=overlap_end,
            last_points_only=last_points_only,
            metric=metric,
            reduction=reduction,
            verbose=verbose,
            show_warnings=show_warnings,
            predict_likelihood_parameters=predict_likelihood_parameters,
            enable_optimization=enable_optimization,
            metric_kwargs=metric_kwargs,
            fit_kwargs=fit_kwargs,
            predict_kwargs=predict_kwargs,
            sample_weight=sample_weight,
        )

    def residuals(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
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
        metric_kwargs: Optional[Dict[str, Any]] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
        values_only: bool = False,
    ) -> Union[TimeSeries, List[TimeSeries], List[List[TimeSeries]]]:
        """Compute the residuals that the model produced for historical forecasts on (potentially multiple) `series`.

        This function computes the difference (or one of Darts' "per time step" metrics) between the actual
        observations from `series` and the fitted values obtained by training the model on `series` (or using a
        pre-trained model with `retrain=False`). Not all models support fitted values, so we use historical forecasts
        as an approximation for them.

        In sequence this method performs:

        - use pre-computed `historical_forecasts` or compute historical forecasts for each series (see
          :meth:`~darts.models.forecasting.conformal_models.ConformalModel.historical_forecasts` for more details).
          How the historical forecasts are generated can be configured with parameters `num_samples`, `train_length`,
          `start`, `start_format`, `forecast_horizon`, `stride`, `retrain`, `last_points_only`, `fit_kwargs`, and
          `predict_kwargs`.
        - compute a backtest using a "per time step" `metric` between the historical forecasts and `series` per
          component/column and time step (see
          :meth:`~darts.models.forecasting.conformal_models.ConformalModel.backtest` for more details). By default,
          uses the residuals :func:`~darts.metrics.metrics.err` (error) as a `metric`.
        - create and return `TimeSeries` (or simply a np.ndarray with `values_only=True`) with the time index from
          historical forecasts, and values from the metrics per component and time step.

        This method works for single or multiple univariate or multivariate series.
        It uses the median prediction (when dealing with stochastic forecasts).

        Notes
        -----
        Darts has several metrics to evaluate probabilistic forecasts. For conformal models, we recommend using
        "per time step" quantile interval metrics (see `here
        <https://unit8co.github.io/darts/generated_api/darts.metrics.html>`_). You can specify which intervals to
        evaluate by setting `metric_kwargs={'q_interval': my_intervals}`. To check all intervals used by your conformal
        model `my_model`, you can set ``{'q_interval': my_model.q_interval}``.

        Parameters
        ----------
        series
            A (sequence of) target time series used to successively compute the historical forecasts. If `cal_series`
            is `None`, will use the past of this series for calibration.
        past_covariates
            Optionally, a (sequence of) past-observed covariate time series for every input time series in `series`.
            Their dimension must match that of the past covariates used for training. If `cal_series` is `None`, will
            use this series for calibration.
        future_covariates
            Optionally, a (sequence of) future-known covariate time series for every input time series in `series`.
            Their dimension must match that of the past covariates used for training. If `cal_series` is `None`, will
            use this series for calibration.
        cal_series
            Optionally, a (sequence of) target series for every input time series in `series` to use as a fixed
            calibration set instead of `series`.
        cal_past_covariates
            Optionally, a (sequence of) past covariates series for every input time series in `series` to use as a fixed
            calibration set instead of `past_covariates`.
        cal_future_covariates
            Optionally, a future covariates series for every input time series in `series` to use as a fixed
            calibration set instead of `future_covariates`.
        historical_forecasts
            Optionally, the (or a sequence of / a sequence of sequences of) historical forecasts time series to be
            evaluated. Corresponds to the output of :meth:`historical_forecasts()
            <darts.models.forecasting.conformal_models.ConformalModel.historical_forecasts>`. The same `series` and
            `last_points_only` values must be passed that were used to generate the historical forecasts. If provided,
            will skip historical forecasting and ignore all parameters except `series`, `last_points_only`, `metric`,
            and `reduction`.
        forecast_horizon
            The forecast horizon for the predictions.
        num_samples
            Number of times a prediction is sampled from the calibrated quantile predictions using linear
            interpolation in-between the quantiles. For larger values, the sample distribution approximates the
            calibrated quantile predictions.
        train_length
            Currently ignored by conformal models.
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

            Note: If the model uses a shifted output (`output_chunk_shift > 0`), then the first predicted point is also
            shifted by `output_chunk_shift` points into the future.
            Note: Raises a ValueError if `start` yields a time outside the time index of `series`.
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
            Currently ignored by conformal models.
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
            If set to `True`, generates the quantile predictions directly. Only supported with `num_samples = 1`.
        enable_optimization
            Whether to use the optimized version of `historical_forecasts` when supported and available.
            Default: ``True``.
        metric_kwargs
            Additional arguments passed to `metric()`, such as `'n_jobs'` for parallelization, `'m'` for scaled
            metrics, etc. Will pass arguments only if they are present in the corresponding metric signature. Ignores
            reduction arguments `"series_reduction", "component_reduction", "time_reduction"`, and parameter
            `'insample'` for scaled metrics (e.g. mase`, `rmsse`, ...), as they are handled internally.
        fit_kwargs
            Currently ignored by conformal models.
        predict_kwargs
            Optionally, some additional arguments passed to the model `predict()` method.
        sample_weight
            Currently ignored by conformal models.
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
        historical_forecasts = historical_forecasts or self.historical_forecasts(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            cal_series=cal_series,
            cal_past_covariates=cal_past_covariates,
            cal_future_covariates=cal_future_covariates,
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
            fit_kwargs=fit_kwargs,
            predict_kwargs=predict_kwargs,
            overlap_end=overlap_end,
            sample_weight=sample_weight,
        )
        return super().residuals(
            series=series,
            historical_forecasts=historical_forecasts,
            forecast_horizon=forecast_horizon,
            num_samples=num_samples,
            train_length=train_length,
            start=start,
            start_format=start_format,
            stride=stride,
            retrain=retrain,
            overlap_end=overlap_end,
            last_points_only=last_points_only,
            metric=metric,
            verbose=verbose,
            show_warnings=show_warnings,
            predict_likelihood_parameters=predict_likelihood_parameters,
            enable_optimization=enable_optimization,
            metric_kwargs=metric_kwargs,
            fit_kwargs=fit_kwargs,
            predict_kwargs=predict_kwargs,
            sample_weight=sample_weight,
            values_only=values_only,
        )

    @random_method
    def _calibrate_forecasts(
        self,
        series: Sequence[TimeSeries],
        forecasts: Union[Sequence[Sequence[TimeSeries]], Sequence[TimeSeries]],
        cal_series: Optional[Sequence[TimeSeries]] = None,
        cal_forecasts: Optional[
            Union[Sequence[Sequence[TimeSeries]], Sequence[TimeSeries]]
        ] = None,
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
    ) -> Union[TimeSeries, List[TimeSeries], List[List[TimeSeries]]]:
        """Generate calibrated historical forecasts.

        In general the workflow of the models to produce one calibrated forecast/prediction per step in the horizon
        is as follows:

        - Generate historical forecasts for `series` and optional calibration set (`cal_series`) (using the forecasting
          model)
        - Extract a calibration set: The forecasts from the most recent past to use as calibration
          for one conformal prediction. The number of examples to use can be defined at model creation with parameter
          `cal_length`. We support two modes:
            - Automatic extraction of the calibration set from the past of your input series (`series`,
              `past_covariates`, ...). This is the default mode and our predict/forecasting/backtest/.... API is
              identical to any other forecasting model
            - Supply a fixed calibration set with parameters `cal_series`, `cal_past_covariates`, ... .
        - Compute the errors/non-conformity scores (specific to each conformal model) on these historical forecasts
        - Compute the quantile values from the errors / non-conformity scores (using our desired quantiles set at model
          creation with parameter `quantiles`).
        - Compute the conformal prediction: Add the calibrated intervals to (or adjust the existing intervals of) the
          forecasting model's predictions.
        """
        # TODO: add proper handling of `cal_stride` > 1
        # cal_stride = stride if self.stride_cal else 1
        cal_length = self.cal_length
        metric, metric_kwargs = self._residuals_metric
        residuals = self.model.residuals(
            series=series if cal_series is None else cal_series,
            historical_forecasts=forecasts if cal_series is None else cal_forecasts,
            overlap_end=overlap_end if cal_series is None else True,
            last_points_only=last_points_only,
            verbose=verbose,
            show_warnings=show_warnings,
            values_only=True,
            metric=metric,
            metric_kwargs=metric_kwargs,
        )

        outer_iterator = enumerate(zip(series, forecasts, residuals))
        if len(series) > 1:
            # Use tqdm on the outer loop only if there's more than one series to iterate over
            # (otherwise use tqdm on the inner loop).
            outer_iterator = _build_tqdm_iterator(
                outer_iterator,
                verbose,
                total=len(series),
                desc="conformal forecasts",
            )

        cp_hfcs = []
        for series_idx, (series_, s_hfcs, res) in outer_iterator:
            cp_preds = []

            # no historical forecasts were generated
            if not s_hfcs:
                cp_hfcs.append(cp_preds)
                continue

            last_hfc = s_hfcs if last_points_only else s_hfcs[-1]

            # compute the minimum required number of useful calibration residuals
            # at least one or `cal_length` examples
            min_n_cal = cal_length or 1
            # `last_points_only=False` requires additional examples to use most recent information
            # from all steps in the horizon
            if not last_points_only:
                min_n_cal += forecast_horizon - 1

            # determine first forecast index for conformal prediction
            if cal_series is None:
                # we need at least one residual per point in the horizon prior to the first conformal forecast
                first_idx_train = forecast_horizon + self.output_chunk_shift
                # plus some additional examples based on `cal_length`
                if cal_length is not None:
                    first_idx_train += cal_length - 1
                # check if later we need to drop some residuals without useful information (unknown residuals)
                if overlap_end:
                    delta_end = n_steps_between(
                        end=last_hfc.end_time(),
                        start=series_.end_time(),
                        freq=series_.freq,
                    )
                else:
                    delta_end = 0
            else:
                # calibration set is decoupled from `series` forecasts; we can start with the first forecast
                first_idx_train = 0
                # check if we need to drop some residuals without useful information
                cal_series_ = cal_series[series_idx]
                cal_last_hfc = cal_forecasts[series_idx][-1]
                delta_end = n_steps_between(
                    end=cal_last_hfc.end_time(),
                    start=cal_series_.end_time(),
                    freq=cal_series_.freq,
                )

            # drop residuals without useful information
            last_res_idx = None
            if last_points_only and delta_end > 0:
                # useful residual information only up until the forecast
                # ending at the last time step in `series`
                last_res_idx = -delta_end
            elif not last_points_only and delta_end >= forecast_horizon:
                # useful residual information only up until the forecast
                # starting at the last time step in `series`
                last_res_idx = -(delta_end - forecast_horizon + 1)
            if last_res_idx is None and cal_series is None:
                # drop at least the one residuals/forecast from the end, since we can only use prior residuals
                last_res_idx = -(self.output_chunk_shift + 1)
                # with last points only, ignore the last `horizon` residuals to avoid look-ahead bias
                if last_points_only:
                    last_res_idx -= forecast_horizon - 1

            if last_res_idx is not None:
                res = res[:last_res_idx]

            if first_idx_train >= len(s_hfcs) or len(res) < min_n_cal:
                set_name = "" if cal_series is None else "cal_"
                raise_log(
                    ValueError(
                        "Could not build the minimum required calibration input with the provided "
                        f"`{set_name}series` and `{set_name}*_covariates` at series index: {series_idx}. "
                        f"Expected to generate at least `{min_n_cal}` calibration forecasts with known residuals "
                        f"before the first conformal forecast, but could only generate `{len(res)}`."
                    ),
                    logger=logger,
                )
            # adjust first index based on `start`
            first_idx_start = 0
            if start is not None:
                # adjust forecastable index in case of output shift or `last_points_only=True`
                adjust_idx = (
                    self.output_chunk_shift
                    + int(last_points_only) * (forecast_horizon - 1)
                ) * series_.freq
                historical_forecastable_index = (
                    s_hfcs[first_idx_train].start_time() - adjust_idx,
                    s_hfcs[-1].start_time() - adjust_idx,
                )
                # TODO: add proper start handling with `cal_stride>1`
                # adjust forecastable index based on start, assuming hfcs were generated with `stride=1`
                first_idx_start, _ = _adjust_historical_forecasts_time_index(
                    series=series_,
                    series_idx=series_idx,
                    start=start,
                    start_format=start_format,
                    stride=stride,
                    historical_forecasts_time_index=historical_forecastable_index,
                    show_warnings=show_warnings,
                )
                # find position relative to start
                first_idx_start = n_steps_between(
                    first_idx_start + adjust_idx,
                    s_hfcs[0].start_time(),
                    freq=series_.freq,
                )

            # get final first index
            first_fc_idx = max([first_idx_train, first_idx_start])
            # bring into shape (forecasting steps, n components, n samples * n examples)
            if last_points_only:
                # -> (1, n components, n samples * n examples)
                res = res.T
            else:
                res = np.array(res)
                # -> (forecast horizon, n components, n samples * n examples)
                # rearrange the residuals to avoid look-ahead bias and to have the same number of examples per
                # point in the horizon. We want the most recent residuals in the past for each step in the horizon.
                # Meaning that to conformalize any forecast at some time `t` with `horizon=n`:
                #   - for `horizon=1` of that forecast calibrate with residuals from all 1-step forecasts up until
                #     forecast time `t-1`
                #   - for `horizon=n` of that forecast calibrate with residuals from all n-step forecasts up until
                #     forecast time `t-n`
                # The rearranged residuals will look as follows, where `res_ti_cj_hk` is the
                # residuals at time `ti` for component `cj` at forecasted step/horizon `hk`.
                # ```
                # [  # forecast horizon
                #     [  # components
                #         [res_t0_c0_h1, ...]  # residuals at different times
                #         [..., res_tn_cn_h1],
                #     ],
                #     ...,
                #     [
                #         [res_t0_c0_hn, ...],
                #         [..., res_tn_cn_hn],
                #     ],
                # ]
                # ```
                res_ = []
                for irr in range(forecast_horizon - 1, -1, -1):
                    res_end_idx = -(forecast_horizon - (irr + 1))
                    res_.append(res[irr : res_end_idx or None, abs(res_end_idx)])
                res = np.concatenate(res_, axis=2).T

            # get the last forecast index based on the residual examples
            if cal_series is None:
                last_fc_idx = res.shape[2] + (
                    forecast_horizon + self.output_chunk_shift
                )
            else:
                last_fc_idx = len(s_hfcs)

            q_hat = None
            # with a calibration set, the calibrated interval is constant across all forecasts
            if cal_series is not None:
                if cal_length is not None:
                    res = res[:, :, -cal_length:]
                q_hat = self._calibrate_interval(res)

            def conformal_predict(idx_, pred_vals_):
                if cal_series is None:
                    # get the last residual index for calibration, `cal_end` is exclusive
                    # to avoid look-ahead bias, use only residuals from before the historical forecast start point;
                    # for `last_points_only=True`, the last residual historically available at the forecasting
                    # point is `forecast_horizon + self.output_chunk_shift - 1` steps before. The same applies to
                    # `last_points_only=False` thanks to the residual rearrangement
                    cal_end = (
                        first_fc_idx
                        + idx_ * stride
                        - (forecast_horizon + self.output_chunk_shift - 1)
                    )
                    # first residual index is shifted back by the horizon to get `cal_length` points for
                    # the last point in the horizon
                    cal_start = cal_end - cal_length if cal_length is not None else None

                    cal_res = res[:, :, cal_start:cal_end]
                    q_hat_ = self._calibrate_interval(cal_res)
                else:
                    # with a calibration set, use a constant q_hat
                    q_hat_ = q_hat
                vals = self._apply_interval(pred_vals_, q_hat_)
                if not predict_likelihood_parameters:
                    vals = sample_from_quantiles(
                        vals, self.quantiles, num_samples=num_samples
                    )
                return vals

            # historical conformal prediction
            # for each forecast, compute calibrated quantile intervals based on past residuals
            if last_points_only:
                inner_iterator = enumerate(
                    s_hfcs.all_values(copy=False)[first_fc_idx:last_fc_idx:stride]
                )
            else:
                inner_iterator = enumerate(s_hfcs[first_fc_idx:last_fc_idx:stride])
            comp_names_out = (
                self._cp_component_names(series_)
                if predict_likelihood_parameters
                else None
            )
            if len(series) == 1:
                # Only use progress bar if there's no outer loop
                inner_iterator = _build_tqdm_iterator(
                    inner_iterator,
                    verbose,
                    total=(last_fc_idx - 1 - first_fc_idx) // stride + 1,
                    desc="conformal forecasts",
                )

            if last_points_only:
                for idx, pred_vals in inner_iterator:
                    pred_vals = np.expand_dims(pred_vals, 0)
                    cp_pred = conformal_predict(idx, pred_vals)
                    cp_preds.append(cp_pred)
                cp_preds = _build_forecast_series(
                    points_preds=np.concatenate(cp_preds, axis=0),
                    input_series=series_,
                    custom_columns=comp_names_out,
                    time_index=generate_index(
                        start=s_hfcs._time_index[first_fc_idx],
                        length=len(cp_preds),
                        freq=series_.freq * stride,
                        name=series_._time_index.name,
                    ),
                    with_static_covs=False,
                    with_hierarchy=False,
                )
            else:
                for idx, pred in inner_iterator:
                    pred_vals = pred.all_values(copy=False)
                    cp_pred = conformal_predict(idx, pred_vals)
                    cp_pred = _build_forecast_series(
                        points_preds=cp_pred,
                        input_series=series_,
                        custom_columns=comp_names_out,
                        time_index=pred._time_index,
                        with_static_covs=False,
                        with_hierarchy=False,
                    )
                    cp_preds.append(cp_pred)
            cp_hfcs.append(cp_preds)
        return cp_hfcs

    def save(
        self, path: Optional[Union[str, os.PathLike, BinaryIO]] = None, **pkl_kwargs
    ) -> None:
        """
        Saves the conformal model under a given path or file handle.

        Additionally, two files are stored if `self.model` is a `TorchForecastingModel`.

        Example for saving and loading a :class:`ConformalNaiveModel`:

            .. highlight:: python
            .. code-block:: python

                from darts.datasets import AirPassengersDataset
                from darts.models import ConformalNaiveModel, LinearRegressionModel

                series = AirPassengersDataset().load()
                forecasting_model = LinearRegressionModel(lags=4).fit(series)

                model = ConformalNaiveModel(
                    model=forecasting_model,
                    quantiles=[0.1, 0.5, 0.9],
                )

                model.save("my_model.pkl")
                model_loaded = ConformalNaiveModel.load("my_model.pkl")
            ..

        Parameters
        ----------
        path
            Path or file handle under which to save the ensemble model at its current state. If no path is specified,
            the ensemble model is automatically saved under ``"{ConformalNaiveModel}_{YYYY-mm-dd_HH_MM_SS}.pkl"``.
            If the forecasting model is a `TorchForecastingModel`, two files (model object and checkpoint) are saved
            under ``"{path}.{ModelClass}.pt"`` and ``"{path}.{ModelClass}.ckpt"``.
        pkl_kwargs
            Keyword arguments passed to `pickle.dump()`
        """

        if path is None:
            # default path
            path = self._default_save_path() + ".pkl"

        super().save(path, **pkl_kwargs)

        if TORCH_AVAILABLE and issubclass(type(self.model), TorchForecastingModel):
            path_tfm = f"{path}.{type(self.model).__name__}.pt"
            self.model.save(path=path_tfm)

    @staticmethod
    def load(path: Union[str, os.PathLike, BinaryIO]) -> "ConformalModel":
        model: ConformalModel = GlobalForecastingModel.load(path)

        if TORCH_AVAILABLE and issubclass(type(model.model), TorchForecastingModel):
            path_tfm = f"{path}.{type(model.model).__name__}.pt"
            model.model = TorchForecastingModel.load(path_tfm)
        return model

    @abstractmethod
    def _calibrate_interval(
        self, residuals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the lower and upper calibrated forecast intervals based on residuals.

        Parameters
        ----------
        residuals
            The residuals are expected to have shape (horizon, n components, n historical forecasts * n samples)
        """
        pass

    @abstractmethod
    def _apply_interval(self, pred: np.ndarray, q_hat: Tuple[np.ndarray, np.ndarray]):
        """Applies the calibrated interval to the predicted quantiles. Returns an array with `len(quantiles)`
        conformalized quantile predictions (lower quantiles, model forecast, upper quantiles) per component.

        E.g. output is `(target1_q1, target1_pred, target1_q2, target2_q1, ...)`
        """
        pass

    @property
    @abstractmethod
    def _residuals_metric(self) -> Tuple[METRIC_TYPE, Optional[dict]]:
        """Gives the "per time step" metric and optional metric kwargs used to compute residuals /
        non-conformity scores."""
        pass

    def _cp_component_names(self, input_series) -> List[str]:
        """Gives the component names for generated forecasts."""
        return likelihood_component_names(
            input_series.components, quantile_names(self.quantiles)
        )

    @property
    def output_chunk_length(self) -> Optional[int]:
        # conformal models can predict any horizon if the calibration set is large enough
        return None

    @property
    def output_chunk_shift(self) -> int:
        return self.model.output_chunk_shift

    @property
    def _model_encoder_settings(self):
        raise NotImplementedError(f"not supported by `{self.__class__.__name__}`.")

    @property
    def extreme_lags(
        self,
    ) -> Tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        int,
        Optional[int],
    ]:
        raise NotImplementedError(f"not supported by `{self.__class__.__name__}`.")

    @property
    def min_train_series_length(self) -> int:
        raise NotImplementedError(f"not supported by `{self.__class__.__name__}`.")

    @property
    def min_train_samples(self) -> int:
        raise NotImplementedError(f"not supported by `{self.__class__.__name__}`.")

    @property
    def supports_multivariate(self) -> bool:
        return self.model.supports_multivariate

    @property
    def supports_past_covariates(self) -> bool:
        return self.model.supports_past_covariates

    @property
    def supports_future_covariates(self) -> bool:
        return self.model.supports_future_covariates

    @property
    def supports_static_covariates(self) -> bool:
        return self.model.supports_static_covariates

    @property
    def supports_sample_weight(self) -> bool:
        return self.model.supports_sample_weight

    @property
    def supports_likelihood_parameter_prediction(self) -> bool:
        return True

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return True

    @property
    def uses_past_covariates(self) -> bool:
        return self.model.uses_past_covariates

    @property
    def uses_future_covariates(self) -> bool:
        return self.model.uses_future_covariates

    @property
    def uses_static_covariates(self) -> bool:
        return self.model.uses_static_covariates

    @property
    def considers_static_covariates(self) -> bool:
        return self.model.considers_static_covariates

    @property
    def likelihood(self) -> str:
        return self._likelihood


class ConformalNaiveModel(ConformalModel):
    def __init__(
        self,
        model: GlobalForecastingModel,
        quantiles: List[float],
        symmetric: bool = True,
        cal_length: Optional[int] = None,
        num_samples: int = 500,
        random_state: Optional[int] = None,
        stride_cal: bool = False,
    ):
        """Naive Conformal Prediction Model.

        A probabilistic model that adds calibrated intervals around the median forecast from a pre-trained
        global forecasting model. It does not have to be trained and can generated calibrated forecasts
        directly using the underlying trained forecasting model. It supports two symmetry modes:

        - `symmetric=True`:
            - The lower and upper interval bounds are calibrated with the same magnitude.
            - Non-conformity scores: uses metric `ae()` (see absolute error :func:`~darts.metrics.metrics.ae`) to
              compute the non-conformity scores on the calibration set.
        - `symmetric=False`
            - The lower and upper interval bounds are calibrated separately.
            - Non-conformity scores: uses metric `err()` (see error :func:`~darts.metrics.metrics.err`) to compute the
              non-conformity scores on the calibration set for the upper bounds, an `-err()` for the lower bounds.

        Since it is a probabilistic model, you can generate forecasts in two ways (when calling `predict()`,
        `historical_forecasts()`, ...):

        - Predict the calibrated quantile intervals directly: Pass parameters `predict_likelihood_parameters=True`, and
          `num_samples=1` to the forecast method.
        - Predict stochastic samples from the calibrated quantile intervals: Pass parameters
          `predict_likelihood_parameters=False`, and `num_samples>>1` to the forecast method.

        Conformal models can be applied to any of Darts' global forecasting model, as long as the model has been
        fitted before. In general the workflow of the models to produce one calibrated forecast/prediction is as
        follows:

        - Extract a calibration set: The number of calibration examples from the most recent past to use for one
          conformal prediction can be defined at model creation with parameter `cal_length`. If `stride_cal` is `True`,
          then the same `stride` from the forecasting methods is applied to the calibration set, and more calibration
          examples are required (`cal_length * stride` historical forecasts that were generated with `stride=1`).
          To make your life simpler, we support two modes:
            - Automatic extraction of the calibration set from the past of your input series (`series`,
              `past_covariates`, ...). This is the default mode and our predict/forecasting/backtest/.... API is
              identical to any other forecasting model
            - Supply a fixed calibration set with parameters `cal_series`, `cal_past_covariates`, ... .
        - Generate historical forecasts on the calibration set (using the forecasting model)
        - Compute the errors/non-conformity scores (as defined above) on these historical forecasts
        - Compute the quantile values from the errors / non-conformity scores (using our desired quantiles set at model
          creation with parameter `quantiles`).
        - Compute the conformal prediction: Add the calibrated intervals to the forecasting model's predictions.

        Some notes:

        - When computing historical_forecasts(), backtest(), residuals(), ... the above is applied for each forecast
          (the forecasting model's historical forecasts are only generated once for efficiency).
        - For multi-horizon forecasts, the above is applied for each step in the horizon separately

        Parameters
        ----------
        model
            A pre-trained global forecasting model.
        quantiles
            A list of quantiles centered around the median `q=0.5` to use. For example quantiles
            [0.1, 0.2, 0.5, 0.8 0.9] correspond to two intervals with (0.9 - 0.1) = 80%, and (0.8 - 0.2) 60% coverage
            around the median (model forecast).
        symmetric
            Whether to use symmetric non-conformity scores. If `True`, uses metric `ae()` (see
            :func:`~darts.metrics.metrics.ae`) to compute the non-conformity scores. If `False`, uses metric `-err()`
            (see :func:`~darts.metrics.metrics.err`) for the lower, and `err()` for the upper quantile interval bound.
        cal_length
            The number of past forecast residuals/errors to consider as calibration input for each conformal forecast.
            If `None`, considers all past residuals.
        num_samples
            Number of times a prediction is sampled from the underlying `model` if it is probabilistic. Uses `1` for
            deterministic models. This is different to the `num_samples` produced by the conformal model which can be
            set in downstream forecasting tasks.
        random_state
            Control the randomness of probabilistic conformal forecasts (sample generation) across different runs.
        stride_cal
            Whether to apply the same historical forecast `stride` to the non-conformity scores of the calibration set.
        """
        super().__init__(
            model=model,
            quantiles=quantiles,
            symmetric=symmetric,
            cal_length=cal_length,
            num_samples=num_samples,
            random_state=random_state,
            stride_cal=stride_cal,
        )

    def _calibrate_interval(
        self, residuals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        def q_hat_from_residuals(residuals_):
            # compute quantiles of shape (forecast horizon, n components, n quantile intervals)
            return np.quantile(
                residuals_,
                q=self.interval_range_sym,
                method="higher",
                axis=2,
            ).transpose((1, 2, 0))

        # residuals shape (horizon, n components, n past forecasts)
        if self.symmetric:
            # symmetric (from metric `ae()`)
            q_hat = q_hat_from_residuals(residuals)
            return -q_hat, q_hat[:, :, ::-1]
        else:
            # asymmetric (from metric `err()`)
            q_hat = q_hat_from_residuals(
                np.concatenate([-residuals, residuals], axis=1)
            )
            n_comps = residuals.shape[1]
            return -q_hat[:, :n_comps, :], q_hat[:, n_comps:, ::-1]

    def _apply_interval(self, pred: np.ndarray, q_hat: Tuple[np.ndarray, np.ndarray]):
        # convert stochastic predictions to median
        if pred.shape[2] != 1:
            pred = np.expand_dims(np.quantile(pred, 0.5, axis=2), -1)
        # shape (forecast horizon, n components, n quantiles)
        pred = np.concatenate([pred + q_hat[0], pred, pred + q_hat[1]], axis=2)
        # -> (forecast horizon, n components * n quantiles)
        return pred.reshape(len(pred), -1)

    @property
    def _residuals_metric(self) -> Tuple[METRIC_TYPE, Optional[dict]]:
        return (metrics.ae if self.symmetric else metrics.err), None


class ConformalQRModel(ConformalModel):
    def __init__(
        self,
        model: GlobalForecastingModel,
        quantiles: List[float],
        symmetric: bool = True,
        cal_length: Optional[int] = None,
        num_samples: int = 500,
        random_state: Optional[int] = None,
        stride_cal: bool = False,
    ):
        """Conformalized Quantile Regression Model.

        A probabilistic model that calibrates the quantile predictions from a pre-trained probabilistic global
        forecasting model. It does not have to be trained and can generated calibrated forecasts
        directly using the underlying trained forecasting model. It supports two symmetry modes:

        - `symmetric=True`:
            - The lower and upper quantile predictions are calibrated with the same magnitude.
            - Non-conformity scores: uses metric `incs_qr(symmetric=True)` (see Non-Conformity Score for Quantile
              Regression :func:`~darts.metrics.metrics.incs_qr`) to compute the non-conformity scores on the calibration
              set.
        - `symmetric=False`
            - The lower and upper quantile predictions are calibrated separately.
            - Non-conformity scores: uses metric `incs_qr(symmetric=False)` (see Non-Conformity Score for Quantile
              Regression :func:`~darts.metrics.metrics.incs_qr`) to compute the non-conformity scores for the upper and
              lower bound separately.

        Since it is a probabilistic model, you can generate forecasts in two ways (when calling `predict()`,
        `historical_forecasts()`, ...):

        - Predict the calibrated quantile intervals directly: Pass parameters `predict_likelihood_parameters=True`, and
          `num_samples=1` to the forecast method.
        - Predict stochastic samples from the calibrated quantile intervals: Pass parameters
          `predict_likelihood_parameters=False`, and `num_samples>>1` to the forecast method.

        Conformal models can be applied to any of Darts' global forecasting model, as long as the model has been
        fitted before. In general the workflow of the models to produce one calibrated forecast/prediction is as
        follows:

        - Extract a calibration set: The number of calibration examples from the most recent past to use for one
          conformal prediction can be defined at model creation with parameter `cal_length`. If `stride_cal` is `True`,
          then the same `stride` from the forecasting methods is applied to the calibration set, and more calibration
          examples are required (`cal_length * stride` historical forecasts that were generated with `stride=1`).
          To make your life simpler, we support two modes:
            - Automatic extraction of the calibration set from the past of your input series (`series`,
              `past_covariates`, ...). This is the default mode and our predict/forecasting/backtest/.... API is
              identical to any other forecasting model
            - Supply a fixed calibration set with parameters `cal_series`, `cal_past_covariates`, ... .
        - Generate historical forecasts (quantile predictions) on the calibration set (using the forecasting model)
        - Compute the errors/non-conformity scores (as defined above) on these historical quantile predictions
        - Compute the quantile values from the errors / non-conformity scores (using our desired quantiles set at model
          creation with parameter `quantiles`).
        - Compute the conformal prediction: Calibrate the predicted quantiles from the forecasting model's predictions.

        Some notes:

        - When computing historical_forecasts(), backtest(), residuals(), ... the above is applied for each forecast
          (the forecasting model's historical forecasts are only generated once for efficiency).
        - For multi-horizon forecasts, the above is applied for each step in the horizon separately

        Parameters
        ----------
        model
            A pre-trained probabilistic global forecasting model using a `likelihood`.
        quantiles
            A list of quantiles centered around the median `q=0.5` to use. For example quantiles
            [0.1, 0.2, 0.5, 0.8 0.9] correspond to two intervals with (0.9 - 0.1) = 80%, and (0.8 - 0.2) 60% coverage
            around the median (model forecast).
        symmetric
            Whether to use symmetric non-conformity scores. If `True`, uses symmetric metric
            `incs_qr(..., symmetric=True)` (see :func:`~darts.metrics.metrics.incs_qr`) to compute the non-conformity
            scores. If `False`, uses asymmetric metric `incs_qr(..., symmetric=False)` with individual scores for the
            lower- and upper quantile interval bounds.
        cal_length
            The number of past forecast residuals/errors to consider as calibration input for each conformal forecast.
            If `None`, considers all past residuals.
        num_samples
            Number of times a prediction is sampled from the underlying `model` if it is probabilistic. Uses `1` for
            deterministic models. This is different to the `num_samples` produced by the conformal model which can be
            set in downstream forecasting tasks.
        random_state
            Control the randomness of probabilistic conformal forecasts (sample generation) across different runs.
        stride_cal
            Whether to apply the same historical forecast `stride` to the non-conformity scores of the calibration set.
        """
        if not model.supports_probabilistic_prediction:
            raise_log(
                ValueError(
                    "`model` must must support probabilistic forecasting. Consider using a `likelihood` at "
                    "forecasting model creation, or use another conformal model."
                ),
                logger=logger,
            )
        super().__init__(
            model=model,
            quantiles=quantiles,
            symmetric=symmetric,
            cal_length=cal_length,
            num_samples=num_samples,
            random_state=random_state,
            stride_cal=stride_cal,
        )

    def _calibrate_interval(
        self, residuals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_comps = residuals.shape[1] // (
            len(self.interval_range) * (1 + int(not self.symmetric))
        )
        n_intervals = len(self.interval_range)

        def q_hat_from_residuals(residuals_):
            # TODO: is there a more efficient way?
            # compute quantiles with shape (horizon, n components, n quantile intervals)
            # over all past residuals
            q_hat_tmp = np.quantile(
                residuals_, q=self.interval_range_sym, method="higher", axis=2
            ).transpose((1, 2, 0))
            q_hat_ = np.empty((len(residuals_), n_comps, n_intervals))
            for i in range(n_intervals):
                for c in range(n_comps):
                    q_hat_[:, c, i] = q_hat_tmp[:, i + c * n_intervals, i]
            return q_hat_

        if self.symmetric:
            # symmetric has one nc-score per interval (from metric `incs_qr(symmetric=True)`)
            # residuals shape (horizon, n components * n intervals, n past forecasts)
            q_hat = q_hat_from_residuals(residuals)
            return -q_hat, q_hat[:, :, ::-1]
        else:
            # asymmetric has two nc-score per interval (for lower and upper quantiles, from metric
            # `incs_qe(symmetric=False)`)
            # lower and upper residuals are concatenated along axis=1;
            # residuals shape (horizon, n components * n intervals * 2, n past forecasts)
            half_idx = residuals.shape[1] // 2
            q_hat_lo = q_hat_from_residuals(residuals[:, :half_idx])
            q_hat_hi = q_hat_from_residuals(residuals[:, half_idx:])
            return -q_hat_lo, q_hat_hi[:, :, ::-1]

    def _apply_interval(self, pred: np.ndarray, q_hat: Tuple[np.ndarray, np.ndarray]):
        # get quantile predictions with shape (n times, n components, n quantiles)
        pred = np.quantile(pred, self.quantiles, axis=2).transpose((1, 2, 0))
        # shape (forecast horizon, n components, n quantiles)
        pred = np.concatenate(
            [
                pred[:, :, : self.idx_median] + q_hat[0],  # lower quantiles
                pred[:, :, self.idx_median : self.idx_median + 1],  # model forecast
                pred[:, :, self.idx_median + 1 :] + q_hat[1],  # upper quantiles
            ],
            axis=2,
        )
        # -> (forecast horizon, n components * n quantiles)
        return pred.reshape(len(pred), -1)

    @property
    def _residuals_metric(self) -> Tuple[METRIC_TYPE, Optional[dict]]:
        return metrics.incs_qr, {
            "q_interval": self.q_interval,
            "symmetric": self.symmetric,
        }
