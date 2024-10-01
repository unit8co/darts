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
from darts.utils import _with_sanity_checks
from darts.utils.historical_forecasts.utils import _historical_forecasts_start_warnings
from darts.utils.timeseries_generation import _build_forecast_series
from darts.utils.ts_utils import (
    SeriesType,
    get_series_seq_type,
    get_single_series,
    series2seq,
)
from darts.utils.utils import (
    _check_quantiles,
    generate_index,
    likelihood_component_names,
    n_steps_between,
    quantile_names,
)

if TORCH_AVAILABLE:
    from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
else:
    TorchForecastingModel = None

logger = get_logger(__name__)


class ConformalModel(GlobalForecastingModel, ABC):
    def __init__(
        self,
        model: GlobalForecastingModel,
        quantiles: List[float],
        symmetric: bool = True,
        cal_length: Optional[int] = None,
        num_samples: int = 500,
    ):
        """Base Conformal Prediction Model.

        Parameters
        ----------
        model
            A pre-trained global forecasting model.
        quantiles
            A list of quantiles centered around the median `q=0.5` to use. For example quantiles
            [0.1, 0.5, 0.9] correspond to a (0.9 - 0.1) = 80% coverage interval around the median (model forecast).
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
        """
        if not isinstance(model, GlobalForecastingModel) or not model._fit_called:
            raise_log(
                ValueError("`model` must be a pre-trained `GlobalForecastingModel`."),
                logger=logger,
            )
        _check_quantiles(quantiles)
        super().__init__(add_encoders=None)

        # quantiles and interval setup
        self.quantiles = quantiles
        self.idx_median = quantiles.index(0.5)
        self.interval_bounds = [
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
        self.num_samples = num_samples if model.supports_probabilistic_prediction else 1
        self._likelihood = "quantile"
        self._fit_called = True

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> "ConformalModel":
        # does not have to be trained
        return self

    def predict(
        self,
        n: int,
        series: Union[TimeSeries, Sequence[TimeSeries]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        verbose: bool = False,
        predict_likelihood_parameters: bool = False,
        show_warnings: bool = True,
        cal_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
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
            predict_likelihood_parameters=predict_likelihood_parameters,
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
            predict_likelihood_parameters=predict_likelihood_parameters,
        )
        cal_preds = self._calibrate_forecasts(
            series=series,
            forecasts=preds,
            cal_series=cal_series,
            cal_forecasts=cal_hfcs,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=False,
            verbose=verbose,
            show_warnings=show_warnings,
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
        num_samples: int = 1,
        train_length: Optional[int] = None,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        start_format: Literal["position", "value"] = "value",
        forecast_horizon: int = 1,
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
        cal_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        cal_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> Union[TimeSeries, List[TimeSeries], List[List[TimeSeries]]]:
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
            predict_likelihood_parameters=predict_likelihood_parameters,
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
                predict_likelihood_parameters=predict_likelihood_parameters,
                enable_optimization=enable_optimization,
                fit_kwargs=fit_kwargs,
                predict_kwargs=predict_kwargs,
            )
        calibrated_forecasts = self._calibrate_forecasts(
            series=series,
            forecasts=hfcs,
            cal_series=cal_series,
            cal_forecasts=cal_hfcs,
            start=start,
            start_format=start_format,
            forecast_horizon=forecast_horizon,
            stride=stride,
            overlap_end=overlap_end,
            last_points_only=last_points_only,
            verbose=verbose,
            show_warnings=show_warnings,
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
        historical_forecasts: Optional[
            Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]
        ] = None,
        num_samples: int = 1,
        train_length: Optional[int] = None,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        start_format: Literal["position", "value"] = "value",
        forecast_horizon: int = 1,
        stride: int = 1,
        retrain: Union[bool, int, Callable[..., bool]] = True,
        overlap_end: bool = False,
        last_points_only: bool = False,
        metric: Union[METRIC_TYPE, List[METRIC_TYPE]] = metrics.miw,
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
        # make user's life easier by adding quantile intervals, or quantiles directly
        if metric_kwargs is None:
            metric = [metric] if not isinstance(metric, list) else metric
            metric_kwargs = []
            for metric_ in metric:
                if metric_ in metrics.ALL_METRICS:
                    if metric_ in metrics.Q_INTERVAL_METRICS:
                        metric_kwargs.append({"q_interval": self.interval_bounds})
                    elif metric_ not in metrics.NON_Q_METRICS:
                        metric_kwargs.append({"q": self.quantiles})
                    else:
                        metric_kwargs.append({})
                else:
                    metric_kwargs.append({})
        return super().backtest(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            historical_forecasts=historical_forecasts,
            num_samples=num_samples,
            train_length=train_length,
            start=start,
            start_format=start_format,
            forecast_horizon=forecast_horizon,
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
        historical_forecasts: Optional[
            Union[TimeSeries, Sequence[TimeSeries], Sequence[Sequence[TimeSeries]]]
        ] = None,
        num_samples: int = 1,
        train_length: Optional[int] = None,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        start_format: Literal["position", "value"] = "value",
        forecast_horizon: int = 1,
        stride: int = 1,
        retrain: Union[bool, int, Callable[..., bool]] = True,
        overlap_end: bool = False,
        last_points_only: bool = True,
        metric: METRIC_TYPE = metrics.iw,
        verbose: bool = False,
        show_warnings: bool = True,
        predict_likelihood_parameters: bool = False,
        enable_optimization: bool = True,
        metric_kwargs: Optional[Dict[str, Any]] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None,
        values_only: bool = False,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ) -> Union[TimeSeries, List[TimeSeries], List[List[TimeSeries]]]:
        # make user's life easier by adding quantile intervals, or quantiles directly
        if metric_kwargs is None and metric in metrics.ALL_METRICS:
            if metric in metrics.Q_INTERVAL_METRICS:
                metric_kwargs = {"q_interval": self.interval_bounds}
            elif metric not in metrics.NON_Q_METRICS:
                metric_kwargs = {"q": self.quantiles}
            else:
                metric_kwargs = {}
        return super().residuals(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            historical_forecasts=historical_forecasts,
            num_samples=num_samples,
            train_length=train_length,
            start=start,
            start_format=start_format,
            forecast_horizon=forecast_horizon,
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
            values_only=values_only,
            sample_weight=sample_weight,
        )

    def _calibrate_forecasts(
        self,
        series: Sequence[TimeSeries],
        forecasts: Union[Sequence[Sequence[TimeSeries]], Sequence[TimeSeries]],
        cal_series: Optional[Sequence[TimeSeries]] = None,
        cal_forecasts: Optional[
            Union[Sequence[Sequence[TimeSeries]], Sequence[TimeSeries]]
        ] = None,
        start: Optional[Union[pd.Timestamp, float, int]] = None,
        start_format: Literal["position", "value"] = "value",
        forecast_horizon: int = 1,
        stride: int = 1,
        overlap_end: bool = False,
        last_points_only: bool = True,
        verbose: bool = False,
        show_warnings: bool = True,
    ) -> Union[TimeSeries, List[TimeSeries], List[List[TimeSeries]]]:
        # TODO: add support for:
        # - num_samples
        # - predict_likelihood_parameters
        # - tqdm iterator over series
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

        cp_hfcs = []
        for series_idx, (series_, s_hfcs, res) in enumerate(
            zip(series, forecasts, residuals)
        ):
            cp_preds = []

            # no historical forecasts were generated
            if not s_hfcs:
                cp_hfcs.append(cp_preds)
                continue

            first_hfc = get_single_series(s_hfcs)
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
            # skip solely based on `start`
            first_idx_start = 0
            if start is not None:
                if isinstance(start, pd.Timestamp) or start_format == "value":
                    start_time = start
                else:
                    start_time = series_._time_index[start]

                first_idx_start = n_steps_between(
                    end=start_time,
                    start=first_hfc.start_time(),
                    freq=series_.freq,
                )
                # hfcs have shifted output; skip until end of shift
                first_idx_start += self.output_chunk_shift
                # hfcs only contain last predicted points; skip until end of first forecast
                if last_points_only:
                    first_idx_start += forecast_horizon - 1

                # if start is out of bounds, we ignore it
                last_idx = len(s_hfcs) - 1
                if (
                    first_idx_start < 0
                    or first_idx_start > last_idx
                    or first_idx_start < first_idx_train
                ):
                    first_idx_start = 0
                    if show_warnings:
                        # adjust to actual start point in case of output shift or `last_points_only=True`
                        adjust_idx = (
                            self.output_chunk_shift
                            + int(last_points_only) * (forecast_horizon - 1)
                        ) * series_.freq
                        hfc_predict_index = (
                            s_hfcs[first_idx_train].start_time() - adjust_idx,
                            s_hfcs[last_idx].start_time() - adjust_idx,
                        )
                        _historical_forecasts_start_warnings(
                            idx=series_idx,
                            start=start,
                            start_time_=start_time,
                            historical_forecasts_time_index=hfc_predict_index,
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
                return self._apply_interval(pred_vals_, q_hat_)

            # historical conformal prediction
            # for each forecast, compute calibrated quantile intervals based on past residuals
            if last_points_only:
                for idx, pred_vals in enumerate(
                    s_hfcs.all_values(copy=False)[first_fc_idx:last_fc_idx:stride]
                ):
                    pred_vals = np.expand_dims(pred_vals, 0)
                    cp_pred = conformal_predict(idx, pred_vals)
                    cp_preds.append(cp_pred)
                cp_preds = _build_forecast_series(
                    points_preds=np.concatenate(cp_preds, axis=0),
                    input_series=series_,
                    custom_columns=self._cp_component_names(series_),
                    time_index=generate_index(
                        start=s_hfcs._time_index[first_fc_idx],
                        length=len(cp_preds),
                        freq=series_.freq * stride,
                        name=series_._time_index.name,
                    ),
                    with_static_covs=False,
                    with_hierarchy=False,
                )
                cp_hfcs.append(cp_preds)
            else:
                for idx, pred in enumerate(s_hfcs[first_fc_idx:last_fc_idx:stride]):
                    pred_vals = pred.all_values(copy=False)
                    cp_pred = conformal_predict(idx, pred_vals)
                    cp_pred = _build_forecast_series(
                        points_preds=cp_pred,
                        input_series=series_,
                        custom_columns=self._cp_component_names(series_),
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

    def _cp_component_names(self, input_series) -> List[str]:
        """Gives the component names for generated forecasts."""
        return likelihood_component_names(
            input_series.components, quantile_names(self.quantiles)
        )

    @property
    def output_chunk_length(self) -> Optional[int]:
        return self.model.output_chunk_length

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
        """Whether the model supports a validation set during training."""
        return False

    @property
    def supports_likelihood_parameter_prediction(self) -> bool:
        """EnsembleModel can predict likelihood parameters if all its forecasting models were fitted with the
        same likelihood.
        """
        return True

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return True

    @property
    def uses_past_covariates(self) -> bool:
        """
        Whether the model uses past covariates, once fitted.
        """
        return self.model.uses_past_covariates

    @property
    def uses_future_covariates(self) -> bool:
        """
        Whether the model uses future covariates, once fitted.
        """
        return self.model.uses_future_covariates

    @property
    def uses_static_covariates(self) -> bool:
        """
        Whether the model uses static covariates, once fitted.
        """
        return self.model.uses_static_covariates

    @property
    def considers_static_covariates(self) -> bool:
        """
        Whether the model considers static covariates, if there are any.
        """
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
    ):
        """Naive Conformal Prediction Model.

        Parameters
        ----------
        model
            A pre-trained global forecasting model.
        quantiles
            A list of quantiles centered around the median `q=0.5` to use. For example quantiles
            [0.1, 0.5, 0.9] correspond to a (0.9 - 0.1) = 80% coverage interval around the median (model forecast).
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
        """
        super().__init__(
            model=model,
            quantiles=quantiles,
            symmetric=symmetric,
            cal_length=cal_length,
            num_samples=num_samples,
        )

    def _calibrate_interval(
        self, residuals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.symmetric:
            # shape (forecast horizon, n components, n quantile intervals)
            q_hat = np.quantile(residuals, q=self.interval_range_sym, axis=2).transpose((
                1,
                2,
                0,
            ))
            return -q_hat, q_hat[:, :, ::-1]

        # asymmetric
        n_comps = residuals.shape[1]
        res = np.concatenate([-residuals, residuals], axis=1)
        q_hat = np.quantile(res, q=self.interval_range_sym, axis=2).transpose((1, 2, 0))
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
    ):
        """Conformalized Quantile Regression Model.

        Parameters
        ----------
        model
            A pre-trained probabilistic global forecasting model using a `likelihood`.
        quantiles
            A list of quantiles centered around the median `q=0.5` to use. For example quantiles
            [0.1, 0.5, 0.9] correspond to a (0.9 - 0.1) = 80% coverage interval around the median (model forecast).
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
                residuals_, q=self.interval_range_sym, axis=2
            ).transpose((
                1,
                2,
                0,
            ))
            q_hat_ = np.empty((len(residuals_), n_comps, n_intervals))
            for i in range(n_intervals):
                for c in range(n_comps):
                    q_hat_[:, c, i] = q_hat_tmp[:, i + c * n_intervals, i]
            return q_hat_

        if self.symmetric:
            # symmetric has one nc-score per intervals
            # residuals shape (horizon, n components * n intervals, n past forecasts)
            q_hat = q_hat_from_residuals(residuals)
            return -q_hat, q_hat[:, :, ::-1]
        else:
            # asymmetric has two nc-score per intervals (for lower and upper quantiles)
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
            "q_interval": self.interval_bounds,
            "symmetric": self.symmetric,
        }
