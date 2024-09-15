import os
import re
from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd

import darts.metrics
from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.utils import _with_sanity_checks
from darts.utils.historical_forecasts.utils import _historical_forecasts_start_warnings
from darts.utils.timeseries_generation import _build_forecast_series
from darts.utils.ts_utils import (
    SeriesType,
    get_series_seq_type,
    get_single_series,
    series2seq,
)
from darts.utils.utils import generate_index, n_steps_between

logger = get_logger(__name__)


def cqr_score_sym(row, quantile_lo_col, quantile_hi_col):
    return (
        [None, None]
        if row[quantile_lo_col] is None or row[quantile_hi_col] is None
        else [
            max(row[quantile_lo_col] - row["y"], row["y"] - row[quantile_hi_col]),
            0
            if row[quantile_lo_col] - row["y"] > row["y"] - row[quantile_hi_col]
            else 1,
        ]
    )


def cqr_score_asym(row, quantile_lo_col, quantile_hi_col):
    return (
        [None, None]
        if row[quantile_lo_col] is None or row[quantile_hi_col] is None
        else [
            row[quantile_lo_col] - row["y"],
            row["y"] - row[quantile_hi_col],
            0
            if row[quantile_lo_col] - row["y"] > row["y"] - row[quantile_hi_col]
            else 1,
        ]
    )


class ConformalModel(GlobalForecastingModel, ABC):
    def __init__(
        self,
        model,
        alpha: Union[float, Tuple[float, float]],
        quantiles: Optional[List[float]] = None,
    ):
        """Base Conformal Prediction Model

        Parameters
        ----------
        model
            The forecasting model.
        alpha
            Significance level of the prediction interval, float if coverage error spread arbitrarily over left and
            right tails, tuple of two floats for different coverage error over left and right tails respectively
        quantiles
            Optionally, a list of quantiles from the quantile regression `model` to use.
        """
        if not isinstance(model, GlobalForecastingModel) or not model._fit_called:
            raise_log(
                ValueError("`model` must be a pre-trained `GlobalForecastingModel`."),
                logger=logger,
            )
        super().__init__(add_encoders=None)

        if isinstance(alpha, float):
            self.symmetrical = True
            self.q_hats = pd.DataFrame(columns=["q_hat_sym"])
        else:
            self.symmetrical = False
            self.alpha_lo, self.alpha_hi = alpha
            self.q_hats = pd.DataFrame(columns=["q_hat_lo", "q_hat_hi"])

        self.model = model
        self.noncon_scores = dict()
        self.alpha = alpha
        self.quantiles = quantiles
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
            num_samples=num_samples,
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
            num_samples=num_samples,
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
        # for step_number in range(1, self.n_forecasts + 1):
        #     # conformalize
        #     noncon_scores = self._get_nonconformity_scores(df_cal, step_number)
        #     q_hat = self._get_q_hat(df_cal, noncon_scores)
        #     y_hat_col = f"yhat{step_number}"
        #     y_hat_lo_col = f"{y_hat_col} {min(self.quantiles) * 100}%"
        #     y_hat_hi_col = f"{y_hat_col} {max(self.quantiles) * 100}%"
        #     if self.method == "naive" and self.symmetrical:
        #         q_hat_sym = q_hat["q_hat_sym"]
        #         df[y_hat_lo_col] = df[y_hat_col] - q_hat_sym
        #         df[y_hat_hi_col] = df[y_hat_col] + q_hat_sym
        #     elif self.method == "cqr" and self.symmetrical:
        #         q_hat_sym = q_hat["q_hat_sym"]
        #         df[y_hat_lo_col] = df[y_hat_lo_col] - q_hat_sym
        #         df[y_hat_hi_col] = df[y_hat_hi_col] + q_hat_sym
        #     elif self.method == "cqr" and not self.symmetrical:
        #         q_hat_lo = q_hat["q_hat_lo"]
        #         q_hat_hi = q_hat["q_hat_hi"]
        #         df[y_hat_lo_col] = df[y_hat_lo_col] - q_hat_lo
        #         df[y_hat_hi_col] = df[y_hat_hi_col] + q_hat_hi
        #     else:
        #         raise ValueError(
        #             f"Unknown conformal prediction method '{self.method}'. Please input either 'naive' or 'cqr'."
        #         )
        #     if step_number == 1:
        #         # save nonconformity scores of the first timestep
        #         self.noncon_scores = noncon_scores
        #
        #     # append the dictionary of q_hats to the dataframe based on the keys of the dictionary
        #     q_hat_df = pd.DataFrame([q_hat])
        #     self.q_hats = pd.concat([self.q_hats, q_hat_df], ignore_index=True)
        #
        #     # if show_all_PI is True, add the quantile regression prediction intervals
        #     if show_all_PI:
        #         df_quantiles = [col for col in df_qr.columns if "%" in col and f"yhat{step_number}" in col]
        #         df_add = df_qr[df_quantiles]
        #
        #         if self.method == "naive":
        #             cp_lo_col = f"yhat{step_number} - qhat{step_number}"  # e.g. yhat1 - qhat1
        #             cp_hi_col = f"yhat{step_number} + qhat{step_number}"  # e.g. yhat1 + qhat1
        #             df.rename(columns={y_hat_lo_col: cp_lo_col, y_hat_hi_col: cp_hi_col}, inplace=True)
        #         elif self.method == "cqr":
        #             qr_lo_col = (
        #                 f"yhat{step_number} {max(self.quantiles) * 100}% - qhat{step_number}"  #e.g. yhat1 95% - qhat1
        #             )
        #             qr_hi_col = (
        #                 f"yhat{step_number} {min(self.quantiles) * 100}% + qhat{step_number}"  #e.g. yhat1 5% + qhat1
        #             )
        #             df.rename(columns={y_hat_lo_col: qr_lo_col, y_hat_hi_col: qr_hi_col}, inplace=True)
        #
        #         df = pd.concat([df, df_add], axis=1, ignore_index=False)
        #
        # return df

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
            num_samples=num_samples,
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
                num_samples=num_samples,
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
            train_length=train_length,
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

    def _calibrate_forecasts(
        self,
        series: Sequence[TimeSeries],
        forecasts: Union[Sequence[Sequence[TimeSeries]], Sequence[TimeSeries]],
        cal_series: Optional[Sequence[TimeSeries]] = None,
        cal_forecasts: Optional[
            Union[Sequence[Sequence[TimeSeries]], Sequence[TimeSeries]]
        ] = None,
        train_length: Optional[int] = None,
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
        # - support for different CP algorithms
        residuals = self.model.residuals(
            series=series if cal_series is None else cal_series,
            historical_forecasts=forecasts if cal_series is None else cal_forecasts,
            overlap_end=overlap_end if cal_series is None else True,
            last_points_only=last_points_only,
            verbose=verbose,
            show_warnings=show_warnings,
            values_only=True,
            metric=self._residuals_metric,
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
            # at least one or `train_length` examples
            min_n_cal = train_length or 1
            # `last_points_only=False` requires additional examples to use most recent information
            # from all steps in the horizon
            if not last_points_only:
                min_n_cal += forecast_horizon - 1

            # determine first forecast index for conformal prediction
            if cal_series is None:
                # we need at least one residual per point in the horizon prior to the first conformal forecast
                first_idx_train = forecast_horizon + self.output_chunk_shift
                # plus some additional examples based on `train_length`
                if train_length is not None:
                    first_idx_train += train_length - 1
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
            assert not np.isnan(res).any().any()

            # get the last forecast index based on the residual examples
            if cal_series is None:
                last_fc_idx = res.shape[2] + (
                    forecast_horizon + self.output_chunk_shift
                )
            else:
                last_fc_idx = len(s_hfcs)

            if last_fc_idx > len(s_hfcs):
                raise_log(ValueError("blabla"), logger=logger)

            q_hat = None
            if cal_series is not None:
                if train_length is not None:
                    res = res[:, :, -train_length:]
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
                    # first residual index is shifted back by the horizon to get `train_length` points for
                    # the last point in the horizon
                    cal_start = (
                        cal_end - train_length if train_length is not None else None
                    )

                    cal_res = res[:, :, cal_start:cal_end]

                    # TODO: remove checks
                    len_exp = cal_end - (cal_start or 0)
                    if cal_res.shape[2] != len_exp:
                        raise_log(ValueError("Too short cal"), logger=logger)

                    q_hat_ = self._calibrate_interval(cal_res)
                else:
                    # with a calibration set, use a constant q_hat
                    q_hat_ = q_hat
                return self._apply_interval(pred_vals_, q_hat_)

            # historical conformal prediction
            if last_points_only:
                for idx, pred_vals in enumerate(
                    s_hfcs.values(copy=False)[first_fc_idx:last_fc_idx:stride]
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
                        name=series_.time_index.name,
                    ),
                    with_static_covs=False,
                    with_hierarchy=False,
                )
                cp_hfcs.append(cp_preds)
            else:
                for idx, pred in enumerate(s_hfcs[first_fc_idx:last_fc_idx:stride]):
                    pred_vals = pred.values(copy=False)
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
        # TODO: Use new save/load logic from EnsembleModel
        model_name = self.__class__.__name__
        raise_log(
            NotImplementedError(
                f"`{model_name}` does not support saving / loading. Instead, "
                f"save the underlying forecasting model `{self.model.__class__.__name__}` using its dedicated "
                f"save / load functionality, and create a new `{model_name}` with it.",
            ),
            logger=logger,
        )

    @abstractmethod
    def _calibrate_interval(
        self, residuals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the upper and lower calibrated forecast intervals based on residuals."""

    @staticmethod
    def _apply_interval(pred, q_hat):
        """Applies the calibrated interval to the predicted values. Returns an array with 3 predicted columns
        (lower bound, model forecast, upper bound) per component.

        E.g. output is `(target1_cq_low, target1_pred, target1_cq_high, target2_cq_low, ...)`
        """
        n_comps = pred.shape[1]
        pred = np.concatenate([pred + q_hat[0], pred, pred + q_hat[1]], axis=1)
        if n_comps == 1:
            return pred

        n_cal_comps = 3
        # pre-compute axes swap (source and destination) for applying calibration intervals
        axes_src = [i for i in range(n_comps * n_cal_comps)]
        axes_dst = []
        for i in range(n_comps):
            axes_dst += axes_src[i::n_comps]
        return pred[:, axes_dst]

    @property
    @abstractmethod
    def _residuals_metric(self):
        """Gives the "per time step" metric used to compute residuals."""

    def _get_nonconformity_scores(self, df_cal: pd.DataFrame, step_number: int) -> dict:
        """Get the nonconformity scores using the given conformal prediction technique.

        Parameters
        ----------
            df_cal : pd.DataFrame
                calibration dataframe
            step_number : int
                i-th step ahead forecast

            Returns
            -------
                Dict[str, np.ndarray]
                    dictionary with one entry (symmetrical) or two entries (asymmetrical) of nonconformity scores

        """
        y_hat_col = f"yhat{step_number}"
        if self.method == "cqr":
            # CQR nonconformity scoring function
            quantile_lo = str(min(self.quantiles) * 100)
            quantile_hi = str(max(self.quantiles) * 100)
            quantile_lo_col = f"{y_hat_col} {quantile_lo}%"
            quantile_hi_col = f"{y_hat_col} {quantile_hi}%"
            if self.symmetrical:
                scores_df = df_cal.apply(
                    cqr_score_sym,
                    axis=1,
                    result_type="expand",
                    quantile_lo_col=quantile_lo_col,
                    quantile_hi_col=quantile_hi_col,
                )
                scores_df.columns = ["scores", "arg"]
                noncon_scores = scores_df["scores"].values
            else:  # asymmetrical intervals
                scores_df = df_cal.apply(
                    cqr_score_asym,
                    axis=1,
                    result_type="expand",
                    quantile_lo_col=quantile_lo_col,
                    quantile_hi_col=quantile_hi_col,
                )
                scores_df.columns = ["scores_lo", "scores_hi", "arg"]
                noncon_scores_lo = scores_df["scores_lo"].values
                noncon_scores_hi = scores_df["scores_hi"].values
                # Remove NaN values
                noncon_scores_lo: Any = noncon_scores_lo[~pd.isnull(noncon_scores_lo)]
                noncon_scores_hi: Any = noncon_scores_hi[~pd.isnull(noncon_scores_hi)]
                # Sort
                noncon_scores_lo.sort()
                noncon_scores_hi.sort()
                # return dict of nonconformity scores
                return {
                    "noncon_scores_hi": noncon_scores_lo,
                    "noncon_scores_lo": noncon_scores_hi,
                }
        else:  # self.method == "naive"
            # Naive nonconformity scoring function
            noncon_scores = abs(df_cal["y"] - df_cal[y_hat_col]).values
        # Remove NaN values
        noncon_scores: Any = noncon_scores[~pd.isnull(noncon_scores)]
        # Sort
        noncon_scores.sort()

        return {"noncon_scores": noncon_scores}

    def _get_q_hat(self, noncon_scores: dict) -> dict:
        """Get the q_hat that is derived from the nonconformity scores.

        Parameters
        ----------
            noncon_scores : dict
                dictionary with one entry (symmetrical) or two entries (asymmetrical) of nonconformity scores

            Returns
            -------
                Dict[str, float]
                    upper and lower q_hat value, or the one-sided prediction interval width

        """
        # Get the q-hat index and value
        if self.method == "cqr" and self.symmetrical is False:
            noncon_scores_lo = noncon_scores["noncon_scores_lo"]
            noncon_scores_hi = noncon_scores["noncon_scores_hi"]
            q_hat_idx_lo = int(len(noncon_scores_lo) * self.alpha_lo)
            q_hat_idx_hi = int(len(noncon_scores_hi) * self.alpha_hi)
            q_hat_lo = noncon_scores_lo[-q_hat_idx_lo]
            q_hat_hi = noncon_scores_hi[-q_hat_idx_hi]
            return {"q_hat_lo": q_hat_lo, "q_hat_hi": q_hat_hi}
        else:
            noncon_scores = noncon_scores["noncon_scores"]
            q_hat_idx = int(len(noncon_scores) * self.alpha)
            q_hat = noncon_scores[-q_hat_idx]
            return {"q_hat_sym": q_hat}

    def _cp_component_names(self, input_series) -> List[str]:
        return [
            f"{tgt_name}{param_n}"
            for tgt_name in input_series.components
            for param_n in ["_cq_lo", "", "_cq_hi"]
        ]

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


def uncertainty_evaluate(df_forecast: pd.DataFrame) -> pd.DataFrame:
    """Evaluate conformal prediction on test dataframe.

    Parameters
    ----------
        df_forecast : pd.DataFrame
            forecast dataframe with the conformal prediction intervals

    Returns
    -------
        pd.DataFrame
            table containing evaluation metrics such as interval_width and miscoverage_rate
    """
    # Remove beginning rows used as lagged regressors (if any), or future dataframes without y-values
    # therefore, this ensures that all forecast rows for evaluation contains both y and y-hat
    df_forecast_eval = df_forecast.dropna(subset=["y", "yhat1"]).reset_index(drop=True)

    # Get evaluation params
    df_eval = pd.DataFrame()
    cols = df_forecast_eval.columns
    yhat_cols = [col for col in cols if "%" in col]
    n_forecasts = int(re.search("yhat(\\d+)", yhat_cols[-1]).group(1))

    # get the highest and lowest quantile percentages
    quantiles = []
    for col in yhat_cols:
        match = re.search(r"\d+\.\d+", col)
        if match:
            quantiles.append(float(match.group()))
    quantiles = sorted(set(quantiles))

    # Begin conformal evaluation steps
    for step_number in range(1, n_forecasts + 1):
        y = df_forecast_eval["y"].values
        # only relevant if show_all_PI is true
        if len([col for col in cols if "qhat" in col]) > 0:
            qhat_cols = [col for col in cols if f"qhat{step_number}" in col]
            yhat_lo = df_forecast_eval[qhat_cols[0]].values
            yhat_hi = df_forecast_eval[qhat_cols[-1]].values
        else:
            yhat_lo = df_forecast_eval[f"yhat{step_number} {quantiles[0]}%"].values
            yhat_hi = df_forecast_eval[f"yhat{step_number} {quantiles[-1]}%"].values
        interval_width, miscoverage_rate = _get_evaluate_metrics_from_dataset(
            y, yhat_lo, yhat_hi
        )

        # Construct row dataframe with current timestep using its q-hat, interval width, and miscoverage rate
        col_names = ["interval_width", "miscoverage_rate"]
        row = [interval_width, miscoverage_rate]
        df_row = pd.DataFrame(
            [row],
            columns=pd.MultiIndex.from_product([[f"yhat{step_number}"], col_names]),
        )

        # Add row dataframe to overall evaluation dataframe with all forecasted timesteps
        df_eval = pd.concat([df_eval, df_row], axis=1)

    return df_eval


def _get_evaluate_metrics_from_dataset(
    y: np.ndarray, yhat_lo: np.ndarray, yhat_hi: np.ndarray
) -> Tuple[float, float]:
    #     df_forecast_eval: pd.DataFrame,
    #     quantile_lo_col: str,
    #     quantile_hi_col: str,
    # ) -> Tuple[float, float]:
    """Infers evaluation parameters based on the evaluation dataframe columns.

    Parameters
    ----------
        df_forecast_eval : pd.DataFrame
            forecast dataframe with the conformal prediction intervals

    Returns
    -------
        float, float
            conformal prediction evaluation metrics
    """
    # Interval width (efficiency metric)
    quantile_lo_mean = np.mean(yhat_lo)
    quantile_hi_mean = np.mean(yhat_hi)
    interval_width = quantile_hi_mean - quantile_lo_mean

    # Miscoverage rate (validity metric)
    n_covered = np.sum((y >= yhat_lo) & (y <= yhat_hi))
    coverage_rate = n_covered / len(y)
    miscoverage_rate = 1 - coverage_rate

    return interval_width, miscoverage_rate


class ConformalNaiveModel(ConformalModel):
    def __init__(self, model, alpha: Union[float, Tuple[float, float]]):
        if not isinstance(alpha, float):
            raise_log(
                ValueError("`alpha` must be a `float`."),
                logger=logger,
            )
        super().__init__(model=model, alpha=alpha)

    def _calibrate_interval(
        self, residuals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the lower and upper calibrated forecast intervals based on residuals.

        Parameters
        ----------
        residuals
            The residuals are expected to have shape (horizon, n components, n historical forecasts * n samples)
        """
        q_hat = np.quantile(residuals, q=self.alpha, axis=2)
        return -q_hat, q_hat

    @property
    def _residuals_metric(self):
        return darts.metrics.ae