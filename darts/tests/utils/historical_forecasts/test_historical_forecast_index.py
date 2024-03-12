from itertools import product

import pytest

import darts.utils.timeseries_generation as tg
from darts.models import LinearRegressionModel
from darts.utils.historical_forecasts.utils import _get_historical_forecast_train_index

models_reg_cov_cls_kwargs = [
    # target + past covariates
    {"lags": 4, "lags_past_covariates": 6},
    # target + future covariates
    {"lags": 4, "lags_future_covariates": [0, 1]},
    # target + fut cov
    {"lags": 2, "lags_future_covariates": [1, 2]},
    # past cov only
    {"lags_past_covariates": 6},
    # fut cov only
    {"lags_future_covariates": [0, 1]},
    # fut + past cov
    {"lags_past_covariates": 6, "lags_future_covariates": [0, 1]},
    # negative fut + past cov
    {"lags_past_covariates": 2, "lags_future_covariates": [-3, 0]},
    # dual fut + past cov
    {"lags_past_covariates": [-4], "lags_future_covariates": [-3, 2]},
    # all
    {"lags": 3, "lags_past_covariates": 6, "lags_future_covariates": [0, 1]},
]


class MultiTrainingSamplesModel(LinearRegressionModel):
    """Overwritte min_train_samples properties to mimic CatboostModel and LightGBMModel requirements
    without installing the dependencies
    """

    @property
    def min_train_samples(self) -> int:
        return 2


class TestHistoricalForecastUtils:

    ts_len = 20
    ts_tg = tg.linear_timeseries(length=ts_len, start_value=1, end_value=ts_len)
    ts_pc = tg.linear_timeseries(
        length=ts_len, start_value=ts_len + 1, end_value=2 * ts_len
    )
    ts_fc = tg.linear_timeseries(
        length=ts_len, start_value=2 * ts_len + 1, end_value=3 * ts_len
    )

    @pytest.mark.parametrize(
        "model_config",
        product(
            [LinearRegressionModel, MultiTrainingSamplesModel],
            models_reg_cov_cls_kwargs,
            [1, 5],
            [0, 1, 3],
        ),
    )
    def test_historical_forecast_train_index(self, model_config):
        """Verify that `_get_historical_forecast_train_index()` return the expected boundaries."""
        model_cls, lags_kwargs, ocl, ocs = model_config
        model = model_cls(
            **lags_kwargs, output_chunk_length=ocl, output_chunk_shift=ocs
        )
        (
            min_target_lag,
            max_target_lag,
            min_past_cov_lag,
            max_past_cov_lag,
            min_future_cov_lag,
            max_future_cov_lag,
            output_chunk_shift,
        ) = model.extreme_lags
        # all the series have the same start, the (co)variate with the most extreme lag dictates the end boundary
        start_index = (
            self.ts_tg.start_time()
            + (
                model.output_chunk_length  # length require to extract the labels
                - min(
                    min_target_lag if min_target_lag is not None else 0,
                    min_past_cov_lag if min_past_cov_lag is not None else 0,
                    (
                        min_future_cov_lag
                        if min_future_cov_lag is not None and min_future_cov_lag < 0
                        else 0
                    ),
                )  # length required to extract the most extreme lag in the past
                + (
                    model.min_train_samples - 1
                )  # length required to extract the additional(s) training sample(s)
            )
            * self.ts_tg.freq
        )

        # target and future covariate have the same end, the fut cov positive lags dictate the end boundary
        end_index = (
            self.ts_tg.end_time()
            + (
                -(
                    max_future_cov_lag + 1
                    if max_future_cov_lag is not None and max_future_cov_lag >= 0
                    else 0
                )  # fut cov lags look into the future
                + 1  # last timestep is included
            )
            * self.ts_tg.freq
        )

        # no auto-regression, overlap_end = True
        forecast_horizon = model.output_chunk_length
        index = _get_historical_forecast_train_index(
            model,
            series=self.ts_tg,
            past_covariates=self.ts_pc if "past" in model.lags else None,
            future_covariates=self.ts_fc if "future" in model.lags else None,
            series_idx=0,
            forecast_horizon=forecast_horizon,
            overlap_end=True,
        )
        assert (
            index[0] == start_index
        ), f"Wrong start boundary; expected {start_index}, received {index[0]}"
        assert (
            index[1] == end_index
        ), f"Wrong end boundary; expected {end_index}, received {index[1]}"

        # no auto-regression, overlap_end = False
        # if index was going beyond series end when overlap_end=True, the end will be shifted
        # so that the last forecasted timestep is aligned with the end of the target series
        last_forecast_shift = (
            forecast_horizon + model.output_chunk_shift - 1
        )  # - 1 because end boundary is inclusive
        if end_index + last_forecast_shift * self.ts_tg.freq > self.ts_tg.end_time():
            end_index = self.ts_tg.end_time() - last_forecast_shift * self.ts_tg.freq
        index = _get_historical_forecast_train_index(
            model,
            series=self.ts_tg,
            past_covariates=self.ts_pc if "past" in model.lags else None,
            future_covariates=self.ts_fc if "future" in model.lags else None,
            series_idx=0,
            forecast_horizon=forecast_horizon,
            overlap_end=False,
        )
        assert (
            index[0] == start_index
        ), f"Wrong start boundary; expected {start_index}, received {index[0]}"
        assert (
            index[1] == end_index
        ), f"Wrong end boundary; expected {end_index}, received {index[1]}"

    def test_historical_forecast_predict_index(self):
        return

    def test_reconcialiate_historical_index(self):
        return

    def test_adjust_historical_index(self):
        return
