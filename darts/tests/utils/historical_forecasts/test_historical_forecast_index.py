from itertools import product
from typing import Callable

import pytest
from pandas import Timestamp

import darts.utils.timeseries_generation as tg
from darts.models import LinearRegressionModel
from darts.utils.historical_forecasts.utils import (
    _adjust_historical_forecasts_time_index,
    _get_historical_forecast_predict_index,
    _get_historical_forecast_train_index,
    _reconciliate_historical_time_indices,
)
from darts.utils.utils import n_steps_between

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
    def test_historical_forecast_index(self, model_config):
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
            _,
        ) = model.extreme_lags
        # all the series have the same start, the (co)variate with the most extreme lag dictates the end boundary
        start_train_index = (
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
        # similar to train_index, except the model is already fitted and ready for inference
        start_pred_index = (
            self.ts_tg.start_time()
            - min(
                min_target_lag if min_target_lag is not None else 0,
                min_past_cov_lag if min_past_cov_lag is not None else 0,
                (
                    min_future_cov_lag
                    if min_future_cov_lag is not None and min_future_cov_lag < 0
                    else 0
                ),
            )  # length required to extract the most extreme lag in the past
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

        # no auto-regression
        forecast_horizon = model.output_chunk_length
        test_args = {
            "model": model,
            "series": self.ts_tg,
            "past_covariates": self.ts_pc if "past" in model.lags else None,
            "future_covariates": self.ts_fc if "future" in model.lags else None,
            "series_idx": 0,
            "forecast_horizon": forecast_horizon,
        }

        # overlap_end = True
        index_train = _get_historical_forecast_train_index(
            **test_args,
            overlap_end=True,
        )
        assert index_train == (
            start_train_index,
            end_index,
        ), f"Wrong training boundaries; expected {index_train}, received {(start_train_index, end_index)}"

        index_pred = _get_historical_forecast_predict_index(
            **test_args,
            overlap_end=True,
        )
        assert index_pred == (
            start_pred_index,
            end_index,
        ), f"Wrong prediction boundaries; expected {index_pred}, received {(start_pred_index, end_index)}"

        # overlap_end = False
        # if index was going beyond series end when overlap_end=True, the end will be shifted
        # so that the last forecasted timestep is aligned with the end of the target series
        last_forecast_shift = (
            forecast_horizon + model.output_chunk_shift - 1
        )  # - 1 because end boundary is inclusive
        if end_index + last_forecast_shift * self.ts_tg.freq > self.ts_tg.end_time():
            end_index = self.ts_tg.end_time() - last_forecast_shift * self.ts_tg.freq

        index_train = _get_historical_forecast_train_index(
            **test_args,
            overlap_end=False,
        )
        assert index_train == (
            start_train_index,
            end_index,
        ), f"Wrong training boundaries; expected {index_train}, received {(start_train_index, end_index)}"

        index_pred = _get_historical_forecast_predict_index(
            **test_args,
            overlap_end=False,
        )
        assert index_pred == (
            start_pred_index,
            end_index,
        ), f"Wrong prediction boundaries; expected {index_pred}, received {(start_pred_index, end_index)}"

    @pytest.mark.parametrize(
        "config",
        product(
            [
                True,
                False,
                3,
                lambda x: x % 2 == 0,
            ],
            ("date", "integer"),
        ),
    )
    def test_reconcialiate_historical_index(self, config):
        """Verify that the expected index is returned, depending on the value of `retrain` and `train_length`."""
        retrain, index_type = config

        if index_type == "date":
            ts = tg.linear_timeseries(
                start=Timestamp("01-01-2000"), end=Timestamp("01-15-2000"), freq="D"
            )
            # 01-01 to 01-05 are the lags corresponding to the label/horizon 01-06, ie
            # the training sample required to fit the LinearRegressionModel
            hf_train_index = tg.generate_index(
                start=Timestamp("01-07-2000"), end=Timestamp("01-16-2000"), freq="D"
            )
            hf_pred_index = tg.generate_index(
                start=Timestamp("01-06-2000"), end=Timestamp("01-16-2000"), freq="D"
            )
        else:
            ts = tg.linear_timeseries(start=5, end=20, freq=1)
            hf_train_index = tg.generate_index(start=11, end=21, freq=1)
            hf_pred_index = tg.generate_index(start=10, end=21, freq=1)

        # create a very simple model
        model = LinearRegressionModel(lags=5, output_chunk_length=1)

        # check that the generated index are correct
        assert (
            hf_train_index[0],
            hf_train_index[-1],
        ) == _get_historical_forecast_train_index(model, ts, 0, None, None, 1, True)
        assert (
            hf_pred_index[0],
            hf_pred_index[-1],
        ) == _get_historical_forecast_predict_index(model, ts, 0, None, None, 1, True)

        # fit the model
        model.fit(ts)

        if isinstance(retrain, Callable) or not retrain:
            expected_index_length = len(hf_pred_index)
        else:
            expected_index_length = len(hf_train_index)

        test_args = {
            "model": model,
            "historical_forecasts_time_index_predict": hf_pred_index,
            "historical_forecasts_time_index_train": hf_train_index,
            "series": ts,
            "series_idx": 0,
            "retrain": retrain,
            "show_warnings": False,
        }

        # train_length = None
        recon_index, modified_train_length = _reconciliate_historical_time_indices(
            **test_args,
            train_length=None,
        )
        assert modified_train_length == model.min_train_series_length
        assert (
            n_steps_between(recon_index[-1], recon_index[0], freq=ts.freq) + 1
            == expected_index_length
        )

        # model.min_train_series_length < train_length < len(series)
        small_train_length = len(ts) - 5
        recon_index, modified_train_length = _reconciliate_historical_time_indices(
            **test_args,
            train_length=small_train_length,
        )
        # if retraining, the model will be trained with a greater amount of timesteps
        if retrain:
            assert modified_train_length == small_train_length
        else:
            assert modified_train_length == model.min_train_series_length
        assert (
            n_steps_between(recon_index[-1], recon_index[0], freq=ts.freq) + 1
            == expected_index_length
            - modified_train_length
            + model.min_train_series_length
        )

        # train_length > len(series)
        large_train_length = len(ts) + 1
        recon_index, modified_train_length = _reconciliate_historical_time_indices(
            **test_args,
            train_length=large_train_length,
        )
        # if the provided `train_length` is too large, its overwritten with the smallest possible value
        assert modified_train_length == model.min_train_series_length
        assert (
            n_steps_between(recon_index[-1], recon_index[0], freq=ts.freq) + 1
            == expected_index_length
        )

    @pytest.mark.parametrize("index_type", ["date", "integer"])
    def test_adjust_historical_index(self, index_type):
        # assume a model created with lags = 2
        if index_type == "date":
            ts = tg.linear_timeseries(
                start=Timestamp("01-01-2000"), end=Timestamp("01-15-2000"), freq="D"
            )
            hf_index = tg.generate_index(
                start=Timestamp("01-03-2000"), end=Timestamp("01-16-2000"), freq="D"
            )
        else:
            ts = tg.linear_timeseries(start=5, end=20, freq=1)
            hf_index = tg.generate_index(start=7, end=21, freq=1)

        test_args = {
            "series": ts,
            "series_idx": 0,
            "historical_forecasts_time_index": hf_index,
            "show_warnings": False,
        }
        # absolute position of the timestamp, in the original series
        start_value = 4
        adjusted_index = _adjust_historical_forecasts_time_index(
            **test_args, start=start_value, start_format="position"
        )
        assert adjusted_index[0] == ts.time_index[start_value]

        if index_type == "integer":
            adjusted_index = _adjust_historical_forecasts_time_index(
                **test_args, start=6, start_format="value"
            )
            # 6 is not in the forecastable index
            assert adjusted_index[0] == hf_index[0]

            adjusted_index = _adjust_historical_forecasts_time_index(
                **test_args, start=8, start_format="value"
            )
            # 8 is the second timestamp in the forecastable index
            assert adjusted_index[0] == hf_index[1]
        else:
            adjusted_index = _adjust_historical_forecasts_time_index(
                **test_args, start=Timestamp("01-02-2000"), start_format="value"
            )
            # "01-02-2000" is not in the forecastable index
            assert adjusted_index[0] == hf_index[0]

            adjusted_index = _adjust_historical_forecasts_time_index(
                **test_args, start=Timestamp("01-05-2000"), start_format="value"
            )
            # "01-05-2000" is the third timestamp in the forecastable index
            assert adjusted_index[0] == hf_index[2]

        # relative position of the timestamp
        start_value = 0.5
        adjusted_index = _adjust_historical_forecasts_time_index(
            **test_args, start=start_value, start_format="value"
        )
        if index_type == "integer":
            # when slicing with integer, end boundary is exclusive
            start_index = int(start_value * len(ts)) - 1
        else:
            # when slicing with Timestmap, end boundary is inclusive
            start_index = int(start_value * len(ts))
        assert adjusted_index[0] == ts.time_index[start_index]
