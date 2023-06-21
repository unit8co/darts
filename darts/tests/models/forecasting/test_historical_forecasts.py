import unittest

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.logging import get_logger
from darts.models import ARIMA, AutoARIMA, LinearRegressionModel, NaiveSeasonal
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

try:
    import torch

    from darts.models import (
        BlockRNNModel,
        NBEATSModel,
        NLinearModel,
        RNNModel,
        TCNModel,
        TFTModel,
        TransformerModel,
    )
    from darts.utils.likelihood_models import GaussianLikelihood

    TORCH_AVAILABLE = True
except ImportError:
    logger = get_logger(__name__)
    logger.warning(
        "Torch not installed - will be skipping historical forecasts tests for torch models"
    )
    TORCH_AVAILABLE = False

models_reg_no_cov_cls_kwargs = [
    (LinearRegressionModel, {"lags": 8}, (8, 1)),
    # (CatBoostModel, {"lags": 6}, (6, 1)),
    # (LightGBMModel, {"lags": 4}, (4, 1)),
]

models_reg_cov_cls_kwargs = [
    # target + past covariates
    (LinearRegressionModel, {"lags": 4, "lags_past_covariates": 6}, (6, 1)),
    # target + past covariates + outputchunk > 3, 6 > 3
    (
        LinearRegressionModel,
        {"lags": 3, "lags_past_covariates": 6, "output_chunk_length": 5},
        (6, 5),
    ),
    # target + future covariates, 2 because to predict x, require x and x+1
    (LinearRegressionModel, {"lags": 4, "lags_future_covariates": [0, 1]}, (4, 2)),
    # target + fut cov + output_chunk_length > 3,
    (
        LinearRegressionModel,
        {"lags": 2, "lags_future_covariates": [1, 2], "output_chunk_length": 5},
        (2, 5),
    ),
    # fut cov + output_chunk_length > 3, 5 > 2
    (
        LinearRegressionModel,
        {"lags_future_covariates": [0, 1], "output_chunk_length": 5},
        (0, 5),
    ),
    # past cov only
    (LinearRegressionModel, {"lags_past_covariates": 6}, (6, 1)),
    # fut cov only
    (LinearRegressionModel, {"lags_future_covariates": [0, 1]}, (0, 2)),
    # fut + past cov only
    (
        LinearRegressionModel,
        {"lags_past_covariates": 6, "lags_future_covariates": [0, 1]},
        (6, 2),
    ),
    # all
    (
        LinearRegressionModel,
        {"lags": 3, "lags_past_covariates": 6, "lags_future_covariates": [0, 1]},
        (6, 2),
    ),
]

if TORCH_AVAILABLE:
    IN_LEN = 24
    OUT_LEN = 12

    NB_EPOCH = 1

    models_torch_cls_kwargs = [
        (
            BlockRNNModel,
            {
                "input_chunk_length": IN_LEN,
                "output_chunk_length": OUT_LEN,
                "model": "RNN",
                "hidden_dim": 10,
                "n_rnn_layers": 1,
                "batch_size": 32,
                "n_epochs": NB_EPOCH,
            },
            # Min of lags needed and max of lags needed
            (IN_LEN, OUT_LEN),
            "PastCovariates",
        ),
        (
            RNNModel,
            {
                "input_chunk_length": IN_LEN,
                "model": "RNN",
                "hidden_dim": 10,
                "batch_size": 32,
                "n_epochs": NB_EPOCH,
            },
            # autoregressive model
            (IN_LEN, 1),
            "DualCovariates",
        ),
        (
            RNNModel,
            {
                "input_chunk_length": IN_LEN,
                "training_length": 12,
                "n_epochs": NB_EPOCH,
                "likelihood": GaussianLikelihood(),
            },
            (IN_LEN, 1),
            "DualCovariates",
        ),
        (
            TCNModel,
            {
                "input_chunk_length": IN_LEN,
                "output_chunk_length": OUT_LEN,
                "n_epochs": NB_EPOCH,
                "batch_size": 32,
            },
            (IN_LEN, OUT_LEN),
            "PastCovariates",
        ),
        (
            TransformerModel,
            {
                "input_chunk_length": IN_LEN,
                "output_chunk_length": OUT_LEN,
                "d_model": 16,
                "nhead": 2,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "dim_feedforward": 16,
                "batch_size": 32,
                "n_epochs": NB_EPOCH,
            },
            (IN_LEN, OUT_LEN),
            "PastCovariates",
        ),
        (
            NBEATSModel,
            {
                "input_chunk_length": IN_LEN,
                "output_chunk_length": OUT_LEN,
                "num_stacks": 4,
                "num_blocks": 1,
                "num_layers": 2,
                "layer_widths": 12,
                "n_epochs": NB_EPOCH,
            },
            (IN_LEN, OUT_LEN),
            "PastCovariates",
        ),
        (
            TFTModel,
            {
                "input_chunk_length": IN_LEN,
                "output_chunk_length": OUT_LEN,
                "hidden_size": 16,
                "lstm_layers": 1,
                "num_attention_heads": 4,
                "add_relative_index": True,
                "n_epochs": NB_EPOCH,
            },
            (IN_LEN, OUT_LEN),
            "MixedCovariates",
        ),
        (
            NLinearModel,
            {
                "input_chunk_length": IN_LEN,
                "output_chunk_length": OUT_LEN,
                "n_epochs": NB_EPOCH,
            },
            (IN_LEN, OUT_LEN),
            "MixedCovariates",
        ),
    ]


class HistoricalforecastTestCase(DartsBaseTestClass):
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)

    # real timeseries for functionality tests
    ts_val_length = 72
    ts_passengers = AirPassengersDataset().load()
    scaler = Scaler()
    ts_passengers = scaler.fit_transform(ts_passengers)
    ts_pass_train, ts_pass_val = (
        ts_passengers[:-ts_val_length],
        ts_passengers[-ts_val_length:],
    )

    # an additional noisy series
    ts_pass_train_1 = ts_pass_train + 0.01 * tg.gaussian_timeseries(
        length=len(ts_pass_train),
        freq=ts_pass_train.freq_str,
        start=ts_pass_train.start_time(),
    )

    ts_past_cov_train = tg.gaussian_timeseries(
        length=len(ts_pass_train),
        freq=ts_pass_train.freq_str,
        start=ts_pass_train.start_time(),
    )

    ts_fut_cov_train = tg.gaussian_timeseries(
        length=len(ts_pass_train),
        freq=ts_pass_train.freq_str,
        start=ts_pass_train.start_time(),
    )

    ts_past_cov_valid_same_start = tg.gaussian_timeseries(
        length=len(ts_pass_val),
        freq=ts_pass_val.freq_str,
        start=ts_pass_val.start_time(),
    )

    ts_past_cov_valid_10_bef_start = tg.gaussian_timeseries(
        length=len(ts_pass_val) + 10,
        freq=ts_pass_val.freq_str,
        start=ts_pass_val.start_time() - 10 * ts_pass_val.freq,
    )
    ts_past_cov_valid_5_aft_start = tg.gaussian_timeseries(
        length=len(ts_pass_val) - 5,
        freq=ts_pass_val.freq_str,
        start=ts_pass_val.start_time() + 5 * ts_pass_val.freq,
    )

    ts_fut_cov_valid_same_start = tg.gaussian_timeseries(
        length=len(ts_pass_val),
        freq=ts_pass_val.freq_str,
        start=ts_pass_val.start_time(),
    )

    ts_fut_cov_valid_16_bef_start = tg.gaussian_timeseries(
        length=len(ts_pass_val) + 16,
        freq=ts_pass_val.freq_str,
        start=ts_pass_val.start_time() - 16 * ts_pass_val.freq,
    )
    ts_fut_cov_valid_7_aft_start = tg.gaussian_timeseries(
        length=len(ts_pass_val) - 7,
        freq=ts_pass_val.freq_str,
        start=ts_pass_val.start_time() + 7 * ts_pass_val.freq,
    )

    # RangeIndex timeseries
    ts_passengers_range = TimeSeries.from_values(ts_passengers.values())
    ts_pass_train_range, ts_pass_val_range = (
        ts_passengers_range[:-ts_val_length],
        ts_passengers_range[-ts_val_length:],
    )

    ts_past_cov_train_range = tg.gaussian_timeseries(
        length=len(ts_pass_train_range),
        freq=ts_pass_train_range.freq_str,
        start=ts_pass_train_range.start_time(),
    )

    # same starting point
    ts_past_cov_valid_range_same_start = tg.gaussian_timeseries(
        length=len(ts_pass_val_range),
        freq=ts_pass_val_range.freq_str,
        start=ts_pass_val_range.start_time(),
    )

    def test_historical_forecasts_transferrable_future_cov_local_models(self):
        model = ARIMA()
        assert model.min_train_series_length == 30
        series = tg.sine_timeseries(length=31)
        res = model.historical_forecasts(
            series, future_covariates=series, retrain=True, forecast_horizon=1
        )
        # ARIMA has a minimum train length of 30, with horizon=1, we expect one forecast at last point
        # (series has length 31)
        assert len(res) == 1
        assert series.end_time() == res.time_index[0]

        model.fit(series, future_covariates=series)
        res = model.historical_forecasts(
            series, future_covariates=series, retrain=False, forecast_horizon=1
        )
        # currently even though transferrable local models would allow , the models currently still take the
        # min_train_length as input for historical forecast predictions (due to extreme_lags not differentiating
        # between fit and predict)
        # (series has length 31)
        assert len(res) == 1
        assert series.end_time() == res.time_index[0]

    def test_historical_forecasts_future_cov_local_models(self):
        model = AutoARIMA()
        assert model.min_train_series_length == 10
        series = tg.sine_timeseries(length=11)
        res = model.historical_forecasts(
            series, future_covariates=series, retrain=True, forecast_horizon=1
        )
        # AutoARIMA has a minimum train length of 10, with horizon=1, we expect one forecast at last point
        # (series has length 11)
        assert len(res) == 1
        assert series.end_time() == res.time_index[0]

        model.fit(series, future_covariates=series)
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(
                series, future_covariates=series, retrain=False, forecast_horizon=1
            )
        assert str(msg.value).startswith(
            "FutureCovariatesLocalForecastingModel does not support historical forecasting "
            "with `retrain` set to `False`"
        )

    def test_historical_forecasts_local_models(self):
        model = NaiveSeasonal()
        assert model.min_train_series_length == 3
        series = tg.sine_timeseries(length=4)
        res = model.historical_forecasts(series, retrain=True, forecast_horizon=1)
        # NaiveSeasonal has a minimum train length of 3, with horizon=1, we expect one forecast at last point
        # (series has length 4)
        assert len(res) == 1
        assert series.end_time() == res.time_index[0]

        model.fit(series)
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(series, retrain=False, forecast_horizon=1)
        assert str(msg.value).startswith(
            "LocalForecastingModel does not support historical forecasting with `retrain` set to `False`"
        )

    def test_historical_forecasts(self):
        train_length = 10
        forecast_horizon = 8
        # if no fit and retrain=false, should fit at fist iteration
        for model_cls, kwargs, bounds in models_reg_no_cov_cls_kwargs:
            model = model_cls(
                **kwargs,
            )

            # time index
            forecasts = model.historical_forecasts(
                series=self.ts_pass_val,
                forecast_horizon=forecast_horizon,
                stride=1,
                train_length=train_length,
                retrain=True,
                overlap_end=False,
            )

            theorical_forecast_length = (
                self.ts_val_length
                - max(
                    [
                        (
                            bounds[0] + bounds[1] + 1
                        ),  # +1 as sklearn models require min 2 train samples
                        train_length,
                    ]
                )  # because we train
                - forecast_horizon  # because we have overlap_end = False
                + 1  # because we include the first element
            )

            self.assertTrue(
                len(forecasts) == theorical_forecast_length,
                f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
                f"of retrain=True and overlap_end=False, and a time index of type DateTimeIndex. "
                f"Expected {theorical_forecast_length}, got {len(forecasts)}",
            )

            # range index
            forecasts = model.historical_forecasts(
                series=self.ts_pass_val_range,
                forecast_horizon=forecast_horizon,
                train_length=train_length,
                stride=1,
                retrain=True,
                overlap_end=False,
            )

            self.assertTrue(
                len(forecasts) == theorical_forecast_length,
                f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
                f"of retrain=True, overlap_end=False, and a time index of type RangeIndex."
                f"Expected {theorical_forecast_length}, got {len(forecasts)}",
            )

            # stride 2
            forecasts = model.historical_forecasts(
                series=self.ts_pass_val_range,
                forecast_horizon=forecast_horizon,
                train_length=train_length,
                stride=2,
                retrain=True,
                overlap_end=False,
            )

            theorical_forecast_length = np.floor(
                (
                    (
                        self.ts_val_length
                        - max(
                            [
                                (
                                    bounds[0] + bounds[1] + 1
                                ),  # +1 as sklearn models require min 2 train samples
                                train_length,
                            ]
                        )  # because we train
                        - forecast_horizon  # because we have overlap_end = False
                        + 1  # because we include the first element
                    )
                    - 1
                )
                / 2
                + 1  # because of stride
            )  # if odd number of elements, we keep the floor

            self.assertTrue(
                len(forecasts) == theorical_forecast_length,
                f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
                f"of retrain=True and overlap_end=False and stride=2. "
                f"Expected {theorical_forecast_length}, got {len(forecasts)}",
            )

            # stride 3
            forecasts = model.historical_forecasts(
                series=self.ts_pass_val_range,
                forecast_horizon=forecast_horizon,
                train_length=train_length,
                stride=3,
                retrain=True,
                overlap_end=False,
            )

            theorical_forecast_length = np.floor(
                (
                    (
                        self.ts_val_length
                        - max(
                            [
                                (
                                    bounds[0] + bounds[1] + 1
                                ),  # +1 as sklearn models require min 2 train samples
                                train_length,
                            ]
                        )  # because we train
                        - forecast_horizon  # because we have overlap_end = False
                        + 1  # because we include the first element
                    )
                    - 1
                )  # the first is always included, so we calculate a modulo on the rest
                / 3  # because of stride
                + 1  # and we readd the first
            )  # if odd number of elements, we keep the floor

            # Here to adapt if forecast_horizon or train_length change
            self.assertTrue(
                len(forecasts) == theorical_forecast_length,
                f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
                f"of retrain=True and overlap_end=False and stride=3. "
                f"Expected {theorical_forecast_length}, got {len(forecasts)}",
            )

            # last points only False
            forecasts = model.historical_forecasts(
                series=self.ts_pass_val_range,
                forecast_horizon=forecast_horizon,
                train_length=train_length,
                stride=1,
                retrain=True,
                overlap_end=False,
                last_points_only=False,
            )

            theorical_forecast_length = (
                self.ts_val_length
                - max(
                    [
                        (
                            bounds[0] + bounds[1] + 1
                        ),  # +1 as sklearn models require min 2 train samples
                        train_length,
                    ]
                )  # because we train
                - forecast_horizon  # because we have overlap_end = False
                + 1  # because we include the first element
            )

            self.assertTrue(
                len(forecasts) == theorical_forecast_length,
                f"Model {model_cls} does not return the right number of historical forecasts in the case of "
                f"retrain=True and overlap_end=False, and last_points_only=False. "
                f"expected {theorical_forecast_length}, got {len(forecasts)}",
            )

            self.assertTrue(
                len(forecasts[0]) == forecast_horizon,
                f"Model {model_cls} does not return forecast_horizon points per historical forecast in the case of "
                f"retrain=True and overlap_end=False, and last_points_only=False",
            )

    def test_sanity_check_invalid_start(self):
        timeidx_ = tg.linear_timeseries(length=10)
        rangeidx_step1 = tg.linear_timeseries(start=0, length=10, freq=1)
        rangeidx_step2 = tg.linear_timeseries(start=0, length=10, freq=2)

        # index too large
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(timeidx_, start=11)
        assert str(msg.value).startswith("`start` index `11` is out of bounds")
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(rangeidx_step1, start=11)
        assert str(msg.value).startswith("`start` index `11` is out of bounds")
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(rangeidx_step2, start=11)
        assert str(msg.value).startswith("The provided point is not a valid index")

        # value too low
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(
                timeidx_, start=timeidx_.start_time() - timeidx_.freq
            )
        assert str(msg.value).startswith(
            "`start` time `1999-12-31 00:00:00` is before the first timestamp `2000-01-01 00:00:00`"
        )
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(
                rangeidx_step1, start=rangeidx_step1.start_time() - rangeidx_step1.freq
            )
        assert str(msg.value).startswith("if `start` is an integer, must be `>= 0`")
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(
                rangeidx_step2, start=rangeidx_step2.start_time() - rangeidx_step2.freq
            )
        assert str(msg.value).startswith("if `start` is an integer, must be `>= 0`")

        # value too high
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(
                timeidx_, start=timeidx_.end_time() + timeidx_.freq
            )
        assert str(msg.value).startswith(
            "`start` time `2000-01-11 00:00:00` is after the last timestamp `2000-01-10 00:00:00`"
        )
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(
                rangeidx_step1, start=rangeidx_step1.end_time() + rangeidx_step1.freq
            )
        assert str(msg.value).startswith("`start` index `10` is out of bounds")
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(
                rangeidx_step2, start=rangeidx_step2.end_time() + rangeidx_step2.freq
            )
        assert str(msg.value).startswith(
            "`start` index `20` is larger than the last index `18`"
        )

    def test_regression_auto_start_multiple_no_cov(self):
        train_length = 15
        forecast_horizon = 10
        for model_cls, kwargs, bounds in models_reg_no_cov_cls_kwargs:
            model = model_cls(
                **kwargs,
            )
            model.fit(self.ts_pass_train)

            forecasts = model.historical_forecasts(
                series=[self.ts_pass_val, self.ts_pass_val],
                forecast_horizon=forecast_horizon,
                train_length=train_length,
                stride=1,
                retrain=True,
                overlap_end=False,
            )

            self.assertTrue(
                len(forecasts) == 2,
                f"Model {model_cls} did not return a list of historical forecasts",
            )

            theorical_forecast_length = (
                self.ts_val_length
                - max(
                    [
                        (
                            bounds[0] + bounds[1] + 1
                        ),  # +1 as sklearn models require min 2 train samples
                        train_length,
                    ]
                )  # because we train
                - forecast_horizon  # because we have overlap_end = False
                + 1  # because we include the first element
            )

            self.assertTrue(
                len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length,
                f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
                f"of retrain=True and overlap_end=False, and a time index of type DateTimeIndex. "
                f"Expected {theorical_forecast_length}, got {len(forecasts[0])} and {len(forecasts[1])}",
            )

    @pytest.mark.slow
    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "Torch not available. auto start and multiple time series for torch models will be skipped.",
    )
    def test_torch_auto_start_multiple_no_cov(self):
        forecast_hrz = 10
        for model_cls, kwargs, bounds, _ in models_torch_cls_kwargs:
            model = model_cls(
                random_state=0,
                **kwargs,
            )
            model.fit(self.ts_pass_train)

            # check historical forecasts for several time series,
            # retrain True and overlap_end False
            forecasts = model.historical_forecasts(
                series=[self.ts_pass_val, self.ts_pass_val],
                forecast_horizon=forecast_hrz,
                stride=1,
                retrain=True,
                overlap_end=False,
            )
            self.assertTrue(
                len(forecasts) == 2,
                f"Model {model_cls} did not return a list of historical forecasts",
            )
            # If retrain=True and overlap_end=False, as ts has 72 values, we can only forecast
            # (target length)-(training length=input_chunk_length+output_chunk_length) - (horizon - 1)
            # indeed we start to predict after the first trainable point (input_chunk_length+output_chunk_length)
            # and we stop in this case (overlap_end=False) at the end_time:
            # target.end_time() - (horizon - 1) * target.freq

            # explanation:
            # (bounds): train sample length
            # (horizon - 1): with overlap_end=False, if entire horizon is available (overlap_end=False),
            # we can predict 1
            theorical_forecast_length = (
                self.ts_val_length - (bounds[0] + bounds[1]) - (forecast_hrz - 1)
            )
            self.assertTrue(
                len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length,
                f"Model {model_cls} does not return the right number of historical forecasts in the case of "
                f"retrain=True and overlap_end=False. "
                f"Expected {theorical_forecast_length}, got {len(forecasts[0])} and {len(forecasts[1])}",
            )

            model = model_cls(
                random_state=0,
                **kwargs,
            )

            model.fit(self.ts_pass_train)
            # check historical forecasts for several time series,
            # retrain True and overlap_end True
            forecasts = model.historical_forecasts(
                series=[self.ts_pass_val, self.ts_pass_val],
                forecast_horizon=forecast_hrz,
                stride=1,
                retrain=True,
                overlap_end=True,
            )

            self.assertTrue(
                len(forecasts) == 2,
                f"Model {model_cls} did not return a list of historical forecasts",
            )
            theorical_forecast_length = (
                self.ts_val_length
                - (bounds[0] + bounds[1])  # train sample length
                - 0  # with overlap_end=True, we are not restricted by the end of the series or horizon
            )
            self.assertTrue(
                len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length
            )

            model = model_cls(
                random_state=0,
                **kwargs,
            )
            model.fit(self.ts_pass_train)
            # check historical forecasts for several time series,
            # retrain False and overlap_end False
            forecasts = model.historical_forecasts(
                series=[self.ts_pass_val, self.ts_pass_val],
                forecast_horizon=forecast_hrz,
                stride=1,
                retrain=False,
                overlap_end=False,
            )

            self.assertTrue(
                len(forecasts) == 2,
                f"Model {model_cls} did not return a list of historical forecasts",
            )
            theorical_forecast_length = (
                self.ts_val_length
                - bounds[0]  # prediction input sample length
                - (
                    forecast_hrz - 1
                )  # overlap_end=False -> if entire horizon is available, we can predict 1
            )
            self.assertTrue(
                len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length
            )

            model = model_cls(
                random_state=0,
                **kwargs,
            )
            model.fit(self.ts_pass_train)
            # check historical forecasts for several time series,
            # retrain False and overlap_end True
            forecasts = model.historical_forecasts(
                series=[self.ts_pass_val, self.ts_pass_val],
                forecast_horizon=forecast_hrz,
                stride=1,
                retrain=False,
                overlap_end=True,
            )

            self.assertTrue(
                len(forecasts) == 2,
                f"Model {model_cls} did not return a list of historical forecasts",
            )
            theorical_forecast_length = (
                self.ts_val_length
                - bounds[0]  # prediction input sample length
                - 0  # overlap_end=False -> we are not restricted by the end of the series or horizon
            )
            self.assertTrue(
                len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length
            )

    def test_regression_auto_start_multiple_with_cov_retrain(self):
        forecast_hrz = 10
        for model_cls, kwargs, bounds in models_reg_cov_cls_kwargs:
            model = model_cls(
                random_state=0,
                **kwargs,
            )

            forecasts_retrain = model.historical_forecasts(
                series=[self.ts_pass_val, self.ts_pass_val],
                past_covariates=[
                    self.ts_past_cov_valid_same_start,
                    self.ts_past_cov_valid_same_start,
                ]
                if "lags_past_covariates" in kwargs
                else None,
                future_covariates=[
                    self.ts_past_cov_valid_same_start,
                    self.ts_past_cov_valid_same_start,
                ]
                if "lags_future_covariates" in kwargs
                else None,
                last_points_only=True,
                forecast_horizon=forecast_hrz,
                stride=1,
                retrain=True,
                overlap_end=False,
            )

            self.assertTrue(
                len(forecasts_retrain) == 2,
                f"Model {model_cls} did not return a list of historical forecasts",
            )

            (
                min_target_lag,
                max_target_lag,
                min_past_cov_lag,
                max_past_cov_lag,
                min_future_cov_lag,
                max_future_cov_lag,
            ) = model.extreme_lags

            past_lag = min(
                min_target_lag if min_target_lag else 0,
                min_past_cov_lag if min_past_cov_lag else 0,
                min_future_cov_lag
                if min_future_cov_lag is not None and min_future_cov_lag < 0
                else 0,
            )

            future_lag = (
                max_future_cov_lag
                if max_future_cov_lag is not None and max_future_cov_lag > 0
                else 0
            )
            # length input - biggest past lag - biggest future lag - forecast horizon - output_chunk_length
            theorical_retrain_forecast_length = len(self.ts_pass_val) - (
                -past_lag
                + future_lag
                + forecast_hrz
                + kwargs.get("output_chunk_length", 1)
            )

            self.assertTrue(
                len(forecasts_retrain[0])
                == len(forecasts_retrain[1])
                == theorical_retrain_forecast_length,
                f"Model {model_cls} does not return the right number of historical forecasts in the case of "
                f"retrain=True and overlap_end=False. "
                f"Expected {theorical_retrain_forecast_length}, got {len(forecasts_retrain[0])} "
                f"and {len(forecasts_retrain[1])}",
            )

            # with last_points_only=True: start is shifted by biggest past lag + training timestamps
            # (forecast horizon + output_chunk_length)
            expected_start = (
                self.ts_pass_val.start_time()
                + (-past_lag + forecast_hrz + kwargs.get("output_chunk_length", 1))
                * self.ts_pass_val.freq
            )
            self.assertEqual(forecasts_retrain[0].start_time(), expected_start)

            # end is shifted back by the biggest future lag
            expected_end = (
                self.ts_pass_val.end_time() - future_lag * self.ts_pass_val.freq
            )
            self.assertEqual(forecasts_retrain[0].end_time(), expected_end)

    def test_regression_auto_start_multiple_with_cov_no_retrain(self):
        forecast_hrz = 10
        for model_cls, kwargs, bounds in models_reg_cov_cls_kwargs:
            model = model_cls(
                random_state=0,
                **kwargs,
            )

            model.fit(
                series=[self.ts_pass_val, self.ts_pass_val],
                past_covariates=[
                    self.ts_past_cov_valid_same_start,
                    self.ts_past_cov_valid_same_start,
                ]
                if "lags_past_covariates" in kwargs
                else None,
                future_covariates=[
                    self.ts_past_cov_valid_same_start,
                    self.ts_past_cov_valid_same_start,
                ]
                if "lags_future_covariates" in kwargs
                else None,
            )
            forecasts_no_retrain = model.historical_forecasts(
                series=[self.ts_pass_val, self.ts_pass_val],
                past_covariates=[
                    self.ts_past_cov_valid_same_start,
                    self.ts_past_cov_valid_same_start,
                ]
                if "lags_past_covariates" in kwargs
                else None,
                future_covariates=[
                    self.ts_past_cov_valid_same_start,
                    self.ts_past_cov_valid_same_start,
                ]
                if "lags_future_covariates" in kwargs
                else None,
                last_points_only=True,
                forecast_horizon=forecast_hrz,
                stride=1,
                retrain=False,
                overlap_end=False,
            )

            (
                min_target_lag,
                max_target_lag,
                min_past_cov_lag,
                max_past_cov_lag,
                min_future_cov_lag,
                max_future_cov_lag,
            ) = model.extreme_lags

            past_lag = min(
                min_target_lag if min_target_lag else 0,
                min_past_cov_lag if min_past_cov_lag else 0,
                min_future_cov_lag if min_future_cov_lag else 0,
            )

            future_lag = (
                max_future_cov_lag
                if max_future_cov_lag is not None and max_future_cov_lag > 0
                else 0
            )

            # with last_points_only=True: start is shifted by the biggest past lag plus the forecast horizon
            expected_start = (
                self.ts_pass_val.start_time()
                + (-past_lag + forecast_hrz - 1) * self.ts_pass_val.freq
            )
            self.assertEqual(forecasts_no_retrain[0].start_time(), expected_start)

            # end is shifted by the biggest future lag
            expected_end = (
                self.ts_pass_val.end_time() - future_lag * self.ts_pass_val.freq
            )
            self.assertEqual(forecasts_no_retrain[0].end_time(), expected_end)

    @pytest.mark.slow
    @unittest.skipUnless(
        TORCH_AVAILABLE,
        "Torch not available. auto start and multiple time series for torch models and covariates "
        "will be skipped.",
    )
    def test_torch_auto_start_with_cov(self):
        forecast_hrz = 10
        # Past covariates only
        for model_cls, kwargs, bounds, type in models_torch_cls_kwargs:

            if type == "DualCovariates":
                continue

            model = model_cls(
                random_state=0,
                **kwargs,
            )
            model.fit(self.ts_pass_train, self.ts_past_cov_train)

            # same start
            forecasts = model.historical_forecasts(
                series=[self.ts_pass_val, self.ts_pass_val],
                past_covariates=[
                    self.ts_past_cov_valid_same_start,
                    self.ts_past_cov_valid_same_start,
                ],
                forecast_horizon=forecast_hrz,
                stride=1,
                retrain=True,
                overlap_end=False,
            )

            self.assertTrue(
                len(forecasts) == 2,
                f"Model {model_cls} did not return a list of historical forecasts",
            )

            theorical_forecast_length = (
                self.ts_val_length
                - (bounds[0] + bounds[1])  # train sample length
                - (forecast_hrz - 1)  # if entire horizon is available, we can predict 1
                - 0  # past covs have same start as target -> no shift
                - 0  # we don't have future covs in output chunk -> no shift
            )
            self.assertTrue(
                len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length,
                f"Model {model_cls} does not return the right number of historical forecasts in case "
                f"of retrain=True and overlap_end=False and past_covariates with same start. "
                f"Expected {theorical_forecast_length}, got {len(forecasts[0])} and {len(forecasts[1])}",
            )

            model = model_cls(
                random_state=0,
                **kwargs,
            )
            model.fit(self.ts_pass_train, past_covariates=self.ts_past_cov_train)

            # start before, after
            forecasts = model.historical_forecasts(
                series=[self.ts_pass_val, self.ts_pass_val],
                past_covariates=[
                    self.ts_past_cov_valid_5_aft_start,
                    self.ts_past_cov_valid_10_bef_start,
                ],
                forecast_horizon=forecast_hrz,
                stride=1,
                retrain=True,
                overlap_end=False,
            )
            theorical_forecast_length = (
                self.ts_val_length
                - (bounds[0] + bounds[1])  # train sample length
                - (forecast_hrz - 1)  # if entire horizon is available, we can predict 1
                - 5  # past covs start 5 later -> shift
                - 0  # we don't have future covs in output chunk -> no shift
            )
            self.assertTrue(
                len(forecasts[0]) == theorical_forecast_length,
                f"Model {model_cls} does not return the right number of historical forecasts in case "
                f"of retrain=True and overlap_end=False and past_covariates starting after. "
                f"Expected {theorical_forecast_length}, got {len(forecasts[0])}",
            )
            theorical_forecast_length = (
                self.ts_val_length
                - (bounds[0] + bounds[1])  # train sample length
                - (forecast_hrz - 1)  # if entire horizon is available, we can predict 1
                - 0  # past covs have same start as target -> no shift
                - 0  # we don't have future covs in output chunk -> no shift
            )
            self.assertTrue(
                len(forecasts[1]) == theorical_forecast_length,
                f"Model {model_cls} does not return the right number of historical forecasts in case "
                f"of retrain=True and overlap_end=False and past_covariates starting before. "
                f"Expected {theorical_forecast_length}, got {len(forecasts[1])}",
            )

        # Past and future covariates
        for model_cls, kwargs, bounds, type in models_torch_cls_kwargs:
            if not type == "MixedCovariates":
                continue

            model = model_cls(
                random_state=0,
                **kwargs,
            )
            model.fit(
                self.ts_pass_train,
                past_covariates=self.ts_past_cov_train,
                future_covariates=self.ts_fut_cov_train,
            )

            forecasts = model.historical_forecasts(
                series=[self.ts_pass_val, self.ts_pass_val],
                past_covariates=[
                    self.ts_past_cov_valid_5_aft_start,
                    self.ts_past_cov_valid_same_start,
                ],
                future_covariates=[
                    self.ts_fut_cov_valid_7_aft_start,
                    self.ts_fut_cov_valid_16_bef_start,
                ],
                forecast_horizon=forecast_hrz,
                stride=1,
                retrain=True,
                overlap_end=False,
            )
            theorical_forecast_length = (
                self.ts_val_length
                - (bounds[0] + bounds[1])  # train sample length
                - (forecast_hrz - 1)  # if entire horizon is available, we can predict 1
                - 7  # future covs start 7 after target (more than past covs) -> shift
                - 2  # future covs in output chunk -> difference between horizon=10 and output_chunk_length=12
            )
            self.assertTrue(
                len(forecasts[0]) == theorical_forecast_length,
                f"Model {model_cls} does not return the right number of historical forecasts in case "
                f"of retrain=True and overlap_end=False and past_covariates and future_covariates with "
                f"different start. "
                f"Expected {theorical_forecast_length}, got {len(forecasts[0])}",
            )
            theorical_forecast_length = (
                self.ts_val_length
                - (bounds[0] + bounds[1])  # train sample length
                - (
                    forecast_hrz - 1
                )  # if entire horizon is available, we can predict 1,
                - 0  # all covs start at the same time as target -> no shift,
                - 2  # future covs in output chunk -> difference between horizon=10 and output_chunk_length=12
            )
            self.assertTrue(
                len(forecasts[1]) == theorical_forecast_length,
                f"Model {model_cls} does not return the right number of historical forecasts in case "
                f"of retrain=True and overlap_end=False and past_covariates with different start. "
                f"Expected {theorical_forecast_length}, got {len(forecasts[1])}",
            )

        # Future covariates only
        for model_cls, kwargs, bounds, type in models_torch_cls_kwargs:
            # todo case of DualCovariates (RNN)
            if type == "PastCovariates" or type == "DualCovariates":
                continue

            model = model_cls(
                random_state=0,
                **kwargs,
            )
            model.fit(self.ts_pass_train, future_covariates=self.ts_fut_cov_train)

            # Only fut covariate
            forecasts = model.historical_forecasts(
                series=[self.ts_pass_val, self.ts_pass_val],
                future_covariates=[
                    self.ts_fut_cov_valid_7_aft_start,
                    self.ts_fut_cov_valid_16_bef_start,
                ],
                forecast_horizon=forecast_hrz,
                stride=1,
                retrain=True,
                overlap_end=False,
            )

            self.assertTrue(
                len(forecasts) == 2,
                f"Model {model_cls} did not return a list of historical forecasts",
            )
            theorical_forecast_length = (
                self.ts_val_length
                - (bounds[0] + bounds[1])  # train sample length
                - (
                    forecast_hrz - 1
                )  # (horizon - 1): if entire horizon is available, we can predict 1,
                - 7  # future covs start 7 after target (more than past covs) -> shift
                - 2  # future covs in output chunk -> difference between horizon=10 and output_chunk_length=12
            )
            self.assertTrue(
                len(forecasts[0]) == theorical_forecast_length,
                f"Model {model_cls} does not return the right number of historical forecasts in case "
                f"of retrain=True and overlap_end=False and no past_covariates and future_covariates "
                f"with different start. "
                f"Expected {theorical_forecast_length}, got {len(forecasts[0])}",
            )
            theorical_forecast_length = (
                self.ts_val_length
                - (bounds[0] + bounds[1])  # train sample length
                - (forecast_hrz - 1)  # if entire horizon is available, we can predict 1
                - 0  # all covs start at the same time as target -> no shift
                - 2  # future covs in output chunk -> difference between horizon=10 and output_chunk_length=12
            )
            self.assertTrue(
                len(forecasts[1]) == theorical_forecast_length,
                f"Model {model_cls} does not return the right number of historical forecasts in case "
                f"of retrain=True and overlap_end=False and no past_covariates and future_covariates "
                f"with different start. "
                f"Expected {theorical_forecast_length}, got {len(forecasts[1])}",
            )

    def test_retrain(self):
        """test historical_forecasts for an untrained model with different retrain values."""

        def helper_hist_forecasts(retrain_val, start):
            model = LinearRegressionModel(lags=4, output_chunk_length=4)
            return model.historical_forecasts(
                self.ts_passengers, start=start, retrain=retrain_val, verbose=False
            )

        def retrain_f_invalid(
            counter, pred_time, train_series, past_covariates, future_covariates
        ):
            return False

        def retrain_f_missing_arg(
            counter, train_series, past_covariates, future_covariates
        ):
            if len(train_series) % 2 == 0:
                return True
            else:
                return False

        def retrain_f_invalid_ouput_int(
            counter, pred_time, train_series, past_covariates, future_covariates
        ):
            return 1

        def retrain_f_invalid_ouput_str(
            counter, pred_time, train_series, past_covariates, future_covariates
        ):
            return "True"

        def retrain_f_valid(
            counter, pred_time, train_series, past_covariates, future_covariates
        ):
            # only retrain once in first iteration
            if pred_time == pd.Timestamp("1959-09-01 00:00:00"):
                return True
            else:
                return False

        def retrain_f_delayed_true(
            counter, pred_time, train_series, past_covariates, future_covariates
        ):
            if counter > 1:
                return True
            else:
                return False

        # test callable
        helper_hist_forecasts(retrain_f_valid, 0.9)
        # missing the `pred_time` positional argument
        expected_msg = "the Callable `retrain` must have a signature/arguments matching the following positional"
        with pytest.raises(ValueError) as error_msg:
            helper_hist_forecasts(retrain_f_missing_arg, 0.9)
        self.assertTrue(str(error_msg.value).startswith(expected_msg))
        # returning a non-bool value (int)
        expected_msg = "Return value of `retrain` must be bool, received <class 'int'>"
        with pytest.raises(ValueError) as error_msg:
            helper_hist_forecasts(retrain_f_invalid_ouput_int, 0.9)
        self.assertTrue(str(error_msg.value).startswith(expected_msg))
        # returning a non-bool value (str)
        expected_msg = "Return value of `retrain` must be bool, received <class 'str'>"
        with pytest.raises(ValueError) as error_msg:
            helper_hist_forecasts(retrain_f_invalid_ouput_str, 0.9)
        self.assertTrue(str(error_msg.value).startswith(expected_msg))
        # predict fails but model could have been trained before the predict round
        expected_msg = "`retrain` is `False` in the first train iteration at prediction point (in time)"
        with pytest.raises(ValueError) as error_msg:
            helper_hist_forecasts(retrain_f_delayed_true, 0.9)
        self.assertTrue(str(error_msg.value).startswith(expected_msg))
        # always returns False, treated slightly different than `retrain=False` and `retrain=0`
        with pytest.raises(ValueError) as error_msg:
            helper_hist_forecasts(retrain_f_invalid, 0.9)
        self.assertTrue(str(error_msg.value).startswith(expected_msg))

        # test int
        helper_hist_forecasts(10, 0.9)
        expected_msg = "Model has not been fit before the first predict iteration at prediction point (in time)"
        # `retrain=0` with not-trained model, encountering directly a predictable time index
        with pytest.raises(ValueError) as error_msg:
            helper_hist_forecasts(0, 0.9)
        self.assertTrue(str(error_msg.value).startswith(expected_msg))

        # test bool
        helper_hist_forecasts(True, 0.9)
        # `retrain=False` with not-trained model, encountering directly a predictable time index
        expected_msg = "The model has not been fitted yet, and `retrain` is ``False``."
        with pytest.raises(ValueError) as error_msg:
            helper_hist_forecasts(False, 0.9)
        self.assertTrue(str(error_msg.value).startswith(expected_msg))

        expected_start = pd.Timestamp("1949-10-01 00:00:00")
        # start before first trainable time index should still work
        res = helper_hist_forecasts(True, pd.Timestamp("1949-09-01 00:00:00"))
        self.assertTrue(res.time_index[0] == expected_start)
        # start at first trainable time index should still work
        res = helper_hist_forecasts(True, expected_start)
        self.assertTrue(res.time_index[0] == expected_start)
        # start at last trainable time index should still work
        expected_end = pd.Timestamp("1960-12-01 00:00:00")
        res = helper_hist_forecasts(True, expected_end)
        self.assertTrue(res.time_index[0] == expected_end)
