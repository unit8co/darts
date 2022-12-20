import unittest

import numpy as np

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

try:
    import torch

    from darts.models import (
        BlockRNNModel,
        CatBoostModel,
        LightGBMModel,
        LinearRegressionModel,
        NBEATSModel,
        RNNModel,
        TCNModel,
        TFTModel,
        TransformerModel,
    )
    from darts.utils.likelihood_models import GaussianLikelihood

    TORCH_AVAILABLE = True
except ImportError:
    logger = get_logger(__name__)
    logger.warning("Torch not installed - will be skipping historical forecasts tests")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    IN_LEN = 24
    OUT_LEN = 12

    NB_EPOCH = 1

    models_reg_no_cov_cls_kwargs = [
        (LinearRegressionModel, {"lags": 4}, (4, 1)),
        (CatBoostModel, {"lags": 4}, (4, 1)),
        (LightGBMModel, {"lags": 4}, (4, 1)),
    ]

    models_reg_cov_cls_kwargs = [
        # # target + past covariates
        (LinearRegressionModel, {"lags": 4, "lags_past_covariates": 6}, (6, 1)),
        # target + past covariates + outputchunk > 1
        (
            LinearRegressionModel,
            {"lags": 4, "lags_past_covariates": 6, "output_chunk_length": 4},
            (6, 4),
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
        ),
    ]

    class HistoricalforecastTestCase(DartsBaseTestClass):

        np.random.seed(42)
        torch.manual_seed(42)

        # real timeseries for functionality tests
        ts_passengers = AirPassengersDataset().load()
        scaler = Scaler()
        ts_passengers = scaler.fit_transform(ts_passengers)
        ts_pass_train, ts_pass_val = ts_passengers[:-72], ts_passengers[-72:]

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

        ts_past_cov_valid_bef_start = tg.gaussian_timeseries(
            length=len(ts_pass_val) + 10,
            freq=ts_pass_val.freq_str,
            start=ts_pass_val.start_time() - 10 * ts_pass_val.freq,
        )
        ts_past_cov_valid_aft_start = tg.gaussian_timeseries(
            length=len(ts_pass_val) - 5,
            freq=ts_pass_val.freq_str,
            start=ts_pass_val.start_time() + 5 * ts_pass_val.freq,
        )

        ts_past_cov_valid_bef_end = tg.gaussian_timeseries(
            length=len(ts_pass_val) - 7,
            freq=ts_pass_val.freq_str,
            start=ts_pass_val.start_time(),
        )
        ts_past_cov_valid_aft_end = tg.gaussian_timeseries(
            length=len(ts_pass_val) + 15,
            freq=ts_pass_val.freq_str,
            start=ts_pass_val.start_time(),
        )

        ts_fut_cov_valid_same_start = tg.gaussian_timeseries(
            length=len(ts_pass_val),
            freq=ts_pass_val.freq_str,
            start=ts_pass_val.start_time(),
        )

        ts_fut_cov_valid_bef_start = tg.gaussian_timeseries(
            length=len(ts_pass_val) + 16,
            freq=ts_pass_val.freq_str,
            start=ts_pass_val.start_time() - 16 * ts_pass_val.freq,
        )
        ts_fut_cov_valid_aft_start = tg.gaussian_timeseries(
            length=len(ts_pass_val) - 7,
            freq=ts_pass_val.freq_str,
            start=ts_pass_val.start_time() + 7 * ts_pass_val.freq,
        )

        ts_fut_cov_valid_bef_end = tg.gaussian_timeseries(
            length=len(ts_pass_val) - 7,
            freq=ts_pass_val.freq_str,
            start=ts_pass_val.start_time(),
        )
        ts_fut_cov_valid_aft_end = tg.gaussian_timeseries(
            length=len(ts_pass_val) + 15,
            freq=ts_pass_val.freq_str,
            start=ts_pass_val.start_time(),
        )

        # RangeIndex timeseries
        ts_passengers_range = TimeSeries.from_values(ts_passengers.values())
        ts_pass_train_range, ts_pass_val_range = (
            ts_passengers_range[:-72],
            ts_passengers_range[-72:],
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

        def test_historical_forecasts(self):

            train_length = 7
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
                    72
                    - max(
                        (bounds[0] + bounds[1]), train_length
                    )  # because we train we have enough data
                    - forecast_horizon  # because we have overlap_end = False
                    + 1  # because we include the first element
                )

                self.assertTrue(
                    len(forecasts) == theorical_forecast_length,
                    f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
                    "of retrain=True and overlap_end=False, and a time index of type DateTimeIndex.",
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
                    " of retrain=True, overlap_end=False, and a time index of type RangeIndex.",
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
                            72
                            - max(
                                (bounds[0] + bounds[1]), train_length
                            )  # because we train we have enough data
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
                    "of retrain=True and overlap_end=False and stride=2",
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
                            72
                            - max(
                                (bounds[0] + bounds[1]), train_length
                            )  # because we train we have enough data
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
                    "of retrain=True and overlap_end=False and stride=3",
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
                    72
                    - max(
                        (bounds[0] + bounds[1]), train_length
                    )  # because we train we have enough data
                    - forecast_horizon  # because we have overlap_end = False
                    + 1  # because we include the first element
                )

                self.assertTrue(
                    len(forecasts) == theorical_forecast_length,
                    f"Model {model_cls} does not return the right number of historical forecasts in the case of"
                    " retrain=True and overlap_end=False, and last_points_only=False",
                )

                self.assertTrue(
                    len(forecasts[0]) == forecast_horizon,
                    f"Model {model_cls} does not return forecast_horizon points per historical forecast in the case of"
                    " retrain=True and overlap_end=False, and last_points_only=False",
                )

        def test_regression_auto_start_multiple_no_cov(self):

            for model_cls, kwargs, bounds in models_reg_no_cov_cls_kwargs:
                model = model_cls(
                    **kwargs,
                )
                model.fit(self.ts_pass_train)

                forecasts = model.historical_forecasts(
                    series=[self.ts_pass_val, self.ts_pass_val],
                    forecast_horizon=10,
                    train_length=15,
                    stride=1,
                    retrain=True,
                    overlap_end=False,
                )

                self.assertTrue(
                    len(forecasts) == 2,
                    f"Model {model_cls} did not return a list of historical forecasts",
                )

                theorical_forecast_length = (
                    72
                    - max(
                        (bounds[0] + bounds[1]), 15
                    )  # because we train we have enough data
                    - 10  # because we have overlap_end = False
                    + 1  # because we include the first element
                )

                self.assertTrue(
                    len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length,
                    f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
                    "of retrain=True and overlap_end=False, and a time index of type DateTimeIndex.",
                )

        @unittest.skipUnless(
            TORCH_AVAILABLE,
            "Torch not available. auto start and multiple time series for torch models will be skipped.",
        )
        def test_torch_auto_start_multiple_no_cov(self):

            for model_cls, kwargs, bounds in models_torch_cls_kwargs:
                model = model_cls(
                    random_state=0,
                    **kwargs,
                )
                model.fit(self.ts_pass_train)

                # check historical forecasts for several time series,
                # retrain True and overlap_end False
                forecasts = model.historical_forecasts(
                    series=[self.ts_pass_val, self.ts_pass_val],
                    forecast_horizon=10,
                    stride=1,
                    retrain=True,
                    overlap_end=False,
                )
                self.assertTrue(
                    len(forecasts) == 2,
                    f"Model {model_cls} did not return a list of historical forecasts",
                )

                # If retrain=True and overlap_end=False, as ts has 72 values, we can only forecast
                # 72-(input_chunk_length+output_chunk_length+1) - 10 + 1
                # indeed we start to predict after the first trainable point (input_chunk_length+output_chunk_length+1)
                # and we stop in this case (overlap_end=False) at the end_time (-10),
                # and the predictable point is the index -10 instead of index -1, so we
                # have to add 1 to -10.
                # Let's note that output_chunk_length is necessarily 1 for RNN.
                self.assertTrue(
                    len(forecasts[0])
                    == len(forecasts[1])
                    == 72 - (bounds[0] + bounds[1] + 1) - 10 + 1,
                    f"Model {model_cls} does not return the right number of historical forecasts in the case of"
                    " retrain=True and overlap_end=False",
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
                    forecast_horizon=10,
                    stride=1,
                    retrain=True,
                    overlap_end=True,
                )

                self.assertTrue(
                    len(forecasts) == 2,
                    f"Model {model_cls} did not return a list of historical forecasts",
                )

                # If retrain=True and overlap_end=True, as ts has 72 values, we can only forecast
                # 72-(input_chunk_length+output_chunk_length+1)
                # We are not limited thanks to overlap_end=True
                self.assertTrue(
                    len(forecasts[0])
                    == len(forecasts[1])
                    == 72 - (bounds[0] + bounds[1] + 1)
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
                    forecast_horizon=10,
                    stride=1,
                    retrain=False,
                    overlap_end=False,
                )

                self.assertTrue(
                    len(forecasts) == 2,
                    f"Model {model_cls} did not return a list of historical forecasts",
                )

                # If retrain=False and overlap_end=False  as ts has 72 values, we can forecast
                # 72-input_chunk_length - 10 + 1 = 39 values
                self.assertTrue(
                    len(forecasts[0]) == len(forecasts[1]) == 72 - bounds[0] - 10 + 1
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
                    forecast_horizon=10,
                    stride=1,
                    retrain=False,
                    overlap_end=True,
                )

                self.assertTrue(
                    len(forecasts) == 2,
                    f"Model {model_cls} did not return a list of historical forecasts",
                )

                # If retrain=False and overlap_end=False  as ts has 72 values, we can forecast
                # 72-input_chunk_length
                # We are not limited thanks to overlap_end=True
                self.assertTrue(
                    len(forecasts[0]) == len(forecasts[1]) == 72 - bounds[0]
                )

        def test_regression_auto_start_multiple_with_cov(self):

            for model_cls, kwargs, bounds in models_reg_cov_cls_kwargs:
                model = model_cls(
                    random_state=0,
                    **kwargs,
                )

                forecasts = model.historical_forecasts(
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
                    forecast_horizon=10,
                    stride=1,
                    retrain=True,
                    overlap_end=False,
                )

                self.assertTrue(
                    len(forecasts) == 2,
                    f"Model {model_cls} did not return a list of historical forecasts",
                )

                self.assertTrue(
                    len(forecasts[0])
                    == len(forecasts[1])
                    == 72 - (bounds[0] + bounds[1] + 1) - 10 + 1,
                    f"Model {model_cls} does not return the right number of historical forecasts in the case of"
                    " retrain=True and overlap_end=False",
                )

        @unittest.skipUnless(
            TORCH_AVAILABLE,
            "Torch not available. auto start and multiple time series for torch models and covariates "
            "will be skipped.",
        )
        def test_torch_auto_start_with_cov(self):

            # # Torch models
            for model_cls, kwargs, bounds in models_torch_cls_kwargs:

                # RNN models don't have past_covariates
                if model_cls.__name__ == "RNNModel":
                    continue

                model = model_cls(
                    random_state=0,
                    **kwargs,
                )
                model.fit(self.ts_pass_train, self.ts_past_cov_train)

                # Only past covariate
                forecasts = model.historical_forecasts(
                    series=[self.ts_pass_val, self.ts_pass_val],
                    past_covariates=[
                        self.ts_past_cov_valid_same_start,
                        self.ts_past_cov_valid_same_start,
                    ],
                    forecast_horizon=10,
                    stride=1,
                    retrain=True,
                    overlap_end=False,
                )

                self.assertTrue(
                    len(forecasts) == 2,
                    f"Model {model_cls} did not return a list of historical forecasts",
                )

                self.assertTrue(
                    len(forecasts[0]) == 72 - (bounds[0] + bounds[1] + 1) - 10 + 1,
                    f"Model {model_cls} does not return the right number of historical forecasts in case "
                    " of retrain=True and overlap_end=False and past_covariates with different start",
                )

                model = model_cls(
                    random_state=0,
                    **kwargs,
                )
                model.fit(self.ts_pass_train, self.ts_past_cov_train)

                forecasts = model.historical_forecasts(
                    series=[self.ts_pass_val, self.ts_pass_val],
                    past_covariates=[
                        self.ts_past_cov_valid_aft_start,
                        self.ts_past_cov_valid_bef_start,
                    ],
                    forecast_horizon=10,
                    stride=1,
                    retrain=True,
                    overlap_end=False,
                )

                # we substract the shift of the past_cov_val ts (-5)
                self.assertTrue(
                    len(forecasts[0]) == 72 - (bounds[0] + bounds[1] + 1) - 10 + 1 - 5,
                    f"Model {model_cls} does not return the right number of historical forecasts in case "
                    " of retrain=True and overlap_end=False and past_covariates with different start",
                )

                self.assertTrue(
                    len(forecasts[1]) == 72 - (bounds[0] + bounds[1] + 1) - 10 + 1,
                    f"Model {model_cls} does not return the right number of historical forecasts in case "
                    " of retrain=True and overlap_end=False and past_covariates starting before.",
                )

            for model_cls, kwargs, bounds in models_torch_cls_kwargs:

                # no PastCovariates models
                if not model_cls.__name__ == "TFTModel":
                    continue

                model = model_cls(
                    random_state=0,
                    **kwargs,
                )
                model.fit(
                    self.ts_pass_train, self.ts_past_cov_train, self.ts_fut_cov_train
                )

                forecasts = model.historical_forecasts(
                    series=[self.ts_pass_val, self.ts_pass_val],
                    past_covariates=[
                        self.ts_past_cov_valid_aft_start,
                        self.ts_past_cov_valid_same_start,
                    ],
                    future_covariates=[
                        self.ts_fut_cov_valid_aft_start,
                        self.ts_fut_cov_valid_bef_start,
                    ],
                    forecast_horizon=10,
                    stride=1,
                    retrain=True,
                    overlap_end=False,
                )

                self.assertTrue(
                    len(forecasts[0]) == 72 - (bounds[0] + bounds[1] + 1) - 10 + 1 - 10,
                    f"Model {model_cls} does not return the right number of historical forecasts in case "
                    " of retrain=True and overlap_end=False and past_covariates and future_covariates with "
                    "different start",
                )

                self.assertTrue(
                    len(forecasts[1]) == 72 - (bounds[0] + bounds[1] + 1) - 10 + 1 - 3,
                    f"Model {model_cls} does not return the right number of historical forecasts in case "
                    " of retrain=True and overlap_end=False and past_covariates with different start",
                )
