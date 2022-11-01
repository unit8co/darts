import numpy as np

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.logging import get_logger
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
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg
from darts.utils.likelihood_models import GaussianLikelihood

# import pandas as pd


try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    logger = get_logger(__name__)
    logger.warning("Torch not installed - will be skipping Torch models tests")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    IN_LEN = 24
    OUT_LEN = 12

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
                "n_epochs": 2,
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
                "n_epochs": 3,
            },
            # autoregressive model
            (IN_LEN, 1),
        ),
        (
            RNNModel,
            {
                "input_chunk_length": IN_LEN,
                "training_length": 12,
                "n_epochs": 2,
                "likelihood": GaussianLikelihood(),
            },
            (IN_LEN, 1),
        ),
        (
            TCNModel,
            {
                "input_chunk_length": IN_LEN,
                "output_chunk_length": OUT_LEN,
                "n_epochs": 2,
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
                "n_epochs": 2,
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
                "n_epochs": 2,
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
                "n_epochs": 2,
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

        # same starting point
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

                self.assertTrue(
                    len(forecasts)
                    == 72
                    - (bounds[0] + bounds[1] + 1)
                    - forecast_horizon
                    + 1
                    - (train_length - 1),
                    f"Model {model_cls} does not return the right number of historical forecasts in the case of"
                    " retrain=True and overlap_end=False",
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
                    len(forecasts)
                    == 72
                    - (bounds[0] + bounds[1] + 1)
                    - forecast_horizon
                    + 1
                    - (train_length - 1),
                    f"Model {model_cls} does not return the right number of historical forecasts in the case of"
                    " retrain=True and overlap_end=False",
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

                # Here to adapt if forecast_horizon or train_length change
                self.assertTrue(
                    len(forecasts)
                    == (
                        72
                        - (bounds[0] + bounds[1] + 1)
                        - forecast_horizon
                        + 1
                        - (train_length - 1)
                    )
                    // 2
                    + 1,
                    f"Model {model_cls} does not return the right number of historical forecasts in the case of"
                    " retrain=True and overlap_end=False and stride=2",
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

                self.assertTrue(
                    len(forecasts)
                    == 72
                    - (bounds[0] + bounds[1] + 1)
                    - forecast_horizon
                    + 1
                    - (train_length - 1),
                    f"Model {model_cls} does not return the right number of historical forecasts in the case of"
                    " retrain=True and overlap_end=False, and last_points_only=False",
                )

                self.assertTrue(
                    len(forecasts[0]) == forecast_horizon,
                    f"Model {model_cls} does not return forecast_horizon points per historical forecast in the case of"
                    " retrain=True and overlap_end=False, and last_points_only=False",
                )

        def test_historical_forecasts_auto_start_multiple_no_cov(self):
            return
            # Regression models
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
                # If retrain=True and overlap_end=False, as ts has 72 values, we can only forecast
                # 72-(min target lag + output_chunk_length+1) - 10 + 1
                # indeed we start to predict after the first two trainable point (2 samples minimum for training)
                # (min target lag + output_chunk_length+1)
                # and we stop in this case (overlap_end=False) at the end_time (-10),
                # and the predictable point is the index -10 instead of index -1, so we
                # have to add 1 to -10.
                # Let's note that output_chunk_length is necessarily 1 for RNN.
                self.assertTrue(
                    len(forecasts[0])
                    == len(forecasts[1])
                    == 72 - (bounds[0] + bounds[1] + 1) - 10 + 1 - (15 - 1),
                    f"Model {model_cls} does not return the right number of historical forecasts in the case of"
                    " retrain=True and overlap_end=False",
                )

            # Bunch of torch models
            for model_cls, kwargs, bounds in models_torch_cls_kwargs:
                model = model_cls(
                    random_state=0,
                    **kwargs,
                )

                print(model_cls.__name__)
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

        # We test
        # - multiple time series
        # - with any combination of retrain and overlap_end
        # - with any combination of covariates and target
        # - with shifts between series an covariates in time
        def test_historical_forecasts_auto_start_multiple_with_cov(self):
            return
            for model_cls, kwargs, bounds in models_reg_cov_cls_kwargs:
                print(model_cls.__name__)
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

            # Torch models
            for model_cls, kwargs, bounds in models_torch_cls_kwargs:

                print(model_cls.__name__)

                model = model_cls(
                    random_state=0,
                    **kwargs,
                )

                # RNN models don't have past_covariates
                if model_cls.__name__ == "RNNModel":
                    continue
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

                forecasts = model.historical_forecasts(
                    series=[self.ts_pass_val, self.ts_pass_val],
                    past_covariates=[
                        self.ts_past_cov_valid_bef_start,
                        self.ts_past_cov_valid_same_start,
                    ],
                    forecast_horizon=10,
                    stride=1,
                    retrain=True,
                    overlap_end=False,
                )

                self.assertTrue(
                    len(forecasts[0]) == 72 - (bounds[0] + bounds[1] + 1) - 10 + 1,
                    f"Model {model_cls} does not return the right number of historical forecasts in case "
                    " of retrain=True and overlap_end=False and past_covariates with different start",
                )

                forecasts = model.historical_forecasts(
                    series=[self.ts_pass_val, self.ts_pass_val],
                    past_covariates=[
                        self.ts_past_cov_valid_aft_start,
                        self.ts_past_cov_valid_same_start,
                    ],
                    forecast_horizon=10,
                    stride=1,
                    retrain=True,
                    overlap_end=False,
                )

                # we substract the shift of the pasct_cov_val ts (-5)
                self.assertTrue(
                    len(forecasts[0]) == 72 - (bounds[0] + bounds[1] + 1) - 10 + 1 - 5,
                    f"Model {model_cls} does not return the right number of historical forecasts in case "
                    " of retrain=True and overlap_end=False and past_covariates with different start",
                )
