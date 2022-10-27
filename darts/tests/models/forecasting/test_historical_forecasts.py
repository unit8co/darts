import numpy as np

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

# import pandas as pd


logger = get_logger(__name__)

try:
    import torch

    from darts.models import (
        BlockRNNModel,
        LightGBMModel,
        NBEATSModel,
        RNNModel,
        TCNModel,
        TFTModel,
        TransformerModel,
    )
    from darts.utils.likelihood_models import GaussianLikelihood

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not installed - will be skipping Torch models tests")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    IN_LEN = 24
    OUT_LEN = 12

    models_reg_no_cov_cls_kwargs = [
        #      (CatBoostModel, {"lags": 4}, (4, 1)),
        (LightGBMModel, {"lags": 4}, (4, 1)),
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

        def test_historical_forecasts_auto_start_multiple_no_cov(self):

            # Regression models
            for model_cls, kwargs, bounds in models_reg_no_cov_cls_kwargs:
                model = model_cls(
                    **kwargs,
                )
                model.fit(self.ts_pass_train)

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
                    == 72 - (bounds[0] + bounds[1] + 1) - 10 + 1,
                    f"Model {model_cls} does not return the right number of historical forecasts in the case of"
                    " retrain=True and overlap_end=False",
                )

            # Bunch of torch models
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

                print("test 1 " + str(len(forecasts[0])))

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

        def test_historical_forecasts_auto_start_multiple_with_cov(self):

            # Torch models
            for model_cls, kwargs, bounds in models_torch_cls_kwargs:

                print(model_cls)
                model = model_cls(
                    random_state=0,
                    **kwargs,
                )

                # RNN models don't have past_covariates
                if model_cls.__name__ == "RNNModel":
                    continue
                model.fit(self.ts_pass_train, self.ts_past_cov_train)

                # # Only past covariate
                # forecasts = model.historical_forecasts(
                #     series=[self.ts_pass_val, self.ts_pass_val],
                #     past_covariates=[
                #         self.ts_past_cov_valid_same_start,
                #         self.ts_past_cov_valid_same_start,
                #     ],
                #     forecast_horizon=10,
                #     stride=1,
                #     retrain=True,
                #     overlap_end=False,
                # )

                # print(len(forecasts[0]))
                # self.assertTrue(
                #     len(forecasts) == 2,
                #     f"Model {model_cls} did not return a list of historical forecasts",
                # )

                # self.assertTrue(
                #     len(forecasts[0]) == len(forecasts[1]) == 27,
                #     f"Model {model_cls} does not return the right number of historical forecasts in the case of"
                #     " retrain=True and overlap_end=False",
                # )

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

                print(len(forecasts[0]))
                self.assertTrue(
                    len(forecasts[0]) == 72 - (bounds[0] + bounds[1] + 1) - 10 + 1,
                    f"Model {model_cls} does not return the right number of historical forecasts in case "
                    " of retrain=True and overlap_end=False and past_covariates with different start",
                )
