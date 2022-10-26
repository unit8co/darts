import numpy as np
import pandas as pd

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
    import torch

    from darts.models import (  # RNNModel,
        BlockRNNModel,
        NBEATSModel,
        TCNModel,
        TFTModel,
        TransformerModel,
    )

    #   from darts.utils.likelihood_models import GaussianLikelihood

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not installed - will be skipping Torch models tests")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    IN_LEN = 24
    OUT_LEN = 12
    models_torch_cls_kwargs = [
        (
            BlockRNNModel,
            {
                "model": "RNN",
                "hidden_dim": 10,
                "n_rnn_layers": 1,
                "batch_size": 32,
                "n_epochs": 2,
            },
        ),
        # (
        #     RNNModel,
        #     {"model": "RNN", "hidden_dim": 10, "batch_size": 32, "n_epochs": 3},
        # ),
        # (
        #     RNNModel,
        #     {"training_length": 12, "n_epochs": 2, "likelihood": GaussianLikelihood()},
        # ),
        (TCNModel, {"n_epochs": 2, "batch_size": 32}),
        (
            TransformerModel,
            {
                "d_model": 16,
                "nhead": 2,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "dim_feedforward": 16,
                "batch_size": 32,
                "n_epochs": 2,
            },
        ),
        (
            NBEATSModel,
            {
                "num_stacks": 4,
                "num_blocks": 1,
                "num_layers": 2,
                "layer_widths": 12,
                "n_epochs": 2,
            },
        ),
        (
            TFTModel,
            {
                "hidden_size": 16,
                "lstm_layers": 1,
                "num_attention_heads": 4,
                "add_relative_index": True,
                "n_epochs": 2,
            },
        ),
    ]

    class HistoricalforecastTestCase(DartsBaseTestClass):

        forecasting_horizon = 12

        np.random.seed(42)
        torch.manual_seed(42)

        # some arbitrary static covariates
        static_covariates = pd.DataFrame([[0.0, 1.0]], columns=["st1", "st2"])

        # real timeseries for functionality tests
        ts_passengers = (
            AirPassengersDataset().load().with_static_covariates(static_covariates)
        )
        scaler = Scaler()
        ts_passengers = scaler.fit_transform(ts_passengers)
        ts_pass_train, ts_pass_val = ts_passengers[:-72], ts_passengers[-72:]

        # an additional noisy series
        ts_pass_train_1 = ts_pass_train + 0.01 * tg.gaussian_timeseries(
            length=len(ts_pass_train),
            freq=ts_pass_train.freq_str,
            start=ts_pass_train.start_time(),
        )

        def test_historical_forecasts_general(self):

            for model_cls, kwargs in models_torch_cls_kwargs:
                model = model_cls(
                    input_chunk_length=IN_LEN,
                    output_chunk_length=OUT_LEN,
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
                print(len(forecasts[0]))
                self.assertTrue(
                    len(forecasts) == 2,
                    f"Model {model_cls} did not return a list of historical forecasts",
                )

                # If retrain=True and overlap_end=False, as ts has 72 values, we can only forecast
                # 72-(24+12) - 10 + 1= 27 values
                # indeed we start to predict after the first trainable point (24+12)
                # and we stop in this case (overlap_end=False) at the end_time (-10),
                # and the predictable point is the index -10 instead of index -1, so we
                # have to add 1 to -10.
                self.assertTrue(
                    len(forecasts[0]) == len(forecasts[1]) == 27,
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
                # 72-(24+12) = 36 values
                # We are not limited thanks to overlap_end=True
                self.assertTrue(len(forecasts[0]) == len(forecasts[1]) == 36)

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
                # 72-24 - 10 + 1 = 39 values
                self.assertTrue(len(forecasts[0]) == len(forecasts[1]) == 39)

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

                # If retrain=False and overlap_end=False  as ts has 72 values, we can forecast 72-24 = 48 values
                # We are not limited thanks to overlap_end=True
                self.assertTrue(len(forecasts[0]) == len(forecasts[1]) == 48)
