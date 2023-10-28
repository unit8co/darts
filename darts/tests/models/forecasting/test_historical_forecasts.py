import itertools

import numpy as np
import pandas as pd
import pytest

import darts
from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.logging import get_logger
from darts.models import (
    ARIMA,
    AutoARIMA,
    CatBoostModel,
    LightGBMModel,
    LinearRegressionModel,
    NaiveSeasonal,
    NotImportedModule,
)
from darts.tests.conftest import tfm_kwargs
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
        TiDEModel,
        TransformerModel,
    )
    from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression

    TORCH_AVAILABLE = True
except ImportError:
    logger = get_logger(__name__)
    logger.warning(
        "Torch not installed - will be skipping historical forecasts tests for torch models"
    )
    TORCH_AVAILABLE = False

models_reg_no_cov_cls_kwargs = [(LinearRegressionModel, {"lags": 8}, {}, (8, 1))]
if not isinstance(CatBoostModel, NotImportedModule):
    models_reg_no_cov_cls_kwargs.append(
        (CatBoostModel, {"lags": 6}, {"iterations": 1}, (6, 1))
    )
if not isinstance(LightGBMModel, NotImportedModule):
    models_reg_no_cov_cls_kwargs.append(
        (LightGBMModel, {"lags": 4}, {"n_estimators": 1}, (4, 1))
    )

models_reg_cov_cls_kwargs = [
    # target + past covariates
    (LinearRegressionModel, {"lags": 4, "lags_past_covariates": 6}, {}, (6, 1)),
    # target + past covariates + outputchunk > 3, 6 > 3
    (
        LinearRegressionModel,
        {"lags": 3, "lags_past_covariates": 6, "output_chunk_length": 5},
        {},
        (6, 5),
    ),
    # target + future covariates, 2 because to predict x, require x and x+1
    (LinearRegressionModel, {"lags": 4, "lags_future_covariates": [0, 1]}, {}, (4, 2)),
    # target + fut cov + output_chunk_length > 3,
    (
        LinearRegressionModel,
        {"lags": 2, "lags_future_covariates": [1, 2], "output_chunk_length": 5},
        {},
        (2, 5),
    ),
    # fut cov + output_chunk_length > 3, 5 > 2
    (
        LinearRegressionModel,
        {"lags_future_covariates": [0, 1], "output_chunk_length": 5},
        {},
        (0, 5),
    ),
    # past cov only
    (LinearRegressionModel, {"lags_past_covariates": 6}, {}, (6, 1)),
    # fut cov only
    (LinearRegressionModel, {"lags_future_covariates": [0, 1]}, {}, (0, 2)),
    # fut + past cov only
    (
        LinearRegressionModel,
        {"lags_past_covariates": 6, "lags_future_covariates": [0, 1]},
        {},
        (6, 2),
    ),
    # all
    (
        LinearRegressionModel,
        {"lags": 3, "lags_past_covariates": 6, "lags_future_covariates": [0, 1]},
        {},
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
                **tfm_kwargs,
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
                **tfm_kwargs,
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
                **tfm_kwargs,
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
                **tfm_kwargs,
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
                **tfm_kwargs,
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
                **tfm_kwargs,
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
                **tfm_kwargs,
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
                **tfm_kwargs,
            },
            (IN_LEN, OUT_LEN),
            "MixedCovariates",
        ),
        (
            TiDEModel,
            {
                "input_chunk_length": IN_LEN,
                "output_chunk_length": OUT_LEN,
                "n_epochs": NB_EPOCH,
                **tfm_kwargs,
            },
            (IN_LEN, OUT_LEN),
            "MixedCovariates",
        ),
    ]
else:
    models_torch_cls_kwargs = []


class TestHistoricalforecast:
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

    # optimized historical forecasts
    start_ts = pd.Timestamp("2000-01-01")
    ts_univariate = tg.linear_timeseries(
        start_value=1, end_value=100, length=20, start=start_ts
    )
    ts_multivariate = ts_univariate.stack(tg.sine_timeseries(length=20, start=start_ts))

    # slightly longer to not affect the last predictable timestamp
    ts_covs = tg.gaussian_timeseries(length=30, start=start_ts)

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

    def test_historical_forecasts_position_start(self):
        series = tg.sine_timeseries(length=10)

        model = LinearRegressionModel(lags=2)
        model.fit(series[:8])

        # negative index
        forecasts_neg = model.historical_forecasts(
            series=series, start=-2, start_format="position", retrain=False
        )
        assert len(forecasts_neg) == 2
        assert (series.time_index[-2:] == forecasts_neg.time_index).all()

        # positive index
        forecasts_pos = model.historical_forecasts(
            series=series, start=8, start_format="position", retrain=False
        )
        assert forecasts_pos == forecasts_neg

    def test_historical_forecasts_negative_rangeindex(self):
        series = TimeSeries.from_times_and_values(
            times=pd.RangeIndex(start=-5, stop=5, step=1), values=np.arange(10)
        )

        model = LinearRegressionModel(lags=2)
        model.fit(series[:8])

        # start as point
        forecasts = model.historical_forecasts(
            series=series, start=-2, start_format="value", retrain=False
        )
        assert len(forecasts) == 7
        assert (series.time_index[-7:] == forecasts.time_index).all()

        # start as index
        forecasts = model.historical_forecasts(
            series=series, start=-2, start_format="position", retrain=False
        )
        assert len(forecasts) == 2
        assert (series.time_index[-2:] == forecasts.time_index).all()

    @pytest.mark.parametrize("config", models_reg_no_cov_cls_kwargs)
    def test_historical_forecasts(self, config):
        train_length = 10
        forecast_horizon = 8
        # if no fit and retrain=false, should fit at fist iteration
        model_cls, kwargs, model_kwarg, bounds = config
        model = model_cls(**kwargs, **model_kwarg)

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

        assert len(forecasts) == theorical_forecast_length, (
            f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
            f"of retrain=True and overlap_end=False, and a time index of type DateTimeIndex. "
            f"Expected {theorical_forecast_length}, got {len(forecasts)}"
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

        assert len(forecasts) == theorical_forecast_length, (
            f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
            f"of retrain=True, overlap_end=False, and a time index of type RangeIndex."
            f"Expected {theorical_forecast_length}, got {len(forecasts)}"
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

        assert len(forecasts) == theorical_forecast_length, (
            f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
            f"of retrain=True and overlap_end=False and stride=2. "
            f"Expected {theorical_forecast_length}, got {len(forecasts)}"
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
        assert len(forecasts) == theorical_forecast_length, (
            f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
            f"of retrain=True and overlap_end=False and stride=3. "
            f"Expected {theorical_forecast_length}, got {len(forecasts)}"
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

        assert len(forecasts) == theorical_forecast_length, (
            f"Model {model_cls} does not return the right number of historical forecasts in the case of "
            f"retrain=True and overlap_end=False, and last_points_only=False. "
            f"expected {theorical_forecast_length}, got {len(forecasts)}"
        )

        assert len(forecasts[0]) == forecast_horizon, (
            f"Model {model_cls} does not return forecast_horizon points per historical forecast in the case of "
            f"retrain=True and overlap_end=False, and last_points_only=False"
        )

    def test_sanity_check_invalid_start(self):
        timeidx_ = tg.linear_timeseries(length=10)
        rangeidx_step1 = tg.linear_timeseries(start=0, length=10, freq=1)
        rangeidx_step2 = tg.linear_timeseries(start=0, length=10, freq=2)

        # label_index (int), too large
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(timeidx_, start=11)
        assert str(msg.value).startswith("`start` index `11` is out of bounds")
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(
                rangeidx_step1, start=rangeidx_step1.end_time() + rangeidx_step1.freq
            )
        assert str(msg.value).startswith(
            "`start` index `10` is larger than the last index"
        )
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(
                rangeidx_step2, start=rangeidx_step2.end_time() + rangeidx_step2.freq
            )
        assert str(msg.value).startswith(
            "`start` index `20` is larger than the last index"
        )

        # label_index (timestamp) too high
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(
                timeidx_, start=timeidx_.end_time() + timeidx_.freq
            )
        assert str(msg.value).startswith(
            "`start` time `2000-01-11 00:00:00` is after the last timestamp `2000-01-10 00:00:00`"
        )

        # label_index, invalid
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(rangeidx_step2, start=11)
        assert str(msg.value).startswith("The provided point is not a valid index")

        # label_index, too low
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
        assert str(msg.value).startswith(
            "`start` index `-1` is smaller than the first index `0`"
        )
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(
                rangeidx_step2, start=rangeidx_step2.start_time() - rangeidx_step2.freq
            )
        assert str(msg.value).startswith(
            "`start` index `-2` is smaller than the first index `0`"
        )

        # positional_index, predicting only the last position
        LinearRegressionModel(lags=1).historical_forecasts(
            timeidx_, start=9, start_format="position"
        )

        # positional_index, predicting from the first position with retrain=True
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(
                timeidx_, start=-10, start_format="position"
            )
        assert str(msg.value).endswith(", resulting in an empty training set.")

        # positional_index, beyond boundaries
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(
                timeidx_, start=10, start_format="position"
            )
        assert str(msg.value).startswith(
            "`start` index `10` is out of bounds for series of length 10"
        )
        with pytest.raises(ValueError) as msg:
            LinearRegressionModel(lags=1).historical_forecasts(
                timeidx_, start=-11, start_format="position"
            )
        assert str(msg.value).startswith(
            "`start` index `-11` is out of bounds for series of length 10"
        )

    @pytest.mark.parametrize("config", models_reg_no_cov_cls_kwargs)
    def test_regression_auto_start_multiple_no_cov(self, config):
        train_length = 15
        forecast_horizon = 10
        model_cls, kwargs, model_kwargs, bounds = config
        model = model_cls(
            **kwargs,
            **model_kwargs,
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

        assert (
            len(forecasts) == 2
        ), f"Model {model_cls} did not return a list of historical forecasts"

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

        assert len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length, (
            f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
            f"of retrain=True and overlap_end=False, and a time index of type DateTimeIndex. "
            f"Expected {theorical_forecast_length}, got {len(forecasts[0])} and {len(forecasts[1])}"
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [ts_univariate, ts_multivariate],
            models_reg_no_cov_cls_kwargs + models_reg_cov_cls_kwargs,
            [True, False],
            [1, 5],
        ),
    )
    def test_optimized_historical_forecasts_regression(self, config):
        ts, model_config, multi_models, forecast_horizon = config
        # slightly longer to not affect the last predictable timestamp
        ts_covs = self.ts_covs
        start = 14

        model_cls = LinearRegressionModel
        _, model_kwargs, _, _ = model_config
        # cover several covariates combinations and several regression models
        # ocl == forecast horizon
        model_kwargs_same = model_kwargs.copy()
        model_kwargs_same["output_chunk_length"] = forecast_horizon
        model_kwargs_same["multi_models"] = multi_models
        model_same = model_cls(**model_kwargs_same)
        model_same.fit(
            series=ts[:start],
            past_covariates=ts_covs if model_same.supports_past_covariates else None,
            future_covariates=ts_covs
            if model_same.supports_future_covariates
            else None,
        )
        # ocl >= forecast horizon
        model_kwargs_diff = model_kwargs.copy()
        model_kwargs_diff["output_chunk_length"] = 5
        model_kwargs_diff["multi_models"] = multi_models
        model_diff = model_cls(**model_kwargs_same)
        model_diff.fit(
            series=ts[:start],
            past_covariates=ts_covs if model_diff.supports_past_covariates else None,
            future_covariates=ts_covs
            if model_diff.supports_future_covariates
            else None,
        )
        # no parametrization to save time on model training at the cost of test granularity
        for model in [model_same, model_diff]:
            for last_points_only in [True, False]:
                for stride in [1, 2]:
                    hist_fct = model.historical_forecasts(
                        series=ts,
                        past_covariates=ts_covs
                        if model.supports_past_covariates
                        else None,
                        future_covariates=ts_covs
                        if model.supports_future_covariates
                        else None,
                        start=start,
                        retrain=False,
                        last_points_only=last_points_only,
                        stride=stride,
                        forecast_horizon=forecast_horizon,
                        enable_optimization=False,
                    )

                    # manually packing the series in list to match expected inputs
                    opti_hist_fct = model._optimized_historical_forecasts(
                        series=[ts],
                        past_covariates=[ts_covs]
                        if model.supports_past_covariates
                        else None,
                        future_covariates=[ts_covs]
                        if model.supports_future_covariates
                        else None,
                        start=start,
                        last_points_only=last_points_only,
                        stride=stride,
                        forecast_horizon=forecast_horizon,
                    )
                    # pack the output to generalize the tests
                    if last_points_only:
                        hist_fct = [hist_fct]
                        opti_hist_fct = [opti_hist_fct]

                    for fct, opti_fct in zip(hist_fct, opti_hist_fct):
                        assert (fct.time_index == opti_fct.time_index).all()
                        np.testing.assert_array_almost_equal(
                            fct.all_values(), opti_fct.all_values()
                        )

    @pytest.mark.parametrize(
        "config",
        list(
            itertools.product(
                [False, True],  # use covariates
                [True, False],  # last points only
                [False, True],  # overlap end
                [1, 3],  # stride
                [
                    3,  # horizon < ocl
                    5,  # horizon == ocl
                ],
                [True, False],  # multi models
            )
        ),
    )
    def test_optimized_historical_forecasts_regression_with_encoders(self, config):
        use_covs, last_points_only, overlap_end, stride, horizon, multi_models = config
        lags = 3
        ocl = 5
        len_val_series = 10 if multi_models else 10 + (ocl - 1)
        series_train, series_val = (
            self.ts_pass_train[:10],
            self.ts_pass_val[:len_val_series],
        )
        model = LinearRegressionModel(
            lags=lags,
            lags_past_covariates=2,
            lags_future_covariates=[2, 3],
            add_encoders={
                "cyclic": {"future": ["month"]},
                "datetime_attribute": {"past": ["dayofweek"]},
            },
            output_chunk_length=ocl,
            multi_models=multi_models,
        )
        if use_covs:
            pc = tg.gaussian_timeseries(
                start=series_train.start_time() - 2 * series_train.freq,
                end=series_val.end_time(),
                freq=series_train.freq,
            )
            fc = tg.gaussian_timeseries(
                start=series_train.start_time() + 3 * series_train.freq,
                end=series_val.end_time() + 4 * series_train.freq,
                freq=series_train.freq,
            )
        else:
            pc, fc = None, None

        model.fit(self.ts_pass_train, past_covariates=pc, future_covariates=fc)

        hist_fct = model.historical_forecasts(
            series=series_val,
            past_covariates=pc,
            future_covariates=fc,
            retrain=False,
            last_points_only=last_points_only,
            overlap_end=overlap_end,
            stride=stride,
            forecast_horizon=horizon,
            enable_optimization=False,
        )

        opti_hist_fct = model._optimized_historical_forecasts(
            series=[series_val],
            past_covariates=[pc],
            future_covariates=[fc],
            last_points_only=last_points_only,
            overlap_end=overlap_end,
            stride=stride,
            forecast_horizon=horizon,
        )

        if not isinstance(hist_fct, list):
            hist_fct = [hist_fct]
            opti_hist_fct = [opti_hist_fct]

        if not last_points_only and overlap_end:
            n_pred_series_expected = 8
            n_pred_points_expected = horizon
            first_ts_expected = series_val.time_index[lags]
            last_ts_expected = series_val.end_time() + series_val.freq * horizon
        elif not last_points_only:  # overlap_end = False
            n_pred_series_expected = len(series_val) - lags - horizon + 1
            n_pred_points_expected = horizon
            first_ts_expected = series_val.time_index[lags]
            last_ts_expected = series_val.end_time()
        elif overlap_end:  # last_points_only = True
            n_pred_series_expected = 1
            n_pred_points_expected = 8
            first_ts_expected = (
                series_val.time_index[lags] + (horizon - 1) * series_val.freq
            )
            last_ts_expected = series_val.end_time() + series_val.freq * horizon
        else:  # last_points_only = True, overlap_end = False
            n_pred_series_expected = 1
            n_pred_points_expected = len(series_val) - lags - horizon + 1
            first_ts_expected = (
                series_val.time_index[lags] + (horizon - 1) * series_val.freq
            )
            last_ts_expected = series_val.end_time()

        if not multi_models:
            first_ts_expected += series_val.freq * (ocl - 1)
            if not overlap_end:
                if not last_points_only:
                    n_pred_series_expected -= ocl - 1
                else:
                    n_pred_points_expected -= ocl - 1

        # to make it simple in case of stride, we assume that non-optimized hist fc returns correct results
        if stride > 1:
            n_pred_series_expected = len(hist_fct)
            n_pred_points_expected = len(hist_fct[0])
            first_ts_expected = hist_fct[0].start_time()
            last_ts_expected = hist_fct[-1].end_time()

        # check length match between optimized and default hist fc
        assert len(opti_hist_fct) == n_pred_series_expected
        assert len(hist_fct) == len(opti_hist_fct)
        # check hist fc start
        assert opti_hist_fct[0].start_time() == first_ts_expected
        # check hist fc end
        assert opti_hist_fct[-1].end_time() == last_ts_expected
        for hfc, ohfc in zip(hist_fct, opti_hist_fct):
            assert len(ohfc) == n_pred_points_expected
            assert (hfc.time_index == ohfc.time_index).all()
            np.testing.assert_array_almost_equal(hfc.all_values(), ohfc.all_values())

    def test_optimized_historical_forecasts_regression_with_component_specific_lags(
        self,
    ):
        horizon = 1
        lags = 3
        len_val_series = 10
        series_train, series_val = (
            self.ts_pass_train[:10],
            self.ts_pass_val[:len_val_series],
        )
        model = LinearRegressionModel(
            lags=lags,
            lags_past_covariates={"default_lags": 2, "darts_enc_pc_dta_dayofweek": 1},
            lags_future_covariates=[2, 3],
            add_encoders={
                "cyclic": {"future": ["month"]},
                "datetime_attribute": {"past": ["dayofweek"]},
            },
        )
        model.fit(series_train)
        hist_fct = model.historical_forecasts(
            series=series_val,
            retrain=False,
            enable_optimization=False,
        )

        opti_hist_fct = model._optimized_historical_forecasts(series=[series_val])

        if not isinstance(hist_fct, list):
            hist_fct = [hist_fct]
            opti_hist_fct = [opti_hist_fct]

        n_pred_series_expected = 1
        n_pred_points_expected = len(series_val) - lags - horizon + 1
        first_ts_expected = (
            series_val.time_index[lags] + (horizon - 1) * series_val.freq
        )
        last_ts_expected = series_val.end_time()

        # check length match between optimized and default hist fc
        assert len(opti_hist_fct) == n_pred_series_expected
        assert len(hist_fct) == len(opti_hist_fct)
        # check hist fc start
        assert opti_hist_fct[0].start_time() == first_ts_expected
        # check hist fc end
        assert opti_hist_fct[-1].end_time() == last_ts_expected
        for hfc, ohfc in zip(hist_fct, opti_hist_fct):
            assert len(ohfc) == n_pred_points_expected
            assert (hfc.time_index == ohfc.time_index).all()
            np.testing.assert_array_almost_equal(hfc.all_values(), ohfc.all_values())

    @pytest.mark.slow
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    @pytest.mark.parametrize(
        "config",
        list(
            itertools.product(
                [False, True],  # use covariates
                [True, False],  # last points only
                [False, True],  # overlap end
                [1, 3],  # stride
                [
                    3,  # horizon < ocl
                    5,  # horizon == ocl
                    7,  # horizon > ocl -> autoregression
                ],
                [False, True],  # use integer indexed series
                [False, True],  # use multi-series
            )
        ),
    )
    def test_optimized_historical_forecasts_torch_with_encoders(self, config):
        (
            use_covs,
            last_points_only,
            overlap_end,
            stride,
            horizon,
            use_int_idx,
            use_multi_series,
        ) = config
        icl = 3
        ocl = 5
        len_val_series = 10
        series_train, series_val = (
            self.ts_pass_train[:10],
            self.ts_pass_val[:len_val_series],
        )
        if use_int_idx:
            series_train = TimeSeries.from_values(
                series_train.all_values(), columns=series_train.columns
            )
            series_val = TimeSeries.from_times_and_values(
                values=series_val.all_values(),
                times=pd.RangeIndex(
                    start=series_train.end_time() + series_train.freq,
                    stop=series_train.end_time()
                    + (len(series_val) + 1) * series_train.freq,
                    step=series_train.freq,
                ),
                columns=series_train.columns,
            )

        def f_encoder(idx):
            return idx.month if not use_int_idx else idx

        model = NLinearModel(
            input_chunk_length=icl,
            add_encoders={
                "custom": {"past": [f_encoder], "future": [f_encoder]},
            },
            output_chunk_length=ocl,
            n_epochs=1,
            **tfm_kwargs,
        )
        if use_covs:
            pc = tg.gaussian_timeseries(
                start=series_train.start_time(),
                end=series_val.end_time() + max(0, horizon - ocl) * series_train.freq,
                freq=series_train.freq,
            )
            fc = tg.gaussian_timeseries(
                start=series_train.start_time(),
                end=series_val.end_time() + max(ocl, horizon) * series_train.freq,
                freq=series_train.freq,
            )
        else:
            pc, fc = None, None

        model.fit(series_train, past_covariates=pc, future_covariates=fc)

        if use_multi_series:
            series_val = [
                series_val,
                (series_val + 10)
                .shift(1)
                .with_columns_renamed(series_val.columns, "test_col"),
            ]
            pc = [pc, pc.shift(1)] if pc is not None else None
            fc = [fc, fc.shift(1)] if fc is not None else None

        hist_fct = model.historical_forecasts(
            series=series_val,
            past_covariates=pc,
            future_covariates=fc,
            retrain=False,
            last_points_only=last_points_only,
            overlap_end=overlap_end,
            stride=stride,
            forecast_horizon=horizon,
            enable_optimization=False,
        )

        opti_hist_fct = model._optimized_historical_forecasts(
            series=series_val if isinstance(series_val, list) else [series_val],
            past_covariates=pc if (isinstance(pc, list) or pc is None) else [pc],
            future_covariates=fc if (isinstance(fc, list) or fc is None) else [fc],
            last_points_only=last_points_only,
            overlap_end=overlap_end,
            stride=stride,
            forecast_horizon=horizon,
        )

        if not isinstance(series_val, list):
            series_val = [series_val]
            hist_fct = [hist_fct]
            opti_hist_fct = [opti_hist_fct]

        for series, hfc, ohfc in zip(series_val, hist_fct, opti_hist_fct):
            if not isinstance(hfc, list):
                hfc = [hfc]
                ohfc = [ohfc]

            if not last_points_only and overlap_end:
                n_pred_series_expected = 8
                n_pred_points_expected = horizon
                first_ts_expected = series.time_index[icl]
                last_ts_expected = series.end_time() + series.freq * horizon
            elif not last_points_only:  # overlap_end = False
                n_pred_series_expected = len(series) - icl - horizon + 1
                n_pred_points_expected = horizon
                first_ts_expected = series.time_index[icl]
                last_ts_expected = series.end_time()
            elif overlap_end:  # last_points_only = True
                n_pred_series_expected = 1
                n_pred_points_expected = 8
                first_ts_expected = series.time_index[icl] + (horizon - 1) * series.freq
                last_ts_expected = series.end_time() + series.freq * horizon
            else:  # last_points_only = True, overlap_end = False
                n_pred_series_expected = 1
                n_pred_points_expected = len(series) - icl - horizon + 1
                first_ts_expected = series.time_index[icl] + (horizon - 1) * series.freq
                last_ts_expected = series.end_time()

            # to make it simple in case of stride, we assume that non-optimized hist fc returns correct results
            if stride > 1:
                n_pred_series_expected = len(hfc)
                n_pred_points_expected = len(hfc[0])
                first_ts_expected = hfc[0].start_time()
                last_ts_expected = hfc[-1].end_time()

            # check length match between optimized and default hist fc
            assert len(ohfc) == n_pred_series_expected
            assert len(hfc) == len(ohfc)
            # check hist fc start
            assert ohfc[0].start_time() == first_ts_expected
            # check hist fc end
            assert ohfc[-1].end_time() == last_ts_expected
            for hfc, ohfc in zip(hfc, ohfc):
                assert hfc.columns.equals(series.columns)
                assert ohfc.columns.equals(series.columns)
                assert len(ohfc) == n_pred_points_expected
                assert (hfc.time_index == ohfc.time_index).all()
                np.testing.assert_array_almost_equal(
                    hfc.all_values(), ohfc.all_values()
                )

    @pytest.mark.slow
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    @pytest.mark.parametrize("model_config", models_torch_cls_kwargs)
    def test_torch_auto_start_multiple_no_cov(self, model_config):
        forecast_hrz = 10
        model_cls, kwargs, bounds, _ = model_config
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
        assert (
            len(forecasts) == 2
        ), f"Model {model_cls} did not return a list of historical forecasts"
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
        assert len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length, (
            f"Model {model_cls} does not return the right number of historical forecasts in the case of "
            f"retrain=True and overlap_end=False. "
            f"Expected {theorical_forecast_length}, got {len(forecasts[0])} and {len(forecasts[1])}"
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

        assert (
            len(forecasts) == 2
        ), f"Model {model_cls} did not return a list of historical forecasts"
        theorical_forecast_length = (
            self.ts_val_length
            - (bounds[0] + bounds[1])  # train sample length
            + 1  # with overlap_end=True, we are not restricted by the end of the series or horizon
        )
        assert len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length

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

        assert (
            len(forecasts) == 2
        ), f"Model {model_cls} did not return a list of historical forecasts"
        theorical_forecast_length = (
            self.ts_val_length
            - bounds[0]  # prediction input sample length
            - (
                forecast_hrz - 1
            )  # overlap_end=False -> if entire horizon is available, we can predict 1
        )
        assert len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length
        assert (
            forecasts[0].end_time()
            == forecasts[1].end_time()
            == self.ts_pass_val.end_time()
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

        assert (
            len(forecasts) == 2
        ), f"Model {model_cls} did not return a list of historical forecasts"
        theorical_forecast_length = (
            self.ts_val_length
            - bounds[0]  # prediction input sample length
            + 1  # overlap_end=True -> last possible prediction start is one step after end of target
        )
        assert len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length
        assert (
            forecasts[0].end_time()
            == forecasts[1].end_time()
            == self.ts_pass_val.end_time() + forecast_hrz * self.ts_pass_val.freq
        )

    def test_hist_fc_end_exact_with_covs(self):
        model = LinearRegressionModel(
            lags=2,
            lags_past_covariates=2,
            lags_future_covariates=(2, 1),
            output_chunk_length=2,
        )
        series = tg.sine_timeseries(length=10)
        model.fit(series, past_covariates=series, future_covariates=series)
        fc = model.historical_forecasts(
            series,
            past_covariates=series[:-2],
            future_covariates=series,
            forecast_horizon=2,
            stride=2,
            overlap_end=False,
            last_points_only=True,
            retrain=False,
        )
        assert len(fc) == 4
        assert fc.end_time() == series.end_time()

        fc = model.historical_forecasts(
            series,
            past_covariates=series[:-2],
            future_covariates=series,
            forecast_horizon=2,
            stride=2,
            overlap_end=False,
            last_points_only=False,
            retrain=False,
        )
        fc = concatenate(fc)
        assert len(fc) == 8
        assert fc.end_time() == series.end_time()

    @pytest.mark.parametrize("model_config", models_reg_cov_cls_kwargs)
    def test_regression_auto_start_multiple_with_cov_retrain(self, model_config):
        forecast_hrz = 10
        model_cls, kwargs, _, bounds = model_config
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

        assert (
            len(forecasts_retrain) == 2
        ), f"Model {model_cls} did not return a list of historical forecasts"

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
        # length input - largest past lag - forecast horizon - max(largest future lag, output_chunk_length)
        theorical_retrain_forecast_length = len(self.ts_pass_val) - (
            -past_lag
            + forecast_hrz
            + max(future_lag + 1, kwargs.get("output_chunk_length", 1))
        )
        assert (
            len(forecasts_retrain[0])
            == len(forecasts_retrain[1])
            == theorical_retrain_forecast_length
        ), (
            f"Model {model_cls} does not return the right number of historical forecasts in the case of "
            f"retrain=True and overlap_end=False. "
            f"Expected {theorical_retrain_forecast_length}, got {len(forecasts_retrain[0])} "
            f"and {len(forecasts_retrain[1])}"
        )

        # with last_points_only=True: start is shifted by biggest past lag + training timestamps
        # (forecast horizon + output_chunk_length)
        expected_start = (
            self.ts_pass_val.start_time()
            + (-past_lag + forecast_hrz + kwargs.get("output_chunk_length", 1))
            * self.ts_pass_val.freq
        )
        assert forecasts_retrain[0].start_time() == expected_start

        # end is shifted back by the biggest future lag
        if model.output_chunk_length - 1 > future_lag:
            shift = 0
        else:
            shift = future_lag
        expected_end = self.ts_pass_val.end_time() - shift * self.ts_pass_val.freq
        assert forecasts_retrain[0].end_time() == expected_end

    @pytest.mark.parametrize("model_config", models_reg_cov_cls_kwargs)
    def test_regression_auto_start_multiple_with_cov_no_retrain(self, model_config):
        forecast_hrz = 10
        model_cls, kwargs, _, bounds = model_config
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
        assert forecasts_no_retrain[0].start_time() == expected_start

        # end is shifted by the biggest future lag if future lag > output_chunk_length
        shift_back = future_lag if future_lag + 1 > model.output_chunk_length else 0
        expected_end = self.ts_pass_val.end_time() - shift_back * self.ts_pass_val.freq
        assert forecasts_no_retrain[0].end_time() == expected_end

    @pytest.mark.slow
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    @pytest.mark.parametrize("model_config", models_torch_cls_kwargs)
    def test_torch_auto_start_with_past_cov(self, model_config):
        forecast_hrz = 10
        # Past covariates only
        model_cls, kwargs, bounds, type = model_config
        if type == "DualCovariates":
            return

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

        assert (
            len(forecasts) == 2
        ), f"Model {model_cls} did not return a list of historical forecasts"

        theorical_forecast_length = (
            self.ts_val_length
            - (bounds[0] + bounds[1])  # train sample length
            - (forecast_hrz - 1)  # if entire horizon is available, we can predict 1
            - 0  # past covs have same start as target -> no shift
            - 0  # we don't have future covs in output chunk -> no shift
        )
        assert len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length, (
            f"Model {model_cls} does not return the right number of historical forecasts in case "
            f"of retrain=True and overlap_end=False and past_covariates with same start. "
            f"Expected {theorical_forecast_length}, got {len(forecasts[0])} and {len(forecasts[1])}"
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
        assert len(forecasts[0]) == theorical_forecast_length, (
            f"Model {model_cls} does not return the right number of historical forecasts in case "
            f"of retrain=True and overlap_end=False and past_covariates starting after. "
            f"Expected {theorical_forecast_length}, got {len(forecasts[0])}"
        )
        theorical_forecast_length = (
            self.ts_val_length
            - (bounds[0] + bounds[1])  # train sample length
            - (forecast_hrz - 1)  # if entire horizon is available, we can predict 1
            - 0  # past covs have same start as target -> no shift
            - 0  # we don't have future covs in output chunk -> no shift
        )
        assert len(forecasts[1]) == theorical_forecast_length, (
            f"Model {model_cls} does not return the right number of historical forecasts in case "
            f"of retrain=True and overlap_end=False and past_covariates starting before. "
            f"Expected {theorical_forecast_length}, got {len(forecasts[1])}"
        )

    @pytest.mark.slow
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    @pytest.mark.parametrize("model_config", models_torch_cls_kwargs)
    def test_torch_auto_start_with_past_future_cov(self, model_config):
        forecast_hrz = 10
        # Past and future covariates
        for model_cls, kwargs, bounds, type in models_torch_cls_kwargs:
            if not type == "MixedCovariates":
                return

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
            assert len(forecasts[0]) == theorical_forecast_length, (
                f"Model {model_cls} does not return the right number of historical forecasts in case "
                f"of retrain=True and overlap_end=False and past_covariates and future_covariates with "
                f"different start. "
                f"Expected {theorical_forecast_length}, got {len(forecasts[0])}"
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
            assert len(forecasts[1]) == theorical_forecast_length, (
                f"Model {model_cls} does not return the right number of historical forecasts in case "
                f"of retrain=True and overlap_end=False and past_covariates with different start. "
                f"Expected {theorical_forecast_length}, got {len(forecasts[1])}"
            )

    @pytest.mark.slow
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    @pytest.mark.parametrize("model_config", models_torch_cls_kwargs)
    def test_torch_auto_start_with_future_cov(self, model_config):
        forecast_hrz = 10
        # Future covariates only
        for model_cls, kwargs, bounds, type in models_torch_cls_kwargs:
            # todo case of DualCovariates (RNN)
            if type == "PastCovariates" or type == "DualCovariates":
                return

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

            assert (
                len(forecasts) == 2
            ), f"Model {model_cls} did not return a list of historical forecasts"
            theorical_forecast_length = (
                self.ts_val_length
                - (bounds[0] + bounds[1])  # train sample length
                - (
                    forecast_hrz - 1
                )  # (horizon - 1): if entire horizon is available, we can predict 1,
                - 7  # future covs start 7 after target (more than past covs) -> shift
                - 2  # future covs in output chunk -> difference between horizon=10 and output_chunk_length=12
            )
            assert len(forecasts[0]) == theorical_forecast_length, (
                f"Model {model_cls} does not return the right number of historical forecasts in case "
                f"of retrain=True and overlap_end=False and no past_covariates and future_covariates "
                f"with different start. "
                f"Expected {theorical_forecast_length}, got {len(forecasts[0])}"
            )
            theorical_forecast_length = (
                self.ts_val_length
                - (bounds[0] + bounds[1])  # train sample length
                - (forecast_hrz - 1)  # if entire horizon is available, we can predict 1
                - 0  # all covs start at the same time as target -> no shift
                - 2  # future covs in output chunk -> difference between horizon=10 and output_chunk_length=12
            )
            assert len(forecasts[1]) == theorical_forecast_length, (
                f"Model {model_cls} does not return the right number of historical forecasts in case "
                f"of retrain=True and overlap_end=False and no past_covariates and future_covariates "
                f"with different start. "
                f"Expected {theorical_forecast_length}, got {len(forecasts[1])}"
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
        assert str(error_msg.value).startswith(expected_msg)
        # returning a non-bool value (int)
        expected_msg = "Return value of `retrain` must be bool, received <class 'int'>"
        with pytest.raises(ValueError) as error_msg:
            helper_hist_forecasts(retrain_f_invalid_ouput_int, 0.9)
        assert str(error_msg.value).startswith(expected_msg)
        # returning a non-bool value (str)
        expected_msg = "Return value of `retrain` must be bool, received <class 'str'>"
        with pytest.raises(ValueError) as error_msg:
            helper_hist_forecasts(retrain_f_invalid_ouput_str, 0.9)
        assert str(error_msg.value).startswith(expected_msg)
        # predict fails but model could have been trained before the predict round
        expected_msg = "`retrain` is `False` in the first train iteration at prediction point (in time)"
        with pytest.raises(ValueError) as error_msg:
            helper_hist_forecasts(retrain_f_delayed_true, 0.9)
        assert str(error_msg.value).startswith(expected_msg)
        # always returns False, treated slightly different than `retrain=False` and `retrain=0`
        with pytest.raises(ValueError) as error_msg:
            helper_hist_forecasts(retrain_f_invalid, 0.9)
        assert str(error_msg.value).startswith(expected_msg)

        # test int
        helper_hist_forecasts(10, 0.9)
        expected_msg = "Model has not been fit yet."
        # `retrain=0` with not-trained model, encountering directly a predictable time index
        with pytest.raises(ValueError) as error_msg:
            helper_hist_forecasts(0, 0.9)
        assert str(error_msg.value).startswith(expected_msg), str(error_msg.value)

        # test bool
        helper_hist_forecasts(True, 0.9)
        # `retrain=False` with not-trained model, encountering directly a predictable time index
        expected_msg = "The model has not been fitted yet, and `retrain` is ``False``."
        with pytest.raises(ValueError) as error_msg:
            helper_hist_forecasts(False, 0.9)
        assert str(error_msg.value).startswith(expected_msg)

        expected_start = pd.Timestamp("1949-10-01 00:00:00")
        # start before first trainable time index should still work
        res = helper_hist_forecasts(True, pd.Timestamp("1949-09-01 00:00:00"))
        assert res.time_index[0] == expected_start
        # start at first trainable time index should still work
        res = helper_hist_forecasts(True, expected_start)
        assert res.time_index[0] == expected_start
        # start at last trainable time index should still work
        expected_end = pd.Timestamp("1960-12-01 00:00:00")
        res = helper_hist_forecasts(True, expected_end)
        assert res.time_index[0] == expected_end

    @pytest.mark.parametrize("model_type", ["regression", "torch"])
    def test_predict_likelihood_parameters(self, model_type):
        """standard checks that historical forecasts work with direct likelihood parameter predictions
        with regression and torch models."""

        def create_model(ocl, use_ll=True, model_type="regression"):
            if model_type == "regression":
                return LinearRegressionModel(
                    lags=3,
                    likelihood="quantile" if use_ll else None,
                    quantiles=[0.05, 0.4, 0.5, 0.6, 0.95] if use_ll else None,
                    output_chunk_length=ocl,
                )
            else:  # model_type == "torch"
                if not TORCH_AVAILABLE:
                    return None
                return NLinearModel(
                    input_chunk_length=3,
                    likelihood=QuantileRegression([0.05, 0.4, 0.5, 0.6, 0.95])
                    if use_ll
                    else None,
                    output_chunk_length=ocl,
                    n_epochs=1,
                    random_state=42,
                    **tfm_kwargs,
                )

        model = create_model(1, False, model_type=model_type)
        # skip torch models if not installed
        if model is None:
            return
        # model doesn't use likelihood
        with pytest.raises(ValueError):
            model.historical_forecasts(
                self.ts_pass_train,
                predict_likelihood_parameters=True,
            )

        model = create_model(1, model_type=model_type)
        # forecast_horizon > output_chunk_length doesn't work
        with pytest.raises(ValueError):
            model.historical_forecasts(
                self.ts_pass_train,
                predict_likelihood_parameters=True,
                forecast_horizon=2,
            )

        model = create_model(1, model_type=model_type)
        # num_samples != 1 doesn't work
        with pytest.raises(ValueError):
            model.historical_forecasts(
                self.ts_pass_train,
                predict_likelihood_parameters=True,
                forecast_horizon=1,
                num_samples=2,
            )

        n = 3
        target_name = self.ts_pass_train.components[0]
        qs_expected = ["q0.05", "q0.40", "q0.50", "q0.60", "q0.95"]
        qs_expected = pd.Index([target_name + "_" + q for q in qs_expected])
        # check that it works with retrain
        model = create_model(1, model_type=model_type)
        hist_fc = model.historical_forecasts(
            self.ts_pass_train,
            predict_likelihood_parameters=True,
            forecast_horizon=1,
            num_samples=1,
            start=len(self.ts_pass_train) - n,  # predict on last 10 steps
            retrain=True,
        )
        assert hist_fc.components.equals(qs_expected)
        assert len(hist_fc) == n

        # check for equal results between predict and hist fc without retraining
        model = create_model(1, model_type=model_type)
        model.fit(series=self.ts_pass_train[:-n])
        hist_fc = model.historical_forecasts(
            self.ts_pass_train,
            predict_likelihood_parameters=True,
            forecast_horizon=1,
            num_samples=1,
            start=len(self.ts_pass_train) - n,  # predict on last 10 steps
            retrain=False,
        )
        assert hist_fc.components.equals(qs_expected)
        assert len(hist_fc) == n

        preds = []
        for n_i in range(n):
            preds.append(
                model.predict(
                    n=1,
                    series=self.ts_pass_train[: -(n - n_i)],
                    predict_likelihood_parameters=True,
                )
            )
        preds = darts.concatenate(preds)
        np.testing.assert_array_almost_equal(
            preds.all_values(copy=False), hist_fc.all_values(copy=False)
        )

        # check equal results between predict and hist fc with higher output_chunk_length and horizon,
        # and last_points_only=False
        model = create_model(2, model_type=model_type)
        # we take one more training step so that model trained on ocl=1 has the same training samples
        # as model above
        model.fit(series=self.ts_pass_train[: -(n - 1)])
        hist_fc = model.historical_forecasts(
            self.ts_pass_train,
            predict_likelihood_parameters=True,
            forecast_horizon=2,
            num_samples=1,
            start=len(self.ts_pass_train) - n,  # predict on last 10 steps
            retrain=False,
            last_points_only=False,
            overlap_end=True,
        )
        # because of overlap_end, we get an additional prediction
        # generate the same predictions manually
        preds = []
        for n_i in range(n + 1):
            right = -(n - n_i) if n_i < 3 else len(self.ts_pass_train)
            preds.append(
                model.predict(
                    n=2,
                    series=self.ts_pass_train[:right],
                    predict_likelihood_parameters=True,
                )
            )
        for p, hfc in zip(preds, hist_fc):
            assert p.columns.equals(hfc.columns)
            assert p.time_index.equals(hfc.time_index)
            np.testing.assert_array_almost_equal(
                p.all_values(copy=False), hfc.all_values(copy=False)
            )
            assert len(hist_fc) == n + 1
