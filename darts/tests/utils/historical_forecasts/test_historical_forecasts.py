import itertools
import logging
import math
from copy import deepcopy
from itertools import product
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MaxAbsScaler

import darts
from darts import TimeSeries, concatenate
from darts.dataprocessing.pipeline import Pipeline
from darts.dataprocessing.transformers import (
    FittableDataTransformer,
    InvertibleDataTransformer,
    Scaler,
)
from darts.datasets import AirPassengersDataset
from darts.models import (
    ARIMA,
    AutoARIMA,
    CatBoostModel,
    ConformalNaiveModel,
    LightGBMModel,
    LinearRegressionModel,
    NaiveDrift,
    NaiveSeasonal,
    NotImportedModule,
)
from darts.models.forecasting.forecasting_model import (
    LocalForecastingModel,
)
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils import n_steps_between
from darts.utils import timeseries_generation as tg
from darts.utils.ts_utils import SeriesType, get_series_seq_type
from darts.utils.utils import likelihood_component_names, quantile_names

if TORCH_AVAILABLE:
    import torch

    from darts.models import (
        BlockRNNModel,
        GlobalNaiveAggregate,
        GlobalNaiveDrift,
        GlobalNaiveSeasonal,
        NBEATSModel,
        NLinearModel,
        RNNModel,
        TCNModel,
        TFTModel,
        TiDEModel,
        TransformerModel,
        TSMixerModel,
    )
    from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression

models = [LinearRegressionModel, NaiveDrift]
models_reg_no_cov_cls_kwargs = [
    (LinearRegressionModel, {"lags": 8}, {}, (8, 1)),
    # output_chunk_length only
    (LinearRegressionModel, {"lags": 5, "output_chunk_length": 2}, {}, (5, 2)),
    # output_chunk_shift only
    (LinearRegressionModel, {"lags": 5, "output_chunk_shift": 1}, {}, (5, 2)),
    # output_chunk_shift + output_chunk_length only
    (
        LinearRegressionModel,
        {"lags": 5, "output_chunk_shift": 1, "output_chunk_length": 2},
        {},
        (5, 3),
    ),
    (LinearRegressionModel, {"lags": [-5]}, {}, (5, 1)),
    (LinearRegressionModel, {"lags": [-5], "output_chunk_shift": 1}, {}, (5, 2)),
]
if not isinstance(CatBoostModel, NotImportedModule):
    models_reg_no_cov_cls_kwargs.append((
        CatBoostModel,
        {"lags": 6},
        {"iterations": 1},
        (6, 1),
    ))
if not isinstance(LightGBMModel, NotImportedModule):
    models_reg_no_cov_cls_kwargs.append((
        LightGBMModel,
        {"lags": 4},
        {"n_estimators": 1},
        (4, 1),
    ))

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

    models += [NLinearModel]

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
                "training_length": IN_LEN + OUT_LEN - 1,
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
                "training_length": IN_LEN + OUT_LEN - 1,
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
        (
            TSMixerModel,
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
            GlobalNaiveAggregate,
            {
                "input_chunk_length": IN_LEN,
                "output_chunk_length": OUT_LEN,
                **tfm_kwargs,
            },
            (IN_LEN, OUT_LEN),
            "MixedCovariates",
        ),
        (
            GlobalNaiveDrift,
            {
                "input_chunk_length": IN_LEN,
                "output_chunk_length": OUT_LEN,
                **tfm_kwargs,
            },
            (IN_LEN, OUT_LEN),
            "MixedCovariates",
        ),
        (
            GlobalNaiveSeasonal,
            {
                "input_chunk_length": IN_LEN,
                "output_chunk_length": OUT_LEN,
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

    #
    sine_univariate1 = tg.sine_timeseries(length=50) * 2 + 1.5
    sine_univariate2 = tg.sine_timeseries(length=50, value_phase=1.5705) * 5 + 1.5
    sine_univariate3 = tg.sine_timeseries(length=50, value_phase=0.1963125) * -9 + 1.5

    @staticmethod
    def create_model(ocl, use_ll=True, model_type="regression", n_epochs=1, **kwargs):
        if model_type == "regression":
            return LinearRegressionModel(
                lags=3,
                likelihood="quantile" if use_ll else None,
                quantiles=[0.05, 0.4, 0.5, 0.6, 0.95] if use_ll else None,
                output_chunk_length=ocl,
                **kwargs,
            )
        else:  # model_type == "torch"
            if not TORCH_AVAILABLE:
                return None
            return NLinearModel(
                input_chunk_length=3,
                likelihood=(
                    QuantileRegression([0.05, 0.4, 0.5, 0.6, 0.95]) if use_ll else None
                ),
                loss_fn=torch.nn.MSELoss() if not use_ll else None,
                output_chunk_length=ocl,
                n_epochs=n_epochs,
                random_state=42,
                **tfm_kwargs,
                **kwargs,
            )

    @pytest.mark.parametrize(
        "config",
        list(
            itertools.product(
                [True, False],
                [0, 1, 3],
                [0, 1, 2],
            )
        ),
    )
    def test_historical_forecasts_output(self, config):
        """Tests historical forecasts output type and values for all combinations of:

        - uni or multivariate `series`
        - different number of `series`, `0` represents a single `TimeSeries`,
          `1` a list of one `TimeSeries`, and so on.
        - different number of expected forecasts.
        """
        is_univariate, series_list_length, n_fc_expected = config

        model = NaiveDrift()
        horizon = 7
        ts_length = horizon + model.min_train_series_length + (n_fc_expected - 1)

        y = tg.constant_timeseries(value=1.0, length=ts_length)
        if not is_univariate:
            y = y.stack(y + 1.0)
        # remember `y` for expected output
        y_ref = y

        if series_list_length:
            y = [y] * series_list_length

        if not n_fc_expected:
            # cannot generate a single forecast
            with pytest.raises(ValueError) as err:
                _ = model.historical_forecasts(
                    series=y, forecast_horizon=horizon, last_points_only=True
                )
            assert str(err.value).startswith(
                "Cannot build a single input for prediction"
            )
            return

        # last_points_only = True: gives a list with a single forecasts per series,
        # where each forecast contains only the last points of all possible historical
        # forecasts
        hfcs = model.historical_forecasts(
            series=y, forecast_horizon=horizon, last_points_only=True
        )
        if not series_list_length:
            # make output the same as if a list of `series` was used
            hfcs = [hfcs]

        n_series = len(y) if series_list_length else 1
        assert isinstance(hfcs, list) and len(hfcs) == n_series
        for hfc in hfcs:
            assert isinstance(hfc, TimeSeries) and len(hfc) == n_fc_expected
            np.testing.assert_array_almost_equal(
                hfc.values(), y_ref.values()[-n_fc_expected:]
            )

        # last_points_only = False: gives a list of lists, where each inner list
        # contains the forecasts (with the entire forecast horizon) of one series
        hfcs = model.historical_forecasts(
            series=y, forecast_horizon=horizon, last_points_only=False
        )
        if not series_list_length:
            # make output the same as if a list of `series` was used
            hfcs = [hfcs]

        assert isinstance(hfcs, list) and len(hfcs) == n_series
        for hfc_series in hfcs:  # list of forecasts per series
            assert isinstance(hfc_series, list) and len(hfc_series) == n_fc_expected
            for hfc in hfc_series:  # each individual forecast
                assert isinstance(hfc, TimeSeries) and len(hfc) == horizon
                np.testing.assert_array_almost_equal(
                    hfc.values(), y_ref.values()[-horizon:]
                )

    @pytest.mark.parametrize(
        "arima_args",
        [
            {},
            {
                "p": np.array([1, 2, 3, 4]),
                "q": (2, 3),
                "seasonal_order": ([1, 5], 1, (1, 2, 3), 6),
                "trend": [0, 0, 2, 1],
            },
        ],
    )
    def test_historical_forecasts_transferrable_future_cov_local_models(
        self, arima_args: dict
    ):
        model = ARIMA(**arima_args)
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

        # passing non-supported covariates
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(
                series,
                past_covariates=series,
                retrain=False,
            )
        assert str(msg.value).startswith(
            "Model prediction does not support `past_covariates`"
        )

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

        # passing non-supported covariates
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(
                series,
                past_covariates=series,
                retrain=True,
            )
        assert str(msg.value).startswith(
            "Model cannot be fit/trained with `past_covariates`."
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
        """Tests historical forecasts with retraining for expected forecast lengths and times"""
        forecast_horizon = 8
        # if no fit and retrain=false, should fit at first iteration
        model_cls, kwargs, model_kwarg, bounds = config
        model = model_cls(**kwargs, **model_kwarg)
        # set train length to be the minimum required training length
        # +1 as sklearn models require min 2 train samples
        train_length = bounds[0] + bounds[1] + 1

        if model.output_chunk_shift > 0:
            with pytest.raises(ValueError) as err:
                forecasts = model.historical_forecasts(
                    series=self.ts_pass_val,
                    forecast_horizon=forecast_horizon,
                    stride=1,
                    train_length=train_length,
                    retrain=True,
                    overlap_end=False,
                )
            assert str(err.value).startswith(
                "Cannot perform auto-regression `(n > output_chunk_length)`"
            )
            # continue the test without auto-regression if we are using shifts
            forecast_horizon = model.output_chunk_length

        # time index without train length
        forecasts_no_train_length = model.historical_forecasts(
            series=self.ts_pass_val,
            forecast_horizon=forecast_horizon,
            stride=1,
            train_length=None,
            retrain=True,
            overlap_end=False,
        )

        # time index with minimum train length
        forecasts = model.historical_forecasts(
            series=self.ts_pass_val,
            forecast_horizon=forecast_horizon,
            stride=1,
            train_length=train_length,
            retrain=True,
            overlap_end=False,
        )

        assert len(forecasts_no_train_length) == len(forecasts)
        theorical_forecast_length = (
            self.ts_val_length
            - train_length  # because we train
            - forecast_horizon  # because we have overlap_end = False
            + 1  # because we include the first element
        )
        assert len(forecasts) == theorical_forecast_length, (
            f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
            f"of retrain=True and overlap_end=False, and a time index of type DateTimeIndex. "
            f"Expected {theorical_forecast_length}, got {len(forecasts)}"
        )
        assert forecasts.time_index.equals(
            self.ts_pass_val.time_index[-theorical_forecast_length:]
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
        assert forecasts.time_index.equals(
            self.ts_pass_val_range.time_index[-theorical_forecast_length:]
        )
        start_idx = self.ts_pass_val_range.get_index_at_point(forecasts.start_time())

        # stride 2
        forecasts = model.historical_forecasts(
            series=self.ts_pass_val_range,
            forecast_horizon=forecast_horizon,
            train_length=train_length,
            stride=2,
            retrain=True,
            overlap_end=False,
        )

        theorical_forecast_length = int(
            np.floor(
                (
                    (
                        self.ts_val_length
                        - train_length  # because we train
                        - forecast_horizon  # because we have overlap_end = False
                        + 1  # because we include the first element
                    )
                    - 1
                )
                / 2
                + 1  # because of stride
            )  # if odd number of elements, we keep the floor
        )

        assert len(forecasts) == theorical_forecast_length, (
            f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
            f"of retrain=True and overlap_end=False and stride=2. "
            f"Expected {theorical_forecast_length}, got {len(forecasts)}"
        )
        assert forecasts.time_index.equals(
            self.ts_pass_val_range.time_index[start_idx::2]
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
                    - train_length  # because we train
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
        assert forecasts.time_index.equals(
            self.ts_pass_val_range.time_index[start_idx::3]
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
            - train_length  # because we train
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
        last_points_times = np.array([fc.end_time() for fc in forecasts])
        np.testing.assert_equal(
            last_points_times,
            self.ts_pass_val_range.time_index[-theorical_forecast_length:].values,
        )

        if not model.supports_past_covariates:
            with pytest.raises(ValueError) as msg:
                model.historical_forecasts(
                    series=self.ts_pass_val_range,
                    past_covariates=self.ts_passengers,
                    retrain=True,
                )
            assert str(msg.value).startswith(
                "Model cannot be fit/trained with `past_covariates`."
            )

        if not model.supports_future_covariates:
            with pytest.raises(ValueError) as msg:
                model.historical_forecasts(
                    series=self.ts_pass_val_range,
                    future_covariates=self.ts_passengers,
                    last_points_only=False,
                )
            assert str(msg.value).startswith(
                "Model cannot be fit/trained with `future_covariates`."
            )

    def test_sanity_check_start(self):
        timeidx_ = tg.linear_timeseries(length=10)
        rangeidx_step1 = tg.linear_timeseries(start=0, length=10, freq=1)
        rangeidx_step2 = tg.linear_timeseries(start=0, length=10, freq=2)

        # invalid start float
        model = LinearRegressionModel(lags=1)
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(rangeidx_step1, start=1.1)
        assert str(msg.value).startswith(
            "if `start` is a float, must be between 0.0 and 1.0."
        )
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(rangeidx_step1, start=-0.1)
        assert str(msg.value).startswith(
            "if `start` is a float, must be between 0.0 and 1.0."
        )

        # invalid start type
        with pytest.raises(TypeError) as msg:
            model.historical_forecasts(rangeidx_step1, start=[0.1])
        assert str(msg.value).startswith(
            "`start` must be either `float`, `int`, `pd.Timestamp` or `None`."
        )

        # label_index (timestamp) with range index series
        model = LinearRegressionModel(lags=1)
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(
                rangeidx_step1, start=timeidx_.end_time() + timeidx_.freq
            )
        assert str(msg.value).startswith(
            "if `start` is a `pd.Timestamp`, all series must be indexed with a `pd.DatetimeIndex`"
        )

        # label_index (int), too large
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(timeidx_, start=11)
        assert str(msg.value).startswith("`start` position `11` is out of bounds")
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(
                rangeidx_step1, start=rangeidx_step1.end_time() + rangeidx_step1.freq
            )
        assert str(msg.value).startswith(
            "`start` time `10` is larger than the last index"
        )
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(
                rangeidx_step2, start=rangeidx_step2.end_time() + rangeidx_step2.freq
            )
        assert str(msg.value).startswith(
            "`start` time `20` is larger than the last index"
        )

        # label_index (timestamp) too high
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(
                timeidx_, start=timeidx_.end_time() + timeidx_.freq
            )
        assert str(msg.value).startswith(
            "`start` time `2000-01-11 00:00:00` is after the last timestamp `2000-01-10 00:00:00`"
        )

        # label_index (timestamp), before series start and stride does not allow to find valid start point in series
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(
                timeidx_,
                start=timeidx_.start_time() - timeidx_.freq,
                stride=len(timeidx_) + 1,
            )
        assert str(msg.value) == (
            "`start` time `1999-12-31 00:00:00` is smaller than the first time index `2000-01-01 00:00:00` "
            "for series at index: 0, and could not find a valid start point within the time index that lies a "
            "round-multiple of `stride=11` ahead of `start` (first inferred start is `2000-01-11 00:00:00`, "
            "but last time index is `2000-01-10 00:00:00`."
        )

        # label_index (timestamp), before trainable/predictable index and stride does not allow to find valid start
        # point in series
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(
                timeidx_, start=timeidx_.start_time(), stride=len(timeidx_)
            )
        assert str(msg.value) == (
            "`start` time `2000-01-01 00:00:00` is smaller than the first historical forecastable time index "
            "`2000-01-04 00:00:00` for series at index: 0, and could not find a valid start point within the "
            "historical forecastable time index that lies a round-multiple of `stride=10` ahead of `start` "
            "(first inferred start is `2000-01-11 00:00:00`, but last historical forecastable time index is "
            "`2000-01-10 00:00:00`."
        )

        # label_index (int), too low and stride does not allow to find valid start point in series
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(
                rangeidx_step1,
                start=rangeidx_step1.start_time() - rangeidx_step1.freq,
                stride=len(rangeidx_step1) + 1,
            )
        assert str(msg.value) == (
            "`start` time `-1` is smaller than the first time index `0` for series at index: 0, and could not "
            "find a valid start point within the time index that lies a round-multiple of `stride=11` ahead of "
            "`start` (first inferred start is `10`, but last time index is `9`."
        )

        # label_index (int), before trainable/predictable index and stride does not allow to find valid start
        # point in series
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(
                rangeidx_step1,
                start=rangeidx_step1.start_time(),
                stride=len(rangeidx_step1),
            )
        assert str(msg.value) == (
            "`start` time `0` is smaller than the first historical forecastable time index `3` for series at "
            "index: 0, and could not find a valid start point within the historical forecastable time index "
            "that lies a round-multiple of `stride=10` ahead of `start` (first inferred start is `10`, but last "
            "historical forecastable time index is `9`."
        )

        # positional_index with time index, predicting only the last position
        preds = model.historical_forecasts(timeidx_, start=9, start_format="position")
        assert len(preds) == 1
        assert preds.start_time() == timeidx_.time_index[9]

        # positional_index, predicting from the first position with retrain=True
        preds1 = model.historical_forecasts(
            timeidx_, start=-10, start_format="position"
        )
        # positional_index, before start of series gives same results
        preds2 = model.historical_forecasts(
            timeidx_, start=-11, start_format="position"
        )
        assert (
            len(preds1) == len(preds2) == len(timeidx_) - model.min_train_series_length
        )
        assert (
            preds1.start_time()
            == preds2.start_time()
            == timeidx_.time_index[model.min_train_series_length]
        )

        # positional_index, beyond boundaries
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(timeidx_, start=10, start_format="position")
        assert str(msg.value).startswith(
            "`start` position `10` is out of bounds for series of length 10"
        )

        # positional_index with range index, predicting only the last position
        preds = model.historical_forecasts(
            rangeidx_step2, start=9, start_format="position"
        )
        assert len(preds) == 1
        assert preds.start_time() == rangeidx_step2.time_index[9]

        # positional_index, predicting from the first position with retrain=True
        preds1 = model.historical_forecasts(
            rangeidx_step2, start=-10, start_format="position"
        )
        # positional_index, before start of series gives same results
        preds2 = model.historical_forecasts(
            rangeidx_step2, start=-11, start_format="position"
        )
        assert (
            len(preds1)
            == len(preds2)
            == len(rangeidx_step2) - model.min_train_series_length
        )
        assert (
            preds1.start_time()
            == preds2.start_time()
            == rangeidx_step2.time_index[model.min_train_series_length]
        )

        # positional_index, beyond boundaries
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(
                rangeidx_step2, start=10, start_format="position"
            )
        assert str(msg.value).startswith(
            "`start` position `10` is out of bounds for series of length 10"
        )

    @pytest.mark.parametrize(
        "config",
        list(
            itertools.product(
                [
                    (
                        "2000-01-01 00:00:00",  # start
                        1,  # stride
                        "2000-01-01 03:00:00",  # expected start
                        "h",  # freq
                    ),
                    ("2000-01-01 00:00:00", 2, "2000-01-01 04:00:00", "h"),
                    ("1999-01-01 00:00:00", 6, "2000-01-01 06:00:00", "h"),
                    ("2000-01-01 00:00:00", 2, "2000-01-01 08:00:00", "2h"),
                    # special case where start is not in the frequency -> start will be converted
                    # to "2000-01-01 00:00:00", and then it's adjusted to be within the historical fc index
                    ("1999-12-31 23:00:00", 2, "2000-01-01 08:00:00", "2h"),
                    # integer index
                    (0, 1, 3, 1),
                    (0, 2, 4, 1),
                    (-24, 6, 6, 1),
                    (0, 2, 8, 2),
                    # special case where start is not in the frequency -> start will be converted
                    # to 0, and then it's adjusted to be within the historical fc index
                    (-1, 2, 8, 2),
                ],
                ["value", "position"],  # start format
                [True, False],  # retrain
                [True, False] if TORCH_AVAILABLE else [False],  # use torch model
            )
        ),
    )
    def test_historical_forecasts_start_too_early(self, caplog, config):
        """If start is not within the trainable/forecastable index, it should start a round-multiple of `stride` ahead
        of `start`. Checks for:
        - correct warnings
        - datetime / integer index
        - different frequencies
        - different strides
        - start "value" and "position"
        - retrain / no-retrain (optimized and non-optimized)
        - torch and regression model
        """
        # the configuration is defined for `retrain = True` and `start_format = "value"`
        (
            (start, stride, start_expected, freq),
            start_format,
            retrain,
            use_torch_model,
        ) = config
        if isinstance(freq, str):
            start, start_expected = pd.Timestamp(start), pd.Timestamp(start_expected)
            start_series = pd.Timestamp("2000-01-01 00:00:00")
        else:
            start_series = 0

        series = tg.linear_timeseries(
            start=start_series,
            length=7,
            freq=freq,
        )
        # when hist fc `start` is not in the valid frequency range, it is converted to a time that is valid.
        # e.g. `start="1999-12-31 23:00:00:` with `freq="2h"` is converted to `"2000-01-01 00:00:00"`
        start_position = n_steps_between(end=start_series, start=start, freq=freq)
        start_time_expected = series.start_time() - start_position * series.freq

        if start_format == "position":
            start = -start_position
            if start < 0:
                # negative position is relative to the end of the series
                start -= len(series)
            start_format_msg = f"position `{start}` corresponding to time "
        else:
            start_format_msg = "time "

        if use_torch_model:
            kwargs = deepcopy(tfm_kwargs)
            kwargs["pl_trainer_kwargs"]["fast_dev_run"] = True
            # use ocl=2 to have same `min_train_length` as the regression model
            model = BlockRNNModel(
                input_chunk_length=1, output_chunk_length=2, n_epochs=1, **kwargs
            )
        else:
            model = LinearRegressionModel(lags=1)

        model.fit(series)
        # if the stride is shorter than the train series length, retrain=False can start earlier
        if not retrain and stride <= model.min_train_series_length:
            start_expected -= (
                model.min_train_series_length + model.extreme_lags[0]
            ) * series.freq

        # label index
        warning_expected = (
            f"`start` {start_format_msg}`{start_time_expected}` is before the first predictable/trainable historical "
            f"forecasting point for series at index: 0. Using the first historical forecasting point "
            f"`{start_expected}` that lies a round-multiple of `stride={stride}` ahead of `start`. To hide these "
            f"warnings, set `show_warnings=False`."
        )

        # check that warning is raised when too early
        enable_optimizations = [False] if retrain else [False, True]
        for enable_optimization in enable_optimizations:
            with caplog.at_level(logging.WARNING):
                pred = model.historical_forecasts(
                    series,
                    start=start,
                    stride=stride,
                    retrain=retrain,
                    start_format=start_format,
                    enable_optimization=enable_optimization,
                )
                assert warning_expected in caplog.text
                assert pred.start_time() == start_expected
        caplog.clear()
        # but no warning when start is at the right time
        warning_short = (
            f"Using the first historical forecasting point `{start_expected}` that lies a round-multiple "
            f"of `stride={stride}` ahead of `start`. To hide these warnings, set `show_warnings=False`."
        )
        with caplog.at_level(logging.WARNING):
            pred = model.historical_forecasts(
                series,
                start=start_expected,
                stride=stride,
                retrain=False,
                start_format="value",
                enable_optimization=True,
            )
            assert warning_short not in caplog.text
            assert pred.start_time() == start_expected

    @pytest.mark.parametrize("config", models_reg_no_cov_cls_kwargs)
    def test_regression_auto_start_multiple_no_cov(self, config):
        # minimum required train length (+1 since sklearn models require 2 samples)
        forecast_horizon = 10
        model_cls, kwargs, model_kwargs, bounds = config
        train_length = bounds[0] + bounds[1] + 1
        model = model_cls(
            **kwargs,
            **model_kwargs,
        )
        model.fit(self.ts_pass_train)

        if model.output_chunk_shift > 0:
            with pytest.raises(ValueError) as err:
                forecasts = model.historical_forecasts(
                    series=[self.ts_pass_val, self.ts_pass_val],
                    forecast_horizon=forecast_horizon,
                    train_length=train_length,
                    stride=1,
                    retrain=True,
                    overlap_end=False,
                )
            assert str(err.value).startswith(
                "Cannot perform auto-regression `(n > output_chunk_length)`"
            )
            # continue the test without autogregression if we are using shifts
            forecast_horizon = model.output_chunk_length

        forecasts = model.historical_forecasts(
            series=[self.ts_pass_val, self.ts_pass_val],
            forecast_horizon=forecast_horizon,
            train_length=train_length,
            stride=1,
            retrain=True,
            overlap_end=False,
        )

        assert len(forecasts) == 2, (
            f"Model {model_cls} did not return a list of historical forecasts"
        )

        theorical_forecast_length = (
            self.ts_val_length
            - train_length
            - forecast_horizon  # because we have overlap_end = False
            + 1  # because we include the first element
        )

        assert len(forecasts[0]) == len(forecasts[1]) == theorical_forecast_length, (
            f"Model {model_cls.__name__} does not return the right number of historical forecasts in the case "
            f"of retrain=True and overlap_end=False, and a time index of type DateTimeIndex. "
            f"Expected {theorical_forecast_length}, got {len(forecasts[0])} and {len(forecasts[1])}"
        )
        assert forecasts[0].time_index.equals(forecasts[1].time_index) and forecasts[
            0
        ].time_index.equals(self.ts_pass_val.time_index[-theorical_forecast_length:])

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [ts_univariate, ts_multivariate],
            models_reg_no_cov_cls_kwargs + models_reg_cov_cls_kwargs,
            [True, False],
            [True, False],
            [1, 5],
        ),
    )
    def test_optimized_historical_forecasts_regression(self, config):
        ts, model_config, multi_models, overlap_end, forecast_horizon = config
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
            future_covariates=(
                ts_covs if model_same.supports_future_covariates else None
            ),
        )
        # ocl >= forecast horizon
        model_kwargs_diff = model_kwargs.copy()
        model_kwargs_diff["output_chunk_length"] = 5
        model_kwargs_diff["multi_models"] = multi_models
        model_diff = model_cls(**model_kwargs_same)
        model_diff.fit(
            series=ts[:start],
            past_covariates=ts_covs if model_diff.supports_past_covariates else None,
            future_covariates=(
                ts_covs if model_diff.supports_future_covariates else None
            ),
        )
        # no parametrization to save time on model training at the cost of test granularity
        for model in [model_same, model_diff]:
            for last_points_only in [True, False]:
                for stride in [1, 2]:
                    hist_fct = model.historical_forecasts(
                        series=ts,
                        past_covariates=(
                            ts_covs if model.supports_past_covariates else None
                        ),
                        future_covariates=(
                            ts_covs if model.supports_future_covariates else None
                        ),
                        start=start,
                        retrain=False,
                        last_points_only=last_points_only,
                        stride=stride,
                        forecast_horizon=forecast_horizon,
                        overlap_end=overlap_end,
                        enable_optimization=False,
                    )

                    # manually packing the series in list to match expected inputs
                    opti_hist_fct = model._optimized_historical_forecasts(
                        series=ts,
                        past_covariates=(
                            ts_covs if model.supports_past_covariates else None
                        ),
                        future_covariates=(
                            ts_covs if model.supports_future_covariates else None
                        ),
                        start=start,
                        last_points_only=last_points_only,
                        stride=stride,
                        forecast_horizon=forecast_horizon,
                        overlap_end=overlap_end,
                    )

                    self.helper_compare_hf(hist_fct, opti_hist_fct)

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
        np.random.seed(0)
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
            series=series_val,
            past_covariates=pc,
            future_covariates=fc,
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

        opti_hist_fct = model._optimized_historical_forecasts(series=series_val)

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
            series=series_val,
            past_covariates=pc,
            future_covariates=fc,
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
            for hfc_, ohfc_ in zip(hfc, ohfc):
                assert hfc_.columns.equals(series.columns)
                assert ohfc_.columns.equals(series.columns)
                assert len(ohfc_) == n_pred_points_expected
                assert (hfc_.time_index == ohfc_.time_index).all()
                np.testing.assert_array_almost_equal(
                    hfc_.all_values(), ohfc_.all_values()
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
            past_covariates=(
                [
                    self.ts_past_cov_valid_same_start,
                    self.ts_past_cov_valid_same_start,
                ]
                if "lags_past_covariates" in kwargs
                else None
            ),
            future_covariates=(
                [
                    self.ts_past_cov_valid_same_start,
                    self.ts_past_cov_valid_same_start,
                ]
                if "lags_future_covariates" in kwargs
                else None
            ),
            last_points_only=True,
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=True,
            overlap_end=False,
        )

        assert len(forecasts_retrain) == 2, (
            f"Model {model_cls} did not return a list of historical forecasts"
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

        past_lag = min(
            min_target_lag if min_target_lag else 0,
            min_past_cov_lag if min_past_cov_lag else 0,
            (
                min_future_cov_lag
                if min_future_cov_lag is not None and min_future_cov_lag < 0
                else 0
            ),
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
            past_covariates=(
                [
                    self.ts_past_cov_valid_same_start,
                    self.ts_past_cov_valid_same_start,
                ]
                if "lags_past_covariates" in kwargs
                else None
            ),
            future_covariates=(
                [
                    self.ts_past_cov_valid_same_start,
                    self.ts_past_cov_valid_same_start,
                ]
                if "lags_future_covariates" in kwargs
                else None
            ),
        )
        forecasts_no_retrain = model.historical_forecasts(
            series=[self.ts_pass_val, self.ts_pass_val],
            past_covariates=(
                [
                    self.ts_past_cov_valid_same_start,
                    self.ts_past_cov_valid_same_start,
                ]
                if "lags_past_covariates" in kwargs
                else None
            ),
            future_covariates=(
                [
                    self.ts_past_cov_valid_same_start,
                    self.ts_past_cov_valid_same_start,
                ]
                if "lags_future_covariates" in kwargs
                else None
            ),
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
            output_chunk_shift,
            _,
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
    @pytest.mark.parametrize(
        "model_config,retrain",
        itertools.product(models_torch_cls_kwargs, [True, False]),
    )
    def test_torch_auto_start_multiple_no_cov(self, model_config, retrain):
        n_fcs = 3
        forecast_hrz = 10
        model_cls, kwargs, bounds, _ = model_config
        model = model_cls(
            random_state=0,
            **kwargs,
        )

        # we expect first predicted point after `min_train_series_length`
        # model is expected to generate `n_fcs` historical forecasts with `n=forecast_hrz` and
        # `series` of length `length_series_history`
        length_series_history = model.min_train_series_length + forecast_hrz + n_fcs - 1
        series = self.ts_pass_train[:length_series_history]
        if not retrain:
            model.fit(series)

        # check historical forecasts for several time series,
        # retrain True and overlap_end False
        forecasts = model.historical_forecasts(
            series=[series] * 2,
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=False,
        )
        assert len(forecasts) == 2, (
            f"Model {model_cls} did not return a list of historical forecasts"
        )

        # with the required time spans we expect to get `n_fcs` forecasts
        if not retrain:
            # with retrain=False, we can start `output_chunk_length` steps earlier for non-RNNModels
            # and `training_length - input_chunk_length` steps for RNNModels
            if not isinstance(model, RNNModel):
                add_fcs = model.extreme_lags[1] + 1
            else:
                add_fcs = model.extreme_lags[7] + 1
        else:
            add_fcs = 0
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + add_fcs
        assert forecasts[0].end_time() == forecasts[1].end_time() == series.end_time()

        # check historical forecasts for several time series,
        # retrain True and overlap_end True
        forecasts = model.historical_forecasts(
            series=[series] * 2,
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=True,
        )

        assert len(forecasts) == 2, (
            f"Model {model_cls} did not return a list of historical forecasts"
        )
        # with overlap_end=True, we can generate additional `forecast_hrz`
        # with retrain=False, we can start `add_fcs` steps earlier
        # forecasts after the end of `series`
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + forecast_hrz + add_fcs
        assert (
            forecasts[0].end_time()
            == forecasts[1].end_time()
            == series.end_time() + forecast_hrz * series.freq
        )

    @pytest.mark.slow
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    @pytest.mark.parametrize(
        "model_config,retrain",
        itertools.product(models_torch_cls_kwargs, [True, False]),
    )
    def test_torch_auto_start_with_past_cov(self, model_config, retrain):
        n_fcs = 3
        forecast_hrz = 10
        # past covariates only
        model_cls, kwargs, bounds, cov_type = model_config

        model = model_cls(
            random_state=0,
            **kwargs,
        )

        if not model.supports_past_covariates:
            with pytest.raises(ValueError) as err:
                model.fit(
                    series=self.ts_pass_train, past_covariates=self.ts_past_cov_train
                )
            assert str(err.value).startswith(
                "The model does not support `past_covariates`."
            )
            return

        # we expect first predicted point after `min_train_series_length`
        # model is expected to generate `n_fcs` historical forecasts with `n=forecast_hrz`,
        # `series` of length `length_series_history`, and covariates that cover the required time range
        length_series_history = model.min_train_series_length + forecast_hrz + n_fcs - 1
        series = self.ts_pass_train[:length_series_history]

        # for historical forecasts, minimum required past covariates should end
        # `forecast_hrz` before the end of `series`
        pc = series[:-forecast_hrz]

        if not retrain:
            model.fit(series, past_covariates=pc)

        # same start, overlap_end=False
        forecasts = model.historical_forecasts(
            series=[series] * 2,
            past_covariates=[pc] * 2,
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=False,
        )
        assert len(forecasts) == 2, (
            f"Model {model_cls} did not return a list of historical forecasts"
        )

        # with the required time spans we expect to get `n_fcs` forecasts
        if not retrain:
            # with retrain=False, we can start `output_chunk_length` steps earlier for non-RNNModels
            # and `training_length - input_chunk_length` steps for RNNModels
            add_fcs = model.extreme_lags[1] + 1
        else:
            add_fcs = 0
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + add_fcs
        assert forecasts[0].end_time() == forecasts[1].end_time() == series.end_time()

        # check the same for `overlap_end=True`
        forecasts = model.historical_forecasts(
            series=[series] * 2,
            past_covariates=[pc] * 2,
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=True,
        )
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + add_fcs
        assert forecasts[0].end_time() == forecasts[1].end_time() == series.end_time()

        # same time index, `overlap_end=True`
        forecasts = model.historical_forecasts(
            series=[series] * 2,
            past_covariates=[series] * 2,
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=True,
        )
        assert len(forecasts) == 2, (
            f"Model {model_cls} did not return a list of historical forecasts"
        )
        # with overlap_end=True, we can generate additional `forecast_hrz`
        # with retrain=False, we can start `add_fcs` steps earlier
        # forecasts after the end of `series`
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + forecast_hrz + add_fcs
        assert (
            forecasts[0].end_time()
            == forecasts[1].end_time()
            == series.end_time() + forecast_hrz * series.freq
        )

        # `pc_longer` has more than required length
        pc_longer = pc.prepend_values([0.0]).append_values([0.0])
        # `pc_before` starts before and has required times
        pc_longer_start = pc.prepend_values([0.0])
        # `pc_after` has required length but starts one step after `pc`
        pc_start_after = pc[1:].append_values([0.0])
        # `pc_end_before` has required length but end one step before `pc`
        pc_end_before = pc[:-1].prepend_values([0.0])

        # checks for long enough and shorter covariates
        forecasts = model.historical_forecasts(
            series=[series] * 4,
            past_covariates=[
                pc_longer,
                pc_longer_start,
                pc_start_after,
                pc_end_before,
            ],
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=False,
        )

        # for long enough past covariates (but too short for overlapping after the end), we expect `n_fcs` forecast
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + add_fcs
        # `pc_start_after` and `pc_end_before` are one step too short for all `n_fcs`
        assert len(forecasts[2]) == len(forecasts[3]) == n_fcs + add_fcs - 1
        assert all([fc.end_time() == series.end_time() for fc in forecasts[:3]])
        assert forecasts[3].end_time() == series.end_time() - series.freq

    @pytest.mark.slow
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    @pytest.mark.parametrize(
        "model_config,retrain",
        list(itertools.product(models_torch_cls_kwargs, [True, False]))[2:],
    )
    def test_torch_auto_start_with_future_cov(self, model_config, retrain):
        n_fcs = 3
        forecast_hrz = 10
        # future covariates only
        model_cls, kwargs, bounds, cov_type = model_config

        model = model_cls(
            random_state=0,
            **kwargs,
        )
        if not model.supports_future_covariates:
            with pytest.raises(ValueError) as err:
                model.fit(
                    series=self.ts_pass_train, future_covariates=self.ts_fut_cov_train
                )
            assert str(err.value).startswith(
                "The model does not support `future_covariates`."
            )
            return

        # we expect first predicted point after `min_train_series_length`
        # model is expected to generate `n_fcs` historical forecasts with `n=forecast_hrz`,
        # `series` of length `length_series_history`, and covariates that cover the required time range
        length_series_history = model.min_train_series_length + forecast_hrz + n_fcs - 1
        series = self.ts_pass_train[:length_series_history]

        # to generate `n_fcs` historical forecasts, and since `forecast_horizon > output_chunk_length`,
        # we need additional `output_chunk_length - horizon` future covariates steps
        add_n = max(model.extreme_lags[1] + 1 - forecast_hrz, 0)
        fc = series.append_values([0.0] * add_n) if add_n else series

        if not retrain:
            model.fit(series, future_covariates=fc)

        # same start, overlap_end=False
        forecasts = model.historical_forecasts(
            series=[series] * 2,
            future_covariates=[fc] * 2,
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=False,
        )
        assert len(forecasts) == 2, (
            f"Model {model_cls} did not return a list of historical forecasts"
        )

        # with the required time spans we expect to get `n_fcs` forecasts
        if not retrain:
            # with retrain=False, we can start `output_chunk_length` steps earlier for non-RNNModels
            # and `training_length - input_chunk_length` steps for RNNModels
            if not isinstance(model, RNNModel):
                add_fcs = model.extreme_lags[1] + 1
            else:
                add_fcs = model.extreme_lags[7] + 1
        else:
            add_fcs = 0
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + add_fcs
        assert forecasts[0].end_time() == forecasts[1].end_time() == series.end_time()

        # check the same for `overlap_end=True`
        forecasts = model.historical_forecasts(
            series=[series] * 2,
            future_covariates=[fc] * 2,
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=True,
        )
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + add_fcs
        assert forecasts[0].end_time() == forecasts[1].end_time() == series.end_time()

        # `overlap_end=True`, with long enough future covariates
        if not isinstance(model, RNNModel):
            add_n = model.output_chunk_length
        else:
            # RNNModel is a special case with always `output_chunk_length=1`
            add_n = forecast_hrz
        fc_long = fc.append_values([0.0] * add_n)
        forecasts = model.historical_forecasts(
            series=[series] * 2,
            future_covariates=[fc_long] * 2,
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=True,
        )
        assert len(forecasts) == 2, (
            f"Model {model_cls} did not return a list of historical forecasts"
        )
        # with overlap_end=True, we can generate additional `forecast_hrz`
        # with retrain=False, we can start `add_fcs` steps earlier
        # forecasts after the end of `series`
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + forecast_hrz + add_fcs
        assert (
            forecasts[0].end_time()
            == forecasts[1].end_time()
            == series.end_time() + forecast_hrz * series.freq
        )

        # `fc_longer` has more than required length
        fc_longer = fc.prepend_values([0.0]).append_values([0.0])
        # `fc_before` starts before and has required times
        fc_longer_start = fc.prepend_values([0.0])
        # `fc_after` has required length but starts one step after `fc`
        fc_start_after = fc[1:].append_values([0.0])
        # `fc_end_before` has required length but end one step before `fc`
        fc_end_before = fc[:-1].prepend_values([0.0])

        # checks for long enough and shorter covariates
        forecasts = model.historical_forecasts(
            series=[series] * 4,
            future_covariates=[
                fc_longer,
                fc_longer_start,
                fc_start_after,
                fc_end_before,
            ],
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=False,
        )

        # for long enough future covariates (but too short for overlapping after the end), we expect `n_fcs` forecast
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + add_fcs
        # `fc_start_after` and `fc_end_before` are one step too short for all `n_fcs`
        assert len(forecasts[2]) == len(forecasts[3]) == n_fcs + add_fcs - 1
        assert all([fc.end_time() == series.end_time() for fc in forecasts[:3]])
        assert forecasts[3].end_time() == series.end_time() - series.freq

    @pytest.mark.slow
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    @pytest.mark.parametrize(
        "model_config,retrain",
        itertools.product(models_torch_cls_kwargs, [True, False]),
    )
    def test_torch_auto_start_with_past_and_future_cov(self, model_config, retrain):
        n_fcs = 3
        forecast_hrz = 10
        # past and future covariates
        model_cls, kwargs, bounds, cov_type = model_config

        model = model_cls(
            random_state=0,
            **kwargs,
        )
        if not (model.supports_past_covariates and model.supports_future_covariates):
            with pytest.raises(ValueError) as err:
                model.fit(
                    self.ts_pass_train,
                    past_covariates=self.ts_past_cov_train,
                    future_covariates=self.ts_fut_cov_train,
                )
            invalid_covs = []
            if not model.supports_past_covariates:
                invalid_covs.append("`past_covariates`")
            if not model.supports_future_covariates:
                invalid_covs.append("`future_covariates`")
            assert str(err.value).startswith(
                f"The model does not support {', '.join(invalid_covs)}"
            )
            return

        # we expect first predicted point after `min_train_series_length`
        # model is expected to generate `n_fcs` historical forecasts with `n=forecast_hrz`,
        # `series` of length `length_series_history`, and covariates that cover the required time range
        length_series_history = model.min_train_series_length + forecast_hrz + n_fcs - 1
        series = self.ts_pass_train[:length_series_history]

        # for historical forecasts, minimum required past covariates should end
        # `forecast_hrz` before the end of `series`
        pc = series[:-forecast_hrz]

        # to generate `n_fcs` historical forecasts, and since `forecast_horizon > output_chunk_length`,
        # we need additional `output_chunk_length - horizon` future covariates steps
        add_n = max(model.extreme_lags[1] + 1 - forecast_hrz, 0)
        fc = series.append_values([0.0] * add_n) if add_n else series

        if not retrain:
            model.fit(series, past_covariates=pc, future_covariates=fc)

        # same start, overlap_end=False
        forecasts = model.historical_forecasts(
            series=[series] * 2,
            past_covariates=[pc] * 2,
            future_covariates=[fc] * 2,
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=False,
        )
        assert len(forecasts) == 2, (
            f"Model {model_cls} did not return a list of historical forecasts"
        )

        # with the required time spans we expect to get `n_fcs` forecasts
        if not retrain:
            # with retrain=False, we can start `output_chunk_length` steps earlier for non-RNNModels
            # and `training_length - input_chunk_length` steps for RNNModels
            if not isinstance(model, RNNModel):
                add_fcs = model.extreme_lags[1] + 1
            else:
                add_fcs = model.extreme_lags[7] + 1
        else:
            add_fcs = 0
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + add_fcs
        assert forecasts[0].end_time() == forecasts[1].end_time() == series.end_time()

        # check the same for `overlap_end=True`
        forecasts = model.historical_forecasts(
            series=[series] * 2,
            past_covariates=[pc] * 2,
            future_covariates=[fc] * 2,
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=True,
        )
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + add_fcs
        assert forecasts[0].end_time() == forecasts[1].end_time() == series.end_time()

        # `overlap_end=True`, with long enough past and future covariates
        if not isinstance(model, RNNModel):
            add_n = model.output_chunk_length
        else:
            # RNNModel is a special case with always `output_chunk_length=1`
            add_n = forecast_hrz
        fc_long = fc.append_values([0.0] * add_n)
        forecasts = model.historical_forecasts(
            series=[series] * 2,
            past_covariates=[series] * 2,
            future_covariates=[fc_long] * 2,
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=True,
        )
        assert len(forecasts) == 2, (
            f"Model {model_cls} did not return a list of historical forecasts"
        )
        # with overlap_end=True, we can generate additional `forecast_hrz`
        # with retrain=False, we can start `add_fcs` steps earlier
        # forecasts after the end of `series`
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + forecast_hrz + add_fcs
        assert (
            forecasts[0].end_time()
            == forecasts[1].end_time()
            == series.end_time() + forecast_hrz * series.freq
        )

        # `pc_longer` has more than required length
        pc_longer = pc.prepend_values([0.0]).append_values([0.0])
        # `pc_before` starts before and has required times
        pc_longer_start = pc.prepend_values([0.0])
        # `pc_after` has required length but starts one step after `pc`
        pc_start_after = pc[1:].append_values([0.0])
        # `pc_end_before` has required length but end one step before `pc`
        pc_end_before = pc[:-1].prepend_values([0.0])

        # `fc_longer` has more than required length
        fc_longer = fc.prepend_values([0.0]).append_values([0.0])
        # `fc_before` starts before and has required times
        fc_longer_start = fc.prepend_values([0.0])
        # `fc_after` has required length but starts one step after `fc`
        fc_start_after = fc[1:].append_values([0.0])
        # `fc_end_before` has required length but end one step before `fc`
        fc_end_before = fc[:-1].prepend_values([0.0])

        # checks for long enough and shorter covariates
        forecasts = model.historical_forecasts(
            series=[series] * 4,
            past_covariates=[
                pc_longer,
                pc_longer_start,
                pc_start_after,
                pc_end_before,
            ],
            future_covariates=[
                fc_longer,
                fc_longer_start,
                fc_start_after,
                fc_end_before,
            ],
            forecast_horizon=forecast_hrz,
            stride=1,
            retrain=retrain,
            overlap_end=False,
        )

        # for long enough future covariates (but too short for overlapping after the end), we expect `n_fcs` forecast
        assert len(forecasts[0]) == len(forecasts[1]) == n_fcs + add_fcs
        # `*_start_after` and `*_end_bore` are one step too short for all `n_fcs`
        assert len(forecasts[2]) == len(forecasts[3]) == n_fcs + add_fcs - 1
        assert all([fc.end_time() == series.end_time() for fc in forecasts[:3]])
        assert forecasts[3].end_time() == series.end_time() - series.freq

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

        model = self.create_model(1, False, model_type=model_type)
        # skip torch models if not installed
        if model is None:
            return
        # model doesn't use likelihood
        with pytest.raises(ValueError):
            model.historical_forecasts(
                self.ts_pass_train,
                predict_likelihood_parameters=True,
            )

        model = self.create_model(1, model_type=model_type)
        # forecast_horizon > output_chunk_length doesn't work
        with pytest.raises(ValueError):
            model.historical_forecasts(
                self.ts_pass_train,
                predict_likelihood_parameters=True,
                forecast_horizon=2,
            )

        model = self.create_model(1, model_type=model_type)
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
        model = self.create_model(1, model_type=model_type)
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
        model = self.create_model(1, model_type=model_type)
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
        model = self.create_model(2, model_type=model_type)
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

    @pytest.mark.parametrize(
        "config",
        product(
            [False, True],  # last_points_only
            [True, False],  # multi_models
            [1, 2, 3],  # horizon
        ),
    )
    def test_probabilistic_optimized_hist_fc_regression(self, config):
        """Tests optimized probabilistic historical forecasts for regression models."""
        np.random.seed(42)
        lpo, multi_models, n = config
        ocl = 2
        q = [0.05, 0.50, 0.95]

        y = tg.linear_timeseries(length=20)
        y = y.stack(y + 1.0)
        y = [y, y]

        icl = 3
        model = LinearRegressionModel(
            lags=icl,
            output_chunk_length=ocl,
            likelihood="quantile",
            quantiles=q,
            multi_models=multi_models,
        )
        model.fit(y)
        # probabilistic forecasts non-optimized
        hfcs_no_opt = model.historical_forecasts(
            series=y,
            forecast_horizon=n,
            last_points_only=lpo,
            retrain=False,
            enable_optimization=False,
            num_samples=1000,
            stride=n,
        )
        # probabilistic forecasts optimized
        hfcs_opt = model.historical_forecasts(
            series=y,
            forecast_horizon=n,
            last_points_only=lpo,
            retrain=False,
            enable_optimization=True,
            num_samples=1000,
            stride=n,
        )
        if n <= ocl:
            # quantile forecasts optimized
            hfcs_opt_q = model.historical_forecasts(
                series=y,
                forecast_horizon=n,
                last_points_only=lpo,
                retrain=False,
                enable_optimization=True,
                predict_likelihood_parameters=True,
                stride=n,
            )
            if lpo:
                q_med = hfcs_opt_q[0].components[1::3].tolist()
            else:
                q_med = hfcs_opt_q[0][0].components[1::3].tolist()
                hfcs_opt_q = (
                    [concatenate(hfc) for hfc in hfcs_opt_q]
                    if hfcs_opt_q is not None
                    else hfcs_opt_q
                )
            hfcs_opt_q = (
                [hfc[q_med] for hfc in hfcs_opt_q]
                if hfcs_opt_q is not None
                else hfcs_opt_q
            )
        else:
            hfcs_opt_q = [None] * len(hfcs_opt)

        if not lpo:
            hfcs_opt = [concatenate(hfc) for hfc in hfcs_opt]
            hfcs_no_opt = [concatenate(hfc) for hfc in hfcs_no_opt]

        for hfc_opt, mean_opt_q, hfc_no_opt in zip(hfcs_opt, hfcs_opt_q, hfcs_no_opt):
            mean_opt = hfc_opt.all_values().mean(axis=2)
            mean_no_opt = hfc_no_opt.all_values().mean(axis=2)
            assert np.abs(mean_opt - mean_no_opt).max() < 0.1
            if mean_opt_q is not None:
                assert np.abs(mean_opt - mean_opt_q.values()).max() < 0.1

    def helper_manual_scaling_prediction(
        self,
        model,
        ts: dict[str, TimeSeries],
        hf_scaler: dict[str, Scaler],
        retrain: bool,
        end_idx: int,
        ocl: int,
        series_idx: Optional[int] = None,
    ):
        ts_copy = deepcopy(ts)
        hf_scaler_copy = deepcopy(hf_scaler)
        for ts_name in hf_scaler_copy:
            # train the fittable scaler without leaking data
            if isinstance(hf_scaler_copy[ts_name], FittableDataTransformer):
                if ts_name == "series" or ts_name == "past_covariates":
                    tmp_ts = ts_copy[ts_name][:end_idx]
                else:
                    # for future covariates, the scaler may access future information
                    tmp_ts = ts_copy[ts_name][: end_idx + max(0, model.extreme_lags[5])]
                if retrain:
                    hf_scaler_copy[ts_name].fit(tmp_ts)
            # apply the scaler on the whole series
            ts_copy[ts_name] = hf_scaler_copy[ts_name].transform(
                ts_copy[ts_name], series_idx=series_idx
            )

        series = ts_copy.pop("series")[:end_idx]
        if retrain:
            # completly reset model for reproducibility of the predict()
            model = model.untrained_model()
            model.fit(series=series, **ts_copy)

        # local model does not support the "series" argument in predict()
        if isinstance(model, LocalForecastingModel):
            pred = model.predict(n=ocl, **ts_copy)
        else:
            pred = model.predict(n=ocl, series=series, **ts_copy)

        # scale back the forecasts
        if isinstance(hf_scaler_copy.get("series"), InvertibleDataTransformer):
            return hf_scaler_copy["series"].inverse_transform(
                pred, series_idx=series_idx
            )
        else:
            return pred

    def helper_compare_hf(self, ts_A, ts_B):
        """Helper method to compare all the entries between two historical forecasts"""
        type_ts_a = get_series_seq_type(ts_A)
        type_ts_b = get_series_seq_type(ts_B)

        assert type_ts_a == type_ts_b
        assert len(ts_A) == len(ts_B)

        if type_ts_a == SeriesType.SINGLE:
            ts_A = [[ts_A]]
            ts_B = [[ts_B]]
        elif type_ts_a == SeriesType.SEQ:
            ts_A = [ts_A]
            ts_B = [ts_B]

        for ts_a, ts_b in zip(ts_A, ts_B):
            for ts_a_, ts_b_ in zip(ts_a, ts_b):
                assert ts_a_.time_index.equals(ts_b_.time_index)
                np.testing.assert_almost_equal(
                    ts_a_.all_values(),
                    ts_b_.all_values(),
                )

    def helper_get_model_params(
        self, model_cls, series: dict, output_chunk_length: int
    ) -> dict:
        model_params = {}
        if TORCH_AVAILABLE and issubclass(model_cls, NLinearModel):
            model_params["input_chunk_length"] = 5
            model_params["output_chunk_length"] = output_chunk_length
            model_params["n_epochs"] = 1
            model_params["random_state"] = 123
            model_params = {
                **model_params,
                **tfm_kwargs,
            }
        elif issubclass(model_cls, LinearRegressionModel):
            model_params["lags"] = 5
            model_params["output_chunk_length"] = output_chunk_length
            if "past_covariates" in series:
                model_params["lags_past_covariates"] = 4
            if "future_covariates" in series:
                model_params["lags_future_covariates"] = [-3, -2]

        return model_params

    @pytest.mark.parametrize(
        "params",
        product(
            [
                (
                    {
                        "series": sine_univariate1 - 11,
                    },
                    {"series": Scaler(scaler=MaxAbsScaler())},
                ),
                (
                    {
                        "series": sine_univariate3 + 2,
                        "past_covariates": sine_univariate1 * 3 + 3,
                    },
                    {"past_covariates": Scaler()},
                ),
                (
                    {
                        "series": sine_univariate3 + 5,
                        "future_covariates": sine_univariate1 * (-4) + 3,
                    },
                    {"future_covariates": Scaler(scaler=MaxAbsScaler())},
                ),
                (
                    {
                        "series": sine_univariate3 * 2 + 7,
                        "past_covariates": sine_univariate1 + 2,
                        "future_covariates": sine_univariate2 + 3,
                    },
                    {"series": Scaler(), "past_covariates": Scaler()},
                ),
            ],
            [True, False],  # retrain
            [True, False],  # last point only
            models,
        ),
    )
    def test_historical_forecasts_with_scaler(self, params):
        """Apply manually the scaler on the target and covariates to compare with automatic scaling for both
        optimized and un-optimized historical forecasts
        """

        (ts, hf_scaler), retrain, last_points_only, model_cls = params
        ocl = 6
        model_params = self.helper_get_model_params(model_cls, ts, ocl)
        model = model_cls(**model_params)

        # local models do not support historical forecast with retrain=False
        if isinstance(model, LocalForecastingModel) and not retrain:
            return
        # skip test when model does not support the covariate
        if ("past_covariates" in ts and not model.supports_past_covariates) or (
            "future_covariates" in ts and not model.supports_future_covariates
        ):
            return

        # pre-train on the entire unscaled target, overfitting/accuracy is not important
        if not retrain:
            model.fit(**ts)
            for ts_name in hf_scaler.keys():
                hf_scaler[ts_name].fit(ts[ts_name])

        hf_args = {
            "start": -ocl - 1,  # in order to get 2 forecasts since stride=1
            "start_format": "position",
            "forecast_horizon": ocl,
            "stride": 1,
            "retrain": retrain,
            "overlap_end": False,
            "last_points_only": last_points_only,
            "verbose": False,
            "enable_optimization": False,
        }
        # un-transformed series, scaler applied within the method
        hf_auto = model.historical_forecasts(
            **ts,
            **hf_args,
            data_transformers=hf_scaler,
        )

        hf_auto_pipeline = model.historical_forecasts(
            **ts,
            **hf_args,
            data_transformers={
                key_: Pipeline([val_]) for key_, val_ in hf_scaler.items()
            },
        )

        # verify that the results are identical when using single Scaler or a Pipeline
        assert len(hf_auto) == len(hf_auto_pipeline) == 2
        self.helper_compare_hf(hf_auto, hf_auto_pipeline)

        # optimized historical forecast since horizon_length <= ocl and retrain=False
        if not retrain:
            opti_hf_args = {**hf_args, **{"enable_optimization": True}}
            assert opti_hf_args["enable_optimization"]

            opti_hf_auto = model.historical_forecasts(
                **ts,
                **opti_hf_args,
                data_transformers=hf_scaler,
            )
            assert len(opti_hf_auto) == len(hf_auto) == 2
            self.helper_compare_hf(hf_auto, opti_hf_auto)

        # for 2nd to last historical forecast
        manual_hf_0 = self.helper_manual_scaling_prediction(
            model, ts, hf_scaler, retrain, -ocl - 1, ocl
        )
        # for last historical forecast
        manual_hf_1 = self.helper_manual_scaling_prediction(
            model, ts, hf_scaler, retrain, -ocl, ocl
        )

        # verify that automatic and manual pre-scaling produce identical forecasts
        if last_points_only:
            tmp_ts = TimeSeries.from_times_and_values(
                times=manual_hf_1.time_index[-2:],
                values=np.array([manual_hf_0.values()[-1], manual_hf_1.values()[-1]]),
                columns=manual_hf_0.components,
            )
            self.helper_compare_hf(tmp_ts, hf_auto)
        else:
            self.helper_compare_hf(hf_auto, [manual_hf_0, manual_hf_1])

    def test_historical_forecasts_with_scaler_errors(self, caplog):
        """Check that the appropriate exception is raised when providing incorrect parameters or the expected
        warning is display in the corner cases."""
        ocl = 2
        hf_args = {
            "start": -ocl - 1,
            "start_format": "position",
            "forecast_horizon": ocl,
            "verbose": False,
        }
        model = LinearRegressionModel(lags=5, output_chunk_length=ocl)
        model.fit(self.sine_univariate1)

        # retrain=False and unfitted data transformers
        with pytest.raises(ValueError) as err:
            model.historical_forecasts(
                **hf_args,
                series=self.sine_univariate1,
                data_transformers={"series": Scaler()},
                retrain=False,
            )
        assert str(err.value).startswith(
            "All the fittable entries in `data_transformers` must already be fitted when `retrain=False`, the "
        )

        # retrain=False, multiple series not matching the fitted data transformers dimensions
        with pytest.raises(ValueError) as err:
            model.historical_forecasts(
                **hf_args,
                series=[self.sine_univariate1] * 2,
                data_transformers={
                    "series": Scaler(global_fit=False).fit([self.sine_univariate1] * 3)
                },
                retrain=False,
            )
        assert str(err.value).startswith(
            "When multiple series are provided, their number should match the number of "
            "`TimeSeries` used to fit the data transformers `n=3`"
        )

        # retrain=True, multiple series and unfitted data transformers with global_fit=True
        expected_warning = (
            "When `retrain=True` and multiple series are provided, the fittable `data_transformers` "
            "are trained on each series independently (`global_fit=True` will be ignored)."
        )
        with caplog.at_level(logging.WARNING):
            model.historical_forecasts(
                **hf_args,
                series=[self.sine_univariate1, self.sine_univariate2],
                data_transformers={"series": Scaler(global_fit=True)},
                retrain=True,
            )
            assert expected_warning in caplog.text

        # data transformer (global_fit=False) prefitted on several series but only series is forecasted
        expected_warning = (
            "Provided only a single series, but at least one of the `data_transformers` "
            "that use `global_fit=False` was fitted on multiple `TimeSeries`."
        )
        with caplog.at_level(logging.WARNING):
            model.historical_forecasts(
                **hf_args,
                series=[self.sine_univariate2],
                data_transformers={
                    "series": Scaler(global_fit=False).fit([
                        self.sine_univariate1,
                        self.sine_univariate2,
                    ])
                },
                retrain=False,
            )
            assert expected_warning in caplog.text

    @pytest.mark.parametrize("params", product([True, False], [True, False]))
    def test_historical_forecasts_with_scaler_multiple_series(self, params):
        """Verify that the scaling in historical forecasts behave as expected when multiple series are used.

        The difference in behavior is caused by the difference in number of parameters when a scaler is fitted on
        a single series/multiple series with global_fit=True or with multplie series with global_fit=False.
        """
        retrain, global_fit = params
        # due to either of the argument, the scaler will have only one set of parameters
        unique_param_entry = retrain or global_fit
        ocl = 2
        hf_args = {
            "start": -ocl,
            "start_format": "position",
            "forecast_horizon": ocl,
            "last_points_only": False,
            "retrain": retrain,
            "verbose": False,
        }
        series = [self.sine_univariate1, self.sine_univariate2, self.sine_univariate3]

        model = LinearRegressionModel(lags=5, output_chunk_length=ocl)
        model.fit(series)

        def get_scaler(fit: bool):
            if fit:
                return Scaler(global_fit=global_fit).fit(series)
            else:
                return Scaler(global_fit=global_fit)

        # using all the series used to fit the scaler
        hf = model.historical_forecasts(
            **hf_args,
            series=series,
            data_transformers={"series": get_scaler(fit=True)},
        )
        manual_hf_0 = self.helper_manual_scaling_prediction(
            model,
            {"series": series[0]},
            {"series": get_scaler(fit=True)},
            retrain,
            -ocl,
            ocl,
            series_idx=None if unique_param_entry else 0,
        )
        manual_hf_1 = self.helper_manual_scaling_prediction(
            model,
            {"series": series[1]},
            {"series": get_scaler(fit=True)},
            retrain,
            -ocl,
            ocl,
            series_idx=None if unique_param_entry else 1,
        )
        manual_hf_2 = self.helper_manual_scaling_prediction(
            model,
            {"series": series[2]},
            {"series": get_scaler(fit=True)},
            retrain,
            -ocl,
            ocl,
            series_idx=None if unique_param_entry else 2,
        )
        self.helper_compare_hf(hf, [[manual_hf_0], [manual_hf_1], [manual_hf_2]])

        # scaler fit on 3 series, historical forecast only over the first one
        hf = model.historical_forecasts(
            **hf_args,
            series=series[0],
            data_transformers={"series": get_scaler(fit=True)},
        )
        manual_hf_0 = self.helper_manual_scaling_prediction(
            model,
            {"series": series[0]},
            {"series": get_scaler(fit=True)},
            retrain,
            -ocl,
            ocl,
        )
        self.helper_compare_hf(hf, [manual_hf_0])

        # scaler fit on 3 series, historical forecast only over the last one, causing a mismatch
        hf = model.historical_forecasts(
            **hf_args,
            series=series[2],
            data_transformers={"series": get_scaler(fit=True)},
        )
        # note that the series_idx is not specified, only the first transformer is used (instead of the 3rd)
        manual_hf_2 = self.helper_manual_scaling_prediction(
            model,
            {"series": series[2]},
            {"series": get_scaler(fit=True)},
            retrain,
            -ocl,
            ocl,
        )
        self.helper_compare_hf(hf, [manual_hf_2])

        # data_transformers are not pre-fitted
        if retrain:
            hf = model.historical_forecasts(
                **hf_args,
                series=series,
                data_transformers={"series": get_scaler(fit=False)},
            )
            manual_hf_0 = self.helper_manual_scaling_prediction(
                model,
                {"series": series[0]},
                {"series": get_scaler(fit=False)},
                retrain,
                -ocl,
                ocl,
            )
            manual_hf_1 = self.helper_manual_scaling_prediction(
                model,
                {"series": series[1]},
                {"series": get_scaler(fit=False)},
                retrain,
                -ocl,
                ocl,
            )
            manual_hf_2 = self.helper_manual_scaling_prediction(
                model,
                {"series": series[2]},
                {"series": get_scaler(fit=False)},
                retrain,
                -ocl,
                ocl,
            )
            self.helper_compare_hf(hf, [[manual_hf_0], [manual_hf_1], [manual_hf_2]])

    @pytest.mark.parametrize(
        "model_type,enable_optimization",
        product(["regression", "torch"], [True, False]),
    )
    def test_fit_kwargs(self, model_type, enable_optimization):
        """check that the parameters provided in fit_kwargs are correctly processed"""
        valid_fit_kwargs = {"max_samples_per_ts": 3}
        invalid_fit_kwargs = {"series": self.ts_pass_train}
        unsupported_fit_kwargs = {"unsupported": "unsupported"}

        n = 2
        model = self.create_model(1, use_ll=False, model_type=model_type)

        # torch not available
        if model is None:
            return

        model.fit(series=self.ts_pass_train[:-n])

        # supported argument
        hist_fc = model.historical_forecasts(
            self.ts_pass_train,
            forecast_horizon=1,
            num_samples=1,
            start=len(self.ts_pass_train) - n,
            retrain=True,
            enable_optimization=enable_optimization,
            fit_kwargs=valid_fit_kwargs,
        )

        assert hist_fc.components.equals(self.ts_pass_train.components)
        assert len(hist_fc) == n

        # passing unsupported argument
        with pytest.raises(TypeError):
            hist_fc = model.historical_forecasts(
                self.ts_pass_train,
                forecast_horizon=1,
                start=len(self.ts_pass_train) - n,
                retrain=True,
                enable_optimization=enable_optimization,
                fit_kwargs=unsupported_fit_kwargs,
            )

        # passing hist_fc parameters in fit_kwargs, with retrain=False
        hist_fc = model.historical_forecasts(
            self.ts_pass_train,
            forecast_horizon=1,
            start=len(self.ts_pass_train) - n,
            retrain=False,
            enable_optimization=enable_optimization,
            fit_kwargs=invalid_fit_kwargs,
        )

        assert hist_fc.components.equals(self.ts_pass_train.components)
        assert len(hist_fc) == n

        # passing hist_fc parameters in fit_kwargs, interfering with the logic
        with pytest.raises(ValueError) as msg:
            model.historical_forecasts(
                self.ts_pass_train,
                forecast_horizon=1,
                start=len(self.ts_pass_train) - n,
                retrain=True,
                enable_optimization=enable_optimization,
                fit_kwargs=invalid_fit_kwargs,
            )
        assert str(msg.value).startswith(
            "The following parameters cannot be passed in `fit_kwargs`"
        )

    @pytest.mark.parametrize(
        "model_type,enable_optimization",
        product(["regression", "torch"], [True, False]),
    )
    def test_predict_kwargs(self, model_type, enable_optimization):
        """check that the parameters provided in predict_kwargs are correctly processed"""
        invalid_predict_kwargs = {"predict_likelihood_parameters": False}
        unsupported_predict_kwargs = {"unsupported": "unsupported"}
        if model_type == "regression":
            valid_predict_kwargs = {}
        else:
            valid_predict_kwargs = {"batch_size": 10}

        n = 2
        model = self.create_model(1, use_ll=False, model_type=model_type)

        # torch not available
        if model is None:
            return

        model.fit(series=self.ts_pass_train[:-n])

        # supported argument
        hist_fc = model.historical_forecasts(
            self.ts_pass_train,
            forecast_horizon=1,
            start=len(self.ts_pass_train) - n,
            retrain=False,
            enable_optimization=enable_optimization,
            predict_kwargs=valid_predict_kwargs,
        )

        assert hist_fc.components.equals(self.ts_pass_train.components)
        assert len(hist_fc) == n

        # passing unsupported prediction argument
        with pytest.raises(TypeError):
            hist_fc = model.historical_forecasts(
                self.ts_pass_train,
                forecast_horizon=1,
                start=len(self.ts_pass_train) - n,
                retrain=False,
                enable_optimization=enable_optimization,
                predict_kwargs=unsupported_predict_kwargs,
            )

        # passing hist_fc parameters in predict_kwargs, interfering with the logic
        with pytest.raises(ValueError) as msg:
            hist_fc = model.historical_forecasts(
                self.ts_pass_train,
                forecast_horizon=1,
                start=len(self.ts_pass_train) - n,
                retrain=False,
                enable_optimization=enable_optimization,
                predict_kwargs=invalid_predict_kwargs,
            )
        assert str(msg.value).startswith(
            "The following parameters cannot be passed in `predict_kwargs`"
        )

    @pytest.mark.parametrize(
        "config",
        product(["regression", "torch"], [True, False], [True, False]),
    )
    def test_sample_weight(self, config):
        """check that passing sample weights work and that it yields different results than without sample weights."""
        model_type, manual_weight, multi_series = config
        ts = self.ts_pass_train
        if manual_weight:
            sample_weight = np.linspace(0, 1, len(ts))
            sample_weight = ts.with_values(np.expand_dims(sample_weight, -1))
        else:
            sample_weight = "linear"

        if multi_series:
            ts = [ts] * 2
            sample_weight = [sample_weight] * 2 if manual_weight else sample_weight

        model_kwargs = (
            {"n_epochs": 3, "optimizer_kwargs": {"lr": 0.1}}
            if model_type == "torch"
            else {}
        )
        model = self.create_model(
            1, use_ll=False, model_type=model_type, **model_kwargs
        )

        # torch not available
        if model is None:
            return

        start_kwargs = {"start": -1, "start_format": "position"}
        hfc_non_weighted = model.historical_forecasts(series=ts, **start_kwargs)

        model = self.create_model(1, use_ll=False, model_type=model_type)
        hfc_weighted = model.historical_forecasts(
            series=ts, sample_weight=sample_weight, **start_kwargs
        )

        if not multi_series:
            hfc_weighted = [hfc_weighted]
            hfc_non_weighted = [hfc_non_weighted]

        # check that the predictions are different
        for hfc_nw, hfc_w in zip(hfc_non_weighted, hfc_weighted):
            with pytest.raises(AssertionError):
                np.testing.assert_array_almost_equal(
                    hfc_w.all_values(), hfc_nw.all_values()
                )

        if manual_weight:
            if multi_series:
                sample_weight[1] = sample_weight[1][1:]
                invalid_idx = 1
            else:
                sample_weight = sample_weight[:-1]
                invalid_idx = 0

            with pytest.raises(ValueError) as err:
                _ = model.historical_forecasts(
                    series=ts, sample_weight=sample_weight, **start_kwargs
                )
            assert (
                str(err.value)
                == f"`sample_weight` at series index {invalid_idx} must contain "
                f"at least all times of the corresponding target `series`."
            )

    def test_historical_forecast_additional_sanity_checks(self):
        model = LinearRegressionModel(lags=1)

        # `stride <= 0`
        with pytest.raises(ValueError) as err:
            _ = model.historical_forecasts(
                series=self.ts_pass_train,
                stride=0,
            )
        assert (
            str(err.value)
            == "The provided stride parameter must be a positive integer."
        )

        # start_format="position" but `start` is not `int`
        with pytest.raises(ValueError) as err:
            _ = model.historical_forecasts(
                series=self.ts_pass_train,
                start=pd.Timestamp("01-01-2020"),
                start_format="position",
            )
        assert str(err.value).startswith(
            "Since `start_format='position'`, `start` must be an integer, received"
        )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [False, True],  # use covariates
            [True, False],  # last points only
            [True, False],  # overlap end
            [1, 3],  # stride
            [
                3,  # horizon < ocl
                5,  # horizon == ocl
                7,  # horizon > ocl -> autoregression
            ],
            [False, True],  # use integer indexed series
            [False, True],  # use multi-series
            [0, 1],  # output chunk shift
        ),
    )
    def test_conformal_historical_forecasts(self, config):
        """Tests historical forecasts output naive conformal model with last points only, covariates, stride,
        different horizons and overlap end.
        Tests that the returned dimensions, lengths and start / end times are correct.
        """
        (
            use_covs,
            last_points_only,
            overlap_end,
            stride,
            horizon,
            use_int_idx,
            use_multi_series,
            ocs,
        ) = config
        q = [0.1, 0.5, 0.9]
        pred_lklp = {"num_samples": 1, "predict_likelihood_parameters": True}
        # compute minimum series length to generate n forecasts
        icl = 3
        ocl = 5
        horizon_ocs = horizon + ocs
        min_len_val_series = icl + horizon_ocs + int(not overlap_end) * horizon_ocs
        n_forecasts = 3
        # get train and val series of that length
        series = self.ts_pass_val[: min_len_val_series + n_forecasts - 1]
        if use_int_idx:
            series = TimeSeries.from_values(
                values=series.all_values(),
                columns=series.columns,
            )
        # check that too short input raises error
        series_too_short = series[:-n_forecasts]

        # optionally, generate covariates
        if use_covs:
            pc = tg.gaussian_timeseries(
                start=series.start_time(),
                end=series.end_time() + max(0, horizon - ocl) * series.freq,
                freq=series.freq,
            )
            fc = tg.gaussian_timeseries(
                start=series.start_time(),
                end=series.end_time() + (max(ocl, horizon) + ocs) * series.freq,
                freq=series.freq,
            )
        else:
            pc, fc = None, None

        # first train the ForecastingModel
        model_kwargs = (
            {}
            if not use_covs
            else {"lags_past_covariates": icl, "lags_future_covariates": (icl, ocl)}
        )
        forecasting_model = LinearRegressionModel(
            lags=icl, output_chunk_length=ocl, output_chunk_shift=ocs, **model_kwargs
        )
        forecasting_model.fit(series, past_covariates=pc, future_covariates=fc)

        # add an offset and rename columns in second series to make sure that conformal hist fc works as expected
        if use_multi_series:
            series = [
                series,
                (series + 10).shift(1).with_columns_renamed(series.columns, "test_col"),
            ]
            pc = [pc, pc.shift(1)] if pc is not None else None
            fc = [fc, fc.shift(1)] if fc is not None else None

        # conformal model
        model = ConformalNaiveModel(forecasting_model, quantiles=q)

        hfc_kwargs = dict(
            {
                "retrain": False,
                "last_points_only": last_points_only,
                "overlap_end": overlap_end,
                "stride": stride,
                "forecast_horizon": horizon,
            },
            **pred_lklp,
        )
        # cannot perform auto regression with output chunk shift
        if ocs and horizon > ocl:
            with pytest.raises(ValueError) as exc:
                _ = model.historical_forecasts(
                    series=series,
                    past_covariates=pc,
                    future_covariates=fc,
                    **hfc_kwargs,
                )
            assert str(exc.value).startswith("Cannot perform auto-regression")
            return

        # compute conformal historical forecasts
        hist_fct = model.historical_forecasts(
            series=series, past_covariates=pc, future_covariates=fc, **hfc_kwargs
        )
        # raises error with too short target series
        with pytest.raises(ValueError) as exc:
            _ = model.historical_forecasts(
                series=series_too_short,
                past_covariates=pc,
                future_covariates=fc,
                **hfc_kwargs,
            )
        assert str(exc.value).startswith(
            "Could not build the minimum required calibration input with the provided `series`"
        )

        if not isinstance(series, list):
            series = [series]
            hist_fct = [hist_fct]

        for (
            series_,
            hfc,
        ) in zip(series, hist_fct):
            if not isinstance(hfc, list):
                hfc = [hfc]

            n_preds_with_overlap = (
                len(series_)
                - icl  # input for first prediction
                - horizon_ocs  # skip first forecasts to avoid look-ahead bias
                + 1  # minimum one forecast
            )
            if not last_points_only:
                # last points only = False gives a list of forecasts per input series
                # where each forecast contains the predictions over the entire horizon
                n_pred_series_expected = n_preds_with_overlap
                n_pred_points_expected = horizon
                first_ts_expected = series_.time_index[icl] + series_.freq * (
                    horizon_ocs + ocs
                )
                last_ts_expected = series_.end_time() + series_.freq * horizon_ocs
                # no overlapping means less predictions
                if not overlap_end:
                    n_pred_series_expected -= horizon_ocs
            else:
                # last points only = True gives one contiguous time series per input series
                # with only predictions from the last point in the horizon
                n_pred_series_expected = 1
                n_pred_points_expected = n_preds_with_overlap
                first_ts_expected = series_.time_index[icl] + series_.freq * (
                    horizon_ocs + ocs + horizon - 1
                )
                last_ts_expected = series_.end_time() + series_.freq * horizon_ocs
                # no overlapping means less predictions
                if not overlap_end:
                    n_pred_points_expected -= horizon_ocs

            # no overlapping means less predictions
            if not overlap_end:
                last_ts_expected -= series_.freq * horizon_ocs

            # adapt based on stride
            if stride > 1:
                if not last_points_only:
                    n_pred_series_expected = n_pred_series_expected // stride + int(
                        n_pred_series_expected % stride
                    )
                else:
                    n_pred_points_expected = n_pred_points_expected // stride + int(
                        n_pred_points_expected % stride
                    )
                first_ts_expected = hfc[0].start_time()
                last_ts_expected = hfc[-1].end_time()

            cols_excpected = likelihood_component_names(
                series_.columns, quantile_names(q)
            )
            # check length match between optimized and default hist fc
            assert len(hfc) == n_pred_series_expected
            # check hist fc start
            assert hfc[0].start_time() == first_ts_expected
            # check hist fc end
            assert hfc[-1].end_time() == last_ts_expected
            for hfc_ in hfc:
                assert hfc_.columns.tolist() == cols_excpected
                assert len(hfc_) == n_pred_points_expected

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [False, True],  # last points only
            [None, 1, 2],  # cal length
            [False, True],  # use start
            ["value", "position"],  # start format
            [False, True],  # use integer indexed series
            [False, True],  # use multi-series
            [0, 1],  # output chunk shift
        ),
    )
    def test_conformal_historical_start_cal_length(self, config):
        """Tests naive conformal model historical forecasts without `cal_stride`."""
        (
            last_points_only,
            cal_length,
            use_start,
            start_format,
            use_int_idx,
            use_multi_series,
            ocs,
        ) = config
        q = [0.1, 0.5, 0.9]
        pred_lklp = {"num_samples": 1, "predict_likelihood_parameters": True}
        # compute minimum series length to generate n forecasts
        icl = 3
        ocl = 5
        horizon = 5
        horizon_ocs = horizon + ocs
        add_cal_length = cal_length - 1 if cal_length is not None else 0
        add_start = 2 * int(use_start)
        min_len_val_series = icl + 2 * horizon_ocs + add_cal_length + add_start
        n_forecasts = 3
        # get train and val series of that length
        series = self.ts_pass_val[: min_len_val_series + n_forecasts - 1]

        if use_int_idx:
            series = TimeSeries.from_values(
                values=series.all_values(),
                columns=series.columns,
            )

        # first train the ForecastingModel
        forecasting_model = LinearRegressionModel(
            lags=icl,
            output_chunk_length=ocl,
            output_chunk_shift=ocs,
        )
        forecasting_model.fit(series)

        # optionally compute the start as a positional index
        start_position = icl + horizon_ocs + add_cal_length + add_start
        start = None
        if use_start:
            if start_format == "value":
                start = series.time_index[start_position]
            else:
                start = start_position

        # add an offset and rename columns in second series to make sure that conformal hist fc works as expected
        if use_multi_series:
            series = [
                series,
                (series + 10).shift(1).with_columns_renamed(series.columns, "test_col"),
            ]

        # compute conformal historical forecasts (skips some of the first forecasts to get minimum required cal set)
        model = ConformalNaiveModel(
            forecasting_model, quantiles=q, cal_length=cal_length
        )
        hist_fct = model.historical_forecasts(
            series=series,
            retrain=False,
            start=start,
            start_format=start_format,
            last_points_only=last_points_only,
            forecast_horizon=horizon,
            overlap_end=False,
            **pred_lklp,
        )

        if not isinstance(series, list):
            series = [series]
            hist_fct = [hist_fct]

        for idx, (
            series_,
            hfc,
        ) in enumerate(zip(series, hist_fct)):
            if not isinstance(hfc, list):
                hfc = [hfc]

            # multi series: second series is shifted by one time step (+/- idx);
            # start_format = "value" requires a shift
            add_start_series_2 = idx * int(use_start) * int(start_format == "value")

            n_preds_without_overlap = (
                len(series_)
                - icl  # input for first prediction
                - horizon_ocs  # skip first forecasts to avoid look-ahead bias
                - horizon_ocs  # cannot compute with `overlap_end=False`
                + 1  # minimum one forecast
                - add_cal_length  # skip based on train length
                - add_start  # skip based on start
                + add_start_series_2  # skip based on start if second series
            )
            if not last_points_only:
                n_pred_series_expected = n_preds_without_overlap
                n_pred_points_expected = horizon
                # seconds series is shifted by one time step (- idx)
                first_ts_expected = series_.time_index[
                    start_position - add_start_series_2 + ocs
                ]
                last_ts_expected = series_.end_time()
            else:
                n_pred_series_expected = 1
                n_pred_points_expected = n_preds_without_overlap
                # seconds series is shifted by one time step (- idx)
                first_ts_expected = (
                    series_.time_index[start_position - add_start_series_2]
                    + (horizon_ocs - 1) * series_.freq
                )
                last_ts_expected = series_.end_time()

            cols_excpected = likelihood_component_names(
                series_.columns, quantile_names(q)
            )
            # check historical forecasts dimensions
            assert len(hfc) == n_pred_series_expected
            # check hist fc start
            assert hfc[0].start_time() == first_ts_expected
            # check hist fc end
            assert hfc[-1].end_time() == last_ts_expected
            for hfc_ in hfc:
                assert hfc_.columns.tolist() == cols_excpected
                assert len(hfc_) == n_pred_points_expected

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [False, True],  # last points only
            [None, 2],  # cal length
            ["value", "position"],  # start format
            [2, 4],  # stride
            [1, 2],  # cal stride
            [0, 1],  # output chunk shift
        ),
    )
    def test_conformal_historical_forecast_start_stride(self, caplog, config):
        """Tests naive conformal model with `start` being the first forecastable index is identical to a start
        before forecastable index (including stride, cal stride).
        """
        (
            last_points_only,
            cal_length,
            start_format,
            stride,
            cal_stride,
            ocs,
        ) = config
        q = [0.1, 0.5, 0.9]
        pred_lklp = {"num_samples": 1, "predict_likelihood_parameters": True}
        # compute minimum series length to generate n forecasts
        icl = 3
        ocl = 5
        horizon = 2

        # the position of the first conformal forecast start point without look-ahead bias; assuming min cal_length=1
        horizon_ocs = math.ceil((horizon + ocs) / cal_stride) * cal_stride
        # adjust by the number of calibration examples
        add_cal_length = cal_stride * (cal_length - 1) if cal_length is not None else 0
        # the minimum series length is the sum of the above, plus the length of one forecast (horizon + ocs)
        min_len_val_series = icl + horizon_ocs + add_cal_length + horizon + ocs
        n_forecasts = 3
        # to get `n_forecasts` with `stride`, we need more points
        n_forecasts_stride = stride * n_forecasts - int(1 % stride > 0)
        # get train and val series of that length
        series = tg.linear_timeseries(
            length=min_len_val_series + n_forecasts_stride - 1
        )

        # first train the ForecastingModel
        forecasting_model = LinearRegressionModel(
            lags=icl,
            output_chunk_length=ocl,
            output_chunk_shift=ocs,
        )
        forecasting_model.fit(series)

        # optionally compute the start as a positional index
        start_position = icl + horizon_ocs + add_cal_length
        if start_format == "value":
            start = series.time_index[start_position]
            start_too_early = series.time_index[start_position - 1]
            start_too_early_stride = series.time_index[start_position - stride]
        else:
            start = start_position
            start_too_early = start_position - 1
            start_too_early_stride = start_position - stride
        start_first_fc = series.time_index[start_position] + series.freq * (
            horizon + ocs - 1 if last_points_only else ocs
        )
        too_early_warn_exp = "is before the first predictable/trainable historical"

        hfc_params = {
            "series": series,
            "retrain": False,
            "start_format": start_format,
            "stride": stride,
            "last_points_only": last_points_only,
            "forecast_horizon": horizon,
        }
        # compute regular historical forecasts
        hist_fct_all = forecasting_model.historical_forecasts(start=start, **hfc_params)
        assert len(hist_fct_all) == n_forecasts
        assert hist_fct_all[0].start_time() == start_first_fc
        assert (
            hist_fct_all[1].start_time() - stride * series.freq
            == hist_fct_all[0].start_time()
        )

        # compute conformal historical forecasts (starting at first possible conformal forecast)
        model = ConformalNaiveModel(
            forecasting_model, quantiles=q, cal_length=cal_length, cal_stride=cal_stride
        )
        with caplog.at_level(logging.WARNING):
            hist_fct = model.historical_forecasts(
                start=start, **hfc_params, **pred_lklp
            )
            assert too_early_warn_exp not in caplog.text
        caplog.clear()
        assert len(hist_fct) == len(hist_fct_all)
        assert hist_fct_all[0].start_time() == hist_fct[0].start_time()
        assert (
            hist_fct[1].start_time() - stride * series.freq == hist_fct[0].start_time()
        )

        # start one earlier gives warning
        with caplog.at_level(logging.WARNING):
            _ = model.historical_forecasts(
                start=start_too_early, **hfc_params, **pred_lklp
            )
            assert too_early_warn_exp in caplog.text
        caplog.clear()

        # starting stride before first valid start, gives identical results
        hist_fct_too_early = model.historical_forecasts(
            start=start_too_early_stride, **hfc_params, **pred_lklp
        )
        assert hist_fct_too_early == hist_fct
