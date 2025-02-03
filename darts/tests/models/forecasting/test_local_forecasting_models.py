import copy
import itertools
import os
import pathlib
from typing import Callable
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.logging import get_logger
from darts.metrics import mape
from darts.models import (
    ARIMA,
    BATS,
    FFT,
    TBATS,
    VARIMA,
    AutoARIMA,
    Croston,
    ExponentialSmoothing,
    FourTheta,
    KalmanForecaster,
    LinearRegressionModel,
    NaiveDrift,
    NaiveMean,
    NaiveMovingAverage,
    NaiveSeasonal,
    NotImportedModule,
    Prophet,
    RandomForest,
    RegressionModel,
    StatsForecastAutoARIMA,
    StatsForecastAutoCES,
    StatsForecastAutoETS,
    StatsForecastAutoTBATS,
    StatsForecastAutoTheta,
    Theta,
)
from darts.models.forecasting.forecasting_model import (
    LocalForecastingModel,
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.timeseries import TimeSeries
from darts.utils import timeseries_generation as tg
from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode, generate_index

logger = get_logger(__name__)

# (forecasting models, maximum error) tuples
models = [
    (ExponentialSmoothing(), 5.4),
    (ARIMA(12, 2, 1), 5.2),
    (ARIMA(1, 1, 1), 24),
    (StatsForecastAutoARIMA(season_length=12), 4.6),
    (StatsForecastAutoTheta(season_length=12), 5.5),
    (StatsForecastAutoCES(season_length=12, model="Z"), 7.3),
    (StatsForecastAutoETS(season_length=12, model="AAZ"), 7.3),
    (StatsForecastAutoTBATS(season_length=12), 10),
    (Croston(version="classic"), 23),
    (Croston(version="tsb", alpha_d=0.1, alpha_p=0.1), 23),
    (Theta(), 11),
    (Theta(1), 17),
    (Theta(-1), 12),
    (FourTheta(1), 17),
    (FourTheta(-1), 12),
    (FourTheta(trend_mode=TrendMode.EXPONENTIAL), 6.0),
    (FourTheta(model_mode=ModelMode.MULTIPLICATIVE), 10),
    (FourTheta(season_mode=SeasonalityMode.ADDITIVE), 12),
    (FFT(trend="poly"), 13),
    (KalmanForecaster(dim_x=3), 20),
    (LinearRegressionModel(lags=12), 13),
    (RandomForest(lags=12, n_estimators=5, max_depth=3), 14),
    (AutoARIMA(), 12),
    (TBATS(use_trend=True, use_arma_errors=True, use_box_cox=True), 8.5),
    (BATS(use_trend=True, use_arma_errors=True, use_box_cox=True), 11),
]

# forecasting models with exogenous variables support
multivariate_models = [
    (VARIMA(1, 0, 0), 32),
    (VARIMA(1, 1, 1), 25),
    (KalmanForecaster(dim_x=30), 16),
    (NaiveSeasonal(), 32),
    (NaiveMean(), 37),
    (NaiveDrift(), 39),
    (NaiveMovingAverage(input_chunk_length=5), 34),
]

dual_models = [
    ARIMA(),
    StatsForecastAutoARIMA(season_length=12),
    StatsForecastAutoETS(season_length=12),
    AutoARIMA(),
]

# test only a few models for encoder support reduce time
encoder_support_models = [
    VARIMA(1, 0, 0),
    ARIMA(),
    AutoARIMA(),
    KalmanForecaster(dim_x=30),
]
if not isinstance(Prophet, NotImportedModule):
    models.append((Prophet(), 9.0))
    dual_models.append(Prophet())
    encoder_support_models.append(Prophet())


class TestLocalForecastingModels:
    # forecasting horizon used in runnability tests
    forecasting_horizon = 5

    # dummy timeseries for runnability tests
    np.random.seed(1)
    ts_gaussian = tg.gaussian_timeseries(length=100, mean=50)
    # for testing covariate slicing
    ts_gaussian_long = tg.gaussian_timeseries(
        length=len(ts_gaussian) + 2 * forecasting_horizon,
        start=ts_gaussian.start_time() - forecasting_horizon * ts_gaussian.freq,
        mean=50,
    )

    # real timeseries for functionality tests
    ts_passengers = AirPassengersDataset().load()
    ts_pass_train, ts_pass_val = ts_passengers.split_after(pd.Timestamp("19570101"))

    # real multivariate timeseries for functionality tests
    ts_ice_heater = IceCreamHeaterDataset().load()
    ts_ice_heater_train, ts_ice_heater_val = ts_ice_heater.split_after(split_point=0.7)

    def retrain_func(
        counter, pred_time, train_series, past_covariates, future_covariates
    ):
        return len(train_series) % 2 == 0

    def test_save_model_parameters(self):
        # model creation parameters were saved before. check if re-created model has same params as original
        for model, _ in models:
            assert model._model_params == model.untrained_model()._model_params

    @pytest.mark.parametrize("model", [ARIMA(1, 1, 1)])
    def test_save_load_model(self, tmpdir_module, model):
        # check if save and load methods work and if loaded model creates same forecasts as original model
        model_path_str = type(model).__name__
        model_path_pathlike = pathlib.Path(model_path_str + "_pathlike")
        model_path_binary = model_path_str + "_binary"
        model_paths = [model_path_str, model_path_pathlike, model_path_binary]
        full_model_paths = [os.path.join(tmpdir_module, p) for p in model_paths]

        model.fit(self.ts_gaussian)
        model_prediction = model.predict(self.forecasting_horizon)

        # test save
        model.save()
        model.save(model_path_str)
        model.save(model_path_pathlike)
        with open(model_path_binary, "wb") as f:
            model.save(f)

        for p in full_model_paths:
            assert os.path.exists(p)

        assert (
            len([
                p
                for p in os.listdir(tmpdir_module)
                if p.startswith(type(model).__name__)
            ])
            == len(full_model_paths) + 1
        )

        # test load
        loaded_model_str = type(model).load(model_path_str)
        loaded_model_pathlike = type(model).load(model_path_pathlike)
        with open(model_path_binary, "rb") as f:
            loaded_model_binary = type(model).load(f)
        loaded_models = [
            loaded_model_str,
            loaded_model_pathlike,
            loaded_model_binary,
        ]

        for loaded_model in loaded_models:
            assert model_prediction == loaded_model.predict(self.forecasting_horizon)

    def test_save_load_model_invalid_path(self):
        # check if save and load methods raise an error when given an invalid path
        model = ARIMA(1, 1, 1)
        model.fit(self.ts_gaussian)

        # Use a byte string as path (, which is not supported)
        model_path_invalid = b"invalid_path"

        # test save
        with pytest.raises(ValueError):
            model.save(model_path_invalid)

        # test load
        with pytest.raises(ValueError):
            type(model).load(model_path_invalid)

    @pytest.mark.parametrize("config", models)
    def test_models_runnability(self, config):
        model, _ = config
        if not isinstance(model, RegressionModel):
            assert isinstance(model, LocalForecastingModel)
        prediction = model.fit(self.ts_gaussian).predict(self.forecasting_horizon)
        assert len(prediction) == self.forecasting_horizon

    @pytest.mark.parametrize("config", models)
    def test_models_performance(self, config):
        # for every model, check whether its errors do not exceed the given bounds
        model, max_mape = config
        np.random.seed(1)  # some models are probabilist...
        model.fit(self.ts_pass_train)
        prediction = model.predict(len(self.ts_pass_val))
        current_mape = mape(self.ts_pass_val, prediction)
        assert current_mape < max_mape, (
            f"{str(model)} model exceeded the maximum MAPE of {max_mape}. "
            f"with a MAPE of {current_mape}"
        )

    @pytest.mark.parametrize("config", multivariate_models)
    def test_multivariate_models_performance(self, config):
        # for every model, check whether its errors do not exceed the given bounds
        model, max_mape = config
        np.random.seed(1)
        model.fit(self.ts_ice_heater_train)
        prediction = model.predict(len(self.ts_ice_heater_val))
        current_mape = mape(self.ts_ice_heater_val, prediction)
        assert current_mape < max_mape, (
            f"{str(model)} model exceeded the maximum MAPE of {max_mape}. "
            f"with a MAPE of {current_mape}"
        )

    def test_multivariate_input(self):
        es_model = ExponentialSmoothing()
        ts_passengers_enhanced = self.ts_passengers.add_datetime_attribute("month")
        with pytest.raises(ValueError):
            es_model.fit(ts_passengers_enhanced)
        es_model.fit(ts_passengers_enhanced["#Passengers"])
        with pytest.raises(KeyError):
            es_model.fit(ts_passengers_enhanced["2"])

    @pytest.mark.parametrize("model", dual_models)
    def test_exogenous_variables_support(self, model):
        # test case with pd.DatetimeIndex
        target_dt_idx = self.ts_gaussian
        fc_dt_idx = self.ts_gaussian_long

        # test case with numerical pd.RangeIndex
        target_num_idx = TimeSeries.from_times_and_values(
            times=generate_index(start=0, length=len(self.ts_gaussian)),
            values=self.ts_gaussian.all_values(copy=False),
        )
        fc_num_idx = TimeSeries.from_times_and_values(
            times=generate_index(start=0, length=len(self.ts_gaussian_long)),
            values=self.ts_gaussian_long.all_values(copy=False),
        )

        for target, future_covariates in zip(
            [target_dt_idx, target_num_idx], [fc_dt_idx, fc_num_idx]
        ):
            if isinstance(target.time_index, pd.RangeIndex):
                try:
                    # _supports_range_index raises a ValueError if model does not support RangeIndex
                    model._supports_range_index
                except ValueError:
                    continue

            # Test models runnability - proper future covariates slicing
            model.fit(target, future_covariates=future_covariates)
            prediction = model.predict(
                self.forecasting_horizon, future_covariates=future_covariates
            )

            assert len(prediction) == self.forecasting_horizon

            # Test mismatch in length between exogenous variables and forecasting horizon
            with pytest.raises(ValueError):
                model.predict(
                    self.forecasting_horizon,
                    future_covariates=tg.gaussian_timeseries(
                        start=future_covariates.start_time(),
                        length=self.forecasting_horizon - 1,
                    ),
                )

            # Test mismatch in time-index/length between series and exogenous variables
            with pytest.raises(ValueError):
                model.fit(target, future_covariates=target[:-1])
            with pytest.raises(ValueError):
                model.fit(target[1:], future_covariates=target[:-1])

    @pytest.mark.parametrize("model_cls", [NaiveMean, Theta])
    def test_encoders_no_support(self, model_cls):
        # test some models that do not support encoders
        add_encoders = {"custom": {"future": [lambda x: x.dayofweek]}}

        with pytest.raises(TypeError):
            _ = model_cls(add_encoders=add_encoders)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            encoder_support_models, [ts_gaussian, None]
        ),  # tuple of (model class, future covs)
    )
    def test_encoders_support(self, config):
        # test some models that support encoders, once with user supplied covariates and once without
        model_object, fc = config
        # test case with pd.DatetimeIndex
        n = 3

        target = self.ts_gaussian[:-3]

        # encoder must be named function for pickling
        def extract_dayofweek(index):
            return index.dayofweek

        add_encoders = {"custom": {"future": [extract_dayofweek]}}

        series = (
            target
            if not isinstance(model_object, VARIMA)
            else target.stack(target.map(np.log))
        )
        model_params = {
            k: vals for k, vals in copy.deepcopy(model_object.model_params).items()
        }
        model_params["add_encoders"] = add_encoders
        model = model_object.__class__(**model_params)

        # Test models with user supplied covariates
        model.fit(series, future_covariates=fc)

        prediction = model.predict(n, future_covariates=fc)
        assert len(prediction) == n

        if isinstance(model, TransferableFutureCovariatesLocalForecastingModel):
            prediction = model.predict(n, series=series, future_covariates=fc)
            assert len(prediction) == n

    def test_dummy_series(self):
        values = np.random.uniform(low=-10, high=10, size=100)
        ts = TimeSeries.from_dataframe(pd.DataFrame({"V1": values}))

        varima = VARIMA(trend="t")
        with pytest.raises(ValueError):
            varima.fit(series=ts)

        autoarima = AutoARIMA(trend="t")
        with pytest.raises(ValueError):
            autoarima.fit(series=ts)

    def test_forecast_time_index(self):
        # the forecast time index should follow that of the train series

        # integer-index, with step>1
        values = np.random.rand(20)
        idx = pd.RangeIndex(start=10, stop=50, step=2)
        ts = TimeSeries.from_times_and_values(idx, values)

        model = NaiveSeasonal(K=1)
        model.fit(ts)
        pred = model.predict(n=5)
        assert all(pred.time_index == pd.RangeIndex(start=50, stop=60, step=2))

        # datetime-index
        ts = tg.constant_timeseries(start=pd.Timestamp("20130101"), length=20, value=1)
        model = NaiveSeasonal(K=1)
        model.fit(ts)
        pred = model.predict(n=5)
        assert pred.start_time() == pd.Timestamp("20130121")
        assert pred.end_time() == pd.Timestamp("20130125")

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "params",
        [  # tuple of (model class, model params, uni/multivariate case)
            (ARIMA, {}, "univariate"),
            (VARIMA, {"d": 0}, "multivariate"),
            (VARIMA, {"d": 1}, "multivariate"),
        ],
    )
    def test_statsmodels_future_models(self, params):
        # same tests, but VARIMA requires to work on a multivariate target series
        model_cls, kwargs, model_type = params
        pred_len = 5
        if model_type == "multivariate":
            series1 = self.ts_ice_heater_train
            series2 = self.ts_ice_heater_val
        else:
            series1 = self.ts_pass_train
            series2 = self.ts_pass_val

        # creating covariates from series + noise
        noise1 = tg.gaussian_timeseries(length=len(series1))
        noise2 = tg.gaussian_timeseries(length=len(series2))

        for _ in range(1, series1.n_components):
            noise1 = noise1.stack(tg.gaussian_timeseries(length=len(series1)))
            noise2 = noise2.stack(tg.gaussian_timeseries(length=len(series2)))

        exog1 = series1 + noise1
        exog2 = series2 + noise2

        exog1_longer = exog1.concatenate(exog1, ignore_time_axis=True)
        exog2_longer = exog2.concatenate(exog2, ignore_time_axis=True)

        # shortening of pred_len so that exog are enough for the training series prediction
        series1 = series1[:-pred_len]
        series2 = series2[:-pred_len]

        # check runnability with different time series
        model = model_cls(**kwargs)
        model.fit(series1)
        _ = model.predict(n=pred_len)
        _ = model.predict(n=pred_len, series=series2)

        # check probabilistic forecast
        n_samples = 3
        pred1 = model.predict(n=pred_len, num_samples=n_samples)
        pred2 = model.predict(n=pred_len, series=series2, num_samples=n_samples)

        # check that the results with a second custom ts are different from the results given with the training ts
        assert not np.array_equal(pred1.values, pred2.values())

        # check runnability with exogeneous variables
        model = model_cls(**kwargs)
        model.fit(series1, future_covariates=exog1)
        pred1 = model.predict(n=pred_len, future_covariates=exog1)
        pred2 = model.predict(n=pred_len, series=series2, future_covariates=exog2)

        assert not np.array_equal(pred1.values(), pred2.values())

        # check runnability with future covariates with extra time steps in the past compared to the target series
        model = model_cls(**kwargs)
        model.fit(series1, future_covariates=exog1_longer)
        _ = model.predict(n=pred_len, future_covariates=exog1_longer)
        _ = model.predict(n=pred_len, series=series2, future_covariates=exog2_longer)

        # check error is raised if model expects covariates but those are not passed when predicting with new data
        with pytest.raises(ValueError):
            model = model_cls(**kwargs)
            model.fit(series1, future_covariates=exog1)
            model.predict(n=pred_len, series=series2)

        # check error is raised if new future covariates are not wide enough for prediction (on the original series)
        with pytest.raises(ValueError):
            model = model_cls(**kwargs)
            model.fit(series1, future_covariates=exog1)
            model.predict(n=pred_len, future_covariates=exog1[:-pred_len])

        # check error is raised if new future covariates are not wide enough for prediction (on a new series)
        with pytest.raises(ValueError):
            model = model_cls(**kwargs)
            model.fit(series1, future_covariates=exog1)
            model.predict(
                n=pred_len, series=series2, future_covariates=exog2[:-pred_len]
            )
        # and checking the case with insufficient historic future covariates
        with pytest.raises(ValueError):
            model = model_cls(**kwargs)
            model.fit(series1, future_covariates=exog1)
            model.predict(
                n=pred_len, series=series2, future_covariates=exog2[pred_len:]
            )

        # verify that we can still forecast the original training series after predicting a new target series
        model = model_cls(**kwargs)
        model.fit(series1, future_covariates=exog1)
        pred1 = model.predict(n=pred_len, future_covariates=exog1)
        model.predict(n=pred_len, series=series2, future_covariates=exog2)
        pred3 = model.predict(n=pred_len, future_covariates=exog1)

        np.testing.assert_array_equal(pred1.values(), pred3.values())
        model.backtest(series1, future_covariates=exog1, start=0.5, retrain=False)

    @patch("typing.Callable", autospec=retrain_func, return_value=True)
    @pytest.mark.parametrize(
        "params",
        [  # tuple of (model class, retrain-able, multivariate, retrain parameter, model type, uni/multivariate args)
            (
                ExponentialSmoothing,
                False,
                False,
                "hello",
                "LocalForecastingModel",
                None,
            ),
            (ExponentialSmoothing, False, False, True, "LocalForecastingModel", None),
            (ExponentialSmoothing, False, False, -2, "LocalForecastingModel", None),
            (ExponentialSmoothing, False, False, 2, "LocalForecastingModel", None),
            (
                ExponentialSmoothing,
                False,
                False,
                "patch_retrain_func",
                "LocalForecastingModel",
                None,
            ),
            (
                LinearRegressionModel,
                True,
                False,
                True,
                "GlobalForecastingModel",
                "lr_univ_args",
            ),
            (
                LinearRegressionModel,
                True,
                False,
                2,
                "GlobalForecastingModel",
                "lr_univ_args",
            ),
            (
                LinearRegressionModel,
                True,
                False,
                "patch_retrain_func",
                "GlobalForecastingModel",
                "lr_univ_args",
            ),
            (
                LinearRegressionModel,
                True,
                True,
                True,
                "GlobalForecastingModel",
                "lr_multi_args",
            ),
            (
                LinearRegressionModel,
                True,
                True,
                2,
                "GlobalForecastingModel",
                "lr_multi_args",
            ),
            (
                LinearRegressionModel,
                True,
                True,
                "patch_retrain_func",
                "GlobalForecastingModel",
                "lr_multi_args,",
            ),
        ],
    )
    def test_backtest_retrain(self, patch_retrain_func, params):
        """
        Test backtest method with different retrain arguments
        """
        model_cls, retrainable, multivariate, retrain, model_type, variate_args = params
        if variate_args is not None:
            if variate_args == "lr_univ_args":
                model_args = {"lags": [-1, -2, -3]}
            else:  # "lr_multiv_args"
                model_args = {
                    "lags": [-1, -2, -3],
                    "lags_past_covariates": [-1, -2, -3],
                }
        else:
            model_args = dict()
        model = model_cls(**model_args)

        if str(retrain) == "patch_retrain_func":
            retrain = patch_retrain_func

        series = self.ts_pass_train

        if (
            not isinstance(retrain, (int, bool, Callable))
            or (isinstance(retrain, int) and retrain < 0)
            or (isinstance(retrain, (Callable)) and (not retrainable))
            or ((retrain != 1) and (not retrainable))
        ):
            with pytest.raises(ValueError):
                _ = model.historical_forecasts(series, retrain=retrain)

        else:
            if isinstance(retrain, Mock):
                # resets patch_retrain_func call_count to 0
                retrain.call_count = 0
                retrain.side_effect = [True, False] * (len(series) // 2)

            fit_method_to_patch = (
                f"darts.models.forecasting.forecasting_model.{model_type}._fit_wrapper"
            )
            predict_method_to_patch = f"darts.models.forecasting.forecasting_model.{model_type}._predict_wrapper"

            with patch(fit_method_to_patch) as patch_fit_method:
                with patch(
                    predict_method_to_patch, side_effect=series
                ) as patch_predict_method:
                    # Set _fit_called attribute to True, otherwise retrain function is never called
                    model._fit_called = True

                    # run backtest
                    _ = model.historical_forecasts(
                        series,
                        past_covariates=series if multivariate else None,
                        retrain=retrain,
                    )

                    assert patch_predict_method.call_count > 1
                    assert patch_fit_method.call_count > 1

                    if isinstance(retrain, Mock):
                        # check that patch_retrain_func has been called at each iteration
                        assert retrain.call_count > 1

    @pytest.mark.parametrize(
        "config",
        [  # tuple of (model, expected string representation)
            (ExponentialSmoothing(), "ExponentialSmoothing()"),  # no params changed
            (ARIMA(1, 1, 1), "ARIMA(p=1, q=1)"),  # default value for a param
            (
                KalmanForecaster(
                    add_encoders={"cyclic": {"past": ["month"]}}
                ),  # data structure param
                "KalmanForecaster(add_encoders={'cyclic': {'past': ['month']}})",
            ),
            (
                TBATS(
                    use_trend=True, use_arma_errors=True, use_box_cox=True
                ),  # params in wrong order
                "TBATS(use_box_cox=True, use_trend=True)",
            ),
        ],
    )
    def test_model_str_call(self, config):
        model, expected = config
        assert expected == str(model)

    @pytest.mark.parametrize(
        "config",
        [  # tuple of (model, expected representation)
            (
                ExponentialSmoothing(),
                "ExponentialSmoothing(trend=ModelMode.ADDITIVE, damped=False, seasonal=SeasonalityMode.ADDITIVE, "
                + "seasonal_periods=None, random_state=0, kwargs=None)",
            ),  # no params changed
            (
                ARIMA(1, 1, 1),
                "ARIMA(p=1, d=1, q=1, seasonal_order=(0, 0, 0, 0), trend=None, random_state=None, add_encoders=None)",
            ),  # default value for a param
        ],
    )
    def test_model_repr_call(self, config):
        model, expected = config
        assert expected == repr(model)
