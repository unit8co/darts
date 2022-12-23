import copy
import os
import shutil
import tempfile
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
    NaiveSeasonal,
    Prophet,
    RandomForest,
    RegressionModel,
    StatsForecastAutoARIMA,
    StatsForecastETS,
    Theta,
)
from darts.models.forecasting.forecasting_model import (
    LocalForecastingModel,
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.tests.base_test_class import DartsBaseTestClass
from darts.timeseries import TimeSeries
from darts.utils import timeseries_generation as tg
from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode

logger = get_logger(__name__)

# (forecasting models, maximum error) tuples
models = [
    (ExponentialSmoothing(), 5.4),
    (ARIMA(12, 2, 1), 5.2),
    (ARIMA(1, 1, 1), 24),
    (StatsForecastAutoARIMA(season_length=12), 4.6),
    (StatsForecastETS(season_length=12, model="AAZ"), 4.1),
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
    (Prophet(), 9.0),
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
]

dual_models = [
    ARIMA(),
    StatsForecastAutoARIMA(season_length=12),
    StatsForecastETS(season_length=12),
    Prophet(),
    AutoARIMA(),
]

# test only a few models for encoder support reduce time
encoder_support_models = [
    VARIMA(1, 0, 0),
    ARIMA(),
    AutoARIMA(),
    Prophet(),
    KalmanForecaster(dim_x=30),
]


class LocalForecastingModelsTestCase(DartsBaseTestClass):

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

    def setUp(self):
        self.temp_work_dir = tempfile.mkdtemp(prefix="darts")

    def tearDown(self):
        shutil.rmtree(self.temp_work_dir)

    def test_save_model_parameters(self):
        # model creation parameters were saved before. check if re-created model has same params as original
        for model, _ in models:
            self.assertTrue(
                model._model_params == model.untrained_model()._model_params
            )

    def test_save_load_model(self):
        # check if save and load methods work and if loaded model creates same forecasts as original model
        cwd = os.getcwd()
        os.chdir(self.temp_work_dir)

        for model in [ARIMA(1, 1, 1), LinearRegressionModel(lags=12)]:
            model_path_str = type(model).__name__
            model_path_file = model_path_str + "_file"
            model_paths = [model_path_str, model_path_file]
            full_model_paths = [
                os.path.join(self.temp_work_dir, p) for p in model_paths
            ]

            model.fit(self.ts_gaussian)
            model_prediction = model.predict(self.forecasting_horizon)

            # test save
            model.save()
            model.save(model_path_str)
            with open(model_path_file, "wb") as f:
                model.save(f)

            for p in full_model_paths:
                self.assertTrue(os.path.exists(p))

            self.assertTrue(
                len(
                    [
                        p
                        for p in os.listdir(self.temp_work_dir)
                        if p.startswith(type(model).__name__)
                    ]
                )
                == 3
            )

            # test load
            loaded_model_str = type(model).load(model_path_str)
            loaded_model_file = type(model).load(model_path_file)
            loaded_models = [loaded_model_str, loaded_model_file]

            for loaded_model in loaded_models:
                self.assertEqual(
                    model_prediction, loaded_model.predict(self.forecasting_horizon)
                )

        os.chdir(cwd)

    def test_models_runnability(self):
        for model, _ in models:
            if not isinstance(model, RegressionModel):
                self.assertTrue(isinstance(model, LocalForecastingModel))
            prediction = model.fit(self.ts_gaussian).predict(self.forecasting_horizon)
            self.assertTrue(len(prediction) == self.forecasting_horizon)

    def test_models_performance(self):
        # for every model, check whether its errors do not exceed the given bounds
        for model, max_mape in models:
            np.random.seed(1)  # some models are probabilist...
            model.fit(self.ts_pass_train)
            prediction = model.predict(len(self.ts_pass_val))
            current_mape = mape(self.ts_pass_val, prediction)
            self.assertTrue(
                current_mape < max_mape,
                "{} model exceeded the maximum MAPE of {}. "
                "with a MAPE of {}".format(str(model), max_mape, current_mape),
            )

    def test_multivariate_models_performance(self):
        # for every model, check whether its errors do not exceed the given bounds
        for model, max_mape in multivariate_models:
            np.random.seed(1)
            model.fit(self.ts_ice_heater_train)
            prediction = model.predict(len(self.ts_ice_heater_val))
            current_mape = mape(self.ts_ice_heater_val, prediction)
            self.assertTrue(
                current_mape < max_mape,
                "{} model exceeded the maximum MAPE of {}. "
                "with a MAPE of {}".format(str(model), max_mape, current_mape),
            )

    def test_multivariate_input(self):
        es_model = ExponentialSmoothing()
        ts_passengers_enhanced = self.ts_passengers.add_datetime_attribute("month")
        with self.assertRaises(ValueError):
            es_model.fit(ts_passengers_enhanced)
        es_model.fit(ts_passengers_enhanced["#Passengers"])
        with self.assertRaises(KeyError):
            es_model.fit(ts_passengers_enhanced["2"])

    def test_exogenous_variables_support(self):
        # test case with pd.DatetimeIndex
        target_dt_idx = self.ts_gaussian
        fc_dt_idx = self.ts_gaussian_long

        # test case with numerical pd.RangeIndex
        target_num_idx = TimeSeries.from_times_and_values(
            times=tg.generate_index(start=0, length=len(self.ts_gaussian)),
            values=self.ts_gaussian.all_values(copy=False),
        )
        fc_num_idx = TimeSeries.from_times_and_values(
            times=tg.generate_index(start=0, length=len(self.ts_gaussian_long)),
            values=self.ts_gaussian_long.all_values(copy=False),
        )

        for target, future_covariates in zip(
            [target_dt_idx, target_num_idx], [fc_dt_idx, fc_num_idx]
        ):
            for model in dual_models:
                # skip models which do not support RangeIndex
                if isinstance(target.time_index, pd.RangeIndex):
                    try:
                        # _supports_range_index raises a ValueError if model does not support RangeIndex
                        model._supports_range_index()
                    except ValueError:
                        continue

                # Test models runnability - proper future covariates slicing
                model.fit(target, future_covariates=future_covariates)
                prediction = model.predict(
                    self.forecasting_horizon, future_covariates=future_covariates
                )

                self.assertTrue(len(prediction) == self.forecasting_horizon)

                # Test mismatch in length between exogenous variables and forecasting horizon
                with self.assertRaises(ValueError):
                    model.predict(
                        self.forecasting_horizon,
                        future_covariates=tg.gaussian_timeseries(
                            start=future_covariates.start_time(),
                            length=self.forecasting_horizon - 1,
                        ),
                    )

                # Test mismatch in time-index/length between series and exogenous variables
                with self.assertRaises(ValueError):
                    model.fit(target, future_covariates=target[:-1])
                with self.assertRaises(ValueError):
                    model.fit(target[1:], future_covariates=target[:-1])

    def test_encoders_support(self):
        # test case with pd.DatetimeIndex
        n = 3

        target = self.ts_gaussian[:-3]
        future_covariates = self.ts_gaussian

        add_encoders = {"custom": {"future": [lambda x: x.dayofweek]}}

        # test some models that do not support encoders
        no_support_model_cls = [NaiveMean, Theta]
        for model_cls in no_support_model_cls:
            with pytest.raises(TypeError):
                _ = model_cls(add_encoders=add_encoders)

        # test some models that support encoders
        for model_object in encoder_support_models:
            series = (
                target
                if not isinstance(model_object, VARIMA)
                else target.stack(target.map(np.log))
            )
            # test once with user supplied covariates, and once without
            for fc in [future_covariates, None]:
                model_params = {
                    k: vals
                    for k, vals in copy.deepcopy(model_object.model_params).items()
                }
                model_params["add_encoders"] = add_encoders
                model = model_object.__class__(**model_params)

                # Test models with user supplied covariates
                model.fit(series, future_covariates=fc)

                prediction = model.predict(n, future_covariates=fc)
                self.assertTrue(len(prediction) == n)

                if isinstance(model, TransferableFutureCovariatesLocalForecastingModel):
                    prediction = model.predict(n, series=series, future_covariates=fc)
                    self.assertTrue(len(prediction) == n)

    def test_dummy_series(self):
        values = np.random.uniform(low=-10, high=10, size=100)
        ts = TimeSeries.from_dataframe(pd.DataFrame({"V1": values}))

        varima = VARIMA(trend="t")
        with self.assertRaises(ValueError):
            varima.fit(series=ts)

        autoarima = AutoARIMA(trend="t")
        with self.assertRaises(ValueError):
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
        self.assertTrue(
            all(pred.time_index == pd.RangeIndex(start=50, stop=60, step=2))
        )

        # datetime-index
        ts = tg.constant_timeseries(start=pd.Timestamp("20130101"), length=20, value=1)
        model = NaiveSeasonal(K=1)
        model.fit(ts)
        pred = model.predict(n=5)
        self.assertEqual(pred.start_time(), pd.Timestamp("20130121"))
        self.assertEqual(pred.end_time(), pd.Timestamp("20130125"))

    def test_statsmodels_future_models(self):

        # same tests, but VARIMA requires to work on a multivariate target series
        UNIVARIATE = "univariate"
        MULTIVARIATE = "multivariate"

        params = [
            (ARIMA, {}, UNIVARIATE),
            (VARIMA, {"d": 0}, MULTIVARIATE),
            (VARIMA, {"d": 1}, MULTIVARIATE),
        ]

        for model_cls, kwargs, model_type in params:
            pred_len = 5
            if model_type == MULTIVARIATE:
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
            pred1 = model.predict(n=pred_len)
            pred2 = model.predict(n=pred_len, series=series2)

            # check probabilistic forecast
            n_samples = 3
            pred1 = model.predict(n=pred_len, num_samples=n_samples)
            pred2 = model.predict(n=pred_len, series=series2, num_samples=n_samples)

            # check that the results with a second custom ts are different from the results given with the training ts
            self.assertFalse(np.array_equal(pred1.values, pred2.values()))

            # check runnability with exogeneous variables
            model = model_cls(**kwargs)
            model.fit(series1, future_covariates=exog1)
            pred1 = model.predict(n=pred_len, future_covariates=exog1)
            pred2 = model.predict(n=pred_len, series=series2, future_covariates=exog2)

            self.assertFalse(np.array_equal(pred1.values(), pred2.values()))

            # check runnability with future covariates with extra time steps in the past compared to the target series
            model = model_cls(**kwargs)
            model.fit(series1, future_covariates=exog1_longer)
            pred1 = model.predict(n=pred_len, future_covariates=exog1_longer)
            pred2 = model.predict(
                n=pred_len, series=series2, future_covariates=exog2_longer
            )

            # check error is raised if model expects covariates but those are not passed when predicting with new data
            with self.assertRaises(ValueError):
                model = model_cls(**kwargs)
                model.fit(series1, future_covariates=exog1)
                model.predict(n=pred_len, series=series2)

            # check error is raised if new future covariates are not wide enough for prediction (on the original series)
            with self.assertRaises(ValueError):
                model = model_cls(**kwargs)
                model.fit(series1, future_covariates=exog1)
                model.predict(n=pred_len, future_covariates=exog1[:-pred_len])

            # check error is raised if new future covariates are not wide enough for prediction (on a new series)
            with self.assertRaises(ValueError):
                model = model_cls(**kwargs)
                model.fit(series1, future_covariates=exog1)
                model.predict(
                    n=pred_len, series=series2, future_covariates=exog2[:-pred_len]
                )
            # and checking the case with insufficient historic future covariates
            with self.assertRaises(ValueError):
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

            self.assertTrue(np.array_equal(pred1.values(), pred3.values()))

            # check backtesting with retrain=False
            model: TransferableFutureCovariatesLocalForecastingModel = model_cls(
                **kwargs
            )
            model.backtest(series1, future_covariates=exog1, start=0.5, retrain=False)

    @patch("typing.Callable")
    def test_backtest_retrain(
        self,
        patch_retrain_func,
    ):
        """
        Test backtest method with different retrain arguments
        """

        series = self.ts_pass_train

        lr_univ_args = {"lags": [-1, -2, -3]}

        lr_multi_args = {
            "lags": [-1, -2, -3],
            "lags_past_covariates": [-1, -2, -3],
        }
        params = [  # tuple of (model, retrain-able, multivariate, retrain parameter, model type)
            (ExponentialSmoothing(), False, False, "hello", "LocalForecastingModel"),
            (ExponentialSmoothing(), False, False, True, "LocalForecastingModel"),
            (ExponentialSmoothing(), False, False, -2, "LocalForecastingModel"),
            (ExponentialSmoothing(), False, False, 2, "LocalForecastingModel"),
            (
                ExponentialSmoothing(),
                False,
                False,
                patch_retrain_func,
                "LocalForecastingModel",
            ),
            (
                LinearRegressionModel(**lr_univ_args),
                True,
                False,
                True,
                "GlobalForecastingModel",
            ),
            (
                LinearRegressionModel(**lr_univ_args),
                True,
                False,
                2,
                "GlobalForecastingModel",
            ),
            (
                LinearRegressionModel(**lr_univ_args),
                True,
                False,
                patch_retrain_func,
                "GlobalForecastingModel",
            ),
            (
                LinearRegressionModel(**lr_multi_args),
                True,
                True,
                True,
                "GlobalForecastingModel",
            ),
            (
                LinearRegressionModel(**lr_multi_args),
                True,
                True,
                2,
                "GlobalForecastingModel",
            ),
            (
                LinearRegressionModel(**lr_multi_args),
                True,
                True,
                patch_retrain_func,
                "GlobalForecastingModel",
            ),
        ]

        for model_cls, retrainable, multivariate, retrain, model_type in params:

            if (
                not isinstance(retrain, (int, bool, Callable))
                or (isinstance(retrain, int) and retrain < 0)
                or (isinstance(retrain, (Callable)) and (not retrainable))
                or ((retrain != 1) and (not retrainable))
            ):
                with self.assertRaises(ValueError):
                    _ = model_cls.historical_forecasts(series, retrain=retrain)

            else:

                if isinstance(retrain, Mock):
                    # resets patch_retrain_func call_count to 0
                    retrain.call_count = 0
                    retrain.side_effect = [True, False] * (len(series) // 2)

                fit_method_to_patch = f"darts.models.forecasting.forecasting_model.{model_type}._fit_wrapper"
                predict_method_to_patch = f"darts.models.forecasting.forecasting_model.{model_type}._predict_wrapper"

                with patch(fit_method_to_patch) as patch_fit_method:
                    with patch(
                        predict_method_to_patch, side_effect=series
                    ) as patch_predict_method:

                        # Set _fit_called attribute to True, otherwise retrain function is never called
                        model_cls._fit_called = True

                        # run backtest
                        _ = model_cls.historical_forecasts(
                            series,
                            past_covariates=series if multivariate else None,
                            retrain=retrain,
                        )

                        assert patch_predict_method.call_count > 1
                        assert patch_fit_method.call_count > 1

                        if isinstance(retrain, Mock):
                            # check that patch_retrain_func has been called at each iteration
                            assert retrain.call_count > 1
