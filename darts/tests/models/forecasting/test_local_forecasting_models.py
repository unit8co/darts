import numpy as np
import pandas as pd

from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.logging import get_logger
from darts.metrics import mape
from darts.models import (
    ARIMA,
    FFT,
    VARIMA,
    Croston,
    ExponentialSmoothing,
    FourTheta,
    KalmanForecaster,
    NaiveSeasonal,
    StatsForecastAutoARIMA,
    Theta,
)
from darts.tests.base_test_class import DartsBaseTestClass
from darts.timeseries import TimeSeries
from darts.utils import timeseries_generation as tg
from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode

logger = get_logger(__name__)

try:
    from darts.models import LinearRegressionModel, RandomForest

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning(
        "Torch not installed - some local forecasting models tests will be skipped"
    )
    TORCH_AVAILABLE = False

# (forecasting models, maximum error) tuples
models = [
    (ExponentialSmoothing(), 5.6),
    (ARIMA(12, 2, 1), 10),
    (ARIMA(1, 1, 1), 40),
    (StatsForecastAutoARIMA(period=12), 4.8),
    (Croston(version="classic"), 34),
    (Croston(version="tsb", alpha_d=0.1, alpha_p=0.1), 34),
    (Theta(), 11.3),
    (Theta(1), 20.2),
    (Theta(-1), 9.8),
    (FourTheta(1), 20.2),
    (FourTheta(-1), 9.8),
    (FourTheta(trend_mode=TrendMode.EXPONENTIAL), 5.5),
    (FourTheta(model_mode=ModelMode.MULTIPLICATIVE), 11.4),
    (FourTheta(season_mode=SeasonalityMode.ADDITIVE), 14.2),
    (FFT(trend="poly"), 11.4),
    (NaiveSeasonal(), 32.4),
    (KalmanForecaster(dim_x=3), 17.0),
]

if TORCH_AVAILABLE:
    models += [
        (LinearRegressionModel(lags=12), 11.0),
        (RandomForest(lags=12, n_estimators=200, max_depth=3), 15.5),
    ]

# forecasting models with exogenous variables support
multivariate_models = [
    (VARIMA(1, 0, 0), 55.6),
    (VARIMA(1, 1, 1), 57.0),
    (KalmanForecaster(dim_x=30), 30.0),
]

dual_models = [ARIMA(), StatsForecastAutoARIMA(period=12)]


try:
    from darts.models import Prophet

    models.append((Prophet(), 13.5))
    dual_models.append(Prophet())
except ImportError:
    logger.warning("Prophet not installed - will be skipping Prophet tests")

try:
    from darts.models import BATS, TBATS, AutoARIMA

    models.append((AutoARIMA(), 12.2))
    models.append((TBATS(use_trend=True, use_arma_errors=True, use_box_cox=True), 8.0))
    models.append((BATS(use_trend=True, use_arma_errors=True, use_box_cox=True), 10.0))
    dual_models.append(AutoARIMA())
    PMDARIMA_AVAILABLE = True
except ImportError:
    logger.warning("pmdarima not installed - will be skipping AutoARIMA tests")
    PMDARIMA_AVAILABLE = False

try:
    from darts.models import TCNModel  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not installed - will be skipping Torch models tests")
    TORCH_AVAILABLE = False


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

    def test_save_model_parameters(self):
        # model creation parameters were saved before. check if re-created model has same params as original
        for model, _ in models:
            self.assertTrue(
                model._model_params == model.untrained_model()._model_params
            )

    def test_models_runnability(self):
        for model, _ in models:
            prediction = model.fit(self.ts_gaussian).predict(self.forecasting_horizon)
            self.assertTrue(len(prediction) == self.forecasting_horizon)

    def test_models_performance(self):
        # for every model, check whether its errors do not exceed the given bounds
        for model, max_mape in models:
            np.random.seed(1)  # some models are probabilist...
            model.fit(self.ts_pass_train)
            prediction = model.predict(len(self.ts_pass_val))
            current_mape = mape(prediction, self.ts_pass_val)
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
            current_mape = mape(prediction, self.ts_ice_heater_val)
            self.assertTrue(
                current_mape < max_mape,
                "{} model exceeded the maximum MAPE of {}. "
                "with a MAPE of {}".format(str(model), max_mape, current_mape),
            )

    def test_multivariate_input(self):
        es_model = ExponentialSmoothing()
        ts_passengers_enhanced = self.ts_passengers.add_datetime_attribute("month")
        with self.assertRaises(AssertionError):
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
            times=tg._generate_index(start=0, length=len(self.ts_gaussian)),
            values=self.ts_gaussian.all_values(copy=False),
        )
        fc_num_idx = TimeSeries.from_times_and_values(
            times=tg._generate_index(start=0, length=len(self.ts_gaussian_long)),
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

    def test_dummy_series(self):
        values = np.random.uniform(low=-10, high=10, size=100)
        ts = TimeSeries.from_dataframe(pd.DataFrame({"V1": values}))

        varima = VARIMA(trend="t")
        with self.assertRaises(ValueError):
            varima.fit(series=ts)

        if PMDARIMA_AVAILABLE:
            autoarima = AutoARIMA(trend="t")
            with self.assertRaises(ValueError):
                autoarima.fit(series=ts)
