import shutil

import numpy as np
import pandas as pd

from .base_test_class import DartsBaseTestClass
from ..timeseries import TimeSeries
from ..utils import timeseries_generation as tg
from ..metrics import mape
from ..models import (
    NaiveSeasonal,
    ExponentialSmoothing,
    ARIMA,
    Theta,
    FourTheta,
    FFT,
    VARIMA,
    RandomForest,
    LinearRegressionModel,
)
from ..utils.utils import SeasonalityMode, TrendMode, ModelMode
from ..logging import get_logger
from ..datasets import AirPassengersDataset, IceCreamHeaterDataset

logger = get_logger(__name__)

# (forecasting models, maximum error) tuples
models = [
    (ExponentialSmoothing(), 5.6),
    (ARIMA(12, 2, 1), 10),
    (ARIMA(1, 1, 1), 40),
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
    (LinearRegressionModel(lags=12), 11.0),
    (RandomForest(lags=12, n_estimators=200, max_depth=3), 15.5),
]
# forecasting models with exogenous variables support
multivariate_models = [
    (VARIMA(1, 0, 0), 55.6),
    (VARIMA(1, 1, 1), 57.0),
]

dual_models = [ARIMA()]


try:
    from ..models import Prophet
    models.append((Prophet(), 13.5))
except ImportError:
    logger.warning("Prophet not installed - will be skipping Prophet tests")

try:
    from ..models import AutoARIMA

    models.append((AutoARIMA(), 12.2))
    dual_models.append(AutoARIMA())
    PMDARIMA_AVAILABLE = True
except ImportError:
    logger.warning("pmdarima not installed - will be skipping AutoARIMA tests")
    PMDARIMA_AVAILABLE = False

try:
    from ..models import TCNModel

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

    # real timeseries for functionality tests
    ts_passengers = AirPassengersDataset().load()
    ts_pass_train, ts_pass_val = ts_passengers.split_after(pd.Timestamp("19570101"))

    # real multivariate timeseries for functionality tests
    ts_ice_heater = IceCreamHeaterDataset().load()
    ts_ice_heater_train, ts_ice_heater_val = ts_ice_heater.split_after(split_point=0.7)

    def test_models_runnability(self):
        for model, _ in models:
            model.fit(self.ts_gaussian)
            prediction = model.predict(self.forecasting_horizon)
            self.assertTrue(len(prediction) == self.forecasting_horizon)

    def test_models_performance(self):
        # for every model, check whether its errors do not exceed the given bounds
        for model, max_mape in models:
            np.random.seed(1)  # some models are probabilist...
            model.fit(self.ts_pass_train)
            prediction = model.predict(len(self.ts_pass_val))
            current_mape = mape(prediction, self.ts_pass_val)
            self.assertTrue(current_mape < max_mape, "{} model exceeded the maximum MAPE of {}. "
                            "with a MAPE of {}".format(str(model), max_mape, current_mape))

    def test_multivariate_models_performance(self):
        # for every model, check whether its errors do not exceed the given bounds
        for model, max_mape in multivariate_models:
            np.random.seed(1)
            model.fit(self.ts_ice_heater_train)
            prediction = model.predict(len(self.ts_ice_heater_val))
            current_mape = mape(prediction, self.ts_ice_heater_val)
            self.assertTrue(current_mape < max_mape, "{} model exceeded the maximum MAPE of {}. "
                            "with a MAPE of {}".format(str(model), max_mape, current_mape))

    def test_multivariate_input(self):
        es_model = ExponentialSmoothing()
        ts_passengers_enhanced = self.ts_passengers.add_datetime_attribute("month")
        with self.assertRaises(AssertionError):
            es_model.fit(ts_passengers_enhanced)
        es_model.fit(ts_passengers_enhanced["#Passengers"])
        with self.assertRaises(KeyError):
            es_model.fit(ts_passengers_enhanced["2"])

    def test_exogenous_variables_support(self):
        for model in dual_models:

            # Test models runnability
            model.fit(self.ts_gaussian, future_covariates=self.ts_gaussian)

            prediction = model.predict(
                self.forecasting_horizon,
                future_covariates=tg.gaussian_timeseries(
                    length=self.forecasting_horizon,
                    start=self.ts_gaussian.end_time() + self.ts_gaussian.freq))

            self.assertTrue(len(prediction) == self.forecasting_horizon)

            # Test mismatch in length between exogenous variables and forecasting horizon
            with self.assertRaises(ValueError):
                model.predict(
                    self.forecasting_horizon,
                    future_covariates=tg.gaussian_timeseries(length=self.forecasting_horizon - 1))

            # Test mismatch in time-index/length between series and exogenous variables
            with self.assertRaises(ValueError):
                model.fit(self.ts_gaussian, future_covariates=self.ts_gaussian[:-1])
            with self.assertRaises(ValueError):
                model.fit(self.ts_gaussian[1:], future_covariates=self.ts_gaussian[:-1])

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
