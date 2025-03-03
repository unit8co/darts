from itertools import product

from sklearn.preprocessing import MaxAbsScaler

from darts.dataprocessing.transformers import Scaler
from darts.models import LinearRegressionModel
from darts.tests.utils.historical_forecasts.test_historical_forecasts import (
    TestHistoricalforecast,
)
from darts.utils import timeseries_generation as tg

tester = TestHistoricalforecast()

sine_univariate1 = tg.sine_timeseries(length=50) * 2 + 1.5
sine_univariate2 = tg.sine_timeseries(length=50, value_phase=1.5705) * 5 + 1.5
sine_univariate3 = tg.sine_timeseries(length=50, value_phase=0.1963125) * -9 + 1.5

params = product(
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
    [LinearRegressionModel],
)

i = 0

for param in params:
    if i == 2 or i == 3:
        tester.test_historical_forecasts_with_scaler(params=param)
    i += 1
