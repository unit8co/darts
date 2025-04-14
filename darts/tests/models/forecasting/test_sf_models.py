import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from statsforecast.models import ADIDA as SF_ADIDA
from statsforecast.models import GARCH as SF_GARCH
from statsforecast.models import MSTL as SF_MSTL
from statsforecast.models import AutoARIMA as SF_AutoARIMA
from statsforecast.models import AutoMFLES as SF_AutoMFLES
from statsforecast.models import AutoRegressive as SF_AutoRegressive
from statsforecast.models import SeasonalNaive as SF_SeasonalNaive
from statsforecast.models import SimpleExponentialSmoothing as SF_SETS
from statsforecast.models import SklearnModel as SF_SklearnModel
from statsforecast.models import Theta as SF_Theta

import darts.utils.timeseries_generation as tg
from darts.datasets import AirPassengersDataset
from darts.models import (
    AutoARIMA,
    AutoCES,
    AutoETS,
    AutoMFLES,
    AutoTBATS,
    AutoTheta,
    Croston,
    StatsForecastModel,
)

sf_models = [
    (SF_AutoARIMA, {"season_length": 12}),
    (SF_AutoRegressive, {"lags": 12}),
    (SF_Theta, {"season_length": 12}),
    (SF_MSTL, {"season_length": 12}),
    (SF_GARCH, {}),
    (SF_SeasonalNaive, {"season_length": 12}),
    (SF_SETS, {"alpha": 0.1}),
    (SF_ADIDA, {}),
    (SF_SklearnModel, {"model": LinearRegression()}),
]

darts_models = [
    (AutoARIMA, {"season_length": 12}),
    (AutoETS, {"season_length": 12}),
    (AutoCES, {"season_length": 12}),
    (AutoTheta, {"season_length": 12}),
    (AutoTBATS, {"season_length": 12}),
    (AutoMFLES, {"season_length": 12, "test_size": 12}),
    (Croston, {}),
    (StatsForecastModel, {"model": SF_AutoARIMA(season_length=12)}),
]


class TestSFModels:
    series = AirPassengersDataset().load().astype(np.float32)
    fc = tg.datetime_attribute_timeseries(
        series, attribute="month", cyclic=True, add_length=12
    )

    @pytest.mark.parametrize(
        "model",
        [
            AutoARIMA(season_length=12),  # native support
            AutoMFLES(season_length=12, test_size=12),  # custom support
            StatsForecastModel(SF_AutoARIMA(season_length=12)),  # custom support
            StatsForecastModel(
                SF_AutoMFLES(season_length=12, test_size=12)
            ),  # custom support
        ],
    )
    def test_transferrable_series(self, model):
        series = self.series[:24]
        model.fit(series)
        pred1 = model.predict(n=12)
        pred2 = model.predict(n=12, series=series)
        assert pred1 == pred2

    def test_probabilistic_forecast(self):
        pass
