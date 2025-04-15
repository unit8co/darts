import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from statsforecast.models import ADIDA as SF_ADIDA
from statsforecast.models import GARCH as SF_GARCH
from statsforecast.models import MSTL as SF_MSTL
from statsforecast.models import AutoARIMA as SF_AutoARIMA
from statsforecast.models import AutoETS as SF_AutoETS
from statsforecast.models import AutoRegressive as SF_AutoRegressive
from statsforecast.models import SeasonalNaive as SF_SeasonalNaive
from statsforecast.models import SimpleExponentialSmoothing as SF_ETS
from statsforecast.models import SimpleExponentialSmoothing as SF_SETS
from statsforecast.models import SklearnModel as SF_SklearnModel
from statsforecast.models import Theta as SF_Theta

import darts.utils.timeseries_generation as tg
from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.metrics import mae
from darts.models import (
    AutoARIMA,
    AutoCES,
    AutoETS,
    AutoMFLES,
    AutoTBATS,
    AutoTheta,
    Croston,
    LinearRegressionModel,
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
    series, _ = series.split_after(pd.Timestamp("19570101"))

    fc = tg.datetime_attribute_timeseries(
        series, attribute="month", cyclic=True, add_length=12
    )
    # as future covariates we want a trend
    trend_values = np.arange(start=1, stop=len(series) + 1)
    ts_trend = TimeSeries.from_times_and_values(
        times=series.time_index, values=trend_values, columns=["trend"]
    )

    @pytest.mark.parametrize(
        "model",
        [
            # comment: (transferable series support type, future cov support type)
            AutoARIMA(season_length=12),  # (native, native),
            AutoMFLES(season_length=12, test_size=12),  # (custom, native)
            AutoETS(season_length=12),  # (native, custom)
            StatsForecastModel(SF_ETS(alpha=0.1)),  # (custom, custom)
        ][1:2],
    )
    def test_transferrable_series(self, model):
        n = 12
        series = self.series[:24]
        fc = self.fc[: 24 + n]
        model_1 = model.untrained_model().fit(series)
        pred_11 = model_1.predict(n=n)
        pred_12 = model_1.predict(n=n, series=series)
        assert pred_11 == pred_12

        model_2 = model.untrained_model().fit(series, future_covariates=fc)
        pred_21 = model_2.predict(n=n)
        pred_22 = model_2.predict(n=n, series=series)
        pred_23 = model_2.predict(n=n, series=series, future_covariates=fc)
        assert pred_21 == pred_22 == pred_23

    @pytest.mark.parametrize(
        "model",
        [
            AutoETS(season_length=12, model="ZZZ"),
            StatsForecastModel(SF_AutoETS(season_length=12, model="ZZZ")),
        ],
    )
    def test_custom_fc_support_fit_on_residuals(self, model):
        """AutoETS does not support future covariates natively. Check that Darts' OLS trick is applied."""
        # test if we are indeed fitting the AutoETS on the residuals of the linear regression
        model.fit(series=self.series, future_covariates=self.ts_trend)

        # create the residuals from the linear regression
        fitted_values_linreg = model._linreg.model.predict(
            X=self.ts_trend.values(copy=False)
        )
        fitted_values_linreg_ts = TimeSeries.from_times_and_values(
            times=self.series.time_index, values=fitted_values_linreg
        )
        resids = self.series - fitted_values_linreg_ts

        # now make in-sample predictions with the AutoETS model
        in_sample_preds = model.model.predict_in_sample()["fitted"]
        ts_in_sample_preds = TimeSeries.from_times_and_values(
            times=self.series.time_index, values=in_sample_preds
        )

        # compare in-sample predictions to the residuals they have supposedly been fitted on
        current_mae = mae(resids, ts_in_sample_preds)
        assert current_mae < 9

    @pytest.mark.parametrize(
        "model",
        [
            AutoETS(season_length=12, model="ZZZ"),
            StatsForecastModel(SF_AutoETS(season_length=12, model="ZZZ")),
        ],
    )
    def test_custom_fc_support_fit_a_linreg(self, model):
        """AutoETS does not support future covariates natively. Check that Darts' OLS trick is applied."""
        model.fit(series=self.series, future_covariates=self.ts_trend)

        # check if linear regression was fit
        assert model._linreg is not None
        assert model._linreg._fit_called

        # fit a linear regression
        linreg = LinearRegressionModel(lags_future_covariates=[0])
        linreg.fit(series=self.series, future_covariates=self.ts_trend)

        # check if the linear regression was fit on the same data by checking if the coefficients are equal
        assert model._linreg.model.coef_ == linreg.model.coef_
