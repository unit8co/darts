import math

import numpy as np
import pandas as pd
import pytest
from statsforecast.models import AutoETS as SF_AutoETS
from statsforecast.models import SimpleExponentialSmoothing as SF_ETS
from statsforecast.utils import ConformalIntervals

import darts.utils.timeseries_generation as tg
from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.metrics import mae
from darts.models import (
    AutoARIMA,
    AutoETS,
    AutoMFLES,
    Croston,
    LinearRegressionModel,
    StatsForecastModel,
)
from darts.utils.likelihood_models.statsforecast import QuantilePrediction


class TestSFModels:
    series = AirPassengersDataset().load().astype(np.float32)
    series, _ = series.split_after(pd.Timestamp("19570101"))

    fc = tg.datetime_attribute_timeseries(
        series, attribute="month", cyclic=True, add_length=12, dtype=np.float32
    )
    # as future covariates we want a trend
    trend_values = np.arange(start=1, stop=len(series) + 1)
    ts_trend = TimeSeries.from_times_and_values(
        times=series.time_index, values=trend_values, columns=["trend"]
    )

    @pytest.mark.parametrize(
        "config",
        [
            # comment: (transferable series support type, future cov support type)
            (AutoARIMA, {"season_length": 12}),  # (native, native)
            (AutoMFLES, {"season_length": 12, "test_size": 12}),  # (custom, native)
            (AutoETS, {"season_length": 12}),  # (native, custom)
            (StatsForecastModel, {"model": SF_ETS(alpha=0.1)}),  # (custom, custom)
        ],
    )
    def test_transferable_series_forecast(self, config):
        model_cls, kwargs = config

        n = 12
        series = self.series[:24]
        fc = self.fc[: 24 + n]

        # series only
        model = model_cls(**kwargs)
        model_1 = model.untrained_model().fit(series)

        # passing fc but model trained without
        with pytest.raises(ValueError) as exc:
            _ = model.predict(n=n, future_covariates=fc)
        assert str(exc.value).startswith(
            "The model was trained without future covariates"
        )

        pred_11 = model_1.predict(n=n).all_values()
        pred_12 = model_1.predict(n=n, series=series).all_values()
        np.testing.assert_array_almost_equal(pred_11, pred_12)

        pred_13 = model_1.predict(n=n, series=series[:12]).all_values()
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(pred_11, pred_13)

        # with future covariates
        model_2 = model.untrained_model().fit(series, future_covariates=fc)

        # using fc with wrong dimensionality
        with pytest.raises(ValueError) as exc:
            _ = model_2.predict(n=n, future_covariates=fc.stack(fc))
        assert str(exc.value).startswith(
            "The `future_covariates` passed to `predict()` must have "
            "the same number of components"
        )

        pred_21 = model_2.predict(n=n).all_values()
        pred_22 = model_2.predict(n=n, series=series).all_values()
        pred_23 = model_2.predict(n=n, series=series, future_covariates=fc).all_values()
        np.testing.assert_array_almost_equal(pred_21, pred_22)
        np.testing.assert_array_almost_equal(pred_22, pred_23)

        pred_24 = model_2.predict(n=n, series=series[:12]).all_values()
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(pred_21, pred_24)

        # with future covariates gives different results as without
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(pred_21, pred_11)

        # with future encodings only
        model_3 = model_cls(add_encoders={"cyclic": {"future": "month"}}, **kwargs)
        model_3.fit(series)
        pred_31 = model_3.predict(n=n).all_values()
        pred_32 = model_3.predict(n=n, series=series).all_values()
        np.testing.assert_array_almost_equal(pred_31, pred_32)
        # same results as model trained with explicit future covariates
        np.testing.assert_array_almost_equal(pred_31, pred_21)

        # with future covariates and future encodings
        model_4 = model_cls(
            add_encoders={"datetime_attribute": {"future": "year"}}, **kwargs
        )
        model_4.fit(series, future_covariates=fc)

        # using fc with wrong dimensionality and encodings
        with pytest.raises(ValueError) as exc:
            _ = model_4.predict(n=n, future_covariates=fc.stack(fc))
        assert str(exc.value).startswith(
            "The `future_covariates` passed to `predict()` must have "
            "the same number of components"
        )

        pred_41 = model_4.predict(n=n).all_values()
        pred_42 = model_4.predict(n=n, series=series).all_values()
        pred_43 = model_4.predict(n=n, series=series, future_covariates=fc).all_values()
        np.testing.assert_array_almost_equal(pred_41, pred_42)
        np.testing.assert_array_almost_equal(pred_42, pred_43)

        # more future covariates gives different results
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(pred_41, pred_21)

    @pytest.mark.parametrize(
        "config",
        [
            # comment: (transferable series support type, future cov support type, probabilistic support type)
            (AutoARIMA, {"season_length": 12}, False),  # (native, native, native)
            (
                AutoMFLES,
                {"season_length": 12, "test_size": 12},
                True,
            ),  # (custom, native, conformal)
            (AutoETS, {"season_length": 12}, False),  # (native, custom, native)
            (
                StatsForecastModel,
                {"model": SF_ETS(alpha=0.1)},
                True,
            ),  # (custom, custom, conformal)
        ],
    )
    def test_probabilistic_support(self, config):
        """Tests likelihood predictions."""
        model_cls, kwargs, requires_conformal = config

        n = 12
        quantiles = [0.1, 0.5, 0.9]
        med_idx = 1
        series = self.series[:24]

        kwargs["quantiles"] = quantiles
        kwargs["random_state"] = 42
        model = model_cls(**kwargs)

        # check the quantile likelihood
        assert isinstance(model.likelihood, QuantilePrediction)
        assert model.likelihood.quantiles == quantiles
        assert model.likelihood.levels == [80.00]  # (q high - q low)

        model.fit(series)

        if requires_conformal:
            # model requires setting `prediction_intervals` at model creation for prob. support
            with pytest.raises(Exception):
                model.predict(n=n, num_samples=2)
            with pytest.raises(Exception):
                model.predict(n=n, predict_likelihood_parameters=True)

            ci = ConformalIntervals()
            if isinstance(model, AutoMFLES):
                kwargs["prediction_intervals"] = ci
            else:
                kwargs["model"] = SF_ETS(alpha=0.1, prediction_intervals=ci)
            model = model_cls(**kwargs).fit(series)

        with pytest.raises(ValueError) as exc:
            _ = model.predict(n=n, num_samples=2, predict_likelihood_parameters=True)
        assert str(exc.value).startswith(
            "`predict_likelihood_parameters=True` is only supported for `num_samples=1`"
        )

        # series only
        pred_mean1 = model.predict(n=n).all_values()
        pred_mean2 = model.predict(n=n, num_samples=1).all_values()
        # mean (num_samples=1) is the default
        np.testing.assert_array_almost_equal(pred_mean1, pred_mean2)

        # direct quantile prediction
        pred_params = model.predict(n=n, predict_likelihood_parameters=True)
        # same with transferable series
        pred_params_tf = model.predict(
            n=n, predict_likelihood_parameters=True, series=series
        )
        assert pred_params == pred_params_tf
        assert pred_params.n_samples == 1
        # one component for each quantile
        name = series.components[0]
        q_names = [name + f"_{q_name}" for q_name in ["q0.10", "q0.50", "q0.90"]]
        assert list(pred_params.components) == q_names

        # center quantile is the mean
        pred_params = pred_params.all_values()
        np.testing.assert_array_almost_equal(pred_mean1[:, 0], pred_params[:, med_idx])
        # low quantile is below mean
        assert np.all(pred_params[:, 0] < pred_params[:, med_idx])
        # high quantile is above mean
        assert np.all(pred_params[:, 2] > pred_params[:, med_idx])

        # drawing samples from quantiles
        model = model.untrained_model().fit(series)
        pred_sample = model.predict(n=n, num_samples=10000).all_values()
        # quantiles from samples is approximately the quantile prediction
        assert pred_sample.shape[2] == 10000
        for idx, q in enumerate(quantiles):
            q_pred = np.quantile(pred_sample, q=q, axis=2)
            max_q_diff = np.abs(pred_params[:, idx] - q_pred).max()
            assert max_q_diff < 2.0

        # reproducible samples when call order is the same (and also with transferable series)
        model = model.untrained_model().fit(series)
        pred_sample_2 = model.predict(
            n=n, num_samples=10000, series=series
        ).all_values()
        assert (pred_sample == pred_sample_2).all()

        # different samples when call order changes
        pred_sample_3 = model.predict(n=n, num_samples=10000).all_values()
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(pred_sample, pred_sample_3)

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

    def test_croston_creation(self):
        with pytest.raises(ValueError) as exc:
            _ = Croston(version="does_not_exist")
        assert str(exc.value).startswith(
            'The provided "version" parameter must be set to'
        )

        with pytest.raises(ValueError) as exc:
            _ = Croston(version="tsb", alpha_d=None)
        assert str(exc.value).startswith(
            'alpha_d and alpha_p must be specified when using "tsb".'
        )

        _ = Croston(version="optimized")
        _ = Croston(version="sba")
        _ = Croston(version="tsb", alpha_d=0.1, alpha_p=0.1)

    def test_historical_forecasts_no_retrain(self):
        n, stride = 6, 12
        series = self.series[:52]
        model = AutoARIMA()
        model.fit(series)
        hist_fc = model.historical_forecasts(
            forecast_horizon=n,
            series=series,
            retrain=False,
            stride=12,
            overlap_end=False,
        )
        expected_fcs = math.ceil(
            (len(series) - model.min_train_series_length - n + 1) / stride
        )
        assert len(hist_fc) == expected_fcs

    def test_no_multivariate_support(self):
        model = AutoARIMA()
        assert not model.supports_multivariate

        with pytest.raises(ValueError) as exc:
            model.fit(self.series.stack(self.series))
        assert (
            str(exc.value)
            == "Model `AutoARIMA` only supports univariate TimeSeries instances"
        )

    def test_wrong_covariates(self):
        # using fc with wrong dimensionality
        n = 12
        series = self.series[:24]
        fc = self.fc[: 24 + n]
        model = AutoARIMA()
        model.fit(series, future_covariates=fc)

        # fc start too late
        with pytest.raises(ValueError) as exc:
            _ = model.predict(n=n, future_covariates=fc[1:])
        exc_expected = "must contain at least the same timesteps"
        assert exc_expected in str(exc.value)

        # fc end too early
        with pytest.raises(ValueError) as exc:
            _ = model.predict(n=n, future_covariates=fc[:-1])
        assert exc_expected in str(exc.value)
