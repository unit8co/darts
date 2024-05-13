from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.logging import get_logger
from darts.models import NotImportedModule, Prophet
from darts.utils import timeseries_generation as tg
from darts.utils.utils import freqs, generate_index

logger = get_logger(__name__)


@pytest.mark.skipif(isinstance(Prophet, NotImportedModule), reason="requires prophet")
class TestProphet:
    def test_add_seasonality_calls(self):
        # test if adding seasonality at model creation and with method model.add_seasonality() are equal
        kwargs_mandatory = {
            "name": "custom",
            "seasonal_periods": 48,
            "fourier_order": 4,
        }
        kwargs_mandatory2 = {
            "name": "custom2",
            "seasonal_periods": 24.9,
            "fourier_order": 1,
        }
        kwargs_all = dict(
            kwargs_mandatory,
            **{
                "prior_scale": 1.0,
                "mode": "additive",
                "condition_name": "custom_condition",
            },
        )
        model1 = Prophet(add_seasonalities=kwargs_all)
        model2 = Prophet()
        model2.add_seasonality(**kwargs_all)
        assert model1._add_seasonalities == model2._add_seasonalities

        # add multiple seasonalities
        model3 = Prophet(add_seasonalities=[kwargs_mandatory, kwargs_mandatory2])
        assert len(model3._add_seasonalities) == 2

        # seasonality already exists
        with pytest.raises(ValueError):
            model1.add_seasonality(**kwargs_mandatory)

        # missing mandatory arguments
        with pytest.raises(ValueError):
            for kw, arg in kwargs_mandatory.items():
                Prophet(add_seasonalities={kw: arg})

        # invalid keywords
        with pytest.raises(ValueError):
            Prophet(
                add_seasonalities=dict(
                    kwargs_mandatory, **{"some_random_keyword": "custom"}
                )
            )

        # invalid value dtypes
        with pytest.raises(ValueError):
            Prophet(add_seasonalities=dict({kw: None for kw in kwargs_mandatory}))

        with pytest.raises(ValueError):
            Prophet(add_seasonalities=dict([]))

    def test_prophet_model(self):
        """runs `helper_test_model` with several frequencies and periods"""
        perform_full_test = False

        test_cases_all = {
            freqs["YE"]: 12,
            "W": 7,
            freqs["BME"]: 12,
            "C": 5,
            "D": 7,
            "MS": 12,
            "B": 5,
            freqs["h"]: 24,
            freqs["bh"]: 8,
            freqs["QE"]: 4,
            freqs["min"]: 60,
            freqs["s"]: 60,
            "30" + freqs["s"]: 60,
            "24" + freqs["min"]: 60,
        }

        test_cases_fast = {
            key: test_cases_all[key] for key in ["MS", "D", freqs["h"]]
        }  # monthly, daily, hourly

        self.helper_test_freq_coversion(test_cases_all)
        test_cases = test_cases_all if perform_full_test else test_cases_fast
        for i, (freq, period) in enumerate(test_cases.items()):
            if not i:
                self.helper_test_prophet_model(
                    period=period, freq=freq, compare_all_models=True
                )
            else:
                self.helper_test_prophet_model(
                    period=period, freq=freq, compare_all_models=False
                )

    def test_prophet_model_without_stdout_suppression(self):
        model = Prophet(suppress_stdout_stderror=False)
        model._execute_and_suppress_output = Mock(return_value=True)
        model._model_builder = Mock(return_value=Mock(fit=Mock(return_value=True)))
        df = pd.DataFrame({
            "ds": pd.date_range(start="2022-01-01", periods=30, freq="D"),
            "y": np.linspace(0, 10, 30),
        })
        ts = TimeSeries.from_dataframe(df, time_col="ds", value_cols="y")
        model.fit(ts)

        (
            model._execute_and_suppress_output.assert_not_called(),
            "Suppression should not be called",
        )
        model.model.fit.assert_called_once(), "Model should still be fitted"

    def test_prophet_model_with_stdout_suppression(self):
        model = Prophet(suppress_stdout_stderror=True)
        model._execute_and_suppress_output = Mock(return_value=True)
        model._model_builder = Mock(return_value=Mock(fit=Mock(return_value=True)))
        df = pd.DataFrame({
            "ds": pd.date_range(start="2022-01-01", periods=30, freq="D"),
            "y": np.linspace(0, 10, 30),
        })
        ts = TimeSeries.from_dataframe(df, time_col="ds", value_cols="y")
        model.fit(ts)

        (
            model._execute_and_suppress_output.assert_called_once(),
            "Suppression should be called once",
        )

    def test_prophet_model_default_with_prophet_constructor(self):
        from prophet import Prophet as FBProphet

        model = Prophet()
        assert model._model_builder == FBProphet, "model should use Facebook Prophet"

    def test_prophet_model_with_logistic_growth(self):
        model = Prophet(growth="logistic", cap=1)

        # Create timeseries with logistic function
        times = generate_index(
            pd.Timestamp("20200101"), pd.Timestamp("20210101"), freq="D"
        )
        values = np.linspace(-10, 10, len(times))
        f = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
        values = f(values)
        ts = TimeSeries.from_times_and_values(times, values, freq="D")
        # split in the middle, so the only way of predicting the plateau correctly
        # is using the capacity
        train, val = ts.split_after(0.5)

        model.fit(train)
        pred = model.predict(len(val))

        for val_i, pred_i in zip(val.univariate_values(), pred.univariate_values()):
            assert abs(val_i - pred_i) < 0.1

    def helper_test_freq_coversion(self, test_cases):
        for freq, period in test_cases.items():
            ts_sine = tg.sine_timeseries(
                value_frequency=1 / period, length=3, freq=freq
            )
            # this should not raise an error if frequency is known
            _ = Prophet._freq_to_days(freq=ts_sine.freq_str)

        assert (
            abs(
                Prophet._freq_to_days(freq="30" + freqs["s"])
                - 30 * Prophet._freq_to_days(freq=freqs["s"])
            )
            < 10e-9
        )

        # check bad frequency string
        with pytest.raises(ValueError):
            _ = Prophet._freq_to_days(freq="30SS")

    def helper_test_prophet_model(self, period, freq, compare_all_models=False):
        """Test which includes adding custom seasonalities and future covariates. The tests compare the output of
        univariate and stochastic forecasting with the validation timeseries and Prophet's base model output.

        The underlying curve to forecast is a sine timeseries multiplied with another sine timeseries.
        The curve shape repeats every 2*period timesteps (i.e. for period=24 hours -> seasonal_periods=48).
        We take the second sine wave as a covariate for the model.
        With the added custom seasonality and covariate, the model should have a very accurate forecast.
        """
        repetitions = 8
        ts_sine1 = tg.sine_timeseries(
            value_frequency=1 / period, length=period * repetitions, freq=freq
        )
        ts_sine2 = tg.sine_timeseries(
            value_frequency=1 / (period * 2), length=period * repetitions, freq=freq
        )
        ts_sine = ts_sine1 * ts_sine2
        covariate = ts_sine2

        split = int(-period * repetitions / 2)
        train, val = ts_sine[:split], ts_sine[split:]
        train_cov, val_cov = covariate[:split], covariate[split:]

        supress_auto_seasonality = {
            "daily_seasonality": False,
            "weekly_seasonality": False,
            "yearly_seasonality": False,
        }
        custom_seasonality = {
            "name": "custom",
            "seasonal_periods": int(2 * period),
            "fourier_order": 4,
        }
        model = Prophet(
            add_seasonalities=custom_seasonality,
            seasonality_mode="additive",
            **supress_auto_seasonality,
        )

        model.fit(train, future_covariates=train_cov)

        # univariate, stochastic and Prophet's base model forecast
        pred_darts = model.predict(n=len(val), num_samples=1, future_covariates=val_cov)
        compare_preds = [pred_darts]

        if compare_all_models:
            pred_darts_stochastic = model.predict(
                n=len(val), num_samples=200, future_covariates=val_cov
            )
            pred_raw_df = model.predict_raw(n=len(val), future_covariates=val_cov)
            pred_raw = TimeSeries.from_dataframe(
                pred_raw_df[["ds", "yhat"]], time_col="ds"
            )
            compare_preds += [
                pred_darts_stochastic.quantile_timeseries(0.5),
                pred_raw,
            ]

        # all predictions should fit the underlying curve very well
        for pred in compare_preds:
            for val_i, pred_i in zip(val.univariate_values(), pred.univariate_values()):
                assert abs(val_i - pred_i) < 0.1

    def test_conditional_seasonality(self):
        """
        Test that conditional seasonality is correctly incorporated by the model
        """
        duration = 395
        horizon = 7
        df = pd.DataFrame()
        df["ds"] = pd.date_range(start="2022-01-02", periods=395)
        df["y"] = [i + 10 * (i % 7 == 0) for i in range(duration)]
        df["is_sunday"] = df["ds"].apply(lambda x: int(x.weekday() == 6))

        ts = TimeSeries.from_dataframe(
            df[:-horizon], time_col="ds", value_cols="y", freq="D"
        )
        future_covariates = TimeSeries.from_dataframe(
            df, time_col="ds", value_cols=["is_sunday"], freq="D"
        )
        expected_result = TimeSeries.from_dataframe(
            df[-horizon:], time_col="ds", value_cols="y", freq="D"
        )

        model = Prophet(seasonality_mode="additive")
        model.add_seasonality(
            name="weekly_sun",
            seasonal_periods=7,
            fourier_order=2,
            condition_name="is_sunday",
        )

        model.fit(ts, future_covariates=future_covariates)

        forecast = model.predict(horizon, future_covariates=future_covariates)

        for val_i, pred_i in zip(
            expected_result.univariate_values(), forecast.univariate_values()
        ):
            assert abs(val_i - pred_i) < 0.1

        invalid_future_covariates = future_covariates.with_values(
            np.reshape(np.random.randint(0, 3, duration), (-1, 1, 1)).astype("float")
        )

        with pytest.raises(ValueError):
            model.fit(ts, future_covariates=invalid_future_covariates)

        with pytest.raises(ValueError):
            model.fit(
                ts,
                future_covariates=invalid_future_covariates.drop_columns("is_sunday"),
            )
