import numpy as np
import pytest

from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.utils import timeseries_generation as tg


class TestExponentialSmoothing:
    series = tg.sine_timeseries(length=100, freq="h")

    @pytest.mark.parametrize(
        "freq_string,expected_seasonal_periods",
        [
            ("D", 7),
            ("h", 24),
            ("ME", 12),
            ("W", 52),
            ("QE", 4),
            ("B", 5),
        ],
    )
    def test_seasonality_inference(
        self, freq_string: str, expected_seasonal_periods: int
    ):
        series = tg.sine_timeseries(length=200, freq=freq_string)
        model = ExponentialSmoothing()
        model.fit(series)
        assert model.seasonal_periods == expected_seasonal_periods

    def test_default_parameters(self):
        """Test default selection for integer index"""
        series = TimeSeries.from_values(np.arange(1, 30, 1))
        model = ExponentialSmoothing()
        model.fit(series)
        assert model.seasonal_periods == 12

    def test_multiple_fit(self):
        """Test whether a model that inferred a seasonality period before will do it again for a new series"""
        series1 = tg.sine_timeseries(length=100, freq="ME")
        series2 = tg.sine_timeseries(length=100, freq="D")
        model = ExponentialSmoothing()
        model.fit(series1)
        model.fit(series2)
        assert model.seasonal_periods == 7

    def test_constructor_kwargs(self):
        """Using kwargs to pass additional parameters to the constructor"""
        constructor_kwargs = {
            "initialization_method": "known",
            "initial_level": 0.5,
            "initial_trend": 0.2,
            "initial_seasonal": np.arange(1, 25),
        }
        model = ExponentialSmoothing(kwargs=constructor_kwargs)
        model.fit(self.series)
        # must be checked separately, name is not consistent
        np.testing.assert_array_almost_equal(
            model.model.model.params["initial_seasons"],
            constructor_kwargs["initial_seasonal"],
        )
        for param_name in ["initial_level", "initial_trend"]:
            assert (
                model.model.model.params[param_name] == constructor_kwargs[param_name]
            )

    def test_fit_kwargs(self):
        """Using kwargs to pass additional parameters to the fit()"""
        # using default optimization method
        model = ExponentialSmoothing()
        model.fit(self.series)
        assert model.fit_kwargs == {}
        pred = model.predict(n=2)

        model_bis = ExponentialSmoothing()
        model_bis.fit(self.series)
        assert model_bis.fit_kwargs == {}
        pred_bis = model_bis.predict(n=2)

        # two methods with the same parameters should yield the same forecasts
        assert pred.time_index.equals(pred_bis.time_index)
        np.testing.assert_array_almost_equal(pred.values(), pred_bis.values())

        # change optimization method
        model_ls = ExponentialSmoothing(method="least_squares")
        model_ls.fit(self.series)
        assert model_ls.fit_kwargs == {"method": "least_squares"}
        pred_ls = model_ls.predict(n=2)

        # forecasts should be slightly different
        assert pred.time_index.equals(pred_ls.time_index)
        assert all(np.not_equal(pred.values(), pred_ls.values()))

    def test_random_errors(self):
        """Test whether random_errors parameter is correctly passed to simulate()"""
        series = tg.sine_timeseries(length=100, freq="h")
        model = ExponentialSmoothing(random_state=42)
        model.fit(series)
        pred = model.predict(n=10, num_samples=10, random_state=42)

        model_boot = ExponentialSmoothing(random_errors="bootstrap", random_state=42)
        model_boot.fit(series)
        pred_boot = model_boot.predict(n=10, num_samples=10, random_state=42)

        # methods with different random_errors set should yield different forecasts
        assert not np.allclose(pred.values(), pred_boot.values(), atol=1e-5)

    def test_error(self):
        """Test whether error parameter is correctly passed to simulate()"""
        series = tg.sine_timeseries(length=100, freq="h")
        model = ExponentialSmoothing(random_state=42)
        model.fit(series)
        pred = model.predict(n=10, num_samples=10, random_state=42)

        model_boot = ExponentialSmoothing(error="mul", random_state=42)
        model_boot.fit(series)
        pred_boot = model_boot.predict(n=10, num_samples=10, random_state=42)

        # methods with different error set should yield different forecasts
        assert not np.allclose(pred.values(), pred_boot.values(), atol=1e-5)

    def test_min_train_length(self):
        # default requirement for a non-seasonal model
        assert ExponentialSmoothing().min_train_series_length == 3
        # the override lowers it
        assert ExponentialSmoothing(min_train_length=2).min_train_series_length == 2
        with pytest.raises(ValueError, match="`min_train_length` must be"):
            ExponentialSmoothing(min_train_length=0)

    def test_future_covariates(self):
        """Future covariates are used through OLS + Exponential Smoothing on the residuals."""
        n = 10
        covs = tg.linear_timeseries(
            length=100 + n, freq="h", start_value=0, end_value=5
        )
        series = (
            tg.sine_timeseries(length=100, freq="h", value_frequency=0.05)
            + 3 * covs[:100]
        )
        kwargs = {"trend": None, "seasonal": None}

        model = ExponentialSmoothing(**kwargs)
        assert model.supports_future_covariates
        model.fit(series, future_covariates=covs)
        pred = model.predict(n, future_covariates=covs)
        assert len(pred) == n
        assert pred.start_time() == series.end_time() + series.freq

        # the regression part is added back to the forecast
        pred_no_cov = ExponentialSmoothing(**kwargs).fit(series).predict(n)
        true = 3 * covs[100:]
        assert (
            np.abs(pred.values() - true.values()).mean()
            < np.abs(pred_no_cov.values() - true.values()).mean() + 1.0
        )
        assert not np.allclose(pred.values(), pred_no_cov.values())

        # probabilistic forecasts include the regression
        pred_prob = model.predict(n, future_covariates=covs, num_samples=5)
        assert pred_prob.n_samples == 5
        assert pred_prob.all_values().shape[0] == n

        # covariates passed at fit time are re-used when none are passed to predict() (darts convention)
        np.testing.assert_array_almost_equal(model.predict(n).values(), pred.values())

        # a model trained without covariates rejects covariates at prediction time
        with pytest.raises(ValueError):
            ExponentialSmoothing(**kwargs).fit(series).predict(
                n, future_covariates=covs
            )

    def test_add_encoders(self):
        series = tg.sine_timeseries(length=100, freq="h")
        model = ExponentialSmoothing(
            trend=None, seasonal=None, add_encoders={"cyclic": {"future": ["hour"]}}
        )
        model.fit(series)
        assert len(model.predict(5)) == 5
