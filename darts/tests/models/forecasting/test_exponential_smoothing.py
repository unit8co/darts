import numpy as np
import pytest

from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.utils import timeseries_generation as tg
from darts.utils.utils import freqs


class TestExponentialSmoothing:
    series = tg.sine_timeseries(length=100, freq=freqs["h"])

    @pytest.mark.parametrize(
        "freq_string,expected_seasonal_periods",
        [
            ("D", 7),
            (freqs["h"], 24),
            (freqs["ME"], 12),
            ("W", 52),
            (freqs["QE"], 4),
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
        series1 = tg.sine_timeseries(length=100, freq=freqs["ME"])
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
