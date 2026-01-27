import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.utils.statistics import (
    check_seasonality,
    extract_trend_and_seasonality,
    granger_causality_tests,
    plot_acf,
    plot_ccf,
    plot_pacf,
    plot_residuals_analysis,
    plot_tolerance_curve,
    remove_seasonality,
    remove_trend,
    stationarity_test_adf,
    stationarity_test_kpss,
    stationarity_tests,
)
from darts.utils.timeseries_generation import (
    constant_timeseries,
    gaussian_timeseries,
    linear_timeseries,
)
from darts.utils.utils import ModelMode, SeasonalityMode


class TestTimeSeries:
    def test_check_seasonality(self):
        pd_series = pd.Series(range(50), index=pd.date_range("20130101", "20130219"))
        pd_series = pd_series.map(lambda x: np.sin(x * np.pi / 3 + np.pi / 2))
        series = TimeSeries.from_series(pd_series)

        assert (True, 6) == check_seasonality(series)
        assert (False, 3) == check_seasonality(series, m=3)

        with pytest.raises(AssertionError):
            check_seasonality(series.stack(series))

    def test_granger_causality(self):
        series_cause_1 = constant_timeseries(start=0, end=9999).stack(
            constant_timeseries(start=0, end=9999)
        )
        series_cause_2 = gaussian_timeseries(start=0, end=9999)
        series_effect_1 = constant_timeseries(start=0, end=999)
        series_effect_2 = TimeSeries.from_values(np.random.uniform(0, 1, 10000))
        series_effect_3 = TimeSeries.from_values(
            np.random.uniform(0, 1, (1000, 2, 1000))
        )
        series_effect_4 = constant_timeseries(
            start=pd.Timestamp("2000-01-01"), length=10000
        )

        # Test univariate
        with pytest.raises(AssertionError):
            granger_causality_tests(series_cause_1, series_effect_1, 10)
        with pytest.raises(AssertionError):
            granger_causality_tests(series_effect_1, series_cause_1, 10)

        # Test deterministic
        with pytest.raises(AssertionError):
            granger_causality_tests(series_cause_1, series_effect_3, 10)
        with pytest.raises(AssertionError):
            granger_causality_tests(series_effect_3, series_cause_1, 10)

        # Test Frequency
        with pytest.raises(ValueError):
            granger_causality_tests(series_cause_2, series_effect_4, 10)

        # Test granger basics
        tests = granger_causality_tests(series_effect_2, series_effect_2, 10)
        assert tests[1][0]["ssr_ftest"][1] > 0.99
        tests = granger_causality_tests(series_cause_2, series_effect_2, 10)
        assert tests[1][0]["ssr_ftest"][1] > 0.01

    def test_stationarity_tests(self):
        np.random.seed(42)
        series_1 = constant_timeseries(start=0, end=9999).stack(
            constant_timeseries(start=0, end=9999)
        )

        series_2 = TimeSeries.from_values(np.random.uniform(0, 1, (1000, 2, 1000)))
        series_3 = gaussian_timeseries(start=0, end=9999)

        # Test univariate
        with pytest.raises(AssertionError):
            stationarity_tests(series_1)
        with pytest.raises(AssertionError):
            stationarity_test_adf(series_1)
        with pytest.raises(AssertionError):
            stationarity_test_kpss(series_1)

        # Test deterministic
        with pytest.raises(AssertionError):
            stationarity_tests(series_2)
        with pytest.raises(AssertionError):
            stationarity_test_adf(series_2)
        with pytest.raises(AssertionError):
            stationarity_test_kpss(series_2)

        # Test basics
        assert stationarity_test_kpss(series_3)[1] > 0.05
        assert stationarity_test_adf(series_3)[1] < 0.05
        assert stationarity_tests


class TestSeasonalDecompose:
    pd_series = pd.Series(range(50), index=pd.date_range("20130101", "20130219"))
    pd_series = pd_series.map(lambda x: np.sin(x * np.pi / 3 + np.pi / 2))
    season = TimeSeries.from_series(pd_series)
    trend = linear_timeseries(
        start_value=1, end_value=10, start=season.start_time(), end=season.end_time()
    )
    ts = trend + season

    def test_extract(self):
        series_copy = self.ts.copy()
        # test default (naive) method
        calc_trend, _ = extract_trend_and_seasonality(self.ts, freq=6)
        diff = self.trend - calc_trend
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # test default (naive) method additive
        calc_trend, _ = extract_trend_and_seasonality(
            self.ts, freq=6, model=ModelMode.ADDITIVE
        )
        diff = self.trend - calc_trend
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # test STL method
        calc_trend, _ = extract_trend_and_seasonality(
            self.ts, freq=6, method="STL", model=ModelMode.ADDITIVE
        )
        diff = self.trend - calc_trend
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # test MSTL method
        calc_trend, calc_seasonality = extract_trend_and_seasonality(
            self.ts, freq=[3, 6], method="MSTL", model=ModelMode.ADDITIVE
        )
        assert len(calc_seasonality.components) == 2
        diff = self.trend - calc_trend
        # relaxed tolerance for MSTL since it will have a larger error from the
        # extrapolation of the trend, it is still a small number but is more
        # than STL or naive trend extraction
        assert np.isclose(np.mean(diff.values() ** 2), 0.0, atol=1e-5)

        # test MSTL method with single freq
        calc_trend, calc_seasonality = extract_trend_and_seasonality(
            self.ts, freq=6, method="MSTL", model=ModelMode.ADDITIVE
        )
        assert len(calc_seasonality.components) == 1
        diff = self.trend - calc_trend
        assert np.isclose(np.mean(diff.values() ** 2), 0.0, atol=1e-5)

        # make sure non MSTL methods fail with multiple freqs
        with pytest.raises(ValueError):
            calc_trend, calc_seasonality = extract_trend_and_seasonality(
                self.ts, freq=[1, 4, 6], method="STL", model=ModelMode.ADDITIVE
            )

        # check if error is raised when using multiplicative model
        with pytest.raises(ValueError):
            calc_trend, _ = extract_trend_and_seasonality(
                self.ts, freq=6, method="STL", model=ModelMode.MULTIPLICATIVE
            )

        with pytest.raises(ValueError):
            calc_trend, _ = extract_trend_and_seasonality(
                self.ts, freq=[3, 6], method="MSTL", model=ModelMode.MULTIPLICATIVE
            )

        assert self.ts == series_copy

    def test_remove_seasonality(self):
        # test default (naive) method
        calc_trend = remove_seasonality(self.ts, freq=6)
        diff = self.trend - calc_trend
        assert np.mean(diff.values() ** 2).item() < 0.5

        # test default (naive) method additive
        calc_trend = remove_seasonality(self.ts, freq=6, model=SeasonalityMode.ADDITIVE)
        diff = self.trend - calc_trend
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # test STL method
        calc_trend = remove_seasonality(
            self.ts,
            freq=6,
            method="STL",
            model=SeasonalityMode.ADDITIVE,
            low_pass=9,
        )
        diff = self.trend - calc_trend
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # check if error is raised
        with pytest.raises(ValueError):
            calc_trend = remove_seasonality(
                self.ts, freq=6, method="STL", model=SeasonalityMode.MULTIPLICATIVE
            )

    def test_remove_trend(self):
        # test naive method
        calc_season = remove_trend(self.ts, freq=6)
        diff = self.season - calc_season
        assert np.mean(diff.values() ** 2).item() < 1.5

        # test naive method additive
        calc_season = remove_trend(self.ts, freq=6, model=ModelMode.ADDITIVE)
        diff = self.season - calc_season
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # test STL method
        calc_season = remove_trend(
            self.ts,
            freq=6,
            method="STL",
            model=ModelMode.ADDITIVE,
            low_pass=9,
        )
        diff = self.season - calc_season
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # check if error is raised
        with pytest.raises(ValueError):
            calc_season = remove_trend(
                self.ts, freq=6, method="STL", model=ModelMode.MULTIPLICATIVE
            )


class TestPlot:
    series = AirPassengersDataset().load()

    def test_statistics_plot(self):
        plot_residuals_analysis(self.series)
        plt.close()
        plot_residuals_analysis(self.series, acf_max_lag=10)
        plt.close()
        plot_residuals_analysis(self.series[:10])
        plt.close()
        plot_acf(self.series)
        plot_pacf(self.series)
        plot_ccf(self.series, self.series)
        plt.close()


class TestPlotToleranceCurve:
    # univariate series
    actual_uni = TimeSeries.from_values(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    pred_uni = TimeSeries.from_values(np.array([1.1, 2.2, 2.9, 4.1, 5.0]))

    # multivariate series
    actual_multi = TimeSeries.from_values(
        np.column_stack([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]]),
        columns=["c1", "c2"],
    )
    pred_multi = TimeSeries.from_values(
        np.column_stack([[1.1, 2.2, 2.9, 4.1, 5.0], [11.0, 22.0, 29.0, 41.0, 50.0]]),
        columns=["c1", "c2"],
    )

    # stochastic series
    pred_stoch = TimeSeries.from_values(
        np.random.rand(5, 1, 10) + np.arange(1.0, 6.0).reshape(-1, 1, 1)
    )

    @pytest.mark.parametrize(
        "actual,pred,kwargs",
        [
            ("actual_uni", "pred_uni", {}),
            ("actual_multi", "pred_multi", {}),
            ("actual_uni", "pred_stoch", {}),
            ("actual_uni", "pred_stoch", {"q": 0.25}),
            ("actual_uni", "pred_uni", {"default_formatting": False}),
            ("actual_uni", "pred_uni", {"min_tolerance": 0.1, "max_tolerance": 0.9}),
            ("actual_uni", "pred_uni", {"step": 0.05}),
        ],
    )
    def test_plot_tolerance_curve_params(self, actual, pred, kwargs):
        plot_tolerance_curve(getattr(self, actual), getattr(self, pred), **kwargs)
        plt.close()

    def test_plot_tolerance_curve_with_axis(self):
        _, ax = plt.subplots()
        plot_tolerance_curve(self.actual_uni, self.pred_uni, axis=ax)
        plt.close()

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"min_tolerance": -0.1}, "min_tolerance must be >= 0"),
            ({"max_tolerance": 1.5}, "max_tolerance must be <= 1"),
            (
                {"min_tolerance": 0.8, "max_tolerance": 0.5},
                "min_tolerance must be >= 0",
            ),
            ({"step": 0}, "step must be positive"),
            ({"step": -0.1}, "step must be positive"),
            ({"step": 2.0}, "step must be positive"),
            ({"q": 1.5}, "q must be between 0 and 1"),
        ],
    )
    def test_plot_tolerance_curve_invalid_params(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            plot_tolerance_curve(self.actual_uni, self.pred_uni, **kwargs)

    def test_plot_tolerance_curve_component_mismatch(self):
        with pytest.raises(ValueError, match="must have the same number of components"):
            plot_tolerance_curve(self.actual_multi, self.pred_uni)

    def test_plot_tolerance_curve_intersect(self):
        # indexes partially overlap
        actual = TimeSeries.from_times_and_values(
            pd.date_range("2020-01-01", periods=10, freq="D"), np.arange(10.0)
        )
        pred = TimeSeries.from_times_and_values(
            pd.date_range("2020-01-05", periods=10, freq="D"), np.arange(10.0)
        )
        plot_tolerance_curve(actual, pred, intersect=True)
        plt.close()

    def test_plot_tolerance_curve_no_overlap(self):
        actual = TimeSeries.from_times_and_values(
            pd.date_range("2020-01-01", periods=5, freq="D"), np.arange(5.0)
        )
        pred = TimeSeries.from_times_and_values(
            pd.date_range("2020-02-01", periods=5, freq="D"), np.arange(5.0)
        )
        with pytest.raises(ValueError, match="at least one overlapping time step"):
            plot_tolerance_curve(actual, pred, intersect=True)

    def test_plot_tolerance_curve_constant_series(self):
        actual_const = TimeSeries.from_values(np.array([5.0, 5.0, 5.0, 5.0, 5.0]))
        with pytest.raises(ValueError, match="range of actual values"):
            plot_tolerance_curve(actual_const, self.pred_uni)
