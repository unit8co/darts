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
