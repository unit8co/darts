import numpy as np
from pandas import date_range

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers.reconciliation import (
    BottomUpReconciliator,
    MinTReconciliator,
    TopDownReconciliator,
    _get_summation_matrix,
)
from darts.models import LinearRegressionModel
from darts.utils import timeseries_generation as tg


class TestReconciliation:
    np.random.seed(42)

    """ test case with a more intricate hierarchy """
    LENGTH = 200
    total_series = (
        tg.sine_timeseries(value_frequency=0.03, length=LENGTH)
        + 1
        + tg.gaussian_timeseries(length=LENGTH) * 0.2
    )
    bottom_1 = total_series / 3 + tg.gaussian_timeseries(length=LENGTH) * 0.01
    bottom_2 = 2 * total_series / 3 + tg.gaussian_timeseries(length=LENGTH) * 0.01
    series = concatenate([total_series, bottom_1, bottom_2], axis=1)
    hierarchy = {"sine_1": ["sine"], "sine_2": ["sine"]}
    series = series.with_hierarchy(hierarchy)

    # get a single forecast
    model = LinearRegressionModel(lags=30, output_chunk_length=10)
    model.fit(series)
    pred = model.predict(n=20)

    # get a backtest forecast to get residuals
    pred_back = model.historical_forecasts(series, start=0.75, forecast_horizon=10)
    intersection = series.slice_intersect(pred_back)
    residuals = intersection - pred_back

    """ test case with a more intricate hierarchy """
    components_complex = ["total", "a", "b", "x", "y", "ax", "ay", "bx", "by"]

    hierarchy_complex = {
        "ax": ["a", "x"],
        "ay": ["a", "y"],
        "bx": ["b", "x"],
        "by": ["b", "y"],
        "a": ["total"],
        "b": ["total"],
        "x": ["total"],
        "y": ["total"],
    }

    series_complex = TimeSeries.from_values(
        values=np.random.rand(50, len(components_complex), 5),
        columns=components_complex,
        hierarchy=hierarchy_complex,
    )

    def _assert_reconciliation(self, fitted_recon):
        pred_r = fitted_recon.transform(self.pred)
        np.testing.assert_almost_equal(
            pred_r["sine"].values(copy=False),
            (pred_r["sine_1"] + pred_r["sine_2"]).values(copy=False),
        )

    def _assert_reconciliation_complex(self, fitted_recon):
        reconciled = fitted_recon.transform(self.series_complex)

        def _assert_comps(comp, comps):
            np.testing.assert_almost_equal(
                reconciled[comp].values(copy=False),
                sum(reconciled[c] for c in comps).values(copy=False),
            )

        _assert_comps("a", ["ax", "ay"])
        _assert_comps("b", ["bx", "by"])
        _assert_comps("x", ["ax", "bx"])
        _assert_comps("y", ["ay", "by"])
        _assert_comps("total", ["ax", "ay", "bx", "by"])
        _assert_comps("total", ["a", "b"])
        _assert_comps("total", ["x", "y"])

    def test_bottom_up(self):
        recon = BottomUpReconciliator()
        self._assert_reconciliation(recon)

    def test_top_down(self):
        # should work when fitting on training series
        recon = TopDownReconciliator()
        recon.fit(self.series)
        self._assert_reconciliation(recon)

        # or when fitting on forecasts
        recon = TopDownReconciliator()
        recon.fit(self.pred)
        self._assert_reconciliation(recon)

        # fit_transform() should also work
        recon = TopDownReconciliator()
        _ = recon.fit_transform(self.pred)

    def test_mint(self):
        # ols
        recon = MinTReconciliator("ols")
        recon.fit(self.series)
        self._assert_reconciliation(recon)

        # wls_struct
        recon = MinTReconciliator("wls_struct")
        recon.fit(self.series)
        self._assert_reconciliation(recon)

        # wls_var
        recon = MinTReconciliator("wls_var")
        recon.fit(self.residuals)
        self._assert_reconciliation(recon)

        # mint_cov
        recon = MinTReconciliator("mint_cov")
        recon.fit(self.residuals)
        self._assert_reconciliation(recon)

        # wls_val
        recon = MinTReconciliator("wls_val")
        recon.fit(self.series)
        self._assert_reconciliation(recon)

        # fit_transform() should also work
        recon = MinTReconciliator()
        _ = recon.fit_transform(self.series)

    def test_summation_matrix(self):
        np.testing.assert_equal(
            _get_summation_matrix(self.series_complex),
            np.array([
                [1, 1, 1, 1],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]),
        )

    def test_hierarchy_preserved_after_predict(self):
        assert self.pred.hierarchy == self.series.hierarchy

    def test_more_intricate_hierarchy(self):
        recon = BottomUpReconciliator()
        self._assert_reconciliation_complex(recon)

        recon = TopDownReconciliator()
        recon.fit(self.series_complex)
        self._assert_reconciliation_complex(recon)

        recon = MinTReconciliator("ols")
        recon.fit(self.series_complex)
        self._assert_reconciliation_complex(recon)

        recon = MinTReconciliator("wls_struct")
        recon.fit(self.series_complex)
        self._assert_reconciliation_complex(recon)

        recon = MinTReconciliator("wls_val")
        recon.fit(self.series_complex)
        self._assert_reconciliation_complex(recon)

    def test_reconcilliation_is_order_independent(self):
        dates = date_range("2020-01-01", "2020-12-31", freq="D")
        nr_dates = len(dates)
        t1 = TimeSeries.from_times_and_values(
            dates, 2 * np.ones(nr_dates), columns=["T1"]
        )
        t2 = TimeSeries.from_times_and_values(
            dates, 5 * np.ones(nr_dates), columns=["T2"]
        )
        t3 = TimeSeries.from_times_and_values(dates, np.ones(nr_dates), columns=["T3"])
        tsum = TimeSeries.from_times_and_values(
            dates, 9 * np.ones(nr_dates), columns=["T_sum"]
        )
        ts_1 = concatenate([t1, t2, t3, tsum], axis="component")
        ts_2 = concatenate([tsum, t1, t2, t3], axis="component")

        def assert_ts_are_equal(ts1, ts2):
            for comp in ["T1", "T2", "T3", "T_sum"]:
                assert ts1[comp] == ts2[comp]

        hierarchy = {"T1": ["T_sum"], "T2": ["T_sum"], "T3": ["T_sum"]}
        ts_1 = ts_1.with_hierarchy(hierarchy)
        ts_2 = ts_2.with_hierarchy(hierarchy)
        assert_ts_are_equal(ts_1, ts_2)

        ts_1_reconc = TopDownReconciliator().fit_transform(ts_1)
        ts_2_reconc = TopDownReconciliator().fit_transform(ts_2)

        assert_ts_are_equal(ts_1_reconc, ts_2_reconc)

        ts_1_reconc = MinTReconciliator().fit_transform(ts_1)
        ts_2_reconc = MinTReconciliator().fit_transform(ts_2)

        assert_ts_are_equal(ts_1_reconc, ts_2_reconc)

        ts_1_reconc = BottomUpReconciliator().transform(ts_1)
        ts_2_reconc = BottomUpReconciliator().transform(ts_2)

        assert_ts_are_equal(ts_1_reconc, ts_2_reconc)
