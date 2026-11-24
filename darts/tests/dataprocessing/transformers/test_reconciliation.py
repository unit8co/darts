import numpy as np
import pytest
from pandas import date_range

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import FittableDataTransformer
from darts.dataprocessing.transformers.reconciliation import (
    BottomUpReconciliator,
    MinTReconciliator,
    TopDownReconciliator,
    _get_summation_matrix,
)
from darts.models import LinearRegressionModel
from darts.tests.parametrize_helpers import param_product
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
    residuals_complex = TimeSeries.from_values(
        values=np.random.randn(50, len(components_complex)),
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

    def test_mint_nonnegative_closed_form(self):
        # with W = I, minimizing (10 - a - b)^2 + (0.5 - a)^2 + (20 - b)^2 gives a = -3 and b = 16.5;
        # under the constraint a, b >= 0 the solution is a = 0 and b = 15
        series = TimeSeries.from_values(
            np.array([[10.0, 0.5, 20.0], [3.0, 1.0, 2.0]]),
            columns=["total", "a", "b"],
            hierarchy={"a": ["total"], "b": ["total"]},
        )
        unconstrained = MinTReconciliator("ols").fit_transform(series).values()
        np.testing.assert_almost_equal(unconstrained[0], [13.5, -3.0, 16.5])

        recon = MinTReconciliator("ols", nonnegative=True)
        reconciled = recon.fit_transform(series).values()
        np.testing.assert_almost_equal(reconciled[0], [15.0, 0.0, 15.0])
        # the second time step has no negative values and is left unchanged
        np.testing.assert_array_equal(reconciled[1], unconstrained[1])

    @pytest.mark.parametrize(
        "method, fit_on",
        [
            ("ols", "series_complex"),
            ("wls_struct", "series_complex"),
            ("wls_val", "series_complex"),
            ("wls_var", "residuals_complex"),
            ("mint_cov", "residuals_complex"),
        ],
    )
    def test_mint_nonnegative(self, method, fit_on):
        # incoherent stochastic base forecasts, some of them negative
        base = self.series_complex - 0.2
        y_hat = base.all_values()
        series_fit = getattr(self, fit_on)
        out = MinTReconciliator(method).fit(series_fit).transform(base).all_values()
        recon = MinTReconciliator(method, nonnegative=True).fit(series_fit)
        out_nn = recon.transform(base).all_values()

        # (time step, sample) pairs with a negative value without the constraint
        resolved = (out < 0).any(axis=1)
        assert resolved.any() and not resolved.all()
        assert (out_nn >= 0).all()
        # all other values are unchanged
        np.testing.assert_array_equal(
            out_nn.transpose(0, 2, 1)[~resolved], out.transpose(0, 2, 1)[~resolved]
        )

        # the reconciled forecasts are coherent
        S = _get_summation_matrix(base)
        bottom = [list(base.components).index(c) for c in base.bottom_level_components]
        b = out_nn[:, bottom, :]
        np.testing.assert_allclose(out_nn, S @ b, atol=1e-12)

        # KKT conditions of min_b (y_hat - S b)' inv(W) (y_hat - S b) s.t. b >= 0: the gradient is zero
        # for positive values of b and non-negative for values at zero
        _, _, W_inv = MinTReconciliator._get_matrices(series_fit, method)
        grad = (S.T @ W_inv @ (S @ b - y_hat)).transpose(0, 2, 1)
        b = b.transpose(0, 2, 1)
        assert np.abs(grad[b > 0]).max() < 1e-8
        assert grad[b == 0].min() > -1e-8

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
        series_input_copy = [ts_1.copy(), ts_2.copy()]
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
        assert [ts_1, ts_2] == series_input_copy

    @pytest.mark.parametrize(
        "dtype, config",
        param_product(
            ["float64", "float32"],
            [
                (BottomUpReconciliator(), "series"),
                (TopDownReconciliator(), "series"),
                (MinTReconciliator("ols"), "series"),
                (MinTReconciliator("wls_struct"), "series"),
                (MinTReconciliator("wls_val"), "series"),
                (MinTReconciliator("wls_var"), "residuals"),
                (MinTReconciliator("mint_cov"), "residuals"),
                (MinTReconciliator("ols", nonnegative=True), "series"),
                (MinTReconciliator("mint_cov", nonnegative=True), "residuals"),
            ],
        ),
    )
    def test_reconciliation_dtype_preservation(self, dtype, config):
        recon, series = config
        series_fit = getattr(self, series).astype(dtype)
        if isinstance(recon, FittableDataTransformer):
            recon.fit(series_fit)

        transformed = recon.transform(series_fit)
        assert transformed.dtype == dtype

        dtype_other = "float32" if dtype == "float64" else "float64"
        series_transform = self.series.astype(dtype_other)
        transformed = recon.transform(series_transform)
        assert transformed.dtype == dtype_other


class TestReconciliationInputValidation:
    def test_get_summation_matrix_no_hierarchy(self):
        ts = tg.constant_timeseries(value=1, length=10)
        with pytest.raises(ValueError, match="must have a hierarchy defined"):
            _get_summation_matrix(ts)

    def test_mint_invalid_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            MinTReconciliator(method="invalid")

    def test_mint_wls_var_stochastic_series(self):
        values = np.random.rand(50, 3, 5)
        ts = TimeSeries.from_values(values, columns=["total", "a", "b"])
        ts = ts.with_hierarchy({"a": ["total"], "b": ["total"]})
        reconciliator = MinTReconciliator(method="wls_var")
        with pytest.raises(ValueError, match="stochastic"):
            reconciliator.fit(ts)

    def test_mint_nonnegative_requires_positive_definite_w(self):
        # `wls_val` weights by the average component values, which are negative here
        values = -np.random.rand(50, 3)
        ts = TimeSeries.from_values(values, columns=["total", "a", "b"])
        ts = ts.with_hierarchy({"a": ["total"], "b": ["total"]})
        with pytest.raises(ValueError, match="positive definite"):
            MinTReconciliator(method="wls_val", nonnegative=True).fit(ts)

    def test_mint_unknown_method_in_get_matrices(self):
        values = np.random.rand(50, 3)
        ts = TimeSeries.from_values(values, columns=["total", "a", "b"])
        ts = ts.with_hierarchy({"a": ["total"], "b": ["total"]})
        with pytest.raises(ValueError, match="Unknown method"):
            MinTReconciliator.get_matrices(ts, method="wls")
