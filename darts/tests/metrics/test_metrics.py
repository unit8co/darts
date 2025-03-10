import copy
import inspect
import itertools

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics

from darts import TimeSeries, concatenate
from darts.metrics import metrics
from darts.utils.utils import likelihood_component_names, quantile_names


def sklearn_mape(*args, **kwargs):
    return sklearn.metrics.mean_absolute_percentage_error(*args, **kwargs) * 100.0


def metric_residuals(y_true, y_pred, **kwargs):
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    return np.mean(y_true - y_pred)


def metric_wmape(y_true, y_pred, **kwargs):
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    return 100.0 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def metric_smape(y_true, y_pred, **kwargs):
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    return (
        100.0
        / len(y_true)
        * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    )


def metric_ope(y_true, y_pred, **kwargs):
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    return 100.0 * np.abs((np.sum(y_true) - np.sum(y_pred)) / np.sum(y_true))


def metric_cov(y_true, y_pred, **kwargs):
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    return (
        100.0
        * sklearn.metrics.root_mean_squared_error(y_true, y_pred)
        / np.mean(y_true)
    )


def metric_marre(y_true, y_pred, **kwargs):
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    return (
        100.0
        / len(y_true)
        * np.sum(np.abs((y_true - y_pred) / (np.max(y_true) - np.min(y_true))))
    )


def metric_rmsle(y_true, y_pred, **kwargs):
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    return np.sqrt(
        1 / len(y_true) * np.sum((np.log(y_true + 1) - np.log(y_pred + 1)) ** 2)
    )


def metric_iw(y_true, y_pred, q_interval=None, **kwargs):
    # this tests assumes `y_pred` are stochastic values
    if isinstance(q_interval, tuple):
        q_interval = [q_interval]
    q_interval = np.array(q_interval)
    q_lo = q_interval[:, 0]
    q_hi = q_interval[:, 1]
    y_pred_lo = np.quantile(y_pred, q_lo, axis=2).transpose(1, 2, 0)
    y_pred_hi = np.quantile(y_pred, q_hi, axis=2).transpose(1, 2, 0)
    res = y_pred_hi - y_pred_lo
    return res.reshape(len(y_pred), -1)


def metric_iws(y_true, y_pred, q_interval=None, **kwargs):
    # this tests assumes `y_pred` are stochastic values
    if isinstance(q_interval, tuple):
        q_interval = [q_interval]
    q_interval = np.array(q_interval)
    q_lo = q_interval[:, 0]
    q_hi = q_interval[:, 1]
    y_pred_lo = np.quantile(y_pred, q_lo, axis=2).transpose(1, 2, 0)
    y_pred_hi = np.quantile(y_pred, q_hi, axis=2).transpose(1, 2, 0)
    interval_width = y_pred_hi - y_pred_lo
    res = np.where(
        y_true < y_pred_lo,
        interval_width + 1 / q_lo * (y_pred_lo - y_true),
        interval_width,
    )
    res = np.where(
        y_true > y_pred_hi, interval_width + 1 / (1 - q_hi) * (y_true - y_pred_hi), res
    )
    return res.reshape(len(y_pred), -1)


def metric_ic(y_true, y_pred, q_interval=None, **kwargs):
    # this tests assumes `y_pred` are stochastic values
    if isinstance(q_interval, tuple):
        q_interval = [q_interval]
    q_interval = np.array(q_interval)
    q_lo = q_interval[:, 0]
    q_hi = q_interval[:, 1]
    y_pred_lo = np.quantile(y_pred, q_lo, axis=2).transpose(1, 2, 0)
    y_pred_hi = np.quantile(y_pred, q_hi, axis=2).transpose(1, 2, 0)
    res = np.where((y_pred_lo <= y_true) & (y_true <= y_pred_hi), 1, 0)
    return res.reshape(len(y_pred), -1)


def metric_incs_qr(y_true, y_pred, q_interval=None, **kwargs):
    # this tests assumes `y_pred` are stochastic values
    if isinstance(q_interval, tuple):
        q_interval = [q_interval]
    q_interval = np.array(q_interval)
    q_lo = q_interval[:, 0]
    q_hi = q_interval[:, 1]
    y_pred_lo = np.quantile(y_pred, q_lo, axis=2).transpose(1, 2, 0)
    y_pred_hi = np.quantile(y_pred, q_hi, axis=2).transpose(1, 2, 0)
    res = np.maximum(y_pred_lo - y_true, y_true - y_pred_hi)
    return res.reshape(len(y_pred), -1)


class TestMetrics:
    np.random.seed(42)
    pd_train = pd.Series(
        np.sin(np.pi * np.arange(31) / 4) + 1,
        index=pd.date_range("20121201", "20121231"),
    )
    pd_train_not_periodic = pd.Series(
        range(31), index=pd.date_range("20121201", "20121231")
    )
    pd_series1 = pd.Series(
        range(10), index=pd.date_range("20130101", "20130110")
    ).astype("float64")
    pd_series2 = pd.Series(
        np.random.rand(10) * 10 + 1, index=pd.date_range("20130101", "20130110")
    )
    pd_series3 = pd.Series(
        np.sin(np.pi * np.arange(20) / 4) + 1,
        index=pd.date_range("20130101", "20130120"),
    )
    series_train = TimeSeries.from_series(pd_train)
    series_train_not_periodic = TimeSeries.from_series(pd_train_not_periodic)
    series1: TimeSeries = TimeSeries.from_series(pd_series1)
    pd_series1[:] = pd_series1.mean()
    series0: TimeSeries = TimeSeries.from_series(pd_series1)
    series2: TimeSeries = TimeSeries.from_series(pd_series2)
    series3: TimeSeries = TimeSeries.from_series(pd_series3)
    series12: TimeSeries = series1.stack(series2)
    series21: TimeSeries = series2.stack(series1)
    series1b = TimeSeries.from_times_and_values(
        pd.date_range("20130111", "20130120"), series1.values()
    )
    series2b = TimeSeries.from_times_and_values(
        pd.date_range("20130111", "20130120"), series2.values()
    )
    series12_mean = (series1 + series2) / 2
    series11_stochastic = TimeSeries.from_times_and_values(
        series1.time_index, np.stack([series1.values(), series1.values()], axis=2)
    )
    series22_stochastic = TimeSeries.from_times_and_values(
        series2.time_index, np.stack([series2.values(), series2.values()], axis=2)
    )
    series33_stochastic = TimeSeries.from_times_and_values(
        series3.time_index, np.stack([series3.values(), series3.values()], axis=2)
    )
    series12_stochastic = TimeSeries.from_times_and_values(
        series1.time_index, np.stack([series1.values(), series2.values()], axis=2)
    )

    @pytest.mark.parametrize(
        "metric",
        [
            metrics.ape,
            metrics.sape,
            metrics.mape,
            metrics.smape,
        ],
    )
    def test_ape_zero(self, metric):
        with pytest.raises(ValueError):
            metric(self.series1, self.series1)

        with pytest.raises(ValueError):
            metric(self.series1, self.series1)

    def test_ope_zero(self):
        with pytest.raises(ValueError):
            metrics.ope(
                self.series1 - self.series1.to_series().mean(),
                self.series1 - self.series1.to_series().mean(),
            )

    @pytest.mark.parametrize(
        "config",
        [
            # time dependent but with time reduction
            (metrics.err, False, {"time_reduction": np.mean}),
            (metrics.ae, False, {"time_reduction": np.mean}),
            (metrics.se, False, {"time_reduction": np.mean}),
            (metrics.sle, False, {"time_reduction": np.mean}),
            (metrics.ase, False, {"time_reduction": np.mean}),
            (metrics.sse, False, {"time_reduction": np.mean}),
            (metrics.ape, False, {"time_reduction": np.mean}),
            (metrics.sape, False, {"time_reduction": np.mean}),
            (metrics.arre, False, {"time_reduction": np.mean}),
            (metrics.ql, True, {"time_reduction": np.mean}),
            # time aggregates
            (metrics.merr, False, {}),
            (metrics.mae, False, {}),
            (metrics.mse, False, {}),
            (metrics.rmse, False, {}),
            (metrics.rmsle, False, {}),
            (metrics.mase, False, {}),
            (metrics.msse, False, {}),
            (metrics.rmsse, False, {}),
            (metrics.mape, False, {}),
            (metrics.wmape, False, {}),
            (metrics.smape, False, {}),
            (metrics.ope, False, {}),
            (metrics.marre, False, {}),
            (metrics.r2_score, False, {}),
            (metrics.coefficient_of_variation, False, {}),
            (metrics.qr, True, {}),
            (metrics.mql, True, {}),
            (metrics.dtw_metric, False, {}),
        ],
    )
    def test_output_type_time_aggregated(self, config):
        """Test output types and shapes for time aggregated metrics:
        for single and multiple univariate or multivariate series, in combination
        with different component and series reduction functions."""
        metric, is_probabilistic, kwargs = config
        params = inspect.signature(metric).parameters

        # y true
        y_t_mv = self.series12 + 1
        y_t_uv = y_t_mv.univariate_component(0)
        y_t_multi_mv = [y_t_mv] * 2
        y_t_multi_uv = [y_t_uv] * 2

        # y pred
        y_p_mv = (
            self.series12
            if not is_probabilistic
            else self.series12_stochastic.stack(self.series12_stochastic)
        ) + 1
        y_p_uv = y_p_mv.univariate_component(0)
        y_p_multi_mv = [y_p_mv] * 2
        y_p_multi_uv = [y_p_uv] * 2

        # insample
        kwargs_uv = copy.deepcopy(kwargs)
        kwargs_mv = copy.deepcopy(kwargs)
        kwargs_list_single_uv = copy.deepcopy(kwargs)
        kwargs_list_single_mv = copy.deepcopy(kwargs)
        kwargs_multi_uv = copy.deepcopy(kwargs)
        kwargs_multi_mv = copy.deepcopy(kwargs)
        if "insample" in params:
            insample = self.series_train.stack(self.series_train) + 1
            kwargs_uv["insample"] = insample.univariate_component(0)
            kwargs_mv["insample"] = insample
            kwargs_list_single_uv["insample"] = [kwargs_uv["insample"]]
            kwargs_list_single_mv["insample"] = [kwargs_mv["insample"]]
            kwargs_multi_uv["insample"] = [kwargs_uv["insample"]] * 2
            kwargs_multi_mv["insample"] = [kwargs_mv["insample"]] * 2

        # SINGLE UNIVARIATE SERIES
        # no reduction
        res = metric(
            y_t_uv, y_p_uv, **kwargs_uv, series_reduction=None, component_reduction=None
        )
        assert isinstance(res, float)
        # series reduction
        res = metric(
            y_t_uv,
            y_p_uv,
            **kwargs_uv,
            series_reduction=np.mean,
            component_reduction=None,
        )
        assert isinstance(res, float)
        # comp reduction
        res = metric(
            y_t_uv,
            y_p_uv,
            **kwargs_uv,
            series_reduction=None,
            component_reduction=np.mean,
        )
        assert isinstance(res, float)
        # series and comp reduction
        res = metric(
            y_t_uv,
            y_p_uv,
            **kwargs_uv,
            series_reduction=np.mean,
            component_reduction=np.mean,
        )
        assert isinstance(res, float)

        # LIST OF SINGLE UNIVARIATE SERIES
        # no reduction
        res = metric(
            [y_t_uv],
            [y_p_uv],
            **kwargs_list_single_uv,
            series_reduction=None,
            component_reduction=None,
        )
        assert isinstance(res, list) and len(res) == 1
        assert isinstance(res[0], float)
        # series reduction
        res = metric(
            [y_t_uv],
            [y_p_uv],
            **kwargs_list_single_uv,
            series_reduction=np.mean,
            component_reduction=None,
        )
        assert isinstance(res, float)
        # comp reduction
        res = metric(
            [y_t_uv],
            [y_p_uv],
            **kwargs_list_single_uv,
            series_reduction=None,
            component_reduction=np.mean,
        )
        assert isinstance(res, list) and len(res) == 1
        assert isinstance(res[0], float)
        # series and comp reduction
        res = metric(
            [y_t_uv],
            [y_p_uv],
            **kwargs_list_single_uv,
            series_reduction=np.mean,
            component_reduction=np.mean,
        )
        assert isinstance(res, float)

        # SINGLE MULTIVARIATE SERIES
        # no reduction
        res = metric(
            y_t_mv, y_p_mv, **kwargs_mv, series_reduction=None, component_reduction=None
        )
        assert isinstance(res, np.ndarray)
        assert res.shape == (2,)
        # series reduction
        res = metric(
            y_t_mv,
            y_p_mv,
            **kwargs_mv,
            series_reduction=np.mean,
            component_reduction=None,
        )
        assert isinstance(res, np.ndarray)
        assert res.shape == (2,)
        # comp reduction
        res = metric(
            y_t_mv,
            y_p_mv,
            **kwargs_mv,
            series_reduction=None,
            component_reduction=np.mean,
        )
        assert isinstance(res, float)
        # series and comp reduction
        res = metric(
            y_t_mv,
            y_p_mv,
            **kwargs_mv,
            series_reduction=np.mean,
            component_reduction=np.mean,
        )
        assert isinstance(res, float)

        # LIST OF SINGLE MULTIVARIATE SERIES
        # no reduction
        res = metric(
            [y_t_mv],
            [y_p_mv],
            **kwargs_list_single_mv,
            series_reduction=None,
            component_reduction=None,
        )
        assert isinstance(res, list) and len(res) == 1
        assert isinstance(res[0], np.ndarray) and res[0].shape == (2,)
        # series reduction
        res = metric(
            [y_t_mv],
            [y_p_mv],
            **kwargs_list_single_mv,
            series_reduction=np.mean,
            component_reduction=None,
        )
        assert isinstance(res, np.ndarray) and res.shape == (2,)
        # comp reduction
        res = metric(
            [y_t_mv],
            [y_p_mv],
            **kwargs_list_single_mv,
            series_reduction=None,
            component_reduction=np.mean,
        )
        assert isinstance(res, list) and len(res) == 1
        assert isinstance(res[0], float)
        # series and comp reduction
        res = metric(
            [y_t_mv],
            [y_p_mv],
            **kwargs_list_single_mv,
            series_reduction=np.mean,
            component_reduction=np.mean,
        )
        assert isinstance(res, float)

        # MULTIPLE UNIVARIATE SERIES
        # no reduction
        res = metric(
            y_t_multi_uv,
            y_p_multi_uv,
            **kwargs_multi_uv,
            series_reduction=None,
            component_reduction=None,
        )
        assert isinstance(res, list)
        assert len(res) == 2
        # series reduction
        res = metric(
            y_t_multi_uv,
            y_p_multi_uv,
            **kwargs_multi_uv,
            series_reduction=np.mean,
            component_reduction=None,
        )
        assert isinstance(res, float)
        # comp reduction
        res = metric(
            y_t_multi_uv,
            y_p_multi_uv,
            **kwargs_multi_uv,
            series_reduction=None,
            component_reduction=np.mean,
        )
        assert isinstance(res, list)
        assert len(res) == 2
        # series and comp reduction
        res = metric(
            y_t_multi_uv,
            y_p_multi_uv,
            **kwargs_multi_uv,
            series_reduction=np.mean,
            component_reduction=np.mean,
        )
        assert isinstance(res, float)

        # MULTIPLE MULTIVARIATE SERIES
        # no reduction
        res = metric(
            y_t_multi_mv,
            y_p_multi_mv,
            **kwargs_multi_mv,
            series_reduction=None,
            component_reduction=None,
        )
        assert isinstance(res, list)
        assert len(res) == 2
        assert all(isinstance(el, np.ndarray) for el in res)
        assert all(el.shape == (2,) for el in res)
        # series reduction
        res = metric(
            y_t_multi_mv,
            y_p_multi_mv,
            **kwargs_multi_mv,
            series_reduction=np.mean,
            component_reduction=None,
        )
        assert isinstance(res, np.ndarray)
        assert res.shape == (2,)
        # comp reduction
        res = metric(
            y_t_multi_mv,
            y_p_multi_mv,
            **kwargs_multi_mv,
            series_reduction=None,
            component_reduction=np.mean,
        )
        assert isinstance(res, list)
        assert len(res) == 2
        assert all(isinstance(el, float) for el in res)
        # series and comp reduction
        res = metric(
            y_t_multi_mv,
            y_p_multi_mv,
            **kwargs_multi_mv,
            series_reduction=np.mean,
            component_reduction=np.mean,
        )
        assert isinstance(res, float)

    @pytest.mark.parametrize(
        "config",
        [
            # time dependent
            (metrics.err, False),
            (metrics.ae, False),
            (metrics.se, False),
            (metrics.sle, False),
            (metrics.ase, False),
            (metrics.sse, False),
            (metrics.ape, False),
            (metrics.sape, False),
            (metrics.arre, False),
            (metrics.ql, True),
        ],
    )
    def test_output_type_time_dependent(self, config):
        """Test output types and shapes for time dependent metrics:
        for single and multiple univariate or multivariate series, in combination
        with different component and series reduction functions."""
        metric, is_probabilistic = config
        params = inspect.signature(metric).parameters

        # y true
        y_t_mv = self.series12 + 1
        y_t_uv = y_t_mv.univariate_component(0)
        y_t_multi_mv = [y_t_mv] * 2
        y_t_multi_uv = [y_t_uv] * 2

        # y pred
        y_p_mv = (
            self.series12
            if not is_probabilistic
            else self.series12_stochastic.stack(self.series12_stochastic)
        ) + 1
        y_p_uv = y_p_mv.univariate_component(0)
        y_p_multi_mv = [y_p_mv] * 2
        y_p_multi_uv = [y_p_uv] * 2

        # insample
        kwargs_uv = {}
        kwargs_mv = {}
        kwargs_list_single_uv = {}
        kwargs_list_single_mv = {}
        kwargs_multi_uv = {}
        kwargs_multi_mv = {}
        if "insample" in params:
            insample = self.series_train.stack(self.series_train) + 1
            kwargs_uv["insample"] = insample.univariate_component(0)
            kwargs_mv["insample"] = insample
            kwargs_list_single_uv["insample"] = [kwargs_uv["insample"]]
            kwargs_list_single_mv["insample"] = [kwargs_mv["insample"]]
            kwargs_multi_uv["insample"] = [kwargs_uv["insample"]] * 2
            kwargs_multi_mv["insample"] = [kwargs_mv["insample"]] * 2

        # SINGLE UNIVARIATE SERIES
        # no reduction
        res = metric(
            y_t_uv, y_p_uv, **kwargs_uv, series_reduction=None, component_reduction=None
        )
        assert isinstance(res, np.ndarray) and res.shape == (len(y_p_uv),)
        # series reduction
        res = metric(
            y_t_uv,
            y_p_uv,
            **kwargs_uv,
            series_reduction=np.mean,
            component_reduction=None,
        )
        assert isinstance(res, np.ndarray) and res.shape == (len(y_p_uv),)
        # comp reduction
        res = metric(
            y_t_uv,
            y_p_uv,
            **kwargs_uv,
            series_reduction=None,
            component_reduction=np.mean,
        )
        assert isinstance(res, np.ndarray) and res.shape == (len(y_p_uv),)
        # series and comp reduction
        res = metric(
            y_t_uv,
            y_p_uv,
            **kwargs_uv,
            series_reduction=np.mean,
            component_reduction=np.mean,
        )
        assert isinstance(res, np.ndarray) and res.shape == (len(y_p_uv),)

        # LIST OF SINGLE UNIVARIATE SERIES
        # no reduction
        res = metric(
            [y_t_uv],
            [y_p_uv],
            **kwargs_list_single_uv,
            series_reduction=None,
            component_reduction=None,
        )
        assert isinstance(res, list) and len(res) == 1
        assert isinstance(res[0], np.ndarray) and res[0].shape == (len(y_p_uv),)
        # series reduction
        res = metric(
            [y_t_uv],
            [y_p_uv],
            **kwargs_list_single_uv,
            series_reduction=np.mean,
            component_reduction=None,
        )
        assert isinstance(res, np.ndarray) and res.shape == (len(y_p_uv),)
        # comp reduction
        res = metric(
            [y_t_uv],
            [y_p_uv],
            **kwargs_list_single_uv,
            series_reduction=None,
            component_reduction=np.mean,
        )
        assert isinstance(res, list) and len(res) == 1
        assert isinstance(res[0], np.ndarray) and res[0].shape == (len(y_p_uv),)

        # series and comp reduction
        res = metric(
            [y_t_uv],
            [y_p_uv],
            **kwargs_list_single_uv,
            series_reduction=np.mean,
            component_reduction=np.mean,
        )
        assert isinstance(res, np.ndarray) and res.shape == (len(y_p_uv),)

        # SINGLE MULTIVARIATE SERIES
        # no reduction
        res = metric(
            y_t_mv, y_p_mv, **kwargs_mv, series_reduction=None, component_reduction=None
        )
        assert isinstance(res, np.ndarray) and res.shape == (len(y_t_mv), 2)
        # series reduction
        res = metric(
            y_t_mv,
            y_p_mv,
            **kwargs_mv,
            series_reduction=np.mean,
            component_reduction=None,
        )
        assert isinstance(res, np.ndarray) and res.shape == (len(y_t_mv), 2)
        # comp reduction
        res = metric(
            y_t_mv,
            y_p_mv,
            **kwargs_mv,
            series_reduction=None,
            component_reduction=np.mean,
        )
        assert isinstance(res, np.ndarray) and res.shape == (len(y_t_mv),)
        # series and comp reduction
        res = metric(
            y_t_mv,
            y_p_mv,
            **kwargs_mv,
            series_reduction=np.mean,
            component_reduction=np.mean,
        )
        assert isinstance(res, np.ndarray) and res.shape == (len(y_t_mv),)

        # LIST OF SINGLE MULTIVARIATE SERIES
        # no reduction
        res = metric(
            [y_t_mv],
            [y_p_mv],
            **kwargs_list_single_mv,
            series_reduction=None,
            component_reduction=None,
        )
        assert isinstance(res, list) and len(res) == 1
        assert isinstance(res[0], np.ndarray) and res[0].shape == (10, 2)
        # series reduction
        res = metric(
            [y_t_mv],
            [y_p_mv],
            **kwargs_list_single_mv,
            series_reduction=np.mean,
            component_reduction=None,
        )
        assert isinstance(res, np.ndarray) and res.shape == (10, 2)
        # comp reduction
        res = metric(
            [y_t_mv],
            [y_p_mv],
            **kwargs_list_single_mv,
            series_reduction=None,
            component_reduction=np.mean,
        )
        assert isinstance(res, list) and len(res) == 1
        assert isinstance(res[0], np.ndarray) and res[0].shape == (10,)
        # series and comp reduction
        res = metric(
            [y_t_mv],
            [y_p_mv],
            **kwargs_list_single_mv,
            series_reduction=np.mean,
            component_reduction=np.mean,
        )
        assert isinstance(res, np.ndarray) and res.shape == (10,)

        # MULTIPLE UNIVARIATE SERIES
        # no reduction
        res = metric(
            y_t_multi_uv,
            y_p_multi_uv,
            **kwargs_multi_uv,
            series_reduction=None,
            component_reduction=None,
        )
        assert isinstance(res, list) and len(res) == 2
        assert all(el.shape == (10,) for el in res)
        # series reduction
        res = metric(
            y_t_multi_uv,
            y_p_multi_uv,
            **kwargs_multi_uv,
            series_reduction=np.mean,
            component_reduction=None,
        )
        assert isinstance(res, np.ndarray) and res.shape == (10,)
        # comp reduction
        res = metric(
            y_t_multi_uv,
            y_p_multi_uv,
            **kwargs_multi_uv,
            series_reduction=None,
            component_reduction=np.mean,
        )
        assert isinstance(res, list) and len(res) == 2
        assert all(el.shape == (10,) for el in res)
        # series and comp reduction
        res = metric(
            y_t_multi_uv,
            y_p_multi_uv,
            **kwargs_multi_uv,
            series_reduction=np.mean,
            component_reduction=np.mean,
        )
        assert isinstance(res, np.ndarray) and res.shape == (10,)

        # MULTIPLE MULTIVARIATE SERIES
        # no reduction
        res = metric(
            y_t_multi_mv,
            y_p_multi_mv,
            **kwargs_multi_mv,
            series_reduction=None,
            component_reduction=None,
        )
        assert isinstance(res, list) and len(res) == 2
        assert all(el.shape == (10, 2) for el in res)
        # series reduction
        res = metric(
            y_t_multi_mv,
            y_p_multi_mv,
            **kwargs_multi_mv,
            series_reduction=np.mean,
            component_reduction=None,
        )
        assert isinstance(res, np.ndarray) and res.shape == (10, 2)
        # comp reduction
        res = metric(
            y_t_multi_mv,
            y_p_multi_mv,
            **kwargs_multi_mv,
            series_reduction=None,
            component_reduction=np.mean,
        )
        assert isinstance(res, list) and len(res) == 2
        assert all(el.shape == (10,) for el in res)
        # series and comp reduction
        res = metric(
            y_t_multi_mv,
            y_p_multi_mv,
            **kwargs_multi_mv,
            series_reduction=np.mean,
            component_reduction=np.mean,
        )
        assert isinstance(res, np.ndarray) and res.shape == (10,)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [
                # time dependent
                (metrics.err, False),
                (metrics.ae, False),
                (metrics.se, False),
                (metrics.sle, False),
                (metrics.ase, False),
                (metrics.sse, False),
                (metrics.ape, False),
                (metrics.sape, False),
                (metrics.arre, False),
                (metrics.ql, True),
                # time aggregates
                (metrics.merr, False),
                (metrics.mae, False),
                (metrics.mse, False),
                (metrics.rmse, False),
                (metrics.rmsle, False),
                (metrics.mase, False),
                (metrics.msse, False),
                (metrics.rmsse, False),
                (metrics.mape, False),
                (metrics.wmape, False),
                (metrics.smape, False),
                (metrics.ope, False),
                (metrics.marre, False),
                (metrics.r2_score, False),
                (metrics.coefficient_of_variation, False),
                (metrics.qr, True),
                (metrics.mql, True),
                (metrics.dtw_metric, False),
            ],
            ["time", "component", "series"],
        ),
    )
    def test_reduction_fn_validity(self, config):
        """Tests reduction functions sanity checks."""
        (metric, is_probabilistic), red_name = config
        params = inspect.signature(metric).parameters
        has_time_red = "time_reduction" in params

        # y true
        y_t = self.series12 + 1

        # y pred
        y_p = (
            self.series12
            if not is_probabilistic
            else self.series12_stochastic.stack(self.series12_stochastic)
        ) + 1

        # insample
        kwargs = {}
        if "insample" in params:
            kwargs["insample"] = self.series_train.stack(self.series_train) + 1

        red_param = red_name + "_reduction"
        if red_name == "time" and not has_time_red:
            # time_reduction not an argument
            with pytest.raises(TypeError):
                _ = metric(y_t, y_p, **kwargs, **{red_param: np.nanmean})
            return

        # check that valid fn works
        _ = metric(y_t, y_p, **kwargs, **{red_param: np.nanmean})

        # no axis in fn
        with pytest.raises(ValueError) as err:
            _ = metric(y_t, y_p, **kwargs, **{red_param: lambda x: np.nanmean(x)})
        assert str(err.value).endswith("Must have a parameter called `axis`.")
        # with axis it works
        _ = metric(
            y_t, y_p, **kwargs, **{red_param: lambda x, axis: np.nanmean(x, axis)}
        )

        # invalid output type: list
        with pytest.raises(ValueError) as err:
            _ = metric(
                y_t,
                y_p,
                **kwargs,
                **{red_param: lambda x, axis: np.nanmean(x, axis).tolist()},
            )
        assert str(err.value).endswith(
            "Expected type `np.ndarray`, received type=`<class 'list'>`."
        )

        # invalid output type: reduced to float
        with pytest.raises(ValueError) as err:
            _ = metric(y_t, y_p, **kwargs, **{red_param: lambda x, axis: x[0, 0]})
        assert str(err.value).endswith(
            "Expected type `np.ndarray`, received type=`<class 'numpy.float64'>`."
        )

        # invalid output shape: did not reduce correctly
        with pytest.raises(ValueError) as err:
            _ = metric(y_t, y_p, **kwargs, **{red_param: lambda x, axis: x[:2, :2]})
        assert str(err.value).startswith(
            f"Invalid `{red_param}` function output shape:"
        )

    @pytest.mark.parametrize(
        "config",
        [
            # time dependent
            (metrics.err, 0, False, {"time_reduction": np.mean}),
            (metrics.ae, 0, False, {"time_reduction": np.mean}),
            (metrics.se, 0, False, {"time_reduction": np.mean}),
            (metrics.sle, 0, False, {"time_reduction": np.mean}),
            (metrics.ase, 0, False, {"time_reduction": np.mean}),
            (metrics.sse, 0, False, {"time_reduction": np.mean}),
            (metrics.ape, 0, False, {"time_reduction": np.mean}),
            (metrics.sape, 0, False, {"time_reduction": np.mean}),
            (metrics.arre, 0, False, {"time_reduction": np.mean}),
            (metrics.ql, 0, True, {"time_reduction": np.mean}),
            # time aggregates
            (metrics.merr, 0, False, {}),
            (metrics.mae, 0, False, {}),
            (metrics.mse, 0, False, {}),
            (metrics.rmse, 0, False, {}),
            (metrics.rmsle, 0, False, {}),
            (metrics.mase, 0, False, {}),
            (metrics.msse, 0, False, {}),
            (metrics.rmsse, 0, False, {}),
            (metrics.mape, 0, False, {}),
            (metrics.wmape, 0, False, {}),
            (metrics.smape, 0, False, {}),
            (metrics.ope, 0, False, {}),
            (metrics.marre, 0, False, {}),
            (metrics.r2_score, 1, False, {}),
            (metrics.coefficient_of_variation, 0, False, {}),
            (metrics.qr, 0, True, {}),
            (metrics.mql, 0, True, {}),
            (metrics.dtw_metric, 0, False, {}),
        ],
    )
    def test_same(self, config):
        metric, score_exp, is_probabilistic, kwargs = config
        params = inspect.signature(metric).parameters
        y_true = self.series1 + 1
        y_pred = (
            self.series1 + 1 if not is_probabilistic else self.series11_stochastic + 1
        )
        if "insample" in params:
            assert metric(y_true, y_pred, self.series_train + 1, **kwargs) == score_exp
        else:
            assert metric(y_true, y_pred, **kwargs) == score_exp

    def test_r2(self):
        from sklearn.metrics import r2_score

        assert metrics.r2_score(self.series1, self.series0) == 0
        assert metrics.r2_score(self.series1, self.series2) == r2_score(
            self.series1.values(), self.series2.values()
        )

        self.helper_test_multivariate_duplication_equality(metrics.r2_score)
        self.helper_test_multiple_ts_duplication_equality(metrics.r2_score)
        self.helper_test_nan(metrics.r2_score)

    @pytest.mark.parametrize(
        "config",
        [
            (metrics.se, False, {"time_reduction": np.nanmean}),
            (metrics.mse, True, {}),
        ],
    )
    def test_se(self, config):
        metric, is_aggregate, kwargs = config
        self.helper_test_shape_equality(metric, **kwargs)
        self.helper_test_nan(metric, **kwargs)
        self.helper_test_non_aggregate(metric, is_aggregate)

    @pytest.mark.parametrize(
        "config",
        [
            (metrics.ae, False, {"time_reduction": np.nanmean}),
            (metrics.mae, True, {}),
        ],
    )
    def test_ae(self, config):
        metric, is_aggregate, kwargs = config
        self.helper_test_shape_equality(metric, **kwargs)
        self.helper_test_nan(metric, **kwargs)
        self.helper_test_non_aggregate(metric, is_aggregate)

    def test_rmse(self):
        self.helper_test_multivariate_duplication_equality(metrics.rmse)
        self.helper_test_multiple_ts_duplication_equality(metrics.rmse)

        np.testing.assert_array_almost_equal(
            metrics.rmse(
                self.series1.append(self.series2b),
                self.series2.append(self.series1b),
            ),
            metrics.mse(
                self.series12,
                self.series21,
                component_reduction=lambda x, axis: np.sqrt(np.mean(x, axis=axis)),
            ),
        )
        self.helper_test_nan(metrics.rmse)

    @pytest.mark.parametrize(
        "config",
        [
            (metrics.sle, False, {"time_reduction": np.nanmean}),
            (metrics.rmsle, True, {}),
        ],
    )
    def test_sle(self, config):
        metric, is_aggregate, kwargs = config
        self.helper_test_multivariate_duplication_equality(metric, **kwargs)
        self.helper_test_multiple_ts_duplication_equality(metric, **kwargs)
        self.helper_test_nan(metric, **kwargs)
        self.helper_test_non_aggregate(metric, is_aggregate)

    @pytest.mark.parametrize(
        "config",
        [
            (metrics.arre, False, {"time_reduction": np.nanmean}),
            (metrics.marre, True, {}),
        ],
    )
    def test_arre(self, config):
        metric, is_aggregate, kwargs = config
        np.testing.assert_array_almost_equal(
            metric(self.series1, self.series2, **kwargs),
            metric(self.series1 + 100, self.series2 + 100, **kwargs),
        )
        self.helper_test_multivariate_duplication_equality(metric, **kwargs)
        self.helper_test_multiple_ts_duplication_equality(metric, **kwargs)
        self.helper_test_nan(metric, **kwargs)
        self.helper_test_non_aggregate(metric, is_aggregate)

        with pytest.raises(ValueError) as exc:
            _ = metric(
                TimeSeries.from_values(np.ones((3, 1, 1))),
                TimeSeries.from_values(np.ones((3, 1, 1))),
            )
        assert str(exc.value).startswith(
            "The difference between the max and min values must "
        )

    @pytest.mark.parametrize(
        "metric",
        [
            metrics.ase,
            metrics.sse,
            metrics.mase,
            metrics.msse,
            metrics.rmsse,
        ],
    )
    def test_season(self, metric):
        with pytest.raises(ValueError):
            metric(self.series3, self.series3 * 1.3, self.series_train, 8)

    @pytest.mark.parametrize(
        "config",
        [
            (metrics.err, False, {"time_reduction": np.nanmean}),
            (metrics.merr, True, {}),
        ],
    )
    def test_res(self, config):
        metric, is_aggregate, kwargs = config
        self.helper_test_shape_equality(metric, **kwargs)
        self.helper_test_nan(metric, **kwargs)

        assert metric(self.series1, self.series1 + 1, **kwargs) == -1.0
        assert metric(self.series1, self.series1 - 1, **kwargs) == 1.0
        self.helper_test_non_aggregate(metric, is_aggregate, val_exp=-1.0)

    def test_coefficient_of_variation(self):
        self.helper_test_multivariate_duplication_equality(
            metrics.coefficient_of_variation
        )
        self.helper_test_multiple_ts_duplication_equality(
            metrics.coefficient_of_variation
        )
        self.helper_test_nan(metrics.coefficient_of_variation)

    @pytest.mark.parametrize(
        "config",
        [
            (metrics.ape, False, {"time_reduction": np.nanmean}),
            (metrics.mape, True, {}),
        ],
    )
    def test_ape(self, config):
        metric, is_aggregate, kwargs = config
        self.helper_test_multivariate_duplication_equality(metric, **kwargs)
        self.helper_test_multiple_ts_duplication_equality(metric, **kwargs)
        self.helper_test_nan(metric, **kwargs)
        self.helper_test_non_aggregate(metric, is_aggregate)

    @pytest.mark.parametrize(
        "config",
        [
            (metrics.ape, False, {"time_reduction": np.nanmean}),
            (metrics.mape, True, {}),
        ],
    )
    def test_sape(self, config):
        metric, is_aggregate, kwargs = config
        self.helper_test_multivariate_duplication_equality(metric, **kwargs)
        self.helper_test_multiple_ts_duplication_equality(metric, **kwargs)
        self.helper_test_nan(metric, **kwargs)
        self.helper_test_non_aggregate(metric, is_aggregate)

    @pytest.mark.parametrize(
        "config",
        [
            (metrics.ase, False, {"time_reduction": np.nanmean}),
            (metrics.sse, False, {"time_reduction": np.nanmean}),
            (metrics.mase, True, {}),
            (metrics.msse, True, {}),
            (metrics.rmsse, True, {}),
        ],
    )
    def test_scaled_errors(self, config):
        metric, is_aggregate, kwargs = config
        insample = self.series_train
        test_cases, _ = self.get_test_cases()
        for s1, s2 in test_cases:
            # multivariate, series as args
            np.testing.assert_array_almost_equal(
                metric(s1.stack(s1), s2.stack(s2), insample.stack(insample), **kwargs),
                metric(s1, s2, insample, **kwargs),
            )

            # test that internal slicing gives identical results with longer `insample` series
            np.testing.assert_array_almost_equal(
                metric(s1, s2, insample, **kwargs),
                metric(
                    s1,
                    s2,
                    insample.append_values(np.array([100.0, 200.0, 300.0])),
                    **kwargs,
                ),
            )

            # multi-ts, series as kwargs
            np.testing.assert_array_almost_equal(
                metric(
                    actual_series=[s1] * 2,
                    pred_series=[s2] * 2,
                    insample=[insample] * 2,
                    **kwargs,
                ),
                metric(s1, s2, insample, **kwargs),
            )

            # checking with n_jobs and verbose
            np.testing.assert_array_almost_equal(
                metric(
                    [s1] * 5, pred_series=[s2] * 5, insample=[insample] * 5, **kwargs
                ),
                metric(
                    [s1] * 5,
                    [s2] * 5,
                    insample=[insample] * 5,
                    n_jobs=-1,
                    verbose=True,
                    **kwargs,
                ),
            )

        # fails with type `m` different from `int`
        with pytest.raises(ValueError) as err:
            metric(self.series2, self.series2, insample, m=None)
        assert str(err.value).startswith("Seasonality `m` must be of type `int`")
        # fails if `insample` ends more than one time step before start of `pred_series`
        with pytest.raises(ValueError) as err:
            metric(self.series1, self.series2, insample[:-1], m=1)
        assert str(err.value).startswith(
            "The `insample` series must start before the `pred_series`"
        )
        # fails if `insample` starts at the beginning of `pred_series`
        with pytest.raises(ValueError) as err:
            metric(self.series1, self.series2, self.series2, m=1)
        assert str(err.value).startswith(
            "The `insample` series must start before the `pred_series`"
        )
        # fails if `insample` starts after the beginning of `pred_series`
        with pytest.raises(ValueError) as err:
            metric(self.series1, self.series2, self.series2[1:], m=1)
        assert str(err.value).startswith(
            "The `insample` series must start before the `pred_series`"
        )
        # wrong number of components
        with pytest.raises(ValueError):
            metric(self.series1, self.series2, insample.stack(insample))
        # multi-ts, second series is not a TimeSeries
        with pytest.raises(ValueError):
            metric([self.series1] * 2, self.series2, [insample] * 2)
        # multi-ts, insample series is not a TimeSeries
        with pytest.raises(ValueError):
            metric([self.series1] * 2, [self.series2] * 2, insample)
        # multi-ts one array has different length
        with pytest.raises(ValueError):
            metric([self.series1] * 2, [self.series2] * 2, [insample] * 3)

    def test_ope(self):
        self.helper_test_multivariate_duplication_equality(metrics.ope)
        self.helper_test_multiple_ts_duplication_equality(metrics.ope)
        self.helper_test_nan(metrics.ope)

    def test_rho_risk(self):
        # deterministic not supported
        with pytest.raises(ValueError):
            metrics.qr(self.series1, self.series1)

        # general univariate, multivariate and multi-ts tests
        self.helper_test_multivariate_duplication_equality(
            metrics.qr, is_stochastic=True
        )
        self.helper_test_multiple_ts_duplication_equality(
            metrics.qr, is_stochastic=True
        )
        self.helper_test_nan(metrics.qr, is_stochastic=True)

        # test perfect predictions -> risk = 0
        for q in [0.25, 0.5]:
            np.testing.assert_array_almost_equal(
                metrics.qr(self.series1, self.series11_stochastic, q=q), 0.0
            )
        np.testing.assert_array_almost_equal(
            metrics.qr(self.series12_mean, self.series12_stochastic, q=0.5), 0.0
        )

        # test whether stochastic sample from two TimeSeries (ts) represents the individual ts at 0. and 1. quantiles
        s1 = self.series1
        s2 = self.series1 * 2
        s12_stochastic = TimeSeries.from_times_and_values(
            s1.time_index, np.stack([s1.values(), s2.values()], axis=2)
        )
        np.testing.assert_array_almost_equal(metrics.qr(s1, s12_stochastic, q=0.0), 0.0)
        np.testing.assert_array_almost_equal(metrics.qr(s2, s12_stochastic, q=1.0), 0.0)

        # preds must be probabilistic
        q_names = likelihood_component_names(
            self.series1.components,
            quantile_names([0.5]),
        )
        with pytest.raises(ValueError) as exc:
            metrics.qr(
                self.series1,
                self.series1.with_columns_renamed(self.series1.components, q_names),
                q=0.5,
            )
        assert (
            str(exc.value)
            == "quantile risk (qr) should only be computed for stochastic predicted TimeSeries."
        )

    @pytest.mark.parametrize(
        "config",
        [
            (metrics.ql, False, {"time_reduction": np.nanmean}),
            (metrics.mql, True, {}),
        ],
    )
    def test_quantile_loss(self, config):
        metric, is_aggregate, kwargs = config
        # deterministic not supported
        with pytest.raises(ValueError):
            metric(self.series1, self.series1, **kwargs)

        # general univariate, multivariate and multi-ts tests
        self.helper_test_multivariate_duplication_equality(
            metric, is_stochastic=True, **kwargs
        )
        self.helper_test_multiple_ts_duplication_equality(
            metric, is_stochastic=True, **kwargs
        )
        self.helper_test_nan(metric, is_stochastic=True, **kwargs)

        # test perfect predictions -> risk = 0
        for q in [0.25, 0.5]:
            np.testing.assert_array_almost_equal(
                metric(self.series1, self.series11_stochastic, q=q, **kwargs), 0.0
            )

        # test whether stochastic sample from two TimeSeries (ts) represents the individual ts at 0. and 1. quantiles
        s1 = self.series1
        s2 = self.series1 * 2
        s12_stochastic = TimeSeries.from_times_and_values(
            s1.time_index, np.stack([s1.values(), s2.values()], axis=2)
        )
        np.testing.assert_array_almost_equal(
            metric(s1, s12_stochastic, q=1.0, **kwargs), 0.0
        )
        np.testing.assert_array_almost_equal(
            metric(s2, s12_stochastic, q=0.0, **kwargs), 0.0
        )

    def test_metrics_arguments(self):
        series00 = self.series0.stack(self.series0)
        series11 = self.series1.stack(self.series1)
        assert (
            metrics.r2_score(series11, series00, True, component_reduction=np.mean) == 0
        )
        assert metrics.r2_score(series11, series00, component_reduction=np.mean) == 0
        assert (
            metrics.r2_score(
                series11, pred_series=series00, component_reduction=np.mean
            )
            == 0
        )
        assert (
            metrics.r2_score(
                series00, actual_series=series11, component_reduction=np.mean
            )
            == 0
        )
        assert (
            metrics.r2_score(
                True,
                component_reduction=np.mean,
                pred_series=series00,
                actual_series=series11,
            )
            == 0
        )
        assert (
            metrics.r2_score(
                series00, True, component_reduction=np.mean, actual_series=series11
            )
            == 0
        )
        assert (
            metrics.r2_score(
                series11, True, component_reduction=np.mean, pred_series=series00
            )
            == 0
        )

        # should fail if kwargs are passed as args, because of the "*"
        with pytest.raises(TypeError):
            metrics.r2_score(series00, series11, False, 0.5, np.mean)

    def test_multiple_ts_rmse(self):
        # simple test
        multi_ts_1 = [self.series1 + 1, self.series1 + 1]
        multi_ts_2 = [self.series1 + 2, self.series1 + 1]
        assert (
            metrics.rmse(
                multi_ts_1,
                multi_ts_2,
                component_reduction=np.mean,
                series_reduction=np.mean,
            )
            == 0.5
        )

    @pytest.mark.parametrize(
        "config",
        [
            (metrics.err, "min", {"time_reduction": np.nanmean}),
            (metrics.ae, "max", {"time_reduction": np.nanmean}),
            (metrics.se, "max", {"time_reduction": np.nanmean}),
            (metrics.sle, "max", {"time_reduction": np.nanmean}),
            (metrics.ape, "max", {"time_reduction": np.nanmean}),
            (metrics.sape, "max", {"time_reduction": np.nanmean}),
            (metrics.arre, "max", {"time_reduction": np.nanmean}),
            (metrics.merr, "min", {}),
            (metrics.mae, "max", {}),
            (metrics.mse, "max", {}),
            (metrics.rmse, "max", {}),
            (metrics.rmsle, "max", {}),
            (metrics.mape, "max", {}),
            (metrics.wmape, "max", {}),
            (metrics.smape, "max", {}),
            (metrics.ope, "max", {}),
            (metrics.marre, "max", {}),
            (metrics.r2_score, "min", {}),
            (metrics.coefficient_of_variation, "max", {}),
        ],
    )
    def test_multiple_ts(self, config):
        """Tests that univariate, multivariate and multi-ts give same metrics with same values."""
        metric, series_reduction, kwargs = config
        series_reduction = getattr(np, series_reduction)

        dim = 2
        series11 = self.series1.stack(self.series1) + 1
        series22 = self.series2.stack(self.series2)
        multi_1 = [series11] * dim
        multi_2 = [series22] * dim

        np.testing.assert_array_almost_equal(
            metric(self.series1 + 1, self.series2, **kwargs),
            metric(series11, series22, **kwargs),
        )
        np.testing.assert_array_almost_equal(
            np.array([metric(series11, series22, **kwargs)] * 2),
            np.array(metric(multi_1, multi_2, **kwargs)),
        )

        # trying different functions
        shifted_1 = self.series1 + 1
        shifted_2 = self.series1 + 2
        shifted_3 = self.series1 + 3

        assert metric(
            [shifted_1, shifted_1],
            [shifted_2, shifted_3],
            component_reduction=np.mean,
            series_reduction=series_reduction,
            **kwargs,
        ) == metric(shifted_1, shifted_3, **kwargs)

        # checking if the result is the same with different n_jobs and verbose True
        assert metric(
            [shifted_1, shifted_1],
            [shifted_2, shifted_3],
            component_reduction=np.mean,
            series_reduction=np.max,
            **kwargs,
        ) == metric(
            [shifted_1, shifted_1],
            [shifted_2, shifted_3],
            component_reduction=np.mean,
            series_reduction=np.max,
            n_jobs=-1,
            verbose=True,
            **kwargs,
        )

    @pytest.mark.parametrize(
        "config",
        [
            (metrics.err, metric_residuals, {}, {"time_reduction": np.nanmean}),
            (
                metrics.ae,
                sklearn.metrics.mean_absolute_error,
                {},
                {"time_reduction": np.nanmean},
            ),
            (
                metrics.se,
                sklearn.metrics.mean_squared_error,
                {},
                {"time_reduction": np.nanmean},
            ),
            (
                lambda *args: np.sqrt(metrics.sle(*args, time_reduction=np.nanmean)),
                metric_rmsle,
                {},
                {},
            ),
            (metrics.ape, sklearn_mape, {}, {"time_reduction": np.nanmean}),
            (metrics.sape, metric_smape, {}, {"time_reduction": np.nanmean}),
            (metrics.arre, metric_marre, {}, {"time_reduction": np.nanmean}),
            (metrics.merr, metric_residuals, {}, {}),
            (metrics.mae, sklearn.metrics.mean_absolute_error, {}, {}),
            (metrics.mse, sklearn.metrics.mean_squared_error, {}, {}),
            (metrics.rmse, sklearn.metrics.root_mean_squared_error, {}, {}),
            (metrics.rmsle, metric_rmsle, {}, {}),
            (metrics.mape, sklearn_mape, {}, {}),
            (metrics.wmape, metric_wmape, {}, {}),
            (metrics.smape, metric_smape, {}, {}),
            (metrics.ope, metric_ope, {}, {}),
            (metrics.marre, metric_marre, {}, {}),
            (metrics.r2_score, sklearn.metrics.r2_score, {}, {}),
            (metrics.coefficient_of_variation, metric_cov, {}, {}),
        ],
    )
    def test_metrics_deterministic(self, config):
        """Tests deterministic metrics against a reference metric"""
        metric, metric_ref, ref_kwargs, kwargs = config
        y_true = self.series1.stack(self.series1) + 1
        y_pred = y_true + 1

        y_true = [y_true] * 2
        y_pred = [y_pred] * 2

        score = metric(y_true, y_pred, **kwargs)
        score_ref = metric_ref(y_true[0].values(), y_pred[0].values(), **ref_kwargs)
        np.testing.assert_array_almost_equal(score, np.array(score_ref))

    @pytest.mark.parametrize(
        "config",
        [
            (
                metrics.ql,
                [(0.30, 0.30), (0.030, 0.030), (0.30, 0.30)],
                "q",
                {"time_reduction": np.nanmean},
            ),
            (metrics.mql, [(0.30, 0.30), (0.030, 0.030), (0.30, 0.30)], "q", {}),
            (
                metrics.qr,
                [(0.30, 0.025), (0.030, 0.0025), (0.30, 0.025)],
                "q",
                {},
            ),
        ],
    )
    def test_metrics_probabilistic(self, config):
        """Tests probabilistic metrics against reference scores"""
        metric, scores_exp, q_param, kwargs = config
        np.random.seed(0)
        x = np.random.normal(loc=0.0, scale=1.0, size=10000)
        y = np.array([
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
        ]).reshape(3, 2, 1)

        y_true = [TimeSeries.from_values(y)] * 2
        y_pred = [TimeSeries.from_values(y + x)] * 2

        for quantile, score_exp in zip([0.1, 0.5, 0.9], scores_exp):
            scores = metric(
                y_true,
                y_pred,
                **{q_param: quantile},
                component_reduction=None,
                **kwargs,
            )
            assert (scores < np.array(score_exp).reshape(1, -1)).all()

    def helper_test_shape_equality(self, metric, **kwargs):
        np.testing.assert_array_almost_equal(
            metric(self.series12, self.series21, **kwargs),
            metric(
                self.series1.append(self.series2b),
                self.series2.append(self.series1b),
                **kwargs,
            ),
        )

    def get_test_cases(self, **kwargs):
        # stochastic metrics (q-risk) behave similar to deterministic metrics if all samples have equal values
        if "is_stochastic" in kwargs and kwargs["is_stochastic"]:
            test_cases = [
                (self.series1 + 1, self.series22_stochastic),
                (self.series1 + 1, self.series33_stochastic),
                (self.series2, self.series33_stochastic),
            ]
            kwargs.pop("is_stochastic", 0)
        else:
            test_cases = [
                (self.series1 + 1, self.series2),
                (self.series1 + 1, self.series3),
                (self.series2, self.series3),
            ]
        return test_cases, kwargs

    def helper_test_multivariate_duplication_equality(self, metric, **kwargs):
        test_cases, kwargs = self.get_test_cases(**kwargs)

        for s1, s2 in test_cases:
            s11 = s1.stack(s1)
            s22 = s2.stack(s2)
            # default intra
            np.testing.assert_array_almost_equal(
                metric(s1, s2, **kwargs), metric(s11, s22, **kwargs)
            )
            # custom intra
            np.testing.assert_array_almost_equal(
                metric(
                    s1,
                    s2,
                    **kwargs,
                    component_reduction=(lambda x, axis: x[0, 0:1]),
                ),
                metric(
                    s11,
                    s22,
                    **kwargs,
                    component_reduction=(lambda x, axis: x[0, 0:1]),
                ),
            )

    def helper_test_multiple_ts_duplication_equality(self, metric, **kwargs):
        test_cases, kwargs = self.get_test_cases(**kwargs)

        for s1, s2 in test_cases:
            s11 = [s1.stack(s1)] * 2
            s22 = [s2.stack(s2)] * 2
            # default intra and inter
            np.testing.assert_almost_equal(
                actual=np.array([metric(s1, s2, **kwargs)] * 2),
                desired=np.array(metric(s11, s22, **kwargs)),
            )

            # custom intra and inter
            np.testing.assert_almost_equal(
                metric(
                    s1,
                    s2,
                    **kwargs,
                    component_reduction=np.mean,
                    series_reduction=np.max,
                ),
                metric(
                    s11,
                    s22,
                    **kwargs,
                    component_reduction=np.mean,
                    series_reduction=np.max,
                ),
            )

    def helper_test_nan(self, metric, **kwargs):
        test_cases, kwargs = self.get_test_cases(**kwargs)

        for s1, s2 in test_cases:
            # univariate
            non_nan_metric = metric(s1[:9] + 1, s2[:9], **kwargs)
            nan_s1 = s1.copy()
            nan_s1._xa.values[-1, :, :] = np.nan
            nan_metric = metric(nan_s1 + 1, s2, **kwargs)
            assert non_nan_metric == nan_metric

            # multivariate + multi-TS
            s11 = [s1.stack(s1)] * 2
            s22 = [s2.stack(s2)] * 2
            non_nan_metric = metric(
                [s[:9] + 1 for s in s11], [s[:9] for s in s22], **kwargs
            )
            nan_s11 = s11.copy()
            for s in nan_s11:
                s._xa.values[-1, :, :] = np.nan
            nan_metric = metric([s + 1 for s in nan_s11], s22, **kwargs)
            np.testing.assert_array_equal(non_nan_metric, nan_metric)

    def helper_test_non_aggregate(self, metric, is_aggregate, val_exp=None):
        if is_aggregate:
            return

        # do not aggregate over time
        res = metric(self.series1 + 1, self.series1 + 2)
        assert len(res) == len(self.series1)

        if val_exp is not None:
            assert (res == -1.0).all()

    @pytest.mark.parametrize(
        "config",
        list(
            itertools.product(
                [
                    # time dependent but with time reduction
                    metrics.err,
                    metrics.ae,
                    metrics.se,
                    metrics.sle,
                    metrics.ase,
                    metrics.sse,
                    metrics.ape,
                    metrics.sape,
                    metrics.arre,
                    metrics.ql,
                    # time aggregates
                    metrics.merr,
                    metrics.mae,
                    metrics.mse,
                    metrics.rmse,
                    metrics.rmsle,
                    metrics.mase,
                    metrics.msse,
                    metrics.rmsse,
                    metrics.mape,
                    metrics.wmape,
                    metrics.smape,
                    metrics.ope,
                    metrics.marre,
                    metrics.r2_score,
                    metrics.coefficient_of_variation,
                    metrics.mql,
                ],
                [True, False],  # univariate series
                [True, False],  # single series
            )
        ),
    )
    def test_metric_quantiles(self, config):
        """Test output types and shapes for time aggregated metrics with quantiles:
        for single and multiple univariate or multivariate series, in combination
        with different component and series reduction functions."""
        np.random.seed(42)
        metric, is_univar, is_single = config
        params = inspect.signature(metric).parameters

        n_comp = 1 if is_univar else 2

        qs_all = [0.1, 0.5, 0.8]
        components = [str(i) for i in range(n_comp)]

        series_vals = np.random.random((10, n_comp, 1))

        pred_prob_vals = np.random.random((10, n_comp, 100))

        pred_vals_qs = []
        for i in range(n_comp):
            pred_vals_qs.append(
                np.quantile(pred_prob_vals[:, [i]], qs_all, axis=2).transpose(1, 0, 2)
            )
        pred_vals_qs = np.concatenate(pred_vals_qs, axis=1)
        pred_components = likelihood_component_names(
            components=components, parameter_names=quantile_names(q=qs_all)
        )

        series = TimeSeries.from_values(series_vals, columns=components)
        series_q_exp = concatenate(
            [series[comp] for comp in components for _ in qs_all], axis=1
        )
        pred_prob = TimeSeries.from_values(pred_prob_vals, columns=components)
        pred_qs = TimeSeries.from_values(pred_vals_qs, columns=pred_components)
        insample = series.shift(-len(series))
        insample_q_exp = concatenate(
            [insample[comp] for comp in components for _ in qs_all], axis=1
        )
        shape_time = (len(pred_qs),) if "time_reduction" in params else tuple()

        if not is_single:
            series = [series] * 2
            series_q_exp = [series_q_exp] * 2
            pred_prob = [pred_prob] * 2
            pred_qs = [pred_qs] * 2
            insample = [insample] * 2
            insample_q_exp = [insample_q_exp] * 2

        kwargs = {"actual_series": series}
        if "insample" in params:
            kwargs["insample"] = insample

        def check_res(
            pred_prob_, pred_qs_, shape_exp, series_reduction=None, **test_kwargs
        ):
            res_prob = metric(
                pred_series=pred_prob_,
                series_reduction=series_reduction,
                **kwargs,
                **test_kwargs,
            )
            res_qs = metric(
                pred_series=pred_qs_,
                series_reduction=series_reduction,
                **kwargs,
                **test_kwargs,
            )
            if is_single or series_reduction is not None:
                res_prob = [res_prob]
                res_qs = [res_qs]
            if series_reduction is None and not is_single:
                assert len(res_prob) == len(res_qs) == len(pred_prob_)

            for res_p, res_q in zip(res_prob, res_qs):
                assert res_p.shape == res_q.shape == shape_exp
                np.testing.assert_array_almost_equal(res_p, res_q)

        check_res(pred_prob, pred_qs, shape_time, q=0.1)
        # one quantile as list
        check_res(pred_prob, pred_qs, shape_time, q=[0.1])
        # multiple quantiles
        check_res(pred_prob, pred_qs, shape_time + (2,), q=[0.1, 0.8])
        # all quantiles
        check_res(pred_prob, pred_qs, shape_time + (3,), q=[0.1, 0.5, 0.8])
        qs = [0.1, 0.8]
        # component and series reduction
        check_res(
            pred_prob,
            pred_qs,
            shape_time + (len(qs),),
            q=qs,
            component_reduction=np.mean,
            series_reduction=np.mean,
        )
        # no component reduction
        check_res(
            pred_prob,
            pred_qs,
            shape_time + (len(qs) * n_comp,),
            q=qs,
            component_reduction=None,
            series_reduction=np.mean,
        )
        # no series reduction
        check_res(
            pred_prob,
            pred_qs,
            shape_time + (len(qs),),
            q=qs,
            component_reduction=np.mean,
            series_reduction=None,
        )
        # no series and component reduction
        check_res(
            pred_prob,
            pred_qs,
            shape_time + (len(qs) * n_comp,),
            q=qs,
            component_reduction=None,
            series_reduction=None,
        )

        # check that we get identical results as when computing each quantile component against the actual
        # target component directly
        kwargs_direct = copy.deepcopy(kwargs)
        q_direct = {}
        if metric.__name__ not in ["ql", "mql"]:
            kwargs_direct["actual_series"] = series_q_exp
            if "insample" in params:
                kwargs_direct["insample"] = insample_q_exp
        else:
            q_direct["q"] = qs_all
            kwargs_direct["actual_series"] = series

        res_direct = metric(
            pred_series=pred_qs, component_reduction=None, **kwargs_direct, **q_direct
        )
        res_qs = metric(
            pred_series=pred_qs,
            component_reduction=None,
            q=qs_all,
            **kwargs,
        )
        np.testing.assert_array_almost_equal(res_direct, res_qs)

    def test_invalid_quantiles(self):
        np.random.seed(42)
        series_a = TimeSeries.from_values(np.random.random((10, 2, 1)))
        series_b = TimeSeries.from_values(np.random.random((10, 2, 10)))

        # unsorted quantiles
        with pytest.raises(ValueError) as exc:
            _ = metrics.mae(series_a, series_b, q=[0.2, 0.1])
        assert "a sequence of increasing order" in str(exc.value)

        # non-unique values metrics
        with pytest.raises(ValueError) as exc:
            _ = metrics.mae(series_a, series_b, q=[0.2, 0.2])
        assert "with unique values only" in str(exc.value)

        # q > 1
        with pytest.raises(ValueError) as exc:
            _ = metrics.mae(series_a, series_b, q=[0.2, 1.01])
        assert "must be in the range `(>=0,<=1)`" in str(exc.value)

        # q < 0
        with pytest.raises(ValueError) as exc:
            _ = metrics.mae(series_a, series_b, q=[-0.01, 0.2])
        assert "must be in the range `(>=0,<=1)`" in str(exc.value)

        # but sorted, unique, and valid quantiles work
        _ = metrics.mae(series_a, series_b, q=[0.0, 0.5, 1.0])

    def test_quantile_as_tuple(self):
        """Test that `q` as tuple (list of quantiles, quantile component names) gives same results as `q`
        as quantile values list."""
        np.random.seed(42)
        q = [0.25, 0.75]

        series_a = TimeSeries.from_values(np.random.random((10, 2, 1)))
        q_names = pd.Index(
            likelihood_component_names(series_a.components, quantile_names(q))
        )
        series_b = TimeSeries.from_values(np.random.random((10, 4, 1)), columns=q_names)

        np.testing.assert_array_almost_equal(
            metrics.mae(series_a, series_b, q=(q, q_names)),
            metrics.mae(series_a, series_b, q=q),
        )

    def test_custom_metric_wrong_output_shape(self):
        """Test that custom metrics must have correct output dim."""

        @metrics.multi_ts_support
        @metrics.multivariate_support
        def custom_metric(
            actual_series,
            pred_series,
            intersect=True,
            *,
            q=None,
            time_reduction=None,
            component_reduction=np.nanmean,
            series_reduction=None,
            n_jobs=1,
            verbose=False,
            out_ndim=1,
        ):
            return np.ones(tuple(1 for _ in range(out_ndim)))

        for ndim in [1, 4]:
            with pytest.raises(ValueError) as exc:
                custom_metric(self.series1, self.series2, out_ndim=ndim)
            assert str(exc.value).startswith(
                "Metric output must have 2 dimensions (n components, n quantiles) for aggregated metrics"
            )
        for ndim in [2, 3]:
            _ = custom_metric(self.series1, self.series2, out_ndim=ndim)

    def test_wrong_error_scale(self):
        with pytest.raises(ValueError) as exc:
            _ = metrics._get_error_scale(
                self.series1.shift(-len(self.series1)),
                self.series1,
                m=1,
                metric="wrong_metric",
            )
        assert str(exc.value).startswith("unknown `metric=wrong_metric`")

    @pytest.mark.parametrize(
        "config",
        [
            # only time dependent quantile interval metrics
            (metrics.iw, metric_iw),
            (metrics.iws, metric_iws),
            (metrics.ic, metric_ic),
            (metrics.incs_qr, metric_incs_qr),
        ],
    )
    def test_metric_quantile_interval_accuracy(self, config):
        """Test output types and shapes for time dependent metrics with quantile intervals:
        for single and multiple univariate or multivariate series, in combination
        with different component and series reduction functions."""
        np.random.seed(42)
        metric, metric_ref = config
        n_comp = 2
        components = [str(i) for i in range(n_comp)]
        series_vals = np.random.random((10, n_comp, 1))
        pred_prob_vals = np.random.random((10, n_comp, 100))
        series = TimeSeries.from_values(series_vals, columns=components)
        pred_prob = TimeSeries.from_values(pred_prob_vals, columns=components)

        def check_ref(**test_kwargs):
            res_prob = metric(
                actual_series=series,
                pred_series=pred_prob,
                series_reduction=None,
                component_reduction=None,
                time_reduction=None,
                **test_kwargs,
            )
            res_ref = metric_ref(
                y_true=series.all_values(),
                y_pred=pred_prob.all_values(),
                **test_kwargs,
            )
            np.testing.assert_array_almost_equal(res_prob, res_ref)

        # one interval as tuple
        check_ref(q_interval=(0.1, 0.5))
        # one interval in list
        check_ref(q_interval=[(0.1, 0.5)])
        # multiple intervals
        check_ref(q_interval=[(0.1, 0.5), (0.5, 0.8)])

    @pytest.mark.parametrize(
        "config",
        list(
            itertools.product(
                [
                    # time dependent but with time reduction
                    metrics.iw,
                    metrics.miw,
                    metrics.iws,
                    metrics.miws,
                    metrics.ic,
                    metrics.mic,
                    metrics.incs_qr,
                    metrics.mincs_qr,
                ],
                [True, False],  # univariate series
                [True, False],  # single series
            )
        ),
    )
    def test_metric_quantile_interval(self, config):
        """Test output types and shapes for time aggregated metrics with quantile intervals:
        for single and multiple univariate or multivariate series, in combination
        with different component and series reduction functions."""
        np.random.seed(42)
        metric, is_univar, is_single = config
        params = inspect.signature(metric).parameters

        n_comp = 1 if is_univar else 2

        qs_all = [0.1, 0.5, 0.8]
        components = [str(i) for i in range(n_comp)]

        series_vals = np.random.random((10, n_comp, 1))
        pred_prob_vals = np.random.random((10, n_comp, 100))

        pred_vals_qs = []
        for i in range(n_comp):
            pred_vals_qs.append(
                np.quantile(pred_prob_vals[:, [i]], qs_all, axis=2).transpose(1, 0, 2)
            )
        pred_vals_qs = np.concatenate(pred_vals_qs, axis=1)
        pred_components = likelihood_component_names(
            components=components, parameter_names=quantile_names(q=qs_all)
        )

        series = TimeSeries.from_values(series_vals, columns=components)
        pred_prob = TimeSeries.from_values(pred_prob_vals, columns=components)
        pred_qs = TimeSeries.from_values(pred_vals_qs, columns=pred_components)
        shape_time = (len(pred_qs),) if "time_reduction" in params else tuple()

        if not is_single:
            series = [series] * 2
            pred_prob = [pred_prob] * 2
            pred_qs = [pred_qs] * 2

        kwargs = {"actual_series": series}

        def check_res(
            pred_prob_, pred_qs_, shape_exp, series_reduction=None, **test_kwargs
        ):
            res_prob = metric(
                actual_series=series,
                pred_series=pred_prob_,
                series_reduction=series_reduction,
                **test_kwargs,
            )
            res_qs = metric(
                actual_series=series,
                pred_series=pred_qs_,
                series_reduction=series_reduction,
                **test_kwargs,
            )
            if is_single or series_reduction is not None:
                res_prob = [res_prob]
                res_qs = [res_qs]
            if series_reduction is None and not is_single:
                assert len(res_prob) == len(res_qs) == len(pred_prob_)

            for res_p, res_q in zip(res_prob, res_qs):
                assert res_p.shape == res_q.shape == shape_exp
                np.testing.assert_array_almost_equal(res_p, res_q)
            return res_qs

        # one interval as tuple
        res = check_res(pred_prob, pred_qs, shape_time, q_interval=(0.1, 0.5))
        # one interval in list
        res2 = check_res(pred_prob, pred_qs, shape_time, q_interval=[(0.1, 0.5)])
        np.testing.assert_array_almost_equal(res, res2)
        # multiple intervals
        check_res(
            pred_prob, pred_qs, shape_time + (2,), q_interval=[(0.1, 0.5), (0.5, 0.8)]
        )
        # all intervals
        check_res(
            pred_prob,
            pred_qs,
            shape_time + (3,),
            q_interval=[(0.1, 0.5), (0.5, 0.8), (0.1, 0.8)],
        )
        q_intervals = [(0.1, 0.5), (0.5, 0.8)]
        # component and series reduction
        check_res(
            pred_prob,
            pred_qs,
            shape_time + (len(q_intervals),),
            q_interval=q_intervals,
            component_reduction=np.mean,
            series_reduction=np.mean,
        )
        # no component reduction
        check_res(
            pred_prob,
            pred_qs,
            shape_time + (len(q_intervals) * n_comp,),
            q_interval=q_intervals,
            component_reduction=None,
            series_reduction=np.mean,
        )
        # no series reduction
        check_res(
            pred_prob,
            pred_qs,
            shape_time + (len(q_intervals),),
            q_interval=q_intervals,
            component_reduction=np.mean,
            series_reduction=None,
        )
        # no series and component reduction
        check_res(
            pred_prob,
            pred_qs,
            shape_time + (len(q_intervals) * n_comp,),
            q_interval=q_intervals,
            component_reduction=None,
            series_reduction=None,
        )

        # check that we get identical results as when computing intervals separately (on the time aggregated case)
        if "time_reduction" in params:
            kwargs["time_reduction"] = np.mean
        res_lo = metric(
            pred_series=pred_qs,
            component_reduction=None,
            q_interval=(0.1, 0.5),
            **kwargs,
        )
        res_hi = metric(
            pred_series=pred_qs,
            component_reduction=None,
            q_interval=(0.5, 0.8),
            **kwargs,
        )
        res_multi = metric(
            pred_series=pred_qs,
            component_reduction=None,
            q_interval=[(0.1, 0.5), (0.5, 0.8)],
            **kwargs,
        )
        if is_single:
            res_lo = [res_lo]
            res_hi = [res_hi]
            res_multi = [res_multi]
        res_lo_hi = []
        for res_lo_, res_hi_ in zip(res_lo, res_hi):
            if res_lo_.ndim == 0:
                res_lo_ = np.expand_dims(res_lo_, -1)
                res_hi_ = np.expand_dims(res_hi_, -1)
                res_lo_hi_ = np.concatenate([res_lo_, res_hi_])
            else:
                res_lo_hi_ = np.concatenate(
                    [(res_lo_[i], res_hi_[i]) for i in range(n_comp)],
                )
            res_lo_hi.append(res_lo_hi_)
        np.testing.assert_array_almost_equal(res_lo_hi, res_multi)

    def test_invalid_quantile_intervals(self):
        np.random.seed(42)
        series_a = TimeSeries.from_values(np.random.random((10, 2, 1)))
        series_b = TimeSeries.from_values(np.random.random((10, 2, 10)))

        # q not supported
        with pytest.raises(ValueError) as exc:
            _ = metrics.iw(series_a, series_b, q=[0.2])
        assert str(exc.value).startswith(
            "`q` is not supported for quantile interval metrics"
        )

        # no quantile interval
        with pytest.raises(ValueError) as exc:
            _ = metrics.iw(series_a, series_b, q_interval=None)
        assert str(exc.value).startswith(
            "Quantile interval metrics require setting `q_interval`."
        )

        # invalid interval type
        with pytest.raises(ValueError) as exc:
            _ = metrics.iw(series_a, series_b, q_interval=0.6)
        assert (
            str(exc.value)
            == "`q_interval` must be a tuple (float, float) or a sequence of tuples (float, float)."
        )

        # invalid tuple length
        with pytest.raises(ValueError) as exc:
            _ = metrics.iw(series_a, series_b, q_interval=(0.1, 0.2, 0.3))
        assert (
            str(exc.value)
            == "`q_interval` must be a tuple (float, float) or a sequence of tuples (float, float)."
        )

        # one tuple has invalid length invalid tuple length (raises a numpy error)
        with pytest.raises(ValueError):
            _ = metrics.iw(series_a, series_b, q_interval=[(0.1, 0.2), (0.2, 0.3, 0.4)])

        # interval upper bound too high
        with pytest.raises(ValueError) as exc:
            _ = metrics.iw(series_a, series_b, q_interval=(0.1, 1.1))
        assert str(exc.value).startswith(
            "All `q` values must be in the range `(>=0,<=1)`."
        )

        # interval lower bound too low
        with pytest.raises(ValueError) as exc:
            _ = metrics.iw(series_a, series_b, q_interval=(-0.01, 0.1))
        assert str(exc.value).startswith(
            "All `q` values must be in the range `(>=0,<=1)`."
        )

        # lower interval equal to higher interval
        with pytest.raises(ValueError) as exc:
            _ = metrics.iw(series_a, series_b, q_interval=(0.2, 0.2))
        assert str(exc.value).startswith(
            "all intervals in `q_interval` must be tuples of (lower q, upper q)"
        )

        # lower interval higher than higher interval
        with pytest.raises(ValueError) as exc:
            _ = metrics.iw(series_a, series_b, q_interval=(0.3, 0.2))
        assert str(exc.value).startswith(
            "all intervals in `q_interval` must be tuples of (lower q, upper q)"
        )
