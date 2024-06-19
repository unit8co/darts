import itertools

import numpy as np
import pytest

import darts.metrics as metrics
from darts.datasets import AirPassengersDataset
from darts.logging import get_logger
from darts.models import LinearRegressionModel, NaiveDrift, NaiveSeasonal
from darts.tests.models.forecasting.test_regression_models import dummy_timeseries
from darts.utils.timeseries_generation import constant_timeseries as ct
from darts.utils.timeseries_generation import linear_timeseries as lt

logger = get_logger(__name__)


class TestResiduals:
    np.random.seed(42)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],
            [False, True],
            [(metrics.err, (-1.0, -2.0)), (metrics.ape, (100.0, 100.0))],
        ),
    )
    def test_output_single_series_hfc_lpo_true(self, config):
        """Tests backtest based on historical forecasts generated on a single `series` (or list of one `series`)
        with last_points_only=True"""
        is_univariate, series_as_list, (metric, score_exp) = config
        n_ts = 10
        y = ct(value=1.0, length=n_ts)
        hfc = ct(value=2.0, length=n_ts)
        if not is_univariate:
            y = y.stack(y + 1.0)
            hfc = hfc.stack(hfc + 2.0)
        n_comps = y.n_components
        y = y if not series_as_list else [y]
        hfc = hfc if not series_as_list else [hfc]

        # expected residuals values of shape (n time steps, n components, n samples=1)
        score_exp = np.array([score_exp[:n_comps]] * 10).reshape(n_ts, -1, 1)
        model = NaiveDrift()

        # check that input does not work with `last_points_only=False``
        with pytest.raises(ValueError) as err:
            _ = model.residuals(
                series=y,
                historical_forecasts=hfc,
                metric=metric,
                last_points_only=False,
            )
        if series_as_list:
            error_msg = "Expected `historical_forecasts` of type `Sequence[Sequence[TimeSeries]]`"
        else:
            error_msg = "Expected `historical_forecasts` of type `Sequence[TimeSeries]`"
        assert str(err.value).startswith(error_msg)

        for vals_only in [False, True]:
            res = model.residuals(
                series=y,
                historical_forecasts=hfc,
                metric=metric,
                last_points_only=True,
                values_only=vals_only,
            )
            res = res if series_as_list else [res]
            assert isinstance(res, list) and len(res) == 1
            res = res[0]
            vals = res if vals_only else res.all_values()
            np.testing.assert_array_almost_equal(vals, score_exp)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],
            [False, True],
            [
                (metrics.err, ((0.0, 0.0), (-1.0, -2.0))),
                (metrics.ape, ((0.0, 0.0), (100.0, 100.0))),
            ],
            [1, 2],
        ),
    )
    def test_output_single_series_hfc_lpo_false(self, config):
        """Tests residuals based on historical forecasts generated on a single `series` (or list of one `series`)
        with last_points_only=False"""
        is_univariate, series_as_list, (metric, score_exp), n_forecasts = config
        n_ts = 10
        y = ct(value=1.0, length=n_ts)
        hfc = ct(value=2.0, length=n_ts)
        if not is_univariate:
            y = y.stack(y + 1.0)
            hfc = hfc.stack(hfc + 2.0)
        n_comps = y.n_components

        hfc = [y, hfc]
        hfc = hfc[:n_forecasts]
        y = y if not series_as_list else [y]
        hfc = hfc if not series_as_list else [hfc]

        # expected residuals values of shape (n time steps, n components, n samples=1) per forecast
        scores_exp = []
        for i in range(n_forecasts):
            scores_exp.append(
                np.array([score_exp[i][:n_comps]] * 10).reshape(n_ts, -1, 1)
            )

        model = NaiveDrift()

        # check that input does not work with `last_points_only=True``
        with pytest.raises(ValueError) as err:
            _ = model.residuals(
                series=y,
                historical_forecasts=hfc,
                metric=metric,
                last_points_only=True,
            )
        if series_as_list:
            error_msg = "Expected `historical_forecasts` of type `Sequence[TimeSeries]`"
        else:
            error_msg = "Expected `historical_forecasts` of type `TimeSeries`"
        assert str(err.value).startswith(error_msg)

        for vals_only in [False, True]:
            res = model.residuals(
                series=y,
                historical_forecasts=hfc,
                metric=metric,
                last_points_only=False,
                values_only=vals_only,
            )
            res = res if series_as_list else [res]
            assert isinstance(res, list) and len(res) == 1
            res = res[0]
            assert isinstance(res, list) and len(res) == n_forecasts
            for res_, score_exp_ in zip(res, scores_exp):
                vals = res_ if vals_only else res_.all_values()
                np.testing.assert_array_almost_equal(vals, score_exp_)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],
            [
                (metrics.err, ((0.0, 0.0), (-1.0, -2.0))),
                (metrics.ape, ((0.0, 0.0), (100.0, 100.0))),
            ],
        ),
    )
    def test_output_multi_series_hfc_lpo_true(self, config):
        """Tests residuals based on historical forecasts generated on multiple `series` with last_points_only=True"""
        is_univariate, (metric, score_exp) = config
        n_ts = 10
        y = ct(value=1.0, length=n_ts)
        hfc = ct(value=2.0, length=n_ts)
        if not is_univariate:
            y = y.stack(y + 1.0)
            hfc = hfc.stack(hfc + 2.0)
        n_comps = y.n_components
        hfc = [y, hfc]
        y = [y, y]

        # expected residuals values of shape (n time steps, n components, n samples=1) per forecast
        scores_exp = []
        for i in range(len(hfc)):
            scores_exp.append(
                np.array([score_exp[i][:n_comps]] * 10).reshape(n_ts, -1, 1)
            )

        model = NaiveDrift()

        # check that input does not work with `last_points_only=False``
        with pytest.raises(ValueError) as err:
            _ = model.residuals(
                series=y,
                historical_forecasts=hfc,
                metric=metric,
                last_points_only=False,
            )
        error_msg = (
            "Expected `historical_forecasts` of type `Sequence[Sequence[TimeSeries]]`"
        )
        assert str(err.value).startswith(error_msg)

        for vals_only in [False, True]:
            res = model.residuals(
                series=y,
                historical_forecasts=hfc,
                metric=metric,
                last_points_only=True,
                values_only=vals_only,
            )
            assert isinstance(res, list) and len(res) == len(y)
            for res_, score_exp_ in zip(res, scores_exp):
                vals = res_ if vals_only else res_.all_values()
                np.testing.assert_array_almost_equal(vals, score_exp_)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],
            [
                (metrics.err, ((0.0, 0.0), (-1.0, -2.0))),
                (metrics.ape, ((0.0, 0.0), (100.0, 100.0))),
            ],
        ),
    )
    def test_output_multi_series_hfc_lpo_false(self, config):
        """Tests residuals based on historical forecasts generated on multiple `series` with
        last_points_only=False.
        """
        is_univariate, (metric, score_exp) = config
        n_ts = 10
        y = ct(value=1.0, length=n_ts)
        hfc = ct(value=2.0, length=n_ts)
        if not is_univariate:
            y = y.stack(y + 1.0)
            hfc = hfc.stack(hfc + 2.0)
        n_comps = y.n_components
        hfc = [[y], [hfc]]
        y = [y, y]

        # expected residuals values of shape (n time steps, n components, n samples=1) per forecast
        scores_exp = []
        for i in range(len(hfc)):
            scores_exp.append(
                np.array([score_exp[i][:n_comps]] * 10).reshape(n_ts, -1, 1)
            )

        model = NaiveDrift()

        # check that input does not work with `last_points_only=False``
        with pytest.raises(ValueError) as err:
            _ = model.residuals(
                series=y,
                historical_forecasts=hfc,
                metric=metric,
                last_points_only=True,
            )
        error_msg = "Expected `historical_forecasts` of type `Sequence[TimeSeries]`"
        assert str(err.value).startswith(error_msg)

        for vals_only in [False, True]:
            res = model.residuals(
                series=y,
                historical_forecasts=hfc,
                metric=metric,
                last_points_only=False,
                values_only=vals_only,
            )
            assert isinstance(res, list) and len(res) == len(y)
            for res_list, score_exp_ in zip(res, scores_exp):
                assert isinstance(res_list, list) and len(res_list) == 1
                res_ = res_list[0]
                vals = res_ if vals_only else res_.all_values()
                np.testing.assert_array_almost_equal(vals, score_exp_)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],
            [
                (metrics.err, ((0.0, 0.0), (-1.0, -2.0))),
                (metrics.ape, ((0.0, 0.0), (100.0, 100.0))),
            ],
        ),
    )
    def test_output_multi_series_hfc_lpo_false_different_n_fcs(self, config):
        """Tests residuals based on historical forecasts generated on multiple `series` with
        last_points_only=False, and the historical forecasts have different lengths
        """
        is_univariate, (metric, score_exp) = config
        n_ts = 10
        y = ct(value=1.0, length=n_ts)
        hfc = ct(value=2.0, length=n_ts)
        if not is_univariate:
            y = y.stack(y + 1.0)
            hfc = hfc.stack(hfc + 2.0)
        n_comps = y.n_components
        hfc = [[y], [hfc, hfc]]
        y = [y, y]

        # expected residuals values of shape (n time steps, n components, n samples=1) per forecast
        scores_exp = []
        for i in range(len(hfc)):
            scores_exp.append(
                np.array([score_exp[i][:n_comps]] * 10).reshape(n_ts, -1, 1)
            )
        # repeat following `hfc`
        scores_exp = [[scores_exp[0]], [scores_exp[1]] * 2]

        model = NaiveDrift()

        # check that input does not work with `last_points_only=False``
        with pytest.raises(ValueError) as err:
            _ = model.residuals(
                series=y,
                historical_forecasts=hfc,
                metric=metric,
                last_points_only=True,
            )
        error_msg = "Expected `historical_forecasts` of type `Sequence[TimeSeries]`"
        assert str(err.value).startswith(error_msg)

        for vals_only in [False, True]:
            res = model.residuals(
                series=y,
                historical_forecasts=hfc,
                metric=metric,
                last_points_only=False,
                values_only=vals_only,
            )
            assert isinstance(res, list) and len(res) == len(y)
            for res_list, hfc_list, score_exp_list in zip(res, hfc, scores_exp):
                assert isinstance(res_list, list) and len(res_list) == len(hfc_list)
                for res_, score_exp_ in zip(res_list, score_exp_list):
                    vals = res_ if vals_only else res_.all_values()
                    np.testing.assert_array_almost_equal(vals, score_exp_)

    def test_wrong_metric(self):
        y = ct(value=1.0, length=10)
        hfc = ct(value=2.0, length=10)

        model = NaiveDrift()

        with pytest.raises(TypeError) as err:
            _ = model.residuals(
                series=y,
                historical_forecasts=hfc,
                metric=metrics.mape,
                last_points_only=True,
            )
        assert str(err.value).endswith(
            "got an unexpected keyword argument 'time_reduction'"
        )

    def test_forecasting_residuals_nocov_output(self):
        model = NaiveSeasonal(K=1)

        # test zero residuals
        constant_ts = ct(length=20)
        residuals = model.residuals(constant_ts)
        np.testing.assert_almost_equal(
            residuals.univariate_values(), np.zeros(len(residuals))
        )
        residuals_vals = model.residuals(constant_ts, values_only=True)
        np.testing.assert_almost_equal(residuals.all_values(), residuals_vals)

        # test constant, positive residuals
        linear_ts = lt(length=20)
        residuals = model.residuals(linear_ts)
        np.testing.assert_almost_equal(
            np.diff(residuals.univariate_values()), np.zeros(len(residuals) - 1)
        )
        np.testing.assert_array_less(
            np.zeros(len(residuals)), residuals.univariate_values()
        )
        residuals_vals = model.residuals(linear_ts, values_only=True)
        np.testing.assert_almost_equal(residuals.all_values(), residuals_vals)

    def test_forecasting_residuals_multiple_series(self):
        # test input types past and/or future covariates

        # dummy covariates and target TimeSeries instances
        series, past_covariates, future_covariates = dummy_timeseries(
            length=10,
            n_series=1,
            comps_target=1,
            comps_pcov=1,
            comps_fcov=1,
        )  # outputs Sequences[TimeSeries] and not TimeSeries

        model = LinearRegressionModel(
            lags=1, lags_past_covariates=1, lags_future_covariates=(1, 1)
        )
        model.fit(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # residuals TimeSeries zero
        res = model.residuals(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        assert isinstance(res, list) and len(res) == len(series) == 1
        res_vals = res[0].all_values(copy=False)
        np.testing.assert_almost_equal(res_vals, np.zeros((len(res[0]), 1, 1)))

        # return values only
        res_vals_direct = model.residuals(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            values_only=True,
        )
        assert (
            isinstance(res_vals_direct, list)
            and len(res_vals_direct) == len(series) == 1
        )
        np.testing.assert_almost_equal(res_vals_direct[0], res_vals)

        # with precomputed historical forecasts
        hfc = model.historical_forecasts(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        res_hfc = model.residuals(series, historical_forecasts=hfc)
        assert res == res_hfc

        # with pretrained model
        res_pretrained = model.residuals(
            series,
            start=model.min_train_series_length,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            retrain=False,
            values_only=True,
        )
        np.testing.assert_almost_equal(res_pretrained[0], res_vals)

        # if model is trained with covariates, should raise error when covariates are missing in residuals()
        with pytest.raises(ValueError):
            model.residuals(series)

        with pytest.raises(ValueError):
            model.residuals(series, past_covariates=past_covariates)

        with pytest.raises(ValueError):
            model.residuals(series, future_covariates=future_covariates)

    @pytest.mark.parametrize(
        "series",
        [
            ct(value=0.5, length=10),
            lt(length=10),
        ],
    )
    def test_forecasting_residuals_cov_output(self, series):
        # if covariates are constant and the target is constant/linear,
        # residuals should be zero (for a LinearRegression model)
        past_covariates = ct(value=0.2, length=10)
        future_covariates = ct(value=0.1, length=10)

        model = LinearRegressionModel(
            lags=1, lags_past_covariates=1, lags_future_covariates=(1, 1)
        )
        model.fit(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # residuals TimeSeries zero
        res = model.residuals(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        np.testing.assert_almost_equal(res.univariate_values(), np.zeros(len(res)))

        # return values only
        res_vals = model.residuals(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            values_only=True,
        )
        np.testing.assert_almost_equal(res.all_values(), res_vals)

        # with precomputed historical forecasts
        hfc = model.historical_forecasts(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        res_hfc = model.residuals(series, historical_forecasts=hfc)
        assert res == res_hfc

        # with pretrained model
        res_pretrained = model.residuals(
            series,
            start=model.min_train_series_length,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            retrain=False,
            values_only=True,
        )
        np.testing.assert_almost_equal(res_vals, res_pretrained)

        # if model is trained with covariates, should raise error when covariates are missing in residuals()
        with pytest.raises(ValueError):
            model.residuals(series)

        with pytest.raises(ValueError):
            model.residuals(series, past_covariates=past_covariates)

        with pytest.raises(ValueError):
            model.residuals(series, future_covariates=future_covariates)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [
                metrics.ase,
                metrics.sse,
            ],
            [1, 2],
        ),
    )
    def test_scaled_metrics(self, config):
        """Tests residuals for scaled metrics based on historical forecasts generated on a sequence
        `series` with last_points_only=False"""
        metric, m = config
        y = lt(length=20)
        hfc = lt(length=10, start=y.start_time() + 10 * y.freq)
        y = [y, y]
        hfc = [[hfc, hfc], [hfc]]

        model = NaiveDrift()
        bts = model.residuals(
            series=y,
            historical_forecasts=hfc,
            metric=metric,
            last_points_only=False,
            metric_kwargs={"m": m},
            values_only=True,
        )
        assert isinstance(bts, list) and len(bts) == 2

        bt_expected = metric(y[0], hfc[0][0], insample=y[0], m=m)
        bt_expected = np.reshape(bt_expected, (len(hfc[0][0]), y[0].n_components, 1))
        for bt_list in bts:
            for bt in bt_list:
                np.testing.assert_array_almost_equal(bt, bt_expected)

    def test_metric_kwargs(self):
        """Tests residuals with different metric_kwargs based on historical forecasts generated on a sequence
        `series` with last_points_only=False"""
        y = lt(length=20)
        y = y.stack(y + 1.0)
        hfc = lt(length=10, start=y.start_time() + 10 * y.freq)
        hfc = hfc.stack(hfc + 1.0)
        y = [y, y]
        hfc = [[hfc, hfc], [hfc]]

        model = NaiveDrift()
        # reduction `metric_kwargs` are bypassed, n_jobs not
        bts = model.residuals(
            series=y,
            historical_forecasts=hfc,
            metric=metrics.ae,
            last_points_only=False,
            metric_kwargs={
                "component_reduction": np.median,
                "time_reduction": np.mean,
                "n_jobs": -1,
            },
            values_only=True,
        )
        assert isinstance(bts, list) and len(bts) == 2

        # `ae` with time and component reduction is equal to `mae` with component reduction
        bt_expected = metrics.ae(
            y[0],
            hfc[0][0],
            series_reduction=None,
            time_reduction=None,
            component_reduction=None,
        )[:, :, None]
        for bt_list in bts:
            for bt in bt_list:
                np.testing.assert_array_almost_equal(bt, bt_expected)

    @pytest.mark.parametrize(
        "config",
        itertools.product([True, False], [True, False]),
    )
    def test_sample_weight(self, config):
        """check that passing sample weights work and that it yields different results than without sample weights."""
        manual_weight, multi_series = config
        ts = AirPassengersDataset().load()
        if manual_weight:
            sample_weight = np.linspace(0, 1, len(ts))
            sample_weight = ts.with_values(np.expand_dims(sample_weight, -1))
        else:
            sample_weight = "linear"

        if multi_series:
            ts = [ts] * 2
            sample_weight = [sample_weight] * 2 if manual_weight else sample_weight

        model = LinearRegressionModel(lags=3, output_chunk_length=1)
        start_kwargs = {"start": -1, "start_format": "position"}
        res_non_weighted = model.residuals(series=ts, values_only=True, **start_kwargs)

        model = LinearRegressionModel(lags=3, output_chunk_length=1)
        res_weighted = model.residuals(
            series=ts, sample_weight=sample_weight, values_only=True, **start_kwargs
        )

        if not multi_series:
            res_weighted = [res_weighted]
            res_non_weighted = [res_non_weighted]

        # check that the predictions are different
        for res_nw, res_w in zip(res_non_weighted, res_weighted):
            with pytest.raises(AssertionError):
                np.testing.assert_array_almost_equal(res_w, res_nw)
