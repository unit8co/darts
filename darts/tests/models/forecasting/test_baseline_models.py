import itertools

import numpy as np
import pytest

from darts import TimeSeries
from darts.logging import get_logger
from darts.models import NaiveDrift, NaiveMean, NaiveMovingAverage, NaiveSeasonal
from darts.models.forecasting.forecasting_model import (
    GlobalForecastingModel,
    LocalForecastingModel,
)
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)


icl = 5
local_models = [
    (NaiveDrift, {}),
    (NaiveMean, {}),
    (NaiveMovingAverage, {}),
    (NaiveSeasonal, {}),
]
global_models = []


if TORCH_AVAILABLE:
    import torch

    from darts.models import GlobalNaiveAggregate, GlobalNaiveDrift, GlobalNaiveSeasonal

    global_models += [
        (
            GlobalNaiveAggregate,
            {"input_chunk_length": icl, "output_chunk_length": 3, **tfm_kwargs},
        ),
        (
            GlobalNaiveAggregate,
            {"input_chunk_length": icl, "output_chunk_length": 1, **tfm_kwargs},
        ),
        (
            GlobalNaiveDrift,
            {"input_chunk_length": icl, "output_chunk_length": 3, **tfm_kwargs},
        ),
        (
            GlobalNaiveDrift,
            {"input_chunk_length": icl, "output_chunk_length": 1, **tfm_kwargs},
        ),
        (
            GlobalNaiveSeasonal,
            {"input_chunk_length": icl, "output_chunk_length": 3, **tfm_kwargs},
        ),
        (
            GlobalNaiveSeasonal,
            {"input_chunk_length": icl, "output_chunk_length": 1, **tfm_kwargs},
        ),
    ]

    def custom_mean_valid(x, dim):
        return torch.mean(x, dim)

    def custom_mean_invalid_out_shape(x, dim):
        return x[:1]

    def custom_mean_invalid_signature(x):
        return torch.mean(x, dim=1)

    def custom_mean_invalid_output_type(x, dim):
        return torch.mean(x, dim=1).detach().numpy()

else:
    custom_mean_valid = None
    custom_mean_invalid_out_shape = None
    custom_mean_invalid_signature = None
    custom_mean_invalid_output_type = None


class TestBaselineModels:
    np.random.seed(42)
    if TORCH_AVAILABLE:
        torch.manual_seed(42)

    @pytest.mark.parametrize(
        "config", itertools.product(local_models + global_models, [False, True])
    )
    def test_fit_predict(self, config):
        """Tests fit and predict for univariate and multivariate time series."""
        (model_cls, model_kwargs), is_multivariate = config

        # min train series length for global naive models
        series = tg.linear_timeseries(length=icl)
        if is_multivariate:
            series.stack(series + 100)

        model = model_cls(**model_kwargs)
        assert not model.supports_probabilistic_prediction
        assert not model.supports_likelihood_parameter_prediction

        # calling predict before fit
        with pytest.raises(ValueError):
            model.predict(n=10)

        # calling fit with covariates
        if isinstance(model, GlobalForecastingModel):
            err_type = ValueError
            err_msg_content = "The model does not support"
        else:  # for local models, covariates are not part of signature
            err_type = TypeError
            err_msg_content = "got an unexpected keyword argument"
        with pytest.raises(err_type) as err:
            model.fit(series=series, past_covariates=series)
        assert err_msg_content in str(err.value)
        with pytest.raises(err_type) as err:
            model.fit(series=series, future_covariates=series)
        assert err_msg_content in str(err.value)

        model.fit(series=series)
        # calling predict with covariates
        with pytest.raises(err_type) as err:
            model.predict(n=10, past_covariates=series)
        assert err_msg_content in str(err.value)
        with pytest.raises(err_type) as err:
            model.predict(n=10, future_covariates=series)
        assert err_msg_content in str(err.value)

        # single series predict works with all models
        preds = model.predict(n=10)
        preds_start = series.end_time() + series.freq
        assert isinstance(preds, TimeSeries)
        assert len(preds) == 10
        assert preds.start_time() == preds_start
        assert preds.components.equals(series.components)

        if isinstance(model, LocalForecastingModel):
            # no series at prediction time
            with pytest.raises(err_type) as err:
                _ = model.predict(n=10, series=series)
            assert err_msg_content in str(err.value)
            # no multiple series prediction
            with pytest.raises(err_type) as err:
                _ = model.predict(n=10, series=[series, series])
            assert err_msg_content in str(err.value)
        else:
            preds = model.predict(n=10, series=series)
            assert isinstance(preds, TimeSeries)
            assert len(preds) == 10
            assert preds.start_time() == preds_start
            assert preds.components.equals(series.components)
            preds = model.predict(n=10, series=[series, series])
            assert isinstance(preds, list)
            assert len(preds) == 2
            assert all([isinstance(p, TimeSeries) for p in preds])
            assert all([len(p) == 10 for p in preds])
            assert all([p.start_time() == preds_start for p in preds])
            assert all([p.components.equals(series.components) for p in preds])

        # multiple series training only with global baselines
        if isinstance(model, LocalForecastingModel):
            with pytest.raises(ValueError) as err:
                model.fit(series=[series, series])
            assert "Train `series` must be a single `TimeSeries`." == str(err.value)
        else:
            model.fit(series=[series, series])

    def test_naive_seasonal(self):
        # min train series length for global naive models
        series = tg.linear_timeseries(length=icl)
        series = series.stack(series + 25.0)

        vals_exp = series.values(copy=False)

        # local naive seasonal
        local_model = NaiveSeasonal(K=icl)
        preds = local_model.fit(series).predict(n=icl)
        np.testing.assert_array_almost_equal(preds.values(copy=False), vals_exp)

        if not TORCH_AVAILABLE:
            return

        # equivalent global naive seasonal
        global_model = GlobalNaiveSeasonal(
            input_chunk_length=icl, output_chunk_length=1, **tfm_kwargs
        )
        preds = global_model.fit(series).predict(n=icl)
        np.testing.assert_array_almost_equal(preds.values(copy=False), vals_exp)

        preds_multi = global_model.predict(n=icl, series=[series, series + 100.0])
        np.testing.assert_array_almost_equal(
            preds_multi[0].values(copy=False), vals_exp
        )
        np.testing.assert_array_almost_equal(
            preds_multi[1].values(copy=False), vals_exp + 100.0
        )

        # global naive seasonal that repeats values `output_chunk_length` times
        global_model = GlobalNaiveSeasonal(
            input_chunk_length=icl, output_chunk_length=icl, **tfm_kwargs
        )
        preds = global_model.fit(series).predict(n=icl)
        np.testing.assert_array_almost_equal(
            preds.values(copy=False), np.repeat(vals_exp[0:1, :], icl, axis=0)
        )

        preds_multi = global_model.predict(n=icl, series=[series, series + 100.0])
        np.testing.assert_array_almost_equal(
            preds_multi[0].values(copy=False), np.repeat(vals_exp[0:1, :], icl, axis=0)
        )
        np.testing.assert_array_almost_equal(
            preds_multi[1].values(copy=False),
            np.repeat(vals_exp[0:1, :] + 100.0, icl, axis=0),
        )

    def test_naive_drift(self):
        # min train series length for global naive models
        series_total = tg.linear_timeseries(length=2 * icl)
        series_total = series_total.stack(series_total + 25.0)
        series = series_total[:icl]
        series_drift = series_total[icl:]

        vals_exp = series_drift.values(copy=False)

        # local naive drift
        local_model = NaiveDrift()
        preds = local_model.fit(series).predict(n=icl)
        np.testing.assert_array_almost_equal(preds.values(copy=False), vals_exp)

        if not TORCH_AVAILABLE:
            return

        # identical global naive drift
        global_model = GlobalNaiveDrift(
            input_chunk_length=icl, output_chunk_length=icl, **tfm_kwargs
        )
        preds = global_model.fit(series).predict(n=icl)
        np.testing.assert_array_almost_equal(preds.values(copy=False), vals_exp)

        preds_multi = global_model.predict(n=icl, series=[series, series + 100.0])
        np.testing.assert_array_almost_equal(
            preds_multi[0].values(copy=False), vals_exp
        )
        np.testing.assert_array_almost_equal(
            preds_multi[1].values(copy=False), vals_exp + 100.0
        )

        # global naive moving drift
        global_model = GlobalNaiveDrift(
            input_chunk_length=icl, output_chunk_length=1, **tfm_kwargs
        )
        preds = global_model.fit(series).predict(n=icl)

        # manually compute the moving/autoregressive drift
        series_vals = series.values(copy=False)
        preds_vals = preds.values(copy=False)
        preds_exp = []
        x, y = 1, None
        for i in range(0, icl):
            y_0 = y if y is not None else series_vals[-1]
            m = (y_0 - series_vals[i]) / (icl - 1)
            y = m * x + y_0
            preds_exp.append(np.expand_dims(y, 0))
        preds_exp = np.concatenate(preds_exp)
        np.testing.assert_array_almost_equal(preds_vals, preds_exp)

        preds_multi = global_model.predict(n=icl, series=[series, series + 100.0])
        np.testing.assert_array_almost_equal(
            preds_multi[0].values(copy=False), preds_exp
        )
        np.testing.assert_array_almost_equal(
            preds_multi[1].values(copy=False), preds_exp + 100.0
        )

    def test_naive_mean(self):
        # min train series length for global naive models
        series = tg.linear_timeseries(length=icl)
        series = series.stack(series + 25.0)

        # mean repeated n times
        vals_exp = np.repeat(
            np.expand_dims(series.values(copy=False).mean(axis=0), 0), icl, axis=0
        )

        # local naive mean
        local_model = NaiveMean()
        preds = local_model.fit(series).predict(n=icl)
        np.testing.assert_array_almost_equal(preds.values(copy=False), vals_exp)

        if not TORCH_AVAILABLE:
            return

        # identical global naive mean
        global_model = GlobalNaiveAggregate(
            input_chunk_length=icl, output_chunk_length=icl, agg_fn="mean", **tfm_kwargs
        )
        preds = global_model.fit(series).predict(n=icl)
        np.testing.assert_array_almost_equal(preds.values(copy=False), vals_exp)

        preds_multi = global_model.predict(n=icl, series=[series, series + 100.0])
        np.testing.assert_array_almost_equal(
            preds_multi[0].values(copy=False), vals_exp
        )
        np.testing.assert_array_almost_equal(
            preds_multi[1].values(copy=False), vals_exp + 100.0
        )

    def test_naive_moving_average(self):
        # min train series length for global naive models
        series = tg.linear_timeseries(length=icl)
        series = series.stack(series + 25.0)

        # manually compute the moving/autoregressive average/mean
        series_vals = series.values(copy=False)
        vals_exp = []
        y = None
        for i in range(0, icl):
            if y is None:
                y_moving = series_vals
            else:
                y_moving = np.concatenate(
                    [series_vals[i:], np.concatenate(vals_exp)], axis=0
                )
            y = np.expand_dims(y_moving.mean(axis=0), 0)
            vals_exp.append(y)
        vals_exp = np.concatenate(vals_exp)

        # local naive mean
        local_model = NaiveMovingAverage(input_chunk_length=icl)
        preds = local_model.fit(series).predict(n=icl)
        np.testing.assert_array_almost_equal(preds.values(copy=False), vals_exp)

        if not TORCH_AVAILABLE:
            return

        # identical global naive moving average
        global_model = GlobalNaiveAggregate(
            input_chunk_length=icl, output_chunk_length=1, agg_fn="mean", **tfm_kwargs
        )
        preds = global_model.fit(series).predict(n=icl)
        np.testing.assert_array_almost_equal(preds.values(copy=False), vals_exp)

        preds_multi = global_model.predict(n=icl, series=[series, series + 100.0])
        np.testing.assert_array_almost_equal(
            preds_multi[0].values(copy=False), vals_exp
        )
        np.testing.assert_array_almost_equal(
            preds_multi[1].values(copy=False), vals_exp + 100.0
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    @pytest.mark.parametrize(
        "agg_fn_config",
        [
            ("nanmean", "nanmean"),
            ("mean", "mean"),
            (custom_mean_valid, "mean"),
        ],
    )
    def test_global_naive_aggregate(self, agg_fn_config):
        agg_fn, agg_name = agg_fn_config

        # min train series length for global naive models
        series = tg.linear_timeseries(length=icl)
        series = series.stack(series + 25.0)

        # manually compute the moving/autoregressive average/mean
        series_vals = series.values(copy=False)
        vals_exp = []

        agg_fn_np = getattr(np, agg_name)
        y = None
        for i in range(0, icl):
            if y is None:
                y_moving = series_vals
            else:
                y_moving = np.concatenate(
                    [series_vals[i:], np.concatenate(vals_exp)], axis=0
                )

            y = np.expand_dims(agg_fn_np(y_moving, axis=0), 0)
            vals_exp.append(y)
        vals_exp = np.concatenate(vals_exp)

        # identical global naive moving average
        global_model = GlobalNaiveAggregate(
            input_chunk_length=icl, output_chunk_length=1, agg_fn=agg_fn, **tfm_kwargs
        )
        preds = global_model.fit(series).predict(n=icl)
        np.testing.assert_array_almost_equal(preds.values(copy=False), vals_exp)

        preds_multi = global_model.predict(n=icl, series=[series, series + 100.0])
        np.testing.assert_array_almost_equal(
            preds_multi[0].values(copy=False), vals_exp
        )
        np.testing.assert_array_almost_equal(
            preds_multi[1].values(copy=False), vals_exp + 100.0
        )

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires torch")
    @pytest.mark.parametrize(
        "agg_fn_config",
        [
            ("mmean", "When `agg_fn` is a string"),
            (1, "`agg_fn` must be a string or callable"),
            (
                custom_mean_invalid_output_type,
                "`agg_fn` output must be a torch Tensor.",
            ),
            (custom_mean_invalid_signature, "got an unexpected keyword argument 'dim'"),
            (custom_mean_invalid_out_shape, "Unexpected `agg_fn` output shape."),
        ],
    )
    def test_global_naive_aggregate_invalid_agg_fn(self, agg_fn_config):
        agg_fn, err_msg_content = agg_fn_config
        # identical global naive moving average
        with pytest.raises(ValueError) as err:
            _ = GlobalNaiveAggregate(
                input_chunk_length=icl,
                output_chunk_length=1,
                agg_fn=agg_fn,
                **tfm_kwargs,
            )
        assert err_msg_content in str(err.value)
