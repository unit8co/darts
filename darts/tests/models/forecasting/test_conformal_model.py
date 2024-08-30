import copy
import itertools
import os

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries, concatenate
from darts.datasets import AirPassengersDataset
from darts.metrics import ae
from darts.models import (
    ConformalNaiveModel,
    LinearRegressionModel,
    NaiveSeasonal,
    NLinearModel,
)
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils import timeseries_generation as tg

IN_LEN = 3
OUT_LEN = 3
regr_kwargs = {"lags": IN_LEN, "output_chunk_length": OUT_LEN}
tfm_kwargs = copy.deepcopy(tfm_kwargs)
tfm_kwargs["pl_trainer_kwargs"]["fast_dev_run"] = True
torch_kwargs = dict(
    {"input_chunk_length": IN_LEN, "output_chunk_length": OUT_LEN, "random_state": 0},
    **tfm_kwargs,
)


def train_model(*args, model_type="regression", model_params=None, **kwargs):
    model_params = model_params or {}
    if model_type == "regression":
        return LinearRegressionModel(**regr_kwargs, **model_params).fit(*args, **kwargs)
    else:
        return NLinearModel(**torch_kwargs, **model_params).fit(*args, **kwargs)


# pre-trained global model for conformal models
models_cls_kwargs_errs = [
    (
        ConformalNaiveModel,
        {"alpha": 0.8},
        "regression",
    ),
]

if TORCH_AVAILABLE:
    models_cls_kwargs_errs.append((
        ConformalNaiveModel,
        {"alpha": 0.8},
        "torch",
    ))


class TestConformalModel:
    np.random.seed(42)

    # forecasting horizon used in runnability tests
    horizon = OUT_LEN + 1

    # some arbitrary static covariates
    static_covariates = pd.DataFrame([[0.0, 1.0]], columns=["st1", "st2"])

    # real timeseries for functionality tests
    ts_length = 13 + horizon
    ts_passengers = (
        AirPassengersDataset()
        .load()[:ts_length]
        .with_static_covariates(static_covariates)
    )
    ts_pass_train, ts_pass_val = (
        ts_passengers[:-horizon],
        ts_passengers[-horizon:],
    )

    # an additional noisy series
    ts_pass_train_1 = ts_pass_train + 0.01 * tg.gaussian_timeseries(
        length=len(ts_pass_train),
        freq=ts_pass_train.freq_str,
        start=ts_pass_train.start_time(),
    )

    # an additional time series serving as covariates
    year_series = tg.datetime_attribute_timeseries(ts_passengers, attribute="year")
    month_series = tg.datetime_attribute_timeseries(ts_passengers, attribute="month")
    time_covariates = year_series.stack(month_series)
    time_covariates_train = time_covariates[:-horizon]

    # various ts with different static covariates representations
    ts_w_static_cov = tg.linear_timeseries(length=ts_length).with_static_covariates(
        pd.Series([1, 2])
    )
    ts_shared_static_cov = ts_w_static_cov.stack(tg.sine_timeseries(length=ts_length))
    ts_comps_static_cov = ts_shared_static_cov.with_static_covariates(
        pd.DataFrame([[0, 1], [2, 3]], columns=["st1", "st2"])
    )

    def test_model_construction(self):
        local_model = NaiveSeasonal(K=5)
        global_model = LinearRegressionModel(**regr_kwargs)
        series = self.ts_pass_train

        model_err_msg = "`model` must be a pre-trained `GlobalForecastingModel`."
        # un-trained local model
        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=local_model, alpha=0.8)
        assert str(exc.value) == model_err_msg

        # pre-trained local model
        local_model.fit(series)
        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=local_model, alpha=0.8)
        assert str(exc.value) == model_err_msg

        # un-trained global model
        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=global_model, alpha=0.0)
        assert str(exc.value) == model_err_msg

        # pre-trained local model should work
        global_model.fit(series)
        _ = ConformalNaiveModel(model=global_model, alpha=0.8)

    @pytest.mark.parametrize("config", models_cls_kwargs_errs)
    def test_save_model_parameters(self, config):
        # model creation parameters were saved before. check if re-created model has same params as original
        model_cls, kwargs, model_type = config
        model = model_cls(
            model=train_model(self.ts_pass_train, model_type=model_type), **kwargs
        )
        model_fresh = model.untrained_model()
        assert model._model_params.keys() == model_fresh._model_params.keys()
        for param, val in model._model_params.items():
            if isinstance(val, ForecastingModel):
                # Conformal Models require a forecasting model as input, which has no equality
                continue
            assert val == model_fresh._model_params[param]

    @pytest.mark.parametrize("config", models_cls_kwargs_errs)
    def test_save_load_model(self, tmpdir_fn, config):
        # check if save and load methods work and if loaded model creates same forecasts as original model
        model_cls, kwargs, model_type = config
        model = model_cls(
            train_model(self.ts_pass_train, model_type=model_type), **kwargs
        )

        model_path = os.path.join(tmpdir_fn, "model_test.pkl")
        with pytest.raises(NotImplementedError) as exc:
            model.save(model_path)
        assert "does not support saving / loading" in str(exc.value)

    @pytest.mark.parametrize("config", models_cls_kwargs_errs)
    def test_single_ts(self, config):
        model_cls, kwargs, model_type = config
        model = model_cls(
            train_model(self.ts_pass_train, model_type=model_type), **kwargs
        )
        pred = model.predict(n=self.horizon)
        assert pred.n_components == self.ts_pass_train.n_components * 3
        assert not np.isnan(pred.all_values()).any().any()

        pred_fc = model.model.predict(n=self.horizon)
        assert pred_fc.time_index.equals(pred.time_index)
        # the center forecasts must be equal to the forecasting model forecast
        np.testing.assert_array_almost_equal(
            pred[self.ts_pass_val.columns.tolist()].all_values(), pred_fc.all_values()
        )
        assert pred.static_covariates is None

        # using a different `n`, gives different results, since we can generate more residuals for the horizon
        pred1 = model.predict(n=1)
        assert not pred1 == pred

        # giving the same series as calibration set must give the same results
        pred_cal = model.predict(n=self.horizon, cal_series=self.ts_pass_train)
        np.testing.assert_array_almost_equal(pred.all_values(), pred_cal.all_values())

        # wrong dimension
        with pytest.raises(ValueError):
            model.predict(
                n=self.horizon, series=self.ts_pass_train.stack(self.ts_pass_train)
            )

    @pytest.mark.parametrize("config", models_cls_kwargs_errs)
    def test_multi_ts(self, config):
        model_cls, kwargs, model_type = config
        model = model_cls(
            train_model(
                [self.ts_pass_train, self.ts_pass_train_1], model_type=model_type
            ),
            **kwargs,
        )
        with pytest.raises(ValueError):
            # when model is fit from >1 series, one must provide a series in argument
            model.predict(n=1)

        pred = model.predict(n=self.horizon, series=self.ts_pass_train)
        assert pred.n_components == self.ts_pass_train.n_components * 3
        assert not np.isnan(pred.all_values()).any().any()

        # the center forecasts must be equal to the forecasting model forecast
        pred_fc = model.model.predict(n=self.horizon, series=self.ts_pass_train)
        assert pred_fc.time_index.equals(pred.time_index)
        np.testing.assert_array_almost_equal(
            pred[self.ts_pass_val.columns.tolist()].all_values(), pred_fc.all_values()
        )

        # using a calibration series also requires an input series
        with pytest.raises(ValueError):
            # when model is fit from >1 series, one must provide a series in argument
            model.predict(n=1, cal_series=self.ts_pass_train)
        # giving the same series as calibration set must give the same results
        pred_cal = model.predict(
            n=self.horizon,
            series=self.ts_pass_train,
            cal_series=self.ts_pass_train,
        )
        np.testing.assert_array_almost_equal(pred.all_values(), pred_cal.all_values())

        # check prediction for several time series
        pred_list = model.predict(
            n=self.horizon,
            series=[self.ts_pass_train, self.ts_pass_train_1],
        )
        pred_fc_list = model.model.predict(
            n=self.horizon,
            series=[self.ts_pass_train, self.ts_pass_train_1],
        )
        assert (
            len(pred_list) == 2
        ), f"Model {model_cls} did not return a list of prediction"
        for pred, pred_fc in zip(pred_list, pred_fc_list):
            assert pred.n_components == self.ts_pass_train.n_components * 3
            assert pred_fc.time_index.equals(pred.time_index)
            assert not np.isnan(pred.all_values()).any().any()
            np.testing.assert_array_almost_equal(
                pred_fc.all_values(),
                pred[self.ts_pass_val.columns.tolist()].all_values(),
            )

        # using a calibration series requires to have same number of series as target
        with pytest.raises(ValueError) as exc:
            # when model is fit from >1 series, one must provide a series in argument
            model.predict(
                n=1,
                series=[self.ts_pass_train, self.ts_pass_val],
                cal_series=self.ts_pass_train,
            )
        assert (
            str(exc.value)
            == "Mismatch between number of `cal_series` (1) and number of `series` (2)."
        )
        # using a calibration series requires to have same number of series as target
        with pytest.raises(ValueError) as exc:
            # when model is fit from >1 series, one must provide a series in argument
            model.predict(
                n=1,
                series=[self.ts_pass_train, self.ts_pass_val],
                cal_series=[self.ts_pass_train] * 3,
            )
        assert (
            str(exc.value)
            == "Mismatch between number of `cal_series` (3) and number of `series` (2)."
        )

        # giving the same series as calibration set must give the same results
        pred_cal_list = model.predict(
            n=self.horizon,
            series=[self.ts_pass_train, self.ts_pass_train_1],
            cal_series=[self.ts_pass_train, self.ts_pass_train_1],
        )
        for pred, pred_cal in zip(pred_list, pred_cal_list):
            np.testing.assert_array_almost_equal(
                pred.all_values(), pred_cal.all_values()
            )

        # using copies of the same series as calibration set must give the same interval widths for
        # each target series
        pred_cal_list = model.predict(
            n=self.horizon,
            series=[self.ts_pass_train, self.ts_pass_train_1],
            cal_series=[self.ts_pass_train, self.ts_pass_train],
        )

        pred_0_vals = pred_cal_list[0].all_values()
        pred_1_vals = pred_cal_list[1].all_values()

        # lower range
        np.testing.assert_array_almost_equal(
            pred_0_vals[:, 1] - pred_0_vals[:, 0], pred_1_vals[:, 1] - pred_1_vals[:, 0]
        )
        # upper range
        np.testing.assert_array_almost_equal(
            pred_0_vals[:, 2] - pred_0_vals[:, 1], pred_1_vals[:, 2] - pred_1_vals[:, 1]
        )

        # wrong dimension
        with pytest.raises(ValueError):
            model.predict(
                n=self.horizon,
                series=[
                    self.ts_pass_train,
                    self.ts_pass_train.stack(self.ts_pass_train),
                ],
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [(ConformalNaiveModel, {"alpha": 0.8}, "regression")],
            [
                {"lags_past_covariates": IN_LEN},
                {"lags_future_covariates": (IN_LEN, OUT_LEN)},
                {},
            ],
        ),
    )
    def test_covariates(self, config):
        (model_cls, kwargs, model_type), covs_kwargs = config
        model_fc = LinearRegressionModel(**regr_kwargs, **covs_kwargs)
        # Here we rely on the fact that all non-Dual models currently are Past models
        if model_fc.supports_future_covariates:
            cov_name = "future_covariates"
            is_past = False
        elif model_fc.supports_past_covariates:
            cov_name = "past_covariates"
            is_past = True
        else:
            cov_name = None
            is_past = None

        covariates = [self.time_covariates_train, self.time_covariates_train]
        if cov_name is not None:
            cov_kwargs = {cov_name: covariates}
            cov_kwargs_train = {cov_name: self.time_covariates_train}
            cov_kwargs_notrain = {cov_name: self.time_covariates}
        else:
            cov_kwargs = {}
            cov_kwargs_train = {}
            cov_kwargs_notrain = {}

        model_fc.fit(series=[self.ts_pass_train, self.ts_pass_train_1], **cov_kwargs)

        model = model_cls(model=model_fc, **kwargs)
        if cov_name == "future_covariates":
            assert model.supports_future_covariates
            assert not model.supports_past_covariates
            assert model.uses_future_covariates
            assert not model.uses_past_covariates
        elif cov_name == "past_covariates":
            assert not model.supports_future_covariates
            assert model.supports_past_covariates
            assert not model.uses_future_covariates
            assert model.uses_past_covariates
        else:
            assert not model.supports_future_covariates
            assert not model.supports_past_covariates
            assert not model.uses_future_covariates
            assert not model.uses_past_covariates

        with pytest.raises(ValueError):
            # when model is fit from >1 series, one must provide a series in argument
            model.predict(n=1)

        if cov_name is not None:
            with pytest.raises(ValueError):
                # when model is fit using multiple covariates, covariates are required at prediction time
                model.predict(n=1, series=self.ts_pass_train)

            with pytest.raises(ValueError):
                # when model is fit using covariates, n cannot be greater than output_chunk_length...
                # (for short covariates)
                # past covariates model can predict up until output_chunk_length
                # with train future covariates we cannot predict at all after end of series
                model.predict(
                    n=OUT_LEN + 1 if is_past else 1,
                    series=self.ts_pass_train,
                    **cov_kwargs_train,
                )
        else:
            # model does not support covariates
            with pytest.raises(ValueError):
                model.predict(
                    n=1,
                    series=self.ts_pass_train,
                    past_covariates=self.time_covariates,
                )
            with pytest.raises(ValueError):
                model.predict(
                    n=1,
                    series=self.ts_pass_train,
                    future_covariates=self.time_covariates,
                )

        # ... unless future covariates are provided
        _ = model.predict(
            n=self.horizon, series=self.ts_pass_train, **cov_kwargs_notrain
        )

        pred = model.predict(
            n=self.horizon, series=self.ts_pass_train, **cov_kwargs_notrain
        )
        pred_fc = model_fc.predict(
            n=self.horizon,
            series=self.ts_pass_train,
            **cov_kwargs_notrain,
        )
        np.testing.assert_array_almost_equal(
            pred[self.ts_pass_val.columns.tolist()].all_values(),
            pred_fc.all_values(),
        )

        if cov_name is None:
            return

        # when model is fit using 1 training and 1 covariate series, time series args are optional
        model_fc = LinearRegressionModel(**regr_kwargs, **covs_kwargs)
        model_fc.fit(series=self.ts_pass_train, **cov_kwargs_train)
        model = model_cls(model_fc, **kwargs)

        if is_past:
            # can only predict up until ocl
            with pytest.raises(ValueError):
                _ = model.predict(n=OUT_LEN + 1)
            # wrong covariates dimension
            with pytest.raises(ValueError):
                covs = cov_kwargs_train[cov_name]
                covs = {cov_name: covs.stack(covs)}
                _ = model.predict(n=OUT_LEN + 1, **covs)
            # with past covariates from train we can predict up until output_chunk_length
            pred1 = model.predict(n=OUT_LEN)
            pred2 = model.predict(n=OUT_LEN, series=self.ts_pass_train)
            pred3 = model.predict(n=OUT_LEN, **cov_kwargs_train)
            pred4 = model.predict(
                n=OUT_LEN, **cov_kwargs_train, series=self.ts_pass_train
            )
        else:
            # with future covariates we need additional time steps to predict
            with pytest.raises(ValueError):
                _ = model.predict(n=1)
            with pytest.raises(ValueError):
                _ = model.predict(n=1, series=self.ts_pass_train)
            with pytest.raises(ValueError):
                _ = model.predict(n=1, **cov_kwargs_train)
            with pytest.raises(ValueError):
                _ = model.predict(n=1, **cov_kwargs_train, series=self.ts_pass_train)
            # wrong covariates dimension
            with pytest.raises(ValueError):
                covs = cov_kwargs_notrain[cov_name]
                covs = {cov_name: covs.stack(covs)}
                _ = model.predict(n=OUT_LEN + 1, **covs)
            pred1 = model.predict(n=OUT_LEN, **cov_kwargs_notrain)
            pred2 = model.predict(
                n=OUT_LEN, series=self.ts_pass_train, **cov_kwargs_notrain
            )
            pred3 = model.predict(n=OUT_LEN, **cov_kwargs_notrain)
            pred4 = model.predict(
                n=OUT_LEN, **cov_kwargs_notrain, series=self.ts_pass_train
            )

        assert pred1 == pred2
        assert pred1 == pred3
        assert pred1 == pred4

    @pytest.mark.parametrize(
        "config,ts",
        itertools.product(
            models_cls_kwargs_errs,
            [ts_w_static_cov, ts_shared_static_cov, ts_comps_static_cov],
        ),
    )
    def test_use_static_covariates(self, config, ts):
        """
        Check that both static covariates representations are supported (component-specific and shared)
        for both uni- and multivariate series when fitting the model.
        Also check that the static covariates are present in the forecasted series
        """
        model_cls, kwargs, model_type = config
        model = model_cls(train_model(ts, model_type=model_type), **kwargs)
        assert model.uses_static_covariates
        pred = model.predict(OUT_LEN)
        assert pred.static_covariates is None

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],  # univariate series
            [True, False],  # single series
            [True, False],  # use covariates
            [True, False],  # datetime index
            [1, 3, 5],  # different horizons
        ),
    )
    def test_predict(self, config):
        (is_univar, is_single, use_covs, is_datetime, horizon) = config
        series = self.ts_pass_train
        if not is_univar:
            series = series.stack(series)
        if not is_datetime:
            series = TimeSeries.from_values(series.all_values(), columns=series.columns)
        if use_covs:
            pc, fc = series, series
            fc = fc.append_values(fc.values()[: max(horizon, OUT_LEN)])
            if horizon > OUT_LEN:
                pc = pc.append_values(pc.values()[: horizon - OUT_LEN])
            model_kwargs = {
                "lags_past_covariates": IN_LEN,
                "lags_future_covariates": (IN_LEN, OUT_LEN),
            }
        else:
            pc, fc = None, None
            model_kwargs = {}
        if not is_single:
            series = [
                series,
                series.with_columns_renamed(
                    col_names=series.columns.tolist(),
                    col_names_new=(series.columns + "_s2").tolist(),
                ),
            ]
            if use_covs:
                pc = [pc] * 2
                fc = [fc] * 2

        # testing lags_past_covariates None but past_covariates during prediction
        model_instance = LinearRegressionModel(
            lags=IN_LEN, output_chunk_length=OUT_LEN, **model_kwargs
        )
        model_instance.fit(series=series, past_covariates=pc, future_covariates=fc)
        model = ConformalNaiveModel(model_instance, alpha=0.8)

        preds = model.predict(
            n=horizon, series=series, past_covariates=pc, future_covariates=fc
        )

        if is_single:
            series = [series]
            preds = [preds]

        for s_, preds_ in zip(series, preds):
            cols_expected = []
            for col in s_.columns:
                cols_expected += [f"{col}{q}" for q in ["_cq_lo", "", "_cq_hi"]]
            assert preds_.columns.tolist() == cols_expected
            assert len(preds_) == horizon
            assert preds_.start_time() == s_.end_time() + s_.freq
            assert preds_.freq == s_.freq

    def test_output_chunk_shift(self):
        model_params = {"output_chunk_shift": 1}
        model = ConformalNaiveModel(
            train_model(self.ts_pass_train, model_params=model_params), alpha=0.8
        )
        pred = model.predict(n=1)
        pred_fc = model.model.predict(n=1)

        assert pred_fc.time_index.equals(pred.time_index)
        # the center forecasts must be equal to the forecasting model forecast
        np.testing.assert_array_almost_equal(
            pred[self.ts_pass_train.columns.tolist()].all_values(), pred_fc.all_values()
        )

        pred_cal = model.predict(n=1, cal_series=self.ts_pass_train)
        assert pred_fc.time_index.equals(pred_cal.time_index)
        # the center forecasts must be equal to the forecasting model forecast
        np.testing.assert_array_almost_equal(pred_cal.all_values(), pred.all_values())

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [1, 3, 5],  # horizon
            [True, False],  # univariate series
            [True, False],  # single series
        ),
    )
    def test_naive_conformal_model_predict(self, config):
        """Verifies that naive conformal model computes the correct intervals
        The naive approach computes it as follows:

        - pred_upper = pred + q_alpha(absolute error, past)
        - pred_middle = pred
        - pred_lower = pred - q_alpha(absolute error, past)

        Where q_alpha(absolute error) is the `alpha` quantile of all historic absolute errors between
        `pred`, and the target series.
        """
        n, is_univar, is_single = config
        alpha = 0.8
        series = self.helper_prepare_series(is_univar, is_single)
        model_fc = train_model(series)
        pred_fc_list = model_fc.predict(n, series=series)
        model = ConformalNaiveModel(model=model_fc, alpha=alpha)
        pred_cal_list = model.predict(n, series=series)
        pred_cal_list_with_cal = model.predict(n, series=series, cal_series=series)

        # compute the expected intervals
        residuals_list = model_fc.residuals(
            series,
            retrain=False,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=False,
            stride=1,
            values_only=True,
            metric=ae,  # absolute error
        )
        if is_single:
            pred_fc_list = [pred_fc_list]
            pred_cal_list = [pred_cal_list]
            residuals_list = [residuals_list]
            pred_cal_list_with_cal = [pred_cal_list_with_cal]

        for pred_fc, pred_cal, residuals in zip(
            pred_fc_list, pred_cal_list, residuals_list
        ):
            residuals = np.concatenate(residuals[:-1], axis=2)

            pred_vals = pred_fc.all_values()
            pred_vals_expected = self.helper_compute_naive_pred_cal(
                residuals, pred_vals, n, alpha
            )
            np.testing.assert_array_almost_equal(
                pred_cal.all_values(), pred_vals_expected
            )
        assert pred_cal_list_with_cal == pred_cal_list

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [1, 3, 5],  # horizon
            [True, False],  # univariate series
            [True, False],  # single series,
            [0, 1],  # output chunk shift
            [None, 1],  # train length
            [False, True],  # use covariates
        ),
    )
    def test_naive_conformal_model_historical_forecasts(self, config):
        """Checks correctness of naive conformal model historical forecasts for:
        - different horizons (smaller, equal and larger the OCL)
        - uni and multivariate series
        - single and multiple series
        - with and without output shift
        - with and without training length
        - with and without covariates in the forecast and calibration sets.
        """
        n, is_univar, is_single, ocs, train_length, use_covs = config
        if ocs and n > OUT_LEN:
            # auto-regression not allowed with ocs
            return

        alpha = 0.8
        series = self.helper_prepare_series(is_univar, is_single)
        model_params = {"output_chunk_shift": ocs}

        # for covariates, we check that shorter & longer covariates in the calibration set give expected results
        covs_kwargs = {}
        cal_covs_kwargs_overlap = {}
        cal_covs_kwargs_short = {}
        cal_covs_kwargs_exact = {}
        if use_covs:
            model_params["lags_past_covariates"] = regr_kwargs["lags"]
            past_covs = series
            if n > OUT_LEN:
                append_vals = [[[1.0]] * (1 if is_univar else 2)] * (n - OUT_LEN)
                if is_single:
                    past_covs = past_covs.append_values(append_vals)
                else:
                    past_covs = [pc.append_values(append_vals) for pc in past_covs]
            covs_kwargs["past_covariates"] = past_covs
            # produces examples with all points in `overlap_end=True` (last example has no useful information)
            cal_covs_kwargs_overlap["cal_past_covariates"] = past_covs
            # produces one example less (drops the one with unuseful information)
            cal_covs_kwargs_exact["cal_past_covariates"] = (
                past_covs[: -(1 + ocs)]
                if is_single
                else [pc[: -(1 + ocs)] for pc in past_covs]
            )
            # produces another example less (drops the last one which contains useful information)
            cal_covs_kwargs_short["cal_past_covariates"] = (
                past_covs[: -(2 + ocs)]
                if is_single
                else [pc[: -(2 + ocs)] for pc in past_covs]
            )

        # forecasts from forecasting model
        model_fc = train_model(series, model_params=model_params, **covs_kwargs)
        hfc_fc_list = model_fc.historical_forecasts(
            series,
            retrain=False,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=False,
            stride=1,
            **covs_kwargs,
        )
        # residuals to compute the conformal intervals
        residuals_list = model_fc.residuals(
            series,
            historical_forecasts=hfc_fc_list,
            overlap_end=True,
            last_points_only=False,
            values_only=True,
            metric=ae,  # absolute error
            **covs_kwargs,
        )

        # conformal forecasts
        model = ConformalNaiveModel(model=model_fc, alpha=alpha)
        # without calibration set
        hfc_conf_list = model.historical_forecasts(
            series=series,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=False,
            stride=1,
            train_length=train_length,
            **covs_kwargs,
        )
        # with calibration set and covariates that can generate all calibration forecasts in the overlap
        hfc_conf_list_with_cal = model.historical_forecasts(
            series=series,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=False,
            stride=1,
            train_length=train_length,
            cal_series=series,
            **covs_kwargs,
            **cal_covs_kwargs_overlap,
        )

        if is_single:
            hfc_conf_list = [hfc_conf_list]
            residuals_list = [residuals_list]
            hfc_conf_list_with_cal = [hfc_conf_list_with_cal]
            hfc_fc_list = [hfc_fc_list]

        # validate computed conformal intervals that did not use a calibration set
        # conformal models start later since they need past residuals as input
        first_fc_idx = len(hfc_fc_list[0]) - len(hfc_conf_list[0])
        for hfc_fc, hfc_conf, hfc_residuals in zip(
            hfc_fc_list, hfc_conf_list, residuals_list
        ):
            for idx, (pred_fc, pred_cal) in enumerate(
                zip(hfc_fc[first_fc_idx:], hfc_conf)
            ):
                # need to ignore additional `ocs` (output shift) residuals
                residuals = np.concatenate(
                    hfc_residuals[: first_fc_idx - ocs + idx], axis=2
                )

                pred_vals = pred_fc.all_values()
                pred_vals_expected = self.helper_compute_naive_pred_cal(
                    residuals, pred_vals, n, alpha, train_length=train_length
                )
                np.testing.assert_array_almost_equal(
                    pred_cal.all_values(), pred_vals_expected
                )

        # validate computed conformal intervals that used a calibration set
        for hfc_conf_with_cal, hfc_conf in zip(hfc_conf_list_with_cal, hfc_conf_list):
            # last forecast with calibration set must be equal to the last without calibration set
            # (since calibration set is the same series)
            assert hfc_conf_with_cal[-1] == hfc_conf[-1]
            hfc_0_vals = hfc_conf_with_cal[0].all_values()
            for hfc_i in hfc_conf_with_cal[1:]:
                hfc_i_vals = hfc_i.all_values()
                np.testing.assert_array_almost_equal(
                    hfc_0_vals[:, 1::3] - hfc_0_vals[:, 0::3],
                    hfc_i_vals[:, 1::3] - hfc_i_vals[:, 0::3],
                )
                np.testing.assert_array_almost_equal(
                    hfc_0_vals[:, 2::3] - hfc_0_vals[:, 1::3],
                    hfc_i_vals[:, 2::3] - hfc_i_vals[:, 1::3],
                )

        if use_covs:
            # `cal_covs_kwargs_exact` will not compute the last example in overlap_end (this one has anyways no
            # useful information). Result is expected to be identical to the case when using `cal_covs_kwargs_overlap`
            hfc_conf_list_with_cal_exact = model.historical_forecasts(
                series=series,
                forecast_horizon=n,
                overlap_end=True,
                last_points_only=False,
                stride=1,
                train_length=train_length,
                cal_series=series,
                **covs_kwargs,
                **cal_covs_kwargs_exact,
            )

            # `cal_covs_kwargs_short` will compute example less that contains useful information
            hfc_conf_list_with_cal_short = model.historical_forecasts(
                series=series,
                forecast_horizon=n,
                overlap_end=True,
                last_points_only=False,
                stride=1,
                train_length=train_length,
                cal_series=series,
                **covs_kwargs,
                **cal_covs_kwargs_short,
            )
            if is_single:
                hfc_conf_list_with_cal_exact = [hfc_conf_list_with_cal_exact]
                hfc_conf_list_with_cal_short = [hfc_conf_list_with_cal_short]

            # must match
            assert hfc_conf_list_with_cal_exact == hfc_conf_list_with_cal

            # second last forecast with shorter calibration set (that has one example less) must be equal to the
            # second last without calibration set
            for hfc_conf_with_cal, hfc_conf in zip(
                hfc_conf_list_with_cal_short, hfc_conf_list
            ):
                assert hfc_conf_with_cal[-2] == hfc_conf[-2]

        # checking that last points only is equal to the last forecasted point
        hfc_lpo_list = model.historical_forecasts(
            series=series,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=True,
            stride=1,
            train_length=train_length,
            **covs_kwargs,
        )
        hfc_lpo_list_with_cal = model.historical_forecasts(
            series=series,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=True,
            stride=1,
            train_length=train_length,
            cal_series=series,
            **covs_kwargs,
            **cal_covs_kwargs_overlap,
        )
        if is_single:
            hfc_lpo_list = [hfc_lpo_list]
            hfc_lpo_list_with_cal = [hfc_lpo_list_with_cal]

        for hfc_lpo, hfc_conf in zip(hfc_lpo_list, hfc_conf_list):
            hfc_conf_lpo = concatenate([hfc[-1:] for hfc in hfc_conf], axis=0)
            assert hfc_lpo == hfc_conf_lpo

        for hfc_lpo, hfc_conf in zip(hfc_lpo_list_with_cal, hfc_conf_list_with_cal):
            hfc_conf_lpo = concatenate([hfc[-1:] for hfc in hfc_conf], axis=0)
            assert hfc_lpo == hfc_conf_lpo

    def helper_prepare_series(self, is_univar, is_single):
        series = self.ts_pass_train
        if not is_univar:
            series = series.stack(series + 3.0)
        if not is_single:
            series = [series, series + 5]
        return series

    @staticmethod
    def helper_compute_naive_pred_cal(
        residuals, pred_vals, n, alpha, train_length=None
    ):
        train_length = train_length or 0
        q_hats = []
        # compute the quantile `alpha` of all past residuals (absolute "per time step" errors between historical
        # forecasts and the target series)
        for idx in range(n):
            res_end = residuals.shape[2] - idx
            if train_length:
                res_start = res_end - train_length
            else:
                res_start = n - (idx + 1)
            res_n = residuals[idx][:, res_start:res_end]
            q_hat_n = np.quantile(res_n, q=alpha, axis=1)
            q_hats.append(q_hat_n)
        q_hats = np.expand_dims(np.array(q_hats), -1)
        # the prediciton interval is given by pred +/- q_hat
        n_comps = pred_vals.shape[1]
        pred_vals_expected = []
        for col_idx in range(n_comps):
            q_col = q_hats[:, col_idx]
            pred_col = pred_vals[:, col_idx]
            pred_col_expected = np.concatenate(
                [pred_col - q_col, pred_col, pred_col + q_col], axis=1
            )
            pred_col_expected = np.expand_dims(pred_col_expected, -1)
            pred_vals_expected.append(pred_col_expected)
        pred_vals_expected = np.concatenate(pred_vals_expected, axis=1)
        return pred_vals_expected

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [1, 3, 5],  # horizon
            [0, 1],  # output chunk shift
            [False, True],  # use covariates
        ),
    )
    def test_too_short_input_predict(self, config):
        """Checks conformal model predict with minimum required input and too short input."""
        n, ocs, use_covs = config
        if ocs and n > OUT_LEN:
            return
        icl = IN_LEN
        min_len = icl + ocs + n
        series = tg.linear_timeseries(length=min_len)
        series_train = [tg.linear_timeseries(length=IN_LEN + OUT_LEN + ocs)] * 2

        model_params = {"output_chunk_shift": ocs}
        covs_kwargs = {}
        cal_covs_kwargs = {}
        covs_kwargs_train = {}
        covs_kwargs_too_short = {}
        cal_covs_kwargs_short = {}
        if use_covs:
            model_params["lags_past_covariates"] = regr_kwargs["lags"]
            covs_kwargs_train["past_covariates"] = series_train
            # use shorter covariates, to test whether residuals are still properly extracted
            past_covs = series
            # for auto-regression, we require longer past covariates
            if n > OUT_LEN:
                past_covs = past_covs.append_values([1.0] * (n - OUT_LEN))
            covs_kwargs["past_covariates"] = past_covs
            covs_kwargs_too_short["past_covariates"] = past_covs[:-1]
            # giving covs in calibration set requires one calibration example less
            cal_covs_kwargs["cal_past_covariates"] = past_covs[: -(1 + ocs)]
            cal_covs_kwargs_short["cal_past_covariates"] = past_covs[: -(2 + ocs)]

        model = ConformalNaiveModel(
            train_model(
                series=series_train,
                model_params=model_params,
                **covs_kwargs_train,
            ),
            alpha=0.8,
        )

        # prediction works with long enough input
        preds1 = model.predict(n=n, series=series, **covs_kwargs)
        assert not np.isnan(preds1.all_values()).any().any()
        preds2 = model.predict(
            n=n, series=series, **covs_kwargs, cal_series=series, **cal_covs_kwargs
        )
        assert not np.isnan(preds2.all_values()).any().any()
        # series too short: without covariates, make `series` shorter. Otherwise, use the shorter covariates
        series_ = series[:-1] if not use_covs else series

        with pytest.raises(ValueError) as exc:
            _ = model.predict(n=n, series=series_, **covs_kwargs_too_short)
        if not use_covs:
            assert str(exc.value).startswith(
                "Could not build the minimum required calibration input with the provided `cal_series`"
            )
        else:
            # if `past_covariates` are too short, then it raises error from the forecasting_model.predict()
            assert str(exc.value).startswith(
                "The `past_covariates` at list/sequence index 0 are not long enough."
            )

        with pytest.raises(ValueError) as exc:
            _ = model.predict(
                n=n,
                series=series,
                cal_series=series_,
                **covs_kwargs,
                **cal_covs_kwargs_short,
            )
        if not use_covs or n > 1:
            assert str(exc.value).startswith(
                "Could not build the minimum required calibration input with the provided `cal_series`"
            )
        else:
            # if `cal_past_covariates` are too short and `horizon=1`, then it raises error from the forecasting model
            assert str(exc.value).startswith(
                "Cannot build a single input for prediction with the provided model"
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [False, True],  # last points only
            [False, True],  # overlap end
            [None, 2],  # train length
            [0, 1],  # output chunk shift
            [1, 3, 5],  # horizon
            [True, False],  # use covs
        ),
    )
    def test_too_short_input_hfc(self, config):
        """Checks conformal model historical forecasts with minimum required input and too short input."""
        (
            last_points_only,
            overlap_end,
            train_length,
            ocs,
            n,
            use_covs,
        ) = config
        if ocs and n > OUT_LEN:
            return

        icl = IN_LEN
        ocl = OUT_LEN
        horizon_ocs = n + ocs
        add_train_length = train_length - 1 if train_length is not None else 0
        # min length to generate 1 conformal forecast
        min_len_val_series = (
            icl + horizon_ocs * (1 + int(not overlap_end)) + add_train_length
        )

        series_train = [tg.linear_timeseries(length=icl + ocl + ocs)] * 2
        series = tg.linear_timeseries(length=min_len_val_series)

        # define cal series to get the minimum required cal set
        if overlap_end:
            # with overlap_end `series` has the exact length to generate one forecast after the end of the input series
            # Therefore, `series` has already the minimum length for one calibrated forecast
            cal_series = series
        else:
            # without overlap_end, we use a shorter input, since the last forecast is within the input series
            # (it generates more residuals with useful information than the minimum requirements)
            cal_series = series[:-horizon_ocs]

        series_with_cal = series[: -(horizon_ocs + add_train_length)]

        model_params = {"output_chunk_shift": ocs}
        covs_kwargs_train = {}
        covs_kwargs = {}
        covs_with_cal_kwargs = {}
        cal_covs_kwargs = {}
        covs_kwargs_short = {}
        cal_covs_kwargs_short = {}
        if use_covs:
            model_params["lags_past_covariates"] = regr_kwargs["lags"]
            covs_kwargs_train["past_covariates"] = series_train

            # `- horizon_ocs` to generate forecasts extending up until end of target series
            if not overlap_end:
                past_covs = series[:-horizon_ocs]
            else:
                past_covs = series

            # calibration set is always generated internally with `overlap_end=True`
            # make shorter to not compute residuals without useful information
            cal_past_covs = cal_series[: -(1 + ocs)]

            # last_points_only requires `horizon` residuals less
            if last_points_only:
                cal_past_covs = cal_past_covs[: (-(n - 1) or None)]

            # for auto-regression, we require longer past covariates
            if n > OUT_LEN:
                past_covs = past_covs.append_values([1.0] * (n - OUT_LEN))
                cal_past_covs = cal_past_covs.append_values([1.0] * (n - OUT_LEN))

            # covariates lengths to generate exactly one forecast
            covs_kwargs["past_covariates"] = past_covs
            # giving a calibration set requires fewer forecasts
            covs_with_cal_kwargs["past_covariates"] = past_covs[:-horizon_ocs]
            cal_covs_kwargs["cal_past_covariates"] = cal_past_covs

            # use too short covariates to check that errors are raised
            covs_kwargs_short["past_covariates"] = covs_kwargs["past_covariates"][:-1]
            cal_covs_kwargs_short["cal_past_covariates"] = cal_covs_kwargs[
                "cal_past_covariates"
            ][:-1]

        model = ConformalNaiveModel(
            train_model(
                series=series_train,
                model_params=model_params,
                **covs_kwargs_train,
            ),
            alpha=0.8,
        )

        hfc_kwargs = {
            "last_points_only": last_points_only,
            "train_length": train_length,
            "overlap_end": overlap_end,
            "forecast_horizon": n,
        }
        # prediction works with long enough input
        hfcs = model.historical_forecasts(
            series=series,
            **covs_kwargs,
            **hfc_kwargs,
        )
        hfcs_cal = model.historical_forecasts(
            series=series_with_cal,
            cal_series=cal_series,
            **covs_with_cal_kwargs,
            **cal_covs_kwargs,
            **hfc_kwargs,
        )
        if last_points_only:
            hfcs = [hfcs]
            hfcs_cal = [hfcs_cal]

        assert len(hfcs) == len(hfcs_cal) == 1
        for hfc, hfc_cal in zip(hfcs, hfcs_cal):
            assert not np.isnan(hfc.all_values()).any().any()
            assert not np.isnan(hfc_cal.all_values()).any().any()

        # input too short: without covariates, make `series` shorter. Otherwise, use the shorter covariates
        series_ = series[:-1] if not use_covs else series
        cal_series_ = cal_series[:-1] if not use_covs else cal_series

        with pytest.raises(ValueError) as exc:
            _ = model.historical_forecasts(
                series=series_,
                **covs_kwargs_short,
                **hfc_kwargs,
            )
        assert str(exc.value).startswith(
            "Could not build the minimum required calibration input with the provided `series` and `*_covariates`"
        )

        with pytest.raises(ValueError) as exc:
            _ = model.historical_forecasts(
                series=series_with_cal,
                cal_series=cal_series_,
                **covs_with_cal_kwargs,
                **cal_covs_kwargs_short,
                **hfc_kwargs,
            )
        if (not use_covs or n > 1 or (train_length or 1) > 1) and not (
            last_points_only and use_covs and train_length is None
        ):
            assert str(exc.value).startswith(
                "Could not build the minimum required calibration input with the provided `cal_series`"
            )
        else:
            assert str(exc.value).startswith(
                "Cannot build a single input for prediction with the provided model"
            )
