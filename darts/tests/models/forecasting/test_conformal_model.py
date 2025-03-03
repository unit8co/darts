import copy
import itertools
import math
import os

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries, concatenate
from darts.datasets import AirPassengersDataset
from darts.metrics import ae, err, ic, incs_qr, mic
from darts.models import (
    ConformalNaiveModel,
    ConformalQRModel,
    LinearRegressionModel,
    NaiveSeasonal,
    NLinearModel,
)
from darts.models.forecasting.conformal_models import _get_calibration_hfc_start
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.tests.conftest import TORCH_AVAILABLE, tfm_kwargs
from darts.utils import n_steps_between
from darts.utils import timeseries_generation as tg
from darts.utils.timeseries_generation import linear_timeseries
from darts.utils.utils import (
    likelihood_component_names,
    quantile_interval_names,
    quantile_names,
)

IN_LEN = 3
OUT_LEN = 3
regr_kwargs = {"lags": IN_LEN, "output_chunk_length": OUT_LEN}
tfm_kwargs = copy.deepcopy(tfm_kwargs)
tfm_kwargs["pl_trainer_kwargs"]["fast_dev_run"] = True
torch_kwargs = dict(
    {"input_chunk_length": IN_LEN, "output_chunk_length": OUT_LEN, "random_state": 0},
    **tfm_kwargs,
)
pred_lklp = {"num_samples": 1, "predict_likelihood_parameters": True}
q = [0.1, 0.5, 0.9]


def train_model(
    *args, model_type="regression", model_params=None, quantiles=None, **kwargs
):
    model_params = model_params or {}
    if model_type == "regression":
        return LinearRegressionModel(
            **regr_kwargs,
            **model_params,
            random_state=42,
        ).fit(*args, **kwargs)
    elif model_type in ["regression_prob", "regression_qr"]:
        return LinearRegressionModel(
            likelihood="quantile",
            quantiles=quantiles,
            **regr_kwargs,
            **model_params,
            random_state=42,
        ).fit(*args, **kwargs)
    else:
        return NLinearModel(**torch_kwargs, **model_params).fit(*args, **kwargs)


# pre-trained global model for conformal models
models_cls_kwargs_errs = [
    (
        ConformalNaiveModel,
        {"quantiles": q},
        "regression",
    ),
]

if TORCH_AVAILABLE:
    models_cls_kwargs_errs.append((
        ConformalNaiveModel,
        {"quantiles": q},
        "torch",
    ))


class TestConformalModel:
    """
    Tests all general model behavior for Naive Conformal Model with symmetric non-conformity score.
    Additionally, checks correctness of predictions for:
    - ConformalNaiveModel with symmetric & asymmetric non-conformity scores
    - ConformalQRModel with symmetric & asymmetric non-conformity scores
    """

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

    def test_model_construction_naive(self):
        local_model = NaiveSeasonal(K=5)
        global_model = LinearRegressionModel(**regr_kwargs)
        series = self.ts_pass_train

        model_err_msg = "`model` must be a pre-trained `GlobalForecastingModel`."
        # un-trained local model
        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=local_model, quantiles=q)
        assert str(exc.value) == model_err_msg

        # pre-trained local model
        local_model.fit(series)
        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=local_model, quantiles=q)
        assert str(exc.value) == model_err_msg

        # un-trained global model
        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=global_model, quantiles=q)
        assert str(exc.value) == model_err_msg

        # pre-trained local model should work
        global_model.fit(series)
        model = ConformalNaiveModel(model=global_model, quantiles=q)
        assert model.likelihood == "quantile"

        # non-centered quantiles
        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=global_model, quantiles=[0.2, 0.5, 0.6])
        assert str(exc.value) == (
            "quantiles lower than `q=0.5` need to share same difference to `0.5` as quantiles higher than `q=0.5`"
        )

        # quantiles missing median
        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=global_model, quantiles=[0.1, 0.9])
        assert str(exc.value) == "median quantile `q=0.5` must be in `quantiles`"

        # too low and high quantiles
        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=global_model, quantiles=[-0.1, 0.5, 1.1])
        assert str(exc.value) == "All provided quantiles must be between 0 and 1."

        # `cal_length` must be `>=1` or `None`
        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=global_model, quantiles=q, cal_length=0)
        assert str(exc.value) == "`cal_length` must be `>=1` or `None`."

        # `cal_stride` must be `>=1`
        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=global_model, quantiles=q, cal_stride=0)
        assert str(exc.value) == "`cal_stride` must be `>=1`."

        # `num_samples` must be `>=1`
        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=global_model, quantiles=q, cal_num_samples=0)
        assert str(exc.value) == "`cal_num_samples` must be `>=1`."

    def test_model_hfc_stride_checks(self):
        series = self.ts_pass_train
        model = LinearRegressionModel(**regr_kwargs).fit(series)
        cp_model = ConformalNaiveModel(model=model, quantiles=q, cal_stride=2)

        expected_error_start = (
            "The provided `stride` parameter must be a round-multiple of "
            "`cal_stride=2` and `>=cal_stride`."
        )
        # `stride` must be >= `cal_stride`
        with pytest.raises(ValueError) as exc:
            cp_model.historical_forecasts(series=series, stride=1)
        assert str(exc.value).startswith(expected_error_start)

        # `stride` must be a round multiple of `cal_stride`
        with pytest.raises(ValueError) as exc:
            cp_model.historical_forecasts(series=series, stride=3)
        assert str(exc.value).startswith(expected_error_start)

        # valid stride
        _ = cp_model.historical_forecasts(series=series, stride=4)

    def test_model_construction_cqr(self):
        model_det = train_model(self.ts_pass_train, model_type="regression")
        model_prob_q = train_model(
            self.ts_pass_train, model_type="regression_prob", quantiles=q
        )
        model_prob_poisson = train_model(
            self.ts_pass_train,
            model_type="regression",
            model_params={"likelihood": "poisson"},
        )

        # deterministic global model
        with pytest.raises(ValueError) as exc:
            ConformalQRModel(model=model_det, quantiles=q)
        assert str(exc.value).startswith(
            "`model` must support probabilistic forecasting."
        )
        # probabilistic model works
        _ = ConformalQRModel(model=model_prob_q, quantiles=q)
        # works also with different likelihood
        _ = ConformalQRModel(model=model_prob_poisson, quantiles=q)

    def test_unsupported_properties(self):
        """Tests only here for coverage, maybe at some point we support these properties."""
        model = ConformalNaiveModel(train_model(self.ts_pass_train), quantiles=q)
        unsupported_properties = [
            "_model_encoder_settings",
            "extreme_lags",
            "min_train_series_length",
            "min_train_samples",
        ]
        for prop in unsupported_properties:
            with pytest.raises(NotImplementedError):
                getattr(model, prop)

    @pytest.mark.parametrize("config", models_cls_kwargs_errs)
    def test_save_model_parameters(self, config):
        # model creation parameters were saved before. check if re-created model has same params as original
        model_cls, kwargs, model_type = config
        model = model_cls(
            model=train_model(
                self.ts_pass_train, model_type=model_type, quantiles=kwargs["quantiles"]
            ),
            **kwargs,
        )
        model_fresh = model.untrained_model()
        assert model._model_params.keys() == model_fresh._model_params.keys()
        for param, val in model._model_params.items():
            if isinstance(val, ForecastingModel):
                # Conformal Models require a forecasting model as input, which has no equality
                continue
            assert val == model_fresh._model_params[param]

    @pytest.mark.parametrize(
        "config", itertools.product(models_cls_kwargs_errs, [{}, pred_lklp])
    )
    def test_save_load_model(self, tmpdir_fn, config):
        # check if save and load methods work and if loaded model creates same forecasts as original model
        (model_cls, kwargs, model_type), pred_kwargs = config
        model = model_cls(
            train_model(
                self.ts_pass_train, model_type=model_type, quantiles=kwargs["quantiles"]
            ),
            **kwargs,
        )

        # check if save and load methods work and
        # if loaded conformal model creates same forecasts as original ensemble models
        expected_suffixes = [
            ".pkl",
            ".pkl.NLinearModel.pt",
            ".pkl.NLinearModel.pt.ckpt",
        ]

        # test save
        model.save()
        model.save(os.path.join(tmpdir_fn, f"{model_cls.__name__}.pkl"))

        model_prediction = model.predict(5, **pred_kwargs)

        assert os.path.exists(tmpdir_fn)
        files = os.listdir(tmpdir_fn)
        if model_type == "torch":
            # 1 from conformal model, 2 from torch, * 2 as `save()` was called twice
            assert len(files) == 6
            for f in files:
                assert f.startswith(model_cls.__name__)
            suffix_counts = {
                suffix: sum(1 for p in os.listdir(tmpdir_fn) if p.endswith(suffix))
                for suffix in expected_suffixes
            }
            assert all(count == 2 for count in suffix_counts.values())
        else:
            assert len(files) == 2
            for f in files:
                assert f.startswith(model_cls.__name__) and f.endswith(".pkl")

        # test load
        pkl_files = []
        for filename in os.listdir(tmpdir_fn):
            if filename.endswith(".pkl"):
                pkl_files.append(os.path.join(tmpdir_fn, filename))
        for p in pkl_files:
            loaded_model = model_cls.load(p)
            assert model_prediction == loaded_model.predict(5, **pred_kwargs)

            # test pl_trainer_kwargs (only for torch models)
            loaded_model = model_cls.load(p, pl_trainer_kwargs={"accelerator": "cuda"})
            if model_type == "torch":
                assert loaded_model.model.trainer_params["accelerator"] == "cuda"

        # test clean save
        clean_model_path = os.path.join(tmpdir_fn, f"clean_{model_cls.__name__}.pkl")
        model.save(clean_model_path, clean=True)
        clean_model = model_cls.load(
            clean_model_path, pl_trainer_kwargs={"accelerator": "cpu"}
        )
        assert clean_model.model.training_series is None
        assert clean_model.model.past_covariate_series is None
        assert clean_model.model.future_covariate_series is None

        clean_model_prediction = clean_model.predict(
            5, self.ts_pass_train, **pred_kwargs
        )
        # Need the same number of previous call to predict (for random state)
        assert model.predict(5, **pred_kwargs) == clean_model_prediction

    def test_fit(self):
        model = ConformalNaiveModel(train_model(self.ts_pass_train), quantiles=q)
        assert model.model._fit_called

        # check kwargs will be passed to `model.model.fit()`
        assert model.supports_sample_weight
        model.model._fit_called = False
        model.fit(self.ts_pass_train, sample_weight="linear")
        assert model.model._fit_called

    @pytest.mark.parametrize("config", models_cls_kwargs_errs)
    def test_single_ts(self, config):
        model_cls, kwargs, model_type = config
        model = model_cls(
            train_model(
                self.ts_pass_train, model_type=model_type, quantiles=kwargs["quantiles"]
            ),
            **kwargs,
        )
        pred = model.predict(n=self.horizon, **pred_lklp)
        assert pred.n_components == self.ts_pass_train.n_components * len(
            kwargs["quantiles"]
        )
        assert not np.isnan(pred.all_values()).any().any()

        pred_fc = model.model.predict(n=self.horizon)
        assert pred_fc.time_index.equals(pred.time_index)
        # the center forecasts must be equal to the forecasting model forecast
        fc_columns = likelihood_component_names(
            self.ts_pass_val.columns, quantile_names([0.5])
        )
        np.testing.assert_array_almost_equal(
            pred[fc_columns].all_values(), pred_fc.all_values()
        )
        assert pred.static_covariates is None

        # using a different `n`, gives different results, since we can generate more residuals for the horizon
        pred1 = model.predict(n=self.horizon - 1, **pred_lklp)
        assert not pred1 == pred[: len(pred1)]

        # wrong dimension
        with pytest.raises(ValueError):
            model.predict(
                n=self.horizon,
                series=self.ts_pass_train.stack(self.ts_pass_train),
                **pred_lklp,
            )

    @pytest.mark.parametrize("config", models_cls_kwargs_errs)
    def test_multi_ts(self, config):
        model_cls, kwargs, model_type = config
        model = model_cls(
            train_model(
                [self.ts_pass_train, self.ts_pass_train_1],
                model_type=model_type,
                quantiles=kwargs["quantiles"],
            ),
            **kwargs,
        )
        with pytest.raises(ValueError):
            # when model is fit from >1 series, one must provide a series in argument
            model.predict(n=1)

        pred = model.predict(n=self.horizon, series=self.ts_pass_train, **pred_lklp)
        assert pred.n_components == self.ts_pass_train.n_components * len(
            kwargs["quantiles"]
        )
        assert not np.isnan(pred.all_values()).any().any()

        # the center forecasts must be equal to the forecasting model forecast
        fc_columns = likelihood_component_names(
            self.ts_pass_val.columns, quantile_names([0.5])
        )
        pred_fc = model.model.predict(n=self.horizon, series=self.ts_pass_train)
        assert pred_fc.time_index.equals(pred.time_index)
        np.testing.assert_array_almost_equal(
            pred[fc_columns].all_values(), pred_fc.all_values()
        )

        # check prediction for several time series
        pred_list = model.predict(
            n=self.horizon,
            series=[self.ts_pass_train, self.ts_pass_train_1],
            **pred_lklp,
        )
        pred_fc_list = model.model.predict(
            n=self.horizon,
            series=[self.ts_pass_train, self.ts_pass_train_1],
        )
        assert len(pred_list) == 2, (
            f"Model {model_cls} did not return a list of prediction"
        )
        for pred, pred_fc in zip(pred_list, pred_fc_list):
            assert pred.n_components == self.ts_pass_train.n_components * len(
                kwargs["quantiles"]
            )
            assert pred_fc.time_index.equals(pred.time_index)
            assert not np.isnan(pred.all_values()).any().any()
            np.testing.assert_array_almost_equal(
                pred_fc.all_values(),
                pred[fc_columns].all_values(),
            )

        # wrong dimension
        with pytest.raises(ValueError):
            model.predict(
                n=self.horizon,
                series=[
                    self.ts_pass_train,
                    self.ts_pass_train.stack(self.ts_pass_train),
                ],
                **pred_lklp,
            )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [(ConformalNaiveModel, {"quantiles": [0.1, 0.5, 0.9]}, "regression")],
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
            n=self.horizon, series=self.ts_pass_train, **cov_kwargs_notrain, **pred_lklp
        )
        pred_fc = model_fc.predict(
            n=self.horizon,
            series=self.ts_pass_train,
            **cov_kwargs_notrain,
        )
        fc_columns = likelihood_component_names(
            self.ts_pass_val.columns, quantile_names([0.5])
        )
        np.testing.assert_array_almost_equal(
            pred[fc_columns].all_values(),
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
                _ = model.predict(n=OUT_LEN + 1, **pred_lklp)
            # wrong covariates dimension
            with pytest.raises(ValueError):
                covs = cov_kwargs_train[cov_name]
                covs = {cov_name: covs.stack(covs)}
                _ = model.predict(n=OUT_LEN, **covs, **pred_lklp)
            # with past covariates from train we can predict up until output_chunk_length
            pred1 = model.predict(n=OUT_LEN, **pred_lklp)
            pred2 = model.predict(n=OUT_LEN, series=self.ts_pass_train, **pred_lklp)
            pred3 = model.predict(n=OUT_LEN, **cov_kwargs_train, **pred_lklp)
            pred4 = model.predict(
                n=OUT_LEN, **cov_kwargs_train, series=self.ts_pass_train, **pred_lklp
            )
        else:
            # with future covariates we need additional time steps to predict
            with pytest.raises(ValueError):
                _ = model.predict(n=1, **pred_lklp)
            with pytest.raises(ValueError):
                _ = model.predict(n=1, series=self.ts_pass_train, **pred_lklp)
            with pytest.raises(ValueError):
                _ = model.predict(n=1, **cov_kwargs_train, **pred_lklp)
            with pytest.raises(ValueError):
                _ = model.predict(
                    n=1, **cov_kwargs_train, series=self.ts_pass_train, **pred_lklp
                )
            # wrong covariates dimension
            with pytest.raises(ValueError):
                covs = cov_kwargs_notrain[cov_name]
                covs = {cov_name: covs.stack(covs)}
                _ = model.predict(n=OUT_LEN, **covs, **pred_lklp)
            pred1 = model.predict(n=OUT_LEN, **cov_kwargs_notrain, **pred_lklp)
            pred2 = model.predict(
                n=OUT_LEN, series=self.ts_pass_train, **cov_kwargs_notrain, **pred_lklp
            )
            pred3 = model.predict(n=OUT_LEN, **cov_kwargs_notrain, **pred_lklp)
            pred4 = model.predict(
                n=OUT_LEN, **cov_kwargs_notrain, series=self.ts_pass_train, **pred_lklp
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
        model = model_cls(
            train_model(ts, model_type=model_type, quantiles=kwargs["quantiles"]),
            **kwargs,
        )
        assert model.considers_static_covariates
        assert model.supports_static_covariates
        assert model.uses_static_covariates
        pred = model.predict(OUT_LEN)
        assert pred.static_covariates.equals(ts.static_covariates)

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
        model = ConformalNaiveModel(model_instance, quantiles=q)

        preds = model.predict(
            n=horizon,
            series=series,
            past_covariates=pc,
            future_covariates=fc,
            **pred_lklp,
        )

        if is_single:
            series = [series]
            preds = [preds]

        for s_, preds_ in zip(series, preds):
            cols_expected = likelihood_component_names(s_.columns, quantile_names(q))
            assert preds_.columns.tolist() == cols_expected
            assert len(preds_) == horizon
            assert preds_.start_time() == s_.end_time() + s_.freq
            assert preds_.freq == s_.freq

    def test_output_chunk_shift(self):
        model_params = {"output_chunk_shift": 1}
        model = ConformalNaiveModel(
            train_model(self.ts_pass_train, model_params=model_params, quantiles=q),
            quantiles=q,
        )
        pred = model.predict(n=1, **pred_lklp)
        pred_fc = model.model.predict(n=1)

        assert pred_fc.time_index.equals(pred.time_index)
        # the center forecasts must be equal to the forecasting model forecast
        fc_columns = likelihood_component_names(
            self.ts_pass_train.columns, quantile_names([0.5])
        )

        np.testing.assert_array_almost_equal(
            pred[fc_columns].all_values(), pred_fc.all_values()
        )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [1, 3, 5],  # horizon
            [True, False],  # univariate series
            [True, False],  # single series
            [q, [0.2, 0.3, 0.5, 0.7, 0.8]],
            [
                (ConformalNaiveModel, "regression"),
                (ConformalNaiveModel, "regression_prob"),
                (ConformalQRModel, "regression_qr"),
            ],  # model type
            [True, False],  # symmetric non-conformity score
            [None, 1],  # train length
        ),
    )
    def test_conformal_model_predict_accuracy(self, config):
        """Verifies that naive conformal model computes the correct intervals for:
        - different horizons (smaller, equal, larger than ocl)
        - uni/multivariate series
        - single/multi series
        - single/multi quantile intervals
        - deterministic/probabilistic forecasting model
        - naive conformal and conformalized quantile regression
        - symmetric/asymmetric non-conformity scores

        The naive approach computes it as follows:

        - pred_upper = pred + q_interval(absolute error, past)
        - pred_middle = pred
        - pred_lower = pred - q_interval(absolute error, past)

        Where q_interval(absolute error) is the `q_hi - q_hi` quantile value of all historic absolute errors
        between `pred`, and the target series.
        """
        (
            n,
            is_univar,
            is_single,
            quantiles,
            (model_cls, model_type),
            symmetric,
            cal_length,
        ) = config
        idx_med = quantiles.index(0.5)
        q_intervals = [
            (q_hi, q_lo)
            for q_hi, q_lo in zip(quantiles[:idx_med], quantiles[idx_med + 1 :][::-1])
        ]
        series = self.helper_prepare_series(is_univar, is_single)
        pred_kwargs = (
            {"num_samples": 1000}
            if model_type in ["regression_prob", "regression_qr"]
            else {}
        )

        model_fc = train_model(series, model_type=model_type, quantiles=q)
        model = model_cls(
            model=model_fc,
            quantiles=quantiles,
            symmetric=symmetric,
            cal_length=cal_length,
        )
        pred_fc_list = model.model.predict(n, series=series, **pred_kwargs)
        pred_cal_list = model.predict(n, series=series, **pred_lklp)

        if issubclass(model_cls, ConformalNaiveModel):
            metric = ae if symmetric else err
            metric_kwargs = {}
        else:
            metric = incs_qr
            metric_kwargs = {"q_interval": q_intervals, "symmetric": symmetric}
        # compute the expected intervals
        residuals_list = model.model.residuals(
            series,
            retrain=False,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=False,
            stride=1,
            values_only=True,
            metric=metric,
            metric_kwargs=metric_kwargs,
            **pred_kwargs,
        )
        if is_single:
            pred_fc_list = [pred_fc_list]
            pred_cal_list = [pred_cal_list]
            residuals_list = [residuals_list]

        for pred_fc, pred_cal, residuals in zip(
            pred_fc_list, pred_cal_list, residuals_list
        ):
            residuals = np.concatenate(residuals[:-1], axis=2)

            pred_vals = pred_fc.all_values()
            pred_vals_expected = self.helper_compute_pred_cal(
                residuals,
                pred_vals,
                n,
                quantiles,
                model_type,
                symmetric,
                cal_length=cal_length,
            )
            self.helper_compare_preds(pred_cal, pred_vals_expected, model_type)

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [1, 3, 5],  # horizon
            [True, False],  # univariate series
            [True, False],  # single series,
            [0, 1],  # output chunk shift
            [None, 1],  # train length
            [False, True],  # use covariates
            [q, [0.2, 0.3, 0.5, 0.7, 0.8]],  # quantiles
        ),
    )
    def test_naive_conformal_model_historical_forecasts(self, config):
        """Checks correctness of naive conformal model historical forecasts for:
        - different horizons (smaller, equal and larger the OCL)
        - uni and multivariate series
        - single and multiple series
        - with and without output shift
        - with and without training length
        - with and without covariates
        """
        n, is_univar, is_single, ocs, cal_length, use_covs, quantiles = config
        if ocs and n > OUT_LEN:
            # auto-regression not allowed with ocs
            return

        series = self.helper_prepare_series(is_univar, is_single)
        model_params = {"output_chunk_shift": ocs}

        # for covariates, we check that shorter & longer covariates in the calibration set give expected results
        covs_kwargs = {}
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
        model = ConformalNaiveModel(
            model=model_fc, quantiles=quantiles, cal_length=cal_length
        )
        hfc_conf_list = model.historical_forecasts(
            series=series,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=False,
            stride=1,
            **covs_kwargs,
            **pred_lklp,
        )

        if is_single:
            hfc_conf_list = [hfc_conf_list]
            residuals_list = [residuals_list]
            hfc_fc_list = [hfc_fc_list]

        # validate computed conformal intervals; conformal models start later since they need past residuals as input
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
                pred_vals_expected = self.helper_compute_pred_cal(
                    residuals,
                    pred_vals,
                    n,
                    quantiles,
                    cal_length=cal_length,
                    model_type="regression",
                    symmetric=True,
                )
                np.testing.assert_array_almost_equal(
                    pred_cal.all_values(), pred_vals_expected
                )

        # checking that last points only is equal to the last forecasted point
        hfc_lpo_list = model.historical_forecasts(
            series=series,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=True,
            stride=1,
            **covs_kwargs,
            **pred_lklp,
        )
        if is_single:
            hfc_lpo_list = [hfc_lpo_list]

        for hfc_lpo, hfc_conf in zip(hfc_lpo_list, hfc_conf_list):
            hfc_conf_lpo = concatenate([hfc[-1:] for hfc in hfc_conf], axis=0)
            assert hfc_lpo == hfc_conf_lpo

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [1, 3, 5],  # horizon
            [0, 1],  # output chunk shift
            [None, 1],  # cal length,
            [1, 2],  # cal stride
            [False, True],  # use start
        ),
    )
    def test_stridden_conformal_model(self, config):
        """Checks correctness of naive conformal model historical forecasts for:
        - different horizons (smaller, equal and larger the OCL)
        - uni and multivariate series
        - single and multiple series
        - with and without output shift
        - with and without training length
        - with and without covariates
        """
        is_univar, is_single = True, False
        n, ocs, cal_length, cal_stride, use_start = config
        if ocs and n > OUT_LEN:
            # auto-regression not allowed with ocs
            return

        series = self.helper_prepare_series(is_univar, is_single)
        # shift second series ahead to cover the non overlapping multi series case
        series = [series[0], series[1].shift(120)]
        model_params = {"output_chunk_shift": ocs}

        # forecasts from forecasting model
        model_fc = train_model(series, model_params=model_params)
        hfc_fc_list = model_fc.historical_forecasts(
            series,
            retrain=False,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=False,
            stride=cal_stride,
        )
        # residuals to compute the conformal intervals
        residuals_list = model_fc.residuals(
            series,
            historical_forecasts=hfc_fc_list,
            overlap_end=True,
            last_points_only=False,
            values_only=True,
            metric=ae,  # absolute error
        )

        # conformal forecasts
        model = ConformalNaiveModel(
            model=model_fc,
            quantiles=q,
            cal_length=cal_length,
            cal_stride=cal_stride,
        )
        # the expected positional index of the first conformal forecast
        # index = (skip n + ocs points (relative to cal_stride) to avoid look-ahead bias) + (number of cal examples)
        first_fc_idx = math.ceil((n + ocs) / cal_stride) + (
            cal_length - 1 if cal_length else 0
        )
        first_start = n_steps_between(
            hfc_fc_list[0][first_fc_idx].start_time() - ocs * series[0].freq,
            series[0].start_time(),
            freq=series[0].freq,
        )

        hfc_conf_list = model.historical_forecasts(
            series=series,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=False,
            start=first_start if use_start else None,
            start_format="position" if use_start else "value",
            stride=cal_stride,
            **pred_lklp,
        )

        # also, skip some residuals from output chunk shift
        ignore_ocs = math.ceil(ocs / cal_stride) if ocs >= cal_stride else 0
        for hfc_fc, hfc_conf, hfc_residuals in zip(
            hfc_fc_list, hfc_conf_list, residuals_list
        ):
            for idx, (pred_fc, pred_cal) in enumerate(
                zip(hfc_fc[first_fc_idx:], hfc_conf)
            ):
                residuals = np.concatenate(
                    hfc_residuals[: first_fc_idx - ignore_ocs + idx], axis=2
                )
                pred_vals = pred_fc.all_values()
                pred_vals_expected = self.helper_compute_pred_cal(
                    residuals,
                    pred_vals,
                    n,
                    q,
                    cal_length=cal_length,
                    model_type="regression",
                    symmetric=True,
                    cal_stride=cal_stride,
                )
                assert pred_fc.time_index.equals(pred_cal.time_index)
                np.testing.assert_array_almost_equal(
                    pred_cal.all_values(), pred_vals_expected
                )

        # check that with a round-multiple of `cal_stride` we get identical forecasts
        assert model.historical_forecasts(
            series=series,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=False,
            start=first_start if use_start else None,
            start_format="position" if use_start else "value",
            stride=2 * cal_stride,
            **pred_lklp,
        ) == [hfc[::2] for hfc in hfc_conf_list]

        # checking that last points only is equal to the last forecasted point
        hfc_lpo_list = model.historical_forecasts(
            series=series,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=True,
            stride=cal_stride,
            **pred_lklp,
        )
        for hfc_lpo, hfc_conf in zip(hfc_lpo_list, hfc_conf_list):
            hfc_conf_lpo = concatenate(
                [hfc[-1::cal_stride] for hfc in hfc_conf], axis=0
            )
            assert hfc_lpo == hfc_conf_lpo

        # checking that predict gives the same results as last historical forecast
        preds = model.predict(
            series=series,
            n=n,
            **pred_lklp,
        )
        hfcs_conf_end = model.historical_forecasts(
            series=series,
            forecast_horizon=n,
            overlap_end=True,
            last_points_only=False,
            start=-cal_stride,
            start_format="position",
            stride=cal_stride,
            **pred_lklp,
        )
        hfcs_conf_end = [hfc[-1] for hfc in hfcs_conf_end]
        for pred, last_hfc in zip(preds, hfcs_conf_end):
            assert pred == last_hfc

    def test_probabilistic_historical_forecast(self):
        """Checks correctness of naive conformal historical forecast from probabilistic fc model compared to
        deterministic one,
        """
        series = self.helper_prepare_series(False, False)
        # forecasts from forecasting model
        model_det = ConformalNaiveModel(
            train_model(series, model_type="regression", quantiles=q),
            quantiles=q,
        )
        model_prob = ConformalNaiveModel(
            train_model(series, model_type="regression_prob", quantiles=q),
            quantiles=q,
        )
        hfcs_det = model_det.historical_forecasts(
            series,
            forecast_horizon=2,
            last_points_only=True,
            stride=1,
            **pred_lklp,
        )
        hfcs_prob = model_prob.historical_forecasts(
            series,
            forecast_horizon=2,
            last_points_only=True,
            stride=1,
            **pred_lklp,
        )
        assert isinstance(hfcs_det, list) and len(hfcs_det) == 2
        assert isinstance(hfcs_prob, list) and len(hfcs_prob) == 2
        for hfc_det, hfc_prob in zip(hfcs_det, hfcs_prob):
            assert hfc_det.columns.equals(hfc_prob.columns)
            assert hfc_det.time_index.equals(hfc_prob.time_index)
            self.helper_compare_preds(
                hfc_prob, hfc_det.all_values(), model_type="regression_prob"
            )

    def helper_prepare_series(self, is_univar, is_single):
        series = self.ts_pass_train
        if not is_univar:
            series = series.stack(series + 3.0)
        if not is_single:
            series = [series, series + 5]
        return series

    @staticmethod
    def helper_compare_preds(cp_pred, pred_expected, model_type, tol_rel=0.1):
        if isinstance(cp_pred, TimeSeries):
            cp_pred = cp_pred.all_values(copy=False)
        if model_type == "regression":
            # deterministic fc model should give almost identical results
            np.testing.assert_array_almost_equal(cp_pred, pred_expected)
        else:
            # probabilistic fc models have some randomness
            diffs_rel = np.abs((cp_pred - pred_expected) / pred_expected)
            assert (diffs_rel < tol_rel).all().all()

    @staticmethod
    def helper_compute_pred_cal(
        residuals,
        pred_vals,
        horizon,
        quantiles,
        model_type,
        symmetric,
        cal_length=None,
        cal_stride=1,
    ):
        """Generates expected prediction results for naive conformal model from:

        - residuals and predictions from deterministic/probabilistic model
        - any forecast horizon
        - any quantile intervals
        - symmetric/ asymmetric non-conformity scores
        - any train length
        """
        cal_length = cal_length or 0
        n_comps = pred_vals.shape[1]
        half_idx = len(quantiles) // 2

        # get alphas from quantiles (alpha = q_hi - q_lo) per interval
        alphas = np.array(quantiles[half_idx + 1 :][::-1]) - np.array(
            quantiles[:half_idx]
        )
        if not symmetric:
            # asymmetric non-conformity scores look only on one tail -> alpha/2
            alphas = 1 - (1 - alphas) / 2
        if model_type == "regression_prob":
            # naive conformal model converts probabilistic forecasts to median (deterministic)
            pred_vals = np.expand_dims(np.quantile(pred_vals, 0.5, axis=2), -1)
        elif model_type == "regression_qr":
            # conformalized quantile regression consumes quantile forecasts
            pred_vals = np.quantile(pred_vals, quantiles, axis=2).transpose(1, 2, 0)

        is_naive = model_type in ["regression", "regression_prob"]
        pred_expected = []
        for alpha_idx, alpha in enumerate(alphas):
            q_hats = []
            # compute the quantile `alpha` of all past residuals (absolute "per time step" errors between historical
            # forecasts and the target series)
            for idx_horizon in range(horizon):
                n = idx_horizon + 1
                # ignore residuals at beginning
                idx_fc_start = math.floor((horizon - n) / cal_stride)
                # keep as many residuals as possible from end
                idx_fc_end = -(math.ceil(horizon / cal_stride) - (idx_fc_start + 1))
                res_n = residuals[idx_horizon, :, idx_fc_start : idx_fc_end or None]
                if cal_length is not None:
                    res_n = res_n[:, -cal_length:]
                if is_naive and symmetric:
                    # identical correction for upper and lower bounds
                    # metric is `ae()`
                    q_hat_n = np.quantile(res_n, q=alpha, method="higher", axis=1)
                    q_hats.append((-q_hat_n, q_hat_n))
                elif is_naive:
                    # correction separately for upper and lower bounds
                    # metric is `err()`
                    q_hat_hi = np.quantile(res_n, q=alpha, method="higher", axis=1)
                    q_hat_lo = np.quantile(-res_n, q=alpha, method="higher", axis=1)
                    q_hats.append((-q_hat_lo, q_hat_hi))
                elif symmetric:  # CQR symmetric
                    # identical correction for upper and lower bounds
                    # metric is `incs_qr(symmetric=True)`
                    q_hat_n = np.quantile(res_n, q=alpha, method="higher", axis=1)
                    q_hats.append((-q_hat_n, q_hat_n))
                else:  # CQR asymmetric
                    # correction separately for upper and lower bounds
                    # metric is `incs_qr(symmetric=False)`
                    half_idx = len(res_n) // 2

                    # residuals have shape (n components * n intervals * 2)
                    # the factor 2 comes from the metric being computed for lower, and upper bounds separately
                    # (comp_1_qlow_1, comp_1_qlow_2, ... comp_n_qlow_m, comp_1_qhigh_1, ...)
                    q_hat_lo = np.quantile(
                        res_n[:half_idx], q=alpha, method="higher", axis=1
                    )
                    q_hat_hi = np.quantile(
                        res_n[half_idx:], q=alpha, method="higher", axis=1
                    )
                    q_hats.append((
                        -q_hat_lo[alpha_idx :: len(alphas)],
                        q_hat_hi[alpha_idx :: len(alphas)],
                    ))
            # bring to shape (horizon, n components, 2)
            q_hats = np.array(q_hats).transpose((0, 2, 1))
            # the prediction interval is given by pred +/- q_hat
            pred_vals_expected = []
            for col_idx in range(n_comps):
                q_col = q_hats[:, col_idx]
                pred_col = pred_vals[:, col_idx]
                if is_naive:
                    # conformal model corrects deterministic predictions
                    idx_q_lo = slice(0, None)
                    idx_q_med = slice(0, None)
                    idx_q_hi = slice(0, None)
                else:
                    # conformal model corrects quantile predictions
                    idx_q_lo = slice(alpha_idx, alpha_idx + 1)
                    idx_q_med = slice(len(alphas), len(alphas) + 1)
                    idx_q_hi = slice(
                        pred_col.shape[1] - (alpha_idx + 1),
                        pred_col.shape[1] - alpha_idx,
                    )
                # correct lower and upper bounds
                pred_col_expected = np.concatenate(
                    [
                        pred_col[:, idx_q_lo] + q_col[:, :1],  # lower quantile
                        pred_col[:, idx_q_med],  # median forecast
                        pred_col[:, idx_q_hi] + q_col[:, 1:],
                    ],  # upper quantile
                    axis=1,
                )
                pred_col_expected = np.expand_dims(pred_col_expected, 1)
                pred_vals_expected.append(pred_col_expected)
            pred_vals_expected = np.concatenate(pred_vals_expected, axis=1)
            pred_expected.append(pred_vals_expected)

        # reorder to have columns going from lowest quantiles to highest per component
        pred_expected_reshaped = []
        for comp_idx in range(n_comps):
            for q_idx in [0, 1, 2]:
                for pred_idx in range(len(pred_expected)):
                    # upper quantiles will have reversed order
                    if q_idx == 2:
                        pred_idx = len(pred_expected) - 1 - pred_idx
                    pred_ = pred_expected[pred_idx][:, comp_idx, q_idx]
                    pred_ = pred_.reshape(-1, 1, 1)

                    # q_hat_idx = q_idx + comp_idx * 3 + alpha_idx * 3 * n_comps
                    pred_expected_reshaped.append(pred_)
                    # only add median quantile once
                    if q_idx == 1:
                        break
        return np.concatenate(pred_expected_reshaped, axis=1)

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
        covs_kwargs_train = {}
        covs_kwargs_too_short = {}
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

        model = ConformalNaiveModel(
            train_model(
                series=series_train,
                model_params=model_params,
                **covs_kwargs_train,
            ),
            quantiles=q,
        )

        # prediction works with long enough input
        preds1 = model.predict(n=n, series=series, **covs_kwargs)
        assert not np.isnan(preds1.all_values()).any().any()

        # series too short: without covariates, make `series` shorter. Otherwise, use the shorter covariates
        series_ = series[:-1] if not use_covs else series
        with pytest.raises(ValueError) as exc:
            _ = model.predict(n=n, series=series_, **covs_kwargs_too_short)
        if not use_covs:
            assert str(exc.value).startswith(
                "Could not build the minimum required calibration input with the provided `series`"
            )
        else:
            # if `past_covariates` are too short, then it raises error from the forecasting_model.predict()
            assert str(exc.value).startswith(
                "The `past_covariates` are not long enough."
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
            cal_length,
            ocs,
            n,
            use_covs,
        ) = config
        if ocs and n > OUT_LEN:
            return

        icl = IN_LEN
        ocl = OUT_LEN
        horizon_ocs = n + ocs
        add_cal_length = cal_length - 1 if cal_length is not None else 0
        # min length to generate 1 conformal forecast
        min_len_val_series = (
            icl + horizon_ocs * (1 + int(not overlap_end)) + add_cal_length
        )

        series_train = [tg.linear_timeseries(length=icl + ocl + ocs)] * 2
        series = tg.linear_timeseries(length=min_len_val_series)

        model_params = {"output_chunk_shift": ocs}
        covs_kwargs_train = {}
        covs_kwargs = {}
        covs_kwargs_short = {}
        if use_covs:
            model_params["lags_past_covariates"] = regr_kwargs["lags"]
            covs_kwargs_train["past_covariates"] = series_train

            # `- horizon_ocs` to generate forecasts extending up until end of target series
            if not overlap_end:
                past_covs = series[:-horizon_ocs]
            else:
                past_covs = series

            # for auto-regression, we require longer past covariates
            if n > OUT_LEN:
                past_covs = past_covs.append_values([1.0] * (n - OUT_LEN))

            # covariates lengths to generate exactly one forecast
            covs_kwargs["past_covariates"] = past_covs

            # use too short covariates to check that errors are raised
            covs_kwargs_short["past_covariates"] = covs_kwargs["past_covariates"][:-1]

        model = ConformalNaiveModel(
            train_model(
                series=series_train,
                model_params=model_params,
                **covs_kwargs_train,
            ),
            quantiles=q,
            cal_length=cal_length,
        )

        hfc_kwargs = {
            "last_points_only": last_points_only,
            "overlap_end": overlap_end,
            "forecast_horizon": n,
        }
        # prediction works with long enough input
        hfcs = model.historical_forecasts(
            series=series,
            **covs_kwargs,
            **hfc_kwargs,
        )
        if last_points_only:
            hfcs = [hfcs]

        assert len(hfcs) == 1
        for hfc in hfcs:
            assert not np.isnan(hfc.all_values()).any().any()

        # input too short: without covariates, make `series` shorter. Otherwise, use the shorter covariates
        series_ = series[:-1] if not use_covs else series
        with pytest.raises(ValueError) as exc:
            _ = model.historical_forecasts(
                series=series_,
                **covs_kwargs_short,
                **hfc_kwargs,
            )
        assert str(exc.value).startswith(
            "Could not build the minimum required calibration input with the provided `series` and `*_covariates`"
        )

    @pytest.mark.parametrize("quantiles", [[0.1, 0.5, 0.9], [0.1, 0.3, 0.5, 0.7, 0.9]])
    def test_backtest_and_residuals(self, quantiles):
        """Residuals and backtest are already tested for quantile, and interval metrics based on stochastic or quantile
        forecasts. So, a simple check that they give expected results should be enough.
        """
        n_q = len(quantiles)
        half_idx = n_q // 2
        q_interval = [
            (q_lo, q_hi)
            for q_lo, q_hi in zip(quantiles[:half_idx], quantiles[half_idx + 1 :][::-1])
        ]
        lpo = False

        # series long enough for 2 hfcs
        series = self.helper_prepare_series(True, True).append_values([0.1])
        # conformal model
        model = ConformalNaiveModel(model=train_model(series), quantiles=quantiles)

        hfc = model.historical_forecasts(
            series=series, forecast_horizon=5, last_points_only=lpo, **pred_lklp
        )
        bt = model.backtest(
            series=series,
            historical_forecasts=hfc,
            last_points_only=lpo,
            metric=mic,
            metric_kwargs={"q_interval": model.q_interval},
        )
        # default backtest is equal to backtest with metric kwargs
        np.testing.assert_array_almost_equal(
            bt,
            model.backtest(
                series=series,
                historical_forecasts=hfc,
                last_points_only=lpo,
                metric=mic,
                metric_kwargs={"q_interval": q_interval},
            ),
        )
        np.testing.assert_array_almost_equal(
            mic(
                [series] * len(hfc),
                hfc,
                q_interval=q_interval,
                series_reduction=np.mean,
            ),
            bt,
        )

        residuals = model.residuals(
            series=series,
            historical_forecasts=hfc,
            last_points_only=lpo,
            metric=ic,
            metric_kwargs={"q_interval": q_interval},
        )
        # default residuals is equal to residuals with metric kwargs
        assert residuals == model.residuals(
            series=series,
            historical_forecasts=hfc,
            last_points_only=lpo,
            metric=ic,
            metric_kwargs={"q_interval": q_interval},
        )
        expected_vals = ic([series] * len(hfc), hfc, q_interval=q_interval)
        expected_residuals = []
        for vals, hfc_ in zip(expected_vals, hfc):
            expected_residuals.append(
                TimeSeries.from_times_and_values(
                    times=hfc_.time_index,
                    values=vals,
                    columns=likelihood_component_names(
                        series.components, quantile_interval_names(q_interval)
                    ),
                )
            )
        assert residuals == expected_residuals

    def test_predict_probabilistic_equals_quantile(self):
        """Tests that sampled quantiles predictions have approx. the same quantiles as direct quantile predictions."""
        quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]

        # multiple multivariate series
        series = self.helper_prepare_series(False, False)

        # conformal model
        model = ConformalNaiveModel(model=train_model(series), quantiles=quantiles)
        # direct quantile predictions
        pred_quantiles = model.predict(n=3, series=series, **pred_lklp)
        # sampled predictions
        pred_samples = model.predict(n=3, series=series, num_samples=500)
        for pred_q, pred_s in zip(pred_quantiles, pred_samples):
            assert pred_q.n_samples == 1
            assert pred_q.n_components == series[0].n_components * len(quantiles)
            assert pred_s.n_samples == 500
            assert pred_s.n_components == series[0].n_components

            vals_q = pred_q.all_values()
            vals_s = pred_s.all_values()
            vals_s_q = np.quantile(vals_s, quantiles, axis=2).transpose((1, 2, 0))
            vals_s_q = vals_s_q.reshape(vals_q.shape)
            self.helper_compare_preds(
                vals_s_q,
                vals_q,
                model_type="regression_prob",
            )

    @pytest.mark.parametrize(
        "config",
        [
            # (cal_length, cal_stride, (start_expected, start_format_expected))
            (None, 1, (None, "value")),
            (None, 2, (-4, "position")),
            (None, 3, (-6, "position")),
            (None, 4, (-4, "position")),
            (1, 1, (-3, "position")),
            (1, 2, (-4, "position")),
            (1, 3, (-3, "position")),
            (1, 4, (-4, "position")),
        ],
    )
    def test_calibration_hfc_start_predict(self, config):
        """Test calibration historical forecast start point when calling `predict()` ("end" position)."""
        cal_length, cal_stride, start_expected = config
        series = linear_timeseries(length=4)
        horizon = 2
        output_chunk_shift = 1
        assert (
            _get_calibration_hfc_start(
                series=[series],
                horizon=horizon,
                output_chunk_shift=output_chunk_shift,
                cal_length=cal_length,
                cal_stride=cal_stride,
                start="end",
                start_format="position",
            )
            == start_expected
        )

    @pytest.mark.parametrize(
        "config",
        [
            # (cal_length, cal_stride, start, start_expected)
            (None, 1, None, None),
            (None, 1, 1, None),
            (1, 1, -1, -4),
            (1, 1, 0, 0),
            (1, 2, 0, 0),
            (1, 3, 0, 0),
            (1, 1, 1, 0),
            (1, 2, 1, 1),
            (1, 3, 1, 1),
            (1, 1, -1, -4),
            (1, 2, -1, -5),
            (1, 3, -1, -4),
        ],
    )
    def test_calibration_hfc_start_position_hist_fc(self, config):
        """Test calibration historical forecast start point when calling `historical_forecasts()`
        with start format "position"."""
        cal_length, cal_stride, start, start_expected = config
        series = linear_timeseries(length=4)
        horizon = 2
        output_chunk_shift = 1
        assert _get_calibration_hfc_start(
            series=[series],
            horizon=horizon,
            output_chunk_shift=output_chunk_shift,
            cal_length=cal_length,
            cal_stride=cal_stride,
            start=start,
            start_format="position",
        ) == (start_expected, "position")

    @pytest.mark.parametrize(
        "config",
        [
            # (cal_length, cal_stride, start, start_expected)
            (None, 1, None, None),
            (None, 1, "2020-01-11", None),
            (1, 1, "2020-01-09", "2020-01-06"),  # start before series start
            (1, 1, "2020-01-10", "2020-01-07"),
            (1, 2, "2020-01-10", "2020-01-06"),
            (1, 3, "2020-01-10", "2020-01-07"),
            (2, 1, "2020-01-09", "2020-01-05"),
            (2, 1, "2020-01-10", "2020-01-06"),
            (2, 2, "2020-01-10", "2020-01-04"),
            (2, 3, "2020-01-10", "2020-01-04"),
        ],
    )
    def test_calibration_hfc_start_value_hist_fc(self, config):
        """Test calibration historical forecast start point when calling `historical_forecasts()`
        with start format "value"."""
        cal_length, cal_stride, start, start_expected = config
        if start is not None:
            start = pd.Timestamp(start)
        if start_expected is not None:
            start_expected = pd.Timestamp(start_expected)
        series = linear_timeseries(length=4, start=pd.Timestamp("2020-01-10"), freq="d")
        horizon = 2
        output_chunk_shift = 1
        assert _get_calibration_hfc_start(
            series=[series],
            horizon=horizon,
            output_chunk_shift=output_chunk_shift,
            cal_length=cal_length,
            cal_stride=cal_stride,
            start=start,
            start_format="value",
        ) == (start_expected, "value")

    def test_encoders(self):
        """Tests support of covariates encoders."""
        n = OUT_LEN + 1
        min_length = IN_LEN + n

        # create non-overlapping train and val series
        series = tg.linear_timeseries(length=min_length)
        val_series = tg.linear_timeseries(
            start=series.end_time() + series.freq, length=min_length
        )

        model = train_model(
            series,
            model_params={
                "lags_future_covariates": (IN_LEN, OUT_LEN),
                "add_encoders": {"datetime_attribute": {"future": ["hour"]}},
            },
        )

        cp_model = ConformalNaiveModel(model, quantiles=q)
        assert (
            cp_model.model.encoders is not None
            and cp_model.model.encoders.encoding_available
        )
        assert model.uses_future_covariates

        # predict: encoders using stored train series must work
        _ = cp_model.predict(n=n)
        # predict: encoding of new series without train overlap must work
        _ = cp_model.predict(n=n, series=val_series)

        # check the same for hfc
        _ = cp_model.historical_forecasts(
            forecast_horizon=n, series=series, overlap_end=True
        )
        _ = cp_model.historical_forecasts(
            forecast_horizon=n, series=val_series, overlap_end=True
        )
