import functools
import itertools

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.logging import get_logger
from darts.models import (
    ConformalNaiveModel,
    LinearRegressionModel,
    NaiveSeasonal,
)
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)


def train_test_split(series, split_ts):
    """
    Splits all provided TimeSeries instances into train and test sets according to the provided timestamp.

    Parameters
    ----------
    features : TimeSeries
        Feature TimeSeries instances to be split.
    target : TimeSeries
        Target TimeSeries instance to be split.
    split_ts : TimeStamp
        Time stamp indicating split point.

    Returns
    -------
    TYPE
        4-tuple of the form (train_features, train_target, test_features, test_target)
    """
    if isinstance(series, TimeSeries):
        return series.split_after(split_ts)
    else:
        return list(zip(*[ts.split_after(split_ts) for ts in series]))


def dummy_timeseries(
    length,
    n_series=1,
    comps_target=1,
    comps_pcov=1,
    comps_fcov=1,
    multiseries_offset=0,
    pcov_offset=0,
    fcov_offset=0,
    comps_stride=100,
    type_stride=10000,
    series_stride=1000000,
    target_start_value=1,
    first_target_start_date=pd.Timestamp("2000-01-01"),
    freq="D",
    integer_index=False,
):
    targets, pcovs, fcovs = [], [], []
    for series_idx in range(n_series):
        target_start_date = (
            series_idx * multiseries_offset
            if integer_index
            else first_target_start_date
            + pd.Timedelta(series_idx * multiseries_offset, unit=freq)
        )
        pcov_start_date = (
            target_start_date + pcov_offset
            if integer_index
            else target_start_date + pd.Timedelta(pcov_offset, unit=freq)
        )
        fcov_start_date = (
            target_start_date + fcov_offset
            if integer_index
            else target_start_date + pd.Timedelta(fcov_offset, unit=freq)
        )

        target_start_val = target_start_value + series_stride * series_idx
        pcov_start_val = target_start_val + type_stride
        fcov_start_val = target_start_val + 2 * type_stride

        target_ts = None
        pcov_ts = None
        fcov_ts = None

        for idx in range(comps_target):
            start = target_start_val + idx * comps_stride
            curr_ts = tg.linear_timeseries(
                start_value=start,
                end_value=start + length - 1,
                start=target_start_date,
                length=length,
                freq=freq,
                column_name=f"{series_idx}-trgt-{idx}",
            )
            target_ts = target_ts.stack(curr_ts) if target_ts else curr_ts
        for idx in range(comps_pcov):
            start = pcov_start_val + idx * comps_stride
            curr_ts = tg.linear_timeseries(
                start_value=start,
                end_value=start + length - 1,
                start=pcov_start_date,
                length=length,
                freq=freq,
                column_name=f"{series_idx}-pcov-{idx}",
            )
            pcov_ts = pcov_ts.stack(curr_ts) if pcov_ts else curr_ts
        for idx in range(comps_fcov):
            start = fcov_start_val + idx * comps_stride
            curr_ts = tg.linear_timeseries(
                start_value=start,
                end_value=start + length - 1,
                start=fcov_start_date,
                length=length,
                freq=freq,
                column_name=f"{series_idx}-fcov-{idx}",
            )
            fcov_ts = fcov_ts.stack(curr_ts) if fcov_ts else curr_ts

        targets.append(target_ts)
        pcovs.append(pcov_ts)
        fcovs.append(fcov_ts)

    return targets, pcovs, fcovs


# helper function used to register LightGBMModel/LinearRegressionModel with likelihood
def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


class TestRegressionModels:
    np.random.seed(42)
    # default regression models
    models = [LinearRegressionModel]

    # register likelihood regression models
    QuantileLinearRegressionModel = partialclass(
        LinearRegressionModel,
        likelihood="quantile",
        quantiles=[0.05, 0.5, 0.95],
        random_state=42,
    )
    # targets for poisson regression must be positive, so we exclude them for some tests
    models.extend([
        QuantileLinearRegressionModel,
    ])

    univariate_accuracies = [
        1e-13,  # LinearRegressionModel
        0.8,  # QuantileLinearRegressionModel
    ]
    multivariate_accuracies = [
        1e-13,  # LinearRegressionModel
        0.8,  # QuantileLinearRegressionModel
    ]
    multivariate_multiseries_accuracies = [
        1e-13,  # LinearRegressionModel
        0.8,  # QuantileLinearRegressionModel
    ]

    # dummy feature and target TimeSeries instances
    target_series, past_covariates, future_covariates = dummy_timeseries(
        length=100,
        n_series=3,
        comps_target=3,
        comps_pcov=2,
        comps_fcov=1,
        multiseries_offset=10,
        pcov_offset=0,
        fcov_offset=0,
    )
    # shift sines to positive values for poisson regressors
    sine_univariate1 = tg.sine_timeseries(length=100) + 1.5
    sine_univariate2 = tg.sine_timeseries(length=100, value_phase=1.5705) + 1.5
    sine_univariate3 = tg.sine_timeseries(length=100, value_phase=0.78525) + 1.5
    sine_univariate4 = tg.sine_timeseries(length=100, value_phase=0.392625) + 1.5
    sine_univariate5 = tg.sine_timeseries(length=100, value_phase=0.1963125) + 1.5
    sine_univariate6 = tg.sine_timeseries(length=100, value_phase=0.09815625) + 1.5
    sine_multivariate1 = sine_univariate1.stack(sine_univariate2)
    sine_multivariate2 = sine_univariate2.stack(sine_univariate3)
    sine_multiseries1 = [sine_univariate1, sine_univariate2, sine_univariate3]
    sine_multiseries2 = [sine_univariate4, sine_univariate5, sine_univariate6]

    lags_1 = {"target": [-3, -2, -1], "past": [-4, -2], "future": [-5, 2]}

    def test_model_construction(self):
        local_model = NaiveSeasonal(K=5)
        global_model = LinearRegressionModel(lags=5, output_chunk_length=1)
        series = self.target_series[0][:10]

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

    @pytest.mark.parametrize("model_cls", models)
    def test_predict_runnability(self, model_cls):
        # testing lags_past_covariates None but past_covariates during prediction
        model_instance = model_cls(lags=4, lags_past_covariates=None)
        model_instance.fit(self.sine_univariate1)
        model = ConformalNaiveModel(model_instance, alpha=0.8)
        # cannot pass past covariates
        with pytest.raises(ValueError):
            model.predict(
                n=1,
                series=self.sine_univariate1,
                past_covariates=self.sine_multivariate1,
            )
        # works without covariates
        model.predict(n=1, series=self.sine_univariate1)

        # testing lags_past_covariates but no past_covariates during prediction
        model_instance = model_cls(lags=4, lags_past_covariates=3)
        # make multi series fit so no training set is stored
        model_instance.fit(
            [self.sine_univariate1] * 2, past_covariates=[self.sine_univariate1] * 2
        )
        model = ConformalNaiveModel(model_instance, alpha=0.8)
        with pytest.raises(ValueError) as exc:
            model.predict(n=1, series=self.sine_univariate1)
        assert (
            str(exc.value) == "The model has been trained with past covariates. "
            "Some matching past_covariates have to be provided to `predict()`."
        )
        # works with covariates
        model.predict(
            n=1, series=self.sine_univariate1, past_covariates=self.sine_univariate1
        )
        # too short covariates
        with pytest.raises(ValueError) as exc:
            model.predict(
                n=1,
                series=self.sine_univariate1,
                past_covariates=self.sine_univariate1[:-1],
            )
        assert str(exc.value).startswith(
            "The `past_covariates` at list/sequence index 0 are not long enough."
        )

        # testing lags_future_covariates but no future_covariates during prediction
        model_instance = model_cls(lags=4, lags_future_covariates=(3, 0))
        # make multi series fit so no training set is stored
        model_instance.fit(
            [self.sine_univariate1] * 2, future_covariates=[self.sine_univariate1] * 2
        )
        model = ConformalNaiveModel(model_instance, alpha=0.8)
        with pytest.raises(ValueError) as exc:
            model.predict(n=1, series=self.sine_univariate1)
        assert (
            str(exc.value) == "The model has been trained with future covariates. "
            "Some matching future_covariates have to be provided to `predict()`."
        )
        # works with covariates
        model.predict(
            n=1, series=self.sine_univariate1, future_covariates=self.sine_univariate1
        )
        with pytest.raises(ValueError) as exc:
            model.predict(
                n=1,
                series=self.sine_univariate1,
                future_covariates=self.sine_univariate1[:-1],
            )
        assert str(exc.value).startswith(
            "The `future_covariates` at list/sequence index 0 are not long enough."
        )

        # test input dim
        model_instance = model_cls(lags=4)
        model_instance.fit(self.sine_univariate1)
        model = ConformalNaiveModel(model_instance, alpha=0.8)
        with pytest.raises(ValueError) as exc:
            model.predict(
                n=1, series=self.sine_univariate1.stack(self.sine_univariate1)
            )
        assert str(exc.value).startswith(
            "The number of components of the target series"
        )

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [True, False],  # univariate series
            [True, False],  # single series
            [True, False],  # use covariates
            [True, False],  # datetime index
            [3, 5, 7],  # different horizons
        ),
    )
    def test_predict(self, config):
        (is_univar, is_single, use_covs, is_datetime, horizon) = config

        icl = 3
        ocl = 5
        series = self.sine_univariate1[:10]
        if not is_univar:
            series = series.stack(series)
        if not is_datetime:
            series = TimeSeries.from_values(series.all_values(), columns=series.columns)
        if use_covs:
            pc, fc = series, series
            fc = fc.append_values(fc.values()[: max(horizon, ocl)])
            if horizon > ocl:
                pc = pc.append_values(pc.values()[: horizon - ocl])
            model_kwargs = {
                "lags_past_covariates": icl,
                "lags_future_covariates": (icl, ocl),
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
            lags=icl, output_chunk_length=ocl, **model_kwargs
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
                cols_expected += [f"{col}_q_{q}" for q in ["lo", "md", "hi"]]
            assert preds_.columns.tolist() == cols_expected
            assert len(preds_) == horizon
            assert preds_.start_time() == s_.end_time() + s_.freq
            assert preds_.freq == s_.freq
