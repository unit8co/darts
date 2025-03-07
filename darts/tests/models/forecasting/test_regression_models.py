import functools
import importlib
import inspect
import logging
import math
from copy import deepcopy
from itertools import product
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

import darts
from darts import TimeSeries
from darts.dataprocessing.encoders import (
    FutureCyclicEncoder,
    PastDatetimeAttributeEncoder,
)
from darts.logging import get_logger
from darts.metrics import mae, rmse
from darts.models import (
    CatBoostModel,
    LightGBMModel,
    LinearRegressionModel,
    NotImportedModule,
    RandomForest,
    RegressionModel,
    XGBModel,
)
from darts.utils import timeseries_generation as tg
from darts.utils.multioutput import MultiOutputRegressor
from darts.utils.utils import generate_index

logger = get_logger(__name__)

# replace catboost and lgbm with xgb in case of core requirements
cb_available = not isinstance(CatBoostModel, NotImportedModule)
lgbm_available = not isinstance(LightGBMModel, NotImportedModule)


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


xgb_test_params = {
    "n_estimators": 1,
    "max_depth": 1,
    "max_leaves": 1,
    "random_state": 42,
}
lgbm_test_params = {
    "n_estimators": 1,
    "max_depth": 1,
    "num_leaves": 2,
    "verbosity": -1,
    "random_state": 42,
}
cb_test_params = {
    "iterations": 1,
    "depth": 1,
    "verbose": -1,
    "random_state": 42,
}


class TestRegressionModels:
    np.random.seed(42)
    # default regression models
    models = [RandomForest, LinearRegressionModel, RegressionModel]

    # register likelihood regression models
    QuantileLinearRegressionModel = partialclass(
        LinearRegressionModel,
        likelihood="quantile",
        quantiles=[0.05, 0.5, 0.95],
        random_state=42,
    )
    PoissonLinearRegressionModel = partialclass(
        LinearRegressionModel, likelihood="poisson", random_state=42
    )
    PoissonXGBModel = partialclass(
        XGBModel,
        likelihood="poisson",
        tree_method="exact",
        **xgb_test_params,
    )
    QuantileXGBModel = partialclass(
        XGBModel,
        likelihood="quantile",
        tree_method="exact",
        **xgb_test_params,
    )
    KNeighborsRegressorModel = partialclass(
        RegressionModel,
        model=KNeighborsRegressor(n_neighbors=1),
    )
    # targets for poisson regression must be positive, so we exclude them for some tests
    models.extend([
        QuantileLinearRegressionModel,
        PoissonLinearRegressionModel,
        PoissonXGBModel,
        QuantileXGBModel,
    ])

    univariate_accuracies = [
        0.03,  # RandomForest
        1e-13,  # LinearRegressionModel
        1e-13,  # RegressionModel
        0.8,  # QuantileLinearRegressionModel
        0.4,  # PoissonLinearRegressionModel
        0.75,  # PoissonXGBModel
        0.75,  # QuantileXGBModel
    ]
    multivariate_accuracies = [
        0.3,  # RandomForest
        1e-13,  # LinearRegressionModel
        1e-13,  # RegressionModel
        0.8,  # QuantileLinearRegressionModel
        0.4,  # PoissonLinearRegressionModel
        0.75,  # PoissonXGBModel
        0.75,  # QuantileXGBModel
    ]
    multivariate_multiseries_accuracies = [
        0.05,  # RandomForest
        1e-13,  # LinearRegressionModel
        1e-13,  # RegressionModel
        0.8,  # QuantileLinearRegressionModel
        0.4,  # PoissonLinearRegressionModel
        0.85,  # PoissonXGBModel
        0.65,  # QuantileXGBModel
    ]

    lgbm_w_categorical_covariates = NotImportedModule
    if lgbm_available:
        RegularLightGBMModel = partialclass(LightGBMModel, **lgbm_test_params)
        QuantileLightGBMModel = partialclass(
            LightGBMModel,
            likelihood="quantile",
            quantiles=[0.05, 0.5, 0.95],
            **lgbm_test_params,
        )
        PoissonLightGBMModel = partialclass(
            LightGBMModel,
            likelihood="poisson",
            **lgbm_test_params,
        )
        models += [
            RegularLightGBMModel,
            QuantileLightGBMModel,
            PoissonLightGBMModel,
        ]

        lgbm_w_categorical_covariates = LightGBMModel(
            lags=1,
            lags_past_covariates=1,
            lags_future_covariates=[1],
            output_chunk_length=1,
            categorical_future_covariates=["fut_cov_promo_mechanism"],
            categorical_past_covariates=["past_cov_cat_dummy"],
            categorical_static_covariates=["product_id"],
            **lgbm_test_params,
        )
        univariate_accuracies += [
            0.75,  # LightGBMModel
            0.75,  # QuantileLightGBMModel
            0.75,  # PoissonLightGBMModel
        ]
        multivariate_accuracies += [
            0.7,  # LightGBMModel
            0.75,  # QuantileLightGBMModel
            0.75,  # PoissonLightGBMModel
        ]
        multivariate_multiseries_accuracies += [
            0.7,  # LightGBMModel
            0.7,  # QuantileLightGBMModel
            0.75,  # PoissonLightGBMModel
        ]
    if cb_available:
        RegularCatBoostModel = partialclass(
            CatBoostModel,
            **cb_test_params,
        )
        QuantileCatBoostModel = partialclass(
            CatBoostModel,
            likelihood="quantile",
            quantiles=[0.05, 0.5, 0.95],
            **cb_test_params,
        )
        PoissonCatBoostModel = partialclass(
            CatBoostModel,
            likelihood="poisson",
            **cb_test_params,
        )
        NormalCatBoostModel = partialclass(
            CatBoostModel,
            likelihood="gaussian",
            **cb_test_params,
        )
        models += [
            RegularCatBoostModel,
            QuantileCatBoostModel,
            PoissonCatBoostModel,
            NormalCatBoostModel,
        ]
        univariate_accuracies += [
            0.75,  # CatBoostModel
            0.75,  # QuantileCatBoostModel
            0.9,  # PoissonCatBoostModel
            0.75,  # NormalCatBoostModel
        ]
        multivariate_accuracies += [
            0.75,  # CatBoostModel
            0.75,  # QuantileCatBoostModel
            0.86,  # PoissonCatBoostModel
            0.75,  # NormalCatBoostModel
        ]
        multivariate_multiseries_accuracies += [
            0.75,  # CatBoostModel
            0.75,  # QuantileCatBoostModel
            1.2,  # PoissonCatBoostModel
            0.75,  # NormalCatBoostModel
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

    @property
    def inputs_for_tests_categorical_covariates(self):
        """
        Returns TimeSeries objects that can be used for testing impact of categorical covariates.

        Details:
        - series is a univariate TimeSeries with daily frequency.
        - future_covariates are a TimeSeries with 2 components. The first component represents a "promotion"
            mechanism and has an impact on the target quantiy according to 'apply_promo_mechanism'. The second
            component contains random data that should have no impact on the target quantity. Note that although the
            intention is to model the "promotion_mechanism" as a categorical variable, it is encoded as integers.
            This is required by LightGBM.
        - past_covariates are a TimeSeries with 2 components. It only contains dummy data and does not
            have any impact on the target series.
        """

        def _apply_promo_mechanism(promo_mechanism):
            if promo_mechanism == 0:
                return 0
            elif promo_mechanism == 1:
                return np.random.normal(25, 5)
            elif promo_mechanism == 2:
                return np.random.normal(5, 1)
            elif promo_mechanism == 3:
                return np.random.normal(6, 2)
            elif promo_mechanism == 4:
                return np.random.normal(50, 5)
            elif promo_mechanism == 5:
                return np.random.normal(2, 0.5)
            elif promo_mechanism == 6:
                return np.random.normal(-10, 3)
            elif promo_mechanism == 7:
                return np.random.normal(15, 3)
            elif promo_mechanism == 8:
                return np.random.normal(40, 7)
            elif promo_mechanism == 9:
                return 0
            elif promo_mechanism == 10:
                return np.random.normal(20, 3)

        date_range = pd.date_range(start="2020-01-01", end="2023-01-01", freq="D")
        df = (
            pd.DataFrame({
                "date": date_range,
                "baseline": np.random.normal(100, 10, len(date_range)),
                "fut_cov_promo_mechanism": np.random.randint(0, 11, len(date_range)),
                "fut_cov_dummy": np.random.normal(10, 2, len(date_range)),
                "past_cov_dummy": np.random.normal(10, 2, len(date_range)),
                "past_cov_cat_dummy": np.random.normal(10, 2, len(date_range)),
            })
            .assign(
                target_qty=lambda _df: _df.baseline
                + _df.fut_cov_promo_mechanism.apply(_apply_promo_mechanism)
            )
            .drop(columns=["baseline"])
        )

        series = TimeSeries.from_dataframe(
            df,
            time_col="date",
            value_cols=["target_qty"],
            static_covariates=pd.DataFrame({"product_id": [1]}),
        )
        past_covariates = TimeSeries.from_dataframe(
            df, time_col="date", value_cols=["past_cov_dummy", "past_cov_cat_dummy"]
        )
        future_covariates = TimeSeries.from_dataframe(
            df, time_col="date", value_cols=["fut_cov_promo_mechanism", "fut_cov_dummy"]
        )

        return series, past_covariates, future_covariates

    @pytest.mark.parametrize("config", product(models, [True, False]))
    def test_model_construction(self, config):
        model, mode = config
        # TESTING SINGLE INT
        # testing lags
        model_instance = model(lags=5, multi_models=mode)
        assert model_instance.lags.get("target") == [-5, -4, -3, -2, -1]
        # testing lags_past_covariates
        model_instance = model(lags=None, lags_past_covariates=3, multi_models=mode)
        assert model_instance.lags.get("past") == [-3, -2, -1]
        # lags_future covariates does not support SINGLE INT

        # TESTING TUPLE of int, only supported by lags_future_covariates
        model_instance = model(
            lags=None, lags_future_covariates=(3, 5), multi_models=mode
        )
        assert model_instance.lags.get("future") == [-3, -2, -1, 0, 1, 2, 3, 4]

        # TESTING LIST of int
        # lags
        values = [-5, -3, -1]
        model_instance = model(lags=values, multi_models=mode)
        assert model_instance.lags.get("target") == values
        # testing lags_past_covariates
        model_instance = model(lags_past_covariates=values, multi_models=mode)
        assert model_instance.lags.get("past") == values
        # testing lags_future_covariates
        values = [-5, -1, 5]
        model_instance = model(lags_future_covariates=values, multi_models=mode)
        assert model_instance.lags.get("future") == values

        # TESTING DICT, lags are specified component-wise
        # model.lags contains the extreme across the components
        values = {"comp0": [-4, -2], "comp1": [-5, -3]}
        model_instance = model(lags=values, multi_models=mode)
        assert model_instance.lags.get("target") == [-5, -2]
        assert model_instance.component_lags.get("target") == values
        # testing lags_past_covariates
        model_instance = model(lags_past_covariates=values, multi_models=mode)
        assert model_instance.lags.get("past") == [-5, -2]
        assert model_instance.component_lags.get("past") == values
        # testing lags_future_covariates
        values = {"comp0": [-4, 2], "comp1": [-5, 3]}
        model_instance = model(lags_future_covariates=values, multi_models=mode)
        assert model_instance.lags.get("future") == [-5, 3]
        assert model_instance.component_lags.get("future") == values

        with pytest.raises(ValueError):
            model(multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=0, multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=[-1, 0], multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=[3, 5], multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=[-3, -5.0], multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=-5, multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=3.6, multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=None, lags_past_covariates=False, multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=None, multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=5, lags_future_covariates=True, multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=5, lags_future_covariates=(1, -3), multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=5, lags_future_covariates=(1, 2, 3), multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=5, lags_future_covariates=(1, True), multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=5, lags_future_covariates=(1, 1.0), multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=5, lags_future_covariates={}, multi_models=mode)
        with pytest.raises(ValueError):
            model(lags=None, lags_future_covariates={}, multi_models=mode)

    @pytest.mark.parametrize("mode", [True, False])
    def test_training_data_creation(self, mode):
        """testing _get_training_data function"""
        # lags defined using lists of integers
        model_instance = RegressionModel(
            lags=self.lags_1["target"],
            lags_past_covariates=self.lags_1["past"],
            lags_future_covariates=self.lags_1["future"],
            multi_models=mode,
        )

        max_samples_per_ts = 17

        training_samples, training_labels, _ = model_instance._create_lagged_data(
            series=self.target_series,
            past_covariates=self.past_covariates,
            future_covariates=self.future_covariates,
            max_samples_per_ts=max_samples_per_ts,
        )

        # checking number of dimensions
        assert len(training_samples.shape) == 2  # samples, features
        assert len(training_labels.shape) == 2  # samples, components (multivariate)
        assert training_samples.shape[0] == training_labels.shape[0]
        assert training_samples.shape[0] == len(self.target_series) * max_samples_per_ts
        assert (
            training_samples.shape[1]
            == len(self.lags_1["target"]) * self.target_series[0].width
            + len(self.lags_1["past"]) * self.past_covariates[0].width
            + len(self.lags_1["future"]) * self.future_covariates[0].width
        )

        # check last sample
        assert list(training_samples[0, :]) == [
            79.0,
            179.0,
            279.0,
            80.0,
            180.0,
            280.0,
            81.0,
            181.0,
            281.0,
            10078.0,
            10178.0,
            10080.0,
            10180.0,
            20077.0,
            20084.0,
        ]
        assert list(training_labels[0]) == [82, 182, 282]

        # lags defined using dictionaries
        # cannot use 'default_lags' because it's converted in `fit()`, before calling `_created_lagged_data`
        model_instance = RegressionModel(
            lags={"0-trgt-0": [-4, -3], "0-trgt-1": [-3, -2], "0-trgt-2": [-2, -1]},
            lags_past_covariates={"0-pcov-0": [-10], "0-pcov-1": [-7]},
            lags_future_covariates={"0-fcov-0": (2, 2)},
            multi_models=mode,
        )

        max_samples_per_ts = 3

        # using only one series of each
        training_samples, training_labels, _ = model_instance._create_lagged_data(
            series=self.target_series[0],
            past_covariates=self.past_covariates[0],
            future_covariates=self.future_covariates[0],
            max_samples_per_ts=max_samples_per_ts,
        )

        # checking number of dimensions
        assert len(training_samples.shape) == 2  # samples, features
        assert len(training_labels.shape) == 2  # samples, components (multivariate)
        assert training_samples.shape[0] == training_labels.shape[0]
        assert training_samples.shape[0] == max_samples_per_ts
        assert (
            training_samples.shape[1]
            == 6  # [-4, -3], [-3, -2], [-2, -1]
            + 2  # [-10], [-7]
            + 4  # [-2, -1, 0, 1]
        )

        # check last sample
        assert list(training_labels[0]) == [97, 197, 297]
        # lags are grouped by components instead of lags
        assert list(training_samples[0, :]) == [
            93,
            94,
            194,
            195,
            295,
            296,  # comp_i = comp_0 + i*100
            10087,
            10190,  # past cov; target + 10'000
            20095,
            20096,
            20097,
            20098,  # future cov; target + 20'000
        ]

        # checking the name of the lagged features
        model_instance.fit(
            series=self.target_series[0],
            past_covariates=self.past_covariates[0],
            future_covariates=self.future_covariates[0],
        )
        assert model_instance.lagged_feature_names == [
            "0-trgt-0_target_lag-4",
            "0-trgt-0_target_lag-3",
            "0-trgt-1_target_lag-3",
            "0-trgt-1_target_lag-2",
            "0-trgt-2_target_lag-2",
            "0-trgt-2_target_lag-1",
            "0-pcov-0_pastcov_lag-10",
            "0-pcov-1_pastcov_lag-7",
            "0-fcov-0_futcov_lag-2",
            "0-fcov-0_futcov_lag-1",
            "0-fcov-0_futcov_lag0",
            "0-fcov-0_futcov_lag1",
        ]

    @pytest.mark.parametrize("mode", [True, False])
    def test_prediction_data_creation(self, mode):
        # assigning correct names to variables
        series = [ts[:-50] for ts in self.target_series]
        output_chunk_length = 5
        n = 12

        # prediction preprocessing start
        covariates = {
            "past": (self.past_covariates, self.lags_1.get("past")),
            "future": (self.future_covariates, self.lags_1.get("future")),
        }

        if mode:
            shift = 0
        else:
            shift = output_chunk_length - 1

        # dictionary containing covariate data over time span required for prediction
        covariate_matrices = {}
        # dictionary containing covariate lags relative to minimum covariate lag
        relative_cov_lags = {}
        # number of prediction steps given forecast horizon and output_chunk_length
        n_pred_steps = math.ceil(n / output_chunk_length)
        remaining_steps = n % output_chunk_length  # for multi_models = False
        for cov_type, (covs, lags) in covariates.items():
            if covs is not None:
                relative_cov_lags[cov_type] = np.array(lags) - lags[0]
                covariate_matrices[cov_type] = []
                for idx, (ts, cov) in enumerate(zip(series, covs)):
                    first_pred_ts = ts.end_time() + 1 * ts.freq
                    last_pred_ts = (
                        (
                            first_pred_ts
                            + ((n_pred_steps - 1) * output_chunk_length) * ts.freq
                        )
                        if mode
                        else (first_pred_ts + (n - 1) * ts.freq)
                    )
                    first_req_ts = first_pred_ts + (lags[0] - shift) * ts.freq
                    last_req_ts = last_pred_ts + (lags[-1] - shift) * ts.freq

                    # not enough covariate data checks excluded, they are tested elsewhere

                    if cov.has_datetime_index:
                        covariate_matrices[cov_type].append(
                            cov.slice(first_req_ts, last_req_ts).values(copy=False)
                        )
                    else:
                        # include last_req_ts when slicing series with integer indices
                        covariate_matrices[cov_type].append(
                            cov[first_req_ts : last_req_ts + 1].values(copy=False)
                        )

                covariate_matrices[cov_type] = np.stack(covariate_matrices[cov_type])

        series_matrix = None
        if "target" in self.lags_1:
            series_matrix = np.stack([
                ts.values(copy=False)[self.lags_1["target"][0] - shift :, :]
                for ts in series
            ])
        # prediction preprocessing end
        assert all([lag >= 0 for lags in relative_cov_lags.values() for lag in lags])

        if mode:
            # tests for multi_models = True
            assert covariate_matrices["past"].shape == (
                len(series),
                relative_cov_lags["past"][-1]
                + (n_pred_steps - 1) * output_chunk_length
                + 1,
                covariates["past"][0][0].width,
            )
            assert covariate_matrices["future"].shape == (
                len(series),
                relative_cov_lags["future"][-1]
                + (n_pred_steps - 1) * output_chunk_length
                + 1,
                covariates["future"][0][0].width,
            )
            assert series_matrix.shape == (
                len(series),
                -self.lags_1["target"][0],
                series[0].width,
            )
            assert list(covariate_matrices["past"][0, :, 0]) == [
                10047.0,
                10048.0,
                10049.0,
                10050.0,
                10051.0,
                10052.0,
                10053.0,
                10054.0,
                10055.0,
                10056.0,
                10057.0,
                10058.0,
                10059.0,
            ]
            assert list(covariate_matrices["future"][0, :, 0]) == [
                20046.0,
                20047.0,
                20048.0,
                20049.0,
                20050.0,
                20051.0,
                20052.0,
                20053.0,
                20054.0,
                20055.0,
                20056.0,
                20057.0,
                20058.0,
                20059.0,
                20060.0,
                20061.0,
                20062.0,
                20063.0,
            ]
            assert list(series_matrix[0, :, 0]) == [48.0, 49.0, 50.0]
        else:
            # tests for multi_models = False
            assert covariate_matrices["past"].shape == (
                len(series),
                relative_cov_lags["past"][-1]
                + (n_pred_steps - 1) * output_chunk_length
                + (remaining_steps - 1)
                + 1,
                covariates["past"][0][0].width,
            )
            assert covariate_matrices["future"].shape == (
                len(series),
                relative_cov_lags["future"][-1]
                + (n_pred_steps - 1) * output_chunk_length
                + (remaining_steps - 1)
                + 1,
                covariates["future"][0][0].width,
            )
            assert series_matrix.shape == (
                len(series),
                -self.lags_1["target"][0] + shift,
                series[0].width,
            )
            assert list(covariate_matrices["past"][0, :, 0]) == [
                10043.0,
                10044.0,
                10045.0,
                10046.0,
                10047.0,
                10048.0,
                10049.0,
                10050.0,
                10051.0,
                10052.0,
                10053.0,
                10054.0,
                10055.0,
                10056.0,
            ]
            assert list(covariate_matrices["future"][0, :, 0]) == [
                20042.0,
                20043.0,
                20044.0,
                20045.0,
                20046.0,
                20047.0,
                20048.0,
                20049.0,
                20050.0,
                20051.0,
                20052.0,
                20053.0,
                20054.0,
                20055.0,
                20056.0,
                20057.0,
                20058.0,
                20059.0,
                20060.0,
            ]
            assert list(series_matrix[0, :, 0]) == [
                44.0,
                45.0,
                46.0,
                47.0,
                48.0,
                49.0,
                50.0,
            ]

    @pytest.mark.parametrize("model_cls", models)
    def test_optional_static_covariates(self, model_cls):
        """adding static covariates to lagged data logic is tested in
        `darts.tests.utils.data.tabularization.test_add_static_covariates`
        """
        series = (
            tg.linear_timeseries(length=6)
            .with_static_covariates(pd.DataFrame({"a": [1]}))
            .astype(np.float32)
        )
        # training model with static covs and predicting without will raise an error
        model = model_cls(lags=4, use_static_covariates=True)
        model.fit(series)
        assert model.uses_static_covariates
        assert model._static_covariates_shape == series.static_covariates.shape
        with pytest.raises(ValueError):
            model.predict(n=2, series=series.with_static_covariates(None))

        # with `use_static_covariates=True`, all series must have static covs
        model = model_cls(lags=4, use_static_covariates=True)
        with pytest.raises(ValueError):
            model.fit([series, series.with_static_covariates(None)])

        # with `use_static_covariates=True`, all static covs must have same shape
        model = model_cls(lags=4, use_static_covariates=True)
        with pytest.raises(ValueError):
            model.fit([
                series,
                series.with_static_covariates(pd.DataFrame({"a": [1], "b": [2]})),
            ])

        # with `use_static_covariates=False`, static covariates are ignored and prediction works
        model = model_cls(lags=4, use_static_covariates=False)
        model.fit(series)
        assert not model.uses_static_covariates
        assert model._static_covariates_shape is None
        preds = model.predict(n=2, series=series.with_static_covariates(None))
        assert preds.static_covariates is None

        # with `use_static_covariates=False`, static covariates are ignored and prediction works
        model = model_cls(lags=4, use_static_covariates=False)
        model.fit(series.with_static_covariates(None))
        assert not model.uses_static_covariates
        assert model._static_covariates_shape is None
        preds = model.predict(n=2, series=series)
        np.testing.assert_almost_equal(
            preds.static_covariates.values,
            series.static_covariates.values,
        )

        # with `use_static_covariates=True`, static covariates are included
        model = model_cls(lags=4, use_static_covariates=True)
        model.fit([series, series])
        assert model.uses_static_covariates
        assert model._static_covariates_shape == series.static_covariates.shape
        preds = model.predict(n=2, series=[series, series])
        for pred in preds:
            np.testing.assert_almost_equal(
                pred.static_covariates.values,
                series.static_covariates.values,
            )

    def test_static_cov_accuracy(self):
        """
        Tests that `RandomForest` regression model reproduces same behaviour as
        `examples/15-static-covariates.ipynb` notebook; see this notebook for
        further details. Notebook is also hosted online at:
        https://unit8co.github.io/darts/examples/15-static-covariates.html
        """

        # given
        period = 20
        sine_series = tg.sine_timeseries(
            length=4 * period,
            value_frequency=1 / period,
            column_name="smooth",
            freq="h",
        )

        sine_vals = sine_series.values()
        linear_vals = np.expand_dims(np.linspace(1, -1, num=19), -1)

        sine_vals[21:40] = linear_vals
        sine_vals[61:80] = linear_vals
        irregular_series = TimeSeries.from_times_and_values(
            values=sine_vals, times=sine_series.time_index, columns=["irregular"]
        )

        # no static covs
        train_series_no_cov = [sine_series, irregular_series]

        # categorical static covs
        sine_series_st_cat = sine_series.with_static_covariates(
            pd.DataFrame(data={"curve_type": [0]})
        )
        irregular_series_st_cat = irregular_series.with_static_covariates(
            pd.DataFrame(data={"curve_type": [1]})
        )
        train_series_static_cov = [sine_series_st_cat, irregular_series_st_cat]

        # when
        fitting_series = [series[:60] for series in train_series_no_cov]
        model_no_static_cov = RandomForest(lags=period // 2, bootstrap=False)
        model_no_static_cov.fit(fitting_series)
        pred_no_static_cov = model_no_static_cov.predict(
            n=period, series=fitting_series
        )

        fitting_series = [series[:60] for series in train_series_static_cov]
        model_static_cov = RandomForest(lags=period // 2, bootstrap=False)
        model_static_cov.fit(fitting_series)
        pred_static_cov = model_static_cov.predict(n=period, series=fitting_series)
        # then
        for series, ps_no_st, ps_st_cat in zip(
            train_series_static_cov, pred_no_static_cov, pred_static_cov
        ):
            rmses = [rmse(series, ps) for ps in [ps_no_st, ps_st_cat]]
            assert rmses[1] < rmses[0]

        # given series of different sizes in input
        fitting_series = [
            train_series_no_cov[0][: (60 - period)],
            train_series_no_cov[1][:60],
        ]
        model_no_static_cov = RandomForest(lags=period // 2, bootstrap=False)
        model_no_static_cov.fit(fitting_series)
        pred_no_static_cov = model_no_static_cov.predict(
            n=period, series=fitting_series
        )
        # multiple series with different components names ("smooth" and "irregular"),
        # will take first target name
        expected_features_in = [
            f"smooth_target_lag{str(-i)}" for i in range(period // 2, 0, -1)
        ]
        assert model_no_static_cov.lagged_feature_names == expected_features_in
        assert len(model_no_static_cov.model.feature_importances_) == len(
            expected_features_in
        )

        fitting_series = [
            train_series_static_cov[0][: (60 - period)],
            train_series_static_cov[1][:60],
        ]
        model_static_cov = RandomForest(lags=period // 2, bootstrap=False)
        model_static_cov.fit(fitting_series)

        # multiple univariates series with different names with same static cov, will take name of first series
        expected_features_in = [
            f"smooth_target_lag{str(-i)}" for i in range(period // 2, 0, -1)
        ] + ["curve_type_statcov_target_smooth"]

        assert model_static_cov.lagged_feature_names == expected_features_in
        assert len(model_static_cov.model.feature_importances_) == len(
            expected_features_in
        )

        pred_static_cov = model_static_cov.predict(n=period, series=fitting_series)

        # then
        for series, ps_no_st, ps_st_cat in zip(
            train_series_static_cov, pred_no_static_cov, pred_static_cov
        ):
            rmses = [rmse(series, ps) for ps in [ps_no_st, ps_st_cat]]
            assert rmses[1] < rmses[0]

    @pytest.mark.parametrize("config", product(models, [True, False]))
    def test_models_runnability(self, config):
        model, mode = config
        train_y, test_y = self.sine_univariate1.split_before(0.7)
        # testing past covariates
        model_instance = model(lags=4, lags_past_covariates=None, multi_models=mode)
        with pytest.raises(ValueError):
            # testing lags_past_covariates None but past_covariates during training
            model_instance.fit(
                series=self.sine_univariate1,
                past_covariates=self.sine_multivariate1,
            )

        model_instance = model(lags=4, lags_past_covariates=3, multi_models=mode)
        with pytest.raises(ValueError):
            # testing lags_past_covariates but no past_covariates during fit
            model_instance.fit(series=self.sine_univariate1)

        # testing future_covariates
        model_instance = model(lags=4, lags_future_covariates=None, multi_models=mode)
        with pytest.raises(ValueError):
            # testing lags_future_covariates None but future_covariates during training
            model_instance.fit(
                series=self.sine_univariate1,
                future_covariates=self.sine_multivariate1,
            )

        model_instance = model(lags=4, lags_future_covariates=(0, 3), multi_models=mode)
        with pytest.raises(ValueError):
            # testing lags_covariate but no covariate during fit
            model_instance.fit(series=self.sine_univariate1)

        # testing input_dim
        model_instance = model(lags=4, lags_past_covariates=2, multi_models=mode)
        model_instance.fit(
            series=train_y,
            past_covariates=self.sine_univariate1.stack(self.sine_univariate1),
        )

        assert model_instance.input_dim == {
            "target": 1,
            "past": 2,
            "future": None,
        }

        with pytest.raises(ValueError):
            prediction = model_instance.predict(n=len(test_y) + 2)

        # while it should work with n = 1
        prediction = model_instance.predict(n=1)
        assert len(prediction) == 1

    @pytest.mark.parametrize(
        "config",
        product(models, [True, False], [sine_univariate1, sine_multivariate1]),
    )
    def test_fit(self, config):
        # test fitting both on univariate and multivariate timeseries
        model, mode, series = config

        series = series[:15]
        sine_multivariate1 = self.sine_multivariate1[:15]

        # auto-regression but past_covariates does not extend enough in the future
        with pytest.raises(ValueError):
            model_instance = model(lags=4, lags_past_covariates=4, multi_models=mode)
            model_instance.fit(series=series, past_covariates=sine_multivariate1)
            model_instance.predict(n=10)

        # inconsistent number of components in series Sequence[TimeSeries]
        training_series = [series.stack(series + 10), series]
        with pytest.raises(ValueError) as err:
            model_instance = model(lags=4, multi_models=mode)
            model_instance.fit(series=training_series)
        assert (
            str(err.value)
            == f"Expected {training_series[0].width} components but received {training_series[1].width} "
            f"components at index 1 of `series`."
        )

        # inconsistent number of components in past_covariates Sequence[TimeSeries]
        training_past_covs = [series, series.stack(series * 2)]
        with pytest.raises(ValueError) as err:
            model_instance = model(lags=4, lags_past_covariates=2, multi_models=mode)
            model_instance.fit(
                series=[series, series + 10],
                past_covariates=training_past_covs,
            )
        assert (
            str(err.value)
            == f"Expected {training_past_covs[0].width} components but received {training_past_covs[1].width} "
            f"components at index 1 of `past_covariates`."
        )

        model_instance = model(lags=12, multi_models=mode)
        model_instance.fit(series=series)
        assert model_instance.lags.get("past") is None

        model_instance = model(lags=12, lags_past_covariates=12, multi_models=mode)
        model_instance.fit(series=series, past_covariates=sine_multivariate1)
        assert len(model_instance.lags.get("past")) == 12

        model_instance = model(
            lags=12, lags_future_covariates=(0, 1), multi_models=mode
        )
        model_instance.fit(series=series, future_covariates=sine_multivariate1)
        assert len(model_instance.lags.get("future")) == 1

        model_instance = model(
            lags=12, lags_past_covariates=[-1, -4, -6], multi_models=mode
        )
        model_instance.fit(series=series, past_covariates=sine_multivariate1)
        assert len(model_instance.lags.get("past")) == 3

        model_instance = model(
            lags=12,
            lags_past_covariates=[-1, -4, -6],
            lags_future_covariates=[-2, 0],
            multi_models=mode,
        )
        model_instance.fit(
            series=series,
            past_covariates=sine_multivariate1,
            future_covariates=sine_multivariate1,
        )
        assert len(model_instance.lags.get("past")) == 3

    def helper_test_models_accuracy(
        self,
        series,
        past_covariates,
        min_rmse_model,
        model,
        idx,
        mode,
        output_chunk_length,
    ):
        # for every model, test whether it predicts the target with a minimum r2 score of `min_rmse`
        train_series, test_series = train_test_split(series, 70)
        train_past_covariates, _ = train_test_split(past_covariates, 70)
        model_instance = model(
            lags=12,
            lags_past_covariates=2,
            output_chunk_length=output_chunk_length,
            multi_models=mode,
        )
        model_instance.fit(series=train_series, past_covariates=train_past_covariates)
        prediction = model_instance.predict(
            n=len(test_series),
            series=train_series,
            past_covariates=past_covariates,
        )
        current_rmse = rmse(prediction, test_series)
        # in case of multi-series take mean rmse
        mean_rmse = np.mean(current_rmse)
        assert mean_rmse <= min_rmse_model[idx], (
            f"{str(model_instance)} model was not able to predict data as well as expected. "
            f"A mean rmse score of {mean_rmse} was recorded."
        )

    @pytest.mark.parametrize(
        "config",
        product(zip(models, range(len(models))), [True, False], [1, 5]),
    )
    def test_models_accuracy_univariate(self, config):
        (model, idx), mode, ocl = config
        # for every model, and different output_chunk_lengths test whether it predicts the univariate time series
        # as well as expected, accuracies are defined at the top of the class
        self.helper_test_models_accuracy(
            self.sine_univariate1,
            self.sine_univariate2,
            self.univariate_accuracies,
            model,
            idx,
            mode,
            ocl,
        )

    @pytest.mark.parametrize(
        "config",
        product(zip(models, range(len(models))), [True, False], [1, 5]),
    )
    def test_models_accuracy_multivariate(self, config):
        (model, idx), mode, ocl = config
        # for every model, and different output_chunk_lengths test whether it predicts the multivariate time series
        # as well as expected, accuracies are defined at the top of the class
        self.helper_test_models_accuracy(
            self.sine_multivariate1,
            self.sine_multivariate2,
            self.multivariate_accuracies,
            model,
            idx,
            mode,
            ocl,
        )

    @pytest.mark.parametrize(
        "config",
        product(zip(models, range(len(models))), [True, False], [1, 5]),
    )
    def test_models_accuracy_multiseries_multivariate(self, config):
        (model, idx), mode, ocl = config
        # for every model, and different output_chunk_lengths test whether it predicts the multiseries, multivariate
        # time series as well as expected, accuracies are defined at the top of the class
        self.helper_test_models_accuracy(
            self.sine_multiseries1,
            self.sine_multiseries2,
            self.multivariate_multiseries_accuracies,
            model,
            idx,
            mode,
            ocl,
        )

    @pytest.mark.parametrize("mode", [True, False])
    def test_min_train_series_length(self, mode):
        lgbm_cls = LightGBMModel if lgbm_available else XGBModel
        cb_cls = CatBoostModel if cb_available else XGBModel
        model = lgbm_cls(lags=4, multi_models=mode)
        min_train_series_length_expected = (
            -model.lags["target"][0] + model.output_chunk_length + 1
        )
        assert min_train_series_length_expected == model.min_train_series_length
        model = cb_cls(lags=2, multi_models=mode)
        min_train_series_length_expected = (
            -model.lags["target"][0] + model.output_chunk_length + 1
        )
        assert min_train_series_length_expected == model.min_train_series_length
        model = lgbm_cls(lags=[-4, -3, -2], multi_models=mode)
        min_train_series_length_expected = (
            -model.lags["target"][0] + model.output_chunk_length + 1
        )
        assert min_train_series_length_expected == model.min_train_series_length
        model = XGBModel(lags=[-4, -3, -2], multi_models=mode)
        min_train_series_length_expected = (
            -model.lags["target"][0] + model.output_chunk_length + 1
        )
        assert min_train_series_length_expected == model.min_train_series_length

    @pytest.mark.parametrize("mode", [True, False])
    def test_historical_forecast(self, mode):
        model = self.models[1](lags=5, multi_models=mode)
        result = model.historical_forecasts(
            series=self.sine_univariate1,
            future_covariates=None,
            start=0.8,
            forecast_horizon=1,
            stride=1,
            retrain=True,
            overlap_end=False,
            last_points_only=True,
            verbose=False,
        )
        assert len(result) == 21

        model = self.models[1](lags=5, lags_past_covariates=5, multi_models=mode)
        result = model.historical_forecasts(
            series=self.sine_univariate1,
            past_covariates=self.sine_multivariate1,
            start=0.8,
            forecast_horizon=1,
            stride=1,
            retrain=True,
            overlap_end=False,
            last_points_only=True,
            verbose=False,
        )
        assert len(result) == 21

        model = self.models[1](
            lags=5, lags_past_covariates=5, output_chunk_length=5, multi_models=mode
        )
        result = model.historical_forecasts(
            series=self.sine_univariate1,
            past_covariates=self.sine_multivariate1,
            start=0.8,
            forecast_horizon=1,
            stride=1,
            retrain=True,
            overlap_end=False,
            last_points_only=True,
            verbose=False,
        )
        assert len(result) == 21

    def test_opti_historical_forecast_predict_checks(self):
        """
        Verify that the sanity check implemented in ForecastingModel.predict are also defined for optimized historical
        forecasts as it does not call this method
        """
        model = self.models[1](lags=5)

        msg_expected = (
            "The model has not been fitted yet, and `retrain` is ``False``. Either call `fit()` before "
            "`historical_forecasts()`, or set `retrain` to something different than ``False``."
        )
        # untrained model, optimized
        with pytest.raises(ValueError) as err:
            model.historical_forecasts(
                series=self.sine_univariate1,
                start=0.9,
                forecast_horizon=1,
                retrain=False,
                enable_optimization=True,
                verbose=False,
            )
        assert str(err.value) == msg_expected

        model.fit(
            series=self.sine_univariate1,
        )
        # deterministic model, num_samples > 1, optimized
        with pytest.raises(ValueError) as err:
            model.historical_forecasts(
                series=self.sine_univariate1,
                start=0.9,
                forecast_horizon=1,
                retrain=False,
                enable_optimization=True,
                num_samples=10,
                verbose=False,
            )
        assert (
            str(err.value)
            == "`num_samples > 1` is only supported for probabilistic models."
        )

    @pytest.mark.parametrize(
        "config",
        [
            (RegressionModel(lags=4), True),
            (RegressionModel(lags=4, model=LinearRegression()), True),
            (RegressionModel(lags=4, model=RandomForestRegressor()), True),
            (
                RegressionModel(lags=4, model=HistGradientBoostingRegressor()),
                False,
            ),
        ],
    )
    def test_multioutput_wrapper(self, config):
        """Check that with input_chunk_length=1, wrapping in MultiOutputRegressor occurs only when necessary"""
        model, supports_multioutput_natively = config
        model.fit(series=self.sine_multivariate1)
        if supports_multioutput_natively:
            assert not isinstance(model.model, MultiOutputRegressor)
            # single estimator is responsible for both components
            assert (
                model.model
                == model.get_estimator(horizon=0, target_dim=0)
                == model.get_estimator(horizon=0, target_dim=1)
            )
        else:
            assert isinstance(model.model, MultiOutputRegressor)
            # one estimator (sub-model) per component
            assert model.get_estimator(horizon=0, target_dim=0) != model.get_estimator(
                horizon=0, target_dim=1
            )

    model_configs_multioutput = [
        (
            RegressionModel,
            {"lags": 4, "model": LinearRegression()},
            True,
        ),
        (LinearRegressionModel, {"lags": 4}, True),
        (XGBModel, {"lags": 4}, True),
        (XGBModel, {"lags": 4, "likelihood": "poisson"}, False),
    ]
    if lgbm_available:
        model_configs_multioutput += [(LightGBMModel, {"lags": 4}, False)]
    if cb_available:
        model_configs_multioutput += [
            (CatBoostModel, {"lags": 4, "loss_function": "RMSE"}, False),
            (CatBoostModel, {"lags": 4, "loss_function": "MultiRMSE"}, True),
            (CatBoostModel, {"lags": 4, "loss_function": "RMSEWithUncertainty"}, False),
        ]

    @pytest.mark.parametrize("config", model_configs_multioutput)
    def test_supports_native_multioutput(self, config):
        model_cls, model_config, supports_native_multioutput = config
        model = model_cls(**model_config)
        assert model._supports_native_multioutput == supports_native_multioutput

    model_configs = [(XGBModel, dict({"likelihood": "poisson"}, **xgb_test_params))]
    if lgbm_available:
        model_configs += [(LightGBMModel, lgbm_test_params)]
    if cb_available:
        model_configs += [(CatBoostModel, cb_test_params)]

    @pytest.mark.parametrize("config", product(model_configs, [1, 2], [True, False]))
    def test_multioutput_validation(self, config):
        """Check that models not supporting multi-output are properly wrapped when ocl>1"""
        (model_cls, model_kwargs), ocl, multi_models = config
        train, val = self.sine_univariate1.split_after(0.6)
        model = model_cls(
            **model_kwargs, lags=4, output_chunk_length=ocl, multi_models=multi_models
        )
        model.fit(series=train, val_series=val)
        if model.output_chunk_length > 1 and model.multi_models:
            assert isinstance(model.model, MultiOutputRegressor)
        else:
            assert not isinstance(model.model, MultiOutputRegressor)

    def test_get_estimator_multi_models(self):
        """Craft training data so that estimator_[i].predict(X) == i + 1"""

        def helper_check_overfitted_estimators(ts: TimeSeries, ocl: int):
            # since xgboost==2.1.0, the regular deterministic models have native multi output regression
            # -> we use a quantile likelihood to activate Darts' MultiOutputRegressor
            m = XGBModel(
                lags=3,
                output_chunk_length=ocl,
                multi_models=True,
                likelihood="quantile",
                quantiles=[0.5],
            )
            m.fit(ts)

            assert len(m.model.estimators_) == ocl * ts.width

            dummy_feats = np.array([[0, 0, 0] * ts.width])
            estimator_counter = 0
            for i in range(ocl):
                for j in range(ts.width):
                    sub_model = m.get_estimator(horizon=i, target_dim=j)
                    pred = sub_model.predict(dummy_feats)[0]
                    # sub-model is overfitted on the training series
                    assert np.abs(estimator_counter - pred) < 1e-2
                    estimator_counter += 1

        # univariate, one-sub model per step in output_chunk_length
        ocl = 3
        ts = TimeSeries.from_values(np.array([0, 0, 0, 0, 1, 2]).T)
        # estimators_[0] labels : [0]
        # estimators_[1] labels : [1]
        # estimators_[2] labels : [2]
        helper_check_overfitted_estimators(ts, ocl)

        # multivariate, one sub-model per component
        ocl = 1
        ts = TimeSeries.from_values(
            np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]).T
        )
        # estimators_[0] labels : [0]
        # estimators_[1] labels : [1]
        # estimators_[2] labels : [2]
        helper_check_overfitted_estimators(ts, ocl)

        # multivariate, one sub-model per position, per component
        ocl = 2
        ts = TimeSeries.from_values(
            np.array([
                [0, 0, 0, 0, 2],
                [0, 0, 0, 1, 3],
            ]).T
        )
        # estimators_[0] labels : [0]
        # estimators_[1] labels : [1]
        # estimators_[2] labels : [2]
        # estimators_[3] labels : [3]
        helper_check_overfitted_estimators(ts, ocl)

    def test_get_estimator_single_model(self):
        """Check estimator getter when multi_models=False"""
        # multivariate, one sub-model per component
        ocl = 2
        ts = TimeSeries.from_values(
            np.array([
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 2],
            ]).T
        )
        # estimators_[0] labels : [1]
        # estimators_[1] labels : [2]

        # since xgboost==2.1.0, the regular deterministic models have native multi output regression
        # -> we use a quantile likelihood to activate Darts' MultiOutputRegressor
        m = XGBModel(
            lags=3,
            output_chunk_length=ocl,
            multi_models=False,
            likelihood="quantile",
            quantiles=[0.5],
        )
        m.fit(ts)

        # one estimator is reused for all the horizon of a given component
        assert len(m.model.estimators_) == ts.width

        dummy_feats = np.array([[0, 0, 0] * ts.width])
        for i in range(ocl):
            for j in range(ts.width):
                sub_model = m.get_estimator(horizon=i, target_dim=j)
                pred = sub_model.predict(dummy_feats)[0]
                # sub-model forecast only depend on the target_dim
                assert np.abs(j + 1 - pred) < 1e-2

    @pytest.mark.parametrize("multi_models", [True, False])
    def test_get_estimator_quantile(self, multi_models):
        """Check estimator getter when using quantile value"""
        ocl = 3
        lags = 3
        quantiles = [0.01, 0.5, 0.99]
        ts = tg.sine_timeseries(length=100, column_name="sine").stack(
            tg.linear_timeseries(length=100, column_name="linear"),
        )

        m = XGBModel(
            lags=lags,
            output_chunk_length=ocl,
            multi_models=multi_models,
            likelihood="quantile",
            quantiles=quantiles,
            random_state=1,
        )
        m.fit(ts)

        assert len(m._model_container) == len(quantiles)
        assert sorted(list(m._model_container.keys())) == sorted(quantiles)
        for quantile_container in m._model_container.values():
            # one sub-model per quantile, per component, per horizon
            if multi_models:
                assert len(quantile_container.estimators_) == ocl * ts.width
            # one sub-model per quantile, per component
            else:
                assert len(quantile_container.estimators_) == ts.width

        # check that retrieve sub-models prediction match the "wrapper" model predictions
        pred_input = ts[-lags:] if multi_models else ts[-lags - ocl + 1 :]
        pred = m.predict(
            n=ocl,
            series=pred_input,
            num_samples=1,
            predict_likelihood_parameters=True,
        )
        for j in range(ts.width):
            for i in range(ocl):
                if multi_models:
                    dummy_feats = pred_input.values()[:lags]
                else:
                    dummy_feats = pred_input.values()[i : +i + lags]
                dummy_feats = np.expand_dims(dummy_feats.flatten(), 0)
                for q in quantiles:
                    sub_model = m.get_estimator(horizon=i, target_dim=j, quantile=q)
                    pred_sub_model = sub_model.predict(dummy_feats)[0]
                    assert (
                        pred_sub_model
                        == pred[f"{ts.components[j]}_q{q:.2f}"].values()[i][0]
                    )

    def test_get_estimator_exceptions(self, caplog):
        """Check that all the corner-cases are properly covered by the method"""
        ts = TimeSeries.from_values(
            values=np.array([
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 2],
            ]).T,
            columns=["a", "b"],
        )
        m = LinearRegressionModel(
            lags=2,
            output_chunk_length=2,
            random_state=1,
        )
        m.fit(ts["a"])
        # not wrapped in MultiOutputRegressor because of native multi-output support
        with caplog.at_level(logging.WARNING):
            m.get_estimator(horizon=0, target_dim=0)
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"
        assert caplog.records[0].message == (
            "Model supports multi-output; a single estimator "
            "forecasts all the horizons and components."
        )

        # univariate, deterministic, ocl > 2
        m = RegressionModel(
            model=HistGradientBoostingRegressor(),
            lags=2,
            output_chunk_length=2,
        )
        m.fit(ts["a"])
        # horizon > ocl
        with pytest.raises(ValueError) as err:
            m.get_estimator(horizon=3, target_dim=0)
        assert str(err.value).startswith(
            "`horizon` must be `>= 0` and `< output_chunk_length"
        )
        # target dim > training series width
        with pytest.raises(ValueError) as err:
            m.get_estimator(horizon=0, target_dim=1)
        assert str(err.value).startswith(
            "`target_dim` must be `>= 0`, and `< n_target_components="
        )

        # univariate, probabilistic
        # using the quantiles argument to force wrapping in MultiOutputRegressor
        m = XGBModel(
            lags=2,
            output_chunk_length=2,
            random_state=1,
            likelihood="poisson",
            quantiles=[0.5],
        )
        m.fit(ts["a"])
        # incorrect likelihood
        with pytest.raises(ValueError) as err:
            m.get_estimator(horizon=0, target_dim=0, quantile=0.1)
        assert str(err.value).startswith(
            "`quantile` is only supported for probabilistic models that "
            "use `likelihood='quantile'`."
        )

        # univariate, probabilistic
        m = XGBModel(
            lags=2,
            output_chunk_length=2,
            random_state=1,
            likelihood="quantile",
            quantiles=[0.01, 0.5, 0.99],
        )
        m.fit(ts["a"])
        # retrieving a non-defined quantile
        with pytest.raises(ValueError) as err:
            m.get_estimator(horizon=0, target_dim=0, quantile=0.1)
        assert str(err.value).startswith(
            "Invalid `quantile=0.1`. Must be one of the fitted quantiles "
            "`[0.01, 0.5, 0.99]`."
        )

    @pytest.mark.parametrize("mode", [True, False])
    def test_regression_model(self, mode):
        lags = 4
        models = [
            RegressionModel(lags=lags, multi_models=mode),
            RegressionModel(lags=lags, model=LinearRegression(), multi_models=mode),
            RegressionModel(
                lags=lags, model=RandomForestRegressor(), multi_models=mode
            ),
            RegressionModel(
                lags=lags, model=HistGradientBoostingRegressor(), multi_models=mode
            ),
        ]

        for model in models:
            model.fit(series=self.sine_univariate1)
            assert len(model.lags.get("target")) == lags
            model.predict(n=10)

    @pytest.mark.parametrize("mode", [True, False])
    def test_multiple_ts(self, mode):
        lags = 4
        lags_past_covariates = 3
        model = RegressionModel(
            lags=lags, lags_past_covariates=lags_past_covariates, multi_models=mode
        )

        target_series = tg.linear_timeseries(start_value=0, end_value=49, length=50)
        past_covariates = tg.linear_timeseries(
            start_value=100, end_value=149, length=50
        )
        past_covariates = past_covariates.stack(
            tg.linear_timeseries(start_value=400, end_value=449, length=50)
        )

        target_train, target_test = target_series.split_after(0.7)
        past_covariates_train, past_covariates_test = past_covariates.split_after(0.7)
        model.fit(
            series=[target_train, target_train + 0.5],
            past_covariates=[past_covariates_train, past_covariates_train + 0.5],
        )

        predictions = model.predict(
            10,
            series=[target_train, target_train + 0.5],
            past_covariates=[past_covariates, past_covariates + 0.5],
        )

        assert len(predictions[0]) == 10, f"Found {len(predictions)} instead"

        # multiple TS, both future and past covariates, checking that both covariates lead to better results than
        # using a single one (target series = past_cov + future_cov + noise)
        np.random.seed(42)

        linear_ts_1 = tg.linear_timeseries(start_value=10, end_value=59, length=50)
        linear_ts_2 = tg.linear_timeseries(start_value=40, end_value=89, length=50)

        past_covariates = tg.sine_timeseries(length=50) * 10
        future_covariates = tg.sine_timeseries(length=50, value_frequency=0.015) * 50

        target_series_1 = linear_ts_1 + 4 * past_covariates + 2 * future_covariates
        target_series_2 = linear_ts_2 + 4 * past_covariates + 2 * future_covariates

        target_series_1_noise = (
            linear_ts_1
            + 4 * past_covariates
            + 2 * future_covariates
            + tg.gaussian_timeseries(std=7, length=50)
        )

        target_series_2_noise = (
            linear_ts_2
            + 4 * past_covariates
            + 2 * future_covariates
            + tg.gaussian_timeseries(std=7, length=50)
        )

        target_train_1, target_test_1 = target_series_1.split_after(0.7)
        target_train_2, target_test_2 = target_series_2.split_after(0.7)

        (
            target_train_1_noise,
            target_test_1_noise,
        ) = target_series_1_noise.split_after(0.7)
        (
            target_train_2_noise,
            target_test_2_noise,
        ) = target_series_2_noise.split_after(0.7)

        # testing improved denoise with multiple TS

        # test 1: with single TS, 2 covariates should be better than one
        model = RegressionModel(lags=3, lags_past_covariates=5, multi_models=mode)
        model.fit([target_train_1_noise], [past_covariates])

        prediction_past_only = model.predict(
            n=len(target_test_1),
            series=[target_train_1_noise, target_train_2_noise],
            past_covariates=[past_covariates] * 2,
        )

        model = RegressionModel(
            lags=3,
            lags_past_covariates=5,
            lags_future_covariates=(5, 0),
            multi_models=mode,
        )
        model.fit([target_train_1_noise], [past_covariates], [future_covariates])
        prediction_past_and_future = model.predict(
            n=len(target_test_1),
            series=[target_train_1_noise, target_train_2_noise],
            past_covariates=[past_covariates] * 2,
            future_covariates=[future_covariates] * 2,
        )

        error_past_only = rmse(
            [target_test_1, target_test_2],
            prediction_past_only,
            series_reduction=np.mean,
        )
        error_both = rmse(
            [target_test_1, target_test_2],
            prediction_past_and_future,
            series_reduction=np.mean,
        )

        assert error_past_only > error_both
        # test 2: with both covariates, 2 TS should learn more than one (with little noise)
        model = RegressionModel(
            lags=3,
            lags_past_covariates=5,
            lags_future_covariates=(5, 0),
            multi_models=mode,
        )
        model.fit(
            [target_train_1_noise, target_train_2_noise],
            [past_covariates] * 2,
            [future_covariates] * 2,
        )
        prediction_past_and_future_multi_ts = model.predict(
            n=len(target_test_1),
            series=[target_train_1_noise, target_train_2_noise],
            past_covariates=[past_covariates] * 2,
            future_covariates=[future_covariates] * 2,
        )
        error_both_multi_ts = rmse(
            [target_test_1, target_test_2],
            prediction_past_and_future_multi_ts,
            series_reduction=np.mean,
        )

        assert error_both > error_both_multi_ts

    @pytest.mark.parametrize(
        "config",
        product(
            [
                (LinearRegressionModel, {}),
                (RandomForest, {"bootstrap": False}),
                (XGBModel, xgb_test_params),
                (KNeighborsRegressorModel, {}),  # no weights support
            ]
            + (
                [(CatBoostModel, dict({"allow_const_label": True}, **cb_test_params))]
                if cb_available
                else []
            )
            + ([(LightGBMModel, lgbm_test_params)] if lgbm_available else []),
            [True, False],
        ),
    )
    def test_weights_built_in(self, config):
        (model_cls, model_kwargs), single_series = config

        ts = TimeSeries.from_values(values=np.array([0, 0, 0, 0, 1, 0, 0]))

        model = model_cls(lags=3, output_chunk_length=1, **model_kwargs)
        model.fit(
            ts if single_series else [ts] * 2,
            sample_weight="linear",
        )
        preds = model.predict(n=3, series=ts if single_series else [ts] * 2)

        model_no_weight = model_cls(lags=3, output_chunk_length=1, **model_kwargs)
        model_no_weight.fit(
            ts if single_series else [ts] * 2,
            sample_weight=None,
        )
        preds_no_weight = model_no_weight.predict(
            n=3, series=ts if single_series else [ts] * 2
        )

        if single_series:
            preds = [preds]
            preds_no_weight = [preds_no_weight]

        for pred, pred_no_weight in zip(preds, preds_no_weight):
            if model.supports_sample_weight:
                with pytest.raises(AssertionError):
                    np.testing.assert_array_almost_equal(
                        pred.all_values(), pred_no_weight.all_values()
                    )
            else:
                np.testing.assert_array_almost_equal(
                    pred.all_values(), pred_no_weight.all_values()
                )

    @pytest.mark.parametrize(
        "config",
        product(
            [
                (LinearRegressionModel, {}),
                (RandomForest, {"bootstrap": False}),
                (XGBModel, xgb_test_params),
                (KNeighborsRegressorModel, {}),  # no weights support
            ]
            + (
                [(CatBoostModel, dict({"allow_const_label": True}, **cb_test_params))]
                if cb_available
                else []
            )
            + ([(LightGBMModel, lgbm_test_params)] if lgbm_available else []),
            [True, False],
        ),
    )
    def test_weights_single_step_horizon(self, config):
        (model_cls, model_kwargs), single_series = config
        model = model_cls(lags=3, output_chunk_length=1, **model_kwargs)

        weights = TimeSeries.from_values(np.array([0, 0, 0, 0, 1, 0, 0]))

        ts = TimeSeries.from_values(values=np.array([0, 0, 0, 0, 1, 0, 0]))

        model.fit(
            ts if single_series else [ts] * 2,
            sample_weight=weights if single_series else [weights] * 2,
        )

        preds = model.predict(n=3, series=ts if single_series else [ts] * 2)

        preds = [preds] if single_series else preds
        for pred in preds:
            if model.supports_sample_weight:
                np.testing.assert_array_almost_equal(pred.values()[:, 0], [1, 1, 1])
            else:
                with pytest.raises(AssertionError):
                    np.testing.assert_array_almost_equal(pred.values()[:, 0], [1, 1, 1])

    @pytest.mark.parametrize(
        "config",
        [
            (LinearRegressionModel, {}),
            (RandomForest, {"bootstrap": False}),
            (XGBModel, xgb_test_params),
            (KNeighborsRegressorModel, {}),  # no weights support
        ]
        + (
            [(CatBoostModel, dict({"allow_const_label": True}, **cb_test_params))]
            if cb_available
            else []
        )
        + ([(LightGBMModel, lgbm_test_params)] if lgbm_available else []),
    )
    def test_weights_multi_horizon(self, config):
        (model_cls, model_kwargs) = config
        model = model_cls(lags=3, output_chunk_length=3, **model_kwargs)

        weights = TimeSeries.from_values(np.array([0, 0, 0, 1, 1, 1, 0, 0, 0]))

        # model should only fit on ones in the middle
        ts = TimeSeries.from_values(values=np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]))

        model.fit(ts, sample_weight=weights)

        pred = model.predict(n=3)

        if model.supports_sample_weight:
            np.testing.assert_array_almost_equal(pred.values()[:, 0], [1, 1, 1])
        else:
            with pytest.raises(AssertionError):
                np.testing.assert_array_almost_equal(pred.values()[:, 0], [1, 1, 1])

    def test_weights_multimodel_false_multi_horizon(self):
        model = LinearRegressionModel(lags=3, output_chunk_length=3, multi_models=False)

        weights = TimeSeries.from_values(np.array([0, 0, 0, 0, 0, 1, 0, 0]))

        ts = TimeSeries.from_values(values=np.array([0, 0, 0, 0, 0, 1, 0, 0]))

        model.fit(ts, sample_weight=weights)

        pred = model.predict(n=3)

        np.testing.assert_array_almost_equal(pred.values()[:, 0], [1, 1, 1])

    @pytest.mark.parametrize("mode", [True, False])
    def test_only_future_covariates(self, mode):
        model = RegressionModel(lags_future_covariates=[-2], multi_models=mode)

        target_series = tg.linear_timeseries(start_value=0, end_value=49, length=50)
        covariates = tg.linear_timeseries(start_value=100, end_value=149, length=50)
        covariates = covariates.stack(
            tg.linear_timeseries(start_value=400, end_value=449, length=50)
        )

        target_train, target_test = target_series.split_after(0.7)
        covariates_train, covariates_test = covariates.split_after(0.7)
        model.fit(
            series=[target_train, target_train + 0.5],
            future_covariates=[covariates_train, covariates_train + 0.5],
        )

        predictions = model.predict(
            10,
            series=[target_train, target_train + 0.5],
            future_covariates=[covariates, covariates + 0.5],
        )

        assert len(predictions[0]) == 10, f"Found {len(predictions[0])} instead"

    @pytest.mark.parametrize(
        "config",
        product(
            [True, False],
            [
                (1, 0, 13),
                (5, -4, 9),
                (7, -6, 7),
                (
                    12,
                    -9,
                    4,
                ),  # output_chunk_length > n -> covariate requirements are capped
            ],
        ),
    )
    def test_not_enough_covariates(self, config):
        # mode, output_chunk_length, required past_offset, required future_offset
        mode, (output_chunk_length, req_past_offset, req_future_offset) = config
        target_series = tg.linear_timeseries(start_value=0, end_value=100, length=50)
        past_covariates = tg.linear_timeseries(
            start_value=100, end_value=200, length=50
        )
        future_covariates = tg.linear_timeseries(
            start_value=200, end_value=300, length=50
        )

        model = RegressionModel(
            lags_past_covariates=[-10],
            lags_future_covariates=[-5, 5],
            output_chunk_length=7,
            multi_models=mode,
        )
        model.fit(
            series=target_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            max_samples_per_ts=1,
        )

        n = 10

        model = RegressionModel(
            lags_past_covariates=[-10],
            lags_future_covariates=[-4, 3],
            output_chunk_length=output_chunk_length,
            multi_models=mode,
        )
        model.fit(
            series=target_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # check that given the required offsets no ValueError is raised
        model.predict(
            n,
            series=target_series[:-25],
            past_covariates=past_covariates[: -25 + req_past_offset],
            future_covariates=future_covariates[: -25 + req_future_offset],
        )
        # check that one less past covariate time step causes ValueError
        with pytest.raises(ValueError):
            model.predict(
                n,
                series=target_series[:-25],
                past_covariates=past_covariates[: -26 + req_past_offset],
                future_covariates=future_covariates[: -25 + req_future_offset],
            )
        # check that one less future covariate time step causes ValueError
        with pytest.raises(ValueError):
            model.predict(
                n,
                series=target_series[:-25],
                past_covariates=past_covariates[: -25 + req_past_offset],
                future_covariates=future_covariates[: -26 + req_future_offset],
            )

    @pytest.mark.parametrize(
        "config",
        product(
            [(XGBModel, xgb_test_params)]
            + ([(LightGBMModel, lgbm_test_params)] if lgbm_available else [])
            + ([(CatBoostModel, cb_test_params)] if cb_available else []),
            [True, False],
        ),
    )
    def test_val_set_weights_runnability_trees(self, config):
        """Tests using weights in val set for single and multi series."""
        (model_cls, model_kwargs), single_series = config
        model = model_cls(lags=10, **model_kwargs)

        series = tg.sine_timeseries(length=20)
        weights = tg.linear_timeseries(length=20)
        if not single_series:
            series = [series] * 2
            weights = [weights] * 2

        model.fit(
            series=series,
            val_series=series,
            sample_weight=weights,
            val_sample_weight=weights,
        )
        _ = model.predict(1, series=series)

    @pytest.mark.parametrize(
        "config",
        product(
            [
                (
                    XGBModel,
                    xgb_test_params,
                    "xgboost.xgb.XGBRegressor",
                    "xgboost.XGBRegressor",
                )
            ]
            + (
                [
                    (
                        LightGBMModel,
                        lgbm_test_params,
                        "lgbm.lgb.LGBMRegressor",
                        "lightgbm.LGBMRegressor",
                    )
                ]
                if lgbm_available
                else []
            )
            + (
                [
                    (
                        CatBoostModel,
                        cb_test_params,
                        "catboost_model.CatBoostRegressor",
                        "catboost.CatBoostRegressor",
                    )
                ]
                if cb_available
                else []
            ),
            [False, True],
        ),
    )
    def test_val_set(self, config):
        """Test whether the evaluation set parameters are passed to the wrapped regression model."""
        (model_cls, model_kwargs, model_loc, model_import), use_weights = config
        module_name, model_name = model_import.split(".")
        # mocking `fit` loses function signature. MultiOutputRegressor checks the function signature
        # internally, so we have to overwrite the mocked function signature with the original one.
        fit_sig = inspect.signature(
            getattr(importlib.import_module(module_name), model_name).fit
        )
        with patch(f"darts.models.forecasting.{model_loc}.fit") as fit_patch:
            fit_patch.__signature__ = fit_sig
            self.helper_check_val_set(
                model_cls, model_kwargs, fit_patch, use_weights=use_weights
            )

    def helper_check_val_set(self, model_cls, model_kwargs, fit_patch, use_weights):
        series1 = tg.sine_timeseries(length=10, column_name="tg_1")
        series2 = tg.sine_timeseries(length=10, column_name="tg_2") / 2 + 10
        series = series1.stack(series2)
        series = series.with_static_covariates(
            pd.DataFrame({"sc1": [0, 1], "sc2": [3, 4]})
        )
        pc = series1 * 10 - 3
        fc = TimeSeries.from_times_and_values(
            times=series.time_index, values=series.values() * -1, columns=["fc1", "fc2"]
        )

        weights_kwargs = (
            {
                "sample_weight": tg.linear_timeseries(length=10),
                "val_sample_weight": tg.linear_timeseries(length=10),
            }
            if use_weights
            else {}
        )

        model = model_cls(
            lags={"default_lags": [-4, -3, -2, -1]},
            lags_past_covariates=3,
            lags_future_covariates={
                "default_lags": [-1, 0],
                "fc1": [0],
            },
            likelihood="quantile",
            add_encoders={"cyclic": {"future": ["month"]}},
            quantiles=[0.1, 0.5, 0.9],
            **model_kwargs,
        )

        # check that an error is raised with an invalid validation series
        with pytest.raises(ValueError) as err:
            model.fit(
                series=series,
                past_covariates=pc,
                future_covariates=fc,
                val_series=series["tg_1"],
                val_past_covariates=pc,
                val_future_covariates=fc["fc1"],
                early_stopping_rounds=2,
                **weights_kwargs,
            )
        msg_expected = (
            "The dimensions of the (`series`, `future_covariates`, `static_covariates`) between "
            "the training and validation set do not match."
        )
        assert str(err.value) == msg_expected

        # check that an error is raised if only second validation series are invalid
        with pytest.raises(ValueError) as err:
            model.fit(
                series=series,
                past_covariates=pc,
                future_covariates=fc,
                val_series=[series, series["tg_1"]],
                val_past_covariates=[pc, pc],
                val_future_covariates=[fc, fc["fc1"]],
                early_stopping_rounds=2,
                **weights_kwargs,
            )
        msg_expected = (
            "The dimensions of the (`series`, `future_covariates`, `static_covariates`) between "
            "the training and validation set at sequence/list index `1` do not match."
        )
        assert str(err.value) == msg_expected

        model.fit(
            series=series,
            past_covariates=pc,
            future_covariates=fc,
            val_series=series,
            val_past_covariates=pc,
            val_future_covariates=fc,
            early_stopping_rounds=2,
            **weights_kwargs,
        )
        # fit called 6 times (3 quantiles * 2 target features)
        assert fit_patch.call_count == 6

        X_train, y_train = fit_patch.call_args[0]

        # check weights in training set
        weight_train = None
        if use_weights:
            assert "sample_weight" in fit_patch.call_args[1]
            weight_train = fit_patch.call_args[1]["sample_weight"]

        # check eval set
        eval_set_name, eval_weight_name = model.val_set_params
        assert eval_set_name in fit_patch.call_args[1]
        eval_set = fit_patch.call_args[1]["eval_set"]
        assert eval_set is not None
        assert isinstance(eval_set, list)
        eval_set = eval_set[0]

        weight = None
        if cb_available and isinstance(model, CatBoostModel):
            # CatBoost requires eval set as `Pool`
            from catboost import Pool

            assert isinstance(eval_set, Pool)
            X, y = eval_set.get_features(), eval_set.get_label()
            if use_weights:
                weight = np.array(eval_set.get_weight())

        else:
            assert isinstance(eval_set, tuple) and len(eval_set) == 2
            X, y = eval_set
            if use_weights:
                assert eval_weight_name in fit_patch.call_args[1]
                weight = fit_patch.call_args[1][eval_weight_name]
                assert isinstance(weight, list)
                weight = weight[0]

        # check same number of features for each dataset
        assert X.shape[1:] == X_train.shape[1:]
        assert y.shape[1:] == y_train.shape[1:]
        assert fit_patch.call_args[1]["early_stopping_rounds"] == 2
        if use_weights:
            assert weight_train.shape == y_train.shape
            assert weight.shape == y.shape

    @pytest.mark.parametrize("mode", [True, False])
    def test_integer_indexed_series(self, mode):
        values_target = np.random.rand(30)
        values_past_cov = np.random.rand(30)
        values_future_cov = np.random.rand(30)

        idx1 = pd.RangeIndex(start=0, stop=30, step=1)
        idx2 = pd.RangeIndex(start=10, stop=70, step=2)

        preds = []
        for idx in [idx1, idx2]:
            target = TimeSeries.from_times_and_values(idx, values_target)
            past_cov = TimeSeries.from_times_and_values(idx, values_past_cov)
            future_cov = TimeSeries.from_times_and_values(idx, values_future_cov)

            train, _ = target[:20], target[20:]

            model = LinearRegressionModel(
                lags=[-2, -1],
                lags_past_covariates=[-2, -1],
                lags_future_covariates=[0],
                multi_models=mode,
            )
            model.fit(
                series=train, past_covariates=past_cov, future_covariates=future_cov
            )

            preds.append(model.predict(n=10))

        # the predicted values should not depend on the time axis
        np.testing.assert_equal(preds[0].values(), preds[1].values())

        # the time axis returned by the second model should be as expected
        assert all(preds[1].time_index == pd.RangeIndex(start=50, stop=70, step=2))

    @pytest.mark.parametrize(
        "config",
        product(
            [
                ({"lags": [-3, -2, -1]}, {"lags": {"gaussian": 3}}),
                ({"lags": 3}, {"lags": {"gaussian": 3, "sine": 3}}),
                (
                    {"lags_past_covariates": 2},
                    {"lags_past_covariates": {"lin_past": 2}},
                ),
                (
                    {"lags_future_covariates": [-2, -1]},
                    {"lags_future_covariates": {"lin_future": [-2, -1]}},
                ),
                (
                    {"lags_future_covariates": [1, 2]},
                    {"lags_future_covariates": {"lin_future": [1, 2]}},
                ),
                (
                    {"lags": 5, "lags_future_covariates": [-2, 3]},
                    {
                        "lags": {
                            "gaussian": [-5, -4, -3, -2, -1],
                            "sine": [-5, -4, -3, -2, -1],
                        },
                        "lags_future_covariates": {
                            "lin_future": [-2, 3],
                            "sine_future": [-2, 3],
                        },
                    },
                ),
                (
                    {"lags": 5, "lags_future_covariates": [-2, 3]},
                    {
                        "lags": {
                            "gaussian": [-5, -4, -3, -2, -1],
                            "sine": [-5, -4, -3, -2, -1],
                        },
                        "lags_future_covariates": {
                            "sine_future": [-2, 3],
                            "default_lags": [-2, 3],
                        },
                    },
                ),
            ],
            [0, 5],
            [True, False],
        ),
    )
    def test_component_specific_lags_forecasts(self, config):
        """Verify that the same lags, defined using int/list or dictionaries yield the same results,
        including output_chunk_shift."""
        (list_lags, dict_lags), output_chunk_shift, multiple_series = config
        max_forecast = 3
        series, past_cov, future_cov = self.helper_generate_input_series_from_lags(
            list_lags,
            dict_lags,
            multiple_series,
            output_chunk_shift,
            max_forecast,
        )

        model = LinearRegressionModel(
            **list_lags, output_chunk_shift=output_chunk_shift
        )
        model.fit(
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )

        # the lags are specified for each component, individually
        model2 = LinearRegressionModel(
            **dict_lags, output_chunk_shift=output_chunk_shift
        )
        model2.fit(
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )

        if "lags_future_covariates" in list_lags:
            assert model.lags["future"] == [
                lag_ + output_chunk_shift
                for lag_ in list_lags["lags_future_covariates"]
            ]

            if "default_lags" in dict_lags["lags_future_covariates"]:
                # check that default lags
                default_components = (
                    model2.component_lags["future"].keys()
                    - dict_lags["lags_future_covariates"].keys()
                )
            else:
                default_components = dict()

            lags_specific = {
                comp_: (
                    dict_lags["lags_future_covariates"]["default_lags"]
                    if comp_ in default_components
                    else dict_lags["lags_future_covariates"][comp_]
                )
                for comp_ in model2.component_lags["future"]
            }
            assert model2.component_lags["future"] == {
                comp_: [lag_ + output_chunk_shift for lag_ in lags_]
                for comp_, lags_ in lags_specific.items()
            }

        # n == output_chunk_length
        s_ = series[0] if multiple_series else series
        pred_start_expected = s_.end_time() + (1 + output_chunk_shift) * s_.freq
        pred = model.predict(
            1,
            series=s_,
            past_covariates=(
                past_cov[0]
                if multiple_series and model.supports_past_covariates
                else None
            ),
            future_covariates=(
                future_cov[0]
                if multiple_series and model.supports_future_covariates
                else None
            ),
        )
        assert pred.start_time() == pred_start_expected
        pred2 = model2.predict(
            1,
            series=s_,
            past_covariates=(
                past_cov[0]
                if multiple_series and model2.supports_past_covariates
                else None
            ),
            future_covariates=(
                future_cov[0]
                if multiple_series and model2.supports_future_covariates
                else None
            ),
        )
        assert pred2.start_time() == pred_start_expected
        np.testing.assert_array_almost_equal(pred.values(), pred2.values())
        assert pred.time_index.equals(pred2.time_index)

        # auto-regression not supported for shifted output (tested in `test_output_shift`)
        if output_chunk_shift:
            return

        # n > output_chunk_length
        pred = model.predict(
            max_forecast,
            series=series[0] if multiple_series else None,
            past_covariates=(
                past_cov[0]
                if multiple_series and model.supports_past_covariates
                else None
            ),
            future_covariates=(
                future_cov[0]
                if multiple_series and model.supports_future_covariates
                else None
            ),
        )
        pred2 = model2.predict(
            max_forecast,
            series=series[0] if multiple_series else None,
            past_covariates=(
                past_cov[0]
                if multiple_series and model2.supports_past_covariates
                else None
            ),
            future_covariates=(
                future_cov[0]
                if multiple_series and model2.supports_future_covariates
                else None
            ),
        )
        np.testing.assert_array_almost_equal(pred.values(), pred2.values())
        assert pred.time_index.equals(pred2.time_index)

    @pytest.mark.parametrize(
        "config",
        product(
            [
                {"lags": {"gaussian": [-1, -3], "sine": [-2, -4, -6]}},
                {"lags_past_covariates": {"default_lags": 2}},
                {
                    "lags": {
                        "gaussian": [-5, -2, -1],
                        "sine": [-2, -1],
                    },
                    "lags_future_covariates": {
                        "lin_future": (1, 4),
                        "default_lags": (2, 2),
                    },
                },
                {
                    "lags": {
                        "default_lags": [-5, -4],
                    },
                    "lags_future_covariates": {
                        "sine_future": (1, 1),
                        "default_lags": [-2, 0, 1, 2],
                    },
                },
            ],
            [True, False],
        ),
    )
    def test_component_specific_lags(self, config):
        """Checking various combination of component-specific lags"""
        (dict_lags, multiple_series) = config
        multivar_target = "lags" in dict_lags and len(dict_lags["lags"]) > 1
        multivar_future_cov = (
            "lags_future_covariates" in dict_lags
            and len(dict_lags["lags_future_covariates"]) > 1
        )

        # create series based on the model parameters
        series = tg.gaussian_timeseries(length=20, column_name="gaussian")
        if multivar_target:
            series = series.stack(tg.sine_timeseries(length=20, column_name="sine"))

        future_cov = tg.linear_timeseries(length=30, column_name="lin_future")
        if multivar_future_cov:
            future_cov = future_cov.stack(
                tg.sine_timeseries(length=30, column_name="sine_future")
            )

        past_cov = tg.linear_timeseries(length=30, column_name="lin_past")

        if multiple_series:
            # second series have different component names
            series = [
                series,
                series.with_columns_renamed(
                    ["gaussian", "sine"][: series.width],
                    ["other", "names"][: series.width],
                )
                + 10,
            ]
            past_cov = [past_cov, past_cov]
            future_cov = [future_cov, future_cov]

        model = LinearRegressionModel(**dict_lags, output_chunk_length=4)
        model.fit(
            series=series,
            past_covariates=past_cov if model.supports_past_covariates else None,
            future_covariates=future_cov if model.supports_future_covariates else None,
        )
        # n < output_chunk_length
        model.predict(
            1,
            series=series[0] if multiple_series else None,
            past_covariates=(
                past_cov[0]
                if multiple_series and model.supports_past_covariates
                else None
            ),
            future_covariates=(
                future_cov[0]
                if multiple_series and model.supports_future_covariates
                else None
            ),
        )

        # n > output_chunk_length
        pred = model.predict(
            7,
            series=series[0] if multiple_series else None,
            past_covariates=(
                past_cov[0]
                if multiple_series and model.supports_past_covariates
                else None
            ),
            future_covariates=(
                future_cov[0]
                if multiple_series and model.supports_future_covariates
                else None
            ),
        )
        # check that lagged features are properly extracted during auto-regression
        if multivar_target:
            np.testing.assert_array_almost_equal(
                tg.sine_timeseries(length=27)[-7:].values(), pred["sine"].values()
            )

    @pytest.mark.parametrize(
        "config",
        product(
            [
                {"lags": [-1, -3]},
                {"lags_past_covariates": 2},
                {"lags_future_covariates": [-2, -1]},
                {"lags_future_covariates": [1, 2]},
                {
                    "lags": 5,
                    "lags_past_covariates": [-3, -1],
                },
                {"lags": [-5, -4], "lags_future_covariates": [-2, 0, 1, 2]},
                {
                    "lags": 5,
                    "lags_past_covariates": 4,
                    "lags_future_covariates": [-3, 1],
                },
                # check that component-specific lags with output_chunk_shift works
                {
                    "lags_past_covariates": {"lin_past": [-3, -1]},
                    "lags_future_covariates": [1, 2],
                },
                {
                    "lags_past_covariates": [-3, -1],
                    "lags_future_covariates": {"lin_future": [1, 2]},
                },
                {
                    "lags": {"gaussian": 5},
                    "lags_past_covariates": [-3, -1],
                    "lags_future_covariates": [1, 2],
                },
            ],
            [True, False],
            [3, 5],
            [1, 4],
        ),
    )
    def test_same_result_output_chunk_shift(self, config):
        """Tests that a model with that uses an output shift gets identical results for a multi-model
        without a shift. This only applies to the regressors that overlap.

        Example models:
        * non-shifted model with ocl=5, shift=0, multi_models=True
        * shifted model with ocl=2, shift=3, multi_models=True

        The 4th and 5th regressors from the non-shifted models should generate identical results as the 1st
        and 2nd regressor of the shifted model.
        """
        list_lags, multiple_series, output_chunk_shift, ocl_shifted = config
        ocl = output_chunk_shift + ocl_shifted
        max_forecast = ocl
        series, past_cov, future_cov = self.helper_generate_input_series_from_lags(
            list_lags,
            {},
            multiple_series,
            output_chunk_shift,
            max_forecast,
            output_chunk_length=ocl,
        )

        model = LinearRegressionModel(
            **list_lags, output_chunk_shift=0, output_chunk_length=ocl
        )

        # with output shift, future lags are shifted
        model_shift = LinearRegressionModel(
            **list_lags,
            output_chunk_shift=output_chunk_shift,
            output_chunk_length=ocl_shifted,
        )
        # adjusting the future lags should give identical models to non-shifted
        list_lags_adj = deepcopy(list_lags)
        # this loop works for both component-specific and non-component-specific future lags
        if "lags_future_covariates" in list_lags_adj:
            if isinstance(list_lags_adj["lags_future_covariates"], dict):
                for key in list_lags_adj["lags_future_covariates"]:
                    list_lags_adj["lags_future_covariates"][key] = [
                        lag_ - output_chunk_shift
                        for lag_ in list_lags_adj["lags_future_covariates"][key]
                    ]
            else:
                list_lags_adj["lags_future_covariates"] = [
                    lag_ - output_chunk_shift
                    for lag_ in list_lags_adj["lags_future_covariates"]
                ]
        model_shift_adj = LinearRegressionModel(
            **list_lags_adj,
            output_chunk_shift=output_chunk_shift,
            output_chunk_length=ocl_shifted,
        )

        if not multiple_series:
            series = [series]
            past_cov = [past_cov] if past_cov is not None else past_cov
            future_cov = [future_cov] if future_cov is not None else future_cov

        for m_ in [model, model_shift, model_shift_adj]:
            m_.fit(
                series=series,
                past_covariates=past_cov,
                future_covariates=future_cov,
            )

        pred = model.predict(
            ocl,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        pred_shift = model_shift.predict(
            ocl_shifted,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        pred_shift_adj = model_shift_adj.predict(
            ocl_shifted,
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
        )
        # expected shifted start is `output_chunk_shift` steps after non-shifted pred start
        for s_, pred_, pred_shift_, pred_shift_adj_ in zip(
            series, pred, pred_shift, pred_shift_adj
        ):
            pred_shift_start_expected = (
                s_.end_time() + (1 + output_chunk_shift) * s_.freq
            )
            assert pred_.start_time() == s_.end_time() + pred_.freq
            assert (
                pred_.end_time()
                == pred_shift_start_expected + (ocl_shifted - 1) * pred_.freq
            )
            assert pred_shift_.start_time() == pred_shift_start_expected
            assert (
                pred_shift_.end_time()
                == pred_shift_start_expected + (ocl_shifted - 1) * pred_shift_.freq
            )
            assert pred_shift_.time_index.equals(pred_shift_adj_.time_index)

            if "lags_future_covariates" not in list_lags:
                # without future lags, all lags should be identical between shift and non-shifted model
                np.testing.assert_almost_equal(
                    pred_[-ocl_shifted:].all_values(copy=False),
                    pred_shift_.all_values(copy=False),
                )
            else:
                # without future lags, the shifted model also shifts future lags
                with pytest.raises(AssertionError):
                    np.testing.assert_almost_equal(
                        pred_[-ocl_shifted:].all_values(copy=False),
                        pred_shift_.all_values(copy=False),
                    )

            # with adjusted future lags, the models should be identical
            np.testing.assert_almost_equal(
                pred_[-ocl_shifted:].all_values(copy=False),
                pred_shift_adj_.all_values(copy=False),
            )

    @pytest.mark.parametrize(
        "config",
        product(
            [
                {"lags": [-1, -3]},
                {"lags_past_covariates": 2},
                {"lags_future_covariates": [-2, -1]},
                {"lags_future_covariates": [1, 2]},
                {
                    "lags": 5,
                    "lags_past_covariates": [-3, -1],
                },
                {"lags": [-5, -4], "lags_future_covariates": [-2, 0, 1, 2]},
                {
                    "lags": 5,
                    "lags_past_covariates": 4,
                    "lags_future_covariates": [-3, 1],
                },
            ],
            [3, 7, 10],
        ),
    )
    def test_output_shift(self, config):
        """Tests shifted output for shift smaller than, equal to, and larger than output_chunk_length."""
        np.random.seed(0)
        lags, shift = config
        ocl = 7
        series = tg.gaussian_timeseries(
            length=28, start=pd.Timestamp("2000-01-01"), freq="d"
        )

        model_target_only = LinearRegressionModel(
            lags=3,
            output_chunk_length=ocl,
            output_chunk_shift=shift,
        )
        model_target_only.fit(series)

        # no auto-regression with shifted output
        with pytest.raises(ValueError) as err:
            _ = model_target_only.predict(n=ocl + 1)
        assert str(err.value).startswith("Cannot perform auto-regression")

        # pred starts with a shift
        for ocl_test in [ocl - 1, ocl]:
            pred = model_target_only.predict(n=ocl_test)
            assert pred.start_time() == series.end_time() + (shift + 1) * series.freq
            assert len(pred) == ocl_test
            assert pred.freq == series.freq

        series, past_cov, future_cov = self.helper_generate_input_series_from_lags(
            lags,
            {},
            multiple_series=False,
            output_chunk_shift=shift,
            max_forecast=ocl,
            output_chunk_length=ocl,
            add_length=2,  # add length for hist fc that don't use target lags
        )

        # model trained on encoders
        cov_support = []
        covs = {}
        if "lags_past_covariates" in lags:
            cov_support.append("past")
            covs["past_covariates"] = tg.datetime_attribute_timeseries(
                past_cov,
                attribute="dayofweek",
                add_length=0,
            )
        if "lags_future_covariates" in lags:
            cov_support.append("future")
            covs["future_covariates"] = tg.datetime_attribute_timeseries(
                future_cov,
                attribute="dayofweek",
                add_length=0,
            )

        if not cov_support:
            return

        add_encoders = {
            "datetime_attribute": {cov: ["dayofweek"] for cov in cov_support}
        }
        model_enc_shift = LinearRegressionModel(
            **lags,
            output_chunk_length=ocl,
            output_chunk_shift=shift,
            add_encoders=add_encoders,
        )
        model_enc_shift.fit(series)

        # model trained with identical covariates
        model_fc_shift = LinearRegressionModel(
            **lags,
            output_chunk_length=ocl,
            output_chunk_shift=shift,
        )
        model_fc_shift.fit(series, **covs)

        pred_enc = model_enc_shift.predict(n=ocl)
        pred_fc = model_fc_shift.predict(n=ocl)
        assert pred_enc == pred_fc

        # check that historical forecasts works properly
        hist_fc_start = -(ocl + shift)
        pred_last_hist_fc = model_fc_shift.predict(n=ocl, series=series[:hist_fc_start])
        # non-optimized hist fc
        hist_fc = model_fc_shift.historical_forecasts(
            series=series,
            start=hist_fc_start,
            start_format="position",
            retrain=False,
            forecast_horizon=ocl,
            last_points_only=False,
            enable_optimization=False,
            **covs,
        )
        assert len(hist_fc) == 1
        assert hist_fc[0] == pred_last_hist_fc
        # optimized hist fc, routine: last_points_only=False
        hist_fc_opt = model_fc_shift.historical_forecasts(
            series=series,
            start=hist_fc_start,
            start_format="position",
            retrain=False,
            forecast_horizon=ocl,
            last_points_only=False,
            enable_optimization=True,
            **covs,
        )
        assert len(hist_fc_opt) == 1
        assert hist_fc_opt[0].time_index.equals(pred_last_hist_fc.time_index)
        np.testing.assert_array_almost_equal(
            hist_fc_opt[0].values(copy=False), pred_last_hist_fc.values(copy=False)
        )

        # optimized hist fc, routine: last_points_only=True
        hist_fc_opt = model_fc_shift.historical_forecasts(
            series=series,
            start=hist_fc_start,
            start_format="position",
            retrain=False,
            forecast_horizon=ocl,
            last_points_only=True,
            enable_optimization=True,
            **covs,
        )
        assert isinstance(hist_fc_opt, TimeSeries)
        assert len(hist_fc_opt) == 1
        assert hist_fc_opt.start_time() == pred_last_hist_fc.end_time()
        np.testing.assert_array_almost_equal(
            hist_fc_opt.values(copy=False), pred_last_hist_fc[-1].values(copy=False)
        )

    @pytest.mark.parametrize("lpo", [True, False])
    def test_historical_forecasts_no_target_lags_with_static_covs(self, lpo):
        """Tests that historical forecasts work without target lags but with static covariates.
        For last_points_only `True` and `False`."""
        ocl = 7
        series = tg.linear_timeseries(
            length=28, start=pd.Timestamp("2000-01-01"), freq="d"
        ).with_static_covariates(pd.Series([1.0]))

        model = LinearRegressionModel(
            lags=None,
            lags_future_covariates=(3, 0),
            output_chunk_length=ocl,
            use_static_covariates=True,
        )
        model.fit(series, future_covariates=series)

        preds1 = model.historical_forecasts(
            series,
            future_covariates=series,
            retrain=False,
            enable_optimization=True,
            last_points_only=lpo,
        )
        preds2 = model.historical_forecasts(
            series,
            future_covariates=series,
            retrain=False,
            enable_optimization=False,
            last_points_only=lpo,
        )
        if lpo:
            preds1 = [preds1]
            preds2 = [preds2]

        for p1, p2 in zip(preds1, preds2):
            np.testing.assert_array_almost_equal(p1.values(), p2.values())

    @pytest.mark.parametrize(
        "config",
        product(
            [
                (RegressionModel, {}),
                (LinearRegressionModel, {}),
                (XGBModel, xgb_test_params),
            ]
            + ([(LightGBMModel, lgbm_test_params)] if lgbm_available else []),
            [True, False],
            [1, 2],
        ),
    )
    def test_encoders(self, config):
        (model_cls, model_kwargs), mode, ocl = config
        max_past_lag = -4
        max_future_lag = 4
        # target
        t1 = tg.linear_timeseries(
            start=pd.Timestamp("2000-01-01"), end=pd.Timestamp("2000-12-01"), freq="MS"
        )
        t2 = tg.linear_timeseries(
            start=pd.Timestamp("2001-01-01"), end=pd.Timestamp("2001-12-01"), freq="MS"
        )
        ts = [t1, t2]

        # past and future covariates longer than target
        n_comp = 2
        covs = TimeSeries.from_times_and_values(
            generate_index(
                start=pd.Timestamp("1999-01-01"),
                end=pd.Timestamp("2002-12-01"),
                freq="MS",
            ),
            values=np.random.randn(48, n_comp),
        )
        pc = [covs, covs]
        fc = [covs, covs]
        examples = ["past", "future", "mixed"]
        covariates_examples = {
            "past": {"past_covariates": pc},
            "future": {"future_covariates": fc},
            "mixed": {"past_covariates": pc, "future_covariates": fc},
        }
        encoder_examples = {
            "past": {"datetime_attribute": {"past": ["hour"]}},
            "future": {"cyclic": {"future": ["hour"]}},
            "mixed": {
                "datetime_attribute": {"past": ["hour"]},
                "cyclic": {"future": ["hour"]},
            },
        }
        model_pc_valid0 = model_cls(
            lags=2,
            add_encoders=encoder_examples["past"],
            multi_models=mode,
            output_chunk_length=ocl,
            **model_kwargs,
        )
        model_fc_valid0 = model_cls(
            lags=2,
            add_encoders=encoder_examples["future"],
            multi_models=mode,
            output_chunk_length=ocl,
            **model_kwargs,
        )
        model_mixed_valid0 = model_cls(
            lags=2,
            add_encoders=encoder_examples["mixed"],
            multi_models=mode,
            output_chunk_length=ocl,
            **model_kwargs,
        )

        # encoders will not generate covariates without lags
        for model in [model_pc_valid0, model_fc_valid0, model_mixed_valid0]:
            model.fit(ts)
            assert not model.encoders.encoding_available
            _ = model.predict(n=1, series=ts)
            _ = model.predict(n=3, series=ts)

        model_pc_valid0 = model_cls(
            lags_past_covariates=[-2],
            add_encoders=encoder_examples["past"],
            multi_models=mode,
            output_chunk_length=ocl,
            **model_kwargs,
        )
        model_fc_valid0 = model_cls(
            lags_future_covariates=[-1, 0],
            add_encoders=encoder_examples["future"],
            multi_models=mode,
            output_chunk_length=ocl,
            **model_kwargs,
        )
        model_mixed_valid0 = model_cls(
            lags_past_covariates=[-2, -1],
            lags_future_covariates=[-3, 3],
            add_encoders=encoder_examples["mixed"],
            multi_models=mode,
            output_chunk_length=ocl,
            **model_kwargs,
        )
        # check that fit/predict works with model internal covariate requirement checks
        for model in [model_pc_valid0, model_fc_valid0, model_mixed_valid0]:
            model.fit(ts)
            assert model.encoders.encoding_available
            _ = model.predict(n=1, series=ts)
            _ = model.predict(n=3, series=ts)

        model_pc_valid1 = model_cls(
            lags=2,
            lags_past_covariates=[max_past_lag, -1],
            add_encoders=encoder_examples["past"],
            multi_models=mode,
            output_chunk_length=ocl,
            **model_kwargs,
        )
        model_fc_valid1 = model_cls(
            lags=2,
            lags_future_covariates=[0, max_future_lag],
            add_encoders=encoder_examples["future"],
            multi_models=mode,
            output_chunk_length=ocl,
            **model_kwargs,
        )
        model_mixed_valid1 = model_cls(
            lags=2,
            lags_past_covariates=[max_past_lag, -1],
            lags_future_covariates=[0, max_future_lag],
            add_encoders=encoder_examples["mixed"],
            multi_models=mode,
            output_chunk_length=ocl,
            **model_kwargs,
        )

        for model, ex in zip(
            [model_pc_valid1, model_fc_valid1, model_mixed_valid1], examples
        ):
            covariates = covariates_examples[ex]
            # don't pass covariates, let them be generated by encoders. Test single target series input
            model_copy = deepcopy(model)
            model_copy.fit(ts[0])
            assert model_copy.encoders.encoding_available
            self.helper_test_encoders_settings(model_copy, ex)
            _ = model_copy.predict(n=1, series=ts)
            self.helper_compare_encoded_covs_with_ref(
                model_copy, ts, covariates, n=1, ocl=ocl, multi_model=mode
            )

            _ = model_copy.predict(n=3, series=ts)
            self.helper_compare_encoded_covs_with_ref(
                model_copy, ts, covariates, n=3, ocl=ocl, multi_model=mode
            )

            _ = model_copy.predict(n=8, series=ts)
            self.helper_compare_encoded_covs_with_ref(
                model_copy, ts, covariates, n=8, ocl=ocl, multi_model=mode
            )

            # manually pass covariates, let encoders add more
            model.fit(ts, **covariates)
            assert model.encoders.encoding_available
            self.helper_test_encoders_settings(model, ex)
            _ = model.predict(n=1, series=ts, **covariates)
            _ = model.predict(n=3, series=ts, **covariates)
            _ = model.predict(n=8, series=ts, **covariates)

    @pytest.mark.parametrize("config", product([True, False], [True, False]))
    def test_encoders_from_covariates_input(self, config):
        multi_models, extreme_lags = config
        series = tg.linear_timeseries(length=10, freq="MS")
        pc = tg.linear_timeseries(length=12, freq="MS")
        fc = tg.linear_timeseries(length=14, freq="MS")
        # 1 == output_chunk_length, 3 > output_chunk_length
        ns = [1, 3]
        model = self.helper_create_LinearModel(
            multi_models=multi_models, extreme_lags=extreme_lags
        )
        model.fit(series)
        for n in ns:
            _ = model.predict(n=n)
            with pytest.raises(ValueError):
                _ = model.predict(n=n, past_covariates=pc)
            with pytest.raises(ValueError):
                _ = model.predict(n=n, future_covariates=fc)
            with pytest.raises(ValueError):
                _ = model.predict(n=n, past_covariates=pc, future_covariates=fc)

        model = self.helper_create_LinearModel(
            multi_models=multi_models, extreme_lags=extreme_lags
        )
        for n in ns:
            model.fit(series, past_covariates=pc)
            _ = model.predict(n=n)
            _ = model.predict(n=n, past_covariates=pc)
            with pytest.raises(ValueError):
                _ = model.predict(n=n, future_covariates=fc)
            with pytest.raises(ValueError):
                _ = model.predict(n=n, past_covariates=pc, future_covariates=fc)

        model = self.helper_create_LinearModel(
            multi_models=multi_models, extreme_lags=extreme_lags
        )
        for n in ns:
            model.fit(series, future_covariates=fc)
            _ = model.predict(n=n)
            with pytest.raises(ValueError):
                _ = model.predict(n=n, past_covariates=pc)
            _ = model.predict(n=n, future_covariates=fc)
            with pytest.raises(ValueError):
                _ = model.predict(n=n, past_covariates=pc, future_covariates=fc)

        model = self.helper_create_LinearModel(
            multi_models=multi_models, extreme_lags=extreme_lags
        )
        for n in ns:
            model.fit(series, past_covariates=pc, future_covariates=fc)
            _ = model.predict(n=n)
            _ = model.predict(n=n, past_covariates=pc)
            _ = model.predict(n=n, future_covariates=fc)
            _ = model.predict(n=n, past_covariates=pc, future_covariates=fc)

    @staticmethod
    def helper_compare_encoded_covs_with_ref(
        model, ts, covariates, n, ocl, multi_model
    ):
        """checks that covariates generated by encoders fulfill the requirements compared to some
        reference covariates:
        What has to match:
        - the types should match, i.e., past and / or future covariates
        - same number of covariate TimeSeries in the list/sequence
        - generated/encoded covariates at training time must have the same start time as reference
        - generated/encoded covariates at prediction time must have the same end time as reference
        """

        def generate_expected_times(ts, n_predict=0) -> dict:
            """generates expected start and end times for the corresponding covariates."""
            freq = ts[0].freq

            def to_ts(dt):
                return pd.Timestamp(dt)

            def train_start_end(start_base, end_base):
                start = to_ts(start_base) - int(not multi_model) * (ocl - 1) * freq
                if not n_predict:
                    end = to_ts(end_base) - (ocl - 1) * freq
                else:
                    end = to_ts(end_base) + freq * max(n_predict - ocl, 0)
                return start, end

            if not n_predict:
                # expected train start, and end
                pc1_start, pc1_end = train_start_end("1999-11-01", "2000-11-01")
                pc2_start, pc2_end = train_start_end("2000-11-01", "2001-11-01")
                fc1_start, fc1_end = train_start_end("2000-03-01", "2001-04-01")
                fc2_start, fc2_end = train_start_end("2001-03-01", "2002-04-01")
            else:
                # expected inference start, and end
                pc1_start, pc1_end = train_start_end("2000-09-01", "2000-12-01")
                pc2_start, pc2_end = train_start_end("2001-09-01", "2001-12-01")
                fc1_start, fc1_end = train_start_end("2001-01-01", "2001-05-01")
                fc2_start, fc2_end = train_start_end("2002-01-01", "2002-05-01")

            times = {
                "pc_start": [pc1_start, pc2_start],
                "pc_end": [pc1_end, pc2_end],
                "fc_start": [fc1_start, fc2_start],
                "fc_end": [fc1_end, fc2_end],
            }
            return times

        covs_reference = (
            covariates.get("past_covariates"),
            covariates.get("future_covariates"),
        )
        covs_generated_train = model.encoders.encode_train(target=ts)
        covs_generated_infer = model.encoders.encode_inference(n=n, target=ts)

        refer_past, refer_future = covs_reference[0], covs_reference[1]
        train_past, train_future = covs_generated_train[0], covs_generated_train[1]
        infer_past, infer_future = covs_generated_infer[0], covs_generated_infer[1]

        t_train = generate_expected_times(ts)
        t_infer = generate_expected_times(ts, n_predict=n)
        if train_past is None:
            assert infer_past is None and refer_past is None
        else:
            assert all([
                isinstance(el, list) for el in [train_past, infer_past, refer_past]
            ])
            assert len(train_past) == len(infer_past) == len(refer_past)
            assert all([
                t_p.start_time() == tp_s
                for t_p, tp_s in zip(train_past, t_train["pc_start"])
            ])
            assert all([
                t_p.end_time() == tp_e
                for t_p, tp_e in zip(train_past, t_train["pc_end"])
            ])
            assert all([
                i_p.start_time() == ip_s
                for i_p, ip_s in zip(infer_past, t_infer["pc_start"])
            ])
            assert all([
                i_p.end_time() == ip_e
                for i_p, ip_e in zip(infer_past, t_infer["pc_end"])
            ])

        if train_future is None:
            assert infer_future is None and refer_future is None
        else:
            assert all([
                isinstance(el, list)
                for el in [train_future, infer_future, refer_future]
            ])
            assert len(train_future) == len(infer_future) == len(refer_future)
            assert all([
                t_f.start_time() == tf_s
                for t_f, tf_s in zip(train_future, t_train["fc_start"])
            ])
            assert all([
                t_f.end_time() == tf_e
                for t_f, tf_e in zip(train_future, t_train["fc_end"])
            ])
            assert all([
                i_f.start_time() == if_s
                for i_f, if_s in zip(infer_future, t_infer["fc_start"])
            ])
            assert all([
                i_f.end_time() == if_e
                for i_f, if_e in zip(infer_future, t_infer["fc_end"])
            ])

    @staticmethod
    def helper_test_encoders_settings(model, example: str):
        if example == "past":
            assert model.encoders.takes_past_covariates
            assert len(model.encoders.past_encoders) == 1
            assert isinstance(
                model.encoders.past_encoders[0], PastDatetimeAttributeEncoder
            )
            assert not model.encoders.takes_future_covariates
            assert len(model.encoders.future_encoders) == 0
        elif example == "future":
            assert not model.encoders.takes_past_covariates
            assert len(model.encoders.past_encoders) == 0
            assert model.encoders.takes_future_covariates
            assert len(model.encoders.future_encoders) == 1
            assert isinstance(model.encoders.future_encoders[0], FutureCyclicEncoder)
        else:  # "mixed"
            assert model.encoders.takes_past_covariates
            assert len(model.encoders.past_encoders) == 1
            assert isinstance(
                model.encoders.past_encoders[0], PastDatetimeAttributeEncoder
            )
            assert model.encoders.takes_future_covariates
            assert len(model.encoders.future_encoders) == 1
            assert isinstance(model.encoders.future_encoders[0], FutureCyclicEncoder)

    @pytest.mark.skipif(not lgbm_available, reason="requires lightgbm")
    def test_quality_forecast_with_categorical_covariates(self):
        """Test case: two time series, a full sine wave series and a sine wave series
        with some irregularities every other period. Only models which use categorical
        static covariates should be able to recognize the underlying curve type when input for prediction is only a
        sine wave
        See the test case in section 6 from
        https://github.com/unit8co/darts/blob/master/examples/15-static-covariates.ipynb

        """
        # full sine wave series
        period = 20
        sine_series = tg.sine_timeseries(
            length=4 * period,
            value_frequency=1 / period,
            column_name="smooth",
            freq="h",
        ).with_static_covariates(pd.DataFrame(data={"curve_type": [1]}))

        # irregular sine wave series with linear ramp every other period
        sine_vals = sine_series.values()
        linear_vals = np.expand_dims(np.linspace(1, -1, num=19), -1)
        sine_vals[21:40] = linear_vals
        sine_vals[61:80] = linear_vals
        irregular_series = TimeSeries.from_times_and_values(
            values=sine_vals, times=sine_series.time_index, columns=["irregular"]
        ).with_static_covariates(pd.DataFrame(data={"curve_type": [0]}))

        def fit_predict(model, train_series, predict_series):
            """perform model training and prediction"""
            model.fit(train_series)
            return model.predict(n=int(period / 2), series=predict_series)

        def get_model_params():
            """generate model parameters"""
            return {
                "lags": int(period / 2),
                "output_chunk_length": int(period / 2),
                "verbose": -1,
            }

        # test case without using categorical static covariates
        train_series_no_cat = [
            sine_series.with_static_covariates(None),
            irregular_series.with_static_covariates(None),
        ]
        # test case using categorical static covariates
        train_series_cat = [sine_series, irregular_series]
        for model_no_cat, model_cat in zip(
            [LightGBMModel(**get_model_params())],
            [
                LightGBMModel(
                    categorical_static_covariates=["curve_type"], **get_model_params()
                ),
            ],
        ):
            preds_no_cat = fit_predict(
                model_no_cat,
                train_series_no_cat,
                predict_series=[series[:60] for series in train_series_no_cat],
            )
            preds_cat = fit_predict(
                model_cat,
                train_series_cat,
                predict_series=[series[:60] for series in train_series_cat],
            )

            # categorical covariates make model aware of the underlying curve type -> improves rmse
            rmses_no_cat = rmse(train_series_cat, preds_no_cat)
            rmses_cat = rmse(train_series_cat, preds_cat)
            assert all([
                rmse_no_cat > rmse_cat
                for rmse_no_cat, rmse_cat in zip(rmses_no_cat, rmses_cat)
            ])

    @pytest.mark.skipif(not lgbm_available, reason="requires lightgbm")
    @pytest.mark.parametrize(
        "model",
        (
            [
                LightGBMModel(
                    lags=1,
                    lags_past_covariates=1,
                    output_chunk_length=1,
                    categorical_past_covariates=[
                        "does_not_exist",
                        "past_cov_cat_dummy",
                    ],
                    categorical_static_covariates=["product_id"],
                    **lgbm_test_params,
                ),
                LightGBMModel(
                    lags=1,
                    lags_past_covariates=1,
                    output_chunk_length=1,
                    categorical_past_covariates=[
                        "past_cov_cat_dummy",
                    ],
                    categorical_static_covariates=["does_not_exist"],
                    **lgbm_test_params,
                ),
                LightGBMModel(
                    lags=1,
                    lags_past_covariates=1,
                    output_chunk_length=1,
                    categorical_future_covariates=["does_not_exist"],
                    **lgbm_test_params,
                ),
            ]
            if lgbm_available
            else []
        ),
    )
    def test_fit_with_categorical_features_raises_error(self, model):
        (
            series,
            past_covariates,
            future_covariates,
        ) = self.inputs_for_tests_categorical_covariates
        with pytest.raises(ValueError):
            model.fit(
                series=series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )

    @pytest.mark.skipif(not lgbm_available, reason="requires lightgbm")
    def test_get_categorical_features_helper(self):
        """Test helper function responsible for retrieving indices of categorical features"""
        (
            series,
            past_covariates,
            future_covariates,
        ) = self.inputs_for_tests_categorical_covariates
        (
            indices,
            column_names,
        ) = self.lgbm_w_categorical_covariates._get_categorical_features(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        assert indices == [2, 3, 5]
        assert column_names == [
            "past_cov_past_cov_cat_dummy_lag-1",
            "fut_cov_fut_cov_promo_mechanism_lag1",
            "product_id",
        ]

    @pytest.mark.skipif(not lgbm_available, reason="requires lightgbm")
    @patch.object(
        (
            darts.models.forecasting.lgbm.lgb.LGBMRegressor
            if lgbm_available
            else darts.models.utils.NotImportedModule
        ),
        "fit",
    )
    def test_lgbm_categorical_features_passed_to_fit_correctly(self, lgb_fit_patch):
        """Test whether the categorical features are passed to LightGBMRegressor"""
        (
            series,
            past_covariates,
            future_covariates,
        ) = self.inputs_for_tests_categorical_covariates
        self.lgbm_w_categorical_covariates.fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # Check that mocked super.fit() method was called with correct categorical_feature argument
        args, kwargs = lgb_fit_patch.call_args
        (
            cat_param_name,
            cat_param_default,
        ) = self.lgbm_w_categorical_covariates._categorical_fit_param
        assert kwargs[cat_param_name] == [2, 3, 5]

    def helper_create_LinearModel(self, multi_models=True, extreme_lags=False):
        if not extreme_lags:
            lags, lags_pc, lags_fc = 3, 3, [-3, -2, -1, 0]
        else:
            lags, lags_pc, lags_fc = None, [-3], [1]
        return LinearRegressionModel(
            lags=lags,
            lags_past_covariates=lags_pc,
            lags_future_covariates=lags_fc,
            output_chunk_length=1,
            multi_models=multi_models,
            add_encoders={
                "datetime_attribute": {
                    "past": ["month", "dayofweek"],
                    "future": ["month", "dayofweek"],
                }
            },
        )

    def helper_generate_input_series_from_lags(
        self,
        list_lags,
        dict_lags,
        multiple_series,
        output_chunk_shift,
        max_forecast,
        output_chunk_length: int = 1,
        add_length: int = 0,
    ):
        np.random.seed(0)
        if dict_lags:
            multivar_target = "lags" in dict_lags and len(dict_lags["lags"]) > 1
            multivar_future_cov = (
                "lags_future_covariates" in dict_lags
                and len(dict_lags["lags_future_covariates"]) > 1
            )
        else:
            multivar_target = False
            multivar_future_cov = False

        # the lags are identical across the components for each series
        model = LinearRegressionModel(
            **list_lags,
            output_chunk_shift=output_chunk_shift,
            output_chunk_length=output_chunk_length,
        )
        autoreg_add_steps = max(max_forecast - model.output_chunk_length, 0)

        # create series based on the model parameters
        n_s = model.min_train_series_length + add_length
        series = tg.gaussian_timeseries(length=n_s, column_name="gaussian")
        if multivar_target:
            series = series.stack(tg.sine_timeseries(length=n_s, column_name="sine"))

        if model.supports_future_covariates:
            # prepend values if not target lags are used
            if "target" not in model.lags and min(model.lags["future"]) < 0:
                prep = abs(min(model.lags["future"]))
            else:
                prep = 0

            # minimum future covariates length
            n_fc = n_s + max(model.lags["future"]) + 1 + autoreg_add_steps
            future_cov = tg.gaussian_timeseries(
                start=series.start_time() - prep * series.freq,
                length=n_fc + prep,
                column_name="lin_future",
            )
            if multivar_future_cov:
                future_cov = future_cov.stack(
                    tg.gaussian_timeseries(length=n_fc, column_name="sine_future")
                )
        else:
            future_cov = None

        if model.supports_past_covariates:
            # prepend values if not target lags are used
            if "target" not in model.lags:
                prep = abs(min(model.lags["past"]))
            else:
                prep = 0

            # minimum past covariates length
            n_pc = n_s + autoreg_add_steps

            past_cov = tg.gaussian_timeseries(
                start=series.start_time() - prep * series.freq,
                length=n_pc + prep,
                column_name="lin_past",
            )
        else:
            past_cov = None

        if multiple_series:
            # second series have different component names
            series = [
                series,
                series.with_columns_renamed(
                    ["gaussian", "sine"][: series.width],
                    ["other", "names"][: series.width],
                )
                + 10,
            ]
            past_cov = [past_cov, past_cov] if past_cov else None
            future_cov = [future_cov, future_cov] if future_cov else None
        return series, past_cov, future_cov


class TestProbabilisticRegressionModels:
    models_cls_kwargs_errs = [
        (
            LinearRegressionModel,
            {
                "lags": 2,
                "likelihood": "quantile",
                "random_state": 42,
                "multi_models": True,
            },
            0.6,
        ),
        (
            LinearRegressionModel,
            {
                "lags": 2,
                "likelihood": "poisson",
                "random_state": 42,
                "multi_models": True,
            },
            0.6,
        ),
        (
            XGBModel,
            {
                "lags": 2,
                "likelihood": "poisson",
                "multi_models": True,
                **xgb_test_params,
            },
            0.6,
        ),
        (
            XGBModel,
            {
                "lags": 2,
                "likelihood": "quantile",
                "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9],
                "multi_models": True,
                **xgb_test_params,
            },
            0.4,
        ),
    ]
    if lgbm_available:
        models_cls_kwargs_errs += [
            (
                LightGBMModel,
                {
                    "lags": 2,
                    "likelihood": "quantile",
                    "multi_models": True,
                    **lgbm_test_params,
                },
                0.4,
            ),
            (
                LightGBMModel,
                {
                    "lags": 2,
                    "likelihood": "quantile",
                    "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "multi_models": True,
                    **lgbm_test_params,
                },
                0.4,
            ),
            (
                LightGBMModel,
                {
                    "lags": 2,
                    "likelihood": "poisson",
                    "multi_models": True,
                    **lgbm_test_params,
                },
                0.6,
            ),
        ]
    if cb_available:
        models_cls_kwargs_errs += [
            (
                CatBoostModel,
                {
                    "lags": 2,
                    "likelihood": "quantile",
                    "multi_models": True,
                    **cb_test_params,
                },
                0.05,
            ),
            (
                CatBoostModel,
                {
                    "lags": 2,
                    "likelihood": "quantile",
                    "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9],
                    "multi_models": True,
                    **cb_test_params,
                },
                0.05,
            ),
            (
                CatBoostModel,
                {
                    "lags": 2,
                    "likelihood": "poisson",
                    "multi_models": True,
                    **cb_test_params,
                },
                0.6,
            ),
            (
                CatBoostModel,
                {
                    "lags": 2,
                    "likelihood": "gaussian",
                    "multi_models": True,
                    **cb_test_params,
                },
                0.05,
            ),
        ]

    constant_ts = tg.constant_timeseries(length=200, value=0.5)
    constant_noisy_ts = constant_ts + tg.gaussian_timeseries(length=200, std=0.1)
    constant_multivar_ts = constant_ts.stack(constant_ts)
    constant_noisy_multivar_ts = constant_noisy_ts.stack(constant_noisy_ts)
    num_samples = 5

    @pytest.mark.parametrize("config", product(models_cls_kwargs_errs, [True, False]))
    def test_fit_predict_determinism(self, config):
        (model_cls, model_kwargs, _), mode = config
        # whether the first predictions of two models initiated with the same random state are the same
        model_kwargs["multi_models"] = mode
        model = model_cls(**model_kwargs)
        model.fit(self.constant_noisy_multivar_ts)
        pred1 = model.predict(n=10, num_samples=2).values()

        model = model_cls(**model_kwargs)
        model.fit(self.constant_noisy_multivar_ts)
        pred2 = model.predict(n=10, num_samples=2).values()

        assert (pred1 == pred2).all()

        # test whether the next prediction of the same model is different
        pred3 = model.predict(n=10, num_samples=2).values()
        assert (pred2 != pred3).any()

    @pytest.mark.parametrize("config", product(models_cls_kwargs_errs, [True, False]))
    def test_probabilistic_forecast_accuracy_univariate(self, config):
        (model_cls, model_kwargs, err), mode = config
        model_kwargs["multi_models"] = mode
        model = model_cls(**model_kwargs)
        self.helper_test_probabilistic_forecast_accuracy(
            model,
            err,
            self.constant_ts,
            self.constant_noisy_ts,
        )

    @pytest.mark.parametrize("config", product(models_cls_kwargs_errs, [True, False]))
    def test_probabilistic_forecast_accuracy_multivariate(self, config):
        (model_cls, model_kwargs, err), mode = config
        model_kwargs["multi_models"] = mode
        model = model_cls(**model_kwargs)
        if model.supports_multivariate:
            self.helper_test_probabilistic_forecast_accuracy(
                model,
                err,
                self.constant_multivar_ts,
                self.constant_noisy_multivar_ts,
            )

    def helper_test_probabilistic_forecast_accuracy(self, model, err, ts, noisy_ts):
        model.fit(noisy_ts[:100])
        pred = model.predict(n=100, num_samples=100)

        # test accuracy of the median prediction compared to the noiseless ts
        mae_err_median = mae(ts[100:], pred)
        assert mae_err_median < err

        # test accuracy for increasing quantiles between 0.7 and 1 (it should ~decrease, mae should ~increase)
        tested_quantiles = [0.7, 0.8, 0.9, 0.99]
        mae_err = mae_err_median
        for quantile in tested_quantiles:
            new_mae = mae(ts[100:], pred.quantile_timeseries(quantile=quantile))
            assert mae_err < new_mae + 0.1
            mae_err = new_mae

        # test accuracy for decreasing quantiles between 0.3 and 0 (it should ~decrease, mae should ~increase)
        tested_quantiles = [0.3, 0.2, 0.1, 0.01]
        mae_err = mae_err_median
        for quantile in tested_quantiles:
            new_mae = mae(ts[100:], pred.quantile_timeseries(quantile=quantile))
            assert mae_err < new_mae + 0.1
            mae_err = new_mae
