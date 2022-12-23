import copy
import functools
import math
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

import darts
from darts import TimeSeries
from darts.dataprocessing.encoders import (
    FutureCyclicEncoder,
    PastDatetimeAttributeEncoder,
)
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.logging import get_logger
from darts.metrics import mae, rmse
from darts.models import (
    CatBoostModel,
    LightGBMModel,
    LinearRegressionModel,
    RandomForest,
    RegressionModel,
    XGBModel,
)
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

# from sklearn.multioutput import MultiOutputRegressor
from darts.utils.multioutput import MultiOutputRegressor

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


class RegressionModelsTestCase(DartsBaseTestClass):

    np.random.seed(42)

    # default regression models
    models = [
        RandomForest,
        LinearRegressionModel,
        RegressionModel,
        LightGBMModel,
        CatBoostModel,
    ]

    # register likelihood regression models
    QuantileCatBoostModel = partialclass(
        CatBoostModel,
        likelihood="quantile",
        quantiles=[0.05, 0.5, 0.95],
        random_state=42,
    )
    PoissonCatBoostModel = partialclass(
        CatBoostModel,
        likelihood="poisson",
        random_state=42,
    )
    NormalCatBoostModel = partialclass(
        CatBoostModel,
        likelihood="gaussian",
        random_state=42,
    )
    QuantileLightGBMModel = partialclass(
        LightGBMModel,
        likelihood="quantile",
        quantiles=[0.05, 0.5, 0.95],
        random_state=42,
    )
    PoissonLightGBMModel = partialclass(
        LightGBMModel, likelihood="poisson", random_state=42
    )
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
        random_state=42,
    )
    QuantileXGBModel = partialclass(
        XGBModel,
        likelihood="quantile",
        random_state=42,
    )
    # targets for poisson regression must be positive, so we exclude them for some tests
    models.extend(
        [
            QuantileLightGBMModel,
            QuantileLinearRegressionModel,
            QuantileCatBoostModel,
            PoissonLightGBMModel,
            PoissonLinearRegressionModel,
            PoissonCatBoostModel,
            NormalCatBoostModel,
            PoissonXGBModel,
            QuantileXGBModel,
        ]
    )

    univariate_accuracies = [
        0.03,  # RandomForest
        1e-13,  # LinearRegressionModel
        1e-13,  # RegressionModel
        0.3,  # LightGBMModel
        0.75,  # CatBoostModel
        0.5,  # QuantileLightGBMModel
        0.8,  # QuantileLinearRegressionModel
        1e-03,  # QuantileCatBoostModel
        0.4,  # PoissonLightGBMModel
        0.4,  # PoissonLinearRegressionModel
        1e-01,  # PoissonCatBoostModel
        1e-05,  # NormalCatBoostModel
        1e-01,  # PoissonXGBModel
        0.5,  # QuantileXGBModel
    ]
    multivariate_accuracies = [
        0.3,
        1e-13,
        1e-13,
        0.4,
        0.75,  # CatBoostModel
        0.4,  # QuantileLightGBMModel
        0.8,  # QuantileLinearRegressionModel
        1e-03,
        0.4,
        0.4,
        0.15,
        1e-05,
        0.15,
        0.4,
    ]
    multivariate_multiseries_accuracies = [
        0.05,  # RandomForest
        1e-13,  # LinearRegressionModel
        1e-13,  # RegressionModel
        0.05,  # LightGBMModel
        0.75,  # CatBoostModel
        0.4,  # QuantileLightGBMModel
        0.8,  # QuantileLinearRegressionModel
        1e-03,  # QuantileCatBoostModel
        0.4,  # PoissonLightGBMModel
        0.4,  # PoissonLinearRegressionModel
        1e-01,  # PoissonCatBoostModel
        1e-03,  # NormalCatBoostModel
        1e-01,  # PoissonXGBModel
        0.4,  # QuantileXGBModel
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
        multi_models_modes = [True, False]
        for mode in multi_models_modes:
            for model in self.models:
                # TESTING SINGLE INT
                # testing lags
                model_instance = model(lags=5, multi_models=mode)
                self.assertEqual(
                    model_instance.lags.get("target"), [-5, -4, -3, -2, -1]
                )
                # testing lags_past_covariates
                model_instance = model(
                    lags=None, lags_past_covariates=3, multi_models=mode
                )
                self.assertEqual(model_instance.lags.get("past"), [-3, -2, -1])
                # testing lags_future covariates
                model_instance = model(
                    lags=None, lags_future_covariates=(3, 5), multi_models=mode
                )
                self.assertEqual(
                    model_instance.lags.get("future"), [-3, -2, -1, 0, 1, 2, 3, 4]
                )

                # TESTING LIST of int
                # lags
                values = [-5, -3, -1]
                model_instance = model(lags=values, multi_models=mode)
                self.assertEqual(model_instance.lags.get("target"), values)
                # testing lags_past_covariates
                model_instance = model(lags_past_covariates=values, multi_models=mode)
                self.assertEqual(model_instance.lags.get("past"), values)
                # testing lags_future_covariates

                with self.assertRaises(ValueError):
                    model(multi_models=mode)
                with self.assertRaises(ValueError):
                    model(lags=0, multi_models=mode)
                with self.assertRaises(ValueError):
                    model(lags=[-1, 0], multi_models=mode)
                with self.assertRaises(ValueError):
                    model(lags=[3, 5], multi_models=mode)
                with self.assertRaises(ValueError):
                    model(lags=[-3, -5.0], multi_models=mode)
                with self.assertRaises(ValueError):
                    model(lags=-5, multi_models=mode)
                with self.assertRaises(ValueError):
                    model(lags=3.6, multi_models=mode)
                with self.assertRaises(ValueError):
                    model(lags=None, lags_past_covariates=False, multi_models=mode)
                with self.assertRaises(ValueError):
                    model(lags=None, multi_models=mode)
                with self.assertRaises(ValueError):
                    model(lags=5, lags_future_covariates=True, multi_models=mode)
                with self.assertRaises(ValueError):
                    model(lags=5, lags_future_covariates=(1, -3), multi_models=mode)
                with self.assertRaises(ValueError):
                    model(lags=5, lags_future_covariates=(1, 2, 3), multi_models=mode)
                with self.assertRaises(ValueError):
                    model(lags=5, lags_future_covariates=(1, True), multi_models=mode)
                with self.assertRaises(ValueError):
                    model(lags=5, lags_future_covariates=(1, 1.0), multi_models=mode)

    def test_training_data_creation(self):
        multi_models_modes = [True, False]
        for mode in multi_models_modes:
            # testing _get_training_data function
            model_instance = RegressionModel(
                lags=self.lags_1["target"],
                lags_past_covariates=self.lags_1["past"],
                lags_future_covariates=self.lags_1["future"],
                multi_models=mode,
            )

            max_samples_per_ts = 17

            training_samples, training_labels = model_instance._create_lagged_data(
                target_series=self.target_series,
                past_covariates=self.past_covariates,
                future_covariates=self.future_covariates,
                max_samples_per_ts=max_samples_per_ts,
            )

            # checking number of dimensions
            self.assertEqual(len(training_samples.shape), 2)  # samples, features
            self.assertEqual(
                len(training_labels.shape), 2
            )  # samples, components (multivariate)
            self.assertEqual(training_samples.shape[0], training_labels.shape[0])
            self.assertEqual(
                training_samples.shape[0], len(self.target_series) * max_samples_per_ts
            )
            self.assertEqual(
                training_samples.shape[1],
                len(self.lags_1["target"]) * self.target_series[0].width
                + len(self.lags_1["past"]) * self.past_covariates[0].width
                + len(self.lags_1["future"]) * self.future_covariates[0].width,
            )

            # check last sample
            self.assertListEqual(
                list(training_samples[0, :]),
                [
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
                ],
            )
            self.assertListEqual(list(training_labels[0]), [82, 182, 282])

    def test_prediction_data_creation(self):
        multi_models_modes = [True, False]

        # assigning correct names to variables
        series = [ts[:-50] for ts in self.target_series]
        output_chunk_length = 5
        n = 12

        # prediction preprocessing start
        covariates = {
            "past": (self.past_covariates, self.lags_1.get("past")),
            "future": (self.future_covariates, self.lags_1.get("future")),
        }

        for mode in multi_models_modes:
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

                    covariate_matrices[cov_type] = np.stack(
                        covariate_matrices[cov_type]
                    )

            series_matrix = None
            if "target" in self.lags_1:
                series_matrix = np.stack(
                    [
                        ts.values(copy=False)[self.lags_1["target"][0] - shift :, :]
                        for ts in series
                    ]
                )
            # prediction preprocessing end
            self.assertTrue(
                all([lag >= 0 for lags in relative_cov_lags.values() for lag in lags])
            )

            if mode:
                # tests for multi_models = True
                self.assertEqual(
                    covariate_matrices["past"].shape,
                    (
                        len(series),
                        relative_cov_lags["past"][-1]
                        + (n_pred_steps - 1) * output_chunk_length
                        + 1,
                        covariates["past"][0][0].width,
                    ),
                )
                self.assertEqual(
                    covariate_matrices["future"].shape,
                    (
                        len(series),
                        relative_cov_lags["future"][-1]
                        + (n_pred_steps - 1) * output_chunk_length
                        + 1,
                        covariates["future"][0][0].width,
                    ),
                )
                self.assertEqual(
                    series_matrix.shape,
                    (len(series), -self.lags_1["target"][0], series[0].width),
                )
                self.assertListEqual(
                    list(covariate_matrices["past"][0, :, 0]),
                    [
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
                    ],
                )
                self.assertListEqual(
                    list(covariate_matrices["future"][0, :, 0]),
                    [
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
                    ],
                )
                self.assertListEqual(list(series_matrix[0, :, 0]), [48.0, 49.0, 50.0])
            else:
                # tests for multi_models = False
                self.assertEqual(
                    covariate_matrices["past"].shape,
                    (
                        len(series),
                        relative_cov_lags["past"][-1]
                        + (n_pred_steps - 1) * output_chunk_length
                        + (remaining_steps - 1)
                        + 1,
                        covariates["past"][0][0].width,
                    ),
                )
                self.assertEqual(
                    covariate_matrices["future"].shape,
                    (
                        len(series),
                        relative_cov_lags["future"][-1]
                        + (n_pred_steps - 1) * output_chunk_length
                        + (remaining_steps - 1)
                        + 1,
                        covariates["future"][0][0].width,
                    ),
                )
                self.assertEqual(
                    series_matrix.shape,
                    (len(series), -self.lags_1["target"][0] + shift, series[0].width),
                )
                self.assertListEqual(
                    list(covariate_matrices["past"][0, :, 0]),
                    [
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
                    ],
                )
                self.assertListEqual(
                    list(covariate_matrices["future"][0, :, 0]),
                    [
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
                    ],
                )
                self.assertListEqual(
                    list(series_matrix[0, :, 0]),
                    [44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0],
                )

    def test_static_covs_addition(self):

        static_covs1 = pd.DataFrame(
            data={
                "cont": [0.1, 0.2, 0.3],
                "cat": ["a", "b", "c"],  # should lead to 9 one-hot encoded columns
            }
        ).astype(dtype={"cat": "category"})

        static_covs2 = pd.DataFrame(data={"cont": [0.1, 0.2, 0.3]})

        # default transformer_num = MinMaxScaler()
        scaler = StaticCovariatesTransformer(transformer_cat=OneHotEncoder())
        ref_series = tg.linear_timeseries(length=10)
        series1 = TimeSeries.from_times_and_values(
            times=ref_series.time_index,
            values=np.concatenate([ref_series.values()] * 3, axis=1),
            columns=["comp1", "comp2", "comp3"],
            static_covariates=static_covs1,
        )
        series1 = scaler.fit_transform(series1)

        series2 = TimeSeries.from_times_and_values(
            times=ref_series.time_index,
            values=np.concatenate([ref_series.values() * 100] * 3, axis=1),
            columns=["comp1", "comp2", "comp3"],
            static_covariates=static_covs2,
        )

        series3 = TimeSeries.from_times_and_values(
            times=ref_series.time_index,
            values=np.concatenate([ref_series.values() * 200] * 3, axis=1),
            columns=["comp1", "comp2", "comp3"],
        )

        series4 = TimeSeries.from_times_and_values(
            times=ref_series.time_index,
            values=np.concatenate([ref_series.values()] * 3, axis=1),
            columns=["comp1", "comp2", "comp3"],
        )

        reg_model = RegressionModel(lags=1, output_chunk_length=1)
        all_series = [series1, series2, series3]
        max_samples = 5
        all_series_width = series1.n_components
        max_scovs_width = max(
            s.static_covariates_values(copy=False).reshape(1, -1).shape[1]
            for s in all_series
            if s.has_static_covariates
        )

        # no static covs
        features = reg_model._create_lagged_data(
            series3, None, None, max_samples_per_ts=max_samples
        )[0]
        self.assertEqual(features.shape, (5, 3))

        # static covs with different dims
        features = reg_model._create_lagged_data(
            all_series, None, None, max_samples_per_ts=max_samples
        )[0]
        self.assertEqual(
            features.shape,
            (max_samples * len(all_series), all_series_width + max_scovs_width),
        )

        # no static covs at prediction but static covs at training
        reg_model.fit(all_series)
        pred_features = reg_model._create_lagged_data(
            series4, None, None, max_samples_per_ts=1
        )[
            0
        ]  # simulates features prep at prediction time
        self.assertEqual(pred_features.shape, (1, all_series_width + max_scovs_width))

    def test_static_cov_accuracy(self):
        # based on : https://unit8co.github.io/darts/examples/15-static-covariates.html

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
            pd.DataFrame(data={"curve_type": ["smooth"]})
        )
        irregular_series_st_cat = irregular_series.with_static_covariates(
            pd.DataFrame(data={"curve_type": ["non_smooth"]})
        )
        train_series_static_cov = [sine_series_st_cat, irregular_series_st_cat]

        scaler = StaticCovariatesTransformer(transformer_cat=OneHotEncoder())
        train_series_static_cov = scaler.fit_transform(train_series_static_cov)

        # when
        model_no_static_cov = RandomForest(lags=period // 2, bootstrap=False)
        model_no_static_cov.fit(train_series_no_cov)
        predict_series_no_cov = [series[:60] for series in train_series_no_cov]
        pred_no_static_cov = model_no_static_cov.predict(
            n=int(period / 2), series=predict_series_no_cov
        )

        model_static_cov = RandomForest(lags=period // 2, bootstrap=False)
        model_static_cov.fit(train_series_static_cov)
        predict_series_static_cov = [series[:60] for series in train_series_static_cov]
        pred_static_cov = model_static_cov.predict(
            n=int(period / 2), series=predict_series_static_cov
        )

        # then
        for series, ps_no_st, ps_st_cat in zip(
            train_series_static_cov, pred_no_static_cov, pred_static_cov
        ):
            rmses = [rmse(series, ps) for ps in [ps_no_st, ps_st_cat]]

            self.assertLess(rmses[1], rmses[0])

    def test_models_runnability(self):
        train_y, test_y = self.sine_univariate1.split_before(0.7)
        multi_models_modes = [True, False]
        for mode in multi_models_modes:
            for model in self.models:
                # testing past covariates
                with self.assertRaises(ValueError):
                    # testing lags_past_covariates None but past_covariates during training
                    model_instance = model(
                        lags=4, lags_past_covariates=None, multi_models=mode
                    )
                    model_instance.fit(
                        series=self.sine_univariate1,
                        past_covariates=self.sine_multivariate1,
                    )

                with self.assertRaises(ValueError):
                    # testing lags_past_covariates but no past_covariates during fit
                    model_instance = model(
                        lags=4, lags_past_covariates=3, multi_models=mode
                    )
                    model_instance.fit(series=self.sine_univariate1)

                # testing future_covariates
                with self.assertRaises(ValueError):
                    # testing lags_future_covariates None but future_covariates during training
                    model_instance = model(
                        lags=4, lags_future_covariates=None, multi_models=mode
                    )
                    model_instance.fit(
                        series=self.sine_univariate1,
                        future_covariates=self.sine_multivariate1,
                    )

                with self.assertRaises(ValueError):
                    # testing lags_covariate but no covariate during fit
                    model_instance = model(
                        lags=4, lags_future_covariates=3, multi_models=mode
                    )
                    model_instance.fit(series=self.sine_univariate1)

                # testing input_dim
                model_instance = model(
                    lags=4, lags_past_covariates=2, multi_models=mode
                )
                model_instance.fit(
                    series=train_y,
                    past_covariates=self.sine_univariate1.stack(self.sine_univariate1),
                )

                self.assertEqual(
                    model_instance.input_dim, {"target": 1, "past": 2, "future": None}
                )

                with self.assertRaises(ValueError):
                    prediction = model_instance.predict(n=len(test_y) + 2)

                # while it should work with n = 1
                prediction = model_instance.predict(n=1)
                self.assertEqual(len(prediction), 1)

    def test_fit(self):
        multi_models_modes = [True, False]
        for mode in multi_models_modes:
            for model in self.models:

                # test fitting both on univariate and multivariate timeseries
                for series in [self.sine_univariate1, self.sine_multivariate2]:
                    with self.assertRaises(ValueError):
                        model_instance = model(
                            lags=4, lags_past_covariates=4, multi_models=mode
                        )
                        model_instance.fit(
                            series=series, past_covariates=self.sine_multivariate1
                        )
                        model_instance.predict(n=10)

                    model_instance = model(lags=12, multi_models=mode)
                    model_instance.fit(series=series)
                    self.assertEqual(model_instance.lags.get("past"), None)

                    model_instance = model(
                        lags=12, lags_past_covariates=12, multi_models=mode
                    )
                    model_instance.fit(
                        series=series, past_covariates=self.sine_multivariate1
                    )
                    self.assertEqual(len(model_instance.lags.get("past")), 12)

                    model_instance = model(
                        lags=12, lags_future_covariates=(0, 1), multi_models=mode
                    )
                    model_instance.fit(
                        series=series, future_covariates=self.sine_multivariate1
                    )
                    self.assertEqual(len(model_instance.lags.get("future")), 1)

                    model_instance = model(
                        lags=12, lags_past_covariates=[-1, -4, -6], multi_models=mode
                    )
                    model_instance.fit(
                        series=series, past_covariates=self.sine_multivariate1
                    )
                    self.assertEqual(len(model_instance.lags.get("past")), 3)

                    model_instance = model(
                        lags=12,
                        lags_past_covariates=[-1, -4, -6],
                        lags_future_covariates=[-2, 0],
                        multi_models=mode,
                    )
                    model_instance.fit(
                        series=series,
                        past_covariates=self.sine_multivariate1,
                        future_covariates=self.sine_multivariate1,
                    )
                    self.assertEqual(len(model_instance.lags.get("past")), 3)

    def helper_test_models_accuracy(self, series, past_covariates, min_rmse_model):
        # for every model, test whether it predicts the target with a minimum r2 score of `min_rmse`
        train_series, test_series = train_test_split(series, 70)
        train_past_covariates, _ = train_test_split(past_covariates, 70)

        multi_models_modes = [True, False]
        for mode in multi_models_modes:
            for output_chunk_length in [1, 5]:
                for idx, model in enumerate(self.models):
                    model_instance = model(
                        lags=12,
                        lags_past_covariates=2,
                        output_chunk_length=output_chunk_length,
                        multi_models=mode,
                    )
                    model_instance.fit(
                        series=train_series, past_covariates=train_past_covariates
                    )
                    prediction = model_instance.predict(
                        n=len(test_series),
                        series=train_series,
                        past_covariates=past_covariates,
                    )
                    current_rmse = rmse(prediction, test_series)
                    # in case of multi-series take mean rmse
                    mean_rmse = np.mean(current_rmse)
                    self.assertTrue(
                        mean_rmse <= min_rmse_model[idx],
                        f"{str(model_instance)} model was not able to predict data as well as expected. "
                        f"A mean rmse score of {mean_rmse} was recorded.",
                    )

    def test_models_accuracy_univariate(self):
        # for every model, and different output_chunk_lengths test whether it predicts the univariate time series
        # as well as expected, accuracies are defined at the top of the class
        self.helper_test_models_accuracy(
            self.sine_univariate1,
            self.sine_univariate2,
            self.univariate_accuracies,
        )

    def test_models_accuracy_multivariate(self):
        # for every model, and different output_chunk_lengths test whether it predicts the multivariate time series
        # as well as expected, accuracies are defined at the top of the class
        self.helper_test_models_accuracy(
            self.sine_multivariate1,
            self.sine_multivariate2,
            self.multivariate_accuracies,
        )

    def test_models_accuracy_multiseries_multivariate(self):
        # for every model, and different output_chunk_lengths test whether it predicts the multiseries, multivariate
        # time series as well as expected, accuracies are defined at the top of the class
        self.helper_test_models_accuracy(
            self.sine_multiseries1,
            self.sine_multiseries2,
            self.multivariate_multiseries_accuracies,
        )

    def test_min_train_series_length(self):
        mutli_models_modes = [True, False]
        for mode in mutli_models_modes:
            model = LightGBMModel(lags=4, multi_models=mode)
            min_train_series_length_expected = (
                -model.lags["target"][0] + model.output_chunk_length + 1
            )
            self.assertEqual(
                min_train_series_length_expected, model.min_train_series_length
            )
            model = CatBoostModel(lags=2, multi_models=mode)
            min_train_series_length_expected = (
                -model.lags["target"][0] + model.output_chunk_length + 1
            )
            self.assertEqual(
                min_train_series_length_expected, model.min_train_series_length
            )
            model = LightGBMModel(lags=[-4, -3, -2], multi_models=mode)
            min_train_series_length_expected = (
                -model.lags["target"][0] + model.output_chunk_length + 1
            )
            self.assertEqual(
                min_train_series_length_expected, model.min_train_series_length
            )
            model = XGBModel(lags=[-4, -3, -2], multi_models=mode)
            min_train_series_length_expected = (
                -model.lags["target"][0] + model.output_chunk_length + 1
            )
            self.assertEqual(
                min_train_series_length_expected, model.min_train_series_length
            )

    def test_historical_forecast(self):
        mutli_models_modes = [True, False]
        for mode in mutli_models_modes:
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
            self.assertEqual(len(result), 21)

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
            self.assertEqual(len(result), 21)

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
            self.assertEqual(len(result), 21)

    def test_multioutput_wrapper(self):
        lags = 4
        models = [
            (RegressionModel(lags=lags), True),
            (RegressionModel(lags=lags, model=LinearRegression()), True),
            (RegressionModel(lags=lags, model=RandomForestRegressor()), True),
            (
                RegressionModel(lags=lags, model=HistGradientBoostingRegressor()),
                False,
            ),
        ]

        for model, supports_multioutput_natively in models:
            model.fit(series=self.sine_multivariate1)
            if supports_multioutput_natively:
                self.assertFalse(isinstance(model.model, MultiOutputRegressor))
            else:
                self.assertIsInstance(model.model, MultiOutputRegressor)

    def test_multioutput_validation(self):

        lags = 4

        models = [
            LightGBMModel(lags=lags, output_chunk_length=1, multi_models=True),
            LightGBMModel(lags=lags, output_chunk_length=1, multi_models=False),
            LightGBMModel(lags=lags, output_chunk_length=2, multi_models=True),
            LightGBMModel(lags=lags, output_chunk_length=2, multi_models=False),
            CatBoostModel(lags=lags, output_chunk_length=1, multi_models=True),
            CatBoostModel(lags=lags, output_chunk_length=1, multi_models=False),
            CatBoostModel(lags=lags, output_chunk_length=2, multi_models=True),
            CatBoostModel(lags=lags, output_chunk_length=2, multi_models=False),
            XGBModel(lags=lags, output_chunk_length=1, multi_models=True),
            XGBModel(lags=lags, output_chunk_length=1, multi_models=False),
            XGBModel(lags=lags, output_chunk_length=2, multi_models=True),
            XGBModel(lags=lags, output_chunk_length=2, multi_models=False),
        ]

        train, val = self.sine_univariate1.split_after(0.6)

        for model in models:
            model.fit(series=train, val_series=val)
            if model.output_chunk_length > 1 and model.multi_models:
                self.assertIsInstance(model.model, MultiOutputRegressor)

    def test_regression_model(self):
        multi_models_modes = [True, False]
        for mode in multi_models_modes:
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
                self.assertEqual(len(model.lags.get("target")), lags)
                model.predict(n=10)

    def test_multiple_ts(self):
        multi_models_modes = [True, False]
        for mode in multi_models_modes:
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
            past_covariates_train, past_covariates_test = past_covariates.split_after(
                0.7
            )
            model.fit(
                series=[target_train, target_train + 0.5],
                past_covariates=[past_covariates_train, past_covariates_train + 0.5],
            )

            predictions = model.predict(
                10,
                series=[target_train, target_train + 0.5],
                past_covariates=[past_covariates, past_covariates + 0.5],
            )

            self.assertEqual(
                len(predictions[0]), 10, f"Found {len(predictions)} instead"
            )

            # multiple TS, both future and past covariates, checking that both covariates lead to better results than
            # using a single one (target series = past_cov + future_cov + noise)
            np.random.seed(42)

            linear_ts_1 = tg.linear_timeseries(start_value=10, end_value=59, length=50)
            linear_ts_2 = tg.linear_timeseries(start_value=40, end_value=89, length=50)

            past_covariates = tg.sine_timeseries(length=50) * 10
            future_covariates = (
                tg.sine_timeseries(length=50, value_frequency=0.015) * 50
            )

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
                inter_reduction=np.mean,
            )
            error_both = rmse(
                [target_test_1, target_test_2],
                prediction_past_and_future,
                inter_reduction=np.mean,
            )

            self.assertGreater(error_past_only, error_both)
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
                inter_reduction=np.mean,
            )

            self.assertGreater(error_both, error_both_multi_ts)

    def test_only_future_covariates(self):
        multi_models_modes = [True, False]
        for mode in multi_models_modes:
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

            self.assertEqual(
                len(predictions[0]), 10, f"Found {len(predictions[0])} instead"
            )

    def test_not_enough_covariates(self):
        multi_models_modes = [True, False]
        for mode in multi_models_modes:
            target_series = tg.linear_timeseries(
                start_value=0, end_value=100, length=50
            )
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
            # output_chunk_length, required past_offset, required future_offset
            test_cases = [
                (1, 0, 13),
                (5, -4, 9),
                (7, -6, 7),
                (
                    12,
                    -9,
                    4,
                ),  # output_chunk_length > n -> covariate requirements are capped
            ]

            for (output_chunk_length, req_past_offset, req_future_offset) in test_cases:
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
                with self.assertRaises(ValueError):
                    model.predict(
                        n,
                        series=target_series[:-25],
                        past_covariates=past_covariates[: -26 + req_past_offset],
                        future_covariates=future_covariates[: -25 + req_future_offset],
                    )
                # check that one less future covariate time step causes ValueError
                with self.assertRaises(ValueError):
                    model.predict(
                        n,
                        series=target_series[:-25],
                        past_covariates=past_covariates[: -25 + req_past_offset],
                        future_covariates=future_covariates[: -26 + req_future_offset],
                    )

    @patch.object(darts.models.forecasting.lgbm.lgb.LGBMRegressor, "fit")
    def test_gradient_boosted_model_with_eval_set(self, lgb_fit_patch):
        """Test whether these evaluation set parameters are passed to LGBRegressor"""
        model = LightGBMModel(lags=4, lags_past_covariates=2)
        model.fit(
            series=self.sine_univariate1,
            past_covariates=self.sine_multivariate1,
            val_series=self.sine_univariate1,
            val_past_covariates=self.sine_multivariate1,
            early_stopping_rounds=2,
        )

        lgb_fit_patch.assert_called_once()
        assert lgb_fit_patch.call_args[1]["eval_set"] is not None
        assert lgb_fit_patch.call_args[1]["early_stopping_rounds"] == 2

    @patch.object(darts.models.forecasting.xgboost.xgb.XGBRegressor, "fit")
    def test_xgboost_with_eval_set(self, xgb_fit_patch):
        model = XGBModel(lags=4, lags_past_covariates=2)
        model.fit(
            series=self.sine_univariate1,
            past_covariates=self.sine_multivariate1,
            val_series=self.sine_univariate1,
            val_past_covariates=self.sine_multivariate1,
            early_stopping_rounds=2,
        )

        xgb_fit_patch.assert_called_once()
        assert xgb_fit_patch.call_args[1]["eval_set"] is not None
        assert xgb_fit_patch.call_args[1]["early_stopping_rounds"] == 2

    def test_integer_indexed_series(self):
        values_target = np.random.rand(30)
        values_past_cov = np.random.rand(30)
        values_future_cov = np.random.rand(30)

        idx1 = pd.RangeIndex(start=0, stop=30, step=1)
        idx2 = pd.RangeIndex(start=10, stop=70, step=2)

        multi_models_mode = [True, False]
        for mode in multi_models_mode:
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
            self.assertTrue(
                all(preds[1].time_index == pd.RangeIndex(start=50, stop=70, step=2))
            )

    def test_encoders(self):
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
            tg.generate_index(
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

        multi_models_mode = [True, False]
        for mode in multi_models_mode:
            for ocl in [1, 2]:
                for model_cls in [
                    RegressionModel,
                    LinearRegressionModel,
                    LightGBMModel,
                    XGBModel,
                ]:
                    model_pc_valid0 = model_cls(
                        lags=2,
                        add_encoders=encoder_examples["past"],
                        multi_models=mode,
                        output_chunk_length=ocl,
                    )
                    model_fc_valid0 = model_cls(
                        lags=2,
                        add_encoders=encoder_examples["future"],
                        multi_models=mode,
                        output_chunk_length=ocl,
                    )
                    model_mixed_valid0 = model_cls(
                        lags=2,
                        add_encoders=encoder_examples["mixed"],
                        multi_models=mode,
                        output_chunk_length=ocl,
                    )

                    # encoders will not generate covariates without lags
                    for model in [model_pc_valid0, model_fc_valid0, model_mixed_valid0]:
                        model.fit(ts)
                        assert not model.encoders.encoding_available
                        _ = model.predict(n=1, series=ts)
                        _ = model.predict(n=3, series=ts)

                    model_pc_valid1 = model_cls(
                        lags=2,
                        lags_past_covariates=[max_past_lag, -1],
                        add_encoders=encoder_examples["past"],
                        multi_models=mode,
                        output_chunk_length=ocl,
                    )
                    model_fc_valid1 = model_cls(
                        lags=2,
                        lags_future_covariates=[0, max_future_lag],
                        add_encoders=encoder_examples["future"],
                        multi_models=mode,
                        output_chunk_length=ocl,
                    )
                    model_mixed_valid1 = model_cls(
                        lags=2,
                        lags_past_covariates=[max_past_lag, -1],
                        lags_future_covariates=[0, max_future_lag],
                        add_encoders=encoder_examples["mixed"],
                        multi_models=mode,
                        output_chunk_length=ocl,
                    )

                    for model, ex in zip(
                        [model_pc_valid1, model_fc_valid1, model_mixed_valid1], examples
                    ):
                        covariates = covariates_examples[ex]
                        # don't pass covariates, let them be generated by encoders. Test single target series input
                        model_copy = copy.deepcopy(model)
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
                return pd.Timestamp(dt, freq=freq)

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
            assert all(
                [isinstance(el, list) for el in [train_past, infer_past, refer_past]]
            )
            assert len(train_past) == len(infer_past) == len(refer_past)
            assert all(
                [
                    t_p.start_time() == tp_s
                    for t_p, tp_s in zip(train_past, t_train["pc_start"])
                ]
            )
            assert all(
                [
                    t_p.end_time() == tp_e
                    for t_p, tp_e in zip(train_past, t_train["pc_end"])
                ]
            )
            assert all(
                [
                    i_p.start_time() == ip_s
                    for i_p, ip_s in zip(infer_past, t_infer["pc_start"])
                ]
            )
            assert all(
                [
                    i_p.end_time() == ip_e
                    for i_p, ip_e in zip(infer_past, t_infer["pc_end"])
                ]
            )

        if train_future is None:
            assert infer_future is None and refer_future is None
        else:
            assert all(
                [
                    isinstance(el, list)
                    for el in [train_future, infer_future, refer_future]
                ]
            )
            assert len(train_future) == len(infer_future) == len(refer_future)
            assert all(
                [
                    t_f.start_time() == tf_s
                    for t_f, tf_s in zip(train_future, t_train["fc_start"])
                ]
            )
            assert all(
                [
                    t_f.end_time() == tf_e
                    for t_f, tf_e in zip(train_future, t_train["fc_end"])
                ]
            )
            assert all(
                [
                    i_f.start_time() == if_s
                    for i_f, if_s in zip(infer_future, t_infer["fc_start"])
                ]
            )
            assert all(
                [
                    i_f.end_time() == if_e
                    for i_f, if_e in zip(infer_future, t_infer["fc_end"])
                ]
            )

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

    @patch.object(darts.models.forecasting.catboost_model.CatBoostRegressor, "fit")
    def test_catboost_model_with_eval_set(self, lgb_fit_patch):
        """Test whether these evaluation set parameters are passed to CatBoostRegressor"""
        model = CatBoostModel(lags=4, lags_past_covariates=2)
        model.fit(
            series=self.sine_univariate1,
            past_covariates=self.sine_multivariate1,
            val_series=self.sine_univariate1,
            val_past_covariates=self.sine_multivariate1,
            early_stopping_rounds=2,
        )

        lgb_fit_patch.assert_called_once()

        assert lgb_fit_patch.call_args[1]["eval_set"] is not None
        assert lgb_fit_patch.call_args[1]["early_stopping_rounds"] == 2


class ProbabilisticRegressionModelsTestCase(DartsBaseTestClass):
    models_cls_kwargs_errs = [
        (
            LightGBMModel,
            {
                "lags": 2,
                "likelihood": "quantile",
                "random_state": 42,
                "multi_models": True,
            },
            0.4,
        ),
        (
            LightGBMModel,
            {
                "lags": 2,
                "likelihood": "quantile",
                "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9],
                "random_state": 42,
                "multi_models": True,
            },
            0.4,
        ),
        (
            LightGBMModel,
            {
                "lags": 2,
                "likelihood": "poisson",
                "random_state": 42,
                "multi_models": True,
            },
            0.6,
        ),
        (
            CatBoostModel,
            {
                "lags": 2,
                "likelihood": "quantile",
                "random_state": 42,
                "multi_models": True,
            },
            0.05,
        ),
        (
            CatBoostModel,
            {
                "lags": 2,
                "likelihood": "quantile",
                "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9],
                "random_state": 42,
                "multi_models": True,
            },
            0.05,
        ),
        (
            CatBoostModel,
            {
                "lags": 2,
                "likelihood": "poisson",
                "random_state": 42,
                "multi_models": True,
            },
            0.6,
        ),
        (
            CatBoostModel,
            {
                "lags": 2,
                "likelihood": "gaussian",
                "random_state": 42,
                "multi_models": True,
            },
            0.05,
        ),
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
                "random_state": 42,
                "multi_models": True,
            },
            0.6,
        ),
        (
            XGBModel,
            {
                "lags": 2,
                "likelihood": "quantile",
                "quantiles": [0.1, 0.3, 0.5, 0.7, 0.9],
                "random_state": 42,
                "multi_models": True,
            },
            0.4,
        ),
    ]

    constant_ts = tg.constant_timeseries(length=200, value=0.5)
    constant_noisy_ts = constant_ts + tg.gaussian_timeseries(length=200, std=0.1)
    constant_multivar_ts = constant_ts.stack(constant_ts)
    constant_noisy_multivar_ts = constant_noisy_ts.stack(constant_noisy_ts)
    num_samples = 5

    def test_fit_predict_determinism(self):
        multi_models_modes = [False, True]
        for mode in multi_models_modes:
            for model_cls, model_kwargs, _ in self.models_cls_kwargs_errs:
                # whether the first predictions of two models initiated with the same random state are the same
                model_kwargs["multi_models"] = mode
                model = model_cls(**model_kwargs)
                model.fit(self.constant_noisy_multivar_ts)
                pred1 = model.predict(n=10, num_samples=2).values()

                model = model_cls(**model_kwargs)
                model.fit(self.constant_noisy_multivar_ts)
                pred2 = model.predict(n=10, num_samples=2).values()

                self.assertTrue((pred1 == pred2).all())

                # test whether the next prediction of the same model is different
                pred3 = model.predict(n=10, num_samples=2).values()
                self.assertTrue((pred2 != pred3).any())

    def test_probabilistic_forecast_accuracy(self):
        multi_models_modes = [True, False]
        for mode in multi_models_modes:
            for model_cls, model_kwargs, err in self.models_cls_kwargs_errs:
                model_kwargs["multi_models"] = mode
                self.helper_test_probabilistic_forecast_accuracy(
                    model_cls,
                    model_kwargs,
                    err,
                    self.constant_ts,
                    self.constant_noisy_ts,
                )
                if issubclass(model_cls, GlobalForecastingModel):
                    self.helper_test_probabilistic_forecast_accuracy(
                        model_cls,
                        model_kwargs,
                        err,
                        self.constant_multivar_ts,
                        self.constant_noisy_multivar_ts,
                    )

    def helper_test_probabilistic_forecast_accuracy(
        self, model_cls, model_kwargs, err, ts, noisy_ts
    ):
        model = model_cls(**model_kwargs)
        model.fit(noisy_ts[:100])
        pred = model.predict(n=100, num_samples=100)

        # test accuracy of the median prediction compared to the noiseless ts
        mae_err_median = mae(ts[100:], pred)
        self.assertLess(mae_err_median, err)

        # test accuracy for increasing quantiles between 0.7 and 1 (it should ~decrease, mae should ~increase)
        tested_quantiles = [0.7, 0.8, 0.9, 0.99]
        mae_err = mae_err_median
        for quantile in tested_quantiles:
            new_mae = mae(ts[100:], pred.quantile_timeseries(quantile=quantile))
            self.assertLess(mae_err, new_mae + 0.1)
            mae_err = new_mae

        # test accuracy for decreasing quantiles between 0.3 and 0 (it should ~decrease, mae should ~increase)
        tested_quantiles = [0.3, 0.2, 0.1, 0.01]
        mae_err = mae_err_median
        for quantile in tested_quantiles:
            new_mae = mae(ts[100:], pred.quantile_timeseries(quantile=quantile))
            self.assertLess(mae_err, new_mae + 0.1)
            mae_err = new_mae
