import math
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import darts
from darts import TimeSeries
from darts.logging import get_logger
from darts.metrics import rmse
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg

logger = get_logger(__name__)

try:
    from darts.models import (
        LightGBMModel,
        LinearRegressionModel,
        RandomForest,
        RegressionModel,
    )

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not available. Regression models tests will be skipped.")
    TORCH_AVAILABLE = False


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


# Regression models rely on PyTorch for the Datasets
if TORCH_AVAILABLE:

    class RegressionModelsTestCase(DartsBaseTestClass):

        np.random.seed(42)

        # default regression models
        models = [RandomForest, LinearRegressionModel, RegressionModel, LightGBMModel]

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

        sine_univariate1 = tg.sine_timeseries(length=100)
        sine_univariate2 = tg.sine_timeseries(length=100, value_phase=1.5705)
        sine_univariate3 = tg.sine_timeseries(length=100, value_phase=0.78525)
        sine_univariate4 = tg.sine_timeseries(length=100, value_phase=0.392625)
        sine_univariate5 = tg.sine_timeseries(length=100, value_phase=0.1963125)
        sine_univariate6 = tg.sine_timeseries(length=100, value_phase=0.09815625)
        sine_multivariate1 = sine_univariate1.stack(sine_univariate2)
        sine_multivariate2 = sine_univariate2.stack(sine_univariate3)
        sine_multiseries1 = [sine_univariate1, sine_univariate2, sine_univariate3]
        sine_multiseries2 = [sine_univariate4, sine_univariate5, sine_univariate6]

        lags_1 = {"target": [-3, -2, -1], "past": [-4, -2], "future": [-5, 2]}

        def test_model_construction(self):

            for model in self.models:
                # TESTING SINGLE INT
                # testing lags
                model_instance = model(lags=5)
                self.assertEqual(
                    model_instance.lags.get("target"), [-5, -4, -3, -2, -1]
                )
                # testing lags_past_covariates
                model_instance = model(lags=None, lags_past_covariates=3)
                self.assertEqual(model_instance.lags.get("past"), [-3, -2, -1])
                # testing lags_future covariates
                model_instance = model(lags=None, lags_future_covariates=(3, 5))
                self.assertEqual(
                    model_instance.lags.get("future"), [-3, -2, -1, 0, 1, 2, 3, 4]
                )

                # TESTING LIST of int
                # lags
                values = [-5, -3, -1]
                model_instance = model(lags=values)
                self.assertEqual(model_instance.lags.get("target"), values)
                # testing lags_past_covariates
                model_instance = model(lags_past_covariates=values)
                self.assertEqual(model_instance.lags.get("past"), values)
                # testing lags_future_covariates

                with self.assertRaises(ValueError):
                    model()
                with self.assertRaises(ValueError):
                    model(lags=0)
                with self.assertRaises(ValueError):
                    model(lags=[-1, 0])
                with self.assertRaises(ValueError):
                    model(lags=[3, 5])
                with self.assertRaises(ValueError):
                    model(lags=[-3, -5.0])
                with self.assertRaises(ValueError):
                    model(lags=-5)
                with self.assertRaises(ValueError):
                    model(lags=3.6)
                with self.assertRaises(ValueError):
                    model(lags=None, lags_past_covariates=False)
                with self.assertRaises(ValueError):
                    model(lags=None)
                with self.assertRaises(ValueError):
                    model(lags=5, lags_future_covariates=True)
                with self.assertRaises(ValueError):
                    model(lags=5, lags_future_covariates=(1, -3))
                with self.assertRaises(ValueError):
                    model(lags=5, lags_future_covariates=(1, 2, 3))
                with self.assertRaises(ValueError):
                    model(lags=5, lags_future_covariates=(1, True))
                with self.assertRaises(ValueError):
                    model(lags=5, lags_future_covariates=(1, 1.0))

        def test_training_data_creation(self):
            # testing _get_training_data function
            model_instance = RegressionModel(
                lags=self.lags_1["target"],
                lags_past_covariates=self.lags_1["past"],
                lags_future_covariates=self.lags_1["future"],
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

            # assigning correct names to variables
            series = [ts[:-50] for ts in self.target_series]
            output_chunk_length = 5
            n = 12

            # prediction preprocessing start
            covariates = {
                "past": (self.past_covariates, self.lags_1.get("past")),
                "future": (self.future_covariates, self.lags_1.get("future")),
            }

            # dictionary containing covariate data over time span required for prediction
            covariate_matrices = {}
            # dictionary containing covariate lags relative to minimum covariate lag
            relative_cov_lags = {}
            # number of prediction steps given forecast horizon and output_chunk_length
            n_pred_steps = math.ceil(n / output_chunk_length)
            for cov_type, (covs, lags) in covariates.items():
                if covs is not None:
                    relative_cov_lags[cov_type] = np.array(lags) - lags[0]
                    covariate_matrices[cov_type] = []
                    for idx, (ts, cov) in enumerate(zip(series, covs)):
                        first_pred_ts = ts.end_time() + 1 * ts.freq
                        last_pred_ts = (
                            first_pred_ts
                            + ((n_pred_steps - 1) * output_chunk_length) * ts.freq
                        )
                        first_req_ts = first_pred_ts + lags[0] * ts.freq
                        last_req_ts = last_pred_ts + lags[-1] * ts.freq

                        # not enough covariate data checks excluded, they are tested elsewhere

                        if cov.has_datetime_index:
                            covariate_matrices[cov_type].append(
                                cov[first_req_ts:last_req_ts].values()
                            )
                        else:
                            # include last_req_ts when slicing series with integer indices
                            covariate_matrices[cov_type].append(
                                cov[first_req_ts : last_req_ts + 1].values()
                            )

                    covariate_matrices[cov_type] = np.stack(
                        covariate_matrices[cov_type]
                    )

            series_matrix = None
            if "target" in self.lags_1:
                series_matrix = np.stack(
                    [ts[self.lags_1["target"][0] :].values() for ts in series]
                )
            # prediction preprocessing end

            # tests
            self.assertTrue(
                all([lag >= 0 for lags in relative_cov_lags.values() for lag in lags])
            )
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

        def test_models_runnability(self):
            train_y, test_y = self.sine_univariate1.split_before(0.7)
            for model in self.models:
                # testing past covariates
                with self.assertRaises(ValueError):
                    # testing lags_past_covariates None but past_covariates during training
                    model_instance = model(lags=4, lags_past_covariates=None)
                    model_instance.fit(
                        series=self.sine_univariate1,
                        past_covariates=self.sine_multivariate1,
                    )

                with self.assertRaises(ValueError):
                    # testing lags_past_covariates but no past_covariates during fit
                    model_instance = model(lags=4, lags_past_covariates=3)
                    model_instance.fit(series=self.sine_univariate1)

                # testing future_covariates
                with self.assertRaises(ValueError):
                    # testing lags_future_covariates None but future_covariates during training
                    model_instance = model(lags=4, lags_future_covariates=None)
                    model_instance.fit(
                        series=self.sine_univariate1,
                        future_covariates=self.sine_multivariate1,
                    )

                with self.assertRaises(ValueError):
                    # testing lags_covariate but no covariate during fit
                    model_instance = model(lags=4, lags_future_covariates=3)
                    model_instance.fit(series=self.sine_univariate1)

                # testing input_dim
                model_instance = model(lags=4, lags_past_covariates=2)
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
                self.assertTrue(
                    len(prediction) == 1,
                    f"Expected length 1, found {len(prediction)} instead",
                )

        def test_fit(self):
            for model in self.models:
                # test fitting both on univariate and multivariate timeseries
                for series in [self.sine_univariate1, self.sine_multivariate2]:
                    with self.assertRaises(ValueError):
                        model_instance = model(lags=4, lags_past_covariates=4)
                        model_instance.fit(
                            series=series, past_covariates=self.sine_multivariate1
                        )
                        model_instance.predict(n=10)

                    model_instance = model(lags=12)
                    model_instance.fit(series=series)
                    self.assertEqual(model_instance.lags.get("past"), None)

                    model_instance = model(lags=12, lags_past_covariates=12)
                    model_instance.fit(
                        series=series, past_covariates=self.sine_multivariate1
                    )
                    self.assertEqual(len(model_instance.lags.get("past")), 12)

                    model_instance = model(lags=12, lags_future_covariates=(0, 1))
                    model_instance.fit(
                        series=series, future_covariates=self.sine_multivariate1
                    )
                    self.assertEqual(len(model_instance.lags.get("future")), 1)

                    model_instance = model(lags=12, lags_past_covariates=[-1, -4, -6])
                    model_instance.fit(
                        series=series, past_covariates=self.sine_multivariate1
                    )
                    self.assertEqual(len(model_instance.lags.get("past")), 3)

                    model_instance = model(
                        lags=12,
                        lags_past_covariates=[-1, -4, -6],
                        lags_future_covariates=[-2, 0],
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

            for output_chunk_length in [1, 5]:
                for idx, model in enumerate(self.models):
                    model_instance = model(
                        lags=12,
                        lags_past_covariates=2,
                        output_chunk_length=output_chunk_length,
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
            # as well as expected
            self.helper_test_models_accuracy(
                self.sine_univariate1, self.sine_univariate2, [0.03, 1e-13, 1e-13, 0.3]
            )

        def test_models_accuracy_multivariate(self):
            # for every model, and different output_chunk_lengths test whether it predicts the multivariate time series
            # as well as expected
            self.helper_test_models_accuracy(
                self.sine_multivariate1,
                self.sine_multivariate2,
                [0.3, 1e-13, 1e-13, 0.4],
            )

        def test_models_accuracy_multiseries_multivariate(self):
            # for every model, and different output_chunk_lengths test whether it predicts the multiseries, multivariate
            # time series as well as expected
            self.helper_test_models_accuracy(
                self.sine_multiseries1,
                self.sine_multiseries2,
                [0.05, 1e-13, 1e-13, 0.05],
            )

        def test_historical_forecast(self):
            model = self.models[1](lags=5)
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

            model = self.models[1](lags=5, lags_past_covariates=5)
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
                lags=5, lags_past_covariates=5, output_chunk_length=5
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
            lags = 12
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
                    self.assertTrue(isinstance(model.model, MultiOutputRegressor))

        def test_regression_model(self):
            lags = 12
            models = [
                RegressionModel(lags=lags),
                RegressionModel(lags=lags, model=LinearRegression()),
                RegressionModel(lags=lags, model=RandomForestRegressor()),
                RegressionModel(lags=lags, model=HistGradientBoostingRegressor()),
            ]

            for model in models:
                model.fit(series=self.sine_univariate1)
                self.assertEqual(len(model.lags.get("target")), lags)
                model.predict(n=10)

        def test_multiple_ts(self):
            lags = 4
            lags_past_covariates = 3
            model = RegressionModel(
                lags=lags, lags_past_covariates=lags_past_covariates
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
            model = RegressionModel(lags=3, lags_past_covariates=5)
            model.fit([target_train_1_noise], [past_covariates])

            prediction_past_only = model.predict(
                n=len(target_test_1),
                series=[target_train_1_noise, target_train_2_noise],
                past_covariates=[past_covariates] * 2,
            )

            model = RegressionModel(
                lags=3, lags_past_covariates=5, lags_future_covariates=(5, 0)
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

            self.assertTrue(error_past_only > error_both)
            # test 2: with both covariates, 2 TS should learn more than one (with little noise)
            model = RegressionModel(
                lags=3, lags_past_covariates=5, lags_future_covariates=(5, 0)
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

            self.assertTrue(error_both > error_both_multi_ts)

        def test_only_future_covariates(self):

            model = RegressionModel(lags_future_covariates=[-2])

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
            )
            model.fit(
                series=target_series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
                max_samples_per_ts=1,
            )

            # output_chunk_length, required past_offset, required future_offset
            test_cases = [
                (1, 0, 13),
                (5, -4, 9),
                (7, -2, 11),
            ]
            for (output_chunk_length, req_past_offset, req_future_offset) in test_cases:
                model = RegressionModel(
                    lags_past_covariates=[-10],
                    lags_future_covariates=[-4, 3],
                    output_chunk_length=output_chunk_length,
                )
                model.fit(
                    series=target_series,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                )

                # check that given the required offsets no ValueError is raised
                model.predict(
                    10,
                    series=target_series[:-25],
                    past_covariates=past_covariates[: -25 + req_past_offset],
                    future_covariates=future_covariates[: -25 + req_future_offset],
                )
                # check that one less past covariate time step causes ValueError
                with self.assertRaises(ValueError):
                    model.predict(
                        10,
                        series=target_series[:-25],
                        past_covariates=past_covariates[: -26 + req_past_offset],
                        future_covariates=future_covariates[: -25 + req_future_offset],
                    )
                # check that one less future covariate time step causes ValueError
                with self.assertRaises(ValueError):
                    model.predict(
                        10,
                        series=target_series[:-25],
                        past_covariates=past_covariates[: -25 + req_past_offset],
                        future_covariates=future_covariates[: -26 + req_future_offset],
                    )

        @patch.object(
            darts.models.forecasting.gradient_boosted_model.lgb.LGBMRegressor, "fit"
        )
        # @patch.object(darts.models.forecasting.gradient_boosted_model.lgb.LGBMRegressor, 'fit')
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
