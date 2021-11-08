import numpy as np
import pandas as pd

import darts
from ..logging import get_logger
from darts import TimeSeries
from darts.metrics import rmse
from .base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from unittest.mock import patch

# Required to import HistGradientBoostingRegressor in sklearn
from sklearn.experimental import enable_hist_gradient_boosting

logger = get_logger(__name__)

try:
    from darts.models import RegressionModel, RandomForest, LinearRegressionModel, LightGBMModel
    from darts.utils.data.sequential_dataset import MixedCovariatesSequentialDataset
    from darts.utils.data.inference_dataset import MixedCovariatesInferenceDataset
    from darts.models.forecasting.regression_model import _shift_matrices, _update_min_max
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning('Torch not available. Regression models tests will be skipped.')
    TORCH_AVAILABLE = False


def train_test_split(features, target, split_ts):
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
    train_features, test_features = features.split_after(split_ts)
    train_target, test_target = target.split_after(split_ts)

    return (train_features, train_target, test_features, test_target)


# Regression models rely on torch for the Datasets
if TORCH_AVAILABLE:
    class RegressionModelsTestCase(DartsBaseTestClass):

        np.random.seed(42)

        # dummy feature and target TimeSeries instances
        ts_periodic = tg.sine_timeseries(length=500)
        ts_gaussian = tg.gaussian_timeseries(length=500)
        ts_random_walk = tg.random_walk_timeseries(length=500)

        ts_cov1 = ts_periodic.stack(ts_gaussian)
        ts_cov1 = ts_cov1.pd_dataframe()
        ts_cov1.columns = ["Periodic", "Gaussian"]
        ts_cov1 = TimeSeries.from_dataframe(ts_cov1)
        ts_sum1 = ts_periodic + ts_gaussian

        ts_cov2 = ts_sum1.stack(ts_random_walk)
        ts_sum2 = ts_sum1 + ts_random_walk

        ts_multivariate1 = ts_sum1.stack(ts_sum2)

        # default regression models
        models = [
            RandomForest,
            LinearRegressionModel,
            RegressionModel,
            LightGBMModel
        ]

        target_series = tg.linear_timeseries(start_value=0, end_value=49, length=50)
        past_covariates = tg.linear_timeseries(start_value=50, end_value=99, length=50).stack(
            tg.linear_timeseries(start_value=50.5, end_value=99.5, length=50)
        )
        future_covariates = tg.linear_timeseries(start_value=100, end_value=199, length=100)

        lags_1 = [-5, -3, -1]
        lags_past_covariates_1 = [-4, -2]
        lags_future_covariates_1 = [-1, 0, 3]

        def test_model_construction(self):

            for model in self.models:
                # TESTING SINGLE INT
                # testing lags
                model_instance = model(lags=5)
                self.assertEqual(model_instance.lags, [-5, -4, -3, -2, -1])
                self.assertEqual(model_instance.min_lag, -5)
                self.assertEqual(model_instance.max_lag, -1)
                # testing lags_past_covariates
                model_instance = model(lags=None, lags_past_covariates=3)
                self.assertEqual(model_instance.lags_past_covariates, [-3, -2, -1])
                self.assertEqual(model_instance.min_lag, -3)
                self.assertEqual(model_instance.max_lag, -1)
                # testing lags_future covariates
                model_instance = model(lags=None, lags_future_covariates=(3, 5))
                self.assertEqual(model_instance.lags_historical_covariates, [-3, -2, -1])
                self.assertEqual(model_instance.lags_future_covariates, [0, 1, 2, 3, 4])
                self.assertEqual(model_instance.min_lag, -3)
                self.assertEqual(model_instance.max_lag, 4)

                # TESTING LIST of int
                # lags
                values = [-5, -3, -1]
                model_instance = model(lags=values)
                self.assertEqual(model_instance.lags, values)
                self.assertEqual(model_instance.min_lag, -5)
                self.assertEqual(model_instance.max_lag, -1)
                # testing lags_past_covariates
                model_instance = model(lags_past_covariates=values)
                self.assertEqual(model_instance.lags_past_covariates, values)
                self.assertEqual(model_instance.min_lag, -5)
                self.assertEqual(model_instance.max_lag, -1)
                # testing lags_future_covariates

                checks_future_covariates = [
                    # lags as input, lags_historical_covariates, lags_future_covariates, min_lag, max_lag
                    ([-3, -5, 1, 5], [-5, -3], [1, 5], -5, 5),
                    ([-3, -5], [-5, -3], None, -5, -3),
                    ([1, 5], None, [1, 5], 1, 5)
                ]

                for lags, lags_hist, lags_fut, min_l, max_l in checks_future_covariates:
                    model_instance = model(lags_future_covariates=lags)
                    self.assertEqual(model_instance.lags_historical_covariates, lags_hist)
                    self.assertEqual(model_instance.lags_future_covariates, lags_fut)
                    self.assertEqual(model_instance.min_lag, min_l)
                    self.assertEqual(model_instance.max_lag, max_l)

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
                lags=self.lags_1,
                lags_past_covariates=self.lags_past_covariates_1,
                lags_future_covariates=self.lags_future_covariates_1
            )

            input_chunk_length = -model_instance.min_lag
            training_output_chunk_length = max(model_instance.max_lag + 1, 1)

            training_dataset = MixedCovariatesSequentialDataset(
                target_series=self.target_series,
                past_covariates=self.past_covariates,
                future_covariates=self.future_covariates,
                input_chunk_length=input_chunk_length,
                output_chunk_length=training_output_chunk_length
            )

            training_samples, training_labels = model_instance._get_training_data(training_dataset=training_dataset)

            # checking number of dimensions
            self.assertEqual(len(training_samples.shape), 2)  # samples, features
            self.assertEqual(len(training_labels.shape), 2)  # samples, components (multivariate)
            self.assertEqual(training_samples.shape[0], training_labels.shape[0])
            self.assertEqual(training_samples.shape[1],
                             len(self.lags_1) + len(self.lags_past_covariates_1) * 2 + len(self.lags_future_covariates_1))

            # checking first row order lags | lags_past_cov | lags_future_cov
            self.assertListEqual(list(training_samples[-1, :]), [0, 2, 4, 51, 51.5, 53, 53.5, 104, 105, 108])
            self.assertEqual(training_labels[-1], 5)

        def test_prediction_data_creation(self):
            model_instance = RegressionModel(
                lags=self.lags_1,
                lags_past_covariates=self.lags_past_covariates_1,
                lags_future_covariates=self.lags_future_covariates_1
            )

            n = 2
            input_chunk_length = max(1, -model_instance.min_lag)
            prediction_output_chunk_length = max(n + model_instance.max_lag, n)

            prediction_dataset = MixedCovariatesInferenceDataset(
                target_series=self.target_series,
                past_covariates=self.past_covariates,
                future_covariates=self.future_covariates,
                input_chunk_length=input_chunk_length,
                output_chunk_length=prediction_output_chunk_length
            )

            (
                target_matrix,
                past_covariates_matrix,
                historic_future_covariates_matrix,
                future_covariates_matrix,
                future_past_covariates_matrix
            ) = model_instance._get_prediction_data(prediction_dataset)

            # checking matrices sizes and content

            for matrix in [target_matrix, past_covariates_matrix, historic_future_covariates_matrix,
                           future_covariates_matrix]:
                self.assertEqual(matrix.shape[0], 1)  # we are training on a single ts

            # checking matrices dimensions
            self.assertEqual(target_matrix.ndim, 3)  # samples, time, dim (multivariate)
            self.assertEqual(past_covariates_matrix.ndim, 3)  # samples, time, dim (could be multivariate)
            self.assertEqual(historic_future_covariates_matrix.ndim, 3)  # same
            self.assertEqual(future_covariates_matrix.ndim, 3)  # same

            # checking the first sample of the matrices, (should contain input_chunk_length matrices) for
            # past values, while should contain n + max_lag in the future ones (where not empty)
            np.testing.assert_array_equal(target_matrix[0].ravel(), np.arange(45, 50))
            np.testing.assert_array_equal(past_covariates_matrix[0, :, 0], np.arange(95, 100))
            np.testing.assert_array_equal(past_covariates_matrix[0, :, 1], np.arange(95.5, 100.5))
            np.testing.assert_array_equal(historic_future_covariates_matrix[0].ravel(), np.arange(145, 150))
            self.assertIsNone(future_past_covariates_matrix)
            np.testing.assert_array_equal(future_covariates_matrix[0].ravel(), np.arange(150, 151))

        def test_shift_matrices(self):
            a = np.zeros((10, 8, 2))
            b = np.ones((10, 3, 2))
            a, b = _shift_matrices(a, b)
            # last "temporal slice" of a should now contain 1s
            np.testing.assert_array_equal(a[:, -1, :], np.ones((10, 2)))

            # testing empty future_matrix
            with self.assertRaises(ValueError):
                a, b = _shift_matrices(a, None)

        def test_update_min_max(self):
            current_min = None
            current_max = None
            values = [-1, -1, -4, 5, 9, 12]
            current_min, current_max = _update_min_max(current_min, current_max, values)

            self.assertEqual(current_min, -4)
            self.assertEqual(current_max, 12)

        def test_models_runnability(self):
            train_x, test_x = self.ts_cov1.split_before(0.7)
            train_y, test_y = self.ts_sum1.split_before(0.7)
            for model in self.models:
                # testing past covariates
                with self.assertRaises(ValueError):
                    # testing lags_past_covariates None but past_covariates during training
                    model_instance = model(lags=4, lags_past_covariates=None)
                    model_instance.fit(series=self.ts_sum1, past_covariates=self.ts_cov1)

                with self.assertRaises(ValueError):
                    # testing lags_past_covariates but no past_covariates during fit
                    model_instance = model(lags=4, lags_past_covariates=3)
                    model_instance.fit(series=self.ts_sum1)

                # testing future_covariates
                with self.assertRaises(ValueError):
                    # testing lags_future_covariates None but future_covariates during training
                    model_instance = model(lags=4, lags_future_covariates=None)
                    model_instance.fit(series=self.ts_sum1, future_covariates=self.ts_cov1)

                with self.assertRaises(ValueError):
                    # testing lags_covariate but no covariate during fit
                    model_instance = model(lags=4, lags_future_covariates=3)
                    model_instance.fit(series=self.ts_sum1)

                # testing input_dim
                model_instance = model(lags=4, lags_past_covariates=2)
                model_instance.fit(
                    series=train_y,
                    past_covariates=self.ts_sum1.stack(self.ts_sum1)
                )

                self.assertEqual(model_instance.input_dim, 3)

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
                for series in [self.ts_sum1, self.ts_multivariate1]:
                    with self.assertRaises(ValueError):
                        model_instance = model(lags=4, lags_past_covariates=4)
                        model_instance.fit(series=series, past_covariates=self.ts_cov1)
                        model_instance.predict(n=10)

                    model_instance = model(lags=12)
                    model_instance.fit(series=series)
                    self.assertEqual(model_instance.lags_past_covariates, None)

                    model_instance = model(lags=12, lags_past_covariates=12)
                    model_instance.fit(series=series, past_covariates=self.ts_cov1)
                    self.assertEqual(len(model_instance.lags_past_covariates), 12)

                    model_instance = model(lags=12, lags_future_covariates=(0, 1))
                    model_instance.fit(series=series, future_covariates=self.ts_cov1)
                    self.assertEqual(len(model_instance.lags_future_covariates), 1)
                    self.assertIsNone(model_instance.lags_historical_covariates)

                    model_instance = model(lags=12, lags_past_covariates=[-1, -4, -6])
                    model_instance.fit(series=series, past_covariates=self.ts_cov1)
                    self.assertEqual(len(model_instance.lags_past_covariates), 3)

                    model_instance = model(lags=12, lags_past_covariates=[-1, -4, -6], lags_future_covariates=[-2, 0])
                    model_instance.fit(series=series, past_covariates=self.ts_cov1, future_covariates=self.ts_cov1)
                    self.assertEqual(len(model_instance.lags_past_covariates), 3)

        def helper_test_models_accuracy(self, series, past_covariates, min_rmse):
            # for every model, test whether it predicts the target with a minimum r2 score of `min_rmse`
            train_f, train_t, test_f, test_t = train_test_split(past_covariates, series, pd.Timestamp("20010101"))

            for model in self.models:
                model_instance = model(lags=12, lags_past_covariates=2)
                model_instance.fit(series=train_t, past_covariates=train_f)
                prediction = model_instance.predict(n=len(test_t), past_covariates=past_covariates)
                current_rmse = rmse(prediction, test_t)

                self.assertTrue(
                    current_rmse <= min_rmse,
                    f"{str(model_instance)} model was not able to denoise data. A rmse score of {current_rmse} was"
                    "recorded."
                )

        def test_models_denoising(self):
            # for every model, test whether it correctly denoises ts_sum using ts_gaussian and ts_sum as inputs
            self.helper_test_models_accuracy(self.ts_sum1, self.ts_cov1, 1.5)

        def test_models_denoising_multi_input(self):
            # for every model, test whether it correctly denoises ts_sum_2 using ts_random_multi and ts_sum_2 as inputs
            self.helper_test_models_accuracy(self.ts_sum2, self.ts_cov2, 19.5)

        def test_historical_forecast(self):
            model = self.models[0](lags=5)
            result = model.historical_forecasts(
                series=self.ts_sum1[:100],
                future_covariates=None,
                start=0.5,
                forecast_horizon=1,
                stride=1,
                retrain=True,
                overlap_end=False,
                last_points_only=True,
                verbose=False
            )
            self.assertEqual(len(result), 51)

            model = self.models[0](lags=5, lags_past_covariates=5)
            result = model.historical_forecasts(
                series=self.ts_sum1[:100],
                past_covariates=self.ts_cov1[:100],
                start=0.5,
                forecast_horizon=1,
                stride=1,
                retrain=True,
                overlap_end=False,
                last_points_only=True,
                verbose=False
            )
            self.assertEqual(len(result), 51)

        def test_regression_model(self):
            lags = 12
            models = [
                RegressionModel(lags=lags),
                RegressionModel(lags=lags, model=LinearRegression()),
                RegressionModel(lags=lags, model=RandomForestRegressor()),
                RegressionModel(lags=lags, model=HistGradientBoostingRegressor())
            ]

            for model in models:
                model.fit(series=self.ts_sum1)
                self.assertEqual(len(model.lags), lags)
                model.predict(n=10)

        def test_multiple_ts(self):
            lags = 4
            lags_past_covariates = 3
            model = RegressionModel(lags=lags, lags_past_covariates=lags_past_covariates)

            target_series = tg.linear_timeseries(start_value=0, end_value=49, length=50)
            past_covariates = tg.linear_timeseries(start_value=100, end_value=149, length=50)
            past_covariates = past_covariates.stack(
                tg.linear_timeseries(start_value=400, end_value=449, length=50))

            target_train, target_test = target_series.split_after(0.7)
            past_covariates_train, past_covariates_test = past_covariates.split_after(0.7)
            model.fit(
                series=[target_train, target_train + 0.5],
                past_covariates=[past_covariates_train, past_covariates_train + 0.5])

            predictions = model.predict(
                10,
                series=[target_train, target_train + 0.5],
                past_covariates=[past_covariates, past_covariates + 0.5])

            self.assertEqual(len(predictions[0]), 10, f"Found {len(predictions)} instead")

            # multiple TS, both future and past covariates, checking that both covariates lead to better results than using
            # a single one (target series = past_cov + future_cov + noise)
            np.random.seed(42)

            linear_ts_1 = tg.linear_timeseries(start_value=10, end_value=59, length=50)
            linear_ts_2 = tg.linear_timeseries(start_value=40, end_value=89, length=50)

            past_covariates = tg.sine_timeseries(length=50) * 10
            future_covariates = tg.sine_timeseries(length=50, value_frequency=0.015) * 50

            target_series_1 = linear_ts_1 + 4 * past_covariates + 2 * future_covariates
            target_series_2 = linear_ts_2 + 4 * past_covariates + 2 * future_covariates

            target_series_1_noise = (linear_ts_1 + 4 * past_covariates + 2 * future_covariates
                                     + tg.gaussian_timeseries(std=7, length=50))

            target_series_2_noise = (linear_ts_2 + 4 * past_covariates + 2 * future_covariates
                                     + tg.gaussian_timeseries(std=7, length=50))

            target_train_1, target_test_1 = target_series_1.split_after(0.7)
            target_train_2, target_test_2 = target_series_2.split_after(0.7)

            target_train_1_noise, target_test_1_noise = target_series_1_noise.split_after(0.7)
            target_train_2_noise, target_test_2_noise = target_series_2_noise.split_after(0.7)

            # testing improved denoise with multiple TS

            # test 1: with single TS, 2 covariates should be better than one
            model = RegressionModel(lags=3, lags_past_covariates=5)
            model.fit([target_train_1_noise], [past_covariates])

            prediction_past_only = model.predict(
                n=len(target_test_1), series=[target_train_1_noise, target_train_2_noise],
                past_covariates=[past_covariates] * 2
            )

            model = RegressionModel(lags=3, lags_past_covariates=5, lags_future_covariates=(5, 0))
            model.fit([target_train_1_noise], [past_covariates], [future_covariates])
            prediction_past_and_future = model.predict(
                n=len(target_test_1), series=[target_train_1_noise, target_train_2_noise],
                past_covariates=[past_covariates] * 2, future_covariates=[future_covariates] * 2
            )

            error_past_only = rmse([target_test_1, target_test_2], prediction_past_only, inter_reduction=np.mean)
            error_both = rmse([target_test_1, target_test_2], prediction_past_and_future, inter_reduction=np.mean)

            self.assertTrue(error_past_only > error_both)
            # test 2: with both covariates, 2 TS should learn more than one (with little noise)
            model = RegressionModel(lags=3, lags_past_covariates=5, lags_future_covariates=(5, 0))
            model.fit([target_train_1_noise, target_train_2_noise], [past_covariates] * 2, [future_covariates] * 2)
            prediction_past_and_future_multi_ts = model.predict(
                n=len(target_test_1), series=[target_train_1_noise, target_train_2_noise],
                past_covariates=[past_covariates] * 2, future_covariates=[future_covariates] * 2
            )
            error_both_multi_ts = rmse([target_test_1, target_test_2], prediction_past_and_future_multi_ts,
                                       inter_reduction=np.mean)

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

            self.assertEqual(len(predictions[0]), 10, f"Found {len(predictions[0])} instead")

        def test_not_enough_future_covariate(self):

            target_series = tg.linear_timeseries(start_value=0, end_value=19, length=20)
            covariates = tg.linear_timeseries(start_value=0, end_value=20, length=20)

            target_train, target_test = target_series.split_after(9)
            covariates_train, covariates_test = covariates.split_after(9)

            model = RegressionModel(lags_future_covariates=[0])
            model.fit(series=target_train, future_covariates=covariates_train)
            # 11 future covariates, with 0 lags covariate, can predict up to 10
            model.predict(10, series=target_train, future_covariates=covariates)
            with self.assertRaises(ValueError):
                model.predict(11, series=target_train, future_covariates=covariates)

            model = RegressionModel(lags_past_covariates=1)
            model.fit(series=target_train, past_covariates=covariates_train)
            # 11 future covariates, without 0 lags covariate, can predict up to 11
            model.predict(11, series=target_train, past_covariates=covariates)
            with self.assertRaises(ValueError):
                model.predict(12, series=target_train, past_covariates=covariates)

        @patch.object(darts.models.forecasting.gradient_boosted_model.lgb.LGBMRegressor, "fit")
        # @patch.object(darts.models.forecasting.gradient_boosted_model.lgb.LGBMRegressor, 'fit')
        def test_gradient_boosted_model_with_eval_set(self, lgb_fit_patch):
            """test whether these evaluation set parameters are passed to LGBRegressor """
            model = LightGBMModel(lags=4, lags_past_covariates=2)
            split_index = 450
            model.fit(series=self.ts_sum1[:split_index],
                      past_covariates=self.ts_cov1[:split_index],
                      val_series=self.ts_sum1[split_index:],
                      val_past_covariates=self.ts_cov1[split_index:],
                      early_stopping_rounds=2,
                      )

            lgb_fit_patch.assert_called_once()

            assert lgb_fit_patch.call_args[1]['eval_set'] is not None
            assert lgb_fit_patch.call_args[1]['early_stopping_rounds'] == 2