import numpy as np
import pandas as pd

from .. import TimeSeries
from ..metrics import rmse
from ..models import RegressionModel, RandomForest, LinearRegressionModel
from .base_test_class import DartsBaseTestClass
from ..utils import timeseries_generation as tg
from sklearn.linear_model import LinearRegression
from sklearn.experimental import (
    enable_hist_gradient_boosting,
)  # enable import of HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


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


class RegressionModelsTestCase(DartsBaseTestClass):

    np.random.seed(42)

    # dummy feature and target TimeSeries instances
    ts_periodic = tg.sine_timeseries(length=500)
    ts_gaussian = tg.gaussian_timeseries(length=500)
    ts_random_walk = tg.random_walk_timeseries(length=500)

    ts_exog1 = ts_periodic.stack(ts_gaussian)
    ts_exog1 = ts_exog1.pd_dataframe()
    ts_exog1.columns = ["Periodic", "Gaussian"]
    ts_exog1 = TimeSeries.from_dataframe(ts_exog1)
    ts_sum1 = ts_periodic + ts_gaussian

    ts_exog2 = ts_sum1.stack(ts_random_walk)
    ts_sum2 = ts_sum1 + ts_random_walk

    # default regression models
    models = [RandomForest, LinearRegressionModel, RegressionModel]
    lags = 4
    lags_exog = [3, 4, 5]

    def test_model_construction(self):

        for model in self.models:
            model_instance = model(lags=5)
            self.assertEqual(model_instance.lags, [1, 2, 3, 4, 5])

            model_instance = model(lags=None, lags_covariates=3)
            self.assertEqual(model_instance.lags_covariates, [1, 2, 3])

            model_instance = model(lags=5, lags_covariates=[3, 6, 9, 12])
            self.assertEqual(model_instance.lags_covariates, [3, 6, 9, 12])

            model_instance = model(lags=None, lags_covariates=0)
            self.assertEqual(model_instance.lags_covariates, [0])

            with self.assertRaises(ValueError):
                model()
            with self.assertRaises(ValueError):
                model(lags=0)
            with self.assertRaises(ValueError):
                model(lags=[3, 4, 5, 0])
            with self.assertRaises(ValueError):
                model(lags=[3, 4.0, 5])
            with self.assertRaises(ValueError):
                model(lags=-5)
            with self.assertRaises(ValueError):
                model(lags=3.6)
            with self.assertRaises(ValueError):
                model(lags=None, lags_covariates=False)
            with self.assertRaises(ValueError):
                model(lags=0)
            with self.assertRaises(ValueError):
                model(lags=5, lags_covariates=True)

    def test_models_runnability(self):
        train_x, test_x = self.ts_exog1.split_before(0.7)
        train_y, test_y = self.ts_sum1.split_before(0.7)
        for model in self.models:
            model_instance = model(lags=4)
            model_instance.fit(series=self.ts_sum1)
            prediction = model_instance.predict(n=20)
            self.assertTrue(
                len(prediction) == 20,
                f"Expected length 20, found {len(prediction)} instead",
            )

            model_instance = model(lags=4, lags_covariates=2)
            model_instance.fit(
                series=[train_y, train_y + 1], covariates=[train_x, train_x + 100]
            )
            prediction = model_instance.predict(
                n=10,
                series=[train_y, train_y + 1],
                covariates=[self.ts_exog1, self.ts_exog1 + 100],
            )

            self.assertTrue(len(prediction[0]) == 10)

            with self.assertRaises(ValueError):
                model_instance = model(lags=4, lags_covariates=None)
                model_instance.fit(series=self.ts_sum1, covariates=self.ts_exog1)

            with self.assertRaises(ValueError):
                model_instance = model(lags=4, lags_covariates=3)
                model_instance.fit(series=self.ts_sum1)

    def test_fit(self):
        for model in self.models:
            with self.assertRaises(ValueError):
                model_instance = model(lags=4, lags_covariates=4)
                model_instance.fit(series=self.ts_sum1, covariates=self.ts_exog1)
                prediction = model_instance.predict(n=10)

            model_instance = model(lags=12)
            model_instance.fit(series=self.ts_sum1)
            self.assertEqual(model_instance.lags_covariates, None)

            model_instance = model(lags=12, lags_covariates=12)
            model_instance.fit(series=self.ts_sum1, covariates=self.ts_exog1)
            self.assertEqual(len(model_instance.lags_covariates), 12)

            model_instance = model(lags=12, lags_covariates=0)
            model_instance.fit(series=self.ts_sum1, covariates=self.ts_exog1)
            self.assertEqual(len(model_instance.lags_covariates), 1)

            model_instance = model(lags=12, lags_covariates=[1, 4, 6])
            model_instance.fit(series=self.ts_sum1, covariates=self.ts_exog1)
            self.assertEqual(len(model_instance.lags_covariates), 3)

    def helper_test_models_accuracy(self, series, covariates, min_rmse):
        # for every model, test whether it predicts the target with a minimum r2 score of `min_rmse`
        train_f, train_t, test_f, test_t = train_test_split(
            covariates, series, pd.Timestamp("20010101")
        )

        for model in self.models:
            model_instance = model(lags=12, lags_covariates=2)
            model_instance.fit(series=train_t, covariates=train_f)
            prediction = model_instance.predict(n=len(test_t), covariates=covariates)
            current_rmse = rmse(prediction, test_t)

            self.assertTrue(
                current_rmse <= min_rmse,
                (
                    "{} model was not able to denoise data. A rmse score of {} was recorded.".format(
                        str(model_instance), current_rmse
                    )
                ),
            )

    def test_models_denoising(self):
        # for every model, test whether it correctly denoises ts_sum using ts_gaussian and ts_sum as inputs
        self.helper_test_models_accuracy(self.ts_sum1, self.ts_exog1, 1.5)

    def test_models_denoising_multi_input(self):
        # for every model, test whether it correctly denoises ts_sum_2 using ts_random_multi and ts_sum_2 as inputs
        self.helper_test_models_accuracy(self.ts_sum2, self.ts_exog2, 19)

    def test_historical_forecast(self):
        model = self.models[0](lags=5)
        result = model.historical_forecasts(
            series=self.ts_sum1[:100],
            covariates=None,
            start=0.5,
            forecast_horizon=1,
            stride=1,
            retrain=True,
            overlap_end=False,
            last_points_only=True,
            verbose=False,
        )
        self.assertEqual(len(result), 51)

        model = self.models[0](lags=5, lags_covariates=5)
        result = model.historical_forecasts(
            series=self.ts_sum1[:100],
            covariates=self.ts_exog1[:100],
            start=0.5,
            forecast_horizon=1,
            stride=1,
            retrain=True,
            overlap_end=False,
            last_points_only=True,
            verbose=False,
        )
        self.assertEqual(len(result), 51)

    def test_regression_model(self):
        lags = 12
        models = [
            RegressionModel(lags=lags),
            RegressionModel(lags=lags, model=LinearRegression()),
            RegressionModel(lags=lags, model=RandomForestRegressor()),
            RegressionModel(lags=lags, model=HistGradientBoostingRegressor()),
        ]

        for model in models:
            model.fit(series=self.ts_sum1)
            self.assertEqual(len(model.lags), lags)
            model.predict(n=10)

    def test_my_test(self):
        lags = 4
        lags_covariates = 3
        model = RegressionModel(lags=lags, lags_covariates=lags_covariates)

        target_series = tg.linear_timeseries(start_value=0, end_value=49, length=50)
        covariates = tg.linear_timeseries(start_value=100, end_value=149, length=50)
        covariates = covariates.stack(
            tg.linear_timeseries(start_value=400, end_value=449, length=50)
        )

        target_train, target_test = target_series.split_after(0.7)
        covariates_train, covariates_test = covariates.split_after(0.7)
        model.fit(
            series=[target_train, target_train + 0.5],
            covariates=[covariates_train, covariates_train + 0.5],
        )

        predictions = model.predict(
            10,
            series=[target_train, target_train + 0.5],
            covariates=[covariates, covariates + 0.5],
        )

        self.assertEqual(len(predictions[0]), 10, f"Found {len(predictions)} instead")

    def test_only_future_covariates(self):

        model = RegressionModel(lags_covariates=0)

        target_series = tg.linear_timeseries(start_value=0, end_value=49, length=50)
        covariates = tg.linear_timeseries(start_value=100, end_value=149, length=50)
        covariates = covariates.stack(
            tg.linear_timeseries(start_value=400, end_value=449, length=50)
        )

        target_train, target_test = target_series.split_after(0.7)
        covariates_train, covariates_test = covariates.split_after(0.7)
        model.fit(
            series=[target_train, target_train + 0.5],
            covariates=[covariates_train, covariates_train + 0.5],
        )

        predictions = model.predict(
            10,
            series=[target_train, target_train + 0.5],
            covariates=[covariates, covariates + 0.5],
        )

        self.assertEqual(
            len(predictions[0]), 10, f"Found {len(predictions[0])} instead"
        )
