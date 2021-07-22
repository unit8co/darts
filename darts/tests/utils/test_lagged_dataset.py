from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.timeseries_generation import linear_timeseries
from darts.utils.data.lagged_dataset import (
    LaggedDataset,
    LaggedInferenceDataset,
    _process_lags,
)
from darts.models.linear_regression_model import LinearRegressionModel


class LaggedDatasetTestCase(DartsBaseTestClass):

    target_1 = linear_timeseries(start_value=0, end_value=19, length=20)
    covariate_1 = linear_timeseries(start_value=20, end_value=39, length=20)

    def test_process_lags(self):
        lags, lags_covariate = _process_lags(3, 2)
        self.assertEqual(lags, [1, 2, 3])
        self.assertEqual(lags_covariate, [1, 2])

        lags, lags_covariate = _process_lags(None, [3, 0])
        self.assertEqual(lags, None)
        self.assertEqual(lags_covariate, [3, 0])

        lags, lags_covariate = _process_lags(1, 0)
        self.assertEqual(lags, [1])
        self.assertEqual(lags_covariate, [0])

    def test_no_lags_covariates(self):
        # array of lags
        tests = [[3, 2, 1], [3]]
        for lags in tests:
            X, y = LaggedDataset(target_series=self.target_1, lags=lags).get_data()

            # len(series) - (older_lag + 1 (prediction)) + 1
            expected_length = len(self.target_1) - max(lags)

            self.assertEqual(
                X.shape[0],
                expected_length,
                f"Wrong matrix dimension. Expected {expected_length} number of "
                f"samples, found {X.shape[0]}",
            )

        # multiple TS
        tests = [[3, 2, 1], [3]]
        for lags in tests:
            X, y = LaggedDataset(
                target_series=[self.target_1] * 2, lags=lags
            ).get_data()

            # len(series) - (older_lag + 1 (prediction)) + 1
            expected_length = (len(self.target_1) - max(lags)) * 2

            self.assertEqual(
                X.shape[0],
                expected_length,
                f"Wrong matrix dimension. Expected {expected_length} number of "
                f"samples, found {X.shape[0]}",
            )

    def test_get_data_without_covariate(self):
        X, y = LaggedDataset(target_series=[self.target_1] * 2, lags=[3, 2]).get_data()
        self.assertEqual(X.shape[1], 2)
        self.assertEqual(X.shape[0], y.shape[0])

    def test_get_data_with_covariate_univariate(self):
        X, y = LaggedDataset(
            target_series=[self.target_1] * 2,
            covariates=[self.covariate_1] * 2,
            lags=[3, 2],
            lags_covariates=[1],
        ).get_data()
        self.assertEqual(X.shape[1], 3)
        self.assertEqual(X.shape[0], y.shape[0])

    def test_get_data_with_covariate_multivariate(self):
        X, y = LaggedDataset(
            target_series=[self.target_1] * 2,
            covariates=[self.covariate_1.stack(self.covariate_1 + 10)] * 2,
            lags=[3, 2],
            lags_covariates=[2, 1],
        ).get_data()
        self.assertEqual(X.shape[1], 6)
        self.assertEqual(X.shape[0], y.shape[0])

    def test_lags_and_lags_covariates_with_0_covariate(self):
        lags = [5, 3, 1]
        lags_covariates = [4, 2, 0]

        X, y = LaggedDataset(
            target_series=self.target_1,
            covariates=self.covariate_1,
            lags=lags,
            lags_covariates=lags_covariates,
        ).get_data()

        # len(series) - (older_lag + 1 (prediction)) + 1 - 1 (need an additional covariate)
        expected_length = len(self.target_1) - max(max(lags), max(lags_covariates)) - 1
        print("Shape:", X.shape, y.shape)
        self.assertEqual(
            X.shape[0],
            expected_length,
            f"Wrong matrix dimension. Expected {expected_length} number of "
            f"samples, found {X.shape[0]}",
        )


class LaggedInferenceDatasetTestCase(DartsBaseTestClass):
    target_1 = linear_timeseries(start_value=0, end_value=19, length=20)
    covariate_1 = linear_timeseries(start_value=20, end_value=49, length=30)

    def test_future_covariates(self):
        lags = [3, 2, 1]
        lags_covariates = [3, 2, 0]

        model = LinearRegressionModel(lags=lags, lags_covariates=lags_covariates)
        model.fit(series=[self.target_1] * 2, covariates=[self.covariate_1[:20]] * 2)
        model.predict(n=4, series=[self.target_1] * 2, covariates=[self.covariate_1] * 2)
