from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.timeseries_generation import linear_timeseries
from darts.utils.data.lagged_dataset import LaggedDataset


class LaggedDatasetTestCase(DartsBaseTestClass):

    target_1 = linear_timeseries(start_value=0, end_value=19, length=20)
    covariate_1 = linear_timeseries(start_value=20, end_value=39, length=20)

    def test_no_lags_covariates(self):
        # array of lags
        lags = [3, 2, 1]
        X, y = LaggedDataset(target_series=self.target_1, lags=lags).get_data()

        # len(series) - (older_lag + 1 (prediction)) + 1
        expected_length = len(self.target_1) - max(lags)

        self.assertEqual(
            X.shape[0],
            expected_length,
            f"Wrong matrix dimension. Expected {expected_length} number of "
            f"samples, found {X.shape[0]}",
        )

    def test_lags_and_lags_covariates(self):
        pass

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
