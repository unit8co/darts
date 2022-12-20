import numpy as np

from darts.logging import get_logger
from darts.models import LinearRegressionModel, NaiveSeasonal
from darts.tests.base_test_class import DartsBaseTestClass
from darts.tests.models.forecasting.test_regression_models import dummy_timeseries
from darts.utils.timeseries_generation import constant_timeseries as ct
from darts.utils.timeseries_generation import linear_timeseries as lt

logger = get_logger(__name__)


class TestResidualsTestCase(DartsBaseTestClass):

    np.random.seed(42)

    def test_forecasting_residuals_nocov_output(self):
        model = NaiveSeasonal(K=1)

        # test zero residuals
        constant_ts = ct(length=20)
        residuals = model.residuals(constant_ts)
        np.testing.assert_almost_equal(
            residuals.univariate_values(), np.zeros(len(residuals))
        )

        # test constant, positive residuals
        linear_ts = lt(length=20)
        residuals = model.residuals(linear_ts)
        np.testing.assert_almost_equal(
            np.diff(residuals.univariate_values()), np.zeros(len(residuals) - 1)
        )
        np.testing.assert_array_less(
            np.zeros(len(residuals)), residuals.univariate_values()
        )

    def test_forecasting_residuals_inputs(self):
        # test input types past and/or future covariates

        # dummy covariates and target TimeSeries instances

        target_series, past_covariates, future_covariates = dummy_timeseries(
            length=10,
            n_series=1,
            comps_target=1,
            comps_pcov=1,
            comps_fcov=1,
        )  # outputs Sequences[TimeSeries] and not TimeSeries

        model = LinearRegressionModel(
            lags=4, lags_past_covariates=4, lags_future_covariates=(4, 1)
        )
        model.fit(
            series=target_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

    def test_forecasting_residuals_cov_output(self):
        # if covariates are constant and the target is constant/linear,
        # residuals should be zero (for a LinearRegression model)

        target_series_1 = ct(value=0.5, length=10)
        target_series_2 = lt(length=10)
        past_covariates = ct(value=0.2, length=10)
        future_covariates = ct(value=0.1, length=10)

        model_1 = LinearRegressionModel(
            lags=1, lags_past_covariates=1, lags_future_covariates=(1, 1)
        )
        model_2 = LinearRegressionModel(
            lags=1, lags_past_covariates=1, lags_future_covariates=(1, 1)
        )
        model_1.fit(
            target_series_1,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        residuals_1 = model_1.residuals(
            target_series_1,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        model_2.fit(
            target_series_2,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        residuals_2 = model_2.residuals(
            target_series_2,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # residuals zero
        np.testing.assert_almost_equal(
            residuals_1.univariate_values(), np.zeros(len(residuals_1))
        )

        np.testing.assert_almost_equal(
            residuals_2.univariate_values(), np.zeros(len(residuals_2))
        )

        # if model is trained with covariates, should raise error when covariates are missing in residuals()
        with self.assertRaises(ValueError):
            model_1.residuals(target_series_1)

        with self.assertRaises(ValueError):
            model_1.residuals(target_series_1, past_covariates=past_covariates)

        with self.assertRaises(ValueError):
            model_1.residuals(target_series_1, future_covariates=future_covariates)
