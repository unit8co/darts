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

    def test_forecasting_residuals(self):
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

        # test with past and/or future covariates

        # dummy feature and target TimeSeries instances

        target_series, past_covariates, future_covariates = dummy_timeseries(
            length=100,
            n_series=1,
            comps_target=1,
            comps_pcov=1,
            comps_fcov=1,
        )  # outputs lists and not TimeSeries

        model_instance = LinearRegressionModel(
            lags=4, lags_past_covariates=4, lags_future_covariates=(4, 1)
        )
        model_instance.fit(
            series=target_series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        # it seems that models can be fit with data objects that are not necessarily TimeSeries, but residuals() will
        # fail because it starts by asserting that the provided object is a univariate TimeSeries.
        # The following asserts the correct TimeSeries types

        with self.assertRaises(ValueError):
            model.residuals(target_series)

        with self.assertRaises(ValueError):
            model.residuals(target_series, past_covariates=past_covariates)

        with self.assertRaises(ValueError):
            model.residuals(target_series, future_covariates=future_covariates)

        with self.assertRaises(ValueError):
            model.residuals(
                target_series,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )
