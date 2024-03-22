import numpy as np
import pytest

from darts.logging import get_logger
from darts.models import LinearRegressionModel, NaiveSeasonal
from darts.tests.models.forecasting.test_regression_models import dummy_timeseries
from darts.utils.timeseries_generation import constant_timeseries as ct
from darts.utils.timeseries_generation import linear_timeseries as lt

logger = get_logger(__name__)


class TestResiduals:

    np.random.seed(42)

    def test_forecasting_residuals_nocov_output(self):
        model = NaiveSeasonal(K=1)

        # test zero residuals
        constant_ts = ct(length=20)
        residuals = model.residuals(constant_ts)
        np.testing.assert_almost_equal(
            residuals.univariate_values(), np.zeros(len(residuals))
        )
        residuals_vals = model.residuals(constant_ts, values_only=True)
        np.testing.assert_almost_equal(residuals.all_values(), residuals_vals)

        # test constant, positive residuals
        linear_ts = lt(length=20)
        residuals = model.residuals(linear_ts)
        np.testing.assert_almost_equal(
            np.diff(residuals.univariate_values()), np.zeros(len(residuals) - 1)
        )
        np.testing.assert_array_less(
            np.zeros(len(residuals)), residuals.univariate_values()
        )
        residuals_vals = model.residuals(linear_ts, values_only=True)
        np.testing.assert_almost_equal(residuals.all_values(), residuals_vals)

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

    @pytest.mark.parametrize(
        "series",
        [
            ct(value=0.5, length=10),
            lt(length=10),
        ],
    )
    def test_forecasting_residuals_cov_output(self, series):
        # if covariates are constant and the target is constant/linear,
        # residuals should be zero (for a LinearRegression model)
        past_covariates = ct(value=0.2, length=10)
        future_covariates = ct(value=0.1, length=10)

        model = LinearRegressionModel(
            lags=1, lags_past_covariates=1, lags_future_covariates=(1, 1)
        )
        model.fit(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        # residuals TimeSeries zero
        res = model.residuals(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        np.testing.assert_almost_equal(res.univariate_values(), np.zeros(len(res)))

        # return values only
        res_vals = model.residuals(
            series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            values_only=True,
        )
        np.testing.assert_almost_equal(res.all_values(), res_vals)

        # with precomputed historical forecasts
        hfc = model.historical_forecasts(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )
        res_hfc = model.residuals(series, historical_forecasts=hfc)
        assert res == res_hfc

        # with pretrained model
        res_pretrained = model.residuals(
            series,
            start=model.min_train_series_length,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            retrain=False,
            values_only=True,
        )
        np.testing.assert_almost_equal(res_vals, res_pretrained)

        # if model is trained with covariates, should raise error when covariates are missing in residuals()
        with pytest.raises(ValueError):
            model.residuals(series)

        with pytest.raises(ValueError):
            model.residuals(series, past_covariates=past_covariates)

        with pytest.raises(ValueError):
            model.residuals(series, future_covariates=future_covariates)
