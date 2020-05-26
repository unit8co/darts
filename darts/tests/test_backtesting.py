import unittest
import numpy as np

from ..backtesting import forecasting_residuals
from ..models.baselines import NaiveSeasonal
from ..utils.timeseries_generation import constant_timeseries, linear_timeseries


class BacktestingTestCase(unittest.TestCase):

    def test_forecasting_residuals(self):
        model = NaiveSeasonal(K=1)

        # test zero residuals
        constant_ts = constant_timeseries(length=20)
        residuals = forecasting_residuals(model, constant_ts)
        np.testing.assert_almost_equal(residuals.values(), np.zeros(len(residuals)))

        # test constant, positive residuals
        linear_ts = linear_timeseries(length=20)
        residuals = forecasting_residuals(model, linear_ts)
        np.testing.assert_almost_equal(np.diff(residuals.values()), np.zeros(len(residuals) - 1))
        np.testing.assert_array_less(np.zeros(len(residuals)), residuals.values())
