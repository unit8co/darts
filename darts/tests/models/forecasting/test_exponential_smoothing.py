import numpy as np

from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils import timeseries_generation as tg


class ExponentialSmoothingTestCase(DartsBaseTestClass):
    def helper_test_seasonality_inference(self, freq_string, expected_seasonal_periods):
        series = tg.sine_timeseries(length=200, freq=freq_string)
        model = ExponentialSmoothing()
        model.fit(series)
        self.assertEqual(model.seasonal_periods, expected_seasonal_periods)

    def test_seasonality_inference(self):

        # test `seasonal_periods` inference for datetime indices
        freq_str_seasonality_periods_tuples = [
            ("D", 7),
            ("H", 24),
            ("M", 12),
            ("W", 52),
            ("Q", 4),
            ("B", 5),
        ]
        for tuple in freq_str_seasonality_periods_tuples:
            self.helper_test_seasonality_inference(*tuple)

        # test default selection for integer index
        series = TimeSeries.from_values(np.arange(1, 30, 1))
        model = ExponentialSmoothing()
        model.fit(series)
        self.assertEqual(model.seasonal_periods, 12)

        # test whether a model that inferred a seasonality period before will do it again for a new series
        series1 = tg.sine_timeseries(length=100, freq="M")
        series2 = tg.sine_timeseries(length=100, freq="D")
        model = ExponentialSmoothing()
        model.fit(series1)
        model.fit(series2)
        self.assertEqual(model.seasonal_periods, 7)
