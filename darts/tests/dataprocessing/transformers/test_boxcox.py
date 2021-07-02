import unittest
import pandas as pd
from math import log
from copy import deepcopy

from darts.dataprocessing.transformers import BoxCox, Mapper
from darts.utils.timeseries_generation import sine_timeseries, linear_timeseries


class BoxCoxTestCase(unittest.TestCase):

    sine_series = sine_timeseries(length=50, value_y_offset=5, value_frequency=0.05)
    lin_series = linear_timeseries(start_value=1, end_value=10, length=50)
    multi_series = sine_series.stack(lin_series)

    def test_boxbox_lambda(self):
        boxcox = BoxCox(lmbda=0.3)

        boxcox.fit(self.multi_series)
        self.assertEqual(boxcox._fitted_params, [[0.3, 0.3]])

        boxcox = BoxCox(lmbda=[0.3, 0.4])
        boxcox.fit(self.multi_series)
        self.assertEqual(boxcox._fitted_params, [[0.3, 0.4]])

        with self.assertRaises(ValueError):
            boxcox = BoxCox(lmbda=[0.2, 0.4, 0.5])
            boxcox.fit(self.multi_series)

        boxcox = BoxCox(optim_method='mle')
        boxcox.fit(self.multi_series)
        lmbda1 = boxcox._fitted_params[0].tolist()

        boxcox = BoxCox(optim_method='pearsonr')
        boxcox.fit(self.multi_series)
        lmbda2 = boxcox._fitted_params[0].tolist()

        self.assertNotEqual(lmbda1, lmbda2)

    def test_boxcox_transform(self):
        log_mapper = Mapper(lambda x: log(x))
        boxcox = BoxCox(lmbda=0)

        transformed1 = log_mapper.transform(self.sine_series)
        transformed2 = boxcox.fit(self.sine_series).transform(self.sine_series)

        self.assertEqual(transformed1, transformed2)

    def test_boxcox_inverse(self):
        boxcox = BoxCox()
        transformed = boxcox.fit_transform(self.multi_series)
        back = boxcox.inverse_transform(transformed)
        pd.testing.assert_frame_equal(self.multi_series.pd_dataframe(), back.pd_dataframe(), check_exact=False)

    def test_boxcox_multi_ts(self):

        test_cases = [
            ([[0.2, 0.4], [0.3, 0.6]]),  # full lambda
            (0.4),  # single value
            None  # None
        ]

        for lmbda in test_cases:
            box_cox = BoxCox(lmbda=lmbda)
            transformed = box_cox.fit_transform([self.multi_series, self.multi_series])
            back = box_cox.inverse_transform(transformed)
            pd.testing.assert_frame_equal(self.multi_series.pd_dataframe(), back[0].pd_dataframe(), check_exact=False)
            pd.testing.assert_frame_equal(self.multi_series.pd_dataframe(), back[1].pd_dataframe(), check_exact=False)

    def test_boxcox_multiple_calls_to_fit(self):
        """
        This test checks whether calling the scaler twice is calculating new lambdas instead of
        keeping the old ones
        """
        box_cox = BoxCox()

        box_cox.fit(self.sine_series)
        lambda1 = deepcopy(box_cox._fitted_params)[0].tolist()

        box_cox.fit(self.lin_series)
        lambda2 = deepcopy(box_cox._fitted_params)[0].tolist()

        self.assertNotEqual(lambda1, lambda2, "Lambdas should change when the transformer is retrained")
