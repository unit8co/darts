from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.dataprocessing.transformers import BoxCox, Mapper
from darts.utils.timeseries_generation import linear_timeseries, sine_timeseries


class TestBoxCox:
    sine_series = sine_timeseries(length=50, value_y_offset=5, value_frequency=0.05)
    lin_series = linear_timeseries(start_value=1, end_value=10, length=50)
    multi_series = sine_series.stack(lin_series)

    def test_boxbox_lambda(self):
        boxcox = BoxCox(lmbda=0.3)

        boxcox.fit(self.multi_series)
        assert boxcox._fitted_params == [[0.3, 0.3]]

        boxcox = BoxCox(lmbda=[0.3, 0.4])
        boxcox.fit(self.multi_series)
        assert boxcox._fitted_params == [[0.3, 0.4]]

        with pytest.raises(ValueError):
            boxcox = BoxCox(lmbda=[0.2, 0.4, 0.5])
            boxcox.fit(self.multi_series)

        boxcox = BoxCox(optim_method="mle")
        boxcox.fit(self.multi_series)
        lmbda1 = boxcox._fitted_params[0].tolist()

        boxcox = BoxCox(optim_method="pearsonr")
        boxcox.fit(self.multi_series)
        lmbda2 = boxcox._fitted_params[0].tolist()

        assert lmbda1 != lmbda2

    def test_boxcox_transform(self):
        log_mapper = Mapper(lambda x: np.log(x))
        boxcox = BoxCox(lmbda=0)

        transformed1 = log_mapper.transform(self.sine_series)
        transformed2 = boxcox.fit(self.sine_series).transform(self.sine_series)

        np.testing.assert_almost_equal(
            transformed1.all_values(copy=False),
            transformed2.all_values(copy=False),
            decimal=4,
        )

    def test_boxcox_inverse(self):
        boxcox = BoxCox()
        transformed = boxcox.fit_transform(self.multi_series)
        back = boxcox.inverse_transform(transformed)
        pd.testing.assert_frame_equal(
            self.multi_series.to_dataframe(), back.to_dataframe(), check_exact=False
        )

    def test_boxcox_multi_ts(self):
        test_cases = [
            ([[0.2, 0.4], [0.3, 0.6]]),  # full lambda
            (0.4),  # single value
            None,  # None
        ]

        for lmbda in test_cases:
            box_cox = BoxCox(lmbda=lmbda)
            transformed = box_cox.fit_transform([self.multi_series, self.multi_series])
            back = box_cox.inverse_transform(transformed)
            pd.testing.assert_frame_equal(
                self.multi_series.to_dataframe(),
                back[0].to_dataframe(),
                check_exact=False,
            )
            pd.testing.assert_frame_equal(
                self.multi_series.to_dataframe(),
                back[1].to_dataframe(),
                check_exact=False,
            )

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

        assert lambda1 != lambda2, (
            "Lambdas should change when the transformer is retrained"
        )

    def test_multivariate_stochastic_series(self):
        transformer = BoxCox()
        vals = np.random.rand(10, 5, 10)
        series = TimeSeries.from_values(vals)

        new_series = transformer.fit_transform(series)
        series_back = transformer.inverse_transform(new_series)

        # Test inverse transform
        np.testing.assert_allclose(series.all_values(), series_back.all_values())

    def test_global_fitting(self):
        """
        Tests that `BoxCox` correctly handles situation where `global_fit = True`. More
        specifically, test checks that global fitting with two disjoint series
        produces same fitted parameters as local fitting with a single series formed
        by 'gluing' these two disjoint series together.
        """
        series_combined = self.sine_series.append_values(self.lin_series.all_values())
        local_params = BoxCox(global_fit=False).fit(series_combined)._fitted_params
        global_params = (
            BoxCox(global_fit=True)
            .fit([self.sine_series, self.lin_series])
            ._fitted_params
        )
        assert local_params == global_params
