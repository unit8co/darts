import unittest

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import MIDAS
from darts.models import LinearRegressionModel


class MIDASTestCase(unittest.TestCase):
    monthly_values = np.arange(1, 10)
    monthly_times = pd.date_range(start="01-2020", periods=9, freq="M")
    monthly_ts = TimeSeries.from_times_and_values(
        times=monthly_times, values=monthly_values, columns=["values"]
    )

    monthly_not_complete_ts = monthly_ts[2:-1]

    quarterly_values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    quarterly_times = pd.date_range(start="01-2020", periods=3, freq="QS")
    quarterly_ts = TimeSeries.from_times_and_values(
        times=quarterly_times,
        values=quarterly_values,
        columns=["values_0", "values_1", "values_2"],
    )

    quarterly_end_times = pd.date_range(start="01-2020", periods=3, freq="Q")
    quarterly_with_quarter_end_index_ts = TimeSeries.from_times_and_values(
        times=quarterly_end_times,
        values=quarterly_values,
        columns=["values_0", "values_1", "values_2"],
    )

    quarterly_not_complete_values = np.array(
        [[np.nan, np.nan, 3], [4, 5, 6], [7, 8, np.nan]]
    )
    quarterly_not_complete_ts = TimeSeries.from_times_and_values(
        times=quarterly_times,
        values=quarterly_not_complete_values,
        columns=["values_0", "values_1", "values_2"],
    )

    daily_times = pd.date_range(start="01-2020", end="09-30-2020", freq="D")
    daily_values = np.arange(1, len(daily_times) + 1)
    daily_ts = TimeSeries.from_times_and_values(
        times=daily_times, values=daily_values, columns=["values"]
    )

    second_times = pd.date_range(start="01-2020", periods=120, freq="S")
    second_values = np.arange(1, len(second_times) + 1)
    second_ts = TimeSeries.from_times_and_values(
        times=second_times, values=second_values, columns=["values"]
    )

    minute_times = pd.date_range(start="01-2020", periods=2, freq="T")
    minute_values = np.array([[i for i in range(1, 61)], [i for i in range(61, 121)]])
    minute_ts = TimeSeries.from_times_and_values(
        times=minute_times,
        values=minute_values,
        columns=[f"values_{i}" for i in range(60)],
    )

    def test_complete_monthly_to_quarterly(self):
        """
        Tests if monthly series aligned with quarters is transformed into a quarterly series in the expected way.
        """
        # to quarter start
        midas_1 = MIDAS(rule="QS")
        quarterly_ts_midas = midas_1.fit_transform(self.monthly_ts)
        self.assertEqual(
            quarterly_ts_midas,
            self.quarterly_ts,
            "Monthly TimeSeries is not correctly transformed "
            "into a quarterly TimeSeries.",
        )

        inversed_quarterly_ts_midas = midas_1.inverse_transform(quarterly_ts_midas)
        self.assertEqual(
            self.monthly_ts,
            inversed_quarterly_ts_midas,
            "Quarterly TimeSeries is not correctly inverse_transformed "
            "back into into a monthly TimeSeries.",
        )

        # to quarter end
        midas_2 = MIDAS(rule="Q")
        quarterly_ts_midas = midas_2.fit_transform(self.monthly_ts)
        self.assertEqual(
            quarterly_ts_midas,
            self.quarterly_with_quarter_end_index_ts,
            "Monthly TimeSeries is not correctly transformed "
            "into a quarterly TimeSeries. Specifically, when the rule requires an QuarterEnd index.",
        )

        inversed_quarterly_ts_midas = midas_2.inverse_transform(quarterly_ts_midas)
        self.assertEqual(
            self.monthly_ts,
            inversed_quarterly_ts_midas,
            "Quarterly TimeSeries is not correctly inverse_transformed "
            "back into into a monthly TimeSeries.",
        )

    def test_not_complete_monthly_to_quarterly(self):
        """
        Check that an univariate monthly series not aligned with quarters is transformed into a quarterly series
        in the expected way.
        """
        # monthly series with missing values
        midas = MIDAS(rule="QS", strip=False)
        quarterly_not_complete_ts_midas = midas.fit_transform(
            self.monthly_not_complete_ts
        )
        self.assertEqual(
            quarterly_not_complete_ts_midas,
            self.quarterly_not_complete_ts,
            "Monthly TimeSeries is not "
            "correctly transformed when"
            " it is not 'complete'.",
        )
        inversed_quarterly_not_complete_ts_midas = midas.inverse_transform(
            quarterly_not_complete_ts_midas
        )
        self.assertEqual(
            self.monthly_not_complete_ts,
            inversed_quarterly_not_complete_ts_midas,
            "Quarterly TimeSeries is not correctly inverse_transformed "
            "back into into a monthly TimeSeries with missing values.",
        )

    def test_multivariate_monthly_to_quarterly(self):
        """
        Check that multivariate monthly to quarterly is properly transformed
        """
        stacked_monthly_ts = self.monthly_ts.stack(
            TimeSeries.from_times_and_values(
                times=self.monthly_ts.time_index,
                values=np.arange(10, 19),
                columns=["other"],
            )
        )

        # component components are alternating
        expected_quarterly_ts = TimeSeries.from_times_and_values(
            times=self.quarterly_ts.time_index,
            values=np.array(
                [[1, 10, 2, 11, 3, 12], [4, 13, 5, 14, 6, 15], [7, 16, 8, 17, 9, 18]]
            ),
            columns=[
                "values_0",
                "other_0",
                "values_1",
                "other_1",
                "values_2",
                "other_2",
            ],
        )

        midas_1 = MIDAS(rule="QS")
        multivar_quarterly_ts_midas = midas_1.fit_transform(stacked_monthly_ts)
        self.assertEqual(
            multivar_quarterly_ts_midas,
            expected_quarterly_ts,
            "Multivariate monthly TimeSeries is not correctly transformed "
            "into a quarterly TimeSeries.",
        )

        multivar_inversed_quarterly_ts_midas = midas_1.inverse_transform(
            multivar_quarterly_ts_midas
        )
        self.assertEqual(
            stacked_monthly_ts,
            multivar_inversed_quarterly_ts_midas,
            "Multivariate quarterly TimeSeries is not correctly inverse_transformed "
            "back into into a monthly TimeSeries.",
        )

    def test_ts_with_missing_data(self):
        """
        Check that multivariate monthly to quarterly with missing data in the middle is properly transformed.
        """
        stacked_monthly_ts_missing = self.monthly_ts.stack(
            TimeSeries.from_times_and_values(
                times=self.monthly_ts.time_index,
                values=np.array([10, 11, 12, np.nan, np.nan, 15, 16, 17, 18]),
                columns=["other"],
            )
        )

        # component components are alternating
        expected_quarterly_ts = TimeSeries.from_times_and_values(
            times=self.quarterly_ts.time_index,
            values=np.array(
                [
                    [1, 10, 2, 11, 3, 12],
                    [4, np.nan, 5, np.nan, 6, 15],
                    [7, 16, 8, 17, 9, 18],
                ]
            ),
            columns=[
                "values_0",
                "other_0",
                "values_1",
                "other_1",
                "values_2",
                "other_2",
            ],
        )

        midas_1 = MIDAS(rule="QS")
        multivar_quarterly_ts_midas = midas_1.fit_transform(stacked_monthly_ts_missing)
        self.assertEqual(
            multivar_quarterly_ts_midas,
            expected_quarterly_ts,
        )

        multivar_inversed_quarterly_ts_midas = midas_1.inverse_transform(
            multivar_quarterly_ts_midas
        )
        self.assertEqual(
            stacked_monthly_ts_missing,
            multivar_inversed_quarterly_ts_midas,
        )

    def test_from_second_to_minute(self):
        """
        Test to see if other frequencies transforms like second to minute work as well.
        """
        midas = MIDAS(rule="T")
        minute_ts_midas = midas.fit_transform(self.second_ts)
        self.assertEqual(minute_ts_midas, self.minute_ts)
        second_ts_midas = midas.inverse_transform(minute_ts_midas)
        self.assertEqual(second_ts_midas, self.second_ts)

    def test_error_when_from_low_to_high(self):
        """
        Tests if the transformer raises an error when the user asks for a transform in the wrong direction.
        """
        # wrong direction / low to high freq
        midas_1 = MIDAS(rule="M")
        self.assertRaises(ValueError, midas_1.fit_transform, self.quarterly_ts)

        # transform to same index requested
        midas_2 = MIDAS(rule="Q")
        self.assertRaises(ValueError, midas_2.fit_transform, self.quarterly_ts)

    def test_error_when_frequency_not_suitable_for_midas(self):
        """
        MIDAS can only be performed when the high frequency is the same and the exact multiple of the low frequency.
        For example, there are always exactly three months in a quarter, but the number of days in a month differs.
        So the monthly to quarterly transformation is possible, while the daily to monthly MIDAS transform is
        impossible.
        """
        midas = MIDAS(rule="M")
        self.assertRaises(ValueError, midas.fit_transform, self.daily_ts)

    def test_inverse_transform_prediction(self):
        """
        Check that inverse-transforming the prediction of a model generate the correct time index when
        using frequency anchored either at the start or the end of the quarter
        """
        # low frequency : QuarterStart
        monthly_ts = TimeSeries.from_times_and_values(
            times=pd.date_range(start="01-2020", periods=24, freq="M"),
            values=np.arange(0, 24),
            columns=["values"],
        )
        monthly_train_ts, monthly_test_ts = monthly_ts.split_after(0.75)

        model = LinearRegressionModel(lags=2)

        midas_quarterly = MIDAS(rule="QS")
        # shape : [6 quarters, 3 months, 1 sample]
        quarterly_train_ts = midas_quarterly.fit_transform(monthly_train_ts)
        # shape : [2 quarters, 3 months, 1 sample]
        quarterly_test_ts = midas_quarterly.transform(monthly_test_ts)

        model.fit(quarterly_train_ts)

        # 2 quarters = 6 months forecast
        pred_quarterly = model.predict(2)
        pred_monthly = midas_quarterly.inverse_transform(pred_quarterly)
        # verify prediction time index in both frequencies
        self.assertTrue(pred_quarterly.time_index.equals(quarterly_test_ts.time_index))
        self.assertTrue(pred_monthly.time_index.equals(monthly_test_ts.time_index))

        # "Q" = QuarterEnd, the 2 "hidden" months must be retrieved
        midas_quarterly = MIDAS(rule="Q")
        quarterly_train_ts = midas_quarterly.fit_transform(monthly_train_ts)
        quarterly_test_ts = midas_quarterly.transform(monthly_test_ts)

        model.fit(quarterly_train_ts)

        pred_quarterly = model.predict(2)
        pred_monthly = midas_quarterly.inverse_transform(pred_quarterly)
        # verify prediction time index in both frequencies
        self.assertTrue(pred_quarterly.time_index.equals(quarterly_test_ts.time_index))
        self.assertTrue(pred_monthly.time_index.equals(monthly_test_ts.time_index))
