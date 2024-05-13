import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.dataprocessing.transformers import MIDAS
from darts.models import LinearRegressionModel
from darts.utils.timeseries_generation import linear_timeseries
from darts.utils.utils import freqs, generate_index


class TestMIDAS:
    monthly_ts = linear_timeseries(
        start_value=1,
        end_value=9,
        start=pd.Timestamp("01-2020"),
        length=9,
        freq=freqs["ME"],
        column_name="values",
    )

    monthly_not_complete_ts = monthly_ts[2:-1]

    quarterly_values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    quarterly_times = pd.date_range(start="01-2020", periods=3, freq="QS")
    quarterly_ts = TimeSeries.from_times_and_values(
        times=quarterly_times,
        values=quarterly_values,
        columns=["values_midas_0", "values_midas_1", "values_midas_2"],
    )

    quarterly_end_times = pd.date_range(start="01-2020", periods=3, freq=freqs["QE"])
    quarterly_with_quarter_end_index_ts = TimeSeries.from_times_and_values(
        times=quarterly_end_times,
        values=quarterly_values,
        columns=["values_midas_0", "values_midas_1", "values_midas_2"],
    )

    quarterly_not_complete_values = np.array([
        [np.nan, np.nan, 3],
        [4, 5, 6],
        [7, 8, np.nan],
    ])
    quarterly_not_complete_ts = TimeSeries.from_times_and_values(
        times=quarterly_times,
        values=quarterly_not_complete_values,
        columns=["values_midas_0", "values_midas_1", "values_midas_2"],
    )

    def test_complete_monthly_to_quarterly(self):
        """
        Tests if monthly series aligned with quarters is transformed into a quarterly series in the expected way.
        """
        # to quarter start
        midas_1 = MIDAS(low_freq="QS")
        quarterly_ts_midas = midas_1.fit_transform(self.monthly_ts)
        assert quarterly_ts_midas == self.quarterly_ts

        inversed_quarterly_ts_midas = midas_1.inverse_transform(quarterly_ts_midas)
        assert self.monthly_ts == inversed_quarterly_ts_midas

        # to quarter end
        midas_2 = MIDAS(low_freq=freqs["QE"])
        quarterly_ts_midas = midas_2.fit_transform(self.monthly_ts)
        assert quarterly_ts_midas == self.quarterly_with_quarter_end_index_ts

        inversed_quarterly_ts_midas = midas_2.inverse_transform(quarterly_ts_midas)
        assert self.monthly_ts == inversed_quarterly_ts_midas

    def test_not_complete_monthly_to_quarterly(self):
        """
        Check that an univariate monthly series not aligned with quarters is transformed into a quarterly series
        in the expected way.
        """
        # monthly series with missing values
        midas = MIDAS(low_freq="QS", strip=False)
        quarterly_not_complete_ts_midas = midas.fit_transform(
            self.monthly_not_complete_ts
        )
        assert quarterly_not_complete_ts_midas == self.quarterly_not_complete_ts

        inversed_quarterly_not_complete_ts_midas = midas.inverse_transform(
            quarterly_not_complete_ts_midas
        )
        assert self.monthly_not_complete_ts == inversed_quarterly_not_complete_ts_midas

        # when strip=True we only get 1 one quarter with all 3 months
        midas = MIDAS(low_freq="QS", strip=True)
        quarterly_not_complete_ts_midas = midas.fit_transform(
            self.monthly_not_complete_ts
        )
        assert quarterly_not_complete_ts_midas == self.quarterly_not_complete_ts[1:2]

        inversed_quarterly_not_complete_ts_midas = midas.inverse_transform(
            quarterly_not_complete_ts_midas
        )
        assert (
            self.monthly_not_complete_ts[1:4]
            == inversed_quarterly_not_complete_ts_midas
        )

    def test_probabilistic_complete_monthly_to_quarterly(self):
        """
        Tests MIDAS on probabilistic series on example of monthly series aligned with quarters.
        """

        # generate probabilistic monthly series
        all_vals_monthly = self.monthly_ts.all_values(copy=False)
        prob_values_monthly = np.concatenate(
            [all_vals_monthly + i * 0.01 for i in range(3)], axis=2
        )
        ts_prob_monthly = TimeSeries.from_times_and_values(
            times=self.monthly_ts.time_index,
            values=prob_values_monthly,
            columns=self.monthly_ts.columns.tolist(),
        )
        # generate probabilistic quarterly series
        all_vals_quarterly = self.quarterly_ts.all_values(copy=False)
        prob_values_quarterly = np.concatenate(
            [all_vals_quarterly + i * 0.01 for i in range(3)], axis=2
        )
        ts_prob_quarterly = TimeSeries.from_times_and_values(
            times=self.quarterly_ts.time_index,
            values=prob_values_quarterly,
            columns=self.quarterly_ts.columns.tolist(),
        )

        # to quarter start
        midas_1 = MIDAS(low_freq="QS")

        quarterly_ts_midas = midas_1.fit_transform(ts_prob_monthly)
        assert quarterly_ts_midas == ts_prob_quarterly

        inversed_quarterly_ts_midas = midas_1.inverse_transform(quarterly_ts_midas)
        assert inversed_quarterly_ts_midas == ts_prob_monthly

    def test_probabilistic_not_complete_monthly_to_quarterly(self):
        """
        Tests MIDAS on probabilistic series on example of monthly series not aligned with quarters.
        """
        # generate probabilistic monthly series
        all_vals_monthly = self.monthly_not_complete_ts.all_values(copy=False)
        prob_values_monthly = np.concatenate(
            [all_vals_monthly + i * 0.01 for i in range(3)], axis=2
        )
        ts_prob_monthly = TimeSeries.from_times_and_values(
            times=self.monthly_not_complete_ts.time_index,
            values=prob_values_monthly,
            columns=self.monthly_not_complete_ts.columns.tolist(),
        )
        # generate probabilistic quarterly series
        all_vals_quarterly = self.quarterly_not_complete_ts.all_values(copy=False)
        prob_values_quarterly = np.concatenate(
            [all_vals_quarterly + i * 0.01 for i in range(3)], axis=2
        )
        ts_prob_quarterly = TimeSeries.from_times_and_values(
            times=self.quarterly_not_complete_ts.time_index,
            values=prob_values_quarterly,
            columns=self.quarterly_not_complete_ts.columns.tolist(),
        )

        # monthly series with missing values
        midas = MIDAS(low_freq="QS", strip=False)
        quarterly_not_complete_ts_midas = midas.fit_transform(ts_prob_monthly)
        assert quarterly_not_complete_ts_midas == ts_prob_quarterly

        inversed_quarterly_not_complete_ts_midas = midas.inverse_transform(
            quarterly_not_complete_ts_midas
        )
        assert inversed_quarterly_not_complete_ts_midas == ts_prob_monthly

        # when strip=True we only get 1 one quarter with all 3 months
        midas = MIDAS(low_freq="QS", strip=True)
        quarterly_not_complete_ts_midas = midas.fit_transform(ts_prob_monthly)
        assert quarterly_not_complete_ts_midas == ts_prob_quarterly[1:2]

        inversed_quarterly_not_complete_ts_midas = midas.inverse_transform(
            quarterly_not_complete_ts_midas
        )
        assert ts_prob_monthly[1:4] == inversed_quarterly_not_complete_ts_midas

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
            values=np.array([
                [1, 10, 2, 11, 3, 12],
                [4, 13, 5, 14, 6, 15],
                [7, 16, 8, 17, 9, 18],
            ]),
            columns=[
                "values_midas_0",
                "other_midas_0",
                "values_midas_1",
                "other_midas_1",
                "values_midas_2",
                "other_midas_2",
            ],
        )

        midas_1 = MIDAS(low_freq="QS")
        multivar_quarterly_ts_midas = midas_1.fit_transform(stacked_monthly_ts)
        assert multivar_quarterly_ts_midas == expected_quarterly_ts

        multivar_inversed_quarterly_ts_midas = midas_1.inverse_transform(
            multivar_quarterly_ts_midas
        )
        assert stacked_monthly_ts == multivar_inversed_quarterly_ts_midas

    def test_probabilistic_multivariate_monthly_to_quarterly(self):
        """
        Check that probabilistic multivariate monthly to quarterly is properly transformed
        """
        monthly_ts = self.monthly_ts.stack(
            TimeSeries.from_times_and_values(
                times=self.monthly_ts.time_index,
                values=np.arange(10, 19),
                columns=["other"],
            )
        )
        # generate probabilistic monthly series
        all_vals_monthly = monthly_ts.all_values(copy=False)
        prob_values_monthly = np.concatenate(
            [all_vals_monthly + i * 0.01 for i in range(3)], axis=2
        )
        ts_prob_monthly = TimeSeries.from_times_and_values(
            times=monthly_ts.time_index,
            values=prob_values_monthly,
            columns=monthly_ts.columns.tolist(),
        )

        # component components are alternating
        quarterly_ts = TimeSeries.from_times_and_values(
            times=self.quarterly_ts.time_index,
            values=np.array([
                [1, 10, 2, 11, 3, 12],
                [4, 13, 5, 14, 6, 15],
                [7, 16, 8, 17, 9, 18],
            ]),
            columns=[
                "values_midas_0",
                "other_midas_0",
                "values_midas_1",
                "other_midas_1",
                "values_midas_2",
                "other_midas_2",
            ],
        )
        # generate probabilistic quarterly series
        all_vals_quarterly = quarterly_ts.all_values(copy=False)
        prob_values_quarterly = np.concatenate(
            [all_vals_quarterly + i * 0.01 for i in range(3)], axis=2
        )
        ts_prob_quarterly = TimeSeries.from_times_and_values(
            times=quarterly_ts.time_index,
            values=prob_values_quarterly,
            columns=quarterly_ts.columns.tolist(),
        )

        midas_1 = MIDAS(low_freq="QS")
        multivar_quarterly_ts_midas = midas_1.fit_transform(ts_prob_monthly)
        assert multivar_quarterly_ts_midas == ts_prob_quarterly

        multivar_inversed_quarterly_ts_midas = midas_1.inverse_transform(
            multivar_quarterly_ts_midas
        )
        assert ts_prob_monthly == multivar_inversed_quarterly_ts_midas

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

        # components are interleaved
        expected_quarterly_ts = TimeSeries.from_times_and_values(
            times=self.quarterly_ts.time_index,
            values=np.array([
                [1, 10, 2, 11, 3, 12],
                [4, np.nan, 5, np.nan, 6, 15],
                [7, 16, 8, 17, 9, 18],
            ]),
            columns=[
                "values_midas_0",
                "other_midas_0",
                "values_midas_1",
                "other_midas_1",
                "values_midas_2",
                "other_midas_2",
            ],
        )

        midas_1 = MIDAS(low_freq="QS")
        multivar_quarterly_ts_midas = midas_1.fit_transform(stacked_monthly_ts_missing)
        assert multivar_quarterly_ts_midas == expected_quarterly_ts

        multivar_inversed_quarterly_ts_midas = midas_1.inverse_transform(
            multivar_quarterly_ts_midas
        )
        assert stacked_monthly_ts_missing == multivar_inversed_quarterly_ts_midas

    def test_from_second_to_minute(self):
        """
        Test to see if other frequencies transforms like second to minute work as well.
        """

        second_times = pd.date_range(start="01-2020", periods=120, freq=freqs["s"])
        second_values = np.arange(1, len(second_times) + 1)
        second_ts = TimeSeries.from_times_and_values(
            times=second_times, values=second_values, columns=["values"]
        )

        minute_times = pd.date_range(start="01-2020", periods=2, freq=freqs["min"])
        minute_values = np.array([
            [i for i in range(1, 61)],
            [i for i in range(61, 121)],
        ])
        minute_ts = TimeSeries.from_times_and_values(
            times=minute_times,
            values=minute_values,
            columns=[f"values_midas_{i}" for i in range(60)],
        )

        midas = MIDAS(low_freq=freqs["min"])
        minute_ts_midas = midas.fit_transform(second_ts)
        assert minute_ts_midas == minute_ts
        second_ts_midas = midas.inverse_transform(minute_ts_midas)
        assert second_ts_midas == second_ts

    def test_error_with_invalid_freq(self):
        with pytest.raises(ValueError) as err:
            _ = MIDAS(low_freq="MEE")
        assert str(err.value).startswith("Cannot infer period alias for")

    def test_error_when_from_low_to_high(self):
        """
        Tests if the transformer raises an error when the user asks for a transform in the wrong direction.
        """
        # wrong direction : low to high freq
        midas_1 = MIDAS(low_freq=freqs["ME"])
        with pytest.raises(ValueError):
            midas_1.fit_transform(self.quarterly_ts)

        # transform to same index requested
        midas_2 = MIDAS(low_freq=freqs["QE"])
        with pytest.raises(ValueError):
            midas_2.fit_transform(self.quarterly_ts)

    def test_error_when_frequency_not_suitable_for_midas(self):
        """
        MIDAS can only be performed when the high frequency is the same and the exact multiple of the low frequency.
        For example, there are always exactly three months in a quarter, but the number of days in a month differs.
        So the monthly to quarterly transformation is possible, while the daily to monthly MIDAS transform is
        impossible.
        """
        daily_times = pd.date_range(start="01-2020", end="09-30-2020", freq="D")
        daily_values = np.arange(1, len(daily_times) + 1)
        daily_ts = TimeSeries.from_times_and_values(
            times=daily_times, values=daily_values, columns=["values"]
        )

        midas = MIDAS(low_freq=freqs["ME"])
        with pytest.raises(ValueError) as msg:
            midas.fit_transform(daily_ts)
        assert str(msg.value).startswith(
            "The frequency of the input series should be an exact multiple"
        )

    def test_inverse_transform_prediction(self):
        """
        Check that inverse-transforming the prediction of a model generate the correct time index when
        using frequency anchored either at the start or the end of the quarter.
        """
        # low frequency : QuarterStart
        monthly_ts = TimeSeries.from_times_and_values(
            times=pd.date_range(start="01-2020", periods=24, freq=freqs["ME"]),
            values=np.arange(0, 24),
            columns=["values"],
        )
        monthly_train_ts, monthly_test_ts = monthly_ts.split_after(0.75)

        model = LinearRegressionModel(lags=2)

        midas_quarterly = MIDAS(low_freq="QS")
        # shape : [6 quarters, 3 months, 1 sample]
        quarterly_train_ts = midas_quarterly.fit_transform(monthly_train_ts)
        # shape : [2 quarters, 3 months, 1 sample]
        quarterly_test_ts = midas_quarterly.transform(monthly_test_ts)

        model.fit(quarterly_train_ts)

        # 2 quarters = 6 months forecast
        pred_quarterly = model.predict(2)
        pred_monthly = midas_quarterly.inverse_transform(pred_quarterly)
        # verify prediction time index in both frequencies
        assert pred_quarterly.time_index.equals(quarterly_test_ts.time_index)
        assert pred_monthly.time_index.equals(monthly_test_ts.time_index)

        # freqs["QE"] = QuarterEnd, the 2 "hidden" months must be retrieved
        midas_quarterly = MIDAS(low_freq=freqs["QE"])
        quarterly_train_ts = midas_quarterly.fit_transform(monthly_train_ts)
        quarterly_test_ts = midas_quarterly.transform(monthly_test_ts)

        model.fit(quarterly_train_ts)

        pred_quarterly = model.predict(2)
        pred_monthly = midas_quarterly.inverse_transform(pred_quarterly)
        # verify prediction time index in both frequencies
        assert pred_quarterly.time_index.equals(quarterly_test_ts.time_index)
        assert pred_monthly.time_index.equals(monthly_test_ts.time_index)

    def test_inverse_transfrom_sliced(self):
        """Verify that MIDAS works as expected when the series is sliced prior to the inverse transform."""
        midas = MIDAS(low_freq="QS", strip=False)
        quarterly_series = midas.fit_transform(self.monthly_ts)
        inversed_quaterly = midas.inverse_transform(quarterly_series[2:])
        # the shift introduce by the slicing operation on the transformed ts is : 2 * 3 months per quarter = 6
        assert inversed_quaterly == self.monthly_ts[6:]

    def test_multiple_ts(self):
        """
        Verify that MIDAS works as expected with multiple series of different "high" frequencies (monthly and quarterly
        to yearly).
        """
        quarterly_univariate_ts = TimeSeries.from_times_and_values(
            times=pd.date_range(start="2000-01-01", periods=12, freq=freqs["QE"]),
            values=np.arange(0, 12),
        )
        quarterly_multivariate_ts = TimeSeries.from_times_and_values(
            times=pd.date_range(start="2020-01-01", periods=12, freq=freqs["QE"]),
            values=np.arange(0, 24).reshape(-1, 2),
        )

        ts_to_transform = [self.monthly_ts, quarterly_univariate_ts]
        # ==> with stripping: not enough months, first series will be empty
        midas_yearly = MIDAS(low_freq="YS", strip=True)

        list_yearly_ts = midas_yearly.fit_transform(ts_to_transform)
        assert len(list_yearly_ts) == 2
        assert len(list_yearly_ts[0]) == 0
        assert list_yearly_ts[0].freq == "YS"
        assert list_yearly_ts[0].n_components == 12

        # 4 quarters in a year
        np.testing.assert_array_almost_equal(
            list_yearly_ts[1].values(),
            quarterly_univariate_ts.values().reshape(3, 4),
        )
        # verify inverse-transform: first series will be empty, second should be identical to original
        inverse_transformed = midas_yearly.inverse_transform(list_yearly_ts)
        assert len(inverse_transformed) == 2
        assert len(inverse_transformed[0]) == 0
        assert inverse_transformed[0].freq == freqs["ME"]
        assert inverse_transformed[0].n_components == 1

        assert ts_to_transform[1:] == inverse_transformed[1:]

        # ==> without stripping: first series will be partially empty
        midas_yearly = MIDAS(low_freq="YS", strip=False)
        list_yearly_ts = midas_yearly.fit_transform(ts_to_transform)
        # 12 months in a year, original ts contains only 9 values, the missing data are nan
        np.testing.assert_array_almost_equal(
            list_yearly_ts[0].values()[:, :9], self.monthly_ts.values().T
        )
        assert np.isnan(list_yearly_ts[0].values()[:, 9:]).sum() == 3
        # 4 quarters in a year
        np.testing.assert_array_almost_equal(
            list_yearly_ts[1].values(),
            quarterly_univariate_ts.values().reshape(3, 4),
        )
        # verify inverse-transform
        assert ts_to_transform == midas_yearly.inverse_transform(list_yearly_ts)

        # replacing the univariate ts with a multivariate ts (same frequency, different start)
        ts_to_transform = [self.monthly_ts, quarterly_multivariate_ts]
        list_yearly_ts = midas_yearly.transform(ts_to_transform)
        np.testing.assert_array_almost_equal(
            list_yearly_ts[1].values(),
            quarterly_multivariate_ts.values().reshape(3, 8),
        )

        assert (
            quarterly_multivariate_ts
            == midas_yearly.inverse_transform(list_yearly_ts)[1]
        )

    def test_ts_with_static_covariates(self):
        # univarite ts, same number of static covariates as components
        global_static_covs = pd.Series(data=[0, 1], index=["static_0", "static_1"])
        monthly_with_static_covs = self.monthly_ts.with_static_covariates(
            global_static_covs
        )

        # multivariate ts, different number of static covariates than components
        components_static_covs = pd.DataFrame(
            data=[["low", 1, 9], ["high", 0, 2]],
            columns=["static_2", "static_3", "static_4"],
        )
        monthly_multivar_with_static_covs = TimeSeries.from_times_and_values(
            times=generate_index(
                start=pd.Timestamp("2000-01"), length=8, freq=freqs["ME"]
            ),
            values=np.stack([np.arange(2)] * 8),
            static_covariates=components_static_covs,
        )

        # dropping the static covariates
        midas_drop_static_covs = MIDAS(low_freq="QS", drop_static_covariates=True)
        # testing univariate (with/without static covariates), multivariate with static covariates
        for ts in [
            self.monthly_ts,
            monthly_with_static_covs,
            monthly_multivar_with_static_covs,
        ]:
            quartely_ts = midas_drop_static_covs.fit_transform(ts)
            assert quartely_ts.static_covariates is None
            inv_quartely_ts = midas_drop_static_covs.inverse_transform(quartely_ts)
            assert inv_quartely_ts.static_covariates is None

        # keeping the static covariates
        midas_with_static_covs = MIDAS(low_freq="QS", drop_static_covariates=False)
        # univariate, no static covariates
        quartely_no_static = midas_with_static_covs.fit_transform(self.monthly_ts)
        assert quartely_no_static.static_covariates is None
        inv_quartely_no_static = midas_with_static_covs.inverse_transform(
            quartely_no_static
        )
        assert inv_quartely_no_static.static_covariates is None

        # univariate, with static covariates
        expected_static_covs = pd.concat(
            [monthly_with_static_covs.static_covariates] * 3
        )
        expected_static_covs.index = [
            "values_midas_0",
            "values_midas_1",
            "values_midas_2",
        ]
        quartely_univ_dropped_static = midas_with_static_covs.fit_transform(
            monthly_with_static_covs
        )
        assert quartely_univ_dropped_static.static_covariates.equals(
            expected_static_covs
        )

        inv_quartely_univ_dropped_static = midas_with_static_covs.inverse_transform(
            quartely_univ_dropped_static
        )
        assert inv_quartely_univ_dropped_static.static_covariates.equals(
            monthly_with_static_covs.static_covariates
        )

        # testing multivariate, with static covariates
        expected_static_covs = pd.concat(
            [monthly_multivar_with_static_covs.static_covariates] * 3
        )
        expected_static_covs.index = [
            "0_midas_0",
            "1_midas_0",
            "0_midas_1",
            "1_midas_1",
            "0_midas_2",
            "1_midas_2",
        ]
        quartely_multiv_dropped_static = midas_with_static_covs.fit_transform(
            monthly_multivar_with_static_covs
        )
        assert quartely_multiv_dropped_static.static_covariates.equals(
            expected_static_covs
        )

        inv_quartely_multiv_dropped_static = midas_with_static_covs.inverse_transform(
            quartely_multiv_dropped_static
        )
        assert inv_quartely_multiv_dropped_static.static_covariates.equals(
            monthly_multivar_with_static_covs.static_covariates
        )

    def test_aligned_low_freq_anchor(self):
        """
        Check that values are properly extracted when the beginning of the series is aligned with the low frequency
        anchor value (day of the week).
        """
        ts = linear_timeseries(
            start=pd.Timestamp("2000-01-01"),
            freq="D",
            start_value=1,
            end_value=22,
            length=22,
        )
        assert ts.time_index[0].day_name() == "Saturday"

        midas_sat = MIDAS(low_freq="W-SAT", strip=False)
        ts_saturday_anchor = midas_sat.fit_transform(ts)
        # the resampled series start at the same timestamp because `strip=False`
        assert ts_saturday_anchor.start_time() == ts.start_time()
        assert ts_saturday_anchor.end_time() == ts.end_time()

        # padding in the first row
        first_row_exp = np.empty(6)
        first_row_exp.fill(np.nan)
        first_row_exp = np.concatenate([first_row_exp, ts[0].values().flatten()])
        np.testing.assert_array_equal(
            ts_saturday_anchor[0].values().flatten(), first_row_exp
        )

        # last row should be completely filled
        np.testing.assert_array_equal(
            ts_saturday_anchor[-1].values().flatten(), ts[-7:].values().flatten()
        )

        # inverse transform
        ts_inv_saturday_anchor = midas_sat.inverse_transform(ts_saturday_anchor)
        np.testing.assert_array_equal(ts.values(), ts_inv_saturday_anchor.values())
        assert ts.time_index.equals(ts_inv_saturday_anchor.time_index)

    def test_shifted_low_freq_anchor(self):
        """
        Check that values are properly extracted when the beginning of the series is not aligned with the low
        frequency anchor value (day of the week).
        """
        ts = linear_timeseries(
            start=pd.Timestamp("2000-01-01"),
            freq="D",
            start_value=1,
            end_value=22,
            length=22,
        )
        assert ts.time_index[2].day_name() == "Monday"

        midas_mon = MIDAS(low_freq="W-MON", strip=False)
        ts_monday_anchor = midas_mon.fit_transform(ts)
        # the resampled series start at the previous anchor point (incomplete)
        assert ts_monday_anchor.time_index[0] == pd.Timestamp("2000-01-03")
        assert ts_monday_anchor.freq == "W-MON"
        assert len(ts_monday_anchor) == 4
        # incomplete low frequency period
        assert (
            ts_monday_anchor[0].values().flatten()[-3:] == ts[:3].values().flatten()
        ).all()
        # complete low frequency period
        assert (
            ts_monday_anchor[1].values().flatten() == ts[3:10].values().flatten()
        ).all()

        # reverse transform
        ts_inv_monday_anchor = midas_mon.inverse_transform(ts_monday_anchor)
        np.testing.assert_array_almost_equal(ts.values(), ts_inv_monday_anchor.values())
        assert ts.time_index.equals(ts_inv_monday_anchor.time_index)

        # with strip, it skips first and last week
        midas_mon = MIDAS(low_freq="W-MON", strip=True)
        ts_monday_anchor = midas_mon.fit_transform(ts)
        assert ts_monday_anchor.time_index[0] == pd.Timestamp("2000-01-10")
        assert ts_monday_anchor.freq == "W-MON"
        assert len(ts_monday_anchor) == 2
        # first complete low frequency period
        assert (
            ts_monday_anchor[0].values().flatten() == ts[3:10].values().flatten()
        ).all()
