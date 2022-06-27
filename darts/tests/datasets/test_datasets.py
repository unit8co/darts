import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger
from darts.tests.base_test_class import DartsBaseTestClass
from darts.utils.timeseries_generation import gaussian_timeseries

logger = get_logger(__name__)

try:
    from darts.utils.data import (  # noqa: F401
        DualCovariatesInferenceDataset,
        DualCovariatesSequentialDataset,
        DualCovariatesShiftedDataset,
        FutureCovariatesInferenceDataset,
        FutureCovariatesSequentialDataset,
        FutureCovariatesShiftedDataset,
        HorizonBasedDataset,
        MixedCovariatesInferenceDataset,
        MixedCovariatesSequentialDataset,
        MixedCovariatesShiftedDataset,
        PastCovariatesInferenceDataset,
        PastCovariatesSequentialDataset,
        PastCovariatesShiftedDataset,
        SplitCovariatesInferenceDataset,
        SplitCovariatesSequentialDataset,
        SplitCovariatesShiftedDataset,
    )

    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("Torch not installed - dataset tests will be skipped.")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:

    class DatasetTestCase(DartsBaseTestClass):
        target1 = gaussian_timeseries(length=100).with_static_covariates(
            pd.Series([0, 1], index=["st1", "st2"])
        )
        target2 = gaussian_timeseries(length=150).with_static_covariates(
            pd.Series([2, 3], index=["st1", "st2"])
        )
        cov_st1 = target1.static_covariates.values
        cov_st2 = target2.static_covariates.values
        cov_st2_df = pd.Series([2, 3], index=["st1", "st2"])
        vals1, vals2 = target1.values(), target2.values()
        cov1, cov2 = gaussian_timeseries(length=100), gaussian_timeseries(length=150)

        def _assert_eq(self, lefts: tuple, rights: tuple):
            for left, right in zip(lefts, rights):
                left = left.values() if isinstance(left, TimeSeries) else left
                right = right.values() if isinstance(right, TimeSeries) else right
                assert type(left) == type(right)
                assert (
                    isinstance(
                        left, (TimeSeries, pd.Series, pd.DataFrame, np.ndarray, list)
                    )
                    or left is None
                )
                if isinstance(left, (pd.Series, pd.DataFrame)):
                    assert left.equals(right)
                elif isinstance(left, np.ndarray):
                    assert np.array_equal(left, right)
                elif isinstance(left, (list, TimeSeries)):
                    assert left == right
                else:
                    assert right is None

        def test_past_covariates_inference_dataset(self):
            # one target series
            ds = PastCovariatesInferenceDataset(
                target_series=self.target1, input_chunk_length=len(self.target1)
            )
            np.testing.assert_almost_equal(ds[0][0], self.vals1)
            self._assert_eq(ds[0][1:], (None, None, self.cov_st1, self.target1))

            # two target series
            ds = PastCovariatesInferenceDataset(
                target_series=[self.target1, self.target2],
                input_chunk_length=max(len(self.target1), len(self.target2)),
            )
            np.testing.assert_almost_equal(ds[1][0], self.vals2)
            self._assert_eq(ds[1][1:], (None, None, self.cov_st2, self.target2))

            # fail if covariates do not have same size
            with self.assertRaises(ValueError):
                ds = PastCovariatesInferenceDataset(
                    target_series=[self.target1, self.target2], covariates=[self.cov1]
                )

            # with covariates
            ds = PastCovariatesInferenceDataset(
                target_series=[self.target1, self.target2],
                covariates=[self.cov1, self.cov2],
                input_chunk_length=max(len(self.target1), len(self.target2)),
            )
            np.testing.assert_almost_equal(ds[1][0], self.vals2)
            np.testing.assert_almost_equal(ds[1][1], self.cov2.values())
            self._assert_eq(
                ds[1][2:], (None, self.cov_st2, self.target2)
            )  # no "future past" covariate here

            # more complex case with future past covariates:
            times1 = pd.date_range(start="20100101", end="20100701", freq="D")
            times2 = pd.date_range(
                start="20100101", end="20100820", freq="D"
            )  # 50 days longer than times1

            target = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            short_cov = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            )
            long_cov = TimeSeries.from_times_and_values(
                times2, np.random.randn(len(times2))
            )

            ds = PastCovariatesInferenceDataset(
                target_series=target,
                covariates=short_cov,
                input_chunk_length=10,
                output_chunk_length=10,
                n=30,
            )

            # should fail if covariates are too short
            with self.assertRaises(ValueError):
                _ = ds[0]

            # Should return correct values when covariates is long enough
            ds = PastCovariatesInferenceDataset(
                target_series=target,
                covariates=long_cov,
                input_chunk_length=10,
                output_chunk_length=10,
                n=30,
            )

            np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
            np.testing.assert_almost_equal(ds[0][1], long_cov.values()[-60:-50])
            np.testing.assert_almost_equal(ds[0][2], long_cov.values()[-50:-30])
            np.testing.assert_almost_equal(ds[0][3], self.cov_st2)
            self.assertEqual(ds[0][4], target)

            # Should also work for integer-indexed series
            target = TimeSeries.from_times_and_values(
                pd.RangeIndex(start=10, stop=50, step=1), np.random.randn(40)
            ).with_static_covariates(self.cov_st2_df)
            covariate = TimeSeries.from_times_and_values(
                pd.RangeIndex(start=20, stop=80, step=1), np.random.randn(60)
            )

            ds = PastCovariatesInferenceDataset(
                target_series=target,
                covariates=covariate,
                input_chunk_length=10,
                output_chunk_length=10,
                n=20,
            )

            np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
            np.testing.assert_almost_equal(ds[0][1], covariate.values()[20:30])
            np.testing.assert_almost_equal(ds[0][2], covariate.values()[30:40])
            np.testing.assert_almost_equal(ds[0][3], self.cov_st2)
            self.assertEqual(ds[0][4], target)

        def test_future_covariates_inference_dataset(self):
            # one target series
            ds = FutureCovariatesInferenceDataset(
                target_series=self.target1, input_chunk_length=len(self.target1)
            )
            np.testing.assert_almost_equal(ds[0][0], self.vals1)
            self._assert_eq(ds[0][1:], (None, self.cov_st1, self.target1))

            # two target series
            ds = FutureCovariatesInferenceDataset(
                target_series=[self.target1, self.target2],
                input_chunk_length=max(len(self.target1), len(self.target2)),
            )
            np.testing.assert_almost_equal(ds[1][0], self.vals2)
            self._assert_eq(ds[1][1:], (None, self.cov_st2, self.target2))

            # fail if covariates do not have same size
            with self.assertRaises(ValueError):
                ds = FutureCovariatesInferenceDataset(
                    target_series=[self.target1, self.target2], covariates=[self.cov1]
                )

            # With future past covariates:
            times1 = pd.date_range(start="20100101", end="20100701", freq="D")
            times2 = pd.date_range(
                start="20100101", end="20100820", freq="D"
            )  # 50 days longer than times1

            target = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            short_cov = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            )
            long_cov = TimeSeries.from_times_and_values(
                times2, np.random.randn(len(times2))
            )

            ds = FutureCovariatesInferenceDataset(
                target_series=target, covariates=short_cov, input_chunk_length=10, n=30
            )

            # should fail if covariates are too short
            with self.assertRaises(ValueError):
                _ = ds[0]

            # Should return correct values when covariates is long enough
            ds = FutureCovariatesInferenceDataset(
                target_series=target, covariates=long_cov, input_chunk_length=10, n=30
            )

            np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
            np.testing.assert_almost_equal(ds[0][1], long_cov.values()[-50:-20])
            np.testing.assert_almost_equal(ds[0][2], self.cov_st2)
            self.assertEqual(ds[0][3], target)

            # Should also work for integer-indexed series
            target = TimeSeries.from_times_and_values(
                pd.RangeIndex(start=10, stop=50, step=1), np.random.randn(40)
            ).with_static_covariates(self.cov_st2_df)
            covariate = TimeSeries.from_times_and_values(
                pd.RangeIndex(start=20, stop=80, step=1), np.random.randn(60)
            )

            ds = FutureCovariatesInferenceDataset(
                target_series=target, covariates=covariate, input_chunk_length=10, n=20
            )

            np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
            np.testing.assert_almost_equal(ds[0][1], covariate.values()[30:50])
            np.testing.assert_almost_equal(ds[0][2], self.cov_st2)
            self.assertEqual(ds[0][3], target)

        def test_dual_covariates_inference_dataset(self):
            # one target series
            ds = DualCovariatesInferenceDataset(
                target_series=self.target1, input_chunk_length=len(self.target1)
            )
            np.testing.assert_almost_equal(ds[0][0], self.vals1)
            self._assert_eq(ds[0][1:], (None, None, self.cov_st1, self.target1))

            # two target series
            ds = DualCovariatesInferenceDataset(
                target_series=[self.target1, self.target2],
                input_chunk_length=max(len(self.target1), len(self.target2)),
            )
            np.testing.assert_almost_equal(ds[1][0], self.vals2)
            self._assert_eq(ds[1][1:], (None, None, self.cov_st2, self.target2))

            # fail if covariates do not have same size
            with self.assertRaises(ValueError):
                ds = DualCovariatesInferenceDataset(
                    target_series=[self.target1, self.target2], covariates=[self.cov1]
                )

            # With future past covariates:
            times1 = pd.date_range(start="20100101", end="20100701", freq="D")
            times2 = pd.date_range(
                start="20100101", end="20100820", freq="D"
            )  # 50 days longer than times1

            target = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            short_cov = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            )
            long_cov = TimeSeries.from_times_and_values(
                times2, np.random.randn(len(times2))
            )

            ds = DualCovariatesInferenceDataset(
                target_series=target,
                covariates=short_cov,
                input_chunk_length=10,
                output_chunk_length=10,
                n=30,
            )

            # should fail if covariates are too short
            with self.assertRaises(ValueError):
                _ = ds[0]

            # Should return correct values when covariates is long enough
            ds = DualCovariatesInferenceDataset(
                target_series=target,
                covariates=long_cov,
                input_chunk_length=10,
                output_chunk_length=10,
                n=30,
            )

            np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
            np.testing.assert_almost_equal(ds[0][1], long_cov.values()[-60:-50])
            np.testing.assert_almost_equal(ds[0][2], long_cov.values()[-50:-20])
            np.testing.assert_almost_equal(ds[0][3], self.cov_st2)
            self.assertEqual(ds[0][4], target)

            # Should also work for integer-indexed series
            target = TimeSeries.from_times_and_values(
                pd.RangeIndex(start=10, stop=50, step=1), np.random.randn(40)
            ).with_static_covariates(self.cov_st2_df)
            covariate = TimeSeries.from_times_and_values(
                pd.RangeIndex(start=20, stop=80, step=1), np.random.randn(60)
            )

            ds = DualCovariatesInferenceDataset(
                target_series=target,
                covariates=covariate,
                input_chunk_length=10,
                output_chunk_length=10,
                n=20,
            )

            np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
            np.testing.assert_almost_equal(ds[0][1], covariate.values()[20:30])
            np.testing.assert_almost_equal(ds[0][2], covariate.values()[30:50])
            np.testing.assert_almost_equal(ds[0][3], self.cov_st2)
            self.assertEqual(ds[0][4], target)

        def test_mixed_covariates_inference_dataset(self):
            # With future past covariates:
            times1 = pd.date_range(start="20100101", end="20100701", freq="D")
            times2 = pd.date_range(
                start="20100201", end="20100820", freq="D"
            )  # ends 50 days after times1

            target = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            past_cov = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            )
            long_past_cov = TimeSeries.from_times_and_values(
                times2, np.random.randn(len(times2))
            )
            future_cov = TimeSeries.from_times_and_values(
                times2, np.random.randn(len(times2))
            )

            ds = MixedCovariatesInferenceDataset(
                target_series=target,
                past_covariates=past_cov,
                future_covariates=past_cov,
                input_chunk_length=10,
                output_chunk_length=10,
                n=30,
            )

            # should fail if future covariates are too short
            with self.assertRaises(ValueError):
                _ = ds[0]

            # Should return correct values when covariates is long enough
            ds = MixedCovariatesInferenceDataset(
                target_series=target,
                past_covariates=long_past_cov,
                future_covariates=future_cov,
                input_chunk_length=10,
                output_chunk_length=10,
                n=30,
            )

            # It should contain:
            # past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates
            np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
            np.testing.assert_almost_equal(ds[0][1], long_past_cov.values()[-60:-50])
            np.testing.assert_almost_equal(ds[0][2], future_cov.values()[-60:-50])
            np.testing.assert_almost_equal(ds[0][3], future_cov.values()[-50:-20])
            np.testing.assert_almost_equal(ds[0][4], long_past_cov.values()[-50:-30])
            np.testing.assert_almost_equal(ds[0][5], self.cov_st2)
            self.assertEqual(ds[0][6], target)

            # Should also work for integer-indexed series
            target = TimeSeries.from_times_and_values(
                pd.RangeIndex(start=10, stop=50, step=1), np.random.randn(40)
            ).with_static_covariates(self.cov_st2_df)
            past_cov = TimeSeries.from_times_and_values(
                pd.RangeIndex(start=20, stop=80, step=1), np.random.randn(60)
            )
            future_cov = TimeSeries.from_times_and_values(
                pd.RangeIndex(start=30, stop=100, step=1), np.random.randn(70)
            )

            ds = MixedCovariatesInferenceDataset(
                target_series=target,
                past_covariates=past_cov,
                future_covariates=future_cov,
                input_chunk_length=10,
                output_chunk_length=10,
                n=20,
            )

            np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
            np.testing.assert_almost_equal(ds[0][1], past_cov.values()[20:30])
            np.testing.assert_almost_equal(ds[0][2], future_cov.values()[10:20])
            np.testing.assert_almost_equal(ds[0][3], future_cov.values()[20:40])
            np.testing.assert_almost_equal(ds[0][4], past_cov.values()[30:40])
            np.testing.assert_almost_equal(ds[0][5], self.cov_st2)
            self.assertEqual(ds[0][6], target)

        def test_split_covariates_inference_dataset(self):
            # With future past covariates:
            times1 = pd.date_range(start="20100101", end="20100701", freq="D")
            times2 = pd.date_range(
                start="20100201", end="20100820", freq="D"
            )  # ends 50 days after times1

            target = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            past_cov = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            )
            long_past_cov = TimeSeries.from_times_and_values(
                times2, np.random.randn(len(times2))
            )
            future_cov = TimeSeries.from_times_and_values(
                times2, np.random.randn(len(times2))
            )

            ds = SplitCovariatesInferenceDataset(
                target_series=target,
                past_covariates=past_cov,
                future_covariates=past_cov,
                input_chunk_length=10,
                output_chunk_length=10,
                n=30,
            )

            # should fail if future covariates are too short
            with self.assertRaises(ValueError):
                _ = ds[0]

            # Should return correct values when covariates is long enough
            ds = SplitCovariatesInferenceDataset(
                target_series=target,
                past_covariates=long_past_cov,
                future_covariates=future_cov,
                input_chunk_length=10,
                output_chunk_length=10,
                n=30,
            )

            # It should contain:
            # past_target, past_covariates, future_covariates, future_past_covariates
            np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
            np.testing.assert_almost_equal(ds[0][1], long_past_cov.values()[-60:-50])
            np.testing.assert_almost_equal(ds[0][2], future_cov.values()[-50:-20])
            np.testing.assert_almost_equal(ds[0][3], long_past_cov.values()[-50:-30])
            np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
            self.assertEqual(ds[0][5], target)

            # Should also work for integer-indexed series
            target = TimeSeries.from_times_and_values(
                pd.RangeIndex(start=10, stop=50, step=1), np.random.randn(40)
            ).with_static_covariates(self.cov_st2_df)
            past_cov = TimeSeries.from_times_and_values(
                pd.RangeIndex(start=20, stop=80, step=1), np.random.randn(60)
            )
            future_cov = TimeSeries.from_times_and_values(
                pd.RangeIndex(start=30, stop=100, step=1), np.random.randn(70)
            )

            ds = SplitCovariatesInferenceDataset(
                target_series=target,
                past_covariates=past_cov,
                future_covariates=future_cov,
                input_chunk_length=10,
                output_chunk_length=10,
                n=20,
            )

            np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
            np.testing.assert_almost_equal(ds[0][1], past_cov.values()[20:30])
            np.testing.assert_almost_equal(ds[0][2], future_cov.values()[20:40])
            np.testing.assert_almost_equal(ds[0][3], past_cov.values()[30:40])
            np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
            self.assertEqual(ds[0][5], target)

        def test_past_covariates_sequential_dataset(self):
            # one target series
            ds = PastCovariatesSequentialDataset(
                target_series=self.target1,
                input_chunk_length=10,
                output_chunk_length=10,
            )
            self.assertEqual(len(ds), 81)
            self._assert_eq(
                ds[5], (self.target1[75:85], None, self.cov_st1, self.target1[85:95])
            )

            # two target series
            ds = PastCovariatesSequentialDataset(
                target_series=[self.target1, self.target2],
                input_chunk_length=10,
                output_chunk_length=10,
            )
            self.assertEqual(len(ds), 262)
            self._assert_eq(
                ds[5], (self.target1[75:85], None, self.cov_st1, self.target1[85:95])
            )
            self._assert_eq(
                ds[136],
                (self.target2[125:135], None, self.cov_st2, self.target2[135:145]),
            )

            # two target series with custom max_nr_samples
            ds = PastCovariatesSequentialDataset(
                target_series=[self.target1, self.target2],
                input_chunk_length=10,
                output_chunk_length=10,
                max_samples_per_ts=50,
            )
            self.assertEqual(len(ds), 100)
            self._assert_eq(
                ds[5], (self.target1[75:85], None, self.cov_st1, self.target1[85:95])
            )
            self._assert_eq(
                ds[55],
                (self.target2[125:135], None, self.cov_st2, self.target2[135:145]),
            )

            # two targets and one covariate
            with self.assertRaises(ValueError):
                ds = PastCovariatesSequentialDataset(
                    target_series=[self.target1, self.target2], covariates=[self.cov1]
                )

            # two targets and two covariates
            ds = PastCovariatesSequentialDataset(
                target_series=[self.target1, self.target2],
                covariates=[self.cov1, self.cov2],
                input_chunk_length=10,
                output_chunk_length=10,
            )
            self._assert_eq(
                ds[5],
                (
                    self.target1[75:85],
                    self.cov1[75:85],
                    self.cov_st1,
                    self.target1[85:95],
                ),
            )
            self._assert_eq(
                ds[136],
                (
                    self.target2[125:135],
                    self.cov2[125:135],
                    self.cov_st2,
                    self.target2[135:145],
                ),
            )

            # should fail if covariates do not have the required time span, even though covariates are longer
            times1 = pd.date_range(start="20100101", end="20110101", freq="D")
            times2 = pd.date_range(start="20120101", end="20150101", freq="D")
            target = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
            ds = PastCovariatesSequentialDataset(
                target_series=target,
                covariates=cov,
                input_chunk_length=10,
                output_chunk_length=10,
            )
            with self.assertRaises(ValueError):
                _ = ds[5]

            # the same should fail when series are integer-indexed
            times1 = pd.RangeIndex(start=0, stop=100, step=1)
            times2 = pd.RangeIndex(start=200, stop=400, step=1)
            target = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
            ds = PastCovariatesSequentialDataset(
                target_series=target,
                covariates=cov,
                input_chunk_length=10,
                output_chunk_length=10,
            )
            with self.assertRaises(ValueError):
                _ = ds[5]

            # we should get the correct covariate slice even when target and covariates are not aligned
            times1 = pd.date_range(start="20100101", end="20110101", freq="D")
            times2 = pd.date_range(start="20090101", end="20110106", freq="D")
            target = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
            ds = PastCovariatesSequentialDataset(
                target_series=target,
                covariates=cov,
                input_chunk_length=10,
                output_chunk_length=10,
            )

            np.testing.assert_almost_equal(ds[0][1], cov.values()[-25:-15])
            np.testing.assert_almost_equal(ds[5][1], cov.values()[-30:-20])

            # This should also be the case when series are integer indexed
            times1 = pd.RangeIndex(start=100, stop=200, step=1)
            times2 = pd.RangeIndex(start=50, stop=250, step=1)
            target = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
            ds = PastCovariatesSequentialDataset(
                target_series=target,
                covariates=cov,
                input_chunk_length=10,
                output_chunk_length=10,
            )

            np.testing.assert_almost_equal(ds[0][1], cov.values()[-70:-60])
            np.testing.assert_almost_equal(ds[5][1], cov.values()[-75:-65])

        def test_future_covariates_sequential_dataset(self):
            # one target series
            ds = FutureCovariatesSequentialDataset(
                target_series=self.target1,
                input_chunk_length=10,
                output_chunk_length=10,
            )
            self.assertEqual(len(ds), 81)
            self._assert_eq(
                ds[5], (self.target1[75:85], None, self.cov_st1, self.target1[85:95])
            )

            # two target series
            ds = FutureCovariatesSequentialDataset(
                target_series=[self.target1, self.target2],
                input_chunk_length=10,
                output_chunk_length=10,
            )
            self.assertEqual(len(ds), 262)
            self._assert_eq(
                ds[5], (self.target1[75:85], None, self.cov_st1, self.target1[85:95])
            )
            self._assert_eq(
                ds[136],
                (self.target2[125:135], None, self.cov_st2, self.target2[135:145]),
            )

            # two target series with custom max_nr_samples
            ds = FutureCovariatesSequentialDataset(
                target_series=[self.target1, self.target2],
                input_chunk_length=10,
                output_chunk_length=10,
                max_samples_per_ts=50,
            )
            self.assertEqual(len(ds), 100)
            self._assert_eq(
                ds[5], (self.target1[75:85], None, self.cov_st1, self.target1[85:95])
            )
            self._assert_eq(
                ds[55],
                (self.target2[125:135], None, self.cov_st2, self.target2[135:145]),
            )

            # two targets and one covariate
            with self.assertRaises(ValueError):
                ds = FutureCovariatesSequentialDataset(
                    target_series=[self.target1, self.target2], covariates=[self.cov1]
                )

            # two targets and two covariates; covariates not aligned, must contain correct values
            target1 = TimeSeries.from_values(
                np.random.randn(100)
            ).with_static_covariates(self.cov_st2_df)
            target2 = TimeSeries.from_values(
                np.random.randn(50)
            ).with_static_covariates(self.cov_st2_df)
            cov1 = TimeSeries.from_values(np.random.randn(120))
            cov2 = TimeSeries.from_values(np.random.randn(80))

            ds = FutureCovariatesSequentialDataset(
                target_series=[target1, target2],
                covariates=[cov1, cov2],
                input_chunk_length=10,
                output_chunk_length=10,
            )

            np.testing.assert_almost_equal(ds[0][0], target1.values()[-20:-10])
            np.testing.assert_almost_equal(ds[0][1], cov1.values()[-30:-20])
            np.testing.assert_almost_equal(ds[0][2], self.cov_st2)
            np.testing.assert_almost_equal(ds[0][3], target1.values()[-10:])

            np.testing.assert_almost_equal(ds[101][0], target2.values()[-40:-30])
            np.testing.assert_almost_equal(ds[101][1], cov2.values()[-60:-50])
            np.testing.assert_almost_equal(ds[101][2], self.cov_st2)
            np.testing.assert_almost_equal(ds[101][3], target2.values()[-30:-20])

            # Should also contain correct values when time-indexed with covariates not aligned
            times1 = pd.date_range(start="20090201", end="20090220", freq="D")
            times2 = pd.date_range(start="20090201", end="20090222", freq="D")
            target1 = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            cov1 = TimeSeries.from_times_and_values(
                times2, np.random.randn(len(times2))
            )

            ds = FutureCovariatesSequentialDataset(
                target_series=[target1],
                covariates=[cov1],
                input_chunk_length=2,
                output_chunk_length=2,
            )

            np.testing.assert_almost_equal(ds[0][0], target1.values()[-4:-2])
            np.testing.assert_almost_equal(ds[0][1], cov1.values()[-4:-2])
            np.testing.assert_almost_equal(ds[0][2], self.cov_st2)
            np.testing.assert_almost_equal(ds[0][3], target1.values()[-2:])

            # Should fail if covariates are not long enough
            target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
                self.cov_st2_df
            )
            cov1 = TimeSeries.from_values(np.random.randn(7))

            ds = FutureCovariatesSequentialDataset(
                target_series=[target1],
                covariates=[cov1],
                input_chunk_length=2,
                output_chunk_length=2,
            )

            with self.assertRaises(ValueError):
                _ = ds[0]

        def test_dual_covariates_sequential_dataset(self):
            # Must contain (past_target, historic_future_covariates, future_covariates, future_target)

            # one target series
            ds = DualCovariatesSequentialDataset(
                target_series=self.target1,
                input_chunk_length=10,
                output_chunk_length=10,
            )
            self.assertEqual(len(ds), 81)
            self._assert_eq(
                ds[5],
                (self.target1[75:85], None, None, self.cov_st1, self.target1[85:95]),
            )

            # two target series
            ds = DualCovariatesSequentialDataset(
                target_series=[self.target1, self.target2],
                input_chunk_length=10,
                output_chunk_length=10,
            )
            self.assertEqual(len(ds), 262)
            self._assert_eq(
                ds[5],
                (self.target1[75:85], None, None, self.cov_st1, self.target1[85:95]),
            )
            self._assert_eq(
                ds[136],
                (
                    self.target2[125:135],
                    None,
                    None,
                    self.cov_st2,
                    self.target2[135:145],
                ),
            )

            # two target series with custom max_nr_samples
            ds = DualCovariatesSequentialDataset(
                target_series=[self.target1, self.target2],
                input_chunk_length=10,
                output_chunk_length=10,
                max_samples_per_ts=50,
            )
            self.assertEqual(len(ds), 100)
            self._assert_eq(
                ds[5],
                (self.target1[75:85], None, None, self.cov_st1, self.target1[85:95]),
            )
            self._assert_eq(
                ds[55],
                (
                    self.target2[125:135],
                    None,
                    None,
                    self.cov_st2,
                    self.target2[135:145],
                ),
            )

            # two targets and one covariate
            with self.assertRaises(ValueError):
                ds = DualCovariatesSequentialDataset(
                    target_series=[self.target1, self.target2], covariates=[self.cov1]
                )

            # two targets and two covariates; covariates not aligned, must contain correct values
            target1 = TimeSeries.from_values(
                np.random.randn(100)
            ).with_static_covariates(self.cov_st2_df)
            target2 = TimeSeries.from_values(
                np.random.randn(50)
            ).with_static_covariates(self.cov_st2_df)
            cov1 = TimeSeries.from_values(np.random.randn(120))
            cov2 = TimeSeries.from_values(np.random.randn(80))

            ds = DualCovariatesSequentialDataset(
                target_series=[target1, target2],
                covariates=[cov1, cov2],
                input_chunk_length=10,
                output_chunk_length=10,
            )

            np.testing.assert_almost_equal(ds[0][0], target1.values()[-20:-10])
            np.testing.assert_almost_equal(ds[0][1], cov1.values()[-40:-30])
            np.testing.assert_almost_equal(ds[0][2], cov1.values()[-30:-20])
            np.testing.assert_almost_equal(ds[0][3], self.cov_st2)
            np.testing.assert_almost_equal(ds[0][4], target1.values()[-10:])

            np.testing.assert_almost_equal(ds[101][0], target2.values()[-40:-30])
            np.testing.assert_almost_equal(ds[101][1], cov2.values()[-70:-60])
            np.testing.assert_almost_equal(ds[101][2], cov2.values()[-60:-50])
            np.testing.assert_almost_equal(ds[101][3], self.cov_st2)
            np.testing.assert_almost_equal(ds[101][4], target2.values()[-30:-20])

            # Should also contain correct values when time-indexed with covariates not aligned
            times1 = pd.date_range(start="20090201", end="20090220", freq="D")
            times2 = pd.date_range(start="20090201", end="20090222", freq="D")
            target1 = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            cov1 = TimeSeries.from_times_and_values(
                times2, np.random.randn(len(times2))
            )

            ds = DualCovariatesSequentialDataset(
                target_series=[target1],
                covariates=[cov1],
                input_chunk_length=2,
                output_chunk_length=2,
            )

            np.testing.assert_almost_equal(ds[0][0], target1.values()[-4:-2])
            np.testing.assert_almost_equal(ds[0][1], cov1.values()[-6:-4])
            np.testing.assert_almost_equal(ds[0][2], cov1.values()[-4:-2])
            np.testing.assert_almost_equal(ds[0][3], self.cov_st2)
            np.testing.assert_almost_equal(ds[0][4], target1.values()[-2:])

            # Should fail if covariates are not long enough
            target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
                self.cov_st2_df
            )
            cov1 = TimeSeries.from_values(np.random.randn(7))

            ds = DualCovariatesSequentialDataset(
                target_series=[target1],
                covariates=[cov1],
                input_chunk_length=2,
                output_chunk_length=2,
            )

            with self.assertRaises(ValueError):
                _ = ds[0]

        def test_past_covariates_shifted_dataset(self):
            # one target series
            ds = PastCovariatesShiftedDataset(
                target_series=self.target1, length=10, shift=5
            )
            self.assertEqual(len(ds), 86)
            self._assert_eq(
                ds[5], (self.target1[80:90], None, self.cov_st1, self.target1[85:95])
            )

            # two target series
            ds = PastCovariatesShiftedDataset(
                target_series=[self.target1, self.target2], length=10, shift=5
            )
            self.assertEqual(len(ds), 272)
            self._assert_eq(
                ds[5], (self.target1[80:90], None, self.cov_st1, self.target1[85:95])
            )
            self._assert_eq(
                ds[141],
                (self.target2[130:140], None, self.cov_st2, self.target2[135:145]),
            )

            # two target series with custom max_nr_samples
            ds = PastCovariatesShiftedDataset(
                target_series=[self.target1, self.target2],
                length=10,
                shift=5,
                max_samples_per_ts=50,
            )
            self.assertEqual(len(ds), 100)
            self._assert_eq(
                ds[5], (self.target1[80:90], None, self.cov_st1, self.target1[85:95])
            )
            self._assert_eq(
                ds[55],
                (self.target2[130:140], None, self.cov_st2, self.target2[135:145]),
            )

            # two targets and one covariate
            with self.assertRaises(ValueError):
                ds = PastCovariatesShiftedDataset(
                    target_series=[self.target1, self.target2], covariates=[self.cov1]
                )

            # two targets and two covariates
            ds = PastCovariatesShiftedDataset(
                target_series=[self.target1, self.target2],
                covariates=[self.cov1, self.cov2],
                length=10,
                shift=5,
            )
            self._assert_eq(
                ds[5],
                (
                    self.target1[80:90],
                    self.cov1[80:90],
                    self.cov_st1,
                    self.target1[85:95],
                ),
            )
            self._assert_eq(
                ds[141],
                (
                    self.target2[130:140],
                    self.cov2[130:140],
                    self.cov_st2,
                    self.target2[135:145],
                ),
            )

            # Should contain correct values even when covariates are not aligned
            target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
                self.cov_st2_df
            )
            cov1 = TimeSeries.from_values(np.random.randn(10))
            ds = PastCovariatesShiftedDataset(
                target_series=[target1], covariates=[cov1], length=3, shift=2
            )
            np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
            np.testing.assert_almost_equal(ds[0][1], cov1.values()[-7:-4])
            np.testing.assert_almost_equal(ds[0][2], self.cov_st2)
            np.testing.assert_almost_equal(ds[0][3], target1.values()[-3:])

            # Should also contain correct values when time-indexed with covariates not aligned
            times1 = pd.date_range(start="20090201", end="20090220", freq="D")
            times2 = pd.date_range(start="20090201", end="20090222", freq="D")
            target1 = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            cov1 = TimeSeries.from_times_and_values(
                times2, np.random.randn(len(times2))
            )
            ds = PastCovariatesShiftedDataset(
                target_series=[target1], covariates=[cov1], length=3, shift=2
            )
            np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
            np.testing.assert_almost_equal(ds[0][1], cov1.values()[-7:-4])
            np.testing.assert_almost_equal(ds[0][2], self.cov_st2)
            np.testing.assert_almost_equal(ds[0][3], target1.values()[-3:])

            # Should fail if covariates are too short
            target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
                self.cov_st2_df
            )
            cov1 = TimeSeries.from_values(np.random.randn(5))
            ds = PastCovariatesShiftedDataset(
                target_series=[target1], covariates=[cov1], length=3, shift=2
            )
            with self.assertRaises(ValueError):
                _ = ds[0]

        def test_future_covariates_shifted_dataset(self):
            # one target series
            ds = FutureCovariatesShiftedDataset(
                target_series=self.target1, length=10, shift=5
            )
            self.assertEqual(len(ds), 86)
            self._assert_eq(
                ds[5], (self.target1[80:90], None, self.cov_st1, self.target1[85:95])
            )

            # two target series
            ds = FutureCovariatesShiftedDataset(
                target_series=[self.target1, self.target2], length=10, shift=5
            )
            self.assertEqual(len(ds), 272)
            self._assert_eq(
                ds[5], (self.target1[80:90], None, self.cov_st1, self.target1[85:95])
            )
            self._assert_eq(
                ds[141],
                (self.target2[130:140], None, self.cov_st2, self.target2[135:145]),
            )

            # two target series with custom max_nr_samples
            ds = FutureCovariatesShiftedDataset(
                target_series=[self.target1, self.target2],
                length=10,
                shift=5,
                max_samples_per_ts=50,
            )
            self.assertEqual(len(ds), 100)
            self._assert_eq(
                ds[5], (self.target1[80:90], None, self.cov_st1, self.target1[85:95])
            )
            self._assert_eq(
                ds[55],
                (self.target2[130:140], None, self.cov_st2, self.target2[135:145]),
            )

            # two targets and one covariate
            with self.assertRaises(ValueError):
                ds = FutureCovariatesShiftedDataset(
                    target_series=[self.target1, self.target2], covariates=[self.cov1]
                )

            # two targets and two covariates
            ds = FutureCovariatesShiftedDataset(
                target_series=[self.target1, self.target2],
                covariates=[self.cov1, self.cov2],
                length=10,
                shift=5,
            )
            self._assert_eq(
                ds[5],
                (
                    self.target1[80:90],
                    self.cov1[85:95],
                    self.cov_st1,
                    self.target1[85:95],
                ),
            )
            self._assert_eq(
                ds[141],
                (
                    self.target2[130:140],
                    self.cov2[135:145],
                    self.cov_st2,
                    self.target2[135:145],
                ),
            )

            # Should contain correct values even when covariates are not aligned
            target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
                self.cov_st2_df
            )
            cov1 = TimeSeries.from_values(np.random.randn(10))
            ds = FutureCovariatesShiftedDataset(
                target_series=[target1], covariates=[cov1], length=3, shift=2
            )
            np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
            np.testing.assert_almost_equal(ds[0][1], cov1.values()[-5:-2])
            np.testing.assert_almost_equal(ds[0][2], self.cov_st2)
            np.testing.assert_almost_equal(ds[0][3], target1.values()[-3:])

            # Should also contain correct values when time-indexed with covariates not aligned
            times1 = pd.date_range(start="20090201", end="20090220", freq="D")
            times2 = pd.date_range(start="20090201", end="20090222", freq="D")
            target1 = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            cov1 = TimeSeries.from_times_and_values(
                times2, np.random.randn(len(times2))
            )
            ds = FutureCovariatesShiftedDataset(
                target_series=[target1], covariates=[cov1], length=3, shift=2
            )
            np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
            np.testing.assert_almost_equal(ds[0][1], cov1.values()[-5:-2])
            np.testing.assert_almost_equal(ds[0][2], self.cov_st2)
            np.testing.assert_almost_equal(ds[0][3], target1.values()[-3:])

            # Should fail if covariates are too short
            target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
                self.cov_st2_df
            )
            cov1 = TimeSeries.from_values(np.random.randn(7))
            ds = FutureCovariatesShiftedDataset(
                target_series=[target1], covariates=[cov1], length=3, shift=2
            )
            with self.assertRaises(ValueError):
                _ = ds[0]

        def test_dual_covariates_shifted_dataset(self):
            # one target series
            ds = DualCovariatesShiftedDataset(
                target_series=self.target1, length=10, shift=5
            )
            self.assertEqual(len(ds), 86)
            self._assert_eq(
                ds[5],
                (self.target1[80:90], None, None, self.cov_st1, self.target1[85:95]),
            )

            # two target series
            ds = DualCovariatesShiftedDataset(
                target_series=[self.target1, self.target2], length=10, shift=5
            )
            self.assertEqual(len(ds), 272)
            self._assert_eq(
                ds[5],
                (self.target1[80:90], None, None, self.cov_st1, self.target1[85:95]),
            )
            self._assert_eq(
                ds[141],
                (
                    self.target2[130:140],
                    None,
                    None,
                    self.cov_st2,
                    self.target2[135:145],
                ),
            )

            # two target series with custom max_nr_samples
            ds = DualCovariatesShiftedDataset(
                target_series=[self.target1, self.target2],
                length=10,
                shift=5,
                max_samples_per_ts=50,
            )
            self.assertEqual(len(ds), 100)
            self._assert_eq(
                ds[5],
                (self.target1[80:90], None, None, self.cov_st1, self.target1[85:95]),
            )
            self._assert_eq(
                ds[55],
                (
                    self.target2[130:140],
                    None,
                    None,
                    self.cov_st2,
                    self.target2[135:145],
                ),
            )

            # two targets and one covariate
            with self.assertRaises(ValueError):
                ds = DualCovariatesShiftedDataset(
                    target_series=[self.target1, self.target2], covariates=[self.cov1]
                )

            # two targets and two covariates
            ds = DualCovariatesShiftedDataset(
                target_series=[self.target1, self.target2],
                covariates=[self.cov1, self.cov2],
                length=10,
                shift=5,
            )
            self._assert_eq(
                ds[5],
                (
                    self.target1[80:90],
                    self.cov1[80:90],
                    self.cov1[85:95],
                    self.cov_st1,
                    self.target1[85:95],
                ),
            )
            self._assert_eq(
                ds[141],
                (
                    self.target2[130:140],
                    self.cov2[130:140],
                    self.cov2[135:145],
                    self.cov_st2,
                    self.target2[135:145],
                ),
            )

            # Should contain correct values even when covariates are not aligned
            target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
                self.cov_st2_df
            )
            cov1 = TimeSeries.from_values(np.random.randn(10))
            ds = DualCovariatesShiftedDataset(
                target_series=[target1], covariates=[cov1], length=3, shift=2
            )
            np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
            np.testing.assert_almost_equal(ds[0][1], cov1.values()[-7:-4])
            np.testing.assert_almost_equal(ds[0][2], cov1.values()[-5:-2])
            np.testing.assert_almost_equal(ds[0][3], self.cov_st2)
            np.testing.assert_almost_equal(ds[0][4], target1.values()[-3:])

            # Should also contain correct values when time-indexed with covariates not aligned
            times1 = pd.date_range(start="20090201", end="20090220", freq="D")
            times2 = pd.date_range(start="20090201", end="20090222", freq="D")
            target1 = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            cov1 = TimeSeries.from_times_and_values(
                times2, np.random.randn(len(times2))
            )
            ds = DualCovariatesShiftedDataset(
                target_series=[target1], covariates=[cov1], length=3, shift=2
            )
            np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
            np.testing.assert_almost_equal(ds[0][1], cov1.values()[-7:-4])
            np.testing.assert_almost_equal(ds[0][2], cov1.values()[-5:-2])
            np.testing.assert_almost_equal(ds[0][3], self.cov_st2)
            np.testing.assert_almost_equal(ds[0][4], target1.values()[-3:])

            # Should fail if covariates are too short
            target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
                self.cov_st2_df
            )
            cov1 = TimeSeries.from_values(np.random.randn(7))
            ds = DualCovariatesShiftedDataset(
                target_series=[target1], covariates=[cov1], length=3, shift=2
            )
            with self.assertRaises(ValueError):
                _ = ds[0]

        def test_horizon_based_dataset(self):
            # one target series
            ds = HorizonBasedDataset(
                target_series=self.target1,
                output_chunk_length=10,
                lh=(1, 3),
                lookback=2,
            )
            self.assertEqual(len(ds), 20)
            self._assert_eq(
                ds[5], (self.target1[65:85], None, self.cov_st1, self.target1[85:95])
            )

            # two target series
            ds = HorizonBasedDataset(
                target_series=[self.target1, self.target2],
                output_chunk_length=10,
                lh=(1, 3),
                lookback=2,
            )
            self.assertEqual(len(ds), 40)
            self._assert_eq(
                ds[5], (self.target1[65:85], None, self.cov_st1, self.target1[85:95])
            )
            self._assert_eq(
                ds[25],
                (self.target2[115:135], None, self.cov_st2, self.target2[135:145]),
            )

            # two targets and one covariate
            with self.assertRaises(ValueError):
                ds = HorizonBasedDataset(
                    target_series=[self.target1, self.target2], covariates=[self.cov1]
                )

            # two targets and two covariates
            ds = HorizonBasedDataset(
                target_series=[self.target1, self.target2],
                covariates=[self.cov1, self.cov2],
                output_chunk_length=10,
                lh=(1, 3),
                lookback=2,
            )
            self._assert_eq(
                ds[5],
                (
                    self.target1[65:85],
                    self.cov1[65:85],
                    self.cov_st1,
                    self.target1[85:95],
                ),
            )
            self._assert_eq(
                ds[25],
                (
                    self.target2[115:135],
                    self.cov2[115:135],
                    self.cov_st2,
                    self.target2[135:145],
                ),
            )

        def test_get_matching_index(self):
            from darts.utils.data.utils import _get_matching_index

            # Check dividable freq
            times1 = pd.date_range(start="20100101", end="20100330", freq="D")
            times2 = pd.date_range(start="20100101", end="20100320", freq="D")
            target = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
            self.assertEqual(_get_matching_index(target, cov, idx=15), 5)

            # check non-dividable freq
            times1 = pd.date_range(start="20100101", end="20120101", freq="M")
            times2 = pd.date_range(start="20090101", end="20110601", freq="M")
            target = TimeSeries.from_times_and_values(
                times1, np.random.randn(len(times1))
            ).with_static_covariates(self.cov_st2_df)
            cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
            self.assertEqual(_get_matching_index(target, cov, idx=15), 15 - 7)

            # check integer-indexed series
            times2 = pd.RangeIndex(start=10, stop=90)
            target = TimeSeries.from_values(
                np.random.randn(100)
            ).with_static_covariates(self.cov_st2_df)
            cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
            self.assertEqual(_get_matching_index(target, cov, idx=15), 5)
