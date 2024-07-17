import inspect
import itertools

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.tests.conftest import TORCH_AVAILABLE
from darts.utils.timeseries_generation import gaussian_timeseries
from darts.utils.utils import freqs

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

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


class TestDataset:
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
            assert type(left) is type(right)
            assert (
                isinstance(
                    left, (TimeSeries, pd.Series, pd.DataFrame, np.ndarray, list)
                )
                or left is None
            )
            if isinstance(left, (pd.Series, pd.DataFrame)):
                assert left.equals(right)
            elif isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
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
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
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
        assert ds[0][4] == target

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
        assert ds[0][4] == target

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
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
            _ = ds[0]

        # Should return correct values when covariates is long enough
        ds = FutureCovariatesInferenceDataset(
            target_series=target, covariates=long_cov, input_chunk_length=10, n=30
        )

        np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
        np.testing.assert_almost_equal(ds[0][1], long_cov.values()[-50:-20])
        np.testing.assert_almost_equal(ds[0][2], self.cov_st2)
        assert ds[0][3] == target

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
        assert ds[0][3] == target

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
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
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
        assert ds[0][4] == target

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
        assert ds[0][4] == target

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
        with pytest.raises(ValueError):
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
        assert ds[0][6] == target

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
        assert ds[0][6] == target

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
        with pytest.raises(ValueError):
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
        assert ds[0][5] == target

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
        assert ds[0][5] == target

    @pytest.mark.parametrize(
        "config",
        [
            # (dataset class, whether contains future, future batch index)
            (PastCovariatesInferenceDataset, None),
            (FutureCovariatesInferenceDataset, 1),
            (DualCovariatesInferenceDataset, 2),
            (MixedCovariatesInferenceDataset, 3),
            (SplitCovariatesInferenceDataset, 2),
        ],
    )
    def test_inference_dataset_output_chunk_shift(self, config):
        ds_cls, future_idx = config
        ocl = 1
        ocs = 2
        target = self.target1[: -(ocl + ocs)]

        ds_covs = {}
        ds_init_params = set(inspect.signature(ds_cls.__init__).parameters)
        for cov_type in ["covariates", "past_covariates", "future_covariates"]:
            if cov_type in ds_init_params:
                ds_covs[cov_type] = self.cov1

        with pytest.raises(ValueError) as err:
            _ = ds_cls(
                target_series=target,
                input_chunk_length=1,
                output_chunk_length=1,
                output_chunk_shift=1,
                n=2,
                **ds_covs,
            )
        assert str(err.value).startswith("Cannot perform auto-regression")

        # regular dataset with output shift=0 and ocl=3: the 3rd future values should be identical to the 1st future
        # values of a dataset with output shift=2 and ocl=1
        ds_reg = ds_cls(
            target_series=target,
            input_chunk_length=1,
            output_chunk_length=3,
            output_chunk_shift=0,
            n=1,
            **ds_covs,
        )

        ds_shift = ds_cls(
            target_series=target,
            input_chunk_length=1,
            output_chunk_length=1,
            output_chunk_shift=ocs,
            n=1,
            **ds_covs,
        )

        batch_reg, batch_shift = ds_reg[0], ds_shift[0]

        # shifted prediction starts 2 steps after regular prediction
        assert batch_reg[-1] == batch_shift[-1] - ocs * target.freq

        if future_idx is not None:
            # 3rd future values of regular ds must be identical to the 1st future values of shifted dataset
            np.testing.assert_array_equal(
                batch_reg[future_idx][ocs:], batch_shift[future_idx]
            )
            batch_reg = batch_reg[:future_idx] + batch_reg[future_idx + 1 :]
            batch_shift = batch_shift[:future_idx] + batch_shift[future_idx + 1 :]

        # without future part, the input will be identical between regular, and shifted dataset
        assert all([
            np.all(el_reg == el_shift)
            for el_reg, el_shift in zip(batch_reg[:-1], batch_shift[:-1])
        ])

    def test_past_covariates_sequential_dataset(self):
        # one target series
        ds = PastCovariatesSequentialDataset(
            target_series=self.target1,
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 81
        self._assert_eq(
            ds[5], (self.target1[75:85], None, self.cov_st1, None, self.target1[85:95])
        )

        # two target series
        ds = PastCovariatesSequentialDataset(
            target_series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 262
        self._assert_eq(
            ds[5], (self.target1[75:85], None, self.cov_st1, None, self.target1[85:95])
        )
        self._assert_eq(
            ds[136],
            (self.target2[125:135], None, self.cov_st2, None, self.target2[135:145]),
        )

        # two target series with custom max_nr_samples
        ds = PastCovariatesSequentialDataset(
            target_series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            max_samples_per_ts=50,
        )
        assert len(ds) == 100
        self._assert_eq(
            ds[5], (self.target1[75:85], None, self.cov_st1, None, self.target1[85:95])
        )
        self._assert_eq(
            ds[55],
            (self.target2[125:135], None, self.cov_st2, None, self.target2[135:145]),
        )

        # two targets and one covariate
        with pytest.raises(ValueError):
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
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[136],
            (
                self.target2[125:135],
                self.cov2[125:135],
                self.cov_st2,
                None,
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
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
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
        assert len(ds) == 81
        self._assert_eq(
            ds[5], (self.target1[75:85], None, self.cov_st1, None, self.target1[85:95])
        )

        # two target series
        ds = FutureCovariatesSequentialDataset(
            target_series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 262
        self._assert_eq(
            ds[5], (self.target1[75:85], None, self.cov_st1, None, self.target1[85:95])
        )
        self._assert_eq(
            ds[136],
            (self.target2[125:135], None, self.cov_st2, None, self.target2[135:145]),
        )

        # two target series with custom max_nr_samples
        ds = FutureCovariatesSequentialDataset(
            target_series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            max_samples_per_ts=50,
        )
        assert len(ds) == 100
        self._assert_eq(
            ds[5], (self.target1[75:85], None, self.cov_st1, None, self.target1[85:95])
        )
        self._assert_eq(
            ds[55],
            (self.target2[125:135], None, self.cov_st2, None, self.target2[135:145]),
        )

        # two targets and one covariate
        with pytest.raises(ValueError):
            ds = FutureCovariatesSequentialDataset(
                target_series=[self.target1, self.target2], covariates=[self.cov1]
            )

        # two targets and two covariates; covariates not aligned, must contain correct values
        target1 = TimeSeries.from_values(np.random.randn(100)).with_static_covariates(
            self.cov_st2_df
        )
        target2 = TimeSeries.from_values(np.random.randn(50)).with_static_covariates(
            self.cov_st2_df
        )
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
        assert ds[0][3] is None
        np.testing.assert_almost_equal(ds[0][4], target1.values()[-10:])

        np.testing.assert_almost_equal(ds[101][0], target2.values()[-40:-30])
        np.testing.assert_almost_equal(ds[101][1], cov2.values()[-60:-50])
        np.testing.assert_almost_equal(ds[101][2], self.cov_st2)
        assert ds[0][3] is None
        np.testing.assert_almost_equal(ds[101][4], target2.values()[-30:-20])

        # Should also contain correct values when time-indexed with covariates not aligned
        times1 = pd.date_range(start="20090201", end="20090220", freq="D")
        times2 = pd.date_range(start="20090201", end="20090222", freq="D")
        target1 = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov1 = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))

        ds = FutureCovariatesSequentialDataset(
            target_series=[target1],
            covariates=[cov1],
            input_chunk_length=2,
            output_chunk_length=2,
        )

        np.testing.assert_almost_equal(ds[0][0], target1.values()[-4:-2])
        np.testing.assert_almost_equal(ds[0][1], cov1.values()[-4:-2])
        np.testing.assert_almost_equal(ds[0][2], self.cov_st2)
        assert ds[0][3] is None
        np.testing.assert_almost_equal(ds[0][4], target1.values()[-2:])

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

        with pytest.raises(ValueError):
            _ = ds[0]

    def test_dual_covariates_sequential_dataset(self):
        # Must contain (past_target, historic_future_covariates, future_covariates, static covariates,
        # sample weight, future_target)

        # one target series
        ds = DualCovariatesSequentialDataset(
            target_series=self.target1,
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 81
        self._assert_eq(
            ds[5],
            (self.target1[75:85], None, None, self.cov_st1, None, self.target1[85:95]),
        )

        # two target series
        ds = DualCovariatesSequentialDataset(
            target_series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 262
        self._assert_eq(
            ds[5],
            (self.target1[75:85], None, None, self.cov_st1, None, self.target1[85:95]),
        )
        self._assert_eq(
            ds[136],
            (
                self.target2[125:135],
                None,
                None,
                self.cov_st2,
                None,
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
        assert len(ds) == 100
        self._assert_eq(
            ds[5],
            (self.target1[75:85], None, None, self.cov_st1, None, self.target1[85:95]),
        )
        self._assert_eq(
            ds[55],
            (
                self.target2[125:135],
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two targets and one covariate
        with pytest.raises(ValueError):
            ds = DualCovariatesSequentialDataset(
                target_series=[self.target1, self.target2], covariates=[self.cov1]
            )

        # two targets and two covariates; covariates not aligned, must contain correct values
        target1 = TimeSeries.from_values(np.random.randn(100)).with_static_covariates(
            self.cov_st2_df
        )
        target2 = TimeSeries.from_values(np.random.randn(50)).with_static_covariates(
            self.cov_st2_df
        )
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
        assert ds[0][4] is None
        np.testing.assert_almost_equal(ds[0][5], target1.values()[-10:])

        np.testing.assert_almost_equal(ds[101][0], target2.values()[-40:-30])
        np.testing.assert_almost_equal(ds[101][1], cov2.values()[-70:-60])
        np.testing.assert_almost_equal(ds[101][2], cov2.values()[-60:-50])
        np.testing.assert_almost_equal(ds[101][3], self.cov_st2)
        assert ds[101][4] is None
        np.testing.assert_almost_equal(ds[101][5], target2.values()[-30:-20])

        # Should also contain correct values when time-indexed with covariates not aligned
        times1 = pd.date_range(start="20090201", end="20090220", freq="D")
        times2 = pd.date_range(start="20090201", end="20090222", freq="D")
        target1 = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov1 = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))

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
        assert ds[0][4] is None
        np.testing.assert_almost_equal(ds[0][5], target1.values()[-2:])

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

        with pytest.raises(ValueError):
            _ = ds[0]

    def test_past_covariates_shifted_dataset(self):
        # one target series
        ds = PastCovariatesShiftedDataset(
            target_series=self.target1, length=10, shift=5
        )
        assert len(ds) == 86
        self._assert_eq(
            ds[5], (self.target1[80:90], None, self.cov_st1, None, self.target1[85:95])
        )

        # two target series
        ds = PastCovariatesShiftedDataset(
            target_series=[self.target1, self.target2], length=10, shift=5
        )
        assert len(ds) == 272
        self._assert_eq(
            ds[5], (self.target1[80:90], None, self.cov_st1, None, self.target1[85:95])
        )
        self._assert_eq(
            ds[141],
            (self.target2[130:140], None, self.cov_st2, None, self.target2[135:145]),
        )

        # two target series with custom max_nr_samples
        ds = PastCovariatesShiftedDataset(
            target_series=[self.target1, self.target2],
            length=10,
            shift=5,
            max_samples_per_ts=50,
        )
        assert len(ds) == 100
        self._assert_eq(
            ds[5], (self.target1[80:90], None, self.cov_st1, None, self.target1[85:95])
        )
        self._assert_eq(
            ds[55],
            (self.target2[130:140], None, self.cov_st2, None, self.target2[135:145]),
        )

        # two targets and one covariate
        with pytest.raises(ValueError):
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
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[141],
            (
                self.target2[130:140],
                self.cov2[130:140],
                self.cov_st2,
                None,
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
        assert ds[0][3] is None
        np.testing.assert_almost_equal(ds[0][4], target1.values()[-3:])

        # Should also contain correct values when time-indexed with covariates not aligned
        times1 = pd.date_range(start="20090201", end="20090220", freq="D")
        times2 = pd.date_range(start="20090201", end="20090222", freq="D")
        target1 = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov1 = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        ds = PastCovariatesShiftedDataset(
            target_series=[target1], covariates=[cov1], length=3, shift=2
        )
        np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0][1], cov1.values()[-7:-4])
        np.testing.assert_almost_equal(ds[0][2], self.cov_st2)
        assert ds[0][3] is None
        np.testing.assert_almost_equal(ds[0][4], target1.values()[-3:])

        # Should fail if covariates are too short
        target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
            self.cov_st2_df
        )
        cov1 = TimeSeries.from_values(np.random.randn(5))
        ds = PastCovariatesShiftedDataset(
            target_series=[target1], covariates=[cov1], length=3, shift=2
        )
        with pytest.raises(ValueError):
            _ = ds[0]

    def test_future_covariates_shifted_dataset(self):
        # one target series
        ds = FutureCovariatesShiftedDataset(
            target_series=self.target1, length=10, shift=5
        )
        assert len(ds) == 86
        self._assert_eq(
            ds[5], (self.target1[80:90], None, self.cov_st1, None, self.target1[85:95])
        )

        # two target series
        ds = FutureCovariatesShiftedDataset(
            target_series=[self.target1, self.target2], length=10, shift=5
        )
        assert len(ds) == 272
        self._assert_eq(
            ds[5], (self.target1[80:90], None, self.cov_st1, None, self.target1[85:95])
        )
        self._assert_eq(
            ds[141],
            (self.target2[130:140], None, self.cov_st2, None, self.target2[135:145]),
        )

        # two target series with custom max_nr_samples
        ds = FutureCovariatesShiftedDataset(
            target_series=[self.target1, self.target2],
            length=10,
            shift=5,
            max_samples_per_ts=50,
        )
        assert len(ds) == 100
        self._assert_eq(
            ds[5], (self.target1[80:90], None, self.cov_st1, None, self.target1[85:95])
        )
        self._assert_eq(
            ds[55],
            (self.target2[130:140], None, self.cov_st2, None, self.target2[135:145]),
        )

        # two targets and one covariate
        with pytest.raises(ValueError):
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
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[141],
            (
                self.target2[130:140],
                self.cov2[135:145],
                self.cov_st2,
                None,
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
        assert ds[0][3] is None
        np.testing.assert_almost_equal(ds[0][4], target1.values()[-3:])

        # Should also contain correct values when time-indexed with covariates not aligned
        times1 = pd.date_range(start="20090201", end="20090220", freq="D")
        times2 = pd.date_range(start="20090201", end="20090222", freq="D")
        target1 = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov1 = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        ds = FutureCovariatesShiftedDataset(
            target_series=[target1], covariates=[cov1], length=3, shift=2
        )
        np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0][1], cov1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0][2], self.cov_st2)
        assert ds[0][3] is None
        np.testing.assert_almost_equal(ds[0][4], target1.values()[-3:])

        # Should fail if covariates are too short
        target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
            self.cov_st2_df
        )
        cov1 = TimeSeries.from_values(np.random.randn(7))
        ds = FutureCovariatesShiftedDataset(
            target_series=[target1], covariates=[cov1], length=3, shift=2
        )
        with pytest.raises(ValueError):
            _ = ds[0]

    def test_dual_covariates_shifted_dataset(self):
        # one target series
        ds = DualCovariatesShiftedDataset(
            target_series=self.target1, length=10, shift=5
        )
        assert len(ds) == 86
        self._assert_eq(
            ds[5],
            (self.target1[80:90], None, None, self.cov_st1, None, self.target1[85:95]),
        )

        # two target series
        ds = DualCovariatesShiftedDataset(
            target_series=[self.target1, self.target2], length=10, shift=5
        )
        assert len(ds) == 272
        self._assert_eq(
            ds[5],
            (self.target1[80:90], None, None, self.cov_st1, None, self.target1[85:95]),
        )
        self._assert_eq(
            ds[141],
            (
                self.target2[130:140],
                None,
                None,
                self.cov_st2,
                None,
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
        assert len(ds) == 100
        self._assert_eq(
            ds[5],
            (self.target1[80:90], None, None, self.cov_st1, None, self.target1[85:95]),
        )
        self._assert_eq(
            ds[55],
            (
                self.target2[130:140],
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two targets and one covariate
        with pytest.raises(ValueError):
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
                None,
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
                None,
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
        assert ds[0][4] is None
        np.testing.assert_almost_equal(ds[0][5], target1.values()[-3:])

        # Should also contain correct values when time-indexed with covariates not aligned
        times1 = pd.date_range(start="20090201", end="20090220", freq="D")
        times2 = pd.date_range(start="20090201", end="20090222", freq="D")
        target1 = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov1 = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        ds = DualCovariatesShiftedDataset(
            target_series=[target1], covariates=[cov1], length=3, shift=2
        )
        np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0][1], cov1.values()[-7:-4])
        np.testing.assert_almost_equal(ds[0][2], cov1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0][3], self.cov_st2)
        assert ds[0][4] is None
        np.testing.assert_almost_equal(ds[0][5], target1.values()[-3:])

        # Should fail if covariates are too short
        target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
            self.cov_st2_df
        )
        cov1 = TimeSeries.from_values(np.random.randn(7))
        ds = DualCovariatesShiftedDataset(
            target_series=[target1], covariates=[cov1], length=3, shift=2
        )
        with pytest.raises(ValueError):
            _ = ds[0]

    @pytest.mark.parametrize("use_weight", [False, True])
    def test_horizon_based_dataset(self, use_weight):
        weight1 = self.target1 + 1
        weight2 = self.target2 + 1

        weight = weight1 if use_weight else None
        weight_exp = weight1[85:95] if use_weight else None
        # one target series
        ds = HorizonBasedDataset(
            target_series=self.target1,
            output_chunk_length=10,
            lh=(1, 3),
            lookback=2,
            sample_weight=weight,
        )
        assert len(ds) == 20
        self._assert_eq(
            ds[5],
            (self.target1[65:85], None, self.cov_st1, weight_exp, self.target1[85:95]),
        )

        # two target series
        weight = [weight1, weight2] if use_weight else None
        weight_exp1 = weight1[85:95] if use_weight else None
        weight_exp2 = weight2[135:145] if use_weight else None
        ds = HorizonBasedDataset(
            target_series=[self.target1, self.target2],
            output_chunk_length=10,
            lh=(1, 3),
            lookback=2,
            sample_weight=weight,
        )
        assert len(ds) == 40
        self._assert_eq(
            ds[5],
            (self.target1[65:85], None, self.cov_st1, weight_exp1, self.target1[85:95]),
        )
        self._assert_eq(
            ds[25],
            (
                self.target2[115:135],
                None,
                self.cov_st2,
                weight_exp2,
                self.target2[135:145],
            ),
        )

        # two targets and one covariate
        with pytest.raises(ValueError):
            ds = HorizonBasedDataset(
                target_series=[self.target1, self.target2], covariates=[self.cov1]
            )

        # two targets and two covariates
        weight = [weight1, weight2] if use_weight else None
        weight_exp1 = weight1[85:95] if use_weight else None
        weight_exp2 = weight2[135:145] if use_weight else None
        ds = HorizonBasedDataset(
            target_series=[self.target1, self.target2],
            covariates=[self.cov1, self.cov2],
            output_chunk_length=10,
            lh=(1, 3),
            lookback=2,
            sample_weight=weight,
        )
        self._assert_eq(
            ds[5],
            (
                self.target1[65:85],
                self.cov1[65:85],
                self.cov_st1,
                weight_exp1,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[25],
            (
                self.target2[115:135],
                self.cov2[115:135],
                self.cov_st2,
                weight_exp2,
                self.target2[135:145],
            ),
        )

    @pytest.mark.parametrize(
        "config",
        [
            # (dataset class, whether contains future, future batch index)
            (PastCovariatesSequentialDataset, None),
            (FutureCovariatesSequentialDataset, 1),
            (DualCovariatesSequentialDataset, 2),
            (MixedCovariatesSequentialDataset, 3),
            (SplitCovariatesSequentialDataset, 2),
        ],
    )
    def test_sequential_training_dataset_output_chunk_shift(self, config):
        ds_cls, future_idx = config
        ocl = 1
        ocs = 2
        target = self.target1[: -(ocl + ocs)]
        sample_weight = target + 1

        ds_covs = {}
        ds_init_params = set(inspect.signature(ds_cls.__init__).parameters)
        for cov_type in ["covariates", "past_covariates", "future_covariates"]:
            if cov_type in ds_init_params:
                ds_covs[cov_type] = self.cov1

        # regular dataset with output shift=0 and ocl=3: the 3rd future values should be identical to the 1st future
        # values of a dataset with output shift=2 and ocl=1
        ds_reg = ds_cls(
            target_series=target,
            input_chunk_length=1,
            output_chunk_length=3,
            output_chunk_shift=0,
            sample_weight=sample_weight,
            **ds_covs,
        )

        ds_shift = ds_cls(
            target_series=target,
            input_chunk_length=1,
            output_chunk_length=1,
            output_chunk_shift=ocs,
            sample_weight=sample_weight,
            **ds_covs,
        )

        batch_reg, batch_shift = ds_reg[0], ds_shift[0]

        if future_idx is not None:
            # 3rd future values of regular ds must be identical to the 1st future values of shifted dataset
            np.testing.assert_array_equal(
                batch_reg[future_idx][-1:], batch_shift[future_idx]
            )
            batch_reg = batch_reg[:future_idx] + batch_reg[future_idx + 1 :]
            batch_shift = batch_shift[:future_idx] + batch_shift[future_idx + 1 :]

        # last two elements are (sample weight, output chunk of the target series).
        # 3rd future values of regular ds must be identical to the 1st future values of shifted dataset
        batch_reg = batch_reg[:-2] + (batch_reg[-2][ocs:], batch_reg[-1][ocs:])

        # without future part, the input will be identical between regular, and shifted dataset
        assert all([
            np.all(el_reg == el_shift)
            for el_reg, el_shift in zip(batch_reg[:-1], batch_shift[:-1])
        ])

    @pytest.mark.parametrize(
        "config",
        itertools.product(
            [
                PastCovariatesSequentialDataset,
                FutureCovariatesSequentialDataset,
                DualCovariatesSequentialDataset,
                MixedCovariatesSequentialDataset,
                SplitCovariatesSequentialDataset,
            ],
            [True, False],
        ),
    )
    def test_sequential_training_dataset_weight(self, config):
        ds_cls, manual_weight = config

        def get_built_in_weigths(targets):
            if isinstance(targets, list):
                max_steps = max([len(ts) for ts in targets])
            else:
                max_steps = len(targets)
            weight_expected = np.linspace(0, 1, max_steps)[-3:]
            return np.expand_dims(weight_expected, -1)

        target1 = self.target1
        target2 = self.target2
        weight1 = target1 + 1
        weight2 = target2 + 1
        built_in_weight = "linear"

        ds_covs = {}
        ds_init_params = set(inspect.signature(ds_cls.__init__).parameters)
        for cov_type in ["covariates", "past_covariates", "future_covariates"]:
            if cov_type in ds_init_params:
                ds_covs[cov_type] = self.cov1

        # no sample weight
        ds = ds_cls(
            target_series=target1,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=None,
            **ds_covs,
        )
        assert ds[0][-2] is None

        # whenever we use sample weight, the weight are extracted from the same time frame as the target labels
        # since we set the weight to be `target + 1`, the returned batch weight must also be `batch_target_label + 1`

        # single univariate
        target = target1
        weight = weight1 if manual_weight else built_in_weight
        ds = ds_cls(
            target_series=target,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=weight,
            **ds_covs,
        )
        weight_exp = ds[0][-1] + 1 if manual_weight else get_built_in_weigths(target)
        assert np.all(ds[0][-2] == weight_exp)

        # single univariate with longer weight
        target = target1
        weight = (
            weight1.prepend_values([0.0]).append_values([0.0])
            if manual_weight
            else built_in_weight
        )
        ds = ds_cls(
            target_series=target,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=weight,
            **ds_covs,
        )
        weight_exp = ds[0][-1] + 1 if manual_weight else get_built_in_weigths(target)
        assert np.all(ds[0][-2] == weight_exp)

        # single multivariate with multivariate weight
        target = target1.stack(target1 + 1)
        weight = weight1.stack(weight1 + 1) if manual_weight else built_in_weight
        ds = ds_cls(
            target_series=target,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=weight,
            **ds_covs,
        )
        weight_exp = ds[0][-1] + 1 if manual_weight else get_built_in_weigths(target)
        assert np.all(ds[0][-2] == weight_exp)

        # single multivariate with univariate (global) weight
        target = target1.stack(target1 + 1)
        weight = weight1 if manual_weight else built_in_weight
        ds = ds_cls(
            target_series=target,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=weight,
            **ds_covs,
        )
        # output weight corresponds to first target component + 1 (e.g. weight1)
        weight_exp = (
            ds[0][-1][:, 0:1] + 1 if manual_weight else get_built_in_weigths(target)
        )
        assert np.all(ds[0][-2] == weight_exp)

        # single univariate and list of single weight
        target = target1
        weight = [weight1] if manual_weight else built_in_weight
        ds = ds_cls(
            target_series=target,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=weight,
            **ds_covs,
        )
        weight_exp = ds[0][-1] + 1 if manual_weight else get_built_in_weigths(target)
        assert np.all(ds[0][-2] == weight_exp)

        # multiple univariate
        target = [target1, target2]
        weight = [weight1, weight2] if manual_weight else built_in_weight
        ds = ds_cls(
            target_series=target,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=weight,
            **{k: [v] * 2 for k, v in ds_covs.items()},
        )
        weight_exp = ds[0][-1] + 1 if manual_weight else get_built_in_weigths(target)
        assert np.all(ds[0][-2] == weight_exp)

        # multiple multivariate
        target = [target1.stack(target1 + 1), target2.stack(target2 + 1)]
        weight = (
            [weight1.stack(weight1 + 1), weight2.stack(weight2 + 1)]
            if manual_weight
            else built_in_weight
        )
        ds = ds_cls(
            target_series=target,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=weight,
            **{k: [v] * 2 for k, v in ds_covs.items()},
        )
        weight_exp = ds[0][-1] + 1 if manual_weight else get_built_in_weigths(target)
        assert np.all(ds[0][-2] == weight_exp)

    @pytest.mark.parametrize(
        "ds_cls",
        [
            PastCovariatesSequentialDataset,
            FutureCovariatesSequentialDataset,
            DualCovariatesSequentialDataset,
            MixedCovariatesSequentialDataset,
            SplitCovariatesSequentialDataset,
        ],
    )
    def test_sequential_training_dataset_invalid_weight(self, ds_cls):
        ts = self.target1

        # invalid built-in weight
        with pytest.raises(ValueError) as err:
            _ = ds_cls(
                target_series=[ts, ts],
                input_chunk_length=1,
                output_chunk_length=3,
                sample_weight="invalid",
            )
        assert str(err.value).startswith(
            "Invalid `sample_weight` value: `'invalid'`. If a string, must be one of: "
        )

        # mismatch number of target and weight series
        with pytest.raises(ValueError) as err:
            _ = ds_cls(
                target_series=[ts, ts],
                input_chunk_length=1,
                output_chunk_length=3,
                sample_weight=[ts],
            )
        assert (
            str(err.value)
            == "The provided sequence of target `series` must have the same "
            "length as the provided sequence of `sample_weight`."
        )

        # too many weight components
        ds = ds_cls(
            target_series=ts,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=ts.stack(ts + 1),
        )
        with pytest.raises(ValueError) as err:
            _ = ds[0]
        assert (
            str(err.value)
            == "The number of components in `sample_weight` must either be `1` or match "
            "the number of target series components `1`. (0-th series)"
        )

        # weight too short end
        ds = ds_cls(
            target_series=ts,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=ts[:-1],
        )
        with pytest.raises(ValueError) as err:
            _ = ds[0]
        assert (
            str(err.value)
            == "Missing sample weights; could not find sample weights in index value range: "
            "2000-04-07 00:00:00 - 2000-04-09 00:00:00."
        )

        # weight too short start
        ds = ds_cls(
            target_series=ts,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=ts[2:],
        )
        with pytest.raises(ValueError) as err:
            _ = ds[len(ds) - 1]
        assert (
            str(err.value)
            == "Missing sample weights; could not find sample weights in index value range: "
            "2000-01-02 00:00:00 - 2000-01-04 00:00:00."
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
        assert _get_matching_index(target, cov, idx=15) == 5

        # check non-dividable freq
        times1 = pd.date_range(start="20100101", end="20120101", freq=freqs["ME"])
        times2 = pd.date_range(start="20090101", end="20110601", freq=freqs["ME"])
        target = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        assert _get_matching_index(target, cov, idx=15) == 15 - 7

        # check integer-indexed series
        times2 = pd.RangeIndex(start=10, stop=90)
        target = TimeSeries.from_values(np.random.randn(100)).with_static_covariates(
            self.cov_st2_df
        )
        cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        assert _get_matching_index(target, cov, idx=15) == 5
