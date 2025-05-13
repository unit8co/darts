import itertools
import math

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

from darts.utils.data import (
    HorizonBasedTorchTrainingDataset,
    SequentialTorchInferenceDataset,
    SequentialTorchTrainingDataset,
    ShiftedTorchTrainingDataset,
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

    def _check_ds_stride(self, ds_regular, ds_stride, stride: int):
        """
        Every `stride`-th values in a dataset with stride=1 should be identical to the dataset stridden with `stride
        """
        # if the un-stridden length is a multiple of the stride
        if len(ds_regular) % stride == 0:
            assert len(ds_regular) == len(ds_stride) * stride

        for idx, batch_str in enumerate(ds_stride):
            for entry_s, entry_r in zip(batch_str, ds_regular[idx * stride]):
                if entry_s is not None and entry_r is not None:
                    np.testing.assert_almost_equal(entry_s, entry_r)
                else:
                    assert entry_s == entry_r

    def test_past_covariates_inference_dataset(self):
        # one target series
        ds = SequentialTorchInferenceDataset(
            series=self.target1, input_chunk_length=len(self.target1)
        )
        np.testing.assert_almost_equal(ds[0][0], self.vals1)
        self._assert_eq(ds[0][1:], (None, None, None, None, self.cov_st1, self.target1))

        # two target series
        ds = SequentialTorchInferenceDataset(
            series=[self.target1, self.target2],
            input_chunk_length=max(len(self.target1), len(self.target2)),
        )
        np.testing.assert_almost_equal(ds[1][0], self.vals2)
        self._assert_eq(ds[1][1:], (None, None, None, None, self.cov_st2, self.target2))

        # fail if covariates do not have same size
        with pytest.raises(ValueError) as exc:
            ds = SequentialTorchInferenceDataset(
                series=[self.target1, self.target2], past_covariates=[self.cov1]
            )
        assert str(exc.value) == (
            "The sequence of `past_covariates` must have the same length as the sequence of target `series`."
        )

        # with covariates
        ds = SequentialTorchInferenceDataset(
            series=[self.target1, self.target2],
            past_covariates=[self.cov1, self.cov2],
            input_chunk_length=max(len(self.target1), len(self.target2)),
        )
        np.testing.assert_almost_equal(ds[1][0], self.vals2)
        np.testing.assert_almost_equal(ds[1][1], self.cov2.values())
        self._assert_eq(
            ds[1][2:], (None, None, None, self.cov_st2, self.target2)
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

        ds = SequentialTorchInferenceDataset(
            series=target,
            past_covariates=short_cov,
            input_chunk_length=10,
            output_chunk_length=10,
            n=30,
        )

        # should fail if covariates are too short
        with pytest.raises(ValueError) as exc:
            _ = ds[0]
        assert str(exc.value) == (
            "For the given forecasting horizon `n=30`, the provided `past_covariates` at series sequence index "
            "`0` do not extend far enough into the future. As `n > output_chunk_length` the `past_covariates` "
            "must end at or after time step `2010-07-21 00:00:00`, whereas now the end is at time "
            "step `2010-07-01 00:00:00`."
        )

        # Should return correct values when covariates is long enough
        ds = SequentialTorchInferenceDataset(
            series=target,
            past_covariates=long_cov,
            input_chunk_length=10,
            output_chunk_length=10,
            n=30,
        )

        np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
        np.testing.assert_almost_equal(ds[0][1], long_cov.values()[-60:-50])
        np.testing.assert_almost_equal(ds[0][2], long_cov.values()[-50:-30])
        assert ds[0][3] is None
        assert ds[0][4] is None
        np.testing.assert_almost_equal(ds[0][5], self.cov_st2)
        assert ds[0][6] == target

        # Should also work for integer-indexed series
        target = TimeSeries.from_times_and_values(
            pd.RangeIndex(start=10, stop=50, step=1), np.random.randn(40)
        ).with_static_covariates(self.cov_st2_df)
        covariate = TimeSeries.from_times_and_values(
            pd.RangeIndex(start=20, stop=80, step=1), np.random.randn(60)
        )

        ds = SequentialTorchInferenceDataset(
            series=target,
            past_covariates=covariate,
            input_chunk_length=10,
            output_chunk_length=10,
            n=20,
        )

        np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
        np.testing.assert_almost_equal(ds[0][1], covariate.values()[20:30])
        np.testing.assert_almost_equal(ds[0][2], covariate.values()[30:40])
        assert ds[0][3] is None
        assert ds[0][4] is None
        np.testing.assert_almost_equal(ds[0][5], self.cov_st2)
        assert ds[0][6] == target

    def test_future_covariates_inference_dataset(self):
        # one target series
        ds = SequentialTorchInferenceDataset(
            series=self.target1, input_chunk_length=len(self.target1)
        )
        np.testing.assert_almost_equal(ds[0][0], self.vals1)
        self._assert_eq(ds[0][1:], (None, None, None, None, self.cov_st1, self.target1))

        # two target series
        ds = SequentialTorchInferenceDataset(
            series=[self.target1, self.target2],
            input_chunk_length=max(len(self.target1), len(self.target2)),
        )
        np.testing.assert_almost_equal(ds[1][0], self.vals2)
        self._assert_eq(ds[1][1:], (None, None, None, None, self.cov_st2, self.target2))

        # fail if covariates do not have same size
        with pytest.raises(ValueError) as exc:
            ds = SequentialTorchInferenceDataset(
                series=[self.target1, self.target2],
                future_covariates=[self.cov1],
            )
        assert str(exc.value) == (
            "The sequence of `future_covariates` must have the same length as the sequence of target `series`."
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

        ds = SequentialTorchInferenceDataset(
            series=target,
            future_covariates=short_cov,
            input_chunk_length=10,
            n=30,
        )

        # should fail if covariates are too short
        with pytest.raises(ValueError) as exc:
            _ = ds[0]
        assert str(exc.value) == (
            "For the given forecasting horizon `n=30`, the provided `future_covariates` at series sequence "
            "index `0` do not extend far enough into the future. As `n > output_chunk_length` the "
            "`future_covariates` must end at or after time step `2010-07-31 00:00:00`, whereas now "
            "the end is at time step `2010-07-01 00:00:00`."
        )

        # Should return correct values when covariates is long enough
        ds = SequentialTorchInferenceDataset(
            series=target,
            future_covariates=long_cov,
            input_chunk_length=10,
            n=30,
        )

        np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
        assert ds[0][1] is None
        assert ds[0][2] is None
        np.testing.assert_almost_equal(ds[0][3], long_cov.values()[-60:-50])
        np.testing.assert_almost_equal(ds[0][4], long_cov.values()[-50:-20])
        np.testing.assert_almost_equal(ds[0][5], self.cov_st2)
        assert ds[0][6] == target

        # Should also work for integer-indexed series
        target = TimeSeries.from_times_and_values(
            pd.RangeIndex(start=10, stop=50, step=1), np.random.randn(40)
        ).with_static_covariates(self.cov_st2_df)
        covariate = TimeSeries.from_times_and_values(
            pd.RangeIndex(start=20, stop=80, step=1), np.random.randn(60)
        )

        ds = SequentialTorchInferenceDataset(
            series=target,
            future_covariates=covariate,
            input_chunk_length=10,
            n=20,
        )

        np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
        assert ds[0][1] is None
        assert ds[0][2] is None
        np.testing.assert_almost_equal(ds[0][3], covariate.values()[20:30])
        np.testing.assert_almost_equal(ds[0][4], covariate.values()[30:50])
        np.testing.assert_almost_equal(ds[0][5], self.cov_st2)
        assert ds[0][6] == target

    def test_dual_covariates_inference_dataset(self):
        # one target series
        ds = SequentialTorchInferenceDataset(
            series=self.target1, input_chunk_length=len(self.target1)
        )
        np.testing.assert_almost_equal(ds[0][0], self.vals1)
        self._assert_eq(ds[0][1:], (None, None, None, None, self.cov_st1, self.target1))

        # two target series
        ds = SequentialTorchInferenceDataset(
            series=[self.target1, self.target2],
            input_chunk_length=max(len(self.target1), len(self.target2)),
        )
        np.testing.assert_almost_equal(ds[1][0], self.vals2)
        self._assert_eq(ds[1][1:], (None, None, None, None, self.cov_st2, self.target2))

        # fail if covariates do not have same size
        with pytest.raises(ValueError):
            ds = SequentialTorchInferenceDataset(
                series=[self.target1, self.target2],
                future_covariates=[self.cov1],
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

        ds = SequentialTorchInferenceDataset(
            series=target,
            future_covariates=short_cov,
            input_chunk_length=10,
            output_chunk_length=10,
            n=30,
        )

        # should fail if covariates are too short
        with pytest.raises(ValueError):
            _ = ds[0]

        # Should return correct values when covariates is long enough
        ds = SequentialTorchInferenceDataset(
            series=target,
            future_covariates=long_cov,
            input_chunk_length=10,
            output_chunk_length=10,
            n=30,
        )

        np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
        assert ds[0][1] is None
        assert ds[0][2] is None
        np.testing.assert_almost_equal(ds[0][3], long_cov.values()[-60:-50])
        np.testing.assert_almost_equal(ds[0][4], long_cov.values()[-50:-20])
        np.testing.assert_almost_equal(ds[0][5], self.cov_st2)
        assert ds[0][6] == target

        # Should also work for integer-indexed series
        target = TimeSeries.from_times_and_values(
            pd.RangeIndex(start=10, stop=50, step=1), np.random.randn(40)
        ).with_static_covariates(self.cov_st2_df)
        covariate = TimeSeries.from_times_and_values(
            pd.RangeIndex(start=20, stop=80, step=1), np.random.randn(60)
        )

        ds = SequentialTorchInferenceDataset(
            series=target,
            future_covariates=covariate,
            input_chunk_length=10,
            output_chunk_length=10,
            n=20,
        )

        np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
        assert ds[0][1] is None
        assert ds[0][2] is None
        np.testing.assert_almost_equal(ds[0][3], covariate.values()[20:30])
        np.testing.assert_almost_equal(ds[0][4], covariate.values()[30:50])
        np.testing.assert_almost_equal(ds[0][5], self.cov_st2)
        assert ds[0][6] == target

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

        ds = SequentialTorchInferenceDataset(
            series=target,
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
        ds = SequentialTorchInferenceDataset(
            series=target,
            past_covariates=long_past_cov,
            future_covariates=future_cov,
            input_chunk_length=10,
            output_chunk_length=10,
            n=30,
        )

        # It should contain:
        # past_target, past_covariates, future_past_covariates, historic_future_covariates, future_covariates
        np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
        np.testing.assert_almost_equal(ds[0][1], long_past_cov.values()[-60:-50])
        np.testing.assert_almost_equal(ds[0][2], long_past_cov.values()[-50:-30])
        np.testing.assert_almost_equal(ds[0][3], future_cov.values()[-60:-50])
        np.testing.assert_almost_equal(ds[0][4], future_cov.values()[-50:-20])
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

        ds = SequentialTorchInferenceDataset(
            series=target,
            past_covariates=past_cov,
            future_covariates=future_cov,
            input_chunk_length=10,
            output_chunk_length=10,
            n=20,
        )

        np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
        np.testing.assert_almost_equal(ds[0][1], past_cov.values()[20:30])
        np.testing.assert_almost_equal(ds[0][2], past_cov.values()[30:40])
        np.testing.assert_almost_equal(ds[0][3], future_cov.values()[10:20])
        np.testing.assert_almost_equal(ds[0][4], future_cov.values()[20:40])
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

        ds = SequentialTorchInferenceDataset(
            series=target,
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
        ds = SequentialTorchInferenceDataset(
            series=target,
            past_covariates=long_past_cov,
            future_covariates=future_cov,
            input_chunk_length=10,
            output_chunk_length=10,
            n=30,
        )

        # It should contain:
        # past_target, past_covariates, future_past_covariates, historic_future_covariates,
        # future_covariates, future_past_covariates
        np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
        np.testing.assert_almost_equal(ds[0][1], long_past_cov.values()[-60:-50])
        np.testing.assert_almost_equal(ds[0][2], long_past_cov.values()[-50:-30])
        np.testing.assert_almost_equal(ds[0][3], future_cov.values()[-60:-50])
        np.testing.assert_almost_equal(ds[0][4], future_cov.values()[-50:-20])
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

        ds = SequentialTorchInferenceDataset(
            series=target,
            past_covariates=past_cov,
            future_covariates=future_cov,
            input_chunk_length=10,
            output_chunk_length=10,
            n=20,
        )

        np.testing.assert_almost_equal(ds[0][0], target.values()[-10:])
        np.testing.assert_almost_equal(ds[0][1], past_cov.values()[20:30])
        np.testing.assert_almost_equal(ds[0][2], past_cov.values()[30:40])
        np.testing.assert_almost_equal(ds[0][3], future_cov.values()[10:20])
        np.testing.assert_almost_equal(ds[0][4], future_cov.values()[20:40])
        np.testing.assert_almost_equal(ds[0][5], self.cov_st2)
        assert ds[0][6] == target

    @pytest.mark.parametrize(
        "config",
        [
            # (dataset class, whether contains future, future batch index)
            (SequentialTorchInferenceDataset, [], None),
            (SequentialTorchInferenceDataset, ["past"], None),
            (SequentialTorchInferenceDataset, ["future"], 4),
            (SequentialTorchInferenceDataset, ["past", "future"], 4),
        ],
    )
    def test_inference_dataset_output_chunk_shift(self, config):
        ds_cls, use_covs, future_idx = config
        ocl = 1
        ocs = 2
        target = self.target1[: -(ocl + ocs)]

        ds_covs = {}
        for cov_type in use_covs:
            ds_covs[cov_type + "_covariates"] = self.cov1

        with pytest.raises(ValueError) as err:
            _ = ds_cls(
                series=target,
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
            series=target,
            input_chunk_length=1,
            output_chunk_length=3,
            output_chunk_shift=0,
            n=1,
            **ds_covs,
        )

        ds_shift = ds_cls(
            series=target,
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

    def test_inference_dataset_bounds(self):
        # target1 has length 100
        assert len(self.target1) == 100

        kwargs = {
            "input_chunk_length": 3,
            "output_chunk_length": 1,
            "n": 1,
        }

        # missing stride
        with pytest.raises(ValueError) as exc:
            SequentialTorchInferenceDataset(
                series=self.target1, stride=0, bounds=np.array([[3, 100]]), **kwargs
            )
        assert (
            str(exc.value)
            == "Must supply either both `stride` and `bounds`, or none of them."
        )

        # stride = 1
        ds = SequentialTorchInferenceDataset(
            series=self.target1, stride=1, bounds=np.array([[3, 100]]), **kwargs
        )
        # length 98
        assert len(ds) == 100 - 3 + 1
        # first two sample are from beginning of the target with stride 1
        np.testing.assert_array_almost_equal(ds[0][0], self.target1.values()[:3])
        assert ds[0][-2] == self.target1
        assert ds[0][-1] == self.target1._time_index[3]

        np.testing.assert_array_almost_equal(ds[1][0], self.target1.values()[1:4])
        assert ds[1][-2] == self.target1
        assert ds[1][-1] == self.target1._time_index[4]

        # last two sample are from end of the target with stride 1
        np.testing.assert_array_almost_equal(ds[96][0], self.target1.values()[-4:-1])
        assert ds[96][-2] == self.target1
        assert ds[96][-1] == self.target1._time_index[-1]
        np.testing.assert_array_almost_equal(ds[97][0], self.target1.values()[-3:])
        assert ds[97][-2] == self.target1
        assert ds[97][-1] == self.target1._time_index[-1] + self.target1.freq

        # stride = 2, setting bounds upper limit as `100` can still only compute until `99` since starting
        # at `3` with stride
        ds = SequentialTorchInferenceDataset(
            series=self.target1, stride=2, bounds=np.array([[3, 100]]), **kwargs
        )

        # length 49
        assert len(ds) == math.ceil((100 - 3 + 1) / 2)
        # first two sample are from beginning of the target
        np.testing.assert_array_almost_equal(ds[0][0], self.target1.values()[:3])
        assert ds[0][-1] == self.target1._time_index[3]
        np.testing.assert_array_almost_equal(ds[1][0], self.target1.values()[2:5])
        assert ds[1][-1] == self.target1._time_index[5]
        # last two sample are from end of the target
        np.testing.assert_array_almost_equal(ds[47][0], self.target1.values()[-6:-3])
        assert ds[47][-1] == self.target1._time_index[-3]
        np.testing.assert_array_almost_equal(ds[48][0], self.target1.values()[-4:-1])
        assert ds[48][-1] == self.target1._time_index[-1]

        # stride = 2, output_chunk_shift = 1, same past target values but pred time is shifted by `+1`
        ds = SequentialTorchInferenceDataset(
            series=self.target1,
            stride=2,
            output_chunk_shift=1,
            bounds=np.array([[3, 100]]),
            **kwargs,
        )

        # length 49
        assert len(ds) == math.ceil((100 - 3 + 1 - 1) / 2)
        np.testing.assert_array_almost_equal(ds[0][0], self.target1.values()[:3])
        assert ds[0][-1] == self.target1._time_index[4]
        np.testing.assert_array_almost_equal(ds[1][0], self.target1.values()[2:5])
        assert ds[1][-1] == self.target1._time_index[6]
        np.testing.assert_array_almost_equal(ds[47][0], self.target1.values()[-6:-3])
        assert ds[47][-1] == self.target1._time_index[-2]
        np.testing.assert_array_almost_equal(ds[48][0], self.target1.values()[-4:-1])
        assert ds[48][-1] == self.target1._time_index[-1] + self.target1.freq

        # stride = 2, setting bounds upper limit as `101` will result in an index error for sample 50
        ds = SequentialTorchInferenceDataset(
            series=self.target1, stride=2, bounds=np.array([[3, 101]]), **kwargs
        )

        # length 50
        assert len(ds) == math.ceil((101 - 3 + 1) / 2)
        # getting the samples from before works
        np.testing.assert_array_almost_equal(ds[0][0], self.target1.values()[:3])
        assert ds[0][-1] == self.target1._time_index[3]
        np.testing.assert_array_almost_equal(ds[1][0], self.target1.values()[2:5])
        assert ds[1][-1] == self.target1._time_index[5]
        np.testing.assert_array_almost_equal(ds[47][0], self.target1.values()[-6:-3])
        assert ds[47][-1] == self.target1._time_index[-3]
        np.testing.assert_array_almost_equal(ds[48][0], self.target1.values()[-4:-1])
        assert ds[48][-1] == self.target1._time_index[-1]

        # but sample at index 50 raises an error
        with pytest.raises(IndexError):
            _ = ds[50]

    def test_inference_dataset_series_too_short(self):
        # stride = 2, setting bounds upper limit as `101` will result in an index error for sample 50
        ds = SequentialTorchInferenceDataset(
            series=self.target1, input_chunk_length=len(self.target1) + 1
        )
        with pytest.raises(ValueError) as exc:
            _ = ds[0]
        assert str(exc.value).startswith(
            "The dataset contains target `series` that are too short"
        )

        # past covs start too late
        ds = SequentialTorchInferenceDataset(
            series=self.target1,
            past_covariates=self.target1[1:],
            input_chunk_length=len(self.target1),
        )
        with pytest.raises(ValueError) as exc:
            _ = ds[0]
        assert str(exc.value).startswith(
            "For the given forecasting case, the provided `past_covariates` at "
            "series sequence index `0` do not extend far enough into the past."
        )

        # past covs end too early
        ds = SequentialTorchInferenceDataset(
            series=self.target1,
            past_covariates=self.target1[:-1],
            input_chunk_length=len(self.target1),
        )
        with pytest.raises(ValueError) as exc:
            _ = ds[0]
        assert str(exc.value).startswith(
            "For the given forecasting horizon `n=1`, the provided `past_covariates` at "
            "series sequence index `0` do not extend far enough into the future."
        )

        # past covs start too late
        target_short = self.target1[:-1]
        ds = SequentialTorchInferenceDataset(
            series=target_short,
            future_covariates=self.target1[1:],
            input_chunk_length=len(target_short),
            output_chunk_length=1,
            n=1,
        )
        with pytest.raises(ValueError) as exc:
            _ = ds[0]
        assert str(exc.value).startswith(
            "For the given forecasting case, the provided `future_covariates` at "
            "series sequence index `0` do not extend far enough into the past."
        )

        # future covs end too early
        ds = SequentialTorchInferenceDataset(
            series=target_short,
            future_covariates=target_short,
            input_chunk_length=len(target_short),
            output_chunk_length=1,
            n=1,
        )
        with pytest.raises(ValueError) as exc:
            _ = ds[0]
        assert str(exc.value).startswith(
            "For the given forecasting horizon `n=1`, the provided `future_covariates` at "
            "series sequence index `0` do not extend far enough into the future."
        )

    def test_shifted_training_dataset_too_short(self):
        # one target series
        with pytest.raises(ValueError) as exc:
            _ = ShiftedTorchTrainingDataset(
                series=self.target1[:5],
                input_chunk_length=3,
                output_chunk_length=3,
                shift=3,
            )
        assert str(exc.value) == (
            "The input `series` are too short to extract even a single sample. "
            "Expected min length: `6`, received max length: `5`."
        )

        # two target series both too short, will hint at max length of both
        with pytest.raises(ValueError) as exc:
            _ = ShiftedTorchTrainingDataset(
                series=[self.target1[:3], self.target1[:4]],
                input_chunk_length=3,
                output_chunk_length=3,
                shift=3,
            )
        assert str(exc.value) == (
            "The input `series` are too short to extract even a single sample. "
            "Expected min length: `6`, received max length: `4`."
        )

        # two target series, first is long enough, second is too short;
        # error is raised only when going through the dataset
        ds = ShiftedTorchTrainingDataset(
            series=[self.target1[:6], self.target1[:5]],
            input_chunk_length=3,
            output_chunk_length=3,
            shift=3,
        )
        # first sample of first series is okay
        _ = ds[0]
        # first sample of second series failed
        with pytest.raises(ValueError) as exc:
            _ = ds[1]
        assert str(exc.value) == (
            "The dataset contains target `series` that are too short to extract "
            "even a single example. Expected min length: `6`, received length `5` "
            "(at series sequence idx `1`)."
        )

    def test_horizon_training_dataset_too_short(self):
        # two target series, first is long enough, second is too short;
        # horizon based only detects too short series when going through the dataset
        ds = HorizonBasedTorchTrainingDataset(
            series=[self.target1[:6], self.target1[:5]],
            output_chunk_length=3,
            lookback=1,
            lh=(1, 1),
        )
        # first sample of first series is okay
        _ = ds[0]
        # first sample of second series failed
        with pytest.raises(ValueError) as exc:
            _ = ds[1]
        assert str(exc.value) == (
            "The dataset contains target `series` that are too short to extract "
            "even a single example. Expected min length: `6`, received length `5` "
            "(at series sequence idx `1`)."
        )
        # dataset end
        with pytest.raises(IndexError):
            _ = ds[2]

    def test_horizon_training_dataset_invalid_lh(self):
        # lh elements must be >= 1
        with pytest.raises(ValueError) as exc:
            _ = HorizonBasedTorchTrainingDataset(
                series=self.target1,
                output_chunk_length=3,
                lookback=1,
                lh=(1, 0),
            )
        assert str(exc.value) == (
            "Invalid `lh=(1, 0)`. `lh` must be a tuple `(min_lh, max_lh)`, "
            "with `1 <= min_lh <= max_lh`."
        )

    def test_past_covariates_sequential_dataset(self):
        # one target series
        ds = SequentialTorchTrainingDataset(
            series=self.target1,
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 81
        self._assert_eq(
            ds[5],
            (
                self.target1[75:85],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )

        # two target series
        ds = SequentialTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 262
        self._assert_eq(
            ds[5],
            (
                self.target1[75:85],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[136],
            (
                self.target2[125:135],
                None,
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two target series with custom max_nr_samples
        ds = SequentialTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            max_samples_per_ts=50,
        )
        assert len(ds) == 100
        self._assert_eq(
            ds[5],
            (
                self.target1[75:85],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[55],
            (
                self.target2[125:135],
                None,
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two targets and one covariate
        with pytest.raises(ValueError) as exc:
            ds = SequentialTorchTrainingDataset(
                series=[self.target1, self.target2], past_covariates=[self.cov1]
            )
        assert str(exc.value) == (
            "The sequence of `past_covariates` must have the same length as the sequence of target `series`."
        )

        # two targets and two covariates
        ds = SequentialTorchTrainingDataset(
            series=[self.target1, self.target2],
            past_covariates=[self.cov1, self.cov2],
            input_chunk_length=10,
            output_chunk_length=10,
        )
        self._assert_eq(
            ds[5],
            (
                self.target1[75:85],
                self.cov1[75:85],
                None,
                None,
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
                None,
                None,
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
        ds = SequentialTorchTrainingDataset(
            series=target,
            past_covariates=cov,
            input_chunk_length=10,
            output_chunk_length=10,
        )
        with pytest.raises(ValueError) as exc:
            _ = ds[5]

        assert str(exc.value) == (
            "Invalid `past_covariates`; could not find values in index range: "
            "2010-12-08 00:00:00 - 2010-12-17 00:00:00."
        )

        # the same should fail when series are integer-indexed
        times1 = pd.RangeIndex(start=0, stop=100, step=1)
        times2 = pd.RangeIndex(start=200, stop=400, step=1)
        target = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        ds = SequentialTorchTrainingDataset(
            series=target,
            past_covariates=cov,
            input_chunk_length=10,
            output_chunk_length=10,
        )
        with pytest.raises(ValueError) as exc:
            _ = ds[5]
        assert str(exc.value) == (
            "Invalid `past_covariates`; could not find values in index range: 75 - 84."
        )

        # we should get the correct covariate slice even when target and covariates are not aligned
        times1 = pd.date_range(start="20100101", end="20110101", freq="D")
        times2 = pd.date_range(start="20090101", end="20110106", freq="D")
        target = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        ds = SequentialTorchTrainingDataset(
            series=target,
            past_covariates=cov,
            input_chunk_length=10,
            output_chunk_length=10,
        )

        np.testing.assert_almost_equal(ds[0][0], target.values()[-20:-10])
        np.testing.assert_almost_equal(ds[0][1], cov.values()[-25:-15])
        assert ds[0][2] is None  # historic future cov
        assert ds[0][3] is None  # future cov
        np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
        assert ds[0][5] is None  # sample weight
        np.testing.assert_almost_equal(ds[0][6], target.values()[-10:])

        np.testing.assert_almost_equal(ds[5][0], target.values()[-25:-15])
        np.testing.assert_almost_equal(ds[5][1], cov.values()[-30:-20])
        assert ds[5][2] is None  # historic future cov
        assert ds[5][3] is None  # future cov
        np.testing.assert_almost_equal(ds[5][4], self.cov_st2)
        assert ds[5][5] is None  # sample weight
        np.testing.assert_almost_equal(ds[5][6], target.values()[-15:-5])

        # This should also be the case when series are integer indexed
        times1 = pd.RangeIndex(start=100, stop=200, step=1)
        times2 = pd.RangeIndex(start=50, stop=250, step=1)
        target = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        ds = SequentialTorchTrainingDataset(
            series=target,
            past_covariates=cov,
            input_chunk_length=10,
            output_chunk_length=10,
        )

        np.testing.assert_almost_equal(ds[0][0], target.values()[-20:-10])
        np.testing.assert_almost_equal(ds[0][1], cov.values()[-70:-60])
        assert ds[0][2] is None  # historic future cov
        assert ds[0][3] is None  # future cov
        np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
        assert ds[0][5] is None  # sample weight
        np.testing.assert_almost_equal(ds[0][6], target.values()[-10:])

        np.testing.assert_almost_equal(ds[5][0], target.values()[-25:-15])
        np.testing.assert_almost_equal(ds[5][1], cov.values()[-75:-65])
        assert ds[5][2] is None  # historic future cov
        assert ds[5][3] is None  # future cov
        np.testing.assert_almost_equal(ds[5][4], self.cov_st2)
        assert ds[5][5] is None  # sample weight
        np.testing.assert_almost_equal(ds[5][6], target.values()[-15:-5])

    def test_future_covariates_sequential_dataset(self):
        # one target series
        ds = SequentialTorchTrainingDataset(
            series=self.target1,
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 81
        self._assert_eq(
            ds[5],
            (
                self.target1[75:85],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )

        # two target series
        ds = SequentialTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 262
        self._assert_eq(
            ds[5],
            (
                self.target1[75:85],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[136],
            (
                self.target2[125:135],
                None,
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two target series with custom max_nr_samples
        ds = SequentialTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            max_samples_per_ts=50,
        )
        assert len(ds) == 100
        self._assert_eq(
            ds[5],
            (
                self.target1[75:85],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[55],
            (
                self.target2[125:135],
                None,
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two targets and one covariate
        with pytest.raises(ValueError) as exc:
            ds = SequentialTorchTrainingDataset(
                series=[self.target1, self.target2],
                future_covariates=[self.cov1],
            )
        assert str(exc.value) == (
            "The sequence of `future_covariates` must have the same length as the sequence of target `series`."
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

        ds = SequentialTorchTrainingDataset(
            series=[target1, target2],
            future_covariates=[cov1, cov2],
            input_chunk_length=10,
            output_chunk_length=10,
        )

        np.testing.assert_almost_equal(ds[0][0], target1.values()[-20:-10])
        assert ds[0][1] is None  # past cov
        np.testing.assert_almost_equal(ds[0][2], cov1.values()[-40:-30])
        np.testing.assert_almost_equal(ds[0][3], cov1.values()[-30:-20])
        np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
        assert ds[0][5] is None  # sample weight
        np.testing.assert_almost_equal(ds[0][6], target1.values()[-10:])

        np.testing.assert_almost_equal(ds[101][0], target2.values()[-40:-30])
        assert ds[0][1] is None  # past cov
        np.testing.assert_almost_equal(ds[101][2], cov2.values()[-70:-60])
        np.testing.assert_almost_equal(ds[101][3], cov2.values()[-60:-50])
        np.testing.assert_almost_equal(ds[101][4], self.cov_st2)
        assert ds[0][5] is None
        np.testing.assert_almost_equal(ds[101][6], target2.values()[-30:-20])

        # Should also contain correct values when time-indexed with covariates not aligned
        times1 = pd.date_range(start="20090201", end="20090220", freq="D")
        times2 = pd.date_range(start="20090201", end="20090222", freq="D")
        target1 = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov1 = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))

        ds = SequentialTorchTrainingDataset(
            series=[target1],
            future_covariates=[cov1],
            input_chunk_length=2,
            output_chunk_length=2,
        )

        np.testing.assert_almost_equal(ds[0][0], target1.values()[-4:-2])
        assert ds[0][1] is None
        np.testing.assert_almost_equal(ds[0][2], cov1.values()[-6:-4])
        np.testing.assert_almost_equal(ds[0][3], cov1.values()[-4:-2])
        np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
        assert ds[0][5] is None
        np.testing.assert_almost_equal(ds[0][6], target1.values()[-2:])

        # Should fail if covariates are not long enough
        target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
            self.cov_st2_df
        )
        cov1 = TimeSeries.from_values(np.random.randn(7))

        ds = SequentialTorchTrainingDataset(
            series=[target1],
            future_covariates=[cov1],
            input_chunk_length=2,
            output_chunk_length=2,
        )

        with pytest.raises(ValueError) as exc:
            _ = ds[0]
        assert (
            str(exc.value)
            == "Invalid `future_covariates`; could not find values in index range: 6 - 7."
        )

    def test_dual_covariates_sequential_dataset(self):
        # Must contain (past_target, historic_future_covariates, future_covariates, static covariates,
        # sample weight, future_target)

        # one target series
        ds = SequentialTorchTrainingDataset(
            series=self.target1,
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 81
        self._assert_eq(
            ds[5],
            (
                self.target1[75:85],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )

        # two target series
        ds = SequentialTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 262
        self._assert_eq(
            ds[5],
            (
                self.target1[75:85],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[136],
            (
                self.target2[125:135],
                None,
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two target series with custom max_nr_samples
        ds = SequentialTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            max_samples_per_ts=50,
        )
        assert len(ds) == 100
        self._assert_eq(
            ds[5],
            (
                self.target1[75:85],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[55],
            (
                self.target2[125:135],
                None,
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two targets and one covariate
        with pytest.raises(ValueError):
            ds = SequentialTorchTrainingDataset(
                series=[self.target1, self.target2],
                future_covariates=[self.cov1],
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

        ds = SequentialTorchTrainingDataset(
            series=[target1, target2],
            future_covariates=[cov1, cov2],
            input_chunk_length=10,
            output_chunk_length=10,
        )

        np.testing.assert_almost_equal(ds[0][0], target1.values()[-20:-10])
        assert ds[0][1] is None
        np.testing.assert_almost_equal(ds[0][2], cov1.values()[-40:-30])
        np.testing.assert_almost_equal(ds[0][3], cov1.values()[-30:-20])
        np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
        assert ds[0][5] is None
        np.testing.assert_almost_equal(ds[0][6], target1.values()[-10:])

        np.testing.assert_almost_equal(ds[101][0], target2.values()[-40:-30])
        assert ds[0][1] is None
        np.testing.assert_almost_equal(ds[101][2], cov2.values()[-70:-60])
        np.testing.assert_almost_equal(ds[101][3], cov2.values()[-60:-50])
        np.testing.assert_almost_equal(ds[101][4], self.cov_st2)
        assert ds[101][5] is None
        np.testing.assert_almost_equal(ds[101][6], target2.values()[-30:-20])

        # Should also contain correct values when time-indexed with covariates not aligned
        times1 = pd.date_range(start="20090201", end="20090220", freq="D")
        times2 = pd.date_range(start="20090201", end="20090222", freq="D")
        target1 = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov1 = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))

        ds = SequentialTorchTrainingDataset(
            series=[target1],
            future_covariates=[cov1],
            input_chunk_length=2,
            output_chunk_length=2,
        )

        np.testing.assert_almost_equal(ds[0][0], target1.values()[-4:-2])
        assert ds[0][1] is None
        np.testing.assert_almost_equal(ds[0][2], cov1.values()[-6:-4])
        np.testing.assert_almost_equal(ds[0][3], cov1.values()[-4:-2])
        np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
        assert ds[0][5] is None
        np.testing.assert_almost_equal(ds[0][6], target1.values()[-2:])

        # Should fail if covariates are not long enough
        target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
            self.cov_st2_df
        )
        cov1 = TimeSeries.from_values(np.random.randn(7))

        ds = SequentialTorchTrainingDataset(
            series=[target1],
            future_covariates=[cov1],
            input_chunk_length=2,
            output_chunk_length=2,
        )

        with pytest.raises(ValueError):
            _ = ds[0]

    def test_past_covariates_shifted_dataset(self):
        # one target series
        ds = ShiftedTorchTrainingDataset(
            series=self.target1,
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
        )
        assert len(ds) == 86
        self._assert_eq(
            ds[5],
            (
                self.target1[80:90],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )

        # two target series
        ds = ShiftedTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
        )
        assert len(ds) == 272
        self._assert_eq(
            ds[5],
            (
                self.target1[80:90],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[141],
            (
                self.target2[130:140],
                None,
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two target series with custom max_nr_samples
        ds = ShiftedTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
            max_samples_per_ts=50,
        )
        assert len(ds) == 100
        self._assert_eq(
            ds[5],
            (
                self.target1[80:90],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[55],
            (
                self.target2[130:140],
                None,
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two targets and one covariate
        with pytest.raises(ValueError):
            ds = ShiftedTorchTrainingDataset(
                series=[self.target1, self.target2], past_covariates=[self.cov1]
            )

        # covariates end too early
        chunk_length = 3
        series = self.target1[: 2 * chunk_length]
        ds = ShiftedTorchTrainingDataset(
            series=series,
            past_covariates=series[: -(chunk_length + 1)],
            input_chunk_length=chunk_length,
            output_chunk_length=chunk_length,
            shift=chunk_length,
        )
        with pytest.raises(ValueError) as exc:
            _ = ds[0]
        assert str(exc.value) == (
            "Invalid `past_covariates`; could not find values in index range: "
            "2000-01-01 00:00:00 - 2000-01-03 00:00:00."
        )

        # covariates are long enough but don't have the same frequency
        ds = ShiftedTorchTrainingDataset(
            series=series,
            past_covariates=self.target1[::2],
            input_chunk_length=chunk_length,
            output_chunk_length=chunk_length,
            shift=chunk_length,
        )
        with pytest.raises(ValueError) as exc:
            _ = ds[0]
        assert str(exc.value) == (
            "The `past_covariates` frequency `<2 * Days>` does not match "
            "the target `series` frequency `<Day>` (at series sequence idx `0`)."
        )

        # two targets and two covariates
        ds = ShiftedTorchTrainingDataset(
            series=[self.target1, self.target2],
            past_covariates=[self.cov1, self.cov2],
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
        )
        self._assert_eq(
            ds[5],
            (
                self.target1[80:90],
                self.cov1[80:90],
                None,
                None,
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
                None,
                None,
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
        ds = ShiftedTorchTrainingDataset(
            series=[target1],
            past_covariates=[cov1],
            input_chunk_length=3,
            output_chunk_length=3,
            shift=2,
        )
        np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0][1], cov1.values()[-7:-4])
        assert ds[0][2] is None
        assert ds[0][3] is None
        np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
        assert ds[0][5] is None
        np.testing.assert_almost_equal(ds[0][6], target1.values()[-3:])

        # Should also contain correct values when time-indexed with covariates not aligned
        times1 = pd.date_range(start="20090201", end="20090220", freq="D")
        times2 = pd.date_range(start="20090201", end="20090222", freq="D")
        target1 = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov1 = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        ds = ShiftedTorchTrainingDataset(
            series=[target1],
            past_covariates=[cov1],
            input_chunk_length=3,
            output_chunk_length=3,
            shift=2,
        )
        np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0][1], cov1.values()[-7:-4])
        assert ds[0][2] is None
        assert ds[0][3] is None
        np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
        assert ds[0][5] is None
        np.testing.assert_almost_equal(ds[0][6], target1.values()[-3:])

        # Should fail if covariates are too short
        target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
            self.cov_st2_df
        )
        cov1 = TimeSeries.from_values(np.random.randn(5))
        ds = ShiftedTorchTrainingDataset(
            series=[target1],
            past_covariates=[cov1],
            input_chunk_length=3,
            output_chunk_length=3,
            shift=2,
        )
        with pytest.raises(ValueError):
            _ = ds[0]

    def test_future_covariates_shifted_dataset(self):
        # one target series
        ds = ShiftedTorchTrainingDataset(
            series=self.target1,
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
        )
        assert len(ds) == 86
        self._assert_eq(
            ds[5],
            (
                self.target1[80:90],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )

        # two target series
        ds = ShiftedTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
        )
        assert len(ds) == 272
        self._assert_eq(
            ds[5],
            (
                self.target1[80:90],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[141],
            (
                self.target2[130:140],
                None,
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two target series with custom max_nr_samples
        ds = ShiftedTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
            max_samples_per_ts=50,
        )
        assert len(ds) == 100
        self._assert_eq(
            ds[5],
            (
                self.target1[80:90],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[55],
            (
                self.target2[130:140],
                None,
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two targets and one covariate
        with pytest.raises(ValueError):
            ds = ShiftedTorchTrainingDataset(
                series=[self.target1, self.target2],
                future_covariates=[self.cov1],
            )

        # covariates end too early
        chunk_length = 3
        series = self.target1[: 2 * chunk_length]
        ds = ShiftedTorchTrainingDataset(
            series=series,
            future_covariates=series[:-1],
            input_chunk_length=chunk_length,
            output_chunk_length=chunk_length,
            shift=chunk_length,
        )
        with pytest.raises(ValueError) as exc:
            _ = ds[0]
        assert str(exc.value) == (
            "Invalid `future_covariates`; could not find values in index range: "
            "2000-01-04 00:00:00 - 2000-01-06 00:00:00."
        )

        # covariates are long enough but don't have the same frequency
        ds = ShiftedTorchTrainingDataset(
            series=series,
            future_covariates=self.target1[::2],
            input_chunk_length=chunk_length,
            output_chunk_length=chunk_length,
            shift=chunk_length,
        )
        with pytest.raises(ValueError) as exc:
            _ = ds[0]
        assert str(exc.value) == (
            "The `future_covariates` frequency `<2 * Days>` does not match "
            "the target `series` frequency `<Day>` (at series sequence idx `0`)."
        )

        # two targets and two covariates
        ds = ShiftedTorchTrainingDataset(
            series=[self.target1, self.target2],
            future_covariates=[self.cov1, self.cov2],
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
        )
        self._assert_eq(
            ds[5],
            (
                self.target1[80:90],
                None,
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
                None,
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
        ds = ShiftedTorchTrainingDataset(
            series=[target1],
            future_covariates=[cov1],
            input_chunk_length=3,
            output_chunk_length=3,
            shift=2,
        )
        np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
        assert ds[0][1] is None
        np.testing.assert_almost_equal(ds[0][2], cov1.values()[-7:-4])
        np.testing.assert_almost_equal(ds[0][3], cov1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
        assert ds[0][5] is None
        np.testing.assert_almost_equal(ds[0][6], target1.values()[-3:])

        # Should also contain correct values when time-indexed with covariates not aligned
        times1 = pd.date_range(start="20090201", end="20090220", freq="D")
        times2 = pd.date_range(start="20090201", end="20090222", freq="D")
        target1 = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov1 = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        ds = ShiftedTorchTrainingDataset(
            series=[target1],
            future_covariates=[cov1],
            input_chunk_length=3,
            output_chunk_length=3,
            shift=2,
        )
        np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
        assert ds[0][1] is None
        np.testing.assert_almost_equal(ds[0][2], cov1.values()[-7:-4])
        np.testing.assert_almost_equal(ds[0][3], cov1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
        assert ds[0][5] is None
        np.testing.assert_almost_equal(ds[0][6], target1.values()[-3:])

        # Should fail if covariates are too short
        target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
            self.cov_st2_df
        )
        cov1 = TimeSeries.from_values(np.random.randn(7))
        ds = ShiftedTorchTrainingDataset(
            series=[target1],
            future_covariates=[cov1],
            input_chunk_length=3,
            output_chunk_length=3,
            shift=2,
        )
        with pytest.raises(ValueError):
            _ = ds[0]

    def test_dual_covariates_shifted_dataset(self):
        # one target series
        ds = ShiftedTorchTrainingDataset(
            series=self.target1,
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
        )
        assert len(ds) == 86
        self._assert_eq(
            ds[5],
            (
                self.target1[80:90],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )

        # two target series
        ds = ShiftedTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
        )
        assert len(ds) == 272
        self._assert_eq(
            ds[5],
            (
                self.target1[80:90],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[141],
            (
                self.target2[130:140],
                None,
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two target series with custom max_nr_samples
        ds = ShiftedTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
            max_samples_per_ts=50,
        )
        assert len(ds) == 100
        self._assert_eq(
            ds[5],
            (
                self.target1[80:90],
                None,
                None,
                None,
                self.cov_st1,
                None,
                self.target1[85:95],
            ),
        )
        self._assert_eq(
            ds[55],
            (
                self.target2[130:140],
                None,
                None,
                None,
                self.cov_st2,
                None,
                self.target2[135:145],
            ),
        )

        # two targets and one covariate
        with pytest.raises(ValueError):
            ds = ShiftedTorchTrainingDataset(
                series=[self.target1, self.target2],
                future_covariates=[self.cov1],
            )

        # two targets and two covariates
        ds = ShiftedTorchTrainingDataset(
            series=[self.target1, self.target2],
            future_covariates=[self.cov1, self.cov2],
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
        )
        self._assert_eq(
            ds[5],
            (
                self.target1[80:90],
                None,
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
                None,
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
        ds = ShiftedTorchTrainingDataset(
            series=[target1],
            future_covariates=[cov1],
            input_chunk_length=3,
            output_chunk_length=3,
            shift=2,
        )
        np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
        assert ds[0][1] is None
        np.testing.assert_almost_equal(ds[0][2], cov1.values()[-7:-4])
        np.testing.assert_almost_equal(ds[0][3], cov1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
        assert ds[0][5] is None
        np.testing.assert_almost_equal(ds[0][6], target1.values()[-3:])

        # Should also contain correct values when time-indexed with covariates not aligned
        times1 = pd.date_range(start="20090201", end="20090220", freq="D")
        times2 = pd.date_range(start="20090201", end="20090222", freq="D")
        target1 = TimeSeries.from_times_and_values(
            times1, np.random.randn(len(times1))
        ).with_static_covariates(self.cov_st2_df)
        cov1 = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        ds = ShiftedTorchTrainingDataset(
            series=[target1],
            future_covariates=[cov1],
            input_chunk_length=3,
            output_chunk_length=3,
            shift=2,
        )
        np.testing.assert_almost_equal(ds[0][0], target1.values()[-5:-2])
        assert ds[0][1] is None
        np.testing.assert_almost_equal(ds[0][2], cov1.values()[-7:-4])
        np.testing.assert_almost_equal(ds[0][3], cov1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0][4], self.cov_st2)
        assert ds[0][5] is None
        np.testing.assert_almost_equal(ds[0][6], target1.values()[-3:])

        # Should fail if covariates are too short
        target1 = TimeSeries.from_values(np.random.randn(8)).with_static_covariates(
            self.cov_st2_df
        )
        cov1 = TimeSeries.from_values(np.random.randn(7))
        ds = ShiftedTorchTrainingDataset(
            series=[target1],
            future_covariates=[cov1],
            input_chunk_length=3,
            output_chunk_length=3,
            shift=2,
        )
        with pytest.raises(ValueError):
            _ = ds[0]

    @pytest.mark.parametrize("use_weight", [False, True])
    def test_horizon_based_dataset(self, use_weight):
        ds_kwargs = {
            "output_chunk_length": 10,
            "lh": (1, 3),
            "lookback": 2,
        }
        weight1 = self.target1 + 1
        weight2 = self.target2 + 1

        weight_exp = weight1[85:95] if use_weight else None
        # one target series
        ds_kwargs["series"] = self.target1
        ds_kwargs["sample_weight"] = weight1 if use_weight else None
        ds = HorizonBasedTorchTrainingDataset(**ds_kwargs)
        # 21 as both `lh` bounds are inclusive
        assert len(ds) == 21
        self._assert_eq(
            ds[5],
            (
                self.target1[65:85],
                None,
                None,
                None,
                self.cov_st1,
                weight_exp,
                self.target1[85:95],
            ),
        )
        # one target series, with stride
        ds_stride = HorizonBasedTorchTrainingDataset(**ds_kwargs, stride=3)
        self._check_ds_stride(ds_regular=ds, ds_stride=ds_stride, stride=3)

        # two target series
        weight_exp1 = weight1[85:95] if use_weight else None
        weight_exp2 = weight2[135:145] if use_weight else None
        ds_kwargs["series"] = [self.target1, self.target2]
        ds_kwargs["sample_weight"] = [weight1, weight2] if use_weight else None
        ds = HorizonBasedTorchTrainingDataset(**ds_kwargs)
        # 42 as both `lh` bounds are inclusive per series
        assert len(ds) == 42
        self._assert_eq(
            ds[5],
            (
                self.target1[65:85],
                None,
                None,
                None,
                self.cov_st1,
                weight_exp1,
                self.target1[85:95],
            ),
        )
        # 21 samples after comes the second series
        self._assert_eq(
            ds[26],
            (
                self.target2[115:135],
                None,
                None,
                None,
                self.cov_st2,
                weight_exp2,
                self.target2[135:145],
            ),
        )
        # two target series, with stride
        ds_stride = HorizonBasedTorchTrainingDataset(**ds_kwargs, stride=3)
        self._check_ds_stride(ds_regular=ds, ds_stride=ds_stride, stride=3)

        # two targets and one covariate
        with pytest.raises(ValueError):
            ds = HorizonBasedTorchTrainingDataset(
                series=[self.target1, self.target2], past_covariates=[self.cov1]
            )

        # two targets and two covariates
        weight_exp1 = weight1[85:95] if use_weight else None
        weight_exp2 = weight2[135:145] if use_weight else None
        ds_kwargs["series"] = [self.target1, self.target2]
        ds_kwargs["past_covariates"] = [self.cov1, self.cov2]
        ds_kwargs["sample_weight"] = [weight1, weight2] if use_weight else None
        ds = HorizonBasedTorchTrainingDataset(**ds_kwargs)
        self._assert_eq(
            ds[5],
            (
                self.target1[65:85],
                self.cov1[65:85],
                None,
                None,
                self.cov_st1,
                weight_exp1,
                self.target1[85:95],
            ),
        )
        # 21 samples after comes the second series
        self._assert_eq(
            ds[26],
            (
                self.target2[115:135],
                self.cov2[115:135],
                None,
                None,
                self.cov_st2,
                weight_exp2,
                self.target2[135:145],
            ),
        )
        # two targets and two covariates, with stride
        ds_stride = HorizonBasedTorchTrainingDataset(**ds_kwargs, stride=3)
        self._check_ds_stride(ds_regular=ds, ds_stride=ds_stride, stride=3)

    @pytest.mark.parametrize(
        "config",
        [
            # (dataset class, whether contains future, future batch index)
            (SequentialTorchTrainingDataset, [], None),
            (SequentialTorchTrainingDataset, ["past"], None),
            (SequentialTorchTrainingDataset, ["future"], 3),
            (SequentialTorchTrainingDataset, ["past", "future"], 3),
        ],
    )
    def test_sequential_training_dataset_output_chunk_shift(self, config):
        ds_cls, use_covs, future_idx = config
        ocl = 1
        ocs = 2
        target = self.target1[: -(ocl + ocs)]
        sample_weight = target + 1

        ds_covs = {}
        for cov_type in use_covs:
            ds_covs[cov_type + "_covariates"] = self.cov1

        # regular dataset with output shift=0 and ocl=3: the 3rd future values should be identical to the 1st future
        # values of a dataset with output shift=2 and ocl=1
        ds_reg = ds_cls(
            series=target,
            input_chunk_length=1,
            output_chunk_length=3,
            output_chunk_shift=0,
            sample_weight=sample_weight,
            **ds_covs,
        )

        ds_shift = ds_cls(
            series=target,
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
                (SequentialTorchTrainingDataset, []),
                (SequentialTorchTrainingDataset, ["past"]),
                (SequentialTorchTrainingDataset, ["future"]),
                (SequentialTorchTrainingDataset, ["past", "future"]),
            ],
            [True, False],
        ),
    )
    def test_sequential_training_dataset_weight(self, config):
        (ds_cls, use_covs), manual_weight = config

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
        for cov_type in use_covs:
            ds_covs[cov_type + "_covariates"] = self.cov1

        # no sample weight
        ds = ds_cls(
            series=target1,
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
            series=target,
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
            series=target,
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
            series=target,
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
            series=target,
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
            series=target,
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
            series=target,
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
            series=target,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=weight,
            **{k: [v] * 2 for k, v in ds_covs.items()},
        )
        weight_exp = ds[0][-1] + 1 if manual_weight else get_built_in_weigths(target)
        assert np.all(ds[0][-2] == weight_exp)

    def test_sequential_training_dataset_invalid_weight(self):
        ds_cls = SequentialTorchTrainingDataset
        ts = self.target1

        # invalid built-in weight
        with pytest.raises(ValueError) as err:
            _ = ds_cls(
                series=[ts, ts],
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
                series=[ts, ts],
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
            series=ts,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=ts.stack(ts + 1),
        )
        with pytest.raises(ValueError) as err:
            _ = ds[0]
        assert (
            str(err.value)
            == "The number of components in `sample_weight` must either be `1` or match "
            "the number of target series components `1` (at series sequence idx `0`)."
        )

        # weight too short end
        ds = ds_cls(
            series=ts,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=ts[:-1],
        )
        with pytest.raises(ValueError) as err:
            _ = ds[0]
        assert (
            str(err.value)
            == "Invalid `sample_weight`; could not find values in index range: "
            "2000-04-07 00:00:00 - 2000-04-09 00:00:00."
        )

        # weight too short start
        ds = ds_cls(
            series=ts,
            input_chunk_length=1,
            output_chunk_length=3,
            sample_weight=ts[2:],
        )
        with pytest.raises(ValueError) as err:
            _ = ds[len(ds) - 1]
        assert (
            str(err.value)
            == "Invalid `sample_weight`; could not find values in index range: "
            "2000-01-02 00:00:00 - 2000-01-04 00:00:00."
        )

    @pytest.mark.parametrize(
        "config",
        [
            (SequentialTorchTrainingDataset, []),
            (SequentialTorchTrainingDataset, ["past"]),
            (SequentialTorchTrainingDataset, ["future"]),
            (SequentialTorchTrainingDataset, ["past", "future"]),
        ],
    )
    def test_sequential_training_dataset_stride(self, config):
        ds_cls, use_covs = config

        ds_covs = {}
        for cov_type in use_covs:
            ds_covs[cov_type + "_covariates"] = self.cov1

        ds_cls, future_idx = config
        icl = 4
        ocl = 2
        nb_samples = 12
        target = self.target1[: icl + ocl + nb_samples - 1]

        ds_reg = ds_cls(
            series=target,
            input_chunk_length=icl,
            output_chunk_length=ocl,
            stride=1,
            **ds_covs,
        )

        ds_stride = ds_cls(
            series=target,
            input_chunk_length=icl,
            output_chunk_length=ocl,
            stride=3,
            **ds_covs,
        )
        assert len(ds_stride) * 3 == len(ds_reg) == nb_samples
        self._check_ds_stride(ds_regular=ds_reg, ds_stride=ds_stride, stride=3)

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
