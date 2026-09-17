import math
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries, concatenate
from darts.tests.conftest import TORCH_AVAILABLE
from darts.utils.timeseries_generation import gaussian_timeseries, linear_timeseries

if not TORCH_AVAILABLE:
    pytest.skip(
        f"Torch not available. {__name__} tests will be skipped.",
        allow_module_level=True,
    )

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from darts.tests.parametrize_helpers import param_product
from darts.utils.data import (
    HorizonBasedTorchTrainingDataset,
    SequentialTorchInferenceDataset,
    SequentialTorchTrainingDataset,
    ShiftedTorchTrainingDataset,
    TorchInferenceDataset,
    TorchTrainingDataset,
)
from darts.utils.data.torch_datasets.utils import (
    ModuleStage,
    PLModuleInput,
    PLModuleOutput,
    TorchInferenceBatch,
    TorchInferenceSample,
    TorchTrainingBatch,
    TorchTrainingSample,
    _as_inference_sample,
    _as_training_sample,
    _batch_collate_fn_predict,
    _batch_collate_fn_train,
    _coerce_training_sample,
    _train_sample_from_shapes,
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

    def _assert_eq_schema(self, left: dict, right: dict):
        assert all([
            (v == right[k] if not isinstance(v, pd.DataFrame) else v.equals(right[k]))
            for k, v in left.items()
        ])

    def _assert_field(self, actual, expected):
        expected = expected.values() if isinstance(expected, TimeSeries) else expected
        if expected is None:
            assert actual is None
            return
        if isinstance(expected, pd.Series | pd.DataFrame):
            assert actual.equals(expected)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self._assert_eq_schema(actual, expected)
        else:
            assert actual == expected

    def _assert_training_output(self, sample: TorchTrainingSample, **sample_expected):
        assert isinstance(sample, TorchTrainingSample)
        sample_expected = TorchTrainingSample(**sample_expected)
        for f in fields(sample):
            self._assert_field(
                getattr(sample, f.name), getattr(sample_expected, f.name)
            )

    def _assert_inference_output(self, sample: TorchInferenceSample, **sample_expected):
        assert isinstance(sample, TorchInferenceSample)
        sample_expected = TorchInferenceSample(**sample_expected)
        for f in fields(sample):
            self._assert_field(
                getattr(sample, f.name), getattr(sample_expected, f.name)
            )

    def _check_ds_stride(self, ds_regular, ds_stride, stride: int):
        """
        Every `stride`-th values in a dataset with stride=1 should be identical to the dataset stridden with `stride
        """
        # if the un-stridden length is a multiple of the stride
        if len(ds_regular) % stride == 0:
            assert len(ds_regular) == len(ds_stride) * stride

        for idx, sample_stride in enumerate(ds_stride):
            sample_regular = ds_regular[idx * stride]
            assert type(sample_stride) is type(sample_regular)
            for f in fields(sample_stride):
                name = f.name
                self._assert_field(
                    getattr(sample_stride, name), getattr(sample_regular, name)
                )

    def test_past_covariates_inference_dataset(self):
        # one target series
        ds = SequentialTorchInferenceDataset(
            series=self.target1, input_chunk_length=len(self.target1)
        )
        self._assert_inference_output(
            ds[0],
            past_target=self.vals1,
            past_covariates=None,
            future_past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            series_schema=self.target1.schema(),
            pred_time=self.target1.end_time() + self.target1.freq,
        )

        # two target series
        ds = SequentialTorchInferenceDataset(
            series=[self.target1, self.target2],
            input_chunk_length=max(len(self.target1), len(self.target2)),
        )
        self._assert_inference_output(
            ds[1],
            past_target=self.vals2,
            past_covariates=None,
            future_past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            series_schema=self.target2.schema(),
            pred_time=self.target2.end_time() + self.target2.freq,
        )

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
        self._assert_inference_output(
            ds[1],
            past_target=self.vals2,
            past_covariates=self.cov2.values(),
            future_past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            series_schema=self.target2.schema(),
            pred_time=self.target2.end_time() + self.target2.freq,
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

        np.testing.assert_almost_equal(ds[0].past_target, target.values()[-10:])
        np.testing.assert_almost_equal(
            ds[0].past_covariates, long_cov.values()[-60:-50]
        )
        np.testing.assert_almost_equal(
            ds[0].future_past_covariates, long_cov.values()[-50:-30]
        )
        assert ds[0].historic_future_covariates is None
        assert ds[0].future_covariates is None
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        self._assert_eq_schema(ds[0].series_schema, target.schema())

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

        np.testing.assert_almost_equal(ds[0].past_target, target.values()[-10:])
        np.testing.assert_almost_equal(ds[0].past_covariates, covariate.values()[20:30])
        np.testing.assert_almost_equal(
            ds[0].future_past_covariates, covariate.values()[30:40]
        )
        assert ds[0].historic_future_covariates is None
        assert ds[0].future_covariates is None
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        self._assert_eq_schema(ds[0].series_schema, target.schema())

    def test_future_covariates_inference_dataset(self):
        # one target series
        ds = SequentialTorchInferenceDataset(
            series=self.target1, input_chunk_length=len(self.target1)
        )
        self._assert_inference_output(
            ds[0],
            past_target=self.vals1,
            past_covariates=None,
            future_past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            series_schema=self.target1.schema(),
            pred_time=self.target1.end_time() + self.target1.freq,
        )

        # two target series
        ds = SequentialTorchInferenceDataset(
            series=[self.target1, self.target2],
            input_chunk_length=max(len(self.target1), len(self.target2)),
        )
        self._assert_inference_output(
            ds[1],
            past_target=self.vals2,
            past_covariates=None,
            future_past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            series_schema=self.target2.schema(),
            pred_time=self.target2.end_time() + self.target2.freq,
        )

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

        np.testing.assert_almost_equal(ds[0].past_target, target.values()[-10:])
        assert ds[0].past_covariates is None
        assert ds[0].future_past_covariates is None
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, long_cov.values()[-60:-50]
        )
        np.testing.assert_almost_equal(
            ds[0].future_covariates, long_cov.values()[-50:-20]
        )
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        self._assert_eq_schema(ds[0].series_schema, target.schema())

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

        np.testing.assert_almost_equal(ds[0].past_target, target.values()[-10:])
        assert ds[0].past_covariates is None
        assert ds[0].future_past_covariates is None
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, covariate.values()[20:30]
        )
        np.testing.assert_almost_equal(
            ds[0].future_covariates, covariate.values()[30:50]
        )
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        self._assert_eq_schema(ds[0].series_schema, target.schema())

    def test_dual_covariates_inference_dataset(self):
        # one target series
        ds = SequentialTorchInferenceDataset(
            series=self.target1, input_chunk_length=len(self.target1)
        )
        self._assert_inference_output(
            ds[0],
            past_target=self.vals1,
            past_covariates=None,
            future_past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            series_schema=self.target1.schema(),
            pred_time=self.target1.end_time() + self.target1.freq,
        )

        # two target series
        ds = SequentialTorchInferenceDataset(
            series=[self.target1, self.target2],
            input_chunk_length=max(len(self.target1), len(self.target2)),
        )
        self._assert_inference_output(
            ds[1],
            past_target=self.vals2,
            past_covariates=None,
            future_past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            series_schema=self.target2.schema(),
            pred_time=self.target2.end_time() + self.target2.freq,
        )

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

        np.testing.assert_almost_equal(ds[0].past_target, target.values()[-10:])
        assert ds[0].past_covariates is None
        assert ds[0].future_past_covariates is None
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, long_cov.values()[-60:-50]
        )
        np.testing.assert_almost_equal(
            ds[0].future_covariates, long_cov.values()[-50:-20]
        )
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        self._assert_eq_schema(ds[0].series_schema, target.schema())

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

        np.testing.assert_almost_equal(ds[0].past_target, target.values()[-10:])
        assert ds[0].past_covariates is None
        assert ds[0].future_past_covariates is None
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, covariate.values()[20:30]
        )
        np.testing.assert_almost_equal(
            ds[0].future_covariates, covariate.values()[30:50]
        )
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        self._assert_eq_schema(ds[0].series_schema, target.schema())

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
        np.testing.assert_almost_equal(ds[0].past_target, target.values()[-10:])
        np.testing.assert_almost_equal(
            ds[0].past_covariates, long_past_cov.values()[-60:-50]
        )
        np.testing.assert_almost_equal(
            ds[0].future_past_covariates, long_past_cov.values()[-50:-30]
        )
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, future_cov.values()[-60:-50]
        )
        np.testing.assert_almost_equal(
            ds[0].future_covariates, future_cov.values()[-50:-20]
        )
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        self._assert_eq_schema(ds[0].series_schema, target.schema())

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

        np.testing.assert_almost_equal(ds[0].past_target, target.values()[-10:])
        np.testing.assert_almost_equal(ds[0].past_covariates, past_cov.values()[20:30])
        np.testing.assert_almost_equal(
            ds[0].future_past_covariates, past_cov.values()[30:40]
        )
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, future_cov.values()[10:20]
        )
        np.testing.assert_almost_equal(
            ds[0].future_covariates, future_cov.values()[20:40]
        )
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        self._assert_eq_schema(ds[0].series_schema, target.schema())

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
        np.testing.assert_almost_equal(ds[0].past_target, target.values()[-10:])
        np.testing.assert_almost_equal(
            ds[0].past_covariates, long_past_cov.values()[-60:-50]
        )
        np.testing.assert_almost_equal(
            ds[0].future_past_covariates, long_past_cov.values()[-50:-30]
        )
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, future_cov.values()[-60:-50]
        )
        np.testing.assert_almost_equal(
            ds[0].future_covariates, future_cov.values()[-50:-20]
        )
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        self._assert_eq_schema(ds[0].series_schema, target.schema())

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

        np.testing.assert_almost_equal(ds[0].past_target, target.values()[-10:])
        np.testing.assert_almost_equal(ds[0].past_covariates, past_cov.values()[20:30])
        np.testing.assert_almost_equal(
            ds[0].future_past_covariates, past_cov.values()[30:40]
        )
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, future_cov.values()[10:20]
        )
        np.testing.assert_almost_equal(
            ds[0].future_covariates, future_cov.values()[20:40]
        )
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        self._assert_eq_schema(ds[0].series_schema, target.schema())

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
        assert isinstance(batch_reg, TorchInferenceSample)
        assert isinstance(batch_shift, TorchInferenceSample)

        # shifted prediction starts 2 steps after regular prediction
        assert batch_reg.pred_time == batch_shift.pred_time - ocs * target.freq

        if future_idx is not None:
            # 3rd future values of regular ds must be identical to the 1st future values of shifted dataset
            np.testing.assert_array_equal(
                batch_reg.future_covariates[ocs:], batch_shift.future_covariates
            )

        skip = {"pred_time", "series_schema"}
        if future_idx is not None:
            skip.add("future_covariates")
        for f in fields(batch_reg):
            name = f.name
            if name in skip:
                continue
            el_reg, el_shift = getattr(batch_reg, name), getattr(batch_shift, name)
            if el_reg is None:
                assert el_shift is None
            else:
                np.testing.assert_array_equal(el_reg, el_shift)
        self._assert_eq_schema(batch_reg.series_schema, batch_shift.series_schema)

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
        np.testing.assert_array_almost_equal(
            ds[0].past_target, self.target1.values()[:3]
        )
        self._assert_eq_schema(ds[0].series_schema, self.target1.schema())
        assert ds[0].pred_time == self.target1._time_index[3]

        np.testing.assert_array_almost_equal(
            ds[1].past_target, self.target1.values()[1:4]
        )
        self._assert_eq_schema(ds[1].series_schema, self.target1.schema())
        assert ds[1].pred_time == self.target1._time_index[4]

        # last two sample are from end of the target with stride 1
        np.testing.assert_array_almost_equal(
            ds[96].past_target, self.target1.values()[-4:-1]
        )
        self._assert_eq_schema(ds[96].series_schema, self.target1.schema())
        assert ds[96].pred_time == self.target1._time_index[-1]
        np.testing.assert_array_almost_equal(
            ds[97].past_target, self.target1.values()[-3:]
        )
        self._assert_eq_schema(ds[97].series_schema, self.target1.schema())
        assert ds[97].pred_time == self.target1._time_index[-1] + self.target1.freq

        # stride = 2, setting bounds upper limit as `100` can still only compute until `99` since starting
        # at `3` with stride
        ds = SequentialTorchInferenceDataset(
            series=self.target1, stride=2, bounds=np.array([[3, 100]]), **kwargs
        )

        # length 49
        assert len(ds) == math.ceil((100 - 3 + 1) / 2)
        # first two sample are from beginning of the target
        np.testing.assert_array_almost_equal(
            ds[0].past_target, self.target1.values()[:3]
        )
        assert ds[0].pred_time == self.target1._time_index[3]
        np.testing.assert_array_almost_equal(
            ds[1].past_target, self.target1.values()[2:5]
        )
        assert ds[1].pred_time == self.target1._time_index[5]
        # last two sample are from end of the target
        np.testing.assert_array_almost_equal(
            ds[47].past_target, self.target1.values()[-6:-3]
        )
        assert ds[47].pred_time == self.target1._time_index[-3]
        np.testing.assert_array_almost_equal(
            ds[48].past_target, self.target1.values()[-4:-1]
        )
        assert ds[48].pred_time == self.target1._time_index[-1]

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
        np.testing.assert_array_almost_equal(
            ds[0].past_target, self.target1.values()[:3]
        )
        assert ds[0].pred_time == self.target1._time_index[4]
        np.testing.assert_array_almost_equal(
            ds[1].past_target, self.target1.values()[2:5]
        )
        assert ds[1].pred_time == self.target1._time_index[6]
        np.testing.assert_array_almost_equal(
            ds[47].past_target, self.target1.values()[-6:-3]
        )
        assert ds[47].pred_time == self.target1._time_index[-2]
        np.testing.assert_array_almost_equal(
            ds[48].past_target, self.target1.values()[-4:-1]
        )
        assert ds[48].pred_time == self.target1._time_index[-1] + self.target1.freq

        # stride = 2, setting bounds upper limit as `101` will result in an index error for sample 50
        ds = SequentialTorchInferenceDataset(
            series=self.target1, stride=2, bounds=np.array([[3, 101]]), **kwargs
        )

        # length 50
        assert len(ds) == math.ceil((101 - 3 + 1) / 2)
        # getting the samples from before works
        np.testing.assert_array_almost_equal(
            ds[0].past_target, self.target1.values()[:3]
        )
        assert ds[0].pred_time == self.target1._time_index[3]
        np.testing.assert_array_almost_equal(
            ds[1].past_target, self.target1.values()[2:5]
        )
        assert ds[1].pred_time == self.target1._time_index[5]
        np.testing.assert_array_almost_equal(
            ds[47].past_target, self.target1.values()[-6:-3]
        )
        assert ds[47].pred_time == self.target1._time_index[-3]
        np.testing.assert_array_almost_equal(
            ds[48].past_target, self.target1.values()[-4:-1]
        )
        assert ds[48].pred_time == self.target1._time_index[-1]

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

    def test_max_samples_per_ts_upper_bound(self):
        # Use cov1 with length 100
        series = self.cov1

        # With input_chunk_length=11, output_chunk_length=13, and shift=24
        # size_of_both_chunks = max(11, 24 + 13) = 37
        # actual extractable samples = 100 - 37 + 1 = 64

        # Case 1: max_samples_per_ts=None should extract all 64 samples
        ds_no_limit = ShiftedTorchTrainingDataset(
            series=series,
            input_chunk_length=11,
            output_chunk_length=13,
            shift=24,
            max_samples_per_ts=None,
        )
        assert len(ds_no_limit) == 64

        # Case 2: max_samples_per_ts > actual max should cap at actual max (64)
        ds_high_limit = ShiftedTorchTrainingDataset(
            series=series,
            input_chunk_length=11,
            output_chunk_length=13,
            shift=24,
            max_samples_per_ts=5000,  # Much higher than 64
        )
        # Should be capped at 64, not 5000
        assert len(ds_high_limit) == 64

        # Case 3: max_samples_per_ts < actual max should use the limit
        ds_low_limit = ShiftedTorchTrainingDataset(
            series=series,
            input_chunk_length=11,
            output_chunk_length=13,
            shift=24,
            max_samples_per_ts=50,
        )
        assert len(ds_low_limit) == 50

        # Case 4: Test with stride > 1
        # actual extractable samples with stride=2 = ceil(64 / 2) = 32
        ds_stride = ShiftedTorchTrainingDataset(
            series=series,
            input_chunk_length=11,
            output_chunk_length=13,
            shift=24,
            stride=2,
            max_samples_per_ts=100,
        )
        assert len(ds_stride) == 32

        # Case 5: Multiple series with different lengths
        series1 = gaussian_timeseries(length=50)  # 50 - 37 + 1 = 14 samples
        series2 = gaussian_timeseries(length=100)  # 100 - 37 + 1 = 64 samples
        ds_multi = ShiftedTorchTrainingDataset(
            series=[series1, series2],
            input_chunk_length=11,
            output_chunk_length=13,
            shift=24,
            max_samples_per_ts=5000,
        )
        # Should be capped at 64 (max of both series), so 2 * 64 = 128
        assert len(ds_multi) == 2 * 64

    def test_past_covariates_sequential_dataset(self):
        # one target series
        ds = SequentialTorchTrainingDataset(
            series=self.target1,
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 81
        self._assert_training_output(
            ds[5],
            past_target=self.target1[75:85],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )

        # two target series
        ds = SequentialTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 262
        self._assert_training_output(
            ds[5],
            past_target=self.target1[75:85],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[136],
            past_target=self.target2[125:135],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
        )

        # two target series with custom max_nr_samples
        ds = SequentialTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            max_samples_per_ts=50,
        )
        assert len(ds) == 100
        self._assert_training_output(
            ds[5],
            past_target=self.target1[75:85],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[55],
            past_target=self.target2[125:135],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[75:85],
            past_covariates=self.cov1[75:85],
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[136],
            past_target=self.target2[125:135],
            past_covariates=self.cov2[125:135],
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
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

        np.testing.assert_almost_equal(ds[0].past_target, target.values()[-20:-10])
        np.testing.assert_almost_equal(ds[0].past_covariates, cov.values()[-25:-15])
        assert ds[0].historic_future_covariates is None  # historic future cov
        assert ds[0].future_covariates is None  # future cov
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        assert ds[0].sample_weight is None  # sample weight
        np.testing.assert_almost_equal(ds[0].future_target, target.values()[-10:])

        np.testing.assert_almost_equal(ds[5].past_target, target.values()[-25:-15])
        np.testing.assert_almost_equal(ds[5].past_covariates, cov.values()[-30:-20])
        assert ds[5].historic_future_covariates is None  # historic future cov
        assert ds[5].future_covariates is None  # future cov
        np.testing.assert_almost_equal(ds[5].static_covariates, self.cov_st2)
        assert ds[5].sample_weight is None  # sample weight
        np.testing.assert_almost_equal(ds[5].future_target, target.values()[-15:-5])

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

        np.testing.assert_almost_equal(ds[0].past_target, target.values()[-20:-10])
        np.testing.assert_almost_equal(ds[0].past_covariates, cov.values()[-70:-60])
        assert ds[0].historic_future_covariates is None  # historic future cov
        assert ds[0].future_covariates is None  # future cov
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        assert ds[0].sample_weight is None  # sample weight
        np.testing.assert_almost_equal(ds[0].future_target, target.values()[-10:])

        np.testing.assert_almost_equal(ds[5].past_target, target.values()[-25:-15])
        np.testing.assert_almost_equal(ds[5].past_covariates, cov.values()[-75:-65])
        assert ds[5].historic_future_covariates is None  # historic future cov
        assert ds[5].future_covariates is None  # future cov
        np.testing.assert_almost_equal(ds[5].static_covariates, self.cov_st2)
        assert ds[5].sample_weight is None  # sample weight
        np.testing.assert_almost_equal(ds[5].future_target, target.values()[-15:-5])

    def test_future_covariates_sequential_dataset(self):
        # one target series
        ds = SequentialTorchTrainingDataset(
            series=self.target1,
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 81
        self._assert_training_output(
            ds[5],
            past_target=self.target1[75:85],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )

        # two target series
        ds = SequentialTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 262
        self._assert_training_output(
            ds[5],
            past_target=self.target1[75:85],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[136],
            past_target=self.target2[125:135],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
        )

        # two target series with custom max_nr_samples
        ds = SequentialTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            max_samples_per_ts=50,
        )
        assert len(ds) == 100
        self._assert_training_output(
            ds[5],
            past_target=self.target1[75:85],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[55],
            past_target=self.target2[125:135],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
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

        np.testing.assert_almost_equal(ds[0].past_target, target1.values()[-20:-10])
        assert ds[0].past_covariates is None  # past cov
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, cov1.values()[-40:-30]
        )
        np.testing.assert_almost_equal(ds[0].future_covariates, cov1.values()[-30:-20])
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        assert ds[0].sample_weight is None  # sample weight
        np.testing.assert_almost_equal(ds[0].future_target, target1.values()[-10:])

        np.testing.assert_almost_equal(ds[101].past_target, target2.values()[-40:-30])
        assert ds[0].past_covariates is None  # past cov
        np.testing.assert_almost_equal(
            ds[101].historic_future_covariates, cov2.values()[-70:-60]
        )
        np.testing.assert_almost_equal(
            ds[101].future_covariates, cov2.values()[-60:-50]
        )
        np.testing.assert_almost_equal(ds[101].static_covariates, self.cov_st2)
        assert ds[0].sample_weight is None
        np.testing.assert_almost_equal(ds[101].future_target, target2.values()[-30:-20])

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

        np.testing.assert_almost_equal(ds[0].past_target, target1.values()[-4:-2])
        assert ds[0].past_covariates is None
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, cov1.values()[-6:-4]
        )
        np.testing.assert_almost_equal(ds[0].future_covariates, cov1.values()[-4:-2])
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        assert ds[0].sample_weight is None
        np.testing.assert_almost_equal(ds[0].future_target, target1.values()[-2:])

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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[75:85],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )

        # two target series
        ds = SequentialTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
        )
        assert len(ds) == 262
        self._assert_training_output(
            ds[5],
            past_target=self.target1[75:85],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[136],
            past_target=self.target2[125:135],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
        )

        # two target series with custom max_nr_samples
        ds = SequentialTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            max_samples_per_ts=50,
        )
        assert len(ds) == 100
        self._assert_training_output(
            ds[5],
            past_target=self.target1[75:85],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[55],
            past_target=self.target2[125:135],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
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

        np.testing.assert_almost_equal(ds[0].past_target, target1.values()[-20:-10])
        assert ds[0].past_covariates is None
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, cov1.values()[-40:-30]
        )
        np.testing.assert_almost_equal(ds[0].future_covariates, cov1.values()[-30:-20])
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        assert ds[0].sample_weight is None
        np.testing.assert_almost_equal(ds[0].future_target, target1.values()[-10:])

        np.testing.assert_almost_equal(ds[101].past_target, target2.values()[-40:-30])
        assert ds[0].past_covariates is None
        np.testing.assert_almost_equal(
            ds[101].historic_future_covariates, cov2.values()[-70:-60]
        )
        np.testing.assert_almost_equal(
            ds[101].future_covariates, cov2.values()[-60:-50]
        )
        np.testing.assert_almost_equal(ds[101].static_covariates, self.cov_st2)
        assert ds[101].sample_weight is None
        np.testing.assert_almost_equal(ds[101].future_target, target2.values()[-30:-20])

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

        np.testing.assert_almost_equal(ds[0].past_target, target1.values()[-4:-2])
        assert ds[0].past_covariates is None
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, cov1.values()[-6:-4]
        )
        np.testing.assert_almost_equal(ds[0].future_covariates, cov1.values()[-4:-2])
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        assert ds[0].sample_weight is None
        np.testing.assert_almost_equal(ds[0].future_target, target1.values()[-2:])

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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[80:90],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )

        # two target series
        ds = ShiftedTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
        )
        assert len(ds) == 272
        self._assert_training_output(
            ds[5],
            past_target=self.target1[80:90],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[141],
            past_target=self.target2[130:140],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[80:90],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[55],
            past_target=self.target2[130:140],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[80:90],
            past_covariates=self.cov1[80:90],
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[141],
            past_target=self.target2[130:140],
            past_covariates=self.cov2[130:140],
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
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
        np.testing.assert_almost_equal(ds[0].past_target, target1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0].past_covariates, cov1.values()[-7:-4])
        assert ds[0].historic_future_covariates is None
        assert ds[0].future_covariates is None
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        assert ds[0].sample_weight is None
        np.testing.assert_almost_equal(ds[0].future_target, target1.values()[-3:])

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
        np.testing.assert_almost_equal(ds[0].past_target, target1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0].past_covariates, cov1.values()[-7:-4])
        assert ds[0].historic_future_covariates is None
        assert ds[0].future_covariates is None
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        assert ds[0].sample_weight is None
        np.testing.assert_almost_equal(ds[0].future_target, target1.values()[-3:])

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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[80:90],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )

        # two target series
        ds = ShiftedTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
        )
        assert len(ds) == 272
        self._assert_training_output(
            ds[5],
            past_target=self.target1[80:90],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[141],
            past_target=self.target2[130:140],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[80:90],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[55],
            past_target=self.target2[130:140],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[80:90],
            past_covariates=None,
            historic_future_covariates=self.cov1[80:90],
            future_covariates=self.cov1[85:95],
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[141],
            past_target=self.target2[130:140],
            past_covariates=None,
            historic_future_covariates=self.cov2[130:140],
            future_covariates=self.cov2[135:145],
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
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
        np.testing.assert_almost_equal(ds[0].past_target, target1.values()[-5:-2])
        assert ds[0].past_covariates is None
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, cov1.values()[-7:-4]
        )
        np.testing.assert_almost_equal(ds[0].future_covariates, cov1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        assert ds[0].sample_weight is None
        np.testing.assert_almost_equal(ds[0].future_target, target1.values()[-3:])

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
        np.testing.assert_almost_equal(ds[0].past_target, target1.values()[-5:-2])
        assert ds[0].past_covariates is None
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, cov1.values()[-7:-4]
        )
        np.testing.assert_almost_equal(ds[0].future_covariates, cov1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        assert ds[0].sample_weight is None
        np.testing.assert_almost_equal(ds[0].future_target, target1.values()[-3:])

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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[80:90],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )

        # two target series
        ds = ShiftedTorchTrainingDataset(
            series=[self.target1, self.target2],
            input_chunk_length=10,
            output_chunk_length=10,
            shift=5,
        )
        assert len(ds) == 272
        self._assert_training_output(
            ds[5],
            past_target=self.target1[80:90],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[141],
            past_target=self.target2[130:140],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[80:90],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[55],
            past_target=self.target2[130:140],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[80:90],
            past_covariates=None,
            historic_future_covariates=self.cov1[80:90],
            future_covariates=self.cov1[85:95],
            static_covariates=self.cov_st1,
            sample_weight=None,
            future_target=self.target1[85:95],
        )
        self._assert_training_output(
            ds[141],
            past_target=self.target2[130:140],
            past_covariates=None,
            historic_future_covariates=self.cov2[130:140],
            future_covariates=self.cov2[135:145],
            static_covariates=self.cov_st2,
            sample_weight=None,
            future_target=self.target2[135:145],
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
        np.testing.assert_almost_equal(ds[0].past_target, target1.values()[-5:-2])
        assert ds[0].past_covariates is None
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, cov1.values()[-7:-4]
        )
        np.testing.assert_almost_equal(ds[0].future_covariates, cov1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        assert ds[0].sample_weight is None
        np.testing.assert_almost_equal(ds[0].future_target, target1.values()[-3:])

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
        np.testing.assert_almost_equal(ds[0].past_target, target1.values()[-5:-2])
        assert ds[0].past_covariates is None
        np.testing.assert_almost_equal(
            ds[0].historic_future_covariates, cov1.values()[-7:-4]
        )
        np.testing.assert_almost_equal(ds[0].future_covariates, cov1.values()[-5:-2])
        np.testing.assert_almost_equal(ds[0].static_covariates, self.cov_st2)
        assert ds[0].sample_weight is None
        np.testing.assert_almost_equal(ds[0].future_target, target1.values()[-3:])

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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[65:85],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=weight_exp,
            future_target=self.target1[85:95],
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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[65:85],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=weight_exp1,
            future_target=self.target1[85:95],
        )
        # 21 samples after comes the second series
        self._assert_training_output(
            ds[26],
            past_target=self.target2[115:135],
            past_covariates=None,
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=weight_exp2,
            future_target=self.target2[135:145],
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
        self._assert_training_output(
            ds[5],
            past_target=self.target1[65:85],
            past_covariates=self.cov1[65:85],
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st1,
            sample_weight=weight_exp1,
            future_target=self.target1[85:95],
        )
        # 21 samples after comes the second series
        self._assert_training_output(
            ds[26],
            past_target=self.target2[115:135],
            past_covariates=self.cov2[115:135],
            historic_future_covariates=None,
            future_covariates=None,
            static_covariates=self.cov_st2,
            sample_weight=weight_exp2,
            future_target=self.target2[135:145],
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
        assert isinstance(batch_reg, TorchTrainingSample)
        assert isinstance(batch_shift, TorchTrainingSample)

        if future_idx is not None:
            # 3rd future values of regular ds must be identical to the 1st future values of shifted dataset
            np.testing.assert_array_equal(
                batch_reg.future_covariates[-1:], batch_shift.future_covariates
            )

        skip = {"sample_weight", "future_target"}
        if future_idx is not None:
            skip.add("future_covariates")
        for f in fields(batch_reg):
            name = f.name
            if name in skip:
                continue
            el_reg, el_shift = getattr(batch_reg, name), getattr(batch_shift, name)
            if el_reg is None:
                assert el_shift is None
            else:
                np.testing.assert_array_equal(el_reg, el_shift)

        # last two elements are (sample weight, output chunk of the target series).
        # 3rd future values of regular ds must be identical to the 1st future values of shifted dataset
        np.testing.assert_array_equal(
            batch_reg.sample_weight[ocs:], batch_shift.sample_weight
        )

    @pytest.mark.parametrize(
        "config",
        param_product(
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
        assert ds[0].sample_weight is None

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
        weight_exp = (
            ds[0].future_target + 1 if manual_weight else get_built_in_weigths(target)
        )
        assert np.all(ds[0].sample_weight == weight_exp)

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
        weight_exp = (
            ds[0].future_target + 1 if manual_weight else get_built_in_weigths(target)
        )
        assert np.all(ds[0].sample_weight == weight_exp)

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
        weight_exp = (
            ds[0].future_target + 1 if manual_weight else get_built_in_weigths(target)
        )
        assert np.all(ds[0].sample_weight == weight_exp)

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
            ds[0].future_target[:, 0:1] + 1
            if manual_weight
            else get_built_in_weigths(target)
        )
        assert np.all(ds[0].sample_weight == weight_exp)

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
        weight_exp = (
            ds[0].future_target + 1 if manual_weight else get_built_in_weigths(target)
        )
        assert np.all(ds[0].sample_weight == weight_exp)

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
        weight_exp = (
            ds[0].future_target + 1 if manual_weight else get_built_in_weigths(target)
        )
        assert np.all(ds[0].sample_weight == weight_exp)

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
        weight_exp = (
            ds[0].future_target + 1 if manual_weight else get_built_in_weigths(target)
        )
        assert np.all(ds[0].sample_weight == weight_exp)

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

    def test_custom_training_dataset_named_sample(self):
        """Custom training datasets must return a `TorchTrainingSample` (unused fields can be omitted)."""

        class NamedTrainingDataset(TorchTrainingDataset):
            def __init__(self, wrapped):
                super().__init__()
                self._wrapped = wrapped

            def __len__(self):
                return len(self._wrapped)

            def __getitem__(self, index):
                sample = self._wrapped[index]
                return TorchTrainingSample(
                    past_target=sample.past_target,
                    past_covariates=sample.past_covariates,
                    historic_future_covariates=sample.historic_future_covariates,
                    future_covariates=sample.future_covariates,
                    static_covariates=sample.static_covariates,
                    future_target=sample.future_target,
                )

        wrapped = SequentialTorchTrainingDataset(
            series=self.target1,
            past_covariates=self.cov1,
            future_covariates=self.cov1,
            input_chunk_length=4,
            output_chunk_length=3,
        )
        ds = NamedTrainingDataset(wrapped)

        sample = ds[0]
        assert isinstance(sample, TorchTrainingSample)
        expected = wrapped[0]
        self._assert_training_output(
            _as_training_sample(sample),
            past_target=expected.past_target,
            past_covariates=expected.past_covariates,
            historic_future_covariates=expected.historic_future_covariates,
            future_covariates=expected.future_covariates,
            static_covariates=expected.static_covariates,
            sample_weight=None,
            future_target=expected.future_target,
        )

        batch = _batch_collate_fn_train([ds[0], ds[1]])
        assert isinstance(batch, TorchTrainingBatch)
        assert batch.past_target.shape[0] == 2
        np.testing.assert_array_equal(
            batch.past_target[0].numpy(), expected.past_target
        )
        np.testing.assert_array_equal(
            batch.future_target[0].numpy(), expected.future_target
        )

        with pytest.raises(ValueError, match="must return a `TorchTrainingSample`"):
            _as_training_sample((expected.past_target, expected.future_target))

    def test_custom_inference_dataset_named_sample(self):
        """Custom inference datasets must return a `TorchInferenceSample`."""

        class NamedInferenceDataset(TorchInferenceDataset):
            def __init__(self, wrapped):
                super().__init__()
                self._wrapped = wrapped

            def __len__(self):
                return len(self._wrapped)

            def __getitem__(self, index):
                sample = self._wrapped[index]
                return TorchInferenceSample(
                    past_target=sample.past_target,
                    past_covariates=sample.past_covariates,
                    future_past_covariates=sample.future_past_covariates,
                    historic_future_covariates=sample.historic_future_covariates,
                    future_covariates=sample.future_covariates,
                    static_covariates=sample.static_covariates,
                    series_schema=sample.series_schema,
                    pred_time=sample.pred_time,
                )

        wrapped = SequentialTorchInferenceDataset(
            series=[self.target1, self.target2],
            past_covariates=[self.cov1, self.cov2],
            input_chunk_length=4,
            output_chunk_length=3,
            n=1,
        )
        ds = NamedInferenceDataset(wrapped)

        sample = ds[0]
        assert isinstance(sample, TorchInferenceSample)
        expected = wrapped[0]
        self._assert_inference_output(
            _as_inference_sample(sample),
            past_target=expected.past_target,
            past_covariates=expected.past_covariates,
            future_past_covariates=expected.future_past_covariates,
            historic_future_covariates=expected.historic_future_covariates,
            future_covariates=expected.future_covariates,
            static_covariates=expected.static_covariates,
            series_schema=expected.series_schema,
            pred_time=expected.pred_time,
        )

        batch = _batch_collate_fn_predict([ds[0], ds[1]])
        assert isinstance(batch, TorchInferenceBatch)
        assert batch.past_target.shape[0] == 2
        np.testing.assert_array_equal(
            batch.past_target[0].numpy(), expected.past_target
        )
        assert batch.pred_time[0] == expected.pred_time

        with pytest.raises(ValueError, match="must return a `TorchInferenceSample`"):
            _as_inference_sample((expected.past_target, expected.pred_time))

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
        times1 = pd.date_range(start="20100101", end="20120101", freq="ME")
        times2 = pd.date_range(start="20090101", end="20110601", freq="ME")
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

    def test_pl_module_input_stage(self):
        """`stage` defaults to predict, is set on train batches, and is not a pytree leaf."""
        past = torch.zeros(2, 3, 1)
        x_default = PLModuleInput(past_target=past)
        assert x_default.stage is ModuleStage.PREDICT

        batch = TorchTrainingBatch(past_target=past, future_target=torch.zeros(2, 1, 1))
        assert batch.to_module_input().stage is ModuleStage.TRAIN
        assert (
            batch.to_module_input(stage=ModuleStage.VALIDATE).stage
            is ModuleStage.VALIDATE
        )

        x_train = PLModuleInput(past_target=past, stage=ModuleStage.TRAIN)
        leaves, spec = tree_flatten(x_train)
        assert ModuleStage.TRAIN not in leaves
        assert all(leaf is None or torch.is_tensor(leaf) for leaf in leaves)
        restored = tree_unflatten(leaves, spec)
        assert isinstance(restored, PLModuleInput)
        assert restored.stage is ModuleStage.TRAIN
        assert restored.past_target is past

    def test_pl_module_output_pytree(self):
        """`PLModuleOutput` round-trips through torch pytree flatten/unflatten."""
        prediction = torch.zeros(2, 3, 1, 1)
        state = (torch.ones(2, 1, 4), torch.zeros(2, 1, 4))
        out = PLModuleOutput(prediction=prediction, state=state)
        leaves, spec = tree_flatten(out)
        restored = tree_unflatten(leaves, spec)
        assert isinstance(restored, PLModuleOutput)
        assert restored.prediction is prediction
        assert restored.state[0] is state[0]
        assert restored.state[1] is state[1]

    def test_coerce_training_sample_invalid(self):
        """Legacy `train_sample` values must be a 6-tuple or `TorchTrainingSample`."""
        sample = TorchTrainingSample(past_target=np.zeros((2, 1)))
        assert _coerce_training_sample(sample) is sample

        with pytest.raises(ValueError, match="must have 6 elements"):
            _coerce_training_sample((np.zeros((2, 1)),))

        with pytest.raises(ValueError, match="Unsupported `train_sample` type"):
            _coerce_training_sample("not-a-sample")

    def test_train_sample_from_shapes_none(self):
        with pytest.raises(ValueError, match="must not be `None`"):
            _train_sample_from_shapes(None, dtype=np.float32)


def generate_series(n_variables: int, length: int, prefix: str):
    return concatenate(
        [
            linear_timeseries(
                length=length, dtype=np.float32, column_name=f"{prefix}_{i}"
            )
            for i in range(n_variables)
        ],
        axis=1,
    )


class TestVariableICLDataset:
    def test_dataset_padding_training(self):
        """Training dataset should NaN-pad past features for short series."""
        icl, ocl, min_icl = 14, 6, 2
        # use a series long enough that rightmost sample has no padding
        series = generate_series(n_variables=1, length=25, prefix="T")

        ds = SequentialTorchTrainingDataset(
            series=series,
            input_chunk_length=(min_icl, icl),
            output_chunk_length=ocl,
        )
        # min_size = 6 + 2 = 8, n_samples = 25 - 8 + 1 = 18
        assert len(ds) == 18

        # rightmost sample (idx 0): past_start = 25 - 6 - 14 = 5, no padding
        sample = ds[0]
        assert isinstance(sample, TorchTrainingSample)
        pt, ft = sample.past_target, sample.future_target
        assert pt.shape == (icl, 1)
        assert ft.shape == (ocl, 1)
        assert not np.isnan(pt).any()

        # leftmost sample (idx 17): past_start = 8 - 6 - 14 = -12, pad_len = 12
        sample = ds[17]
        assert isinstance(sample, TorchTrainingSample)
        pt, ft = sample.past_target, sample.future_target
        assert pt.shape == (icl, 1)
        assert ft.shape == (ocl, 1)
        pad_len = np.isnan(pt[:, 0]).sum()
        assert pad_len == 12
        assert not np.isnan(pt[pad_len:]).any()
        assert not np.isnan(ft).any()

    def test_dataset_padding_inference(self):
        """Inference dataset should NaN-pad past target for short series."""
        icl, ocl, min_icl = 14, 6, 2
        series = generate_series(n_variables=1, length=5, prefix="T")

        ds = SequentialTorchInferenceDataset(
            series=series,
            input_chunk_length=(min_icl, icl),
            output_chunk_length=ocl,
            n=ocl,
        )
        assert len(ds) == 1

        sample = ds[0]
        assert isinstance(sample, TorchInferenceSample)
        pt = sample.past_target
        assert pt.shape == (icl, 1)
        # first (icl - len(series)) = 9 values should be NaN
        pad_len = icl - len(series)
        assert np.isnan(pt[:pad_len]).all()
        assert not np.isnan(pt[pad_len:]).any()

    def test_dataset_no_padding_without_variable_icl(self):
        """With a fixed input chunk length, datasets should not left-pad."""
        icl, ocl = 14, 6
        series = generate_series(n_variables=1, length=25, prefix="T")

        ds_default = SequentialTorchTrainingDataset(
            series=series,
            input_chunk_length=icl,
            output_chunk_length=ocl,
        )
        ds_explicit = SequentialTorchTrainingDataset(
            series=series,
            input_chunk_length=(icl, icl),
            output_chunk_length=ocl,
        )
        assert len(ds_default) == len(ds_explicit)

        # no NaN padding in any sample
        for i in range(len(ds_default)):
            pt_d = ds_default[i].past_target
            pt_e = ds_explicit[i].past_target
            assert not np.isnan(pt_d).any()
            np.testing.assert_array_equal(pt_d, pt_e)

    def test_dataset_sample_uniqueness(self):
        """Each distinct sample index for a series must have a unique output window."""
        icl, ocl, min_icl = 10, 4, 2
        series = generate_series(n_variables=1, length=15, prefix="T")

        ds = SequentialTorchTrainingDataset(
            series=series,
            input_chunk_length=(min_icl, icl),
            output_chunk_length=ocl,
        )
        # min_size = (10 + 4) - (10 - 2) = 6, n_samples = 15 - 6 + 1 = 10
        assert len(ds) == 10

        # each sample should produce a different future_target window
        ft_windows = []
        for i in range(len(ds)):
            ft = ds[i].future_target
            ft_windows.append(tuple(ft[:, 0].tolist()))
        assert len(set(ft_windows)) == len(ds)

    def test_dataset_padding_with_covariates(self):
        """NaN-padding should apply to past covariates and historic future covariates."""
        icl, ocl, min_icl = 14, 6, 2
        series = generate_series(n_variables=1, length=25, prefix="T")
        past_cov = generate_series(n_variables=2, length=25, prefix="PC")
        future_cov = generate_series(n_variables=3, length=50, prefix="FC")

        ds = SequentialTorchTrainingDataset(
            series=series,
            past_covariates=past_cov,
            future_covariates=future_cov,
            input_chunk_length=(min_icl, icl),
            output_chunk_length=ocl,
        )

        # leftmost sample (most padding)
        sample = ds[len(ds) - 1]
        assert isinstance(sample, TorchTrainingSample)
        pt, pc, hfc, fc = (
            sample.past_target,
            sample.past_covariates,
            sample.historic_future_covariates,
            sample.future_covariates,
        )

        assert pt.shape == (icl, 1)
        assert pc.shape == (icl, 2)
        assert hfc.shape == (icl, 3)
        assert fc.shape == (ocl, 3)

        # padding lengths must match across all past features
        pt_pad = np.isnan(pt[:, 0]).sum()
        pc_pad = np.isnan(pc[:, 0]).sum()
        hfc_pad = np.isnan(hfc[:, 0]).sum()
        assert pt_pad == pc_pad == hfc_pad
        assert pt_pad == 12

        # future covariates in the output chunk are never padded
        assert not np.isnan(fc).any()
