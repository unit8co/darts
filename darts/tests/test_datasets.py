import pandas as pd
import numpy as np

from .base_test_class import DartsBaseTestClass
from ..timeseries import TimeSeries
from ..utils.data import (PastCovariatesInferenceDataset, PastCovariatesSequentialDataset,
                          PastCovariatesShiftedDataset, HorizonBasedDataset)
from ..utils.timeseries_generation import gaussian_timeseries


class DatasetTestCase(DartsBaseTestClass):
    target1, target2 = gaussian_timeseries(length=100), gaussian_timeseries(length=150)
    vals1, vals2 = target1.values(), target2.values()
    cov1, cov2 = gaussian_timeseries(length=100), gaussian_timeseries(length=150)

    def _assert_eq(self, tup_ar, tup_series):
        l1 = []
        for ar_element in tup_ar:
            l1.append(None if ar_element is None else list(ar_element))
        l2 = []
        for series_element in tup_series:
            l2.append(None if series_element is None else list(series_element.values(copy=False)))

        self.assertEqual(l1, l2)

    def test_past_covariates_inference_dataset(self):
        # one target series
        ds = PastCovariatesInferenceDataset(target_series=self.target1, input_chunk_length=len(self.target1))
        np.testing.assert_almost_equal(ds[0][0], self.vals1)
        self.assertEqual(ds[0][1:], (None, None, self.target1))

        # two target series
        ds = PastCovariatesInferenceDataset(target_series=[self.target1, self.target2],
                                            input_chunk_length=max(len(self.target1), len(self.target2)))
        np.testing.assert_almost_equal(ds[1][0], self.vals2)
        self.assertEqual(ds[1][1:], (None, None, self.target2))

        # fail if covariates do not have same size
        with self.assertRaises(ValueError):
            ds = PastCovariatesInferenceDataset(target_series=[self.target1, self.target2], covariates=[self.cov1])

        # with covariates
        ds = PastCovariatesInferenceDataset(target_series=[self.target1, self.target2],
                                            covariates=[self.cov1, self.cov2],
                                            input_chunk_length=max(len(self.target1), len(self.target2)))
        np.testing.assert_almost_equal(ds[1][0], self.vals2)
        np.testing.assert_almost_equal(ds[1][1], self.cov2.values())
        self.assertEqual(ds[1][2:], (None, self.target2))  # no "future past" covariate here

        # more complex case with future past covariates
        t1 =
        ds = PastCovariatesInferenceDataset(target_series=[self.target1, self.target2],
                                            covariates=[self.cov1, self.cov2],
                                            input_chunk_length=10,
                                            output_chunk_length=10,
                                            n=50)


        # TODO: more complex case with series shifted w.r.t. one another and datetimes


    def test_sequential_dataset(self):
        # one target series
        ds = SequentialDataset(target_series=self.target1, input_chunk_length=10, output_chunk_length=10)
        self.assertEqual(len(ds), 81)
        self._assert_eq(ds[5], (self.target1[75:85], self.target1[85:95], None))

        # two target series
        ds = SequentialDataset(target_series=[self.target1, self.target2], input_chunk_length=10, output_chunk_length=10)
        self.assertEqual(len(ds), 262)
        self._assert_eq(ds[5], (self.target1[75:85], self.target1[85:95], None))
        self._assert_eq(ds[136], (self.target2[125:135], self.target2[135:145], None))

        # two target series with custom max_nr_samples
        ds = SequentialDataset(target_series=[self.target1, self.target2],
                               input_chunk_length=10, output_chunk_length=10, max_samples_per_ts=50)
        self.assertEqual(len(ds), 100)
        self._assert_eq(ds[5], (self.target1[75:85], self.target1[85:95], None))
        self._assert_eq(ds[55], (self.target2[125:135], self.target2[135:145], None))

        # two targets and one covariate
        with self.assertRaises(ValueError):
            ds = SequentialDataset(target_series=[self.target1, self.target2], covariates=[self.cov1])

        # two targets and two covariates
        ds = SequentialDataset(target_series=[self.target1, self.target2],
                               covariates=[self.cov1, self.cov2],
                               input_chunk_length=10, output_chunk_length=10)
        self._assert_eq(ds[5], (self.target1[75:85], self.target1[85:95], self.cov1[75:85]))
        self._assert_eq(ds[136], (self.target2[125:135], self.target2[135:145], self.cov2[125:135]))

    def test_shifted_dataset(self):
        # one target series
        ds = ShiftedDataset(target_series=self.target1, length=10, shift=5)
        self.assertEqual(len(ds), 86)
        self._assert_eq(ds[5], (self.target1[80:90], self.target1[85:95], None))

        # two target series
        ds = ShiftedDataset(target_series=[self.target1, self.target2], length=10, shift=5)
        self.assertEqual(len(ds), 272)
        self._assert_eq(ds[5], (self.target1[80:90], self.target1[85:95], None))
        self._assert_eq(ds[141], (self.target2[130:140], self.target2[135:145], None))

        # two target series with custom max_nr_samples
        ds = ShiftedDataset(target_series=[self.target1, self.target2], length=10, shift=5, max_samples_per_ts=50)
        self.assertEqual(len(ds), 100)
        self._assert_eq(ds[5], (self.target1[80:90], self.target1[85:95], None))
        self._assert_eq(ds[55], (self.target2[130:140], self.target2[135:145], None))

        # two targets and one covariate
        with self.assertRaises(ValueError):
            ds = ShiftedDataset(target_series=[self.target1, self.target2], covariates=[self.cov1])

        # two targets and two covariates
        ds = ShiftedDataset(target_series=[self.target1, self.target2],
                            covariates=[self.cov1, self.cov2],
                            length=10, shift=5)
        self._assert_eq(ds[5], (self.target1[80:90], self.target1[85:95], self.cov1[80:90]))
        self._assert_eq(ds[141], (self.target2[130:140], self.target2[135:145], self.cov2[130:140]))

    def test_horizon_based_dataset(self):
        # one target series
        ds = HorizonBasedDataset(target_series=self.target1, output_chunk_length=10, lh=(1, 3), lookback=2)
        self.assertEqual(len(ds), 20)
        self._assert_eq(ds[5], (self.target1[65:85], self.target1[85:95], None))

        # two target series
        ds = HorizonBasedDataset(target_series=[self.target1, self.target2],
                                 output_chunk_length=10, lh=(1, 3), lookback=2)
        self.assertEqual(len(ds), 40)
        self._assert_eq(ds[5], (self.target1[65:85], self.target1[85:95], None))
        self._assert_eq(ds[25], (self.target2[115:135], self.target2[135:145], None))

        # two targets and one covariate
        with self.assertRaises(ValueError):
            ds = HorizonBasedDataset(target_series=[self.target1, self.target2], covariates=[self.cov1])

        # two targets and two covariates
        ds = HorizonBasedDataset(target_series=[self.target1, self.target2],
                                 covariates=[self.cov1, self.cov2],
                                 output_chunk_length=10, lh=(1, 3), lookback=2)
        self._assert_eq(ds[5], (self.target1[65:85], self.target1[85:95], self.cov1[65:85]))
        self._assert_eq(ds[25], (self.target2[115:135], self.target2[135:145], self.cov2[115:135]))

    def test_get_matching_index(self):
        from ..utils.data.utils import _get_matching_index

        # Check dividable freq
        times1 = pd.date_range(start='20100101', end='20100330', freq='D')
        times2 = pd.date_range(start='20100101', end='20100320', freq='D')
        target = TimeSeries.from_times_and_values(times1, np.random.randn(len(times1)))
        cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        self.assertEqual(_get_matching_index(target, cov, idx=15), 5)

        # check non-dividable freq
        times1 = pd.date_range(start='20100101', end='20120101', freq='M')
        times2 = pd.date_range(start='20090101', end='20110601', freq='M')
        target = TimeSeries.from_times_and_values(times1, np.random.randn(len(times1)))
        cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        self.assertEqual(_get_matching_index(target, cov, idx=15), 15 - 7)

        # check integer-indexed series
        times2 = pd.RangeIndex(start=10, stop=90)
        target = TimeSeries.from_values(np.random.randn(100))
        cov = TimeSeries.from_times_and_values(times2, np.random.randn(len(times2)))
        self.assertEqual(_get_matching_index(target, cov, idx=15), 5)
