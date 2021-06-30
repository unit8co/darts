from .base_test_class import DartsBaseTestClass
from ..utils.data import SimpleInferenceDataset, SequentialDataset, ShiftedDataset, HorizonBasedDataset
from ..utils.timeseries_generation import gaussian_timeseries


class DatasetTestCase(DartsBaseTestClass):
    target1, target2 = gaussian_timeseries(length=100), gaussian_timeseries(length=150)
    cov1, cov2 = gaussian_timeseries(length=100), gaussian_timeseries(length=150)

    def _assert_eq(self, tup_ar, tup_series):
        l1 = []
        for ar_element in tup_ar:
            l1.append(None if ar_element is None else list(ar_element))
        l2 = []
        for series_element in tup_series:
            l2.append(None if series_element is None else list(series_element.values(copy=False)))

        self.assertEqual(l1, l2)

    def test_simple_inference_dataset(self):
        # one target series
        ds = SimpleInferenceDataset(series=self.target1, input_chunk_length=len(self.target1))
        self.assertEqual(ds[0], (self.target1, None, None))

        # two target series
        ds = SimpleInferenceDataset(series=[self.target1, self.target2],
                                    input_chunk_length=max(len(self.target1), len(self.target2)))
        self.assertEqual(ds[1], (self.target2, None, None))

        # fail if covariates do not have same size
        with self.assertRaises(ValueError):
            ds = SimpleInferenceDataset(series=[self.target1, self.target2], covariates=[self.cov1])

        # with covariates
        ds = SimpleInferenceDataset(series=[self.target1, self.target2], covariates=[self.cov1, self.cov2],
                                    input_chunk_length=max(len(self.target1), len(self.target2)))
        self.assertEqual(ds[1], (self.target2, self.cov2, None))

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
