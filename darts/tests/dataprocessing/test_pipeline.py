import logging
import unittest

from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import (
    BaseDataTransformer,
    FittableDataTransformer,
    InvertibleDataTransformer,
    InvertibleMapper,
    Mapper,
)
from darts.utils.timeseries_generation import constant_timeseries


class PipelineTestCase(unittest.TestCase):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    class DataTransformerMock1(BaseDataTransformer):
        def __init__(self):
            super().__init__()
            self.transform_called = False
            self.inverse_transform_called = False
            self.fit_called = False

        @staticmethod
        def ts_transform(data: TimeSeries) -> TimeSeries:
            return data.append_values(constant_timeseries(value=1, length=3).values())

        def transform(self, data, *args, **kwargs) -> TimeSeries:
            self.transform_called = True
            return super().transform(data, *args, **kwargs)

    class DataTransformerMock2(FittableDataTransformer, InvertibleDataTransformer):
        def __init__(self):
            super().__init__()

            self.transform_called = False
            self.inverse_transform_called = False
            self.fit_called = False

        @staticmethod
        def ts_fit(series: TimeSeries):
            pass

        @staticmethod
        def ts_transform(series: TimeSeries) -> TimeSeries:
            return series.append_values(constant_timeseries(value=2, length=3).values())

        @staticmethod
        def ts_inverse_transform(series: TimeSeries) -> TimeSeries:
            return series

        def fit(self, data):
            super().fit(data)
            self.fit_called = True
            return self

        def transform(self, data, *args, **kwargs):
            self.transform_called = True
            self.args = args
            self.kwargs = kwargs
            return super().transform(data, *args, **kwargs)

        def inverse_transform(self, data, *args, **kwargs) -> TimeSeries:
            self.inverse_transform_called = True
            self.args = args
            self.kwargs = kwargs
            return super().inverse_transform(data, *args, **kwargs)

    class PlusTenTransformer(InvertibleDataTransformer):
        def __init__(self, name="+10 transformer"):
            super().__init__(name=name)

        @staticmethod
        def ts_transform(series: TimeSeries) -> TimeSeries:
            return series.map(lambda x: x + 10)

        @staticmethod
        def ts_inverse_transform(series: TimeSeries) -> TimeSeries:
            return series.map(lambda x: x - 10)

    class TimesTwoTransformer(InvertibleDataTransformer):
        def __init__(self):
            super().__init__(name="*2 transformer")

        @staticmethod
        def ts_transform(data: TimeSeries) -> TimeSeries:
            return data.map(lambda x: x * 2)

        @staticmethod
        def ts_inverse_transform(data: TimeSeries) -> TimeSeries:
            return data.map(lambda x: x / 2)

    def test_transform(self):
        # given
        mock1 = self.DataTransformerMock1()
        mock2 = self.DataTransformerMock2()
        data = constant_timeseries(value=0, length=3)
        transformers = [mock1] * 10 + [mock2] * 10
        p = Pipeline(transformers)
        # when
        p.fit(data)
        transformed = p.transform(data)

        # then
        self.assertEqual(63, len(transformed))
        self.assertEqual([0] * 3 + [1] * 30 + [2] * 30, list(transformed.values()))
        for t in transformers:
            self.assertTrue(t.transform_called)
            self.assertFalse(t.inverse_transform_called)

    def test_inverse_raise_exception(self):
        # given
        mock = self.DataTransformerMock1()
        p = Pipeline([mock])

        # when & then
        with self.assertRaises(ValueError):
            p.inverse_transform(None)

    def test_transformers_not_modified(self):
        # given
        mock = self.DataTransformerMock1()
        p = Pipeline([mock], copy=True)

        # when
        p.transform(constant_timeseries(value=1, length=10))

        # then
        self.assertFalse(mock.transform_called)

    def test_fit(self):
        # given
        data = constant_timeseries(value=0, length=3)
        transformers = [self.DataTransformerMock1() for _ in range(10)] + [
            self.DataTransformerMock2() for _ in range(10)
        ]
        p = Pipeline(transformers)

        # when
        p.fit(data)

        # then
        for i in range(10):
            self.assertFalse(transformers[i].fit_called)
        for i in range(10, 20):
            self.assertTrue(transformers[i].fit_called)

    def test_fit_skips_superfluous_transforms(self):
        # given
        data = constant_timeseries(value=0, length=100)
        transformers = (
            [self.DataTransformerMock1() for _ in range(10)]
            + [self.DataTransformerMock2()]
            + [self.DataTransformerMock1() for _ in range(10)]
        )
        p = Pipeline(transformers)

        # when
        p.fit(data)

        # then
        for i in range(10):
            self.assertTrue(transformers[i].transform_called)
        self.assertTrue(transformers[10].fit_called)
        self.assertFalse(transformers[10].transform_called)
        for i in range(11, 21):
            self.assertFalse(transformers[i].transform_called)

    def test_transform_fit(self):
        # given
        data = constant_timeseries(value=0, length=3)
        transformers = [self.DataTransformerMock1() for _ in range(10)] + [
            self.DataTransformerMock2() for _ in range(10)
        ]
        p = Pipeline(transformers)

        # when
        _ = p.fit_transform(data)

        # then
        for t in transformers:
            self.assertTrue(t.transform_called)
        for i in range(10):
            self.assertFalse(transformers[i].fit_called)
        for i in range(10, 20):
            self.assertTrue(transformers[i].fit_called)

    def test_inverse_transform(self):
        # given
        data = constant_timeseries(value=0.0, length=3)

        transformers = [self.PlusTenTransformer(), self.TimesTwoTransformer()]
        p = Pipeline(transformers)

        # when
        transformed = p.transform(data)
        back = p.inverse_transform(transformed)

        # then
        self.assertEqual(data, back)

    def test_getitem(self):
        # given

        transformers = [
            self.PlusTenTransformer(name=f"+10 transformer{i}") for i in range(0, 10)
        ]
        p = Pipeline(transformers)

        # when & then
        # note : only compares string representations, since __getitem__() copies the transformers
        self.assertEqual(str(p[1]._transformers), str([transformers[1]]))
        self.assertEqual(str(p[4:8]._transformers), str(transformers[4:8]))

        with self.assertRaises(ValueError):
            p["invalid attempt"]

    def test_raises_on_non_transformers(self):
        # given
        input_list = list(range(10))

        # when & then
        with self.assertRaises(
            ValueError,
            msg="transformers should be objects deriving from BaseDataTransformer",
        ):
            Pipeline(input_list)

    def test_raises_on_bad_key(self):
        # given
        bad_key = 12.0
        p = Pipeline([])

        # when & then
        with self.assertRaises(ValueError, msg="Key must be int, str or slice"):
            p[bad_key]

    def test_multi_ts(self):

        series1 = constant_timeseries(value=0.0, length=3)
        series2 = constant_timeseries(value=1.0, length=3)

        data = [series1, series2]

        mapper1 = InvertibleMapper(fn=lambda x: x + 10, inverse_fn=lambda x: x - 10)
        mapper2 = InvertibleMapper(fn=lambda x: x * 10, inverse_fn=lambda x: x / 10)

        transformers = [mapper1, mapper2]
        p = Pipeline(transformers)

        # when
        transformed = p.transform(data)
        back = p.inverse_transform(transformed)

        # then
        self.assertEqual(data, back)

    def test_pipeline_partial_inverse(self):
        series = constant_timeseries(value=0.0, length=3)

        def plus_ten(x):
            return x + 10

        mapper = Mapper(fn=plus_ten)
        mapper_inv = InvertibleMapper(fn=lambda x: x + 2, inverse_fn=lambda x: x - 2)

        series_plus_ten = mapper.transform(series)

        pipeline = Pipeline([mapper, mapper_inv])

        transformed = pipeline.transform(series)

        # should fail, since partial is False by default
        with self.assertRaises(ValueError):
            pipeline.inverse_transform(transformed)

        back = pipeline.inverse_transform(transformed, partial=True)

        # while the +/- 2 is inverted, the +10 operation is not
        self.assertEqual(series_plus_ten, back)

    def test_pipeline_verbose(self):
        """
        Checks if the verbose param applied to the pipeline is changing the verbosity level in the
        contained transformers.
        """

        def plus_ten(x):
            return x + 10

        mapper = Mapper(fn=plus_ten, verbose=True)
        mapper_inv = InvertibleMapper(
            fn=lambda x: x + 2, inverse_fn=lambda x: x - 2, verbose=True
        )

        verbose_value = False
        pipeline = Pipeline([mapper, mapper_inv], verbose=verbose_value)

        for transformer in pipeline:
            self.assertEqual(transformer._verbose, verbose_value)

    def test_pipeline_n_jobs(self):
        """
        Checks if the n_jobs param applied to the pipeline is changing the verbosity level in the
        contained transformers.
        """

        def plus_ten(x):
            return x + 10

        mapper = Mapper(fn=plus_ten, n_jobs=1)
        mapper_inv = InvertibleMapper(
            fn=lambda x: x + 2, inverse_fn=lambda x: x - 2, n_jobs=2
        )

        n_jobs_value = -1
        pipeline = Pipeline([mapper, mapper_inv], n_jobs=n_jobs_value)

        for transformer in pipeline:
            self.assertEqual(transformer._n_jobs, n_jobs_value)
