import unittest
import logging

from darts import TimeSeries
from darts.utils.timeseries_generation import constant_timeseries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import BaseDataTransformer, FittableDataTransformer, InvertibleDataTransformer


class PipelineTestCase(unittest.TestCase):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    class DataTransformerMock1(BaseDataTransformer[TimeSeries]):
        def __init__(self):
            super().__init__()
            self.transform_called = False
            self.inverse_transform_called = False
            self.fit_called = False

        def transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
            super().transform(data, *args, **kwargs)
            self.transform_called = True
            return data.append_values(constant_timeseries(1, 3).values())

    class DataTransformerMock2(FittableDataTransformer[TimeSeries], InvertibleDataTransformer[TimeSeries]):
        def __init__(self):
            super().__init__()
            self.transform_called = False
            self.inverse_transform_called = False
            self.fit_called = False

        def fit(self, data):
            super().fit(data)
            self.fit_called = True
            return self

        def transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
            super().transform(data, *args, **kwargs)
            self.transform_called = True
            self.args = args
            self.kwargs = kwargs
            return data.append_values(constant_timeseries(2, 3).values())

        def inverse_transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
            super().inverse_transform(data, *args, **kwargs)
            self.inverse_transform_called = True
            self.args = args
            self.kwargs = kwargs
            return data

    class PlusTenTransformer(InvertibleDataTransformer[TimeSeries]):
        def __init__(self, name="+10 transformer"):
            super().__init__(name=name)

        def transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
            super().transform(data, *args, **kwargs)
            return data.map(lambda x: x + 10)

        def inverse_transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
            super().inverse_transform(data, *args, **kwargs)
            return data.map(lambda x: x - 10)

    class TimesTwoTransformer(InvertibleDataTransformer[TimeSeries]):
        def __init__(self):
            super().__init__(name="*2 transformer")

        def transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
            super().transform(data, *args, **kwargs)
            return data.map(lambda x: x * 2)

        def inverse_transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
            super().inverse_transform(data, *args, **kwargs)
            return data.map(lambda x: x / 2)

    def test_transform(self):
        # given
        mock1 = self.DataTransformerMock1()
        mock2 = self.DataTransformerMock2()
        data = constant_timeseries(0, 3)
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
        p.transform(constant_timeseries(1, 10))

        # then
        self.assertFalse(mock.transform_called)

    def test_fit(self):
        # given
        data = constant_timeseries(0, 3)
        transformers = [
            self.DataTransformerMock1() for _ in range(10)
        ] + [self.DataTransformerMock2() for _ in range(10)]
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
        data = constant_timeseries(0, 100)
        transformers = [self.DataTransformerMock1() for _ in range(10)]\
            + [self.DataTransformerMock2()]\
            + [self. DataTransformerMock1() for _ in range(10)]
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
        data = constant_timeseries(0, 3)
        transformers = [
            self.DataTransformerMock1() for _ in range(10)
        ] + [self.DataTransformerMock2() for _ in range(10)]
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
        data = constant_timeseries(0., 3)

        transformers = [self.PlusTenTransformer(), self.TimesTwoTransformer()]
        p = Pipeline(transformers)

        # when
        transformed = p.transform(data)
        back = p.inverse_transform(transformed)

        # then
        self.assertEqual(data, back)

    def test_getitem(self):
        # given

        transformers = [self.PlusTenTransformer(name="+10 transformer{}".format(i)) for i in range(0, 10)]
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
        with self.assertRaises(ValueError, msg="transformers should be objects deriving from BaseDataTransformer"):
            Pipeline(input_list)

    def test_raises_on_bad_key(self):
        # given
        bad_key = 12.0
        p = Pipeline([])

        # when & then
        with self.assertRaises(ValueError, msg="Key must be int, str or slice"):
            p[bad_key]
