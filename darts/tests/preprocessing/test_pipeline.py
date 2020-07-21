import unittest

from darts.preprocessing import BaseTransformer, Pipeline
from darts import TimeSeries
from darts.utils.timeseries_generation import constant_timeseries


class PipelineTestCase(unittest.TestCase):
    __test__ = True

    class TransformerMock1(BaseTransformer[TimeSeries]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.validate_called = False
            self.transform_called = False
            self.inverse_transform_called = False
            self.fit_called = False

        def validate(self, data: TimeSeries) -> bool:
            self.validate_called = True
            return super().validate(data)

        def transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
            self.transform_called = True
            return data.append_values(constant_timeseries(1, 3).values())

    class TransformerMock2(BaseTransformer[TimeSeries]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, fittable=True, reversible=True, can_predict=True)
            self.validate_called = False
            self.transform_called = False
            self.inverse_transform_called = False
            self.fit_called = False
            self.args = None
            self.kwargs = None

        def fit(self, data):
            self.fit_called = True
            return self

        def predict(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
            return constant_timeseries(12, 10)

        def validate(self, data: TimeSeries) -> bool:
            self.validate_called = True
            return super().validate(data)

        def transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
            self.transform_called = True
            self.args = args
            self.kwargs = kwargs
            return data.append_values(constant_timeseries(2, 3).values())

        def inverse_transform(self, data: TimeSeries, *args, **kwargs) -> TimeSeries:
            self.inverse_transform_called = True
            self.args = args
            self.kwargs = kwargs
            return data

    def test_transform(self):
        # given
        mock1 = self.TransformerMock1()
        mock2 = self.TransformerMock2()
        data = constant_timeseries(0, 3)
        transformers = [mock1] * 10 + [mock2] * 10
        p = Pipeline(transformers)
        # when
        transformed = p(data)

        # then
        self.assertEqual(63, len(transformed))
        self.assertEqual([0] * 3 + [1] * 30 + [2] * 30, list(transformed.values()))
        for t in transformers:
            self.assertTrue(t.transform_called)
            self.assertFalse(t.inverse_transform_called)

    def test_inverse_raise_exception(self):
        # given
        mock = self.TransformerMock1()
        p = Pipeline([mock])

        # when & then
        self.assertRaises(ValueError, p, None, inverse=True)

    def test_transformers_not_modified(self):
        # given
        mock = self.TransformerMock1()
        p = Pipeline([mock], deep=True)

        # when
        p(constant_timeseries(1, 10))

        # then
        self.assertFalse(mock.transform_called)

    def test_bad_names_raise_error(self):
        # given
        mocks = [self.TransformerMock1() for _ in range(10)]
        names = ["too few", "names", "here"]
        not_unique_names = ["not unique name"] * 10

        # when & then
        self.assertRaises(ValueError, Pipeline, mocks, names=names)
        self.assertRaises(ValueError, Pipeline, mocks, names=not_unique_names)

    def test_set_correct_names(self):
        # given
        mocks = [self.TransformerMock1() for _ in range(3)]
        names = ["mock1", "mock2", "mock3"]

        # when
        p = Pipeline(mocks, names=names)

        # then
        i = 0
        for t, name in p:
            self.assertEqual(names[i], name)
            i += 1

    def test_set_correct_names_when_not_provided(self):
        # given
        mocks = [self.TransformerMock1() for _ in range(3)]

        # when
        p = Pipeline(mocks)

        # then
        i = 0
        for t, name in p:
            self.assertEqual(str(i), name)
            i += 1

    def test_set_transformer_args_kwargs(self):
        # given
        mocks = [self.TransformerMock2() for _ in range(3)]
        p = Pipeline(mocks, names=["a", "b", "c"])

        # when
        p.set_transformer_args_kwargs(0, [1, 2, 3], blabla=True)
        p.set_transformer_args_kwargs("b", [1, 2], blabla=False)
        p.set_transformer_args_kwargs("c", [1], blabla=False, p="X")
        p(constant_timeseries(10, 10))

        # then
        self.assertEqual([1, 2, 3], *mocks[0].args)
        self.assertEqual({"blabla": True}, mocks[0].kwargs)

        self.assertEqual([1, 2], *mocks[1].args)
        self.assertEqual({"blabla": False}, mocks[1].kwargs)

        self.assertEqual([1], *mocks[2].args)
        self.assertEqual({"blabla": False, "p": "X"}, mocks[2].kwargs)

    def test_fit(self):
        # given
        data = constant_timeseries(0, 3)
        transformers = [
            self.TransformerMock1() for _ in range(10)
        ] + [self.TransformerMock2() for _ in range(10)]
        p = Pipeline(transformers)

        # when
        _ = p.fit(data)

        # then
        for i in range(10):
            self.assertFalse(transformers[i].fit_called)
        for i in range(10, 20):
            self.assertTrue(transformers[i].fit_called)

    def test_transform_fit(self):
        # given
        data = constant_timeseries(0, 3)
        transformers = [
            self.TransformerMock1() for _ in range(10)
        ] + [self.TransformerMock2() for _ in range(10)]
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

    def test_predict_raises(self):
        # given
        data = constant_timeseries(0, 3)
        transformers = [
            self.TransformerMock1() for _ in range(10)
        ]
        p = Pipeline(transformers)

        # when & then
        self.assertRaises(ValueError, p.predict, data)
        for t in transformers:
            self.assertFalse(t.transform_called)

    def test_predict(self):
        # given
        data = constant_timeseries(0, 3)
        transformers = [
            self.TransformerMock2() for _ in range(10)
        ]
        p = Pipeline(transformers)

        # when
        pred = p.predict(data)

        # then
        self.assertEqual(10, len(pred))
        self.assertEqual([12] * 10, list(pred.values()))
        for t in transformers:
            self.assertTrue(t.transform_called)

    def test_inverse_transform(self):
        # given
        data = constant_timeseries(0, 3)
        transformers = [
            self.TransformerMock2() for _ in range(10)
        ]
        p = Pipeline(transformers)

        # when
        pred1 = p.inverse_transform(data)
        pred2 = p(data, inverse=True)

        # then
        self.assertEqual(data, pred1)
        self.assertEqual(data, pred2)
        for t in transformers:
            self.assertTrue(t.inverse_transform_called)

    def test_getitem(self):
        # given
        transformers = [
            self.TransformerMock2() for _ in range(10)
        ]
        p = Pipeline(transformers)

        # then
        def common_routine(p, key, expected_names):
            i = 0
            p1 = p[key]
            for t, name in p1:
                self.assertEqual(expected_names[i], name)
                i += 1

        # when
        keys = [
            slice(1, 5, 1),
            "2",
            3,
            slice(None, None, -1)
        ]
        expected_names = [
            ["1", "2", "3", "4", "5"],
            ["2"],
            ["3"],
            ["9", "8", "7", "6", "5", "4", "3", "2", "1", "0"]
        ]

        for key, en in zip(keys, expected_names):
            common_routine(p, key, en)

    def test_raises_on_non_transformers(self):
        # given
        input_list = list(range(10))

        # when & then
        with self.assertRaises(ValueError, msg="transformers should be objects deriving from BaseTransformer"):
            Pipeline(input_list)

    def test_raises_on_bad_key(self):
        # given
        bad_key = 12.0
        p = Pipeline([])

        # when & then
        with self.assertRaises(ValueError, msg="Key must be int, str or slice"):
            p[bad_key]
