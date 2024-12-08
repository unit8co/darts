from collections.abc import Mapping, Sequence
from typing import Any, Union

import numpy as np
import pytest

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


class TestPipeline:
    class DataTransformerMock1(BaseDataTransformer):
        def __init__(self):
            super().__init__()
            self.transform_called = False
            self.inverse_transform_called = False
            self.fit_called = False

        @staticmethod
        def ts_transform(data: TimeSeries, params: Mapping[str, Any]) -> TimeSeries:
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
        def ts_fit(series: TimeSeries, params: Mapping[str, Any]):
            pass

        @staticmethod
        def ts_transform(series: TimeSeries, params: Mapping[str, Any]) -> TimeSeries:
            return series.append_values(constant_timeseries(value=2, length=3).values())

        @staticmethod
        def ts_inverse_transform(
            series: TimeSeries, params: Mapping[str, Any]
        ) -> TimeSeries:
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
        def ts_transform(series: TimeSeries, params: Mapping[str, Any]) -> TimeSeries:
            return series.map(lambda x: x + 10)

        @staticmethod
        def ts_inverse_transform(
            series: TimeSeries, params: Mapping[str, Any]
        ) -> TimeSeries:
            return series.map(lambda x: x - 10)

    class TimesTwoTransformer(InvertibleDataTransformer):
        def __init__(self):
            super().__init__(name="*2 transformer")

        @staticmethod
        def ts_transform(data: TimeSeries, params: Mapping[str, Any]) -> TimeSeries:
            return data.map(lambda x: x * 2)

        @staticmethod
        def ts_inverse_transform(
            data: TimeSeries, params: Mapping[str, Any]
        ) -> TimeSeries:
            return data.map(lambda x: x / 2)

    class ExtendTransformer(FittableDataTransformer, InvertibleDataTransformer):
        def __init__(self, global_fit: bool, coef: int):
            self.coef = coef
            super().__init__(
                name="fittable extending transformer", global_fit=global_fit
            )

        @staticmethod
        def ts_fit(
            series: Union[TimeSeries, Sequence[TimeSeries]],
            params: Mapping[str, Any],
            *args,
            **kwargs,
        ):
            coef = params["fixed"]["coef"]
            if isinstance(series, Sequence):
                return sum(ts.values()[0] for ts in series) + coef
            else:
                return series.values()[0] + coef

        @staticmethod
        def ts_transform(data: TimeSeries, params: Mapping[str, Any]) -> TimeSeries:
            return data + params["fitted"]

        @staticmethod
        def ts_inverse_transform(
            data: TimeSeries, params: Mapping[str, Any]
        ) -> TimeSeries:
            return data - params["fitted"]

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
        assert 63 == len(transformed)
        assert [0] * 3 + [1] * 30 + [2] * 30 == list(transformed.values())
        for t in transformers:
            assert t.transform_called
            assert not t.inverse_transform_called

    def test_transform_prefitted(self):
        """Check that when multiple series are passed to fit transformers with global_fit=False,
        transform behave as expected when series_idx is specified.

        Note: the transformers are fitted independently to make the expected results more intuitive
        """
        data = [
            constant_timeseries(value=0, length=2),
            constant_timeseries(value=10, length=2),
        ]

        def get_transf(global_fit: bool, fit: bool, coef: int):
            transf = self.ExtendTransformer(global_fit=global_fit, coef=coef)
            if fit:
                transf.fit(data)
            return transf

        # multiple series, global_fit=False
        p = Pipeline([
            get_transf(global_fit=False, fit=True, coef=1),
            get_transf(global_fit=False, fit=True, coef=5),
        ])
        transformed = p.transform(data)

        # ts + (data[0][0] + 1) + (data[0][0] + 5) = 6
        np.testing.assert_array_almost_equal(
            transformed[0].values(), np.array([[6, 6]]).T
        )
        # ts + (data[1][0] + 1) + (data[1][0] + 5) = 10 + 11 + 15 = 36
        np.testing.assert_array_almost_equal(
            transformed[1].values(), np.array([[36, 36]]).T
        )
        # implicitly use the first params of each transformer
        np.testing.assert_array_almost_equal(
            transformed[0].values(), p.transform(data[0]).values()
        )
        # explicitly use the first params of each transformer
        np.testing.assert_array_almost_equal(
            transformed[0].values(), p.transform(data[0], series_idx=[0]).values()
        )
        # implicitly use the first params of each transformer
        # ts + (data[0][0] + 1) + (data[0][0] + 5) = 10 + 1 + 5 = 16
        np.testing.assert_array_almost_equal(
            np.array([[16, 16]]).T, p.transform(data[1]).values()
        )
        # explicitly use the second params of each transformer
        np.testing.assert_array_almost_equal(
            transformed[1].values(), p.transform(data[1], series_idx=[1]).values()
        )

        # multiple series, mixture of local and global transformers
        p = Pipeline([
            get_transf(global_fit=False, fit=True, coef=1),
            get_transf(global_fit=True, fit=True, coef=90),
        ])
        transformed = p.transform(data)
        # ts + (data[0][0] + 1) + (sum(data[;, 0]) + 90) = 0 + 1 + 100
        np.testing.assert_array_almost_equal(
            transformed[0].values(), np.array([[101, 101]]).T
        )
        # ts + (data[1][0] + 1) + (sum(data[;, 0]) + 90) = 10 + 11 + 100
        np.testing.assert_array_almost_equal(
            transformed[1].values(), np.array([[121, 121]]).T
        )
        # implicitly use the first params of first transformer, the second is global
        np.testing.assert_array_almost_equal(
            transformed[0].values(), p.transform(data[0]).values()
        )
        # explicitly use the first params of first transformer, the second is global
        np.testing.assert_array_almost_equal(
            transformed[0].values(), p.transform(data[0], series_idx=[0]).values()
        )
        # implicitly use the first params of first transformer, the second is global
        # ts + (data[0][0] + 1) + (sum(data[;, 0]) + 90) = 10 + 1 + 100
        np.testing.assert_array_almost_equal(
            np.array([[111, 111]]).T, p.transform(data[1]).values()
        )
        # explicitly use the second params of first transformer, the second is global
        np.testing.assert_array_almost_equal(
            transformed[1].values(), p.transform(data[1], series_idx=[1]).values()
        )

        # reversing input, and explicitly selecting reversed indexes
        assert transformed[::-1] == p.transform(data[::-1], series_idx=[1, 0])

    def test_inverse_raise_exception(self):
        # given
        mock = self.DataTransformerMock1()
        p = Pipeline([mock])

        # when & then
        with pytest.raises(ValueError):
            p.inverse_transform(None)

    def test_transformers_not_modified(self):
        # given
        mock = self.DataTransformerMock1()
        p = Pipeline([mock], copy=True)

        # when
        p.transform(constant_timeseries(value=1, length=10))

        # then
        assert not mock.transform_called

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
            assert not transformers[i].fit_called
        for i in range(10, 20):
            assert transformers[i].fit_called

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
            assert transformers[i].transform_called
        assert transformers[10].fit_called
        assert not transformers[10].transform_called
        for i in range(11, 21):
            assert not transformers[i].transform_called

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
            assert t.transform_called
        for i in range(10):
            assert not transformers[i].fit_called
        for i in range(10, 20):
            assert transformers[i].fit_called

    def test_inverse_transform(self):
        # given
        data = constant_timeseries(value=0.0, length=3)

        transformers = [self.PlusTenTransformer(), self.TimesTwoTransformer()]
        p = Pipeline(transformers)

        # when
        transformed = p.transform(data)
        back = p.inverse_transform(transformed)

        # then
        assert data == back

    def test_inverse_transform_prefitted(self):
        """Check that when multiple series are passed to fit transformers with global_fit=False,
        inverse_transform behave as expected when series_idx is specified.

        Note: the transformers are fitted independently to make the expected results more intuitive
        """
        data = [
            constant_timeseries(value=0, length=2),
            constant_timeseries(value=10, length=2),
        ]

        def get_transf(global_fit: bool, fit: bool, coef: int):
            transf = self.ExtendTransformer(global_fit=global_fit, coef=coef)
            if fit:
                transf.fit(data)
            return transf

        # multiple series, global_fit=False
        p = Pipeline([
            get_transf(global_fit=False, fit=True, coef=1),
            get_transf(global_fit=False, fit=True, coef=5),
        ])
        transformed = p.transform(data)

        # implicitly use the first params of each transformer
        np.testing.assert_array_almost_equal(
            data[0].values(), p.inverse_transform(transformed[0]).values()
        )
        # explicitly use the first params of each transformer
        np.testing.assert_array_almost_equal(
            data[0].values(),
            p.inverse_transform(transformed[0], series_idx=[0]).values(),
        )

        # 10 + 11 + 15
        np.testing.assert_array_almost_equal(
            np.array([[36, 36]]).T, transformed[1].values()
        )
        # implicitly use the first params of each transformer
        # inverse_transform[0][0] = lambda x: x - 1, inverse_transform[1][0] = lambda x: x - 5
        np.testing.assert_array_almost_equal(
            np.array([[30, 30]]).T, p.inverse_transform(transformed[1]).values()
        )
        np.testing.assert_array_almost_equal(
            np.array([[30, 30]]).T,
            p.inverse_transform(transformed[1], series_idx=0).values(),
        )
        # explicitly use the second params of each transformer
        # inverse_transform[0][0] = lambda x: x - 11, inverse_transform[1][0] = lambda x: x - 15
        np.testing.assert_array_almost_equal(
            data[1].values(), p.inverse_transform(transformed[1], series_idx=1).values()
        )

        # multiple series, mixture of local and global transformers
        p = Pipeline([
            get_transf(global_fit=False, fit=True, coef=1),
            get_transf(global_fit=True, fit=True, coef=90),
        ])
        transformed = p.transform(data)

        # implicitly use the first params of each transformer
        np.testing.assert_array_almost_equal(
            data[0].values(), p.inverse_transform(transformed[0]).values()
        )
        # explicitly use the first params of each transformer
        np.testing.assert_array_almost_equal(
            data[0].values(),
            p.inverse_transform(transformed[0], series_idx=[0]).values(),
        )
        # 10 + 11 + 100
        np.testing.assert_array_almost_equal(
            np.array([[121, 121]]).T, transformed[1].values()
        )

        # implicitly use the first params of each transformer
        # inverse_transform[0][0] = lambda x: x - 1, inverse_transform = lambda x: x - 100
        np.testing.assert_array_almost_equal(
            np.array([[20, 20]]).T, p.inverse_transform(transformed[1]).values()
        )
        # explicitly use the second params of each transformer
        # inverse_transform[0][1] = lambda x: x - 11, inverse_transform = lambda x: x - 100
        np.testing.assert_array_almost_equal(
            data[1].values(),
            p.inverse_transform(transformed[1], series_idx=[1]).values(),
        )

        # reversing input, and explicitly selecting reversed indexes
        assert transformed[::-1] == p.transform(data[::-1], series_idx=[1, 0])

    def test_getitem(self):
        # given

        transformers = [
            self.PlusTenTransformer(name=f"+10 transformer{i}") for i in range(0, 10)
        ]
        p = Pipeline(transformers)

        # when & then
        # note : only compares string representations, since __getitem__() copies the transformers
        assert str(p[1]._transformers) == str([transformers[1]])
        assert str(p[4:8]._transformers) == str(transformers[4:8])

        with pytest.raises(ValueError):
            p["invalid attempt"]

    def test_raises_on_non_transformers(self):
        # given
        input_list = list(range(10))

        # when & then
        with pytest.raises(ValueError) as err:
            Pipeline(input_list)

        assert (
            str(err.value)
            == "transformers should be objects deriving from BaseDataTransformer"
        )

    def test_raises_on_bad_key(self):
        # given
        bad_key = 12.0
        p = Pipeline([])

        # when & then
        with pytest.raises(ValueError) as err:
            p[bad_key]
        assert str(err.value) == "key must be either an int or a slice"

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
        assert data == back

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
        with pytest.raises(ValueError):
            pipeline.inverse_transform(transformed)

        back = pipeline.inverse_transform(transformed, partial=True)

        # while the +/- 2 is inverted, the +10 operation is not
        assert series_plus_ten == back

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
            assert transformer._verbose == verbose_value

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
            assert transformer._n_jobs == n_jobs_value
