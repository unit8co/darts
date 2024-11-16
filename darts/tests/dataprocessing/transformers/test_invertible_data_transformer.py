from collections.abc import Mapping, Sequence
from typing import Any, Union

import numpy as np

from darts import TimeSeries
from darts.dataprocessing.transformers.invertible_data_transformer import (
    InvertibleDataTransformer,
)
from darts.utils.timeseries_generation import constant_timeseries


class TestInvertibleDataTransformer:
    class DataTransformerMock(InvertibleDataTransformer):
        def __init__(
            self,
            scale: float,
            translation: float,
            stack_samples: bool = False,
            mask_components: bool = True,
            parallel_params: Union[bool, Sequence[str]] = False,
        ):
            """
            Applies the (invertible) transform `transformed_series = scale * series + translation`.

            Parameters
            ----------
            scale
                Scale coefficient of transform.
            translation
                Translational constant of transform.
            stack_samples
                Whether to call `stack_samples` inside of `ts_transform`.
            mask_components
                Whether to automatically apply any provided `component_mask` key word arguments. See
                `BaseDataTransformer` docstring for further details.
            parallel_params
                Specifies which parameters should vary between different parallel jobs, supposing that
                multiple time series are given to `ts_transform`. See `BaseDataTransformer` docstring
                for further details.

            """
            self._scale = scale
            self._translation = translation
            self._stack_samples = stack_samples
            self._mask_components = mask_components
            super().__init__(
                name="DataTransformerMock",
                mask_components=mask_components,
                parallel_params=parallel_params,
            )

        @staticmethod
        def ts_transform(
            series: TimeSeries, params: Mapping[str, Any], **kwargs
        ) -> TimeSeries:
            """
            Implements the transform `scale * series + translation`.

            If `component_mask` is in `kwargs`, this is manually applied and unapplied. If
            `_stack_samples = True` in `params['fixed']`, then `stack_samples` and `unstack_samples`
            all used when computing this transformation.

            """
            fixed_params = params["fixed"]
            stack_samples = fixed_params["_stack_samples"]
            mask_components = fixed_params["_mask_components"]
            scale, translation = fixed_params["_scale"], fixed_params["_translation"]

            # Ensure `component_mask` not passed via `kwargs` if specified `mask_components=True`
            # when transform created
            if mask_components:
                assert "component_mask" not in kwargs

            # Ensure manual masking only performed when `mask_components = False`
            # when transform constructed:
            if not mask_components and ("component_mask" in kwargs):
                vals = TestInvertibleDataTransformer.DataTransformerMock.apply_component_mask(
                    series, kwargs["component_mask"], return_ts=False
                )
            else:
                vals = series.all_values()

            if stack_samples:
                vals = TestInvertibleDataTransformer.DataTransformerMock.stack_samples(
                    vals
                )
            vals = scale * vals + translation
            if stack_samples:
                vals = (
                    TestInvertibleDataTransformer.DataTransformerMock.unstack_samples(
                        vals, series=series
                    )
                )

            if not mask_components and ("component_mask" in kwargs):
                vals = TestInvertibleDataTransformer.DataTransformerMock.unapply_component_mask(
                    series, vals, kwargs["component_mask"]
                )

            return series.with_values(vals)

        @staticmethod
        def ts_inverse_transform(
            series: TimeSeries, params: Mapping[str, Any], **kwargs
        ) -> TimeSeries:
            """
            Implements the inverse transform `(series - translation) / scale`.

            If `component_mask` is in `kwargs`, this is manually applied and unapplied. If
            `_stack_samples = True` in `params['fixed']`, then `stack_samples` and `unstack_samples`
            all used when computing this transformation.

            """
            fixed_params = params["fixed"]
            stack_samples = fixed_params["_stack_samples"]
            mask_components = fixed_params["_mask_components"]
            scale, translation = fixed_params["_scale"], fixed_params["_translation"]

            # Ensure `component_mask` not passed via `kwargs` if specified `mask_components=True`
            # when transform created
            if mask_components:
                assert "component_mask" not in kwargs

            # Ensure manual masking only performed when `mask_components = False`
            # when transform constructed:
            if not mask_components and ("component_mask" in kwargs):
                vals = TestInvertibleDataTransformer.DataTransformerMock.apply_component_mask(
                    series, kwargs["component_mask"], return_ts=False
                )
            else:
                vals = series.all_values()

            if stack_samples:
                vals = TestInvertibleDataTransformer.DataTransformerMock.stack_samples(
                    vals
                )
            vals = (vals - translation) / scale
            if stack_samples:
                vals = (
                    TestInvertibleDataTransformer.DataTransformerMock.unstack_samples(
                        vals, series=series
                    )
                )

            if not mask_components and ("component_mask" in kwargs):
                vals = TestInvertibleDataTransformer.DataTransformerMock.unapply_component_mask(
                    series, vals, kwargs["component_mask"]
                )

            return series.with_values(vals)

    def test_input_transformed_single_series(self):
        """
        Tests for correct (inverse) transformation of single series.
        """
        test_input = constant_timeseries(value=1, length=10)

        mock = self.DataTransformerMock(scale=2, translation=10)

        transformed = mock.transform(test_input)

        # 2 * 1 + 10 = 12
        expected = constant_timeseries(value=12, length=10)
        assert transformed == expected

        # Should get input back:
        assert mock.inverse_transform(transformed) == test_input

    def test_input_transformed_multiple_series(self):
        """
        Tests for correct transformation of multiple series when
        different param values are used for different parallel
        jobs (i.e. test that `parallel_params` argument is treated
        correctly). Also tests that transformer correctly handles
        being provided with fewer input series than fixed parameter
        value sets.
        """
        test_input_1 = constant_timeseries(value=1, length=10)
        test_input_2 = constant_timeseries(value=2, length=11)

        # Don't have different params for different jobs:
        mock = self.DataTransformerMock(scale=2, translation=10, parallel_params=False)
        (transformed_1, transformed_2) = mock.transform((test_input_1, test_input_2))
        # 2 * 1 + 10 = 12
        assert transformed_1 == constant_timeseries(value=12, length=10)
        # 2 * 2 + 10 = 14
        assert transformed_2 == constant_timeseries(value=14, length=11)
        # Should get input back:
        inv_1, inv_2 = mock.inverse_transform((transformed_1, transformed_2))
        assert inv_1 == test_input_1
        assert inv_2 == test_input_2

        # If only one timeseries provided, should apply parameters defined for
        # for the first to that series:
        transformed_1 = mock.transform(test_input_1)
        assert transformed_1 == constant_timeseries(value=12, length=10)
        inv_1 = mock.inverse_transform(transformed_1)
        assert inv_1 == test_input_1

        # Have different `scale` param for different jobs:
        mock = self.DataTransformerMock(
            scale=(2, 3), translation=10, parallel_params=["_scale"]
        )
        (transformed_1, transformed_2) = mock.transform((test_input_1, test_input_2))
        # 2 * 1 + 10 = 12
        assert transformed_1 == constant_timeseries(value=12, length=10)
        # 3 * 2 + 10 = 16
        assert transformed_2 == constant_timeseries(value=16, length=11)
        # Should get input back:
        inv_1, inv_2 = mock.inverse_transform((transformed_1, transformed_2))
        assert inv_1 == test_input_1
        assert inv_2 == test_input_2

        # Have different `scale`, `translation`, and `stack_samples` params for different jobs:
        mock = self.DataTransformerMock(
            scale=(2, 3),
            translation=(10, 11),
            stack_samples=(False, True),
            mask_components=(False, False),
            parallel_params=True,
        )
        (transformed_1, transformed_2) = mock.transform((test_input_1, test_input_2))
        # 2 * 1 + 10 = 12
        assert transformed_1 == constant_timeseries(value=12, length=10)
        # 3 * 2 + 11 = 17
        assert transformed_2 == constant_timeseries(value=17, length=11)
        # Should get input back:
        inv_1, inv_2 = mock.inverse_transform((transformed_1, transformed_2))
        assert inv_1 == test_input_1
        assert inv_2 == test_input_2
        # If only one timeseries provided, should apply parameters defined for
        # for the first to that series:
        transformed_1 = mock.transform(test_input_1)
        assert transformed_1 == constant_timeseries(value=12, length=10)
        inv_1 = mock.inverse_transform(transformed_1)
        assert inv_1 == test_input_1

        # Specify three sets of fixed params, but pass only one or two series as inputs
        # to `transform`/`inverse_transform`; transformer should apply `i`th set of fixed
        # params to the `i`th input passed to `transform`
        mock = self.DataTransformerMock(
            scale=(2, 3, 4),
            translation=(10, 11, 12),
            stack_samples=(False, True, False),
            mask_components=(False, False, False),
            parallel_params=True,
        )
        # If single series provided to transformer with three sets of
        # fixed params, should transform/inverse transform using the first set of fixed
        # parameters:
        transformed_1 = mock.transform(test_input_1)
        assert transformed_1 == constant_timeseries(value=12, length=10)
        inv_1 = mock.inverse_transform(transformed_1)
        assert inv_1 == test_input_1
        # If two series provided to transformer with three sets of
        # fixed params, should transform/inverse transform using the first and
        # second set of fixed parameters:
        transformed_1, transformed_2 = mock.transform((test_input_1, test_input_2))
        assert transformed_1 == constant_timeseries(value=12, length=10)
        assert transformed_2 == constant_timeseries(value=17, length=11)
        inv_1, inv_2 = mock.inverse_transform((transformed_1, transformed_2))
        assert inv_1 == test_input_1
        assert inv_2 == test_input_2

    def test_input_transformed_list_of_lists_of_series(self):
        """
        Tests for correct transformation of multiple series when
        different param values are used for different parallel
        jobs (i.e. test that `parallel_params` argument is treated
        correctly). Also tests that transformer correctly handles
        being provided with fewer input series than fixed parameter
        value sets.
        """
        test_input_1 = constant_timeseries(value=1, length=10)
        test_input_2 = constant_timeseries(value=2, length=11)

        # Don't have different params for different jobs:
        mock = self.DataTransformerMock(scale=2, translation=10, parallel_params=False)
        (transformed_1, transformed_2) = mock.transform((test_input_1, test_input_2))
        # 2 * 1 + 10 = 12
        assert transformed_1 == constant_timeseries(value=12, length=10)
        # 2 * 2 + 10 = 14
        assert transformed_2 == constant_timeseries(value=14, length=11)

        # list of lists of series must get input back
        inv = mock.inverse_transform([[transformed_1], [transformed_2]])
        assert len(inv) == 2
        assert all(
            isinstance(series_list, list) and len(series_list) == 1
            for series_list in inv
        )
        assert all(
            isinstance(series, TimeSeries)
            for series_list in inv
            for series in series_list
        )
        assert inv[0][0] == test_input_1
        assert inv[1][0] == test_input_2

        # one list of lists of is longer than others, must get input back
        inv = mock.inverse_transform([[transformed_1, transformed_1], [transformed_2]])
        assert len(inv) == 2
        assert len(inv[0]) == 2 and len(inv[1]) == 1
        assert all(isinstance(series_list, list) for series_list in inv)
        assert all(
            isinstance(series, TimeSeries)
            for series_list in inv
            for series in series_list
        )
        assert inv[0][0] == test_input_1
        assert inv[0][1] == test_input_1
        assert inv[1][0] == test_input_2

        # different types of Sequences, must get input back
        inv = mock.inverse_transform(((transformed_1, transformed_1), (transformed_2,)))
        assert len(inv) == 2
        assert len(inv[0]) == 2 and len(inv[1]) == 1
        assert all(isinstance(series_list, list) for series_list in inv)
        assert all(
            isinstance(series, TimeSeries)
            for series_list in inv
            for series in series_list
        )
        assert inv[0][0] == test_input_1
        assert inv[0][1] == test_input_1
        assert inv[1][0] == test_input_2

        # one list of lists is empty, returns empty list as well
        inv = mock.inverse_transform([[], [transformed_2, transformed_2]])
        assert len(inv) == 2
        assert len(inv[0]) == 0 and len(inv[1]) == 2
        assert all(isinstance(series_list, list) for series_list in inv)
        assert all(isinstance(series, TimeSeries) for series in inv[1])
        assert inv[1][0] == test_input_2
        assert inv[1][1] == test_input_2

        # more list of lists than used during transform works
        inv = mock.inverse_transform([
            [transformed_1],
            [transformed_2],
            [transformed_2],
        ])
        assert len(inv) == 3
        assert all(
            isinstance(series_list, list) and len(series_list) == 1
            for series_list in inv
        )
        assert all(
            isinstance(series, TimeSeries)
            for series_list in inv
            for series in series_list
        )
        assert inv[0][0] == test_input_1
        assert inv[1][0] == test_input_2
        assert inv[2][0] == test_input_2

    def test_input_transformed_multiple_samples(self):
        """
        Tests that `stack_samples` and `unstack_samples` correctly
        implemented when considering multi-sample timeseries.
        """
        test_input = constant_timeseries(value=1, length=10)
        test_input = test_input.concatenate(
            constant_timeseries(value=2, length=10), axis="sample"
        )

        mock = self.DataTransformerMock(scale=2, translation=10, stack_samples=True)
        transformed = mock.transform(test_input)

        # 2 * 1 + 10 = 12
        expected = constant_timeseries(value=12, length=10)
        # 2 * 2 + 10 = 14
        expected = expected.concatenate(
            constant_timeseries(value=14, length=10), axis="sample"
        )
        assert transformed == expected
        # Should get input back:
        inv = mock.inverse_transform(transformed)
        assert inv == test_input

    def test_input_transformed_masking(self):
        """
        Tests that automatic component masking is correctly implemented,
        and that manual component masking is also handled correctly
        through `kwargs` + `apply_component_mask`/`unapply_component_mask`
        methods.
        """
        test_input = TimeSeries.from_values(np.ones((4, 3, 5)))
        mask = np.array([True, False, True])
        # Second component should be untransformed:
        scale = 2
        translation = 10
        expected = np.stack(
            [12 * np.ones((4, 5)), np.ones((4, 5)), 12 * np.ones((4, 5))], axis=1
        )
        expected = TimeSeries.from_values(expected)

        # Automatically apply component mask:
        mock = self.DataTransformerMock(
            scale=scale, translation=translation, mask_components=True
        )
        transformed = mock.transform(test_input, component_mask=mask)
        assert transformed == expected
        # Should get input back:
        inv = mock.inverse_transform(transformed, component_mask=mask)
        assert inv == test_input

        # Manually apply component mask:
        mock = self.DataTransformerMock(scale=2, translation=10, mask_components=False)
        transformed = mock.transform(test_input, component_mask=mask)
        assert transformed == expected
        # Should get input back:
        inv = mock.inverse_transform(transformed, component_mask=mask)
        assert inv == test_input
