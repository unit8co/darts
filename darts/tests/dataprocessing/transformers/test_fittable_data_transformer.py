import logging
import unittest
from typing import Any, Mapping, Sequence, Union

import numpy as np

from darts import TimeSeries
from darts.dataprocessing.transformers.fittable_data_transformer import (
    FittableDataTransformer,
)
from darts.utils.timeseries_generation import constant_timeseries


class BaseDataTransformerTestCase(unittest.TestCase):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    class DataTransformerMock(FittableDataTransformer):
        def __init__(
            self,
            scale: float,
            translation: float,
            stack_samples: bool = False,
            mask_components: bool = True,
            parallel_params: Union[bool, Sequence[str]] = False,
        ):
            """
            Applies the transform `transformed_series = scale * series + translation`.
            When 'fitting' this transform, the `scale` and `translation` fixed parameters are returned.

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
        def ts_fit(series: TimeSeries, params: Mapping[str, Any], **kwargs):
            """
            'Fits' transform by returning `scale` and `translation` fixed params.
            """
            # Ensure `component_mask` not passed via `kwargs` if specified `mask_components=True`
            # when transform created
            mask_components = params["fixed"]["_mask_components"]
            if mask_components:
                assert "component_mask" not in kwargs
            # Ensure 'fitted' is not in `params`:
            assert "fitted" not in params
            scale, translation = (
                params["fixed"]["_scale"],
                params["fixed"]["_translation"],
            )
            return scale, translation

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
            stack_samples = params["fixed"]["_stack_samples"]
            mask_components = params["fixed"]["_mask_components"]
            scale, translation = params["fitted"]

            # Ensure `component_mask` not passed via `kwargs` if specified `mask_components=True`
            # when transform created
            if mask_components:
                assert "component_mask" not in kwargs

            # Ensure manual masking only performed when `mask_components = False`
            # when transform constructed:
            if not mask_components and ("component_mask" in kwargs):
                vals = BaseDataTransformerTestCase.DataTransformerMock.apply_component_mask(
                    series, kwargs["component_mask"], return_ts=False
                )
            else:
                vals = series.all_values()

            if stack_samples:
                vals = BaseDataTransformerTestCase.DataTransformerMock.stack_samples(
                    vals
                )
            vals = scale * vals + translation
            if stack_samples:
                vals = BaseDataTransformerTestCase.DataTransformerMock.unstack_samples(
                    vals, series=series
                )

            if not mask_components and ("component_mask" in kwargs):
                vals = BaseDataTransformerTestCase.DataTransformerMock.unapply_component_mask(
                    series, vals, kwargs["component_mask"]
                )

            return series.with_values(vals)

    def test_input_transformed_single_series(self):
        """
        Tests for correct transformation of single series.
        """
        test_input = constant_timeseries(value=1, length=10)

        mock = self.DataTransformerMock(scale=2, translation=10)

        transformed = mock.fit_transform(test_input)

        # 2 * 1 + 10 = 12
        expected = constant_timeseries(value=12, length=10)
        self.assertEqual(transformed, expected)

    def test_input_transformed_multiple_series(self):
        """
        Tests for correct transformation of multiple series when
        different param values are used for different parallel
        jobs (i.e. test that `parallel_params` argument is treated
        correctly). Also tests that transformer correctly handles
        being provided with fewer input series than series used
        to fit the transformer.
        """
        test_input_1 = constant_timeseries(value=1, length=10)
        test_input_2 = constant_timeseries(value=2, length=11)
        test_input_3 = constant_timeseries(value=3, length=12)

        # Don't have different params for different jobs:
        mock = self.DataTransformerMock(scale=2, translation=10, parallel_params=False)
        (transformed_1, transformed_2) = mock.fit_transform(
            (test_input_1, test_input_2)
        )
        # 2 * 1 + 10 = 12
        self.assertEqual(transformed_1, constant_timeseries(value=12, length=10))
        # 2 * 2 + 10 = 14
        self.assertEqual(transformed_2, constant_timeseries(value=14, length=11))

        # Have different `scale` param for different jobs:
        mock = self.DataTransformerMock(
            scale=(2, 3), translation=10, parallel_params=["_scale"]
        )
        (transformed_1, transformed_2) = mock.fit_transform(
            (test_input_1, test_input_2)
        )
        # 2 * 1 + 10 = 12
        self.assertEqual(transformed_1, constant_timeseries(value=12, length=10))
        # 3 * 2 + 10 = 16
        self.assertEqual(transformed_2, constant_timeseries(value=16, length=11))

        # If only one timeseries provided, should apply parameters defined for
        # for the first to that series:
        transformed_1 = mock.transform(test_input_1)
        self.assertEqual(transformed_1, constant_timeseries(value=12, length=10))

        # Have different `scale`, `translation`, and `stack_samples` params for different jobs:
        mock = self.DataTransformerMock(
            scale=(2, 3),
            translation=(10, 11),
            stack_samples=(False, True),
            mask_components=(False, False),
            parallel_params=True,
        )
        (transformed_1, transformed_2) = mock.fit_transform(
            (test_input_1, test_input_2)
        )
        # 2 * 1 + 10 = 12
        self.assertEqual(transformed_1, constant_timeseries(value=12, length=10))
        # 3 * 2 + 11 = 17
        self.assertEqual(transformed_2, constant_timeseries(value=17, length=11))

        # If only one timeseries provided, should apply parameters defined for
        # for the first to that series:
        transformed_1 = mock.transform(test_input_1)
        self.assertEqual(transformed_1, constant_timeseries(value=12, length=10))

        # Train on three series with three different fixed param values,
        # but pass only one or two series as inputs to `transform`;
        # transformer should use the parameters fitted to `i`th series to
        # apply `transform`/`inverse_transform` to the `i`th input:
        mock = self.DataTransformerMock(
            scale=(2, 3, 4),
            translation=(10, 11, 12),
            stack_samples=(False, True, False),
            mask_components=(False, False, False),
            parallel_params=True,
        )
        mock.fit([test_input_1, test_input_2, test_input_3])
        # If single series provided to transformer trained on three
        # series, should transform using the first set of fitted parameters:
        transformed_1 = mock.transform(test_input_1)
        self.assertEqual(transformed_1, constant_timeseries(value=12, length=10))
        # If two series provided to transformer trained on three
        # series, should transform using the first and second set of
        # fitted parameters:
        transformed_1, transformed_2 = mock.transform((test_input_1, test_input_2))
        self.assertEqual(transformed_1, constant_timeseries(value=12, length=10))
        self.assertEqual(transformed_2, constant_timeseries(value=17, length=11))

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
        transformed = mock.fit_transform(test_input)

        # 2 * 1 + 10 = 12
        expected = constant_timeseries(value=12, length=10)
        # 2 * 2 + 10 = 14
        expected = expected.concatenate(
            constant_timeseries(value=14, length=10), axis="sample"
        )
        self.assertEqual(transformed, expected)

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
        transformed = mock.fit_transform(test_input, component_mask=mask)
        self.assertEqual(transformed, expected)

        # Manually apply component mask:
        mock = self.DataTransformerMock(scale=2, translation=10, mask_components=False)
        transformed = mock.fit_transform(test_input, component_mask=mask)
        self.assertEqual(transformed, expected)
