from collections.abc import Mapping, Sequence
from typing import Any, Union

import numpy as np

from darts import TimeSeries
from darts.dataprocessing.transformers.fittable_data_transformer import (
    FittableDataTransformer,
)
from darts.utils.timeseries_generation import constant_timeseries


class TestLocalFittableDataTransformer:
    """
    Tests that data transformers inheriting from `FittableDataTransformer` class behave
    correctly when `global_fit` attribute is `False`.
    """

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
                vals = TestLocalFittableDataTransformer.DataTransformerMock.apply_component_mask(
                    series, kwargs["component_mask"], return_ts=False
                )
            else:
                vals = series.all_values()

            if stack_samples:
                vals = (
                    TestLocalFittableDataTransformer.DataTransformerMock.stack_samples(
                        vals
                    )
                )
            vals = scale * vals + translation
            if stack_samples:
                vals = TestLocalFittableDataTransformer.DataTransformerMock.unstack_samples(
                    vals, series=series
                )

            if not mask_components and ("component_mask" in kwargs):
                vals = TestLocalFittableDataTransformer.DataTransformerMock.unapply_component_mask(
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
        assert transformed == expected

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
        (transformed_1, transformed_2) = mock.fit_transform((
            test_input_1,
            test_input_2,
        ))
        # 2 * 1 + 10 = 12
        assert transformed_1 == constant_timeseries(value=12, length=10)
        # 2 * 2 + 10 = 14
        assert transformed_2 == constant_timeseries(value=14, length=11)

        # Have different `scale` param for different jobs:
        mock = self.DataTransformerMock(
            scale=(2, 3), translation=10, parallel_params=["_scale"]
        )
        (transformed_1, transformed_2) = mock.fit_transform((
            test_input_1,
            test_input_2,
        ))
        # 2 * 1 + 10 = 12
        assert transformed_1 == constant_timeseries(value=12, length=10)
        # 3 * 2 + 10 = 16
        assert transformed_2 == constant_timeseries(value=16, length=11)

        # If only one timeseries provided, should apply parameters defined for
        # for the first to that series:
        assert mock.transform(test_input_1) == constant_timeseries(value=12, length=10)
        # 2 * 2 + 10 = 14
        assert mock.transform(test_input_2) == constant_timeseries(value=14, length=11)

        # If the index of another set of parameters is provided, the output changes accordingly:
        # 3 * 1 + 10 = 13
        assert mock.transform(test_input_1, series_idx=1) == constant_timeseries(
            value=13, length=10
        )
        # 3 * 2 + 10 = 16
        assert mock.transform(test_input_2, series_idx=1) == constant_timeseries(
            value=16, length=11
        )

        # Have different `scale`, `translation`, and `stack_samples` params for different jobs:
        mock = self.DataTransformerMock(
            scale=(2, 3),
            translation=(10, 11),
            stack_samples=(False, True),
            mask_components=(False, False),
            parallel_params=True,
        )
        (transformed_1, transformed_2) = mock.fit_transform((
            test_input_1,
            test_input_2,
        ))
        # 2 * 1 + 10 = 12
        assert transformed_1 == constant_timeseries(value=12, length=10)
        # 3 * 2 + 11 = 17
        assert transformed_2 == constant_timeseries(value=17, length=11)

        # If only one timeseries provided, should apply parameters defined for
        # for the first to that series:
        assert mock.transform(test_input_1) == constant_timeseries(value=12, length=10)
        # 2 * 2 + 10 = 14
        assert mock.transform(test_input_2) == constant_timeseries(value=14, length=11)

        # If the index of another set of parameters is provided, the output changes accordingly:
        assert mock.transform(test_input_1, series_idx=0) == constant_timeseries(
            value=12, length=10
        )
        # 3 * 1 + 11 = 14
        assert mock.transform(test_input_1, series_idx=1) == constant_timeseries(
            value=14, length=10
        )
        # 2 * 2 + 10 = 14
        assert mock.transform(test_input_2, series_idx=0) == constant_timeseries(
            value=14, length=11
        )
        # 3 * 2 + 11 = 17
        assert mock.transform(test_input_2, series_idx=1) == constant_timeseries(
            value=17, length=11
        )

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
        assert transformed_1 == constant_timeseries(value=12, length=10)
        # If two series provided to transformer trained on three
        # series, should transform using the first and second set of
        # fitted parameters:
        transformed_1, transformed_2 = mock.transform((test_input_1, test_input_2))
        assert transformed_1 == constant_timeseries(value=12, length=10)
        assert transformed_2 == constant_timeseries(value=17, length=11)

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
        assert transformed == expected

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
        assert transformed == expected

        # Manually apply component mask:
        mock = self.DataTransformerMock(scale=2, translation=10, mask_components=False)
        transformed = mock.fit_transform(test_input, component_mask=mask)
        assert transformed == expected


class TestGlobalFittableDataTransformer:
    """
    Tests that data transformers inheriting from `FittableDataTransformer` class behave
    correctly when `global_fit` attribute is `True`.
    """

    class DataTransformerMock(FittableDataTransformer):
        def __init__(self, global_fit: bool):
            """
            Subtracts off the time-averaged mean of each component in a `TimeSeries`.

            If `global_fit` is `True`, then all of the `TimeSeries` provided to `fit` are
            used to compute a single time-averaged mean that will be subtracted from
            every `TimeSeries` subsequently provided to `transform`.

            Conversely, if `global_fit` is `False`, then the time-averaged mean of each
            `TimeSeries` pass to `fit` is individually computed, resulting in `n` means
            being computed if `n` `TimeSeries` were passed to `fit`. If multiple `TimeSeries`
            are subsequently passed to `transform`, the `i`th computed mean will be subtracted
            from the `i`th provided `TimeSeries`.

            Parameters
            ----------
            global_fit
                Whether global fitting should be performed.
            """
            super().__init__(
                name="DataTransformerMock", global_fit=global_fit, mask_components=True
            )

        @staticmethod
        def ts_fit(
            series: Union[TimeSeries, Sequence[TimeSeries]],
            params: Mapping[str, Any],
            **kwargs,
        ):
            """
            'Fits' transform by computing time-average of each sample and
            component in `series`.

            If `global_fit` is `True`, then `series` is a `Sequence[TimeSeries]` and the time-averaged mean
            of each component over *all* of the `TimeSeries` is computed. If `global_fit` is `False`, then
            `series` is a single `TimeSeries` and the time-averaged mean of the components of this single
            `TimeSeries` are computed.
            """
            if not isinstance(series, Sequence):
                series = [series]
            vals = np.concatenate([ts.all_values(copy=False) for ts in series], axis=0)
            return np.mean(vals, axis=0)

        @staticmethod
        def ts_transform(
            series: TimeSeries, params: Mapping[str, Any], **kwargs
        ) -> TimeSeries:
            """
            Implements the transform `series - mean`.
            """
            mean = params["fitted"]
            vals = series.all_values()
            vals -= mean
            return TimeSeries.from_values(vals)

    def test_global_fitting(self):
        """
        Tests that time-averaged mean subtraction transformation behaves
        correctly when `global_fit = False` and when `global_fit = True`.
        """

        series_1 = TimeSeries.from_values(np.ones((3, 2, 1)))
        series_2 = TimeSeries.from_values(2 * np.ones((3, 2, 1)))

        # Local fitting - subtracting mean of each series from itself should return
        # zero-valued series:
        transformed_1, transformed_2 = self.DataTransformerMock(
            global_fit=False
        ).fit_transform([series_1, series_2])
        assert transformed_1 == TimeSeries.from_values(np.zeros((3, 2, 1)))
        assert transformed_2 == TimeSeries.from_values(np.zeros((3, 2, 1)))

        # Global fitting - mean of `series_1` and `series_2` should be `1.5`, so
        # `series_1` values should be transformed to `-0.5` and `series_2` values
        # should be transformed to `1.5`:
        transformed_1, transformed_2 = self.DataTransformerMock(
            global_fit=True
        ).fit_transform([series_1, series_2])
        assert transformed_1 == TimeSeries.from_values(-0.5 * np.ones((3, 2, 1)))
        assert transformed_2 == TimeSeries.from_values(0.5 * np.ones((3, 2, 1)))

    def test_global_fitting_component_masking(self):
        cols_1 = ["A", "B"]
        cols_2 = ["C", "D"]
        series_1_ = TimeSeries.from_values(np.ones((3, 2, 1)), columns=cols_1)
        series_2_ = TimeSeries.from_values(2 * np.ones((3, 2, 1)), columns=cols_2)
        series_1 = series_1_.stack(series_2_)
        series_2 = series_2_.stack(series_1_)

        component_mask = np.array([True] * len(cols_1) + [False] * len(cols_2))
        # Local fitting - subtracting mean of each series from itself should return
        # zero-valued series:
        transformed_1, transformed_2 = self.DataTransformerMock(
            global_fit=False
        ).fit_transform([series_1, series_2], component_mask=component_mask)
        # transformed components
        assert transformed_1[cols_1] == TimeSeries.from_values(
            np.zeros((3, 2, 1)), columns=cols_1
        )
        assert transformed_2[cols_2] == TimeSeries.from_values(
            np.zeros((3, 2, 1)), columns=cols_2
        )

        # non-transformed components
        assert transformed_1[cols_2] == series_2_
        assert transformed_2[cols_1] == series_1_

        # Global fitting - mean of `series_1` and `series_2` should be `1.5`, so
        # `series_1` values should be transformed to `-0.5` and `series_2` values
        # should be transformed to `1.5`:
        transformed_1, transformed_2 = self.DataTransformerMock(
            global_fit=True
        ).fit_transform([series_1, series_2], component_mask=component_mask)
        # transformed components
        assert transformed_1[cols_1] == TimeSeries.from_values(
            -0.5 * np.ones((3, 2, 1)), columns=cols_1
        )
        assert transformed_2[cols_2] == TimeSeries.from_values(
            0.5 * np.ones((3, 2, 1)), columns=cols_2
        )

        # non-transformed components
        assert transformed_1[cols_2] == series_2_
        assert transformed_2[cols_1] == series_1_
