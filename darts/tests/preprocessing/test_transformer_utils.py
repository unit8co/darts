import unittest
import logging

from darts.preprocessing import (
    transformer_from_ts_functions,
    transformer_from_values_functions,
    BaseTransformer,
    InvertibleTransformer,
    FittableTransformer
)
from darts.utils.timeseries_generation import constant_timeseries


class UtilsTestCase(unittest.TestCase):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    class FitCalled:
        def __init__(self):
            self._fit_called = False

        def __call__(self, *args, **kwargs):
            self._fit_called = True

    def test_inverse_called(self):
        # given
        ts = constant_timeseries(1, 10)
        other = constant_timeseries(2, 10)
        fit_check = self.FitCalled()
        transformer = transformer_from_ts_functions(transform=lambda x: x + other,
                                                    inverse_transform=lambda x: x + 2 * other,
                                                    fit=fit_check)

        # when
        inverse_fitted = transformer.fit(ts).inverse_transform(ts)

        # then
        self.assertSequenceEqual([[5]] * 10, inverse_fitted.values().tolist())
        self.assertTrue(fit_check._fit_called)

    def test_inverse_not_called(self):
        # given
        ts = constant_timeseries(1, 10)
        other = constant_timeseries(2, 10)
        fit_check = self.FitCalled()
        transformer = transformer_from_ts_functions(transform=lambda x: x + other,
                                                    inverse_transform=lambda x: x + 2 * other,
                                                    fit=fit_check)

        # when
        transformed = transformer.fit_transform(ts)

        # then
        self.assertSequenceEqual([[3]] * 10, transformed.values().tolist())
        self.assertTrue(fit_check._fit_called)

    def test_inverse_called_values(self):
        # given
        ts = constant_timeseries(1, 10)
        other = constant_timeseries(2, 10).values()
        fit_check = self.FitCalled()
        transformer = transformer_from_values_functions(transform=lambda x: x + other,
                                                        inverse_transform=lambda x: x + 2 * other,
                                                        fit=fit_check)

        # when
        inverse_fitted = transformer.fit(ts).inverse_transform(ts)

        # then
        self.assertSequenceEqual([[5]] * 10, inverse_fitted.values().tolist())
        self.assertTrue(fit_check._fit_called)

    def test_inverse_not_called_values(self):
        # given
        ts = constant_timeseries(1, 10)
        other = constant_timeseries(2, 10).values()
        fit_check = self.FitCalled()
        transformer = transformer_from_values_functions(transform=lambda x: x + other,
                                                        inverse_transform=lambda x: x + 2 * other,
                                                        fit=fit_check)

        # when
        transformed = transformer.fit_transform(ts)

        # then
        self.assertSequenceEqual([[3]] * 10, transformed.values().tolist())
        self.assertTrue(fit_check._fit_called)

    def test_correct_class(self):

        # given & when
        simple_tf = transformer_from_ts_functions(transform=lambda x: x + 10,
                                                  name="simple transformer")

        invertible_tf = transformer_from_ts_functions(transform=lambda x: x + 10,
                                                      inverse_transform=lambda x: x - 10,
                                                      name="invertible transformer")
        fittable_tf = transformer_from_ts_functions(transform=lambda x: x + 10,
                                                    fit=lambda x: x,
                                                    name="fittable transformer")
        inv_and_fit_tf = transformer_from_ts_functions(transform=lambda x: x + 10,
                                                       inverse_transform=lambda x: x - 10,
                                                       fit=lambda x: x,
                                                       name="invertible and fittable transformer")

        transformers = [simple_tf, invertible_tf, fittable_tf, inv_and_fit_tf]

        # then
        self.assertTrue(all((isinstance(t, BaseTransformer)) for t in transformers))

        self.assertFalse(isinstance(simple_tf, InvertibleTransformer))
        self.assertFalse(isinstance(simple_tf, FittableTransformer))

        self.assertTrue(isinstance(invertible_tf, InvertibleTransformer))
        self.assertFalse(isinstance(invertible_tf, FittableTransformer))

        self.assertTrue(isinstance(fittable_tf, FittableTransformer))
        self.assertFalse(isinstance(fittable_tf, InvertibleTransformer))

        self.assertTrue(isinstance(inv_and_fit_tf, FittableTransformer))
        self.assertTrue(isinstance(inv_and_fit_tf, InvertibleTransformer))
