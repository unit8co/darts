import unittest
import logging

from darts.dataprocessing import data_transformer_from_ts_functions, data_transformer_from_values_functions
from darts.dataprocessing.transformers import BaseDataTransformer, InvertibleDataTransformer, FittableDataTransformer
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
        transformer = data_transformer_from_ts_functions(transform=lambda x: x + other,
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
        transformer = data_transformer_from_ts_functions(transform=lambda x: x + other,
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
        transformer = data_transformer_from_values_functions(transform=lambda x: x + other,
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
        transformer = data_transformer_from_values_functions(transform=lambda x: x + other,
                                                             inverse_transform=lambda x: x + 2 * other,
                                                             fit=fit_check)

        # when
        transformed = transformer.fit_transform(ts)

        # then
        self.assertSequenceEqual([[3]] * 10, transformed.values().tolist())
        self.assertTrue(fit_check._fit_called)

    def test_correct_class(self):

        # given & when
        simple_tf = data_transformer_from_ts_functions(transform=lambda x: x + 10,
                                                       name="simple transformer")

        invertible_tf = data_transformer_from_ts_functions(transform=lambda x: x + 10,
                                                           inverse_transform=lambda x: x - 10,
                                                           name="invertible transformer")
        fittable_tf = data_transformer_from_ts_functions(transform=lambda x: x + 10,
                                                         fit=lambda x: x,
                                                         name="fittable transformer")
        inv_and_fit_tf = data_transformer_from_ts_functions(transform=lambda x: x + 10,
                                                            inverse_transform=lambda x: x - 10,
                                                            fit=lambda x: x,
                                                            name="invertible and fittable transformer")

        transformers = [simple_tf, invertible_tf, fittable_tf, inv_and_fit_tf]

        # then
        self.assertTrue(all((isinstance(t, BaseDataTransformer)) for t in transformers))

        self.assertFalse(isinstance(simple_tf, InvertibleDataTransformer))
        self.assertFalse(isinstance(simple_tf, FittableDataTransformer))

        self.assertTrue(isinstance(invertible_tf, InvertibleDataTransformer))
        self.assertFalse(isinstance(invertible_tf, FittableDataTransformer))

        self.assertTrue(isinstance(fittable_tf, FittableDataTransformer))
        self.assertFalse(isinstance(fittable_tf, InvertibleDataTransformer))

        self.assertTrue(isinstance(inv_and_fit_tf, FittableDataTransformer))
        self.assertTrue(isinstance(inv_and_fit_tf, InvertibleDataTransformer))
