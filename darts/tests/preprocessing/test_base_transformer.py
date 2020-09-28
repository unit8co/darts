import unittest
import logging

from darts.preprocessing import BaseTransformer, Validator


class BaseTransformerTestCase(unittest.TestCase):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    class TransformerMock(BaseTransformer[str]):
        def __init__(self, *args, **kwargs):
            super().__init__(name="TransformerMock", invertible=False, fittable=False, *args, **kwargs)
            self.validate_called = False
            self.transform_called = False

        def validate(self, data: str) -> bool:
            self.validate_called = True
            return super().validate(data)

        def transform(self, data: str, *args, **kwargs) -> str:
            self.transform_called = True
            return data + "transformed"

    def test_validation_fails(self):
        # given
        validator = Validator(lambda x: False, "it was meant to fail")

        mock = self.TransformerMock(validators=[validator])

        # when & then
        with self.assertRaises(ValueError):
            mock.validate("test input")
        self.assertTrue(mock.validate_called)

        self.assertFalse(mock.transform_called)

    def test_raise_unimplemented_exception(self):
        # given
        mock = self.TransformerMock()

        # when & then
        with self.assertRaises(NotImplementedError):
            mock.fit("test").transform("test")

        with self.assertRaises(NotImplementedError):
            mock.inverse_transform("test")

        with self.assertRaises(NotImplementedError):
            mock.fit_transform("test")

    def test_input_transformed(self):
        # given
        test_input = "test"
        mock = self.TransformerMock()

        # when
        transformed = mock.transform(test_input)

        expected = "testtransformed"
        self.assertEqual(transformed, expected)

    def test_validators_calls_in_order(self):
        # given
        def t(data: str, i: int):
            t.called += 1
            return t.called == i
        t.called = -1

        validators = [
            Validator((lambda y: lambda x: t(x, y))(i), str(i)) for i in range(10)
        ]

        mock = self.TransformerMock(validators=validators)
        # when
        result = mock.validate("")

        # then
        self.assertTrue(result)
