import unittest
import logging

from darts.dataprocessing import Validator
from darts.dataprocessing.transformers import BaseDataTransformer


class BaseDataTransformerTestCase(unittest.TestCase):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    class DataTransformerMock(BaseDataTransformer[str]):
        def __init__(self, validators=None):
            super().__init__(name="DataTransformerMock", validators=validators)
            self.validate_called = False
            self.transform_called = False

        def validate(self, data: str) -> bool:
            self.validate_called = True
            return super()._validate(data)

        def transform(self, data: str, *args, **kwargs) -> str:
            self.transform_called = True
            return data + "transformed"

    def test_validation_fails(self):
        # given
        validator = Validator(lambda x: False, "it was meant to fail")

        mock = self.DataTransformerMock(validators=[validator])

        # when & then
        with self.assertRaises(ValueError):
            mock.validate("test input")
        self.assertTrue(mock.validate_called)

        self.assertFalse(mock.transform_called)

    def test_input_transformed(self):
        # given
        test_input = "test"
        mock = self.DataTransformerMock()

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

        mock = self.DataTransformerMock(validators=validators)
        # when
        result = mock.validate("")

        # then
        self.assertTrue(result)
