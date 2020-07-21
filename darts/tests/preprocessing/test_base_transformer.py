import unittest

from darts.preprocessing import BaseTransformer, Validator


class BaseTransformerTestCase(unittest.TestCase):
    __test__ = True

    class TransformerMock(BaseTransformer[str]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
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
        val = (lambda x: False, "test")
        mock = self.TransformerMock(validator_fns=[val])

        # when & then
        self.assertRaises(ValueError, mock, "test input")
        self.assertTrue(mock.validate_called)
        self.assertFalse(mock.transform_called)

    def test_raise_unimplemented_exception(self):
        # given
        mock = self.TransformerMock()

        # when & then
        self.assertRaises(NotImplementedError, mock, "test", fit=True)
        self.assertRaises(NotImplementedError, mock, "test", inverse=True)
        self.assertRaises(NotImplementedError, mock.fit_transform, "test")

    def test_input_transformed(self):
        # given
        test_input = "test"
        mock = self.TransformerMock()

        # when
        transformed = mock(test_input)

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
        validator_fns = [
            ((lambda y: lambda x: t(x, y))(i), str(i)) for i in range(10, 20)
        ]
        mock = self.TransformerMock(validators=validators, validator_fns=validator_fns)
        # when
        result = mock.validate("")

        # then
        self.assertTrue(result)
