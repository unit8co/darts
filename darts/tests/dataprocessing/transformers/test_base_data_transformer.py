import unittest
import logging

from darts.dataprocessing.transformers import BaseDataTransformer


class BaseDataTransformerTestCase(unittest.TestCase):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    class DataTransformerMock(BaseDataTransformer[str]):
        def __init__(self):
            super().__init__(name="DataTransformerMock")
            self.transform_called = False

        def transform(self, data: str, *args, **kwargs) -> str:
            self.transform_called = True
            return data + "transformed"

    def test_input_transformed(self):
        # given
        test_input = "test"
        mock = self.DataTransformerMock()

        # when
        transformed = mock.transform(test_input)

        expected = "testtransformed"
        self.assertEqual(transformed, expected)
