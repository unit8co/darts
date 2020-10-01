import unittest
import logging

from darts.dataprocessing import Validator


class ValidatorTestCase(unittest.TestCase):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_basic(self):
        # given
        validator = Validator(lambda x: x == "test", "reason")

        # when
        result = validator("test")

        self.assertTrue(result)
        self.assertEqual(validator.reason, "reason")
