import unittest

from darts.preprocessing import Validator


class ValidatorTestCase(unittest.TestCase):
    __test__ = True

    def test_basic(self):
        # given
        validator = Validator(lambda x: x == "test", "reason")

        # when
        result = validator("test")

        self.assertTrue(result)
        self.assertEqual(validator.reason, "reason")
