import unittest
import pandas as pd

from darts.dataprocessing.transformers.mappers import Mapper, InvertibleMapper
from darts.utils.timeseries_generation import constant_timeseries, linear_timeseries


class MappersTestCase(unittest.TestCase):

    @staticmethod
    def func(x):
        return x + 10

    @staticmethod
    def inverse_func(x):
        return x - 10

    @staticmethod
    def ts_func(ts, x):
        return x - ts.month

    @staticmethod
    def inverse_ts_func(ts, x):
        return x + ts.month

    plus_ten = Mapper(func.__func__)
    plus_ten_invertible = InvertibleMapper(func.__func__, inverse_func.__func__)

    subtract_month = Mapper(ts_func.__func__)
    subtract_month_invertible = InvertibleMapper(ts_func.__func__, inverse_ts_func.__func__)

    lin_series = linear_timeseries(start_value=1, length=12, freq='MS', start_ts=pd.Timestamp('2000-01-01'), end_value=12)  # noqa: E501
    zeroes = constant_timeseries(value=0.0, length=12, freq='MS', start_ts=pd.Timestamp('2000-01-01'))
    tens = constant_timeseries(value=10.0, length=12, freq='MS', start_ts=pd.Timestamp('2000-01-01'))

    def test_mapper(self):
        transformed = self.plus_ten.transform(self.zeroes)
        self.assertEqual(transformed, self.tens)

    def test_invertible_mapper(self):
        transformed = self.plus_ten_invertible.transform(self.lin_series)
        back = self.plus_ten_invertible.inverse_transform(transformed)
        self.assertEqual(back, self.lin_series)

    def test_mapper_with_timestamp(self):
        transformed = self.subtract_month.transform(self.lin_series)
        self.assertEqual(transformed, self.zeroes)

    def test_invertible_mapper_with_timestamp(self):
        transformed = self.subtract_month_invertible.transform(self.lin_series)
        back = self.subtract_month_invertible.inverse_transform(transformed)
        self.assertEqual(back, self.lin_series)
