import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Optional

from darts import TimeSeries
from darts.utils.data.covariate_index_generators import (
    CovariateIndexGenerator,
    PastCovariateIndexGenerator,
    FutureCovariateIndexGenerator
)
from darts.logging import raise_if_not, get_logger, raise_log, raise_if

from darts.utils.timeseries_generation import datetime_attribute_timeseries

SupportedIndexes = Union[pd.DatetimeIndex, pd.Int64Index, pd.RangeIndex]
logger = get_logger(__name__)


class Encoder(ABC):
    """Abstract class for index encoders encode an index"""

    @abstractmethod
    def __init__(self, index_generator: CovariateIndexGenerator):
        self.index_generator = index_generator
        self.train_encoded = None
        self.inference_encoded = None
        self.dtype = None

    @abstractmethod
    def encode(self, index: SupportedIndexes) -> TimeSeries:
        pass

    def encode_train(self,
                     idx: int,
                     target: TimeSeries,
                     covariate: Optional[TimeSeries] = None) -> TimeSeries:
        self.dtype = target.dtype
        index = self.index_generator.generate_train_series(idx, target, covariate)
        encoded = self.encode(index)
        return self.merge_covariate(encoded, covariate=covariate)

    def encode_inference(self,
                         idx: int,
                         n: int,
                         target: TimeSeries,
                         covariate: Optional[TimeSeries] = None) -> TimeSeries:

        self.dtype = target.dtype
        index = self.index_generator.generate_inference_series(idx, n, target, covariate)
        encoded = self.encode(index)
        return self.merge_covariate(encoded, covariate=covariate)

    def encode_absolute(self,
                        idx: int,
                        target: TimeSeries,
                        covariate: Optional[TimeSeries] = None) -> TimeSeries:
        return self.encode_train(idx, target, covariate)

    @staticmethod
    def merge_covariate(encoded: TimeSeries, covariate: Optional[TimeSeries] = None) -> TimeSeries:
        return covariate.stack(encoded) if covariate is not None else encoded


class TestEncoder(Encoder):
    def __init__(self, index_generator):
        super(TestEncoder, self).__init__(index_generator)

    def encode(self, index: SupportedIndexes) -> TimeSeries:
        super(TestEncoder, self).encode(index)
        return TimeSeries.from_times_and_values(index, np.arange(len(index)))


class CyclicTemporalEncoder(Encoder):
    """adds cyclic time index encoding"""

    def __init__(self, index_generator, attribute):
        super(CyclicTemporalEncoder, self).__init__(index_generator)
        self.attribute = attribute

    def encode(self, index: SupportedIndexes) -> TimeSeries:
        super(CyclicTemporalEncoder, self).encode(index)
        return datetime_attribute_timeseries(index, attribute=self.attribute, cyclic=True, dtype=self.dtype)


class CyclicPastEncoder(CyclicTemporalEncoder):
    def __init__(self, input_chunk_length, output_chunk_length, attribute):
        super(CyclicPastEncoder, self).__init__(
            index_generator=PastCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            attribute=attribute
        )


class CyclicFutureEncoder(CyclicTemporalEncoder):
    def __init__(self, input_chunk_length, output_chunk_length, attribute):
        super(CyclicFutureEncoder, self).__init__(
            index_generator=FutureCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            attribute=attribute
        )


class PositionalEncoder(Encoder):
    """adds absolute positional index encoding"""

    def __init__(self, index_generator, *args):
        super(PositionalEncoder, self).__init__(index_generator)
        raise_if(True, 'NotImplementedError')

    def encode(self, index: SupportedIndexes) -> TimeSeries:
        super(PositionalEncoder, self).encode(index)
        raise_if(True, 'NotImplementedError')
        return TimeSeries.from_times_and_values(index, np.arange(len(index)))


class PositionalPastEncoder(PositionalEncoder):
    def __init__(self, input_chunk_length, output_chunk_length, *args):
        super(PositionalPastEncoder, self).__init__(
            index_generator=PastCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            *args
        )


class PositionalFutureEncoder(PositionalEncoder):
    def __init__(self, input_chunk_length, output_chunk_length, *args):
        super(PositionalFutureEncoder, self).__init__(
            index_generator=FutureCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            *args
        )




