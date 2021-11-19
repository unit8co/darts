import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, List

from darts import TimeSeries
from darts.utils.data.covariate_index_generators import (
    CovariateIndexGenerator,
    PastCovariateIndexGenerator,
    FutureCovariateIndexGenerator
)

from darts.utils.data.training_dataset import TrainingDataset
from darts.utils.data.inference_dataset import InferenceDataset

from darts.logging import raise_if_not, get_logger, raise_log, raise_if

from darts.utils.timeseries_generation import datetime_attribute_timeseries

SupportedIndexes = Union[pd.DatetimeIndex, pd.Int64Index, pd.RangeIndex]
logger = get_logger(__name__)

ENCODER_KWARGS = ['add_cyclic_encoder', 'add_positional_encoder']

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


class EncoderSequence:
    def __init__(self, model_kwargs: Dict, train_dataset: TrainingDataset):
        _, self.kwargs = model_kwargs
        self.shift = train_dataset.ds_past.shift
        self.input_chunk_length = train_dataset.ds_past.input_chunk_length
        self.output_chunk_length = train_dataset.ds_past.output_chunk_length
        self.encoders: List[Encoder] = list()
        self.verify_call()

    @property
    def add_encoders(self):
        """returns dict from relevant encoder kwargs at model creation"""
        return {
            encoder: self.kwargs.get(encoder, None) for encoder in ENCODER_KWARGS if self.kwargs.get(encoder, None)
        }

    def verify_call(self):
        """encoder kwargs must be of form `encoder_kwarg=Dict[str, Union[str, Sequence[str]]]`.
        For example with cyclic encoders

        Parameters
        ----------
        kwargs
        add_cyclic_encoder={
            'past': ['month', 'dayofmonth', ...],  # or simply 'month'
            'future': [],  # or simply omitting `future`
        }
        """
        if not self.add_encoders:
            return

        for kwarg in self.add_encoders:
            pass

    def map_encoders(self):
        mapper = {
            'add_cyclic_encoder': {'past': []}
        }
