import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, List, Sequence, Tuple

from darts import TimeSeries
from darts.utils.data.covariate_index_generators import (
    CovariateIndexGenerator,
    PastCovariateIndexGenerator,
    FutureCovariateIndexGenerator
)

from darts.logging import raise_if_not, get_logger, raise_if

from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries


SupportedTimeSeries = Union[TimeSeries, Sequence[TimeSeries]]
SupportedIndexes = Union[pd.DatetimeIndex, pd.Int64Index, pd.RangeIndex]
logger = get_logger(__name__)

ENCODER_KWARGS = ['add_cyclic_encoder', 'add_positional_encoder']
FUTURE = 'future'
PAST = 'past'
VALID_TIME_PARAMS = [
    FUTURE,
    PAST
    # 'absolute'
]
VALID_DTYPES = (str, Sequence)

DIMS = ('time', 'component', 'sample')


class Encoder(ABC):
    """Abstract class for index encoders"""

    @abstractmethod
    def __init__(self):
        self.train_encoded = None
        self.inference_encoded = None
        self.dtype = None

    @abstractmethod
    def encode_train(self,
                     target: TimeSeries,
                     covariate: Optional[TimeSeries] = None,
                     merge_covariate: bool = True,
                     **kwargs) -> TimeSeries:
        """Each subclass must implement a method to encode covariate index for training"""
        pass

    @abstractmethod
    def encode_inference(self,
                         n: int,
                         target: TimeSeries,
                         covariate: Optional[TimeSeries] = None,
                         merge_covariate: bool = True,
                         **kwargs) -> TimeSeries:
        """Each subclass must implement a method to encode covariate index for prediction"""
        pass

    @abstractmethod
    def encode_absolute(self,
                        target: TimeSeries,
                        covariate: Optional[TimeSeries] = None) -> TimeSeries:
        pass

    @staticmethod
    def merge_covariate(encoded: TimeSeries, covariate: Optional[TimeSeries] = None) -> TimeSeries:
        """If (actual) covariates are given, merge the encoded index with the covariates"""
        return covariate.stack(encoded) if covariate is not None else encoded


class EncoderSingle(Encoder, ABC):
    """Abstract class for single index encoders"""

    def __init__(self, index_generator: CovariateIndexGenerator):
        """Single encoders take an `index_generator` to generate the required index for encoding past and future
        covariates.
        See darts.utils.data.covariate_index_generators.py for the `CovariateIndexGenerator` subclasses.
        For past covariate encoders, use a `PastCovariateIndexGenerator`.
        For future covariate encoders use a `FutureCovariateIndexGenerator`.

        Parameters
        ----------
        index_generator
            An instance of `CovariateIndexGenerator` with methods `generate_train_series()` and
            `generate_inference_series()`. Used to generate the index for encoders.
        """

        super(EncoderSingle, self).__init__()
        self.index_generator = index_generator

    @abstractmethod
    def encode(self, index: SupportedIndexes) -> TimeSeries:
        """Single Encoders must implement an encode() method to encode the index"""
        pass

    def encode_train(self,
                     target: TimeSeries,
                     covariate: Optional[TimeSeries] = None,
                     merge_covariate: bool = True,
                     **kwargs) -> TimeSeries:

        self.dtype = target.dtype
        index = self.index_generator.generate_train_series(target, covariate)
        encoded = self.encode(index)
        if merge_covariate:
            return self.merge_covariate(encoded, covariate=covariate)
        else:
            return encoded

    def encode_inference(self,
                         n: int,
                         target: TimeSeries,
                         covariate: Optional[TimeSeries] = None,
                         merge_covariate: bool = True,
                         **kwargs) -> TimeSeries:

        self.dtype = target.dtype
        index = self.index_generator.generate_inference_series(n, target, covariate)
        encoded = self.encode(index)

        if merge_covariate:
            return self.merge_covariate(encoded, covariate=covariate)
        else:
            return encoded

    def encode_absolute(self,
                        target: TimeSeries,
                        covariate: Optional[TimeSeries] = None) -> TimeSeries:
        return self.encode_train(target, covariate)


class CyclicTemporalEncoder(EncoderSingle):
    """Adds cyclic time index encoding using `darts.utils.timeseries_generation.datetime_attribute_series()`"""

    def __init__(self, index_generator, attribute):
        super(CyclicTemporalEncoder, self).__init__(index_generator)
        self.attribute = attribute

    def encode(self, index: SupportedIndexes) -> TimeSeries:
        super(CyclicTemporalEncoder, self).encode(index)
        return datetime_attribute_timeseries(index, attribute=self.attribute, cyclic=True, dtype=self.dtype)


class CyclicPastEncoder(CyclicTemporalEncoder):
    """Cyclic encoder for past covariates"""
    def __init__(self, input_chunk_length, output_chunk_length, attribute):
        super(CyclicPastEncoder, self).__init__(
            index_generator=PastCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            attribute=attribute
        )


class CyclicFutureEncoder(CyclicTemporalEncoder):
    """Cyclic encoder for future covariates"""
    def __init__(self, input_chunk_length, output_chunk_length, attribute):
        super(CyclicFutureEncoder, self).__init__(
            index_generator=FutureCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            attribute=attribute
        )


class PositionalEncoder(EncoderSingle):
    """Absolute positional index encoding"""

    def __init__(self, index_generator, *args):
        super(PositionalEncoder, self).__init__(index_generator)
        raise_if(True, 'NotImplementedError')

    def encode(self, index: SupportedIndexes) -> TimeSeries:
        super(PositionalEncoder, self).encode(index)
        raise_if(True, 'NotImplementedError')
        return TimeSeries.from_times_and_values(index, np.arange(len(index)))


class PositionalPastEncoder(PositionalEncoder):
    """Absolute index encoder for past covariates"""
    def __init__(self, input_chunk_length, output_chunk_length, *args):
        super(PositionalPastEncoder, self).__init__(
            index_generator=PastCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            *args
        )


class PositionalFutureEncoder(PositionalEncoder):
    """Absolute index encoder for future covariates"""
    def __init__(self, input_chunk_length, output_chunk_length, *args):
        super(PositionalFutureEncoder, self).__init__(
            index_generator=FutureCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            *args
        )


class EncoderSequence(Encoder):
    """Encoder sequence can store and control multiple past and future covariate encoders at once.
    It provides the same features as single encoders (`encode_train()` and `encode_inference`).
    """

    def __init__(self,
                 model_kwargs: Dict,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 shift: int,
                 takes_past_covariates: bool = False,
                 takes_future_covariates: bool = False) -> None:

        """
        EncoderSequence automatically creates encoder objects from parameters used when creating a
        `TorchForecastingModel` model. Currently these parameters include:

        model = MyModel(...,
                        add_cyclic_encoder={'past': ['month', 'dayofweek', ...], 'future': [same as past]},
                        ...)

        A string ID is extracted from the parameters as follows:
            str(key) + str(sub_key) -> 'add_cyclic_encoder' + 'past' -> ID = 'add_cyclic_encoder_past'

        This ID is used to map the model parameters with the corresponding encoder objects.

        New encoders can be added by appending them to the mapping property `EncoderSequence.encoder_map()`

        Paramaters
        ----------
        model_kwargs
            the parameters used at `TorchforeCastingModel` model creation.
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        shift
            The number of time steps by which to shift the output chunks relative to the input chunks.
        takes_past_covariates
            whether or not the `TrainingDataset` takes past covariates
        takes_future_covariates
            whether or not the `TrainingDataset` takes past covariates
        """

        super(EncoderSequence, self).__init__()
        self.params = model_kwargs
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.shift = shift
        self._past_encoders: List[EncoderSingle] = []
        self._future_encoders: List[EncoderSingle] = []
        self.takes_past_covariates = takes_past_covariates
        self.takes_future_covariates = takes_future_covariates
        self.setup_encoders(self.params)

    def encode_train(self,
                     target: SupportedTimeSeries,
                     past_covariate: Optional[SupportedTimeSeries] = None,
                     future_covariate: Optional[SupportedTimeSeries] = True,
                     **kwargs) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:

        return self.launch_encoder(target=target,
                                   past_covariate=past_covariate,
                                   future_covariate=future_covariate,
                                   n=None)

    def encode_inference(self,
                         n: int,
                         target: SupportedTimeSeries,
                         past_covariate: Optional[SupportedTimeSeries] = None,
                         future_covariate: Optional[SupportedTimeSeries] = True,
                         merge_covariate: Optional[SupportedTimeSeries] = True,
                         **kwargs) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:

        return self.launch_encoder(target=target,
                                   past_covariate=past_covariate,
                                   future_covariate=future_covariate,
                                   n=n)

    def launch_encoder(self,
                       target: Sequence[TimeSeries],
                       past_covariate: SupportedTimeSeries,
                       future_covariate: SupportedTimeSeries,
                       n: Optional[int] = None) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:

        if not self.past_encoders and not self.future_encoders:
            return past_covariate, future_covariate

        target = [target] if isinstance(target, TimeSeries) else target

        if self.past_encoders:
            past_covariate = self.encode_sequence(encoders=self.past_encoders, target=target, covariate=past_covariate, n=n)

        if self.future_encoders:
            future_covariate = self.encode_sequence(encoders=self.future_encoders, target=target, covariate=future_covariate, n=n)

        return past_covariate, future_covariate

    def encode_sequence(self,
                        encoders: Sequence[EncoderSingle],
                        target: Sequence[TimeSeries],
                        covariate: Optional[SupportedTimeSeries],
                        n: Optional[int] = None) -> Sequence[TimeSeries]:
        encode_method = 'encode_train' if n is None else 'encode_inference'

        encoded_sequence = []
        if covariate is None:
            covariate = [None] * len(target)
        else:
            covariate = [covariate] if isinstance(covariate, TimeSeries) else covariate

        for ts, pc in zip(target, covariate):
            encoded = concatenate([getattr(enc, encode_method)(target=ts,
                                                               covariate=pc,
                                                               merge_covariate=False,
                                                               n=n) for enc in encoders], axis=DIMS[1])
            encoded_sequence.append(self.merge_covariate(encoded=encoded, covariate=pc))
        return encoded_sequence

    def encode_absolute(self,
                        target: TimeSeries,
                        covariate: Optional[TimeSeries] = None) -> TimeSeries:
        pass

    @property
    def future_encoders(self) -> List[EncoderSingle]:
        """returns the future covariate encoder objects"""
        return self._future_encoders

    @property
    def past_encoders(self) -> List[EncoderSingle]:
        """returns the past covariate encoder objects"""
        return self._past_encoders

    @property
    def encoder_map(self) -> Dict:
        """mapping between encoder identifier string (from parameters at model creations) and the corresponding
        future or past covariate encoder"""
        mapper = {
            'add_cyclic_encoder_past': CyclicPastEncoder,
            'add_cyclic_encoder_future': CyclicFutureEncoder,
            'add_positional_encoder_past': PositionalPastEncoder,
            'add_positional_encoder_future': PositionalFutureEncoder
        }
        return mapper

    def setup_encoders(self, params):
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
        past_encoders, future_encoders = self.process_input(params)

        if not past_encoders and not future_encoders:
            return

        self._past_encoders = [self.encoder_map[enc_id](self.input_chunk_length,
                                                        self.output_chunk_length,
                                                        attr) for enc_id, attr in past_encoders]
        self._future_encoders = [self.encoder_map[enc_id](self.input_chunk_length,
                                                          self.output_chunk_length,
                                                          attr) for enc_id, attr in future_encoders]

    def process_input(self, params) -> Tuple[List, List]:
        """processes input and returns dict from relevant encoder parameters at model creation.

        If model parameters contain parameters for encoders, this method returns two lists with past and future encoder
        identifiers and parameters extracted from model parameters.

        `parameters` must be a dictionary of form `encoder_kwarg=Dict[str, Union[str, Sequence[str]]]`.
        For example with cyclic encoders

        Parameters
        ----------
        params
            add_cyclic_encoder={
                'past': ['month', 'dayofmonth', ...],  # or simply 'month'
                'future': [],  # or simply omitting `future`
            }
        Raises
        ------
        ValueError
            1) if the outermost key is other than (`past`, `future`, `absolute`)
            2) if the innermost values are other than type `str` or `Sequence`
        """
        # extract encoder params
        encoders = {enc: params.get(enc, None) for enc in ENCODER_KWARGS if params.get(enc, None)}

        if not encoders:
            return [], []

        # check input for if invalid temporal types; values other than ('past', 'future', 'absolute')
        invalid_time_params = list()
        for encoder, t_types in encoders.items():
            invalid_time_params += [t_type for t_type in t_types.keys() if t_type not in VALID_TIME_PARAMS]

        raise_if(len(invalid_time_params) > 0,
                 f'Encountered invalid temporal types `{invalid_time_params}` in `add_*_encoder` parameter at model '
                 f'creation. Supported temporal types are: `{VALID_TIME_PARAMS}`.')

        # convert
        past_encoders, future_encoders = list(), list()
        for enc, enc_params in encoders.items():
            for enc_time, enc_attr in enc_params.items():
                raise_if_not(isinstance(enc_attr, VALID_DTYPES),
                             f'Encountered value `{enc_attr}` of invalid type `{type(enc_attr)}` for parameter '
                             f'`{enc}` at model creation. Supported data types are: `{VALID_DTYPES}`.')
                attrs = [enc_attr] if isinstance(enc_attr, str) else enc_attr
                for attr in attrs:
                    encoder_id = '_'.join([enc, enc_time])
                    if enc_time == PAST:
                        past_encoders.append((encoder_id, attr))
                    else:
                        future_encoders.append((encoder_id, attr))

        past_encoders = past_encoders if self.takes_past_covariates else []
        future_encoders = future_encoders if self.takes_future_covariates else []
        return past_encoders, future_encoders
