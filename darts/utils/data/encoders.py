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

ENCODER_KWARG = 'add_encoders'
ENCODER_KEYS = ['cyclic', 'datetime_attribute', 'positional']
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
    """Abstract class for all encoders"""

    @abstractmethod
    def __init__(self):
        self.attribute = None
        self.train_encoded = None
        self.inference_encoded = None
        self.dtype = None

    @abstractmethod
    def encode_train(self,
                     target: TimeSeries,
                     covariate: Optional[TimeSeries] = None,
                     merge_covariate: bool = True,
                     **kwargs) -> TimeSeries:
        """Each subclass must implement a method to encode covariate index for training.

        Parameters
        ----------
        target
            the target TimeSeries used during training or passed to prediction as `series`
        covariate
            optionally, the future covariates used for prediction
        merge_covariate
            whether or not to merge the encoded TimeSeries with `covariate`.
        """
        pass

    @abstractmethod
    def encode_inference(self,
                         n: int,
                         target: TimeSeries,
                         covariate: Optional[TimeSeries] = None,
                         merge_covariate: bool = True,
                         **kwargs) -> TimeSeries:
        """Each subclass must implement a method to encode covariate index for prediction

        Parameters
        ----------
        n
            the forecast horizon
        target
            the target TimeSeries used during training or passed to prediction as `series`
        covariate
            optionally, the future covariates used for prediction
        merge_covariate
            whether or not to merge the encoded TimeSeries with `covariate`.

        """
        pass

    @abstractmethod
    def encode_absolute(self,
                        target: TimeSeries,
                        covariate: Optional[TimeSeries] = None) -> TimeSeries:
        """Tnis is a placeholder for doing absolute encodings"""
        pass

    @staticmethod
    def _merge_covariate(encoded: TimeSeries, covariate: Optional[TimeSeries] = None) -> TimeSeries:
        """If (actual) covariates are given, merge the encoded index with the covariates

        Parameters
        ----------
        encoded
            the encoded TimeSeries either from `encode_train()` or `encode_inference()`
        covariate
            optionally, the future covariates used for prediction
        """
        return covariate.stack(encoded) if covariate is not None else encoded


class SingleEncoder(Encoder, ABC):
    """Abstract class for single index encoders.
    Single encoders can be used to implement new encoding techniques.
    Each single encoder must implement an `_encode()` method that carries the encoding logic.

    The `_encode()` method must take an `index` as input and generate a encoded single `TimeSeries` as output.
    """

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

        super(SingleEncoder, self).__init__()
        self.index_generator = index_generator

    @abstractmethod
    def _encode(self, index: SupportedIndexes) -> TimeSeries:
        """Single Encoders must implement an _encode() method to encode the index.

        Parameters
        ----------
        index
            the index generated from `self.index_generator` for either the train or inference dataset.
        """
        pass

    def encode_train(self,
                     target: TimeSeries,
                     covariate: Optional[TimeSeries] = None,
                     merge_covariate: bool = True,
                     **kwargs) -> TimeSeries:
        """Returns encoded index for training.

        Parameters
        ----------
        target
            the target TimeSeries used during training or passed to prediction as `series`
        covariate
            optionally, the covariate used for training: past covariate if `self.index_generator` is instance of 
            `PastCovariateIndexGenerator`, future covariate if `self.index_generator` is instance of 
            `FutureCovariateIndexGenerator`
        merge_covariate
            whether or not to merge the encoded TimeSeries with `covariate`.
        """
        self.dtype = target.dtype
        index = self.index_generator.generate_train_series(target, covariate)
        encoded = self._encode(index)
        if merge_covariate:
            return self._merge_covariate(encoded, covariate=covariate)
        else:
            return encoded

    def encode_inference(self,
                         n: int,
                         target: TimeSeries,
                         covariate: Optional[TimeSeries] = None,
                         merge_covariate: bool = True,
                         **kwargs) -> TimeSeries:
        """Returns encoded index for inference/prediction.

        Parameters
        ----------
        n
            the forecast horizon
        target
            the target TimeSeries used during training or passed to prediction as `series`
        covariate
            optionally, the covariate used for prediction: past covariate if `self.index_generator` is instance of 
            `PastCovariateIndexGenerator`, future covariate if `self.index_generator` is instance of 
            `FutureCovariateIndexGenerator`
        merge_covariate
            whether or not to merge the encoded TimeSeries with `covariate`.
        """
        self.dtype = target.dtype
        index = self.index_generator.generate_inference_series(n, target, covariate)
        encoded = self._encode(index)

        if merge_covariate:
            return self._merge_covariate(encoded, covariate=covariate)
        else:
            return encoded

    def encode_absolute(self,
                        target: TimeSeries,
                        covariate: Optional[TimeSeries] = None) -> TimeSeries:
        return self.encode_train(target, covariate)


class CyclicTemporalEncoder(SingleEncoder):
    """CyclicTemporalEncoder: Cyclic index encoding for `TimeSeries` that have a time index of type `pd.DatetimeIndex`.
    """

    def __init__(self, index_generator: CovariateIndexGenerator, attribute: str):
        """
        Parameters
        ----------
        index_generator
            an instance of `CovariateIndexGenerator` with methods `generate_train_series()` and
            `generate_inference_series()`. Used to generate the index for encoders.
        attribute
            the attribute of the underlying pd.DatetimeIndex from  for which to apply cyclic encoding.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        """


        super(CyclicTemporalEncoder, self).__init__(index_generator)
        self.attribute = attribute

    def _encode(self, index: SupportedIndexes) -> TimeSeries:
        """applies cyclic encoding from `datetime_attribute_timeseries()` to `self.attribute` of `index`."""
        super(CyclicTemporalEncoder, self)._encode(index)
        return datetime_attribute_timeseries(index, attribute=self.attribute, cyclic=True, dtype=self.dtype)


class CyclicPastEncoder(CyclicTemporalEncoder):
    """Cyclic encoder for past covariates."""

    def __init__(self, input_chunk_length, output_chunk_length, attribute):
        """
        Parameters
        ----------
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        attribute
            the attribute of the underlying pd.DatetimeIndex from  for which to apply cyclic encoding.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        """
        super(CyclicPastEncoder, self).__init__(
            index_generator=PastCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            attribute=attribute
        )


class CyclicFutureEncoder(CyclicTemporalEncoder):
    """Cyclic encoder for future covariates."""

    def __init__(self, input_chunk_length, output_chunk_length, attribute):
        """
        Parameters
        ----------
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        attribute
            the attribute of the underlying pd.DatetimeIndex from  for which to apply cyclic encoding.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        """
        super(CyclicFutureEncoder, self).__init__(
            index_generator=FutureCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            attribute=attribute
        )


class DatetimeAttributeEncoder(SingleEncoder):
    """DatetimeAttributeEncoder: Adds pd.DatatimeIndex attribute information derived from the index as scalars.
    Requires the underlying TimeSeries to have a pd.DatetimeIndex
    """

    def __init__(self, index_generator: CovariateIndexGenerator, attribute: str):
        """
        Parameters
        ----------
        index_generator
            an instance of `CovariateIndexGenerator` with methods `generate_train_series()` and
            `generate_inference_series()`. Used to generate the index for encoders.
        attribute
            the attribute of the underlying pd.DatetimeIndex from  for which to add scalar information.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        """

        super(DatetimeAttributeEncoder, self).__init__(index_generator)
        self.attribute = attribute

    def _encode(self, index: SupportedIndexes) -> TimeSeries:
        """applies cyclic encoding from `datetime_attribute_timeseries()` to `self.attribute` of `index`."""
        super(DatetimeAttributeEncoder, self)._encode(index)
        return datetime_attribute_timeseries(index, attribute=self.attribute, dtype=self.dtype)


class DatetimeAttributePastEncoder(DatetimeAttributeEncoder):
    """Datetime attribute encoder for past covariates."""

    def __init__(self, input_chunk_length, output_chunk_length, attribute):
        """
        Parameters
        ----------
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        attribute
            the attribute of the underlying pd.DatetimeIndex from  for which to add scalar information.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        """
        super(DatetimeAttributePastEncoder, self).__init__(
            index_generator=PastCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            attribute=attribute
        )


class DatetimeAttributeFutureEncoder(DatetimeAttributeEncoder):
    """Datetime attribute encoder for future covariates."""

    def __init__(self, input_chunk_length, output_chunk_length, attribute):
        """
        Parameters
        ----------
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        attribute
            the attribute of the underlying pd.DatetimeIndex from  for which to add scalar information.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        """
        super(DatetimeAttributeFutureEncoder, self).__init__(
            index_generator=FutureCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            attribute=attribute
        )


class PositionalEncoder(SingleEncoder):
    """PLACEHOLDER: absolute positional index encoding"""

    def __init__(self, index_generator, *args):
        super(PositionalEncoder, self).__init__(index_generator)
        raise_if(True, 'NotImplementedError')

    def _encode(self, index: SupportedIndexes) -> TimeSeries:
        super(PositionalEncoder, self)._encode(index)
        return TimeSeries.from_times_and_values(index, np.arange(len(index)))


class PositionalPastEncoder(PositionalEncoder):
    """PLACEHOLDER: absolute index encoder for past covariates"""
    def __init__(self, input_chunk_length, output_chunk_length, *args):
        super(PositionalPastEncoder, self).__init__(
            index_generator=PastCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            *args
        )


class PositionalFutureEncoder(PositionalEncoder):
    """PLACEHOLDER: absolute index encoder for future covariates"""
    def __init__(self, input_chunk_length, output_chunk_length, *args):
        super(PositionalFutureEncoder, self).__init__(
            index_generator=FutureCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            *args
        )


class SequenceEncoder(Encoder):
    """A sequence encoder can store and control multiple past and future covariate encoders at once.
    It provides the same functionality as single encoders (`encode_train()` and `encode_inference`).
    Sequence encoders can be used with Darts' darts `Datasets`.
    """

    def __init__(self,
                 model_kwargs: Dict,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 shift: int,
                 takes_past_covariates: bool = False,
                 takes_future_covariates: bool = False) -> None:

        """
        SequenceEncoder automatically creates encoder objects from parameters used when creating a
        `TorchForecastingModel` model. Currently these parameters include:
        
        * add_encoders={
                'cyclic': {'past': ['month', 'dayofweek', ...], 'future': [same as for 'past']}
            }

        for example:
        model = MyModel(..., add_encoders={...}, ...)

        Tuples of `(encoder_id, attribute)` are extracted from the parameters to instantiate the `SingleEncoder`
        objects:
        * The `encoder_id` is extracted as follows:
            str(key) + str(temporal_key) -> 'cyclic' + 'past' -> `encoder_id` = 'cyclic_past'
            The `encoder_ix` is used to map the model parameters with the corresponding `SingleEncoder` objects.
        * The `attribute` is extracted from the values given by `temporal_key`
            `attribute` = 'month'
            ...
            The `attribute` tells the `SingleEncoder` which attribute of the index to encoder

        The resulting `SingleEncoder` objects will be instantiates as follows:
        self.past_encoders = [
            CyclicPastEncoder(input_chunk_length, output_chunk_length, attribute='month'),
            CyclicPastEncoder(input_chunk_length, output_chunk_length, attribute='dayofweek'),
            ...
        ]
        self.future_encoders = [
            CyclicFutureEncoder(input_chunk_length, output_chunk_length, attribute=future_attribute),
            ...
        ]

        New encoders can be added by appending them to the mapping property `SequenceEncoder.encoder_map()`

        Parameters
        ----------
        model_kwargs
            the parameters used at `TorchForecastingModel` model creation.
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

        super(SequenceEncoder, self).__init__()
        self.params = model_kwargs
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.shift = shift
        self._past_encoders: List[SingleEncoder] = []
        self._future_encoders: List[SingleEncoder] = []
        self.takes_past_covariates = takes_past_covariates
        self.takes_future_covariates = takes_future_covariates
        self._setup_encoders(self.params)

    def encode_train(self,
                     target: SupportedTimeSeries,
                     past_covariate: Optional[SupportedTimeSeries] = None,
                     future_covariate: Optional[SupportedTimeSeries] = True,
                     **kwargs) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        """Returns encoded index for all past and/or future covariates for training.
        Which covariates are generated depends on the parameters used at model creation.
        
        Parameters
        ----------
        target
            the target TimeSeries used during training or passed to prediction as `series`
        past_covariate
            optionally, the past covariates used for training.
        future_covariate
            optionally, the future covariates used for training.

        Returns
        -------
        Tuple[past_covariate, future_covariate]
            The past_covariate and/or future_covariate for training including the encodings.
            If input {x}_covariate is None and no {x}_encoders are given, will return `None`
            for the {x}_covariate.
        """

        return self._launch_encoder(target=target,
                                   past_covariate=past_covariate,
                                   future_covariate=future_covariate,
                                   n=None)

    def encode_inference(self,
                         n: int,
                         target: SupportedTimeSeries,
                         past_covariate: Optional[SupportedTimeSeries] = None,
                         future_covariate: Optional[SupportedTimeSeries] = True,
                         **kwargs) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        """Returns encoded index for all past and/or future covariates for inference/prediction.
        Which covariates are generated depends on the parameters used at model creation.

        Parameters
        ----------
        n
            the forecast horizon
        target
            the target TimeSeries used during training or passed to prediction as `series`
        past_covariate
            optionally, the past covariates used for training.
        future_covariate
            optionally, the future covariates used for training.

        Returns
        -------
        Tuple[past_covariate, future_covariate]
            The past_covariate and/or future_covariate for prediction/inference including the encodings.
            If input {x}_covariate is None and no {x}_encoders are given, will return `None`
            for the {x}_covariate.
        """

        return self._launch_encoder(target=target,
                                   past_covariate=past_covariate,
                                   future_covariate=future_covariate,
                                   n=n)

    def _launch_encoder(self,
                        target: Sequence[TimeSeries],
                        past_covariate: SupportedTimeSeries,
                        future_covariate: SupportedTimeSeries,
                        n: Optional[int] = None) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        """Launches the encode sequence for past covariate and future covariate for either training or
        inference/prediction.

        If `n` is `None` it is a prediction, otherwise it is training.
        """

        if not self.past_encoders and not self.future_encoders:
            return past_covariate, future_covariate

        target = [target] if isinstance(target, TimeSeries) else target

        if self.past_encoders:
            past_covariate = self._encode_sequence(encoders=self.past_encoders, target=target, covariate=past_covariate, n=n)

        if self.future_encoders:
            future_covariate = self._encode_sequence(encoders=self.future_encoders, target=target, covariate=future_covariate, n=n)

        return past_covariate, future_covariate

    def _encode_sequence(self,
                         encoders: Sequence[SingleEncoder],
                         target: Sequence[TimeSeries],
                         covariate: Optional[SupportedTimeSeries],
                         n: Optional[int] = None) -> Sequence[TimeSeries]:
        """Sequentially encodes the index of all input target/covariate TimeSeries

        If `n` is `None` it is a prediction and method `encoder.encode_inference()` is called.
        Otherwise, it is a training case and `encoder.encode_train()` is called.
        """

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
            encoded_sequence.append(self._merge_covariate(encoded=encoded, covariate=pc))
        return encoded_sequence

    def encode_absolute(self,
                        target: TimeSeries,
                        covariate: Optional[TimeSeries] = None) -> TimeSeries:
        pass

    @property
    def future_encoders(self) -> List[SingleEncoder]:
        """returns the future covariate encoder objects"""
        return self._future_encoders

    @property
    def past_encoders(self) -> List[SingleEncoder]:
        """returns the past covariate encoder objects"""
        return self._past_encoders

    @property
    def encoder_map(self) -> Dict:
        """mapping between encoder identifier string (from parameters at model creations) and the corresponding
        future or past covariate encoder"""
        mapper = {
            'cyclic_past': CyclicPastEncoder,
            'cyclic_future': CyclicFutureEncoder,
            'datetime_attribute_past': DatetimeAttributePastEncoder,
            'datetime_attribute_future': DatetimeAttributeFutureEncoder,
            'positional_past': PositionalPastEncoder,
            'positional_future': PositionalFutureEncoder
        }
        return mapper

    def _setup_encoders(self, params: Dict):
        """Sets up/Initializes all past and future encoders from parameters `params` used at model creation.

        Parameters
        ----------
        params
            Parameters (kwargs) used at model creation. Relevant parameters are:
            * add_encoders={
                'cyclic': {'past': ['month', 'dayofweek', ...], 'future': [same as for 'past']}
            }
        """
        past_encoders, future_encoders = self._process_input(params)

        if not past_encoders and not future_encoders:
            return

        self._past_encoders = [self.encoder_map[enc_id](self.input_chunk_length,
                                                        self.output_chunk_length,
                                                        attr) for enc_id, attr in past_encoders]
        self._future_encoders = [self.encoder_map[enc_id](self.input_chunk_length,
                                                          self.output_chunk_length,
                                                          attr) for enc_id, attr in future_encoders]

    def _process_input(self, params: Dict) -> Tuple[List, List]:
        """processes input and returns two lists of tuples `(encoder_id, attribute)` from relevant encoder
        parameters at model creation.

        `params` must be a dictionary of form `encoder_kwarg=Dict[str, Union[str, Sequence[str]]]`.
        For example with cyclic encoders

        Parameters
        ----------
        params
            Parameters (kwargs) used at model creation. Relevant parameters are:

            * add_encoders={
                    'cyclic': {'past': ['month', 'dayofweek', ...], 'future': [same as for 'past']},
                    ...
                }

            for example:
            model = MyModel(..., add_encoders={...}, ...)

            Tuples of `(encoder_id, attribute)` are extracted from the parameters to instantiate the `SingleEncoder`
            objects:
            * The `encoder_id` is extracted as follows:
                str(key) + str(temporal_key) -> 'cyclic' + 'past' -> `encoder_id` = 'cyclic_past'
                The `encoder_ix` is used to map the model parameters with the corresponding `SingleEncoder` objects.
            * The `attribute` is extracted from the values given by `temporal_key`
                `attribute` = 'month'
                ...
                The `attribute` tells the `SingleEncoder` which attribute of the index to encoder

        Raises
        ------
        ValueError
            1) if the outermost key is other than (`past`, `future`, `absolute`)
            2) if the innermost values are other than type `str` or `Sequence`
        """
        # extract encoder params
        params_encoder = params.get(ENCODER_KWARG, {})
        encoders = {enc: params_encoder.get(enc, None) for enc in ENCODER_KEYS if params_encoder.get(enc, None)}

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
