"""
Encoder Classes Main
------------------------------
"""

import pandas as pd
import numpy as np

from typing import Union, Optional, Dict, List, Sequence, Tuple

from darts import TimeSeries
from darts.utils.data.encoder_base import (
    CovariateIndexGenerator,
    PastCovariateIndexGenerator,
    FutureCovariateIndexGenerator,
    Encoder,
    SingleEncoder
)
from darts.utils.data.utils import _index_diff

from darts.logging import raise_if_not, get_logger, raise_if

from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries


SupportedTimeSeries = Union[TimeSeries, Sequence[TimeSeries]]
SupportedIndexes = Union[pd.DatetimeIndex, pd.Int64Index, pd.RangeIndex]
logger = get_logger(__name__)

ENCODER_KEYS = ['cyclic', 'datetime_attribute', 'position']
FUTURE = 'future'
PAST = 'past'
VALID_TIME_PARAMS = [
    FUTURE,
    PAST
    # 'absolute'
]
VALID_DTYPES = (str, Sequence)
DIMS = ('time', 'component', 'sample')


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


class IntegerIndexEncoder(SingleEncoder):
    """IntegerIndexEncoder: Adds integer index value (position) derived from the underlying TimeSeries' time index.
    """

    def __init__(self, index_generator: CovariateIndexGenerator, attribute: str):
        """
        Parameters
        ----------
        index_generator
            an instance of `CovariateIndexGenerator` with methods `generate_train_series()` and
            `generate_inference_series()`. Used to generate the index for encoders.
        attribute
            either 'absolute' or 'relative'.
            If 'absolute', the generated encoded values will range from (0, inf) and the train target series
            will be used as a reference to set the 0-index.
            If 'relative', the generated encoded values will range from (-inf, inf) and the train target series
            end time will be used as a reference to evaluate the relative index positions.
        """
        raise_if_not(attribute in ['absolute', 'relative'],
                     f'Encountered invalid encoder argument `{attribute}` for encoder `position`.'
                     f'Attribute must be one of `("absolute", "relative")`.')

        super(IntegerIndexEncoder, self).__init__(index_generator)
        self.attribute = attribute
        self.was_called = False
        self.current_start_index: Tuple[int, Optional[Union[pd.Timestamp, int]]] = (0, None)
        self.last_end_index: Optional[Tuple[int, Optional[Union[pd.Timestamp, int]]]] = None
        self.train_end_index: Tuple[int, Optional[Union[pd.Timestamp, int]]] = (0, None)
        self.freq = None

    def _encode(self, index: SupportedIndexes) -> TimeSeries:
        """applies cyclic encoding from `datetime_attribute_timeseries()` to `self.attribute` of `index`."""
        super(IntegerIndexEncoder, self)._encode(index)

        # initialize encoder -> required to correctly assign integer index for train, validation and inference datasets
        if not self.was_called:
            self.freq = index.freq
            self.last_end_index = (0, index[0] - self.freq)
            self.prediction_point = (len(index) - 1, index[-1])
            self.was_called = True

        # get the difference between last index and reference index for each case
        current_start_value = index[0]
        if self.attribute == 'absolute':
            reference_index, reference_value = self.last_end_index
            index_diff = _index_diff(self=current_start_value, other=reference_value, freq=self.freq)
            current_start_index = reference_index - index_diff - 1
        else:  # relative
            reference_index, reference_value = self.prediction_point
            index_diff = _index_diff(self=current_start_value, other=reference_value, freq=self.freq)
            current_start_index = -index_diff

        encoded = TimeSeries.from_times_and_values(times=index,
                                                   values=np.arange(current_start_index,
                                                                    current_start_index + len(index)),
                                                   columns=[self.attribute + '_idx'])

        self.last_end_index = (current_start_index + len(encoded), encoded.time_index[-1])
        return encoded


class IntegerIndexPastEncoder(IntegerIndexEncoder):
    """IntegerIndexEncoder: Adds integer index value (position) for past covariates derived from the underlying
    TimeSeries' time index.
    """

    def __init__(self, input_chunk_length, output_chunk_length, attribute):
        """
        Parameters
        ----------
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        attribute
            either 'absolute' or 'relative'.
            If 'absolute', the generated encoded values will range from (0, inf) and the train target series
            will be used as a reference to set the 0-index.
            If 'relative', the generated encoded values will range from (-inf, inf) and the train target series
            end time will be used as a reference to evaluate the relative index positions.
        """
        super(IntegerIndexPastEncoder, self).__init__(
            index_generator=PastCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            attribute=attribute
        )


class IntegerIndexFutureEncoder(IntegerIndexEncoder):
    """IntegerIndexEncoder: Adds integer index value (position) for future covariates derived from the underlying
    TimeSeries' time index.
    """

    def __init__(self, input_chunk_length, output_chunk_length, attribute):
        """
        Parameters
        ----------
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        attribute
            either 'absolute' or 'relative'.
            If 'absolute', the generated encoded values will range from (0, inf) and the train target series
            will be used as a reference to set the 0-index.
            If 'relative', the generated encoded values will range from (-inf, inf) and the train target series
            end time will be used as a reference to evaluate the relative index positions.
        """
        super(IntegerIndexFutureEncoder, self).__init__(
            index_generator=FutureCovariateIndexGenerator(input_chunk_length, output_chunk_length),
            attribute=attribute
        )


class SequenceEncoder(Encoder):
    """A sequence encoder can store and control multiple past and future covariate encoders at once.
    It provides the same functionality as single encoders (`encode_train()` and `encode_inference()`).
    Sequence encoders can be used with Darts' darts `Datasets`.
    """

    def __init__(self,
                 add_encoders: Dict,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 shift: int,
                 takes_past_covariates: bool = False,
                 takes_future_covariates: bool = False) -> None:

        """
        SequenceEncoder automatically creates encoder objects from parameter `add_encoders` used when creating a
        `TorchForecastingModel`.

        *   Only kwarg `add_encoders` of type `Optional[Dict]` will be used to extract the encoders.
            For example: `model = MyModel(..., add_encoders={...}, ...)`

        The `add_encoders` dict must follow this convention:
            `{encoder keyword: {temporal keyword: List[attributes]}}`
        Supported encoder keywords:
            `cyclic` for cyclic temporal encoder. See the docs :meth:`CyclicTemporalEncoder
            <darts.utils.data.encoders.CyclicTemporalEncoder>`;
            `datetime_attribute` for adding scalar information of pd.DatetimeIndex attribute. See the docs
            :meth:`DatetimeAttributeEncoder <darts.utils.data.encoders.DatetimeAttributeEncoder>`
        Supported temporal keywords:
            'past' for adding encoding as past covariates
            'future' for adding encoding as future covariates
        Supported attributes:
            for attributes read the referred docs for the corresponding encoder from above
        An example of a valid `add_encoders` dict for hourly data for :
            add_encoders={
                'cyclic': {'future': ['month']},
                'datetime_attribute': {'past': ['hour'], 'future': ['year', 'dayofweek']}
            }

        Tuples of `(encoder_id, attribute)` are extracted from `add_encoders` to instantiate the `SingleEncoder`
        objects:
        * The `encoder_id` is extracted as follows:
            str(encoder_kw) + str(temporal_kw) -> 'cyclic' + 'past' -> `encoder_id` = 'cyclic_past'
            The `encoder_id` is used to map the parameters with the corresponding `SingleEncoder` objects.
        * The `attribute` is extracted from the values given by values under `temporal_kw`
            `attribute` = 'month'
            ...
            The `attribute` tells the `SingleEncoder` which attribute of the index to encode

        New encoders can be added by appending them to the mapping property `SequenceEncoder.encoder_map()`

        Parameters
        ----------
        add_encoders
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
        self.params = add_encoders
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.shift = shift
        self._past_encoders: List[SingleEncoder] = []
        self._future_encoders: List[SingleEncoder] = []
        self.takes_past_covariates = takes_past_covariates
        self.takes_future_covariates = takes_future_covariates
        self.encoding_available = False
        self.train_called = False

        self._setup_encoders(self.params)

    def encode_train(self,
                     target: SupportedTimeSeries,
                     past_covariate: Optional[SupportedTimeSeries] = None,
                     future_covariate: Optional[SupportedTimeSeries] = None,
                     encode_past: bool = True,
                     encode_future: bool = True,
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

        Raises
        Warning
            If model was created with `add_encoders` and there is suspicion of lazy loading.
            The encodings/covariates are generated pre-train for all individual targets and
            loaded into memory. Depending on the size of target data, this can create memory
            issues. In case this applies, consider setting `add_encoders=None` at model
            creation
        """
        if not self.train_called:
            if not isinstance(target, (TimeSeries, list)):
                logger.warning("Fitting was called with `add_encoders` and suspicion of lazy loading. "
                               "The encodings/covariates are generated pre-train for all individual targets and "
                               "loaded into memory. Depending on the size of your data, this can create memory issues. "
                               "In case this applies, consider setting `add_encoders=None` at model creation.")
            self.train_called = True

        return self._launch_encoder(target=target,
                                    past_covariate=past_covariate,
                                    future_covariate=future_covariate,
                                    n=None,
                                    encode_past=encode_past,
                                    encode_future=encode_future)

    def encode_inference(self,
                         n: int,
                         target: SupportedTimeSeries,
                         past_covariate: Optional[SupportedTimeSeries] = None,
                         future_covariate: Optional[SupportedTimeSeries] = None,
                         encode_past: bool = True,
                         encode_future: bool = True,
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
                                    n=n,
                                    encode_past=encode_past,
                                    encode_future=encode_future)

    def _launch_encoder(self,
                        target: Sequence[TimeSeries],
                        past_covariate: SupportedTimeSeries,
                        future_covariate: SupportedTimeSeries,
                        n: Optional[int] = None,
                        encode_past: bool = True,
                        encode_future: bool = True) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        """Launches the encode sequence for past covariate and future covariate for either training or
        inference/prediction.

        If `n` is `None` it is a prediction, otherwise it is training.
        """

        if not self.encoding_available:
            return past_covariate, future_covariate

        target = [target] if isinstance(target, TimeSeries) else target

        if self.past_encoders and encode_past:
            past_covariate = self._encode_sequence(encoders=self.past_encoders,
                                                   target=target,
                                                   covariate=past_covariate,
                                                   n=n)

        if self.future_encoders and encode_future:
            future_covariate = self._encode_sequence(encoders=self.future_encoders,
                                                     target=target,
                                                     covariate=future_covariate,
                                                     n=n)

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
            'position_past': IntegerIndexPastEncoder,
            'position_future': IntegerIndexFutureEncoder
        }
        return mapper

    def _setup_encoders(self, params: Dict):
        """Sets up/Initializes all past and future encoders from `add_encoder` parameter used at model creation.

        Parameters
        ----------
        params
            Dict from parameter `add_encoders` (kwargs) used at model creation. Relevant parameters are:
            * params={'cyclic': {'past': ['month', 'dayofweek', ...], 'future': [same as for 'past']}}
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
        self.encoding_available = True

    def _process_input(self, params: Dict) -> Tuple[List, List]:
        """processes input and returns two lists of tuples `(encoder_id, attribute)` from relevant encoder
        parameters at model creation.

        Parameters
        ----------
        params
            A dict of type Optional[Dict] from parameter `add_encoders` used at model creation.

            For example: `model = MyModel(..., add_encoders={...}, ...)`

            The `params`/`add_encoders` dict must follow this convention:
                `{encoder keyword: {temporal keyword: List[attributes]}}`
            Supported encoder keywords:
                `cyclic` for cyclic temporal encoder. See the docs :meth:`CyclicTemporalEncoder
                <darts.utils.data.encoders.CyclicTemporalEncoder>`;
                `datetime_attribute` for adding scalar information of pd.DatetimeIndex attribute. See the docs
                :meth:`DatetimeAttributeEncoder <darts.utils.data.encoders.DatetimeAttributeEncoder>`
            Supported temporal keywords:
                'past' for adding encoding as past covariates
                'future' for adding encoding as future covariates
            Supported attributes:
                for attributes read the referred docs for the corresponding encoder from above
            An example of a valid `add_encoders` dict at model creation for hourly data for :
                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'past': ['hour'], 'future': ['year', 'dayofweek']}
                }

            Tuples of `(encoder_id, attribute)` are extracted from `add_encoders` to instantiate the `SingleEncoder`
            objects:
            * The `encoder_id` is extracted as follows:
                str(encoder_kw) + str(temporal_kw) -> 'cyclic' + 'past' -> `encoder_id` = 'cyclic_past'
                The `encoder_id` is used to map the parameters with the corresponding `SingleEncoder` objects.
            * The `attribute` is extracted from the values given by values under `temporal_kw`
                `attribute` = 'month'
                ...
                The `attribute` tells the `SingleEncoder` which attribute of the index to encode

        Raises
        ------
        ValueError
            1) if the outermost key is other than (`past`, `future`, `absolute`)
            2) if the innermost values are other than type `str` or `Sequence`
        """

        if not params:
            return [], []

        encoders = {enc: params.get(enc, None) for enc in ENCODER_KEYS if params.get(enc, None)}

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
