"""
Time Axes Encoders
------------------

Encoders can generate past and/or future covariate series by encoding the index of a TimeSeries `series`.
Each encoder class has an `encode_train()` and `encode_inference()` to generate the encodings for training and
inference.

The encoders extract the index either from the target series or optional additional past/future covariates.
If additional covariates are supplied to `encode_train()` or `encode_inference()`, the time index of those
covariates are used for the encodings. This means that the input covariates must meet the same model-specific
requirements as wihtout encoders.

There are two main types of encoder classes: `SingleEncoder` and `SequentialEncoder`.

*   SingleEncoder
        The SingleEncoder classes carry the encoder logic for past and future covariates, and training and
        inference datasets. They can be used as stand-alone encoders.

        Each SingleEncoder has a dedicated subclass for generating past or future covariates. The naming convention
        is `{X}{SingleEncoder}` where {X} is one of (Past, Future) and {SingleEncoder} is one of the SingleEncoder
        classes described in the next section. An example:

        .. highlight:: python
        .. code-block:: python

            encoder = PastDatetimeAttributeEncoder(input_chunk_length=24,
                                                   output_chunk_length=12,
                                                   attribute='month')

            past_covariates_train = encoder.encode_train(target=target,
                                                         covariate=optional_past_covariates)
            past_covariates_inf = encoder.encode_inference(n=12,
                                                           target=target,
                                                           covariate=optional_past_covariates)

*   SequentialEncoder
        Stores and controls multiple SingleEncoders for both past and/or future covariates all under one hood.
        It provides the same functionality as SingleEncoders (`encode_train()` and `encode_inference()`).
        It can be used both as stand-alone or as an all-in-one solution with Darts' `TorchForecastingModel` models
        through optional parameter `add_encoders`:

        .. highlight:: python
        .. code-block:: python

            model = SomeTorchForecastingModel(..., add_encoders={...})
        ..

        If used at model creation, the SequentialEncoder will handle all past and future encoders autonomously.
        The requirements for model parameter `add_encoders` are described in the next section or in
        :meth:`SequentialEncoder <darts.utils.data.encoders.SequentialEncoder>`.

SingleEncoder
-------------

The SingleEncoders from {X}{SingleEncoder} are:

*   DatetimeAttributeEncoder
        Adds scalar pd.DatatimeIndex attribute information derived from `series.time_index`.
        Requires `series` to have a pd.DatetimeIndex.

        attribute
            An attribute of `pd.DatetimeIndex`: see all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
*   CyclicTemporalEncoder
        Adds cyclic pd.DatetimeIndex attribute information deriveed from `series.time_index`.
        Adds 2 columns, corresponding to sin and cos encodings, to uniquely describe the underlying attribute.
        Requires `series` to have a pd.DatetimeIndex.

        attribute
            An attribute of `pd.DatetimeIndex` that follows a cyclic pattern. One of ('month', 'day', 'weekday',
            'dayofweek', 'day_of_week', 'hour', 'minute', 'second', 'microsecond', 'nanosecond', 'quarter',
            'dayofyear', 'day_of_year', 'week', 'weekofyear', 'week_of_year').
*   IntegerIndexEncoder
        Adds absolute or relative index positions as integer values (positions) derived from `series` time index.
        `series` can either have a pd.DatetimeIndex or an integer index.

        attribute
            Either 'absolute' or 'relative'.
            'absolute' will generate position values ranging from 0 to inf where 0 is set at the start of `series`.
            'relative' will generate position values relative to the forecasting/prediction point. Values range
            from -inf to inf where 0 is set at the forecasting point.
*   CallableIndexEncoder
        Applies a user-defined callable to encode `series`' index.
        `series` can either have a pd.DatetimeIndex or an integer index.

        attribute
            a callable/function to encode the index.
            For `series` with a pd.DatetimeIndex: ``lambda index: (index.year - 1950) / 50``.
            For `series` with an integer index: ``lambda index: index / 50``

SequentialEncoder
-----------------

The SequentialEncoder combines the logic of all SingleEncoders from above and has additional benefits:

*   use multiple encoders at once
*   generate multiple attribute encodings at once
*   generate both past and future at once
*   supports transformers (Scaler)
*   easy to use with TorchForecastingModels

The model parameter `add_encoders` must be a Dict following of this convention:

*   outer keys: `SingleEncoder` and Transformer tags:

    *   'datetime_attribute' for `DatetimeAttributeEncoder`
    *   'cyclic' for `CyclicEncoder`
    *   'position' for `IntegerIndexEncoder`
    *   'custom' for `CallableIndexEncoder`
    *   'transformer' for a transformer
*   inner keys: covariate type

    *   'past' for past covariates
    *   'future' for future covariates
    *   (do not specify for 'transformer')
*   inner key values:

    *   list of attributes for `SingleEncoder`
    *   transformer object for 'transformer'

Below is an example that illustrates a valid `add_encoders` dict for hourly data and how it can be used with a
TorchForecastingModel (this is only meant to illustrate many features at once).

.. highlight:: python
.. code-block:: python

    add_encoders = {
        'cyclic': {'future': ['month']},
        'datetime_attribute': {'future': ['hour', 'dayofweek']},
        'position': {'past': ['absolute'], 'future': ['relative']},
        'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
        'transformer': Scaler()
    }

    model = SomeTorchForecastingModel(..., add_encoders=add_encoders)
..
"""

import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import FittableDataTransformer
from darts.logging import get_logger, raise_if, raise_if_not
from darts.timeseries import DIMS
from darts.utils.data.encoder_base import (
    CovariateIndexGenerator,
    Encoder,
    FutureCovariateIndexGenerator,
    PastCovariateIndexGenerator,
    ReferenceIndexType,
    SequentialEncoderTransformer,
    SingleEncoder,
    SupportedIndex,
)
from darts.utils.data.utils import _index_diff
from darts.utils.timeseries_generation import datetime_attribute_timeseries

SupportedTimeSeries = Union[TimeSeries, Sequence[TimeSeries]]
logger = get_logger(__name__)

ENCODER_KEYS = ["cyclic", "datetime_attribute", "position", "custom"]
FUTURE = "future"
PAST = "past"
VALID_TIME_PARAMS = [FUTURE, PAST]
VALID_ENCODER_DTYPES = (str, Sequence)

TRANSFORMER_KEYS = ["transformer"]
VALID_TRANSFORMER_DTYPES = FittableDataTransformer
INTEGER_INDEX_ATTRIBUTES = ["absolute", "relative"]


class CyclicTemporalEncoder(SingleEncoder):
    def __init__(self, index_generator: CovariateIndexGenerator, attribute: str):
        """
        Cyclic index encoding for `TimeSeries` that have a time index of type `pd.DatetimeIndex`.

        Parameters
        ----------
        index_generator
            An instance of `CovariateIndexGenerator` with methods `generate_train_series()` and
            `generate_inference_series()`. Used to generate the index for encoders.
        attribute
            The attribute of the underlying pd.DatetimeIndex from  for which to apply cyclic encoding.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        """
        super().__init__(index_generator)
        self.attribute = attribute

    def _encode(self, index: SupportedIndex, dtype: np.dtype) -> TimeSeries:
        """applies cyclic encoding from `datetime_attribute_timeseries()` to `self.attribute` of `index`."""
        super()._encode(index, dtype)
        return datetime_attribute_timeseries(
            index, attribute=self.attribute, cyclic=True, dtype=dtype
        )

    @property
    def accept_transformer(self) -> List[bool]:
        """CyclicTemporalEncoder should not be transformed. Returns two elements for sine and cosine waves."""
        return [False, False]


class PastCyclicEncoder(CyclicTemporalEncoder):
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
            The attribute of the underlying pd.DatetimeIndex from  for which to apply cyclic encoding.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        """
        super().__init__(
            index_generator=PastCovariateIndexGenerator(
                input_chunk_length, output_chunk_length
            ),
            attribute=attribute,
        )


class FutureCyclicEncoder(CyclicTemporalEncoder):
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
            The attribute of the underlying pd.DatetimeIndex from  for which to apply cyclic encoding.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        """
        super().__init__(
            index_generator=FutureCovariateIndexGenerator(
                input_chunk_length, output_chunk_length
            ),
            attribute=attribute,
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
            An instance of `CovariateIndexGenerator` with methods `generate_train_series()` and
            `generate_inference_series()`. Used to generate the index for encoders.
        attribute
            The attribute of the underlying pd.DatetimeIndex from  for which to add scalar information.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        """
        super().__init__(index_generator)
        self.attribute = attribute

    def _encode(self, index: SupportedIndex, dtype: np.dtype) -> TimeSeries:
        """Applies cyclic encoding from `datetime_attribute_timeseries()` to `self.attribute` of `index`."""
        super()._encode(index, dtype)
        return datetime_attribute_timeseries(
            index, attribute=self.attribute, dtype=dtype
        )

    @property
    def accept_transformer(self) -> List[bool]:
        """DatetimeAttributeEncoder accepts transformations"""
        return [True]


class PastDatetimeAttributeEncoder(DatetimeAttributeEncoder):
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
            The attribute of the underlying pd.DatetimeIndex from  for which to add scalar information.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        """
        super().__init__(
            index_generator=PastCovariateIndexGenerator(
                input_chunk_length, output_chunk_length
            ),
            attribute=attribute,
        )


class FutureDatetimeAttributeEncoder(DatetimeAttributeEncoder):
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
            The attribute of the underlying pd.DatetimeIndex from  for which to add scalar information.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        """
        super().__init__(
            index_generator=FutureCovariateIndexGenerator(
                input_chunk_length, output_chunk_length
            ),
            attribute=attribute,
        )


class IntegerIndexEncoder(SingleEncoder):
    """IntegerIndexEncoder: Adds integer index value (position) derived from the underlying TimeSeries' time index
    for past and future covariates.
    """

    def __init__(self, index_generator: CovariateIndexGenerator, attribute: str):
        """
        Parameters
        ----------
        index_generator
            An instance of `CovariateIndexGenerator` with methods `generate_train_series()` and
            `generate_inference_series()`. Used to generate the index for encoders.
        attribute
            Either 'absolute' or 'relative'. If 'absolute', the generated encoded values will range from (0, inf)
            and the train target series will be used as a reference to set the 0-index. If 'relative', the generated
            encoded values will range from (-inf, inf) and the train target series end time will be used as a reference
            to evaluate the relative index positions.
        """
        raise_if_not(
            isinstance(attribute, str) and attribute in INTEGER_INDEX_ATTRIBUTES,
            f"Encountered invalid encoder argument `{attribute}` for encoder `position`. "
            f'Attribute must be one of `("absolute", "relative")`.',
            logger,
        )

        super().__init__(index_generator)

        self.attribute = attribute
        self.reference_index: Optional[
            Tuple[int, Optional[Union[pd.Timestamp, int]]]
        ] = None
        self.was_called = False

    def _encode(self, index: SupportedIndex, dtype: np.dtype) -> TimeSeries:
        """Applies cyclic encoding from `datetime_attribute_timeseries()` to `self.attribute` of `index`.
        1)  for attribute=='absolute', the reference point/index is one step before start of the train target series
        2)  for attribute=='relative', the reference point/index is the overall prediction/forecast index
        """
        super()._encode(index, dtype)

        # load reference index from index_generators
        if not self.was_called:
            self.reference_index = self.index_generator.reference_index
            self.was_called = True

        current_start_value = index[0]

        # extract reference index
        reference_index, reference_value = self.reference_index

        # get the difference between last index and reference index for each case
        index_diff = _index_diff(
            self=current_start_value, other=reference_value, freq=index.freq
        )
        # set the start integer index value for the current index
        current_start_index = (
            reference_index - index_diff
            if self.attribute == "absolute"
            else -index_diff
        )

        encoded = TimeSeries.from_times_and_values(
            times=index,
            values=np.arange(current_start_index, current_start_index + len(index)),
            columns=[self.attribute + "_idx"],
        ).astype(np.dtype(dtype))

        # update reference index for 'absolute' case to avoid having to evaluate longer differences (cost-intensive)
        if self.attribute == "absolute":
            self.reference_index = (
                current_start_index + len(encoded) - 1,
                encoded.time_index[-1],
            )
        return encoded

    @property
    def accept_transformer(self) -> List[bool]:
        """IntegerIndexEncoder accepts transformations. Note that transforming 'relative' IntegerIndexEncoder
        will return an 'absolute' index."""
        return [True]


class PastIntegerIndexEncoder(IntegerIndexEncoder):
    """IntegerIndexEncoder: Adds integer index value (position) for past covariates derived from the underlying
    TimeSeries' time index.
    """

    def __init__(
        self, input_chunk_length: int, output_chunk_length: int, attribute: str
    ):
        """
        Parameters
        ----------
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        attribute
            Either 'absolute' or 'relative'. If 'absolute', the generated encoded values will range from (0, inf)
            and the train target series will be used as a reference to set the 0-index. If 'relative', the generated
            encoded values will range from (-inf, inf) and the train target series end time will be used as a reference
            to evaluate the relative index positions.
        """
        reference_index_type = (
            ReferenceIndexType.PREDICTION
            if attribute == "relative"
            else ReferenceIndexType.START
        )

        super().__init__(
            index_generator=PastCovariateIndexGenerator(
                input_chunk_length,
                output_chunk_length,
                reference_index_type=reference_index_type,
            ),
            attribute=attribute,
        )


class FutureIntegerIndexEncoder(IntegerIndexEncoder):
    """IntegerIndexEncoder: Adds integer index value (position) for future covariates derived from the underlying
    TimeSeries' time index.
    """

    def __init__(
        self, input_chunk_length: int, output_chunk_length: int, attribute: str
    ):
        """
        Parameters
        ----------
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        attribute
            Either 'absolute' or 'relative'. If 'absolute', the generated encoded values will range from (0, inf)
            and the train target series will be used as a reference to set the 0-index. If 'relative', the generated
            encoded values will range from (-inf, inf) and the train target series end time will be used as a reference
            to evaluate the relative index positions.
        """
        reference_index_type = (
            ReferenceIndexType.PREDICTION
            if attribute == "relative"
            else ReferenceIndexType.START
        )

        super().__init__(
            index_generator=FutureCovariateIndexGenerator(
                input_chunk_length,
                output_chunk_length,
                reference_index_type=reference_index_type,
            ),
            attribute=attribute,
        )


class CallableIndexEncoder(SingleEncoder):
    """CallableIndexEncoder: Applies a user-defined callable to encode the underlying index for past and future
    covariates.
    """

    def __init__(self, index_generator: CovariateIndexGenerator, attribute: Callable):
        """
        Parameters
        ----------
        index_generator
            An instance of `CovariateIndexGenerator` with methods `generate_train_series()` and
            `generate_inference_series()`. Used to generate the index for encoders.
        attribute
            A callable that takes an index `index` of type `(pd.DatetimeIndex, pd.RangeIndex)` as input
            and returns a np.ndarray of shape `(len(index),)`.
            An example for a correct `attribute` for `index` of type pd.DatetimeIndex:
            ``attribute = lambda index: (index.year - 1950) / 50``. And for pd.RangeIndex:
            ``attribute = lambda index: (index - 1950) / 50``
        """
        raise_if_not(
            callable(attribute),
            f"Encountered invalid encoder argument `{attribute}` for encoder `callable`. "
            f"Attribute must be a callable that returns a `np.ndarray`.",
            logger,
        )

        super().__init__(index_generator)

        self.attribute = attribute

    def _encode(self, index: SupportedIndex, dtype: np.dtype) -> TimeSeries:
        """Apply the user-defined callable to encode the index"""
        super()._encode(index, dtype)

        return TimeSeries.from_times_and_values(
            times=index, values=self.attribute(index), columns=["custom"]
        ).astype(np.dtype(dtype))

    @property
    def accept_transformer(self) -> List[bool]:
        """CallableIndexEncoder accepts transformations."""
        return [True]


class PastCallableIndexEncoder(CallableIndexEncoder):
    """IntegerIndexEncoder: Adds integer index value (position) for past covariates derived from the underlying
    TimeSeries' time index.
    """

    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        attribute: Union[str, Callable],
    ):
        """
        Parameters
        ----------
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        attribute
            A callable that takes an index `index` of type `(pd.DatetimeIndex, pd.RangeIndex)` as input
            and returns a np.ndarray of shape `(len(index),)`.
            An example for a correct `attribute` for `index` of type pd.DatetimeIndex:
            ``attribute = lambda index: (index.year - 1950) / 50``. And for pd.RangeIndex:
            ``attribute = lambda index: (index - 1950) / 50``
        """
        super().__init__(
            index_generator=PastCovariateIndexGenerator(
                input_chunk_length, output_chunk_length
            ),
            attribute=attribute,
        )


class FutureCallableIndexEncoder(CallableIndexEncoder):
    """IntegerIndexEncoder: Adds integer index value (position) for future covariates derived from the underlying
    TimeSeries' time index.
    """

    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,
        attribute: Union[str, Callable],
    ):
        """
        Parameters
        ----------
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        attribute
            A callable that takes an index `index` of type `(pd.DatetimeIndex, pd.RangeIndex)` as input
            and returns a np.ndarray of shape `(len(index),)`.
            An example for a correct `attribute` for `index` of type pd.DatetimeIndex:
            ``attribute = lambda index: (index.year - 1950) / 50``. And for pd.RangeIndex:
            ``attribute = lambda index: (index - 1950) / 50``
        """

        super().__init__(
            index_generator=FutureCovariateIndexGenerator(
                input_chunk_length, output_chunk_length
            ),
            attribute=attribute,
        )


class SequentialEncoder(Encoder):
    """A `SequentialEncoder` object can store and control multiple past and future covariate encoders at once.
    It provides the same functionality as single encoders (`encode_train()` and `encode_inference()`).
    """

    def __init__(
        self,
        add_encoders: Dict,
        input_chunk_length: int,
        output_chunk_length: int,
        takes_past_covariates: bool = False,
        takes_future_covariates: bool = False,
    ) -> None:

        """
        SequentialEncoder automatically creates encoder objects from parameter `add_encoders` used when creating a
        `TorchForecastingModel`.

        *   Only kwarg `add_encoders` of type `Optional[Dict]` will be used to extract the encoders.
            For example: `model = MyModel(..., add_encoders={...}, ...)`

        The `add_encoders` dict must follow this convention:
            `{encoder keyword: {temporal keyword: List[attributes]}, ..., transformer keyword: transformer object}`
        Supported encoder keywords:
            `'cyclic'` for cyclic temporal encoder. See the docs
            :meth:`CyclicTemporalEncoder <darts.utils.data.encoders.CyclicTemporalEncoder>`;
            `'datetime_attribute'` for adding scalar information of pd.DatetimeIndex attribute. See the docs
            :meth:`DatetimeAttributeEncoder <darts.utils.data.encoders.DatetimeAttributeEncoder>`
            `'position'` for integer index position encoder. See the docs
            :meth:`IntegerIndexEncoder <darts.utils.data.encoders.IntegerIndexEncoder>`;
            `'custom'` for encoding index with custom callables (functions). See the docs
            :meth:`CallableIndexEncoder <darts.utils.data.encoders.CallableIndexEncoder>`;
        Supported temporal keywords:
            'past' for adding encoding as past covariates
            'future' for adding encoding as future covariates
        Supported attributes:
            for attributes read the referred docs for the corresponding encoder from above
        Supported transformers:
            a transformer can be added with transformer keyword 'transformer'. The transformer object must be an
            instance of Darts' :meth:`FittableDataTransformer
            <darts.dataprocessing.transformers.fittable_data_transformer.FittableDataTransformer>` such as Scaler() or
            BoxCox(). The transformers will be fitted on the training dataset when calling calling `model.fit()`.
            The training, validation and inference datasets are then transformed equally.

        An example of a valid `add_encoders` dict for hourly data:

            .. highlight:: python
            .. code-block:: python

                from darts.dataprocessing.transformers import Scaler
                add_encoders={
                    'cyclic': {'future': ['month']},
                    'datetime_attribute': {'past': ['hour'], 'future': ['year', 'dayofweek']},
                    'position': {'past': ['absolute'], 'future': ['relative']},
                    'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
                    'transformer': Scaler()
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

        New encoders can be added by appending them to the mapping property `SequentialEncoder.encoder_map()`

        Parameters
        ----------
        add_encoders
            The parameters used at `TorchForecastingModel` model creation.
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        takes_past_covariates
            Whether or not the `TrainingDataset` takes past covariates
        takes_future_covariates
            Whether or not the `TrainingDataset` takes past covariates
        """

        super().__init__()
        self.params = add_encoders
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.train_called = False
        self.encoding_available = False
        self.takes_past_covariates = takes_past_covariates
        self.takes_future_covariates = takes_future_covariates

        # encoders
        self._past_encoders: List[SingleEncoder] = []
        self._future_encoders: List[SingleEncoder] = []

        # transformer
        self._past_transformer: Optional[SequentialEncoderTransformer] = None
        self._future_transformer: Optional[SequentialEncoderTransformer] = None

        # setup encoders and transformer
        self._setup_encoders(self.params)
        self._setup_transformer(self.params)

    def encode_train(
        self,
        target: SupportedTimeSeries,
        past_covariate: Optional[SupportedTimeSeries] = None,
        future_covariate: Optional[SupportedTimeSeries] = None,
        encode_past: bool = True,
        encode_future: bool = True,
    ) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        """Returns encoded index for all past and/or future covariates for training.
        Which covariates are generated depends on the parameters used at model creation.

        Parameters
        ----------
        target
            The target TimeSeries used during training or passed to prediction as `series`
        past_covariate
            Optionally, the past covariates used for training.
        future_covariate
            Optionally, the future covariates used for training.
        encode_past
            Whether or not to apply encoders for past covariates
        encode_future
            Whether or not to apply encoders for future covariates
        Returns
        -------
        Tuple[past_covariate, future_covariate]
            The past_covariate and/or future_covariate for training including the encodings.
            If input {x}_covariate is None and no {x}_encoders are given, will return `None`
            for the {x}_covariate.
        Raises
        ------
        Warning
            If model was created with `add_encoders` and there is suspicion of lazy loading.
            The encodings/covariates are generated eagerly before starting training for all individual targets and
            loaded into memory. Depending on the size of target data, this can create memory
            issues. In case this applies, consider setting `add_encoders=None` at model
            creation and build your encodings covariates manually for lazy loading.
        """
        if not self.train_called:
            if not isinstance(target, (TimeSeries, list)):
                logger.warning(
                    "Fitting was called with `add_encoders` and suspicion of lazy loading. "
                    "The encodings/covariates are generated pre-train for all individual targets and "
                    "loaded into memory. Depending on the size of your data, this can create memory issues. "
                    "In case this applies, consider setting `add_encoders=None` at model creation."
                )

            self.train_called = True

        return self._launch_encoder(
            target=target,
            past_covariate=past_covariate,
            future_covariate=future_covariate,
            n=None,
            encode_past=encode_past,
            encode_future=encode_future,
        )

    def encode_inference(
        self,
        n: int,
        target: SupportedTimeSeries,
        past_covariate: Optional[SupportedTimeSeries] = None,
        future_covariate: Optional[SupportedTimeSeries] = None,
        encode_past: bool = True,
        encode_future: bool = True,
    ) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        """Returns encoded index for all past and/or future covariates for inference/prediction.
        Which covariates are generated depends on the parameters used at model creation.

        Parameters
        ----------
        n
            The forecast horizon
        target
            The target TimeSeries used during training or passed to prediction as `series`
        past_covariate
            Optionally, the past covariates used for training.
        future_covariate
            Optionally, the future covariates used for training.
        encode_past
            Whether or not to apply encoders for past covariates
        encode_future
            Whether or not to apply encoders for future covariates

        Returns
        -------
        Tuple[past_covariate, future_covariate]
            The past_covariate and/or future_covariate for prediction/inference including the encodings.
            If input {x}_covariate is None and no {x}_encoders are given, will return `None`
            for the {x}_covariate.
        """

        return self._launch_encoder(
            target=target,
            past_covariate=past_covariate,
            future_covariate=future_covariate,
            n=n,
            encode_past=encode_past,
            encode_future=encode_future,
        )

    def _launch_encoder(
        self,
        target: Sequence[TimeSeries],
        past_covariate: SupportedTimeSeries,
        future_covariate: SupportedTimeSeries,
        n: Optional[int] = None,
        encode_past: bool = True,
        encode_future: bool = True,
    ) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        """Launches the encode sequence for past covariate and future covariate for either training or
        inference/prediction.

        If `n` is `None` it is a prediction, otherwise it is training.
        """

        if not self.encoding_available:
            return past_covariate, future_covariate

        target = [target] if isinstance(target, TimeSeries) else target

        if self.past_encoders and encode_past:
            past_covariate = self._encode_sequence(
                encoders=self.past_encoders,
                transformer=self.past_transformer,
                target=target,
                covariate=past_covariate,
                n=n,
            )

        if self.future_encoders and encode_future:
            future_covariate = self._encode_sequence(
                encoders=self.future_encoders,
                transformer=self.future_transformer,
                target=target,
                covariate=future_covariate,
                n=n,
            )
        return past_covariate, future_covariate

    def _encode_sequence(
        self,
        encoders: Sequence[SingleEncoder],
        transformer: Optional[SequentialEncoderTransformer],
        target: Sequence[TimeSeries],
        covariate: Optional[SupportedTimeSeries],
        n: Optional[int] = None,
    ) -> List[TimeSeries]:
        """Sequentially encodes the index of all input target/covariate TimeSeries

        If `n` is `None` it is a prediction and method `encoder.encode_inference()` is called.
        Otherwise, it is a training case and `encoder.encode_train()` is called.
        """

        encode_method = "encode_train" if n is None else "encode_inference"

        encoded_sequence = []
        if covariate is None:
            covariate = [None] * len(target)
        else:
            covariate = [covariate] if isinstance(covariate, TimeSeries) else covariate

        for ts, pc in zip(target, covariate):
            encoded = concatenate(
                [
                    getattr(enc, encode_method)(
                        target=ts, covariate=pc, merge_covariate=False, n=n
                    )
                    for enc in encoders
                ],
                axis=DIMS[1],
            )
            encoded_sequence.append(
                self._merge_covariate(encoded=encoded, covariate=pc)
            )

        if transformer is not None:
            encoded_sequence = transformer.transform(encoded_sequence)

        return encoded_sequence

    @property
    def future_encoders(self) -> List[SingleEncoder]:
        """Returns the future covariate encoder objects"""
        return self._future_encoders

    @property
    def past_encoders(self) -> List[SingleEncoder]:
        """Returns the past covariate encoder objects"""
        return self._past_encoders

    @property
    def past_transformer(self) -> SequentialEncoderTransformer:
        """Returns the past transformer object"""
        return self._past_transformer

    @property
    def future_transformer(self) -> SequentialEncoderTransformer:
        """Returns the future transformer object"""
        return self._future_transformer

    @property
    def encoder_map(self) -> Dict:
        """Mapping between encoder identifier string (from parameters at model creations) and the corresponding
        future or past covariate encoder"""
        mapper = {
            "cyclic_past": PastCyclicEncoder,
            "cyclic_future": FutureCyclicEncoder,
            "datetime_attribute_past": PastDatetimeAttributeEncoder,
            "datetime_attribute_future": FutureDatetimeAttributeEncoder,
            "position_past": PastIntegerIndexEncoder,
            "position_future": FutureIntegerIndexEncoder,
            "custom_past": PastCallableIndexEncoder,
            "custom_future": FutureCallableIndexEncoder,
        }
        return mapper

    def _setup_encoders(self, params: Dict) -> None:
        """Sets up/Initializes all past and future encoders and an optional transformer from `add_encoder` parameter
        used at model creation.


        Parameters
        ----------
        params
            Dict from parameter `add_encoders` (kwargs) used at model creation. Relevant parameters are:
            * params={'cyclic': {'past': ['month', 'dayofweek', ...], 'future': [same as for 'past']}}
        """
        past_encoders, future_encoders = self._process_input_encoders(params)

        if not past_encoders and not future_encoders:
            return

        self._past_encoders = [
            self.encoder_map[enc_id](
                self.input_chunk_length, self.output_chunk_length, attr
            )
            for enc_id, attr in past_encoders
        ]
        self._future_encoders = [
            self.encoder_map[enc_id](
                self.input_chunk_length, self.output_chunk_length, attr
            )
            for enc_id, attr in future_encoders
        ]
        self.encoding_available = True

    def _setup_transformer(self, params: Dict) -> None:
        """Sets up/Initializes an optional transformer from `add_encoder` parameter used at model creation.

        Parameters
        ----------
        params
            Dict from parameter `add_encoders` (kwargs) used at model creation. Relevant parameters are:
            * params={..., 'transformer': Scaler()}
        """
        (
            transformer,
            transform_past_mask,
            transform_future_mask,
        ) = self._process_input_transformer(params)
        if transform_past_mask:
            self._past_transformer = SequentialEncoderTransformer(
                copy.deepcopy(transformer), transform_past_mask
            )
        if transform_future_mask:
            self._future_transformer = SequentialEncoderTransformer(
                copy.deepcopy(transformer), transform_future_mask
            )

    def _process_input_encoders(self, params: Dict) -> Tuple[List, List]:
        """Processes input and returns two lists of tuples `(encoder_id, attribute)` from relevant encoder
        parameters at model creation.

        Parameters
        ----------
        params
            The `add_encoders` dict used at model creation. Must follow this convention:
                `{encoder keyword: {temporal keyword: List[attributes]}}`

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

        # check input for invalid encoder types
        invalid_encoders = [
            enc for enc in params if enc not in ENCODER_KEYS + TRANSFORMER_KEYS
        ]
        raise_if(
            len(invalid_encoders) > 0,
            f"Encountered invalid encoder types `{invalid_encoders}` in `add_encoders` parameter at model "
            f"creation. Supported encoder types are: `{ENCODER_KEYS + TRANSFORMER_KEYS}`.",
            logger,
        )

        encoders = {
            enc: params.get(enc, None) for enc in ENCODER_KEYS if params.get(enc, None)
        }

        # check input for invalid temporal types
        invalid_time_params = list()
        for encoder, t_types in encoders.items():
            invalid_time_params += [
                t_type for t_type in t_types.keys() if t_type not in VALID_TIME_PARAMS
            ]

        raise_if(
            len(invalid_time_params) > 0,
            f"Encountered invalid temporal types `{invalid_time_params}` in `add_encoders` parameter at model "
            f"creation. Supported temporal types are: `{VALID_TIME_PARAMS}`.",
            logger,
        )

        # convert into tuples of (encoder string identifier, encoder attribute)
        past_encoders, future_encoders = list(), list()
        for enc, enc_params in encoders.items():
            for enc_time, enc_attr in enc_params.items():
                raise_if_not(
                    isinstance(enc_attr, VALID_ENCODER_DTYPES),
                    f"Encountered value `{enc_attr}` of invalid type `{type(enc_attr)}` for encoder "
                    f"`{enc}` in `add_encoders` at model creation. Supported data types are: "
                    f"`{VALID_ENCODER_DTYPES}`.",
                    logger,
                )
                attrs = [enc_attr] if isinstance(enc_attr, str) else enc_attr
                for attr in attrs:
                    encoder_id = "_".join([enc, enc_time])
                    if enc_time == PAST:
                        past_encoders.append((encoder_id, attr))
                    else:
                        future_encoders.append((encoder_id, attr))

        for temp_enc, takes_temp, temp in [
            (past_encoders, self.takes_past_covariates, "past"),
            (future_encoders, self.takes_future_covariates, "future"),
        ]:
            if temp_enc and not takes_temp:
                logger.warning(
                    f"Specified {temp} encoders in `add_encoders` at model creation but model does not "
                    f"accept {temp} covariates. {temp} encoders will be ignored."
                )

        past_encoders = past_encoders if self.takes_past_covariates else []
        future_encoders = future_encoders if self.takes_future_covariates else []
        return past_encoders, future_encoders

    def _process_input_transformer(
        self, params: Dict
    ) -> Tuple[Optional[FittableDataTransformer], List, List]:
        """Processes input params used at model creation and returns tuple of one transformer object and two masks
        that specify which past / future encoders accept being transformed.

        Parameters
        ----------
        params
            Dict from parameter `add_encoders` (kwargs) used at model creation. Relevant parameters are:
            * params={'transformer': Scaler()}
        """

        if not params:
            return None, [], []

        transformer = params.get(TRANSFORMER_KEYS[0], None)
        if transformer is None:
            return None, [], []

        raise_if_not(
            isinstance(transformer, VALID_TRANSFORMER_DTYPES),
            f"Encountered `{TRANSFORMER_KEYS[0]}` of invalid type `{type(transformer)}` "
            f"in `add_encoders` at model creation. Transformer must be an instance of "
            f"`{VALID_TRANSFORMER_DTYPES}`.",
            logger,
        )

        transform_past_mask = [
            transform
            for enc in self.past_encoders
            for transform in enc.accept_transformer
        ]
        transform_future_mask = [
            transform
            for enc in self.future_encoders
            for transform in enc.accept_transformer
        ]
        return transformer, transform_past_mask, transform_future_mask
