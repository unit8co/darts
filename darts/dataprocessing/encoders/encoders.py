"""
Time Axes Encoders
------------------

Encoders can generate past and/or future covariates series by encoding the index of a TimeSeries `series`.
Each encoder class has an `encode_train()` and `encode_inference()` to generate the encodings for training and
inference.

The encoders extract the index either from the target series or optional additional past/future covariates.
If additional covariates are supplied to `encode_train()` or `encode_inference()`, the time index of those
covariates are used for the encodings. This means that the input covariates must meet the same model-specific
requirements as without encoders.

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
                                                         covariates=optional_past_covariates)
            past_covariates_inf = encoder.encode_inference(n=12,
                                                           target=target,
                                                           covariates=optional_past_covariates)

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
        :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>`.

SingleEncoder
-------------

The SingleEncoders from {X}{SingleEncoder} are:

*   `DatetimeAttributeEncoder`
        Adds scalar pd.DatatimeIndex attribute information derived from `series.time_index`.
        Requires `series` to have a pd.DatetimeIndex.

        attribute
            An attribute of `pd.DatetimeIndex`: see all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
*   `CyclicTemporalEncoder`
        Adds cyclic pd.DatetimeIndex attribute information deriveed from `series.time_index`.
        Adds 2 columns, corresponding to sin and cos encodings, to uniquely describe the underlying attribute.
        Requires `series` to have a pd.DatetimeIndex.

        attribute
            An attribute of `pd.DatetimeIndex` that follows a cyclic pattern. One of ('month', 'day', 'weekday',
            'dayofweek', 'day_of_week', 'hour', 'minute', 'second', 'microsecond', 'nanosecond', 'quarter',
            'dayofyear', 'day_of_year', 'week', 'weekofyear', 'week_of_year').
*   `IntegerIndexEncoder`
        Adds the relative index positions as integer values (positions) derived from `series` time index.
        `series` can either have a pd.DatetimeIndex or an integer index.

        attribute
            Currently, only 'relative' is supported.
            'relative' will generate position values relative to the forecasting/prediction point. Values range
            from -inf to inf where 0 is set at the forecasting point.
*   `CallableIndexEncoder`
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
*   inner keys: covariates type

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
        'position': {'past': ['relative'], 'future': ['relative']},
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
from darts.dataprocessing.encoders.encoder_base import (
    CovariatesIndexGenerator,
    Encoder,
    FutureCovariatesIndexGenerator,
    PastCovariatesIndexGenerator,
    SequentialEncoderTransformer,
    SingleEncoder,
    SupportedIndex,
)
from darts.dataprocessing.transformers import FittableDataTransformer
from darts.logging import get_logger, raise_if, raise_if_not
from darts.timeseries import DIMS
from darts.utils.timeseries_generation import (
    datetime_attribute_timeseries,
    generate_index,
)
from darts.utils.utils import seq2series, series2seq

SupportedTimeSeries = Union[TimeSeries, Sequence[TimeSeries]]
logger = get_logger(__name__)

ENCODER_KEYS = ["cyclic", "datetime_attribute", "position", "custom"]
FUTURE = "future"
PAST = "past"
VALID_TIME_PARAMS = [FUTURE, PAST]
VALID_ENCODER_DTYPES = (str, Sequence)

TRANSFORMER_KEYS = ["transformer"]
VALID_TRANSFORMER_DTYPES = FittableDataTransformer
INTEGER_INDEX_ATTRIBUTES = ["relative"]


class CyclicTemporalEncoder(SingleEncoder):
    """`CyclicTemporalEncoder`: Cyclic encoding of time series datetime attributes."""

    def __init__(self, index_generator: CovariatesIndexGenerator, attribute: str):
        """
        Cyclic index encoding for `TimeSeries` that have a time index of type `pd.DatetimeIndex`.

        Parameters
        ----------
        index_generator
            An instance of `CovariatesIndexGenerator` with methods `generate_train_idx()` and
            `generate_inference_idx()`. Used to generate the index for encoders.
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

    def _encode(
        self, index: SupportedIndex, target_end: pd.Timestamp, dtype: np.dtype
    ) -> TimeSeries:
        """applies cyclic encoding from `datetime_attribute_timeseries()` to `self.attribute` of `index`."""
        super()._encode(index, target_end, dtype)
        return datetime_attribute_timeseries(
            index,
            attribute=self.attribute,
            cyclic=True,
            dtype=dtype,
            with_columns=[
                self.base_component_name + self.attribute + "_sin",
                self.base_component_name + self.attribute + "_cos",
            ],
        )

    @property
    def accept_transformer(self) -> List[bool]:
        """`CyclicTemporalEncoder` should not be transformed. Returns two elements for sine and cosine waves."""
        return [False, False]

    @property
    def requires_fit(self) -> bool:
        return False

    @property
    def base_component_name(self) -> str:
        return super().base_component_name + "_cyc_"


class PastCyclicEncoder(CyclicTemporalEncoder):
    """`CyclicEncoder`: Cyclic encoding of past covariates datetime attributes."""

    def __init__(
        self,
        attribute: str,
        input_chunk_length: Optional[int] = None,
        output_chunk_length: Optional[int] = None,
        lags_covariates: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        attribute
            The attribute of the underlying pd.DatetimeIndex from  for which to apply cyclic encoding.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        input_chunk_length
            Optionally, the number of input target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `input_chunk_length` from :class:`TorchForecastingModel`, or to the absolute
            minimum target lag value `abs(min(lags))` for :class:`RegressionModel`.
        output_chunk_length
            Optionally, the number of output target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `output_chunk_length` from both :class:`TorchForecastingModel`, and
            :class:`RegressionModel`.
        lags_covariates
            Optionally, a list of integers representing the past covariate lags. Accepts integer lag values <= -1.
            Only required for :class:`RegressionModel`.
            Corresponds to the lag values from parameter `lags_past_covariates` of :class:`RegressionModel`.
        """
        super().__init__(
            index_generator=PastCovariatesIndexGenerator(
                input_chunk_length,
                output_chunk_length,
                lags_covariates=lags_covariates,
            ),
            attribute=attribute,
        )


class FutureCyclicEncoder(CyclicTemporalEncoder):
    """`CyclicEncoder`: Cyclic encoding of future covariates datetime attributes."""

    def __init__(
        self,
        attribute: str,
        input_chunk_length: Optional[int] = None,
        output_chunk_length: Optional[int] = None,
        lags_covariates: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        attribute
            The attribute of the underlying pd.DatetimeIndex from  for which to apply cyclic encoding.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        input_chunk_length
            Optionally, the number of input target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `input_chunk_length` from :class:`TorchForecastingModel`, or to the absolute
            minimum target lag value `abs(min(lags))` for :class:`RegressionModel`.
        output_chunk_length
            Optionally, the number of output target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `output_chunk_length` from both :class:`TorchForecastingModel`, and
            :class:`RegressionModel`.
        lags_covariates
            Optionally, a list of integers representing the future covariate lags. Accepts all integer values.
            Only required for :class:`RegressionModel`.
            Corresponds to the lag values from parameter `lags_future_covariates` from :class:`RegressionModel`.
        """
        super().__init__(
            index_generator=FutureCovariatesIndexGenerator(
                input_chunk_length,
                output_chunk_length,
                lags_covariates=lags_covariates,
            ),
            attribute=attribute,
        )


class DatetimeAttributeEncoder(SingleEncoder):
    """`DatetimeAttributeEncoder`: Adds pd.DatatimeIndex attribute information derived from the index as scalars.
    Requires the underlying TimeSeries to have a pd.DatetimeIndex
    """

    def __init__(self, index_generator: CovariatesIndexGenerator, attribute: str):
        """
        Parameters
        ----------
        index_generator
            An instance of `CovariatesIndexGenerator` with methods `generate_train_idx()` and
            `generate_inference_idx()`. Used to generate the index for encoders.
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

    def _encode(
        self, index: SupportedIndex, target_end: pd.Timestamp, dtype: np.dtype
    ) -> TimeSeries:
        """Applies cyclic encoding from `datetime_attribute_timeseries()` to `self.attribute` of `index`."""
        super()._encode(index, target_end, dtype)
        return datetime_attribute_timeseries(
            index,
            attribute=self.attribute,
            dtype=dtype,
            with_columns=self.base_component_name + self.attribute,
        )

    @property
    def accept_transformer(self) -> List[bool]:
        """`DatetimeAttributeEncoder` accepts transformations"""
        return [True]

    @property
    def requires_fit(self) -> bool:
        return False

    @property
    def base_component_name(self) -> str:
        return super().base_component_name + "_dta_"


class PastDatetimeAttributeEncoder(DatetimeAttributeEncoder):
    """Datetime attribute encoder for past covariates."""

    def __init__(
        self,
        attribute: str,
        input_chunk_length: Optional[int] = None,
        output_chunk_length: Optional[int] = None,
        lags_covariates: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        attribute
            The attribute of the underlying pd.DatetimeIndex from  for which to add scalar information.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        input_chunk_length
            Optionally, the number of input target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `input_chunk_length` from :class:`TorchForecastingModel`, or to the absolute
            minimum target lag value `abs(min(lags))` for :class:`RegressionModel`.
        output_chunk_length
            Optionally, the number of output target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `output_chunk_length` from both :class:`TorchForecastingModel`, and
            :class:`RegressionModel`.
        lags_covariates
            Optionally, a list of integers representing the past covariate lags. Accepts integer lag values <= -1.
            Only required for :class:`RegressionModel`.
            Corresponds to the lag values from parameter `lags_past_covariates` of :class:`RegressionModel`.
        """
        super().__init__(
            index_generator=PastCovariatesIndexGenerator(
                input_chunk_length,
                output_chunk_length,
                lags_covariates=lags_covariates,
            ),
            attribute=attribute,
        )


class FutureDatetimeAttributeEncoder(DatetimeAttributeEncoder):
    """Datetime attribute encoder for future covariates."""

    def __init__(
        self,
        attribute: str,
        input_chunk_length: Optional[int] = None,
        output_chunk_length: Optional[int] = None,
        lags_covariates: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        attribute
            The attribute of the underlying pd.DatetimeIndex from  for which to add scalar information.
            Must be an attribute of `pd.DatetimeIndex`, or `week` / `weekofyear` / `week_of_year` - e.g. "month",
            "weekday", "day", "hour", "minute", "second". See all available attributes in
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html#pandas.DatetimeIndex.
            For more information, check out :meth:`datetime_attribute_timeseries()
            <darts.utils.timeseries_generation.datetime_attribute_timeseries>`
        input_chunk_length
            Optionally, the number of input target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `input_chunk_length` from :class:`TorchForecastingModel`, or to the absolute
            minimum target lag value `abs(min(lags))` for :class:`RegressionModel`.
        output_chunk_length
            Optionally, the number of output target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `output_chunk_length` from both :class:`TorchForecastingModel`, and
            :class:`RegressionModel`.
        lags_covariates
            Optionally, a list of integers representing the future covariate lags. Accepts all integer values.
            Only required for :class:`RegressionModel`.
            Corresponds to the lag values from parameter `lags_future_covariates` from :class:`RegressionModel`.
        """
        super().__init__(
            index_generator=FutureCovariatesIndexGenerator(
                input_chunk_length,
                output_chunk_length,
                lags_covariates=lags_covariates,
            ),
            attribute=attribute,
        )


class IntegerIndexEncoder(SingleEncoder):
    """IntegerIndexEncoder: Adds integer index value (position) derived from the underlying TimeSeries' time index
    for past and future covariates.
    """

    def __init__(self, index_generator: CovariatesIndexGenerator, attribute: str):
        """
        Parameters
        ----------
        index_generator
            An instance of `CovariatesIndexGenerator` with methods `generate_train_idx()` and
            `generate_inference_idx()`. Used to generate the index for encoders.
        attribute
            Currently only 'relative' is supported. The generated encoded values will range from (-inf, inf) and the
            target series end time will be used as a reference to evaluate the relative index positions.
        """
        raise_if_not(
            isinstance(attribute, str) and attribute in INTEGER_INDEX_ATTRIBUTES,
            f"Encountered invalid encoder argument `{attribute}` for encoder `position`. "
            f'Attribute must be `"relative"`.',
            logger,
        )
        super().__init__(index_generator)

        self.attribute = attribute

    def _encode(
        self, index: SupportedIndex, target_end: pd.Timestamp, dtype: np.dtype
    ) -> TimeSeries:
        """Applies cyclic encoding from `datetime_attribute_timeseries()` to `self.attribute` of `index`.
        For attribute=='relative', the reference point/index is the prediction/forecast index of the target series.
        """
        super()._encode(index, target_end, dtype)

        idx_larger_end = (index <= target_end).sum()
        freq = index.freq if isinstance(index, pd.DatetimeIndex) else index.step
        if idx_larger_end:
            idx_larger_end -= 1
        if index[0] > target_end:
            idx_diff = (
                len(generate_index(start=target_end, end=index[0], freq=freq)) - 1
            )
        elif index[-1] < target_end:
            idx_diff = (
                -len(generate_index(start=index[-1], end=target_end, freq=freq)) + 1
            )
        else:
            idx_diff = 0
        return TimeSeries.from_times_and_values(
            times=index,
            values=np.arange(
                start=idx_diff - idx_larger_end,
                stop=idx_diff - idx_larger_end + len(index),
            ),
            columns=[self.base_component_name + self.attribute],
        ).astype(np.dtype(dtype))

    @property
    def accept_transformer(self) -> List[bool]:
        """`IntegerIndexEncoder` accepts transformations. Note that transforming 'relative' `IntegerIndexEncoder`
        will return the absolute position (in the transformed space)."""
        return [True]

    @property
    def requires_fit(self) -> bool:
        # requires fitting to get the reference index from `IntegerIndexEncoder.index_generator` for inference
        return True

    @property
    def base_component_name(self) -> str:
        return super().base_component_name + "_pos_"


class PastIntegerIndexEncoder(IntegerIndexEncoder):
    """`IntegerIndexEncoder`: Adds integer index value (position) for past covariates derived from the underlying
    TimeSeries' time index.
    """

    def __init__(
        self,
        attribute: str,
        input_chunk_length: Optional[int] = None,
        output_chunk_length: Optional[int] = None,
        lags_covariates: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        attribute
            Currently only 'relative' is supported. The generated encoded values will range from (-inf, inf) and the
            target series end time will be used as a reference to evaluate the relative index positions.
        input_chunk_length
            Optionally, the number of input target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `input_chunk_length` from :class:`TorchForecastingModel`, or to the absolute
            minimum target lag value `abs(min(lags))` for :class:`RegressionModel`.
        output_chunk_length
            Optionally, the number of output target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `output_chunk_length` from both :class:`TorchForecastingModel`, and
            :class:`RegressionModel`.
        lags_covariates
            Optionally, a list of integers representing the past covariate lags. Accepts integer lag values <= -1.
            Only required for :class:`RegressionModel`.
            Corresponds to the lag values from parameter `lags_past_covariates` of :class:`RegressionModel`.
        """
        super().__init__(
            index_generator=PastCovariatesIndexGenerator(
                input_chunk_length,
                output_chunk_length,
                lags_covariates=lags_covariates,
            ),
            attribute=attribute,
        )


class FutureIntegerIndexEncoder(IntegerIndexEncoder):
    """`IntegerIndexEncoder`: Adds integer index value (position) for future covariates derived from the underlying
    TimeSeries' time index.
    """

    def __init__(
        self,
        attribute: str,
        input_chunk_length: Optional[int] = None,
        output_chunk_length: Optional[int] = None,
        lags_covariates: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        attribute
            Currently only 'relative' is supported. The generated encoded values will range from (-inf, inf) and the
            target series end time will be used as a reference to evaluate the relative index positions.
        input_chunk_length
            Optionally, the number of input target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `input_chunk_length` from :class:`TorchForecastingModel`, or to the absolute
            minimum target lag value `abs(min(lags))` for :class:`RegressionModel`.
        output_chunk_length
            Optionally, the number of output target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `output_chunk_length` from both :class:`TorchForecastingModel`, and
            :class:`RegressionModel`.
        lags_covariates
            Optionally, a list of integers representing the future covariate lags. Accepts all integer values.
            Only required for :class:`RegressionModel`.
            Corresponds to the lag values from parameter `lags_future_covariates` from :class:`RegressionModel`.
        """
        super().__init__(
            index_generator=FutureCovariatesIndexGenerator(
                input_chunk_length,
                output_chunk_length,
                lags_covariates=lags_covariates,
            ),
            attribute=attribute,
        )


class CallableIndexEncoder(SingleEncoder):
    """`CallableIndexEncoder`: Applies a user-defined callable to encode the underlying index for past and future
    covariates.
    """

    def __init__(self, index_generator: CovariatesIndexGenerator, attribute: Callable):
        """
        Parameters
        ----------
        index_generator
            An instance of `CovariatesIndexGenerator` with methods `generate_train_idx()` and
            `generate_inference_idx()`. Used to generate the index for encoders.
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

    def _encode(
        self, index: SupportedIndex, target_end: pd.Timestamp, dtype: np.dtype
    ) -> TimeSeries:
        """Apply the user-defined callable to encode the index"""
        super()._encode(index, target_end, dtype)

        return TimeSeries.from_times_and_values(
            times=index,
            values=self.attribute(index),
            columns=[self.base_component_name + "custom"],
        ).astype(np.dtype(dtype))

    @property
    def accept_transformer(self) -> List[bool]:
        """`CallableIndexEncoder` accepts transformations."""
        return [True]

    @property
    def requires_fit(self) -> bool:
        return False

    @property
    def base_component_name(self) -> str:
        return super().base_component_name + "_cus_"


class PastCallableIndexEncoder(CallableIndexEncoder):
    """`IntegerIndexEncoder`: Adds integer index value (position) for past covariates derived from the underlying
    TimeSeries' time index.
    """

    def __init__(
        self,
        attribute: Callable,
        input_chunk_length: Optional[int] = None,
        output_chunk_length: Optional[int] = None,
        lags_covariates: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        attribute
            A callable that takes an index `index` of type `(pd.DatetimeIndex, pd.RangeIndex)` as input
            and returns a np.ndarray of shape `(len(index),)`.
            An example for a correct `attribute` for `index` of type pd.DatetimeIndex:
            ``attribute = lambda index: (index.year - 1950) / 50``. And for pd.RangeIndex:
            ``attribute = lambda index: (index - 1950) / 50``
        input_chunk_length
            Optionally, the number of input target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `input_chunk_length` from :class:`TorchForecastingModel`, or to the absolute
            minimum target lag value `abs(min(lags))` for :class:`RegressionModel`.
        output_chunk_length
            Optionally, the number of output target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `output_chunk_length` from both :class:`TorchForecastingModel`, and
            :class:`RegressionModel`.
        lags_covariates
            Optionally, a list of integers representing the past covariate lags. Accepts integer lag values <= -1.
            Only required for :class:`RegressionModel`.
            Corresponds to the lag values from parameter `lags_past_covariates` of :class:`RegressionModel`.
        """
        super().__init__(
            index_generator=PastCovariatesIndexGenerator(
                input_chunk_length,
                output_chunk_length,
                lags_covariates=lags_covariates,
            ),
            attribute=attribute,
        )


class FutureCallableIndexEncoder(CallableIndexEncoder):
    """`IntegerIndexEncoder`: Adds integer index value (position) for future covariates derived from the underlying
    TimeSeries' time index.
    """

    def __init__(
        self,
        attribute: Callable,
        input_chunk_length: Optional[int] = None,
        output_chunk_length: Optional[int] = None,
        lags_covariates: Optional[List[int]] = None,
    ):
        """
        Parameters
        ----------
        attribute
            A callable that takes an index `index` of type `(pd.DatetimeIndex, pd.RangeIndex)` as input
            and returns a np.ndarray of shape `(len(index),)`.
            An example for a correct `attribute` for `index` of type pd.DatetimeIndex:
            ``attribute = lambda index: (index.year - 1950) / 50``. And for pd.RangeIndex:
            ``attribute = lambda index: (index - 1950) / 50``
        input_chunk_length
            Optionally, the number of input target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `input_chunk_length` from :class:`TorchForecastingModel`, or to the absolute
            minimum target lag value `abs(min(lags))` for :class:`RegressionModel`.
        output_chunk_length
            Optionally, the number of output target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `output_chunk_length` from both :class:`TorchForecastingModel`, and
            :class:`RegressionModel`.
        lags_covariates
            Optionally, a list of integers representing the future covariate lags. Accepts all integer values.
            Only required for :class:`RegressionModel`.
            Corresponds to the lag values from parameter `lags_future_covariates` from :class:`RegressionModel`.
        """
        super().__init__(
            index_generator=FutureCovariatesIndexGenerator(
                input_chunk_length,
                output_chunk_length,
                lags_covariates=lags_covariates,
            ),
            attribute=attribute,
        )


class SequentialEncoder(Encoder):
    """A `SequentialEncoder` object can store and control multiple past and future covariates encoders at once.
    It provides the same functionality as single encoders (`encode_train()` and `encode_inference()`).
    """

    def __init__(
        self,
        add_encoders: Dict,
        input_chunk_length: Optional[int] = None,
        output_chunk_length: Optional[int] = None,
        lags_past_covariates: Optional[List[int]] = None,
        lags_future_covariates: Optional[List[int]] = None,
        takes_past_covariates: bool = False,
        takes_future_covariates: bool = False,
    ) -> None:
        """
        SequentialEncoder automatically creates encoder objects from parameter `add_encoders`. `add_encoders` can also
        be set directly in all of Darts' `ForecastingModels`. This will automatically set up a
        :class:`SequentialEncoder` tailored to the settings of the underlying forecasting model.

        The `add_encoders` dict must follow this convention:
            `{encoder keyword: {temporal keyword: List[attributes]}, ..., transformer keyword: transformer object}`
        Supported encoder keywords:
            `'cyclic'` for cyclic temporal encoder. See the docs
            :meth:`CyclicTemporalEncoder <darts.dataprocessing.encoders.CyclicTemporalEncoder>`;
            `'datetime_attribute'` for adding scalar information of pd.DatetimeIndex attribute. See the docs
            :meth:`DatetimeAttributeEncoder <darts.dataprocessing.encoders.DatetimeAttributeEncoder>`
            `'position'` for integer index position encoder. See the docs
            :meth:`IntegerIndexEncoder <darts.dataprocessing.encoders.IntegerIndexEncoder>`;
            `'custom'` for encoding index with custom callables (functions). See the docs
            :meth:`CallableIndexEncoder <darts.dataprocessing.encoders.CallableIndexEncoder>`;
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
                    'position': {'past': ['relative'], 'future': ['relative']},
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
            A dictionary with the encoder settings.
        input_chunk_length
            Optionally, the number of input target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `input_chunk_length` from :class:`TorchForecastingModel`, or to the absolute
            minimum target lag value `abs(min(lags))` for :class:`RegressionModel`.
        output_chunk_length
            Optionally, the number of output target time steps per chunk. Only required for
            :class:`TorchForecastingModel`, and :class:`RegressionModel`.
            Corresponds to parameter `output_chunk_length` from both :class:`TorchForecastingModel`, and
            :class:`RegressionModel`.
        lags_past_covariates
            Optionally, a list of integers representing the past covariate lags. Accepts integer lag values <= -1.
            Only required for :class:`RegressionModel`.
            Corresponds to the lag values from parameter `lags_past_covariates` of :class:`RegressionModel`.
        lags_future_covariates
            Optionally, a list of integers representing the future covariate lags. Accepts all integer values.
            Only required for :class:`RegressionModel`.
            Corresponds to the lag values from parameter `lags_future_covariates` from :class:`RegressionModel`.
        takes_past_covariates
            Whether to encode/generate past covariates.
        takes_future_covariates
            Whether to encode/generate future covariates.
        """
        super().__init__()
        self.params = add_encoders
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.encoding_available = False
        self.takes_past_covariates = takes_past_covariates
        self.takes_future_covariates = takes_future_covariates
        self.lags_past_covariates = lags_past_covariates
        self.lags_future_covariates = lags_future_covariates

        # encoders
        self._past_encoders: List[SingleEncoder] = []
        self._past_components: pd.Index = pd.Index([])
        self._future_encoders: List[SingleEncoder] = []
        self._future_components: pd.Index = pd.Index([])

        # transformer
        self._past_transformer: Optional[SequentialEncoderTransformer] = None
        self._future_transformer: Optional[SequentialEncoderTransformer] = None

        # setup encoders and transformer
        self._setup_encoders(self.params)
        self._setup_transformer(self.params)

    def encode_train(
        self,
        target: SupportedTimeSeries,
        past_covariates: Optional[SupportedTimeSeries] = None,
        future_covariates: Optional[SupportedTimeSeries] = None,
        encode_past: bool = True,
        encode_future: bool = True,
    ) -> Tuple[
        Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]
    ]:
        """Returns encoded index for all past and/or future covariates for training.
        Which covariates are generated depends on the parameters used at model creation.

        Parameters
        ----------
        target
            The target TimeSeries used during training or passed to prediction as `series`
        past_covariates
            Optionally, the past covariates used for training.
        future_covariates
            Optionally, the future covariates used for training.
        encode_past
            Whether to apply encoders for past covariates
        encode_future
            Whether to apply encoders for future covariates
        Returns
        -------
        Tuple[past_covariates, future_covariates]
            The past_covariates and/or future_covariates for training including the encodings.
            If input {x}_covariates is None and no {x}_encoders are given, will return `None`
            for the {x}_covariates.
        Raises
        ------
        Warning
            If model was created with `add_encoders` and there is suspicion of lazy loading.
            The encodings/covariates are generated eagerly before starting training for all individual targets and
            loaded into memory. Depending on the size of target data, this can create memory
            issues. In case this applies, consider setting `add_encoders=None` at model
            creation and build your encodings covariates manually for lazy loading.
        """
        if not self.fit_called:
            if not isinstance(target, (TimeSeries, list)):
                logger.warning(
                    "Fitting was called with `add_encoders` and suspicion of lazy loading. "
                    "The encodings/covariates are generated pre-train for all individual targets and "
                    "loaded into memory. Depending on the size of your data, this can create memory issues. "
                    "In case this applies, consider setting `add_encoders=None` at model creation."
                )

            self._fit_called = True
        past_covariates, future_covariates = self._launch_encoder(
            target=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            n=None,
            encode_past=encode_past,
            encode_future=encode_future,
        )
        self._fit_called = True
        return past_covariates, future_covariates

    def encode_inference(
        self,
        n: int,
        target: SupportedTimeSeries,
        past_covariates: Optional[SupportedTimeSeries] = None,
        future_covariates: Optional[SupportedTimeSeries] = None,
        encode_past: bool = True,
        encode_future: bool = True,
    ) -> Tuple[
        Union[TimeSeries, Sequence[TimeSeries]], Union[TimeSeries, Sequence[TimeSeries]]
    ]:
        """Returns encoded index for all past and/or future covariates for inference/prediction.
        Which covariates are generated depends on the parameters used at model creation.

        Parameters
        ----------
        n
            The forecast horizon
        target
            The target TimeSeries used during training or passed to prediction as `series`
        past_covariates
            Optionally, the past covariates used for training.
        future_covariates
            Optionally, the future covariates used for training.
        encode_past
            Whether to apply encoders for past covariates
        encode_future
            Whether to apply encoders for future covariates

        Returns
        -------
        Tuple[past_covariates, future_covariates]
            The past_covariates and/or future_covariates for prediction/inference including the encodings.
            If input {x}_covariates is None and no {x}_encoders are given, will return `None`
            for the {x}_covariates.
        """
        raise_if(
            not self.fit_called and self.requires_fit,
            f"`{self.__class__.__name__}` contains encoders or transformers which must be trained before inference. "
            "Call method `encode_train()` before `encode_inference()`.",
            logger=logger,
        )
        return self._launch_encoder(
            target=target,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            n=n,
            encode_past=encode_past,
            encode_future=encode_future,
        )

    def _launch_encoder(
        self,
        target: Sequence[TimeSeries],
        past_covariates: SupportedTimeSeries,
        future_covariates: SupportedTimeSeries,
        n: Optional[int] = None,
        encode_past: bool = True,
        encode_future: bool = True,
    ) -> Tuple[Sequence[TimeSeries], Sequence[TimeSeries]]:
        """Launches the encode sequence for past covariates and future covariates for either training or
        inference/prediction.

        If `n` is not `None` it is a prediction, otherwise it is training.
        """
        if not self.encoding_available:
            return past_covariates, future_covariates

        # guarantee that all inputs are either a sequence of TimeSeries or None
        single_series = isinstance(target, TimeSeries)
        target = series2seq(target)
        past_covariates = series2seq(past_covariates)
        future_covariates = series2seq(future_covariates)

        # generate past covariates encodings
        if self.past_encoders and encode_past:
            past_covariates = self._encode_sequence(
                encoders=self.past_encoders,
                transformer=self.past_transformer,
                target=target,
                covariates=past_covariates,
                covariates_type=PAST,
                n=n,
            )

        # generate future covariates encodings
        if self.future_encoders and encode_future:
            future_covariates = self._encode_sequence(
                encoders=self.future_encoders,
                transformer=self.future_transformer,
                target=target,
                covariates=future_covariates,
                covariates_type=FUTURE,
                n=n,
            )

        # convert covariates back to single series if single target was used as input
        if single_series:
            past_covariates = seq2series(past_covariates)
            future_covariates = seq2series(future_covariates)
        return past_covariates, future_covariates

    def _encode_sequence(
        self,
        encoders: Sequence[SingleEncoder],
        transformer: Optional[SequentialEncoderTransformer],
        target: Sequence[TimeSeries],
        covariates: Optional[SupportedTimeSeries],
        covariates_type: str,
        n: Optional[int] = None,
    ) -> List[TimeSeries]:
        """Sequentially encodes the index of all input target/covariates TimeSeries

        If `n` is not `None` it is a prediction and method `encoder.encode_inference()` is called.
        Otherwise, it is a training case and `encoder.encode_train()` is called.
        """
        encode_method = "encode_train" if n is None else "encode_inference"

        encoded_sequence = []
        if covariates is None:
            covariates = [None] * len(target)
        else:
            covariates = (
                [covariates] if isinstance(covariates, TimeSeries) else covariates
            )

        for ts, covs in zip(target, covariates):
            # drop encoder components if they are in input covariates
            covs = self._drop_encoded_components(
                covariates=covs,
                components=getattr(self, f"{covariates_type}_components"),
            )
            encoded = concatenate(
                [
                    getattr(enc, encode_method)(
                        target=ts, covariates=covs, merge_covariates=False, n=n
                    )
                    for enc in encoders
                ],
                axis=DIMS[1],
            )
            encoded_sequence.append(
                self._merge_covariates(encoded=encoded, covariates=covs)
            )

        if transformer is not None:
            encoded_sequence = transformer.transform(encoded_sequence)

        # store encoded past/future component names if they were not saved before
        if getattr(self, f"{covariates_type}_components").empty:
            components = encoded_sequence[0].components
            if covariates is not None and covariates[0] is not None:
                components = components[~components.isin(covariates[0].components)]
            setattr(self, f"_{covariates_type}_components", components)

        return encoded_sequence

    @property
    def past_encoders(self) -> List[SingleEncoder]:
        """Returns the past covariates encoders"""
        return self._past_encoders

    @property
    def future_encoders(self) -> List[SingleEncoder]:
        """Returns the future covariates encoders"""
        return self._future_encoders

    @property
    def encoders(self) -> Tuple[List[SingleEncoder], List[SingleEncoder]]:
        """Returns a tuple of (past covariates encoders, future covariates encoders)"""
        return self.past_encoders, self.future_encoders

    @property
    def past_components(self) -> pd.Index:
        """Returns the past covariates component names generated by `SequentialEncoder.past_encoders`.
        Only available after calling `SequentialEncoder.encode_train()`
        """
        return self._past_components

    @property
    def future_components(self) -> pd.Index:
        """Returns the future covariates component names generated by `SequentialEncoder.future_encoders`.
        Only available after calling `SequentialEncoder.encode_train()`
        """
        return self._future_components

    @property
    def components(self) -> Tuple[pd.Index, pd.Index]:
        """Returns the covariates component names generated by `SequentialEncoder.past_encoders` and
        `SequentialEncoder.past_encoders`. A tuple of (past encoded components, future encoded components).
        Only available after calling `SequentialEncoder.encode_train()`
        """
        return self.past_components, self.future_components

    @property
    def past_transformer(self) -> SequentialEncoderTransformer:
        """Returns the past transformer object"""
        return self._past_transformer

    @property
    def future_transformer(self) -> SequentialEncoderTransformer:
        """Returns the future transformer object"""
        return self._future_transformer

    def transformers(
        self,
    ) -> Tuple[SequentialEncoderTransformer, SequentialEncoderTransformer]:
        """Returns a tuple of (past transformer, future transformer)."""
        return self.past_transformer, self.future_transformer

    @property
    def encoder_map(self) -> Dict:
        """Mapping between encoder identifier string (from parameters at model creations) and the corresponding
        future or past covariates encoder"""
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
                attribute=attr,
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                lags_covariates=self.lags_past_covariates,
            )
            for enc_id, attr in past_encoders
        ]
        self._future_encoders = [
            self.encoder_map[enc_id](
                attribute=attr,
                input_chunk_length=self.input_chunk_length,
                output_chunk_length=self.output_chunk_length,
                lags_covariates=self.lags_future_covariates,
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
            1) if the outermost key is other than (`past`, `future`)
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

    @property
    def requires_fit(self) -> bool:
        return any(
            [enc.requires_fit for cov_enc in self.encoders for enc in cov_enc]
        ) or any([tf is not None for tf in self.transformers()])
