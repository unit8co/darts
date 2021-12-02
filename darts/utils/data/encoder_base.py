"""
Encoder Base Classes
------------------------------
"""

import pandas as pd

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Union, Optional, Tuple, Sequence, List

from itertools import compress
from darts import TimeSeries
from darts.logging import get_logger
from darts.utils.timeseries_generation import _generate_index
from darts.dataprocessing.transformers import FittableDataTransformer


SupportedIndexes = Union[pd.DatetimeIndex, pd.Int64Index, pd.RangeIndex]
EncoderOutputType = Optional[Union[Sequence[TimeSeries], List[TimeSeries]]]
logger = get_logger(__name__)


class ReferenceIndexType(Enum):
    PREDICTION = auto()
    START = auto()
    NONE = auto()


class CovariateIndexGenerator(ABC):
    def __init__(self,
                 input_chunk_length: int,
                 output_chunk_length: int,
                 reference_index_type: ReferenceIndexType = ReferenceIndexType.NONE):
        """
        Parameters
        ----------
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        reference_index
            If a reference index should be saved, set `reference_index` to one of `(ReferenceIndexType.PREDICTION,
            ReferenceIndexType.START)`
        """
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.reference_index_type = reference_index_type
        self.reference_index: Optional[Tuple[int, Union[pd.Timestamp, int]]] = None

    @abstractmethod
    def generate_train_series(self,
                              target: TimeSeries,
                              covariate: Optional[TimeSeries] = None) -> SupportedIndexes:
        """
        Implement a method that extracts the required covariate index for training.

        Parameters
        ----------
        target
            The target TimeSeries used during training
        covariate
            Optionally, the future covariates used for training
        """
        pass

    @abstractmethod
    def generate_inference_series(self,
                                  n: int,
                                  target: TimeSeries,
                                  covariate: Optional[TimeSeries] = None) -> SupportedIndexes:
        """
        Implement a method that extracts the required covariate index for prediction.

        Parameters
        ----------
        n
            The forecast horizon
        target
            The target TimeSeries used during training or passed to prediction as `series`
        covariate
            Optionally, the future covariates used for prediction
        """
        pass


class PastCovariateIndexGenerator(CovariateIndexGenerator):
    """Generates index for past covariates on train and inference datasets"""
    def generate_train_series(self,
                              target: TimeSeries,
                              covariate: Optional[TimeSeries] = None) -> SupportedIndexes:

        super(PastCovariateIndexGenerator, self).generate_train_series(target, covariate)

        # save a reference index if specified
        if self.reference_index_type is not ReferenceIndexType.NONE and self.reference_index is None:
            if self.reference_index_type is ReferenceIndexType.PREDICTION:
                self.reference_index = (len(target) - 1, target.end_time())
            else:  # save the time step before start of target series
                self.reference_index = (-1, target.start_time() - target.freq)

        return covariate.time_index if covariate is not None else target.time_index

    def generate_inference_series(self,
                                  n: int,
                                  target: TimeSeries,
                                  covariate: Optional[TimeSeries] = None) -> SupportedIndexes:
        """For prediction (`n` is given) with past covariates we have to distinguish between two cases:
        1)  If past covariates are given, we can use them as reference
        2)  If past covariates are missing, we need to generate a time index that starts `input_chunk_length`
            before the end of `target` and ends `max(0, n - output_chunk_length)` after the end of `target`
        """

        super(PastCovariateIndexGenerator, self).generate_inference_series(n, target, covariate)
        if covariate is not None:
            return covariate.time_index
        else:
            return _generate_index(start=target.end_time() - target.freq * (self.input_chunk_length - 1),
                                   length=self.input_chunk_length + max(0, n - self.output_chunk_length),
                                   freq=target.freq)


class FutureCovariateIndexGenerator(CovariateIndexGenerator):
    """Generates index for future covariates on train and inference datasets."""
    def generate_train_series(self,
                              target: TimeSeries,
                              covariate: Optional[TimeSeries] = None) -> SupportedIndexes:
        """For training (when `n` is `None`) we can simply use the future covariates (if available) or target as
        reference to extract the time index.
        """

        super(FutureCovariateIndexGenerator, self).generate_train_series(target, covariate)

        # save a reference index if specified
        if self.reference_index_type is not ReferenceIndexType.NONE and self.reference_index is None:
            if self.reference_index_type is ReferenceIndexType.PREDICTION:
                self.reference_index = (len(target) - 1, target.end_time())
            else:  # save the time step before start of target series
                self.reference_index = (-1, target.start_time() - target.freq)

        return covariate.time_index if covariate is not None else target.time_index

    def generate_inference_series(self,
                                  n: int,
                                  target: TimeSeries,
                                  covariate: Optional[TimeSeries] = None) -> SupportedIndexes:
        """For prediction (`n` is given) with future covariates we have to distinguish between two cases:
        1)  If future covariates are given, we can use them as reference
        2)  If future covariates are missing, we need to generate a time index that starts `input_chunk_length`
            before the end of `target` and ends `max(n, output_chunk_length)` after the end of `target`
        """
        super(FutureCovariateIndexGenerator, self).generate_inference_series(n, target, covariate)

        if covariate is not None:
            return covariate.time_index
        else:
            return _generate_index(start=target.end_time() - target.freq * (self.input_chunk_length - 1),
                                   length=self.input_chunk_length + max(n, self.output_chunk_length),
                                   freq=target.freq)


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
            The target TimeSeries used during training or passed to prediction as `series`
        covariate
            Optionally, the future covariates used for prediction
        merge_covariate
            Whether or not to merge the encoded TimeSeries with `covariate`.
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
            The forecast horizon
        target
            The target TimeSeries used during training or passed to prediction as `series`
        covariate
            Optionally, the future covariates used for prediction
        merge_covariate
            Whether or not to merge the encoded TimeSeries with `covariate`.

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
            The encoded TimeSeries either from `encode_train()` or `encode_inference()`
        covariate
            Optionally, the future covariates used for prediction
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
            The index generated from `self.index_generator` for either the train or inference dataset.
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
            The target TimeSeries used during training or passed to prediction as `series`
        covariate
            Optionally, the covariate used for training: past covariate if `self.index_generator` is instance of
            `PastCovariateIndexGenerator`, future covariate if `self.index_generator` is instance of
            `FutureCovariateIndexGenerator`
        merge_covariate
            Whether or not to merge the encoded TimeSeries with `covariate`.
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
            The forecast horizon
        target
            The target TimeSeries used during training or passed to prediction as `series`
        covariate
            Optionally, the covariate used for prediction: past covariate if `self.index_generator` is instance of
            `PastCovariateIndexGenerator`, future covariate if `self.index_generator` is instance of
            `FutureCovariateIndexGenerator`
        merge_covariate
            Whether or not to merge the encoded TimeSeries with `covariate`.
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

    @property
    @abstractmethod
    def accept_transformer(self) -> bool:
        """Whether or not the SingleEncoder sub class accepts to be transformed."""
        pass


class SequenceEncoderTransformer:
    """`SequenceEncoderTransformer` applies transformation to the non-transformed past and future covariate output of
    `SequenceEncoder.encode_train()` and `SequenceEncoder.encode_inference()`. The transformer is fitted
    when `transform()` is called for the first time. This ensures proper transformation of train, validation and
    inference dataset covariates."""

    def __init__(self, transformer: FittableDataTransformer, transform_past_mask: List, transform_future_mask: List):
        """
        Parameters
        ----------
        transformer
            A `FittableDataTransformer` object with a `fit_transform()` and `transform()` method.
        transform_past_mask
            A boolean 1-D mask specifying which of the input covariates to :meth:`transform()
            <SequenceEncoderTransformer.transform()>` must be transformed.
        transform_future_mask
            A boolean 1-D mask specifying which of the input future covariates to :meth:`transform()
            <SequenceEncoderTransformer.transform()>` must be transformed.
        """
        self.transformer = transformer
        self.transform_past_mask = transform_past_mask
        self.transform_future_mask = transform_future_mask
        self._fit_called = False

    def transform(self,
                  past_covariate: EncoderOutputType,
                  future_covariate: EncoderOutputType) -> Tuple[EncoderOutputType, EncoderOutputType]:
        """This method applies transformation to the non-transformed past and future covariate output of
        `SequenceEncoder.encode_train()` and `SequenceEncoder.encode_inference()`. The transformer is fitted when
        `transform()` is called for the first time. This ensures proper transformation of train, validation and
        inference dataset covariates.

        Parameters
        ----------
        past_covariate
            The non-transformed past covariate output of `SequenceEncoder`. A sequence or list containing user-defined
            past covariates that were supplied to `fit()` or `predict()` and optional generated encoded past covariates
            from `SequenceEncoder`.
        future_covariate
            The non-transformed future covariate output of `SequenceEncoder`. A sequence or list containing user-
            defined future covariates that were supplied to `fit()` or `predict()` and optional generated encoded
            future covariates from `SequenceEncoder`.
        """
        if not self.fit_called:
            self._update_masks(past_covariate, future_covariate)
            transformed = self.transformer.fit_transform(self._extract(past_covariate, future_covariate))
            self._fit_called = True
        else:
            transformed = self.transformer.transform(self._extract(past_covariate, future_covariate))
        return self._insert(past_covariate, future_covariate, transformed)

    def _extract(self,
                 past_covariate: EncoderOutputType,
                 future_covariate: EncoderOutputType) -> List[TimeSeries]:
        """Extract covariates that need to be transformed.
        """
        extracted = []
        if self.transform_past_mask:
            extracted += list(compress(past_covariate, self.transform_past_mask))
        if self.transform_future_mask:
            extracted += list(compress(future_covariate, self.transform_future_mask))
        return extracted

    def _insert(self,
                past_covariate: EncoderOutputType,
                future_covariate: EncoderOutputType,
                covariate_transformed: List[TimeSeries]) -> Tuple[EncoderOutputType, EncoderOutputType]:
        """Insert transformed covariates into input covariates.
        """
        get_idx = 0
        for idx, insert_past in enumerate(self.transform_past_mask):
            if insert_past:
                past_covariate[idx] = covariate_transformed[get_idx]
                get_idx += 1
        for idx, insert_future in enumerate(self.transform_future_mask):
            if insert_future:
                future_covariate[idx] = covariate_transformed[get_idx]
                get_idx += 1
        return past_covariate, future_covariate

    def _update_masks(self,
                      past_covariate: EncoderOutputType,
                      future_covariate: EncoderOutputType) -> None:
        """A mismatch between lengths of {x}_covariate and transform_{}_mask means that user passed additional
        {}_covariate to `model.fit()`. User-defined covariates are always located at the beginning of {}_covariate.
        In case of a mismatch, the masks are updated so that user-defined covariates are not transformed.
        """
        if past_covariate is not None and len(past_covariate) != len(self.transform_past_mask):
            n_diff = (len(past_covariate) - len(self.transform_past_mask))
            self.transform_past_mask = [False] * n_diff + self.transform_past_mask
        if future_covariate is not None and len(future_covariate) != len(self.transform_future_mask):
            n_diff = (len(future_covariate) - len(self.transform_future_mask))
            self.transform_future_mask = [False] * n_diff + self.transform_future_mask

    @property
    def fit_called(self):
        """Return whether or not the transformer has been fitted."""
        return self._fit_called


