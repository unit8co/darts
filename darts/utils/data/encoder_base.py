"""
Encoder Base Classes
------------------------------
"""

import pandas as pd

from abc import ABC, abstractmethod
from typing import Union, Optional

from darts import TimeSeries
from darts.logging import get_logger
from darts.utils.timeseries_generation import _generate_index


SupportedIndexes = Union[pd.DatetimeIndex, pd.Int64Index, pd.RangeIndex]
logger = get_logger(__name__)


class CovariateIndexGenerator(ABC):
    def __init__(self, input_chunk_length, output_chunk_length):
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

    @abstractmethod
    def generate_train_series(self,
                              target: TimeSeries,
                              covariate: Optional[TimeSeries] = None) -> SupportedIndexes:
        """
        Implement a method that extracts the required covariate index for training.

        Parameters
        ----------
        target
            the target TimeSeries used during training
        covariate
            optionally, the future covariates used for training
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
            the forecast horizon
        target
            the target TimeSeries used during training or passed to prediction as `series`
        covariate
            optionally, the future covariates used for prediction
        """
        pass


class PastCovariateIndexGenerator(CovariateIndexGenerator):
    """generates index for past covariates on train and inference datasets"""
    def generate_train_series(self,
                              target: TimeSeries,
                              covariate: Optional[TimeSeries] = None) -> SupportedIndexes:

        super(PastCovariateIndexGenerator, self).generate_train_series(target, covariate)
        return covariate.time_index if covariate is not None else target.time_index

    def generate_inference_series(self,
                                  n: int,
                                  target: TimeSeries,
                                  covariate: Optional[TimeSeries] = None) -> SupportedIndexes:
        """For prediction (`n` is given) with past covariates we have to distinguish between two cases:
        1)  if past covariates are given, we can use them as reference
        2)  if past covariates are missing, we need to generate a time index that starts `input_chunk_length`
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
    """generates index for future covariates on train and inference datasets."""
    def generate_train_series(self,
                              target: TimeSeries,
                              covariate: Optional[TimeSeries] = None) -> SupportedIndexes:
        """For training (when `n` is `None`) we can simply use the future covariates (if available) or target as
        reference to extract the time index.
        """

        super(FutureCovariateIndexGenerator, self).generate_train_series(target, covariate)

        return covariate.time_index if covariate is not None else target.time_index

    def generate_inference_series(self,
                                  n: int,
                                  target: TimeSeries,
                                  covariate: Optional[TimeSeries] = None) -> SupportedIndexes:
        """For prediction (`n` is given) with future covariates we have to distinguish between two cases:
        1)  if future covariates are given, we can use them as reference
        2)  if future covariates are missing, we need to generate a time index that starts `input_chunk_length`
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
