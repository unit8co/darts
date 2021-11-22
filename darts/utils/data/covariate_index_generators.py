import pandas as pd

from abc import ABC, abstractmethod
from typing import Union, Optional

from darts import TimeSeries
from darts.utils.timeseries_generation import _generate_index
from darts.logging import raise_if_not, get_logger, raise_log, raise_if

logger = get_logger(__name__)


SupportedIndexes = Union[pd.DatetimeIndex, pd.Int64Index, pd.RangeIndex]


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
    """generates index for past covariate"""
    def generate_train_series(self,
                              target: TimeSeries,
                              covariate: Optional[TimeSeries] = None) -> SupportedIndexes:
        # TODO -> we can probably summarize this in CovariateIndexGenerator as I think it's the same for
        #  future & past covs
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
    """generates index for future covariate."""
    def generate_train_series(self,
                              target: TimeSeries,
                              covariate: Optional[TimeSeries] = None) -> SupportedIndexes:
        """For training (when `n` is `None`) we can simply use the future covariates (if available) or target as
        reference to extract the time index.
        """

        # TODO -> we can probably summarize this in CovariateIndexGenerator as I think it's the same for
        #  future & past covs
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


