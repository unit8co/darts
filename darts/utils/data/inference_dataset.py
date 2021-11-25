"""
Inference Dataset
-----------------
"""

from typing import Union, Sequence, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ...timeseries import TimeSeries
from ...logging import raise_if_not
from .utils import _get_matching_index, CovariateType
from .encoders import SequenceEncoder
from .encoder_base import (PastCovariateIndexGenerator,
                           FutureCovariateIndexGenerator)

SampleIndexType = Tuple[int, int, int, int, int, int]


class InferenceDataset(ABC, Dataset):
    def __init__(self):
        """
        Abstract class for all darts torch inference dataset.

        It can be used as models' inputs, to obtain simple forecasts on each `TimeSeries`
        (using covariates if specified).

        The first elements of the tuples it contains are numpy arrays (which will be translated to torch tensors
        by the torch DataLoader). The last elements of the tuples are the (past) target TimeSeries, which is
        needed in order to properly construct the time axis of the forecast series.
        """

        self.lazy_encoders: Optional[SequenceEncoder] = None

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @staticmethod
    def _generate_index(past_target: TimeSeries,
                        main_cov_type: CovariateType,
                        input_chunk_length: int,
                        output_chunk_length: int,
                        n: int):
        """returns tuple of (past_start, past_end, future_start, future_end)"""

        raise_if_not(main_cov_type in [CovariateType.NONE, CovariateType.PAST, CovariateType.FUTURE],
                     '`main_cov_type` must be one of `(CovariateType.NONE, CovariateType.PAST, CovariateType.FUTURE)`')

        if main_cov_type is CovariateType.NONE:
            return past_target.time_index[0], past_target.time_index[1], None, None

        if main_cov_type is CovariateType.PAST:
            index_generator = PastCovariateIndexGenerator(input_chunk_length=input_chunk_length,
                                                          output_chunk_length=output_chunk_length)
        else:  # CovariateType.FUTURE
            index_generator = FutureCovariateIndexGenerator(input_chunk_length=input_chunk_length,
                                                            output_chunk_length=output_chunk_length)

        index = index_generator.generate_inference_series(n=n, target=past_target, covariate=None)
        if len(index) == input_chunk_length:
            return past_target.time_index[0], past_target.time_index[-1], None, None
        else:
            index = index[input_chunk_length:]
            return past_target.time_index[0], past_target.time_index[-1], index[0], index[-1]

    def _generate_covariates(self,
                             n: int,
                             target: TimeSeries,
                             covariate: Optional[TimeSeries],
                             cov_type: CovariateType = CovariateType.NONE):

        raise_if_not(cov_type in [CovariateType.PAST, CovariateType.HISTORIC_FUTURE, CovariateType.FUTURE],
                     '`cov_type` must be one of `(CovariateType.PAST, CovariateType.HISTORIC_FUTURE, '
                     'CovariateType.FUTURE)` ')

        if cov_type is CovariateType.PAST:
            covariate, _ = self.lazy_encoders.encode_inference(n=n,
                                                               target=target,
                                                               past_covariate=covariate,
                                                               future_covariate=None,
                                                               encode_future=False)
        else:
            _, covariate = self.lazy_encoders.encode_inference(n=n,
                                                               target=target,
                                                               past_covariate=None,
                                                               future_covariate=covariate,
                                                               encode_past=False)
        if covariate is None:
            return None, None
        else:
            cov_vals = covariate[0].values(copy=False)
            return cov_vals[: self.lazy_encoders.input_chunk_length], cov_vals[ self.lazy_encoders.input_chunk_length:]


class PastCovariatesInferenceDataset(InferenceDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 n: int = 1,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1,
                 covariate_type: CovariateType = CovariateType.PAST,
                 lazy_encoders: Optional[SequenceEncoder] = None):
        """
        Contains (past_target, past_covariates, future_past_covariates).

        "future_past_covariates" are past covariates that happen to be also known in the future - those
        are needed for forecasting with n > output_chunk_length by any model relying on past covariates.

        For this reason, when n > output_chunk_length, this dataset will also emmit the "future past_covariates".

        Parameters
        ----------
        target_series
            The target series that are to be predicted into the future.
        covariates
            Optionally, some past-observed covariates that are used for predictions. This argument is required
            if the model was trained with past-observed covariates.
        n
            Forecast horizon: The number of time steps to predict after the end of the target series.
        input_chunk_length
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emmits in output.
        lazy_encoders
            Optionally, an instance of `SequenceEncoder`. If data is loaded lazily and lazy_encoders are given,
            covariates are generated at sample loading time.
        """

        super().__init__()

        self.target_series = [target_series] if isinstance(target_series, TimeSeries) else target_series
        self.covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates

        self.covariate_type = covariate_type

        self.n = n
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        self.lazy_encoders = lazy_encoders

        raise_if_not((covariates is None or len(self.target_series) == len(self.covariates)),
                     'The number of target series must be equal to the number of covariates.')

    def __len__(self):
        return len(self.target_series)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], TimeSeries]:
        target_series = self.target_series[idx]
        covariate_series = None if self.covariates is None else self.covariates[idx]

        past_target = target_series[-self.input_chunk_length:]
        past_target_vals = past_target.values(copy=False)

        do_encoding = self.lazy_encoders is not None and (self.lazy_encoders.future_encoders or
                                                          self.lazy_encoders.past_encoders)

        cov_past, cov_future = None, None
        if self.covariates is not None or do_encoding:
            main_cov_type = CovariateType.NONE
            if self.covariates is not None:
                main_cov_type = CovariateType.PAST if self.covariate_type is CovariateType.PAST \
                    else CovariateType.FUTURE

            past_start, past_end, future_start, future_end = \
                self._generate_index(past_target=past_target,
                                     main_cov_type=main_cov_type,
                                     input_chunk_length=self.input_chunk_length,
                                     output_chunk_length=self.output_chunk_length,
                                     n=self.n)

            covariate = covariate_series[past_start:future_end] if self.covariates else covariate_series

            if not do_encoding:
                cov_vals = covariate.values(copy=False)
                cov_past, cov_future = cov_vals[:self.input_chunk_length], cov_vals[self.input_chunk_length:]
            else:
                cov_past, cov_future = self._generate_covariates(n=self.n,
                                                                 target=past_target,
                                                                 covariate=covariate,
                                                                 cov_type=self.covariate_type)
        return past_target_vals, cov_past, cov_future, target_series


class FutureCovariatesInferenceDataset(InferenceDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 n: int = 1,
                 input_chunk_length: int = 12,
                 covariate_type: CovariateType = CovariateType.FUTURE,
                 lazy_encoders: Optional[SequenceEncoder] = None):
        """
        Contains (past_target, future_covariates) tuples

        Parameters
        ----------
        target_series
            The target series that are to be predicted into the future.
        covariates
            Optionally, some future-known covariates that are used for predictions. This argument is required
            if the model was trained with future-known covariates.
        n
            Forecast horizon: The number of time steps to predict after the end of the target series.
        input_chunk_length
            The length of the target series the model takes as input.
        lazy_encoders
            Optionally, an instance of `SequenceEncoder`. If data is loaded lazily and lazy_encoders are given,
            covariates are generated at sample loading time.
        """
        super().__init__()
        self.target_series = [target_series] if isinstance(target_series, TimeSeries) else target_series
        self.covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates
        self.covariate_type = covariate_type

        self.n = n
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = n

        self.lazy_encoders = lazy_encoders

        raise_if_not((covariates is None or len(self.target_series) == len(self.covariates)),
                     'The number of target series must be equal to the number of covariates.')

    def __len__(self):
        return len(self.target_series)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], TimeSeries]:
        target_series = self.target_series[idx]
        covariate_series = None if self.covariates is None else self.covariates[idx]

        past_target = target_series[-self.input_chunk_length:]
        past_target_vals = past_target.values(copy=False)

        cov_past, cov_future = None, None

        do_encoding = self.lazy_encoders is not None and (self.lazy_encoders.future_encoders or
                                                          self.lazy_encoders.past_encoders)

        if self.covariates is not None or do_encoding:
            main_cov_type = CovariateType.NONE
            if self.covariates is not None:
                main_cov_type = CovariateType.PAST if self.covariate_type is CovariateType.PAST \
                    else CovariateType.FUTURE

            past_start, past_end, future_start, future_end = \
                self._generate_index(past_target=past_target,
                                     main_cov_type=main_cov_type,
                                     input_chunk_length=self.input_chunk_length,
                                     output_chunk_length=self.output_chunk_length,
                                     n=self.n)

            covariate = covariate_series[past_start:future_end] if self.covariates else covariate_series

            if not do_encoding:
                cov_vals = covariate.values(copy=False)
                cov_past, cov_future = cov_vals[:self.input_chunk_length], cov_vals[self.input_chunk_length:]
            else:
                cov_past, cov_future = self._generate_covariates(n=self.n,
                                                                 target=past_target,
                                                                 covariate=covariate,
                                                                 cov_type=self.covariate_type)

        return past_target_vals, cov_future, target_series


class DualCovariatesInferenceDataset(InferenceDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 n: int = 1,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1,
                 lazy_encoders: Optional[SequenceEncoder] = None):
        """
        Contains (past_target, historic_future_covariates, future_covariates) tuples.

        Parameters
        ----------
        target_series
            The target series that are to be predicted into the future.
        covariates
            Optionally, some future-known covariates that are used for predictions. This argument is required
            if the model was trained with future-known covariates.
        n
            Forecast horizon: The number of time steps to predict after the end of the target series.
        input_chunk_length
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emmits in output.
        lazy_encoders
            Optionally, an instance of `SequenceEncoder`. If data is loaded lazily and lazy_encoders are given,
            covariates are generated at sample loading time.
        """
        super().__init__()

        # This dataset is in charge of serving historic future covariates
        self.ds_past = PastCovariatesInferenceDataset(target_series=target_series,
                                                      covariates=covariates,
                                                      n=n,
                                                      input_chunk_length=input_chunk_length,
                                                      output_chunk_length=output_chunk_length,
                                                      covariate_type=CovariateType.HISTORIC_FUTURE,
                                                      lazy_encoders=lazy_encoders)

        # This dataset is in charge of serving future covariates
        self.ds_future = FutureCovariatesInferenceDataset(target_series=target_series,
                                                          covariates=covariates,
                                                          n=n,
                                                          input_chunk_length=input_chunk_length,
                                                          covariate_type=CovariateType.FUTURE,
                                                          lazy_encoders=lazy_encoders)

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], TimeSeries]:
        past_target, historic_future_covs, _, ts_target = self.ds_past[idx]
        _, future_covs, _ = self.ds_future[idx]
        return past_target, historic_future_covs, future_covs, ts_target


class MixedCovariatesInferenceDataset(InferenceDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 n: int = 1,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1,
                 lazy_encoders: Optional[SequenceEncoder] = None):
        """
        Contains (past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates)
        tuples. "future_past_covariates" are past covariates that happen to be also known in the future - those
        are needed for forecasting with n > output_chunk_length by any model relying on past covariates.

        Parameters
        ----------
        target_series
            The target series that are to be predicted into the future.
        past_covariates
            Optionally, some past-observed covariates that are used for predictions. This argument is required
            if the model was trained with past-observed covariates.
        future_covariates
            Optionally, some future-known covariates that are used for predictions. This argument is required
            if the model was trained with future-known covariates.
        n
            Forecast horizon: The number of time steps to predict after the end of the target series.
        input_chunk_length
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emmits in output.
        lazy_encoders
            Optionally, an instance of `SequenceEncoder`. If data is loaded lazily and lazy_encoders are given,
            covariates are generated at sample loading time.
        """
        super().__init__()

        # This dataset is in charge of serving past covariates
        self.ds_past = PastCovariatesInferenceDataset(target_series=target_series,
                                                      covariates=past_covariates,
                                                      n=n,
                                                      input_chunk_length=input_chunk_length,
                                                      output_chunk_length=output_chunk_length,
                                                      covariate_type=CovariateType.PAST,
                                                      lazy_encoders=lazy_encoders)

        # This dataset is in charge of serving historic and future future covariates
        self.ds_future = DualCovariatesInferenceDataset(target_series=target_series,
                                                        covariates=future_covariates,
                                                        n=n,
                                                        input_chunk_length=input_chunk_length,
                                                        output_chunk_length=output_chunk_length,
                                                        lazy_encoders=lazy_encoders)

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
                                        Optional[np.ndarray], Optional[np.ndarray], TimeSeries]:

        past_target, past_covs, future_past_covs, ts_target = self.ds_past[idx]
        _, historic_future_covs, future_covs, _ = self.ds_future[idx]
        return past_target, past_covs, historic_future_covs, future_covs, future_past_covs, ts_target


class SplitCovariatesInferenceDataset(InferenceDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 n: int = 1,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1,
                 lazy_encoders: Optional[SequenceEncoder] = None):
        """
        Contains (past_target, past_covariates, future_covariates, future_past_covariates) tuples.
        "future_past_covariates" are past covariates that happen to be also known in the future - those
        are needed for forecasting with n > output_chunk_length by any model relying on past covariates.

        Parameters
        ----------
        target_series
            The target series that are to be predicted into the future.
        past_covariates
            Optionally, some past-observed covariates that are used for predictions. This argument is required
            if the model was trained with past-observed covariates.
        future_covariates
            Optionally, some future-known covariates that are used for predictions. This argument is required
            if the model was trained with future-known covariates.
        n
            Forecast horizon: The number of time steps to predict after the end of the target series.
        input_chunk_length
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emmits in output.
        lazy_encoders
            Optionally, an instance of `SequenceEncoder`. If data is loaded lazily and lazy_encoders are given,
            covariates are generated at sample loading time.
        """
        super().__init__()

        # This dataset is in charge of serving past covariates
        self.ds_past = PastCovariatesInferenceDataset(target_series=target_series,
                                                      covariates=past_covariates,
                                                      n=n,
                                                      input_chunk_length=input_chunk_length,
                                                      output_chunk_length=output_chunk_length,
                                                      covariate_type=CovariateType.PAST,
                                                      lazy_encoders=lazy_encoders)

        # This dataset is in charge of serving future covariates
        self.ds_future = FutureCovariatesInferenceDataset(target_series=target_series,
                                                          covariates=future_covariates,
                                                          n=n,
                                                          input_chunk_length=input_chunk_length,
                                                          covariate_type=CovariateType.FUTURE,
                                                          lazy_encoders=lazy_encoders)

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
                                        Optional[np.ndarray], TimeSeries]:

        past_target, past_covs, future_past_covs, ts_target = self.ds_past[idx]
        _, future_covs, _ = self.ds_future[idx]
        return past_target, past_covs, future_covs, future_past_covs, ts_target
