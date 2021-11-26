"""
Inference Dataset
-----------------
"""

import numpy as np

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Union, Sequence, Optional, Tuple

from ...timeseries import TimeSeries
from ...logging import raise_if_not
from .utils import CovariateType
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

        raise_if_not(main_cov_type in [CovariateType.PAST, CovariateType.FUTURE],
                     '`main_cov_type` must be one of `(CovariateType.PAST, CovariateType.FUTURE)`')

        if main_cov_type is CovariateType.PAST:
            index_generator = PastCovariateIndexGenerator(input_chunk_length=input_chunk_length,
                                                          output_chunk_length=output_chunk_length)
        else:  # CovariateType.FUTURE
            index_generator = FutureCovariateIndexGenerator(input_chunk_length=input_chunk_length,
                                                            output_chunk_length=output_chunk_length)

        index = index_generator.generate_inference_series(n=n, target=past_target, covariate=None)

        if len(index) != input_chunk_length:
            index = index[input_chunk_length:]

        if input_chunk_length != 0:  # for regular models
            return past_target.time_index[0], past_target.time_index[-1], index[0], index[-1]
        else:  # for regression ensemble models
            return index[0], index[0], index[0], index[-1]

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

        cov_vals = covariate[0].values(copy=False)
        if self.lazy_encoders.input_chunk_length != 0:  # regular models
            return cov_vals[:self.lazy_encoders.input_chunk_length], cov_vals[self.lazy_encoders.input_chunk_length:]
        else:  # regression ensemble models
            return cov_vals, cov_vals


class GenericInferenceDataset(InferenceDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 n: int = 1,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1,
                 covariate_type: CovariateType = CovariateType.PAST,
                 lazy_encoders: Optional[SequenceEncoder] = None):
        """
        Contains (past_target, past_covariates | historic_future_covariates, future_past_covariates | future_covariate).

        "future_past_covariates" are past covariates that happen to be also known in the future - those
        are needed for forecasting with n > output_chunk_length by any model relying on past covariates.
        For this reason, when n > output_chunk_length, this dataset will also emmit the "future past_covariates".
        
        "historic_future_covariates" are historic future covariates that are given for the input_chunk in the past.

        Parameters
        ----------
        target_series
            The target series that are to be predicted into the future.
        covariates
            Optionally, one or a sequence of `TimeSeries` containing either past or future covariates. If covariates
            were used during training, the same type of cavariates must be supplied at prediction.
        n
            Forecast horizon: The number of time steps to predict after the end of the target series.
        input_chunk_length
            The length of the target series the model takes as input.
        output_chunk_length
            The length of the target series the model emits in output.
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

        raise_if_not(len(target_series) >= self.input_chunk_length,
                     f"All input series must have length >= `input_chunk_length` ({self.input_chunk_length}).")

        covariate_series = None if self.covariates is None else self.covariates[idx]

        past_target = target_series[-self.input_chunk_length:]
        past_target_vals = past_target.values(copy=False)

        covariate = None
        if covariate_series is not None:
            main_cov_type = CovariateType.PAST if self.covariate_type is CovariateType.PAST \
                else CovariateType.FUTURE

            past_start, past_end, future_start, future_end = \
                self._generate_index(past_target=past_target,
                                     main_cov_type=main_cov_type,
                                     input_chunk_length=self.input_chunk_length,
                                     output_chunk_length=self.output_chunk_length,
                                     n=self.n)
            case_start = future_start if self.covariate_type is CovariateType.FUTURE else past_start
            raise_if_not(
                covariate_series.start_time() <= case_start,
                f"For the given forecasting case, the provided {main_cov_type.value} covariates at dataset index "
                f"`{idx}` do not extend far enough into the past. The {main_cov_type.value} covariates must start at "
                f"time step `{case_start}`, whereas now they start at time step `{covariate_series.start_time()}`.")
            raise_if_not(
                covariate_series.end_time() >= future_end,
                f"For the given forecasting horizon `n={self.n}`, the provided {main_cov_type.value} covariates "
                f"at dataset index `{idx}` do not extend far enough into the future. As `"
                f"{'n > output_chunk_length' if self.n > self.output_chunk_length else 'n <= output_chunk_length'}"
                f"` the {main_cov_type.value} covariates must end at time step `{future_end}`, "
                f"whereas now they end at time step `{covariate_series.end_time()}`.")

            # extract the index position (index) from time_index value
            cov_start = covariate_series.time_index.get_loc(past_start)
            cov_end = covariate_series.time_index.get_loc(future_end) + 1
            covariate = covariate_series[cov_start:cov_end]

        do_encoding = self.lazy_encoders is not None and self.lazy_encoders.encoding_available
        if do_encoding:  # optionally, use encoders to generate (additional) covariates
            cov_past, cov_future = self._generate_covariates(n=self.n,
                                                             target=past_target,
                                                             covariate=covariate,
                                                             cov_type=self.covariate_type)
        elif covariate_series is not None:  # extract past and future covariate parts without encoders
            cov_vals = covariate.values(copy=False)
            if self.input_chunk_length != 0:  # regular models
                cov_past, cov_future = cov_vals[:self.input_chunk_length], cov_vals[self.input_chunk_length:]
            else:  # regression ensemble models
                cov_past, cov_future = cov_vals, cov_vals
        else:
            cov_past, cov_future = None, None

        cov_past = cov_past if cov_past is not None and len(cov_past) > 0 else None
        cov_future = cov_future if cov_future is not None and len(cov_future) > 0 else None

        return past_target_vals, cov_past, cov_future, target_series


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

        self.ds = GenericInferenceDataset(target_series=target_series,
                                          covariates=covariates,
                                          n=n,
                                          input_chunk_length=input_chunk_length,
                                          output_chunk_length=output_chunk_length,
                                          covariate_type=covariate_type,
                                          lazy_encoders=lazy_encoders)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], TimeSeries]:
        return self.ds[idx]


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

        self.ds = GenericInferenceDataset(target_series=target_series,
                                          covariates=covariates,
                                          n=n,
                                          input_chunk_length=input_chunk_length,
                                          output_chunk_length=n,
                                          covariate_type=covariate_type,
                                          lazy_encoders=lazy_encoders)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], TimeSeries]:
        past_target_vals, _, cov_future, target_series = self.ds[idx]
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
