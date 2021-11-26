"""
Inference Dataset
-----------------
"""

from typing import Union, Sequence, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import Dataset

from ...timeseries import TimeSeries
from ...logging import raise_if_not
from .utils import _get_matching_index


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

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass


class PastCovariatesInferenceDataset(InferenceDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 n: int = 1,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1):
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
        """

        super().__init__()

        self.target_series = [target_series] if isinstance(target_series, TimeSeries) else target_series
        self.covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates
        self.n = n
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length

        raise_if_not((covariates is None or len(self.target_series) == len(self.covariates)),
                     'The number of target series must be equal to the number of covariates.')

    def __len__(self):
        return len(self.target_series)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], TimeSeries]:
        target_series = self.target_series[idx]
        covariate_series = None if self.covariates is None else self.covariates[idx]
        target_vals = target_series.values(copy=False)

        raise_if_not(len(target_series) >= self.input_chunk_length,
                     'All input series must have length >= `input_chunk_length` ({}).'.format(
                         self.input_chunk_length))

        tgt_past_vals = target_vals[-self.input_chunk_length:]

        cov_past = cov_future = None
        if covariate_series is not None:
            cov_vals = covariate_series.values(copy=False)
            start_past_cov_idx = _get_matching_index(target_series, covariate_series, self.input_chunk_length)

            # because -0 doesn't do what we want as an indexing bound
            if -start_past_cov_idx+self.input_chunk_length == 0:
                cov_past = cov_vals[-start_past_cov_idx:]
            else:
                cov_past = cov_vals[-start_past_cov_idx:-start_past_cov_idx+self.input_chunk_length]

            raise_if_not(len(cov_past) == self.input_chunk_length,
                         'The dataset contains past covariates that do not have a sufficient time span to obtain '
                         'a slice of length input_chunk_length matching the future target')

            # check whether future covariates are also required
            if self.n > self.output_chunk_length:
                # check that enough "past" covariates are available into the future
                # We need `n - output_chunk_length`
                nr_timestamps_needed = self.n - self.output_chunk_length
                last_req_ts = target_series.end_time() + nr_timestamps_needed * target_series.freq

                raise_if_not(covariate_series.end_time() >= last_req_ts,
                             "When forecasting future values for a horizon `n > output_chunk_length` with models "
                             "requiring past covariates, the past covariates need to be provided for the next "
                             "`(n - output_chunk_length)` steps in advance. For the dataset's {}-th sample, the last "
                             "covariate timestamp is {} whereas it should be {}.".format(
                                 idx, covariate_series.end_time(), last_req_ts
                             ))

                # because -0 doesn't do what we want as an indexing bound
                if -start_past_cov_idx+self.input_chunk_length+nr_timestamps_needed == 0:
                    cov_future = cov_vals[-start_past_cov_idx+self.input_chunk_length:]
                else:
                    cov_future = cov_vals[-start_past_cov_idx+self.input_chunk_length:
                                          -start_past_cov_idx+self.input_chunk_length+nr_timestamps_needed]

        return tgt_past_vals, cov_past, cov_future, target_series


class FutureCovariatesInferenceDataset(InferenceDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 n: int = 1,
                 input_chunk_length: int = 12):
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
        """
        super().__init__()
        self.target_series = [target_series] if isinstance(target_series, TimeSeries) else target_series
        self.covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates
        self.n = n
        self.input_chunk_length = input_chunk_length

        raise_if_not((covariates is None or len(self.target_series) == len(self.covariates)),
                     'The number of target series must be equal to the number of covariates.')

    def __len__(self):
        return len(self.target_series)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], TimeSeries]:
        target_series = self.target_series[idx]
        covariate_series = None if self.covariates is None else self.covariates[idx]
        target_vals = target_series.values(copy=False)

        raise_if_not(len(target_series) >= self.input_chunk_length,
                     'All input series must have length >= `input_chunk_length` ({}).'.format(
                         self.input_chunk_length))

        tgt_past_vals = target_vals[-self.input_chunk_length:]

        cov_future = None
        if covariate_series is not None:
            # check that we have at least n timestamps of future covariates available
            last_req_ts = target_series.end_time() + self.n * target_series.freq

            raise_if_not(covariate_series.end_time() >= last_req_ts,
                         "When forecasting future values for a horizon `n` with models requiring future covariates, "
                         "the future covariates need to be known `n` time steps in advance. "
                         "For the dataset's {}-th sample, the last covariate timestamp is {} whereas it "
                         "should be {}.".format(idx, covariate_series.end_time(), last_req_ts))

            cov_vals = covariate_series.values(copy=False)
            start_idx = _get_matching_index(target_series, covariate_series, 0)

            # because -0 doesn't do what we want as an indexing bound
            if -start_idx+self.n == 0:
                cov_future = cov_vals[-start_idx:]
            else:
                cov_future = cov_vals[-start_idx:-start_idx+self.n]

        return tgt_past_vals, cov_future, target_series


class DualCovariatesInferenceDataset(InferenceDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 n: int = 1,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1):
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
        """
        super().__init__()

        # This dataset is in charge of serving historic future covariates
        self.ds_past = PastCovariatesInferenceDataset(target_series=target_series,
                                                      covariates=covariates,
                                                      n=n,
                                                      input_chunk_length=input_chunk_length,
                                                      output_chunk_length=output_chunk_length)

        # This dataset is in charge of serving future covariates
        self.ds_future = FutureCovariatesInferenceDataset(target_series=target_series,
                                                          covariates=covariates,
                                                          n=n,
                                                          input_chunk_length=input_chunk_length)

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
                 output_chunk_length: int = 1):
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
        """
        super().__init__()

        # This dataset is in charge of serving past covariates
        self.ds_past = PastCovariatesInferenceDataset(target_series=target_series,
                                                      covariates=past_covariates,
                                                      n=n,
                                                      input_chunk_length=input_chunk_length,
                                                      output_chunk_length=output_chunk_length)

        # This dataset is in charge of serving historic and future future covariates
        self.ds_future = DualCovariatesInferenceDataset(target_series=target_series,
                                                        covariates=future_covariates,
                                                        n=n,
                                                        input_chunk_length=input_chunk_length,
                                                        output_chunk_length=output_chunk_length)

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
                 output_chunk_length: int = 1):
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
        """
        super().__init__()

        # This dataset is in charge of serving past covariates
        self.ds_past = PastCovariatesInferenceDataset(target_series=target_series,
                                                      covariates=past_covariates,
                                                      n=n,
                                                      input_chunk_length=input_chunk_length,
                                                      output_chunk_length=output_chunk_length)

        # This dataset is in charge of serving future covariates
        self.ds_future = FutureCovariatesInferenceDataset(target_series=target_series,
                                                          covariates=future_covariates,
                                                          n=n,
                                                          input_chunk_length=input_chunk_length)

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
                                        Optional[np.ndarray], TimeSeries]:

        past_target, past_covs, future_past_covs, ts_target = self.ds_past[idx]
        _, future_covs, _ = self.ds_future[idx]
        return past_target, past_covs, future_covs, future_past_covs, ts_target
