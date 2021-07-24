"""
Inference Dataset
-----------------
"""

from typing import Union, Sequence, Optional, Tuple
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ...timeseries import TimeSeries
from ...logging import raise_if_not, raise_if


class InferenceDataset(ABC, Dataset):
    def __init__(self):
        """
        Abstract class for all darts torch inference dataset.

        It can be used as models' inputs, to obtain simple forecasts on each `TimeSeries`
        (using covariates if specified).

        `TimeSeriesInferenceDataset` inherits from `Sequence`; meaning that the implementations have to
        provide the `__len__()` and `__getitem__()` methods.

        It contains `TimeSeries` (and not e.g. `np.ndarray`), because inference requires the time axes,
        and typically the performance penalty should be lower than for training datasets because there's no slicing.

        TODO for speed: return nd.arrays and rebuild the time axes of the forecasted series separately?
        """
        pass

    @abstractmethod
    def get_target(self, idx: int) -> TimeSeries:
        """
        This is needed in order to build the final forecast TimeSeries (we need the time axis).
        """
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

        raise_if_not((covariates is None or len(target_series) == len(covariates)),
                     'The number of target series must be equal to the number of covariates.')

    def __len__(self):
        return len(self.target_series)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        target_series = self.target_series[idx]
        covariate_series = None if self.covariates is None else self.covariates[idx]

        raise_if_not(len(target_series) >= self.input_chunk_length,
                     'All input series must have length >= `input_chunk_length` ({}).'.format(
                         self.input_chunk_length))

        tgt_past = target_series[-self.input_chunk_length:]

        cov_past = cov_future = None
        if covariate_series is not None:
            raise_if_not(covariate_series.freq == target_series.freq,
                         'The dataset contains some covariate series that do not have the same freq as the '
                         'target series ({}-th)'.format(idx))

            # past target and past covariates must come from the same time slice
            cov_past = covariate_series.slice_intersect(tgt_past)

            raise_if_not(len(cov_past) == self.input_chunk_length,
                         'The dataset contains some covariates that do not have a sufficient time span to obtain '
                         'a slice of length input_chunk_length matching the target')

            # check whether future covariates are also required
            if self.n > self.output_chunk_length:
                # check that enough "past" covariates are available into the future
                # We need `n - output_chunk_length`
                nr_timestamps_needed = self.n - self.output_chunk_length
                last_req_ts = target_series.end_time() + nr_timestamps_needed * target_series.freq

                raise_if_not(covariate_series.end_time() >= last_req_ts,
                             "When forecasting future values for a horizon n with models requiring past covariates, "
                             "the past covariates need to be known (n - output_chunk_length) in advance (needed "
                             "to produce forecasts when n > output_chunk_length). For the dataset's {}-th sample, "
                             "the last covariate timestamp is {} whereas it should be {}.".format(
                                 idx, covariate_series.end_time(), last_req_ts
                             ))

                cov_future = covariate_series[last_req_ts-(nr_timestamps_needed-1)*target_series.freq:last_req_ts+target_series.freq]

        # TODO: change and slice numpy views instead of TimeSeries in the logic above
        tgt_past_tsr = torch.from_numpy(tgt_past.values(copy=False))
        cov_past_tsr = torch.from_numpy(cov_past.values(copy=False))
        cov_future_tsr = torch.from_numpy(cov_future.values(copy=False))
        return tgt_past_tsr, cov_past_tsr, cov_future_tsr

    def get_target(self, idx: int) -> TimeSeries:
        return self.target_series[idx]


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

        raise_if_not((covariates is None or len(target_series) == len(covariates)),
                     'The number of target series must be equal to the number of covariates.')

    def __len__(self):
        return len(self.target_series)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Optional[Tensor]]:
        target_series = self.target_series[idx]
        covariate_series = None if self.covariates is None else self.covariates[idx]

        raise_if_not(len(target_series) >= self.input_chunk_length,
                     'All input series must have length >= `input_chunk_length` ({}).'.format(
                         self.input_chunk_length))

        tgt_past = target_series[-self.input_chunk_length:]

        cov_future = None
        if covariate_series is not None:
            raise_if_not(covariate_series.freq == target_series.freq,
                         'The dataset contains some covariate series that do not have the same freq as the '
                         'target series ({}-th)'.format(idx))

            # check that we have at least n timestamps of future covariates available
            last_req_ts = target_series.end_time() + self.n * target_series.freq

            raise_if_not(covariate_series.end_time() >= last_req_ts,
                         "When forecasting future values for a horizon n with models requiring future covariates, "
                         "the future covariates need to be known n time steps in advance. "
                         "For the dataset's {}-th sample, the last covariate timestamp is {} whereas it "
                         "should be {}.".format(idx, covariate_series.end_time(), last_req_ts))

            cov_future = covariate_series[last_req_ts - (self.n - 1) * target_series.freq:last_req_ts + target_series.freq]

        # TODO: slice numpy views instead of TimeSeries...
        return torch.from_numpy(tgt_past.values(copy=False)), torch.from_numpy(cov_future.values(copy=False))

    def get_target(self, idx: int) -> TimeSeries:
        return self.target_series[idx]


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
        self.ds_past = PastCovariatesInferenceDataset(target_series=target_series,
                                                      covariates=covariates,
                                                      n=n,
                                                      input_chunk_length=input_chunk_length,
                                                      output_chunk_length=output_chunk_length)

        self.ds_future = FutureCovariatesInferenceDataset(target_series=target_series,
                                                          covariates=covariates,
                                                          n=n,
                                                          input_chunk_length=input_chunk_length)

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(self, idx):
        past_target, historic_future_covs, _ = self.ds_past[idx]
        _, future_covs = self.ds_future[idx]
        return past_target, historic_future_covs, future_covs

    def get_target(self, idx: int) -> TimeSeries:
        return self.ds_past.get_target(idx)


class MixedCovariatesInferenceDataset(InferenceDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 n: int = 1,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1):
        """
        Contains (past_target, past_covariates, historic_future_covariates, future_covariates, future_past_covariates) tuples.
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
        self.ds_past = PastCovariatesInferenceDataset(target_series=target_series,
                                                      covariates=past_covariates,
                                                      n=n,
                                                      input_chunk_length=input_chunk_length,
                                                      output_chunk_length=output_chunk_length)

        self.ds_future = DualCovariatesInferenceDataset(target_series=target_series,
                                                        covariates=future_covariates,
                                                        n=n,
                                                        input_chunk_length=input_chunk_length,
                                                        output_chunk_length=output_chunk_length)

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(self, idx):
        past_target, past_covs, future_past_covs = self.ds_past[idx]
        _, historic_future_covs, future_covs = self.ds_future[idx]
        return past_target, past_covs, historic_future_covs, future_covs, future_past_covs

    def get_target(self, idx: int) -> TimeSeries:
        return self.ds_past.get_target(idx)


class SkipCovariatesInferenceDataset(InferenceDataset):
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
        self.ds_past = PastCovariatesInferenceDataset(target_series=target_series,
                                                      covariates=past_covariates,
                                                      n=n,
                                                      input_chunk_length=input_chunk_length,
                                                      output_chunk_length=output_chunk_length)

        self.ds_future = FutureCovariatesInferenceDataset(target_series=target_series,
                                                          covariates=future_covariates,
                                                          n=n,
                                                          input_chunk_length=input_chunk_length)

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(self, idx):
        past_target, past_covs, future_past_covs = self.ds_past[idx]
        _, future_covs = self.ds_future[idx]
        return past_target, past_covs, future_covs, future_past_covs

    def get_target(self, idx: int) -> TimeSeries:
        return self.ds_past.get_target(idx)
