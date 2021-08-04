"""
Shifted Training Dataset
------------------------
"""

from typing import Union, Sequence, Optional, Tuple
import numpy as np

from ...timeseries import TimeSeries
from .training_dataset import (PastCovariatesTrainingDataset,
                               FutureCovariatesTrainingDataset,
                               DualCovariatesTrainingDataset,
                               MixedCovariatesTrainingDataset,
                               SplitCovariatesTrainingDataset)
from .utils import _get_matching_index
from ..utils import raise_if_not


class PastCovariatesShiftedDataset(PastCovariatesTrainingDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 length: int = 12,
                 shift: int = 1,
                 max_samples_per_ts: Optional[int] = None):
        """
        A time series dataset containing tuples of (past_target, past_covariates, future_target)
        arrays, which all have length `length`.
        The "future_target" is the "past_target" target shifted by `shift` time steps forward.
        So if an emitted "past_target" (and "past_covariates") goes from position `i` to `i+length`,
        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `length + shift`.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates
            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        length
            The length of the emitted past and future series.
        shift
            The number of time steps by which to shift the output relative to the input.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        """
        super().__init__()

        self.ds = GenericShiftedDataset(target_series=target_series,
                                        covariates=covariates,
                                        input_chunk_length=length,
                                        output_chunk_length=length,
                                        shift=shift,
                                        shift_covariates=False,
                                        max_samples_per_ts=max_samples_per_ts)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        return self.ds[idx]


class FutureCovariatesShiftedDataset(FutureCovariatesTrainingDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 length: int = 12,
                 shift: int = 1,
                 max_samples_per_ts: Optional[int] = None):
        """
        A time series dataset containing tuples of (past_target, future_covariates, future_target)
        arrays, which all have length `length`.
        The "future_target" is the "past_target" target shifted by `shift` time steps forward.
        So if an emitted "past_target" goes from position `i` to `i+length`,
        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.
        The slicing future covariates matches that of future targets. The slicing
        itself relies on time indexes to align the series if they have unequal lengths.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `length + shift`.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        length
            The length of the emitted past and future series.
        shift
            The number of time steps by which to shift the output relative to the input.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        """

        super().__init__()

        self.ds = GenericShiftedDataset(target_series=target_series,
                                        covariates=covariates,
                                        input_chunk_length=length,
                                        output_chunk_length=length,
                                        shift=shift,
                                        shift_covariates=True,
                                        max_samples_per_ts=max_samples_per_ts)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        return self.ds[idx]


class DualCovariatesShiftedDataset(DualCovariatesTrainingDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 length: int = 12,
                 shift: int = 1,
                 max_samples_per_ts: Optional[int] = None):
        """
        A time series dataset containing tuples of
        (past_target, historic_future_covariates, future_covariates, future_target)
        arrays, which all have length `length`.
        The "future_target" is the "past_target" target shifted by `shift` time steps forward.
        So if an emitted "past_target" goes from position `i` to `i+length`,
        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.
        The slicing "future_covariates" matches that of "futuretarget" and the slicing of "historic_future_covariates"
        matches that of "past_target". The slicing itself relies on time indexes to align the series if they have
        unequal lengths.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `length + shift`.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        length
            The length of the emitted past and future series.
        shift
            The number of time steps by which to shift the output relative to the input.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        """

        super().__init__()

        # This dataset is in charge of serving historical future covariates
        self.ds_past = GenericShiftedDataset(target_series=target_series,
                                             covariates=covariates,
                                             input_chunk_length=length,
                                             output_chunk_length=length,
                                             shift=shift,
                                             shift_covariates=False,
                                             max_samples_per_ts=max_samples_per_ts)

        # This dataset is in charge of serving future covariates
        self.ds_future = GenericShiftedDataset(target_series=target_series,
                                               covariates=covariates,
                                               input_chunk_length=length,
                                               output_chunk_length=length,
                                               shift=shift,
                                               shift_covariates=True,
                                               max_samples_per_ts=max_samples_per_ts)

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        past_target, past_covariate, future_target = self.ds_past[idx]
        _, future_covariate, _ = self.ds_future[idx]
        return past_target, past_covariate, future_covariate, future_target


class MixedCovariatesShiftedDataset(MixedCovariatesTrainingDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 length: int = 12,
                 shift: int = 1,
                 max_samples_per_ts: Optional[int] = None):
        """
        A time series dataset containing tuples of (past_target, past_covariates, historic_future_covariates,
                                                    future_covariates, future_target)
        arrays, which all have length `length`.
        The "future_target" is the "past_target" target shifted by `shift` time steps forward.
        So if an emitted "past_target" goes from position `i` to `i+length`,
        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.
        The slicing of past and future covariates matches that of past and future targets, respectively. The slicing
        itself relies on time indexes to align the series if they have unequal lengths.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `length + shift`.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        past_covariates
            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        future_covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. This has to follow
            the same constraints as `past_covariates`.
        length
            The length of the emitted past and future series.
        shift
            The number of time steps by which to shift the output relative to the input.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        """
        super().__init__()

        # This dataset is in charge of serving past covariates
        self.ds_past = GenericShiftedDataset(target_series=target_series,
                                             covariates=past_covariates,
                                             input_chunk_length=length,
                                             output_chunk_length=length,
                                             shift=shift,
                                             shift_covariates=False,
                                             max_samples_per_ts=max_samples_per_ts)

        # The dual dataset serves both historical and future future covariates
        self.ds_dual = DualCovariatesShiftedDataset(target_series=target_series,
                                                    covariates=future_covariates,
                                                    length=length,
                                                    shift=shift,
                                                    max_samples_per_ts=max_samples_per_ts)

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
                                        Optional[np.ndarray], np.ndarray]:

        past_target, past_covariate, future_target = self.ds_past[idx]
        _, historic_future_covariate, future_covariate, _ = self.ds_dual[idx]
        return past_target, past_covariate, historic_future_covariate, future_covariate, future_target


class SplitCovariatesShiftedDataset(SplitCovariatesTrainingDataset):
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 length: int = 12,
                 shift: int = 1,
                 max_samples_per_ts: Optional[int] = None):
        """
        A time series dataset containing tuples of (past_target, past_covariates, future_covariates, future_target)
        arrays, which all have length `length`.
        The "future_target" is the "past_target" target shifted by `shift` time steps forward.
        So if an emitted "past_target" goes from position `i` to `i+length`,
        the emitted "future_target" will go from position `i+shift` to `i+shift+length`.
        The slicing of past and future covariates matches that of past and future targets, respectively. The slicing
        itself relies on time indexes to align the series if they have unequal lengths.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `length + shift`.
        If these conditions are not satisfied, an error will be raised when trying to access some of the splits.

        The sampling is uniform over the number of time series; i.e., the i-th sample of this dataset has
        a probability 1/N of coming from any of the N time series in the sequence. If the time series have different
        lengths, they will contain different numbers of slices. Therefore, some particular slices may
        be sampled more often than others if they belong to shorter time series.

        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        past_covariates
            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        future_covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. This has to follow
            the same constraints as `past_covariates`.
        length
            The length of the emitted past and future series.
        shift
            The number of time steps by which to shift the output relative to the input.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        """

        super().__init__()

        # This dataset is in charge of serving past covariates
        self.ds_past = GenericShiftedDataset(target_series=target_series,
                                             covariates=past_covariates,
                                             input_chunk_length=length,
                                             output_chunk_length=length,
                                             shift=shift,
                                             shift_covariates=False,
                                             max_samples_per_ts=max_samples_per_ts)

        # This dataset is in charge of serving future covariates
        self.ds_future = GenericShiftedDataset(target_series=target_series,
                                               covariates=future_covariates,
                                               input_chunk_length=length,
                                               output_chunk_length=length,
                                               shift=shift,
                                               shift_covariates=True,
                                               max_samples_per_ts=max_samples_per_ts)

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        past_target, past_covariate, future_target = self.ds_past[idx]
        _, future_covariate, _ = self.ds_future[idx]
        return past_target, past_covariate, future_covariate, future_target


class GenericShiftedDataset:
    def __init__(self,
                 target_series: Union[TimeSeries, Sequence[TimeSeries]],
                 covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
                 input_chunk_length: int = 12,
                 output_chunk_length: int = 1,
                 shift: int = 1,
                 shift_covariates: bool = False,
                 max_samples_per_ts: Optional[int] = None):
        """
        Contains (past_target, <X>_covariate, future_target), where "<X>" is past if `shift_covariates = False`
        and future otherwise.
        The past chunks have length `input_chunk_length` and the future chunks have length `output_chunk_length`.
        The future chunks start `shift` after the past chunks' start.

        This is meant to be a "generic" dataset that can be used to build ShiftedDataset's
        (when `input_chunk_length = output_chunk_length`), or SequenceDataset's (when `shift = input_chunk_length`).

        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates
            Optionally, one or a sequence of `TimeSeries` containing covariates.
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        shift
            The number of time steps by which to shift the output chunks relative to the input chunks.
        shift_covariates
            Whether or not to shift the covariates forward the same way as the target.
            FutureCovariatesModel's require this set to True, while PastCovariatesModel's require this set to False.
        max_samples_per_ts
            This is an upper bound on the number of (input, output, input_covariates) tuples that can be produced
            per time series. It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        """
        super().__init__()

        self.target_series = [target_series] if isinstance(target_series, TimeSeries) else target_series
        self.covariates = [covariates] if isinstance(covariates, TimeSeries) else covariates

        raise_if_not(covariates is None or len(self.target_series) == len(self.covariates),
                     'The provided sequence of target series must have the same length as '
                     'the provided sequence of covariate series.')

        self.input_chunk_length, self.output_chunk_length = input_chunk_length, output_chunk_length
        self.shift, self.shift_covariates = shift, shift_covariates
        self.max_samples_per_ts = max_samples_per_ts

        self.size_of_both_chunks = max(self.input_chunk_length, self.shift + self.output_chunk_length)

        if self.max_samples_per_ts is None:
            # read all time series to get the maximum size
            self.max_samples_per_ts = max(len(ts) for ts in self.target_series) - \
                                      self.size_of_both_chunks + 1

        self.ideal_nr_samples = len(self.target_series) * self.max_samples_per_ts

    def __len__(self):
        return self.ideal_nr_samples

    def __getitem__(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        # determine the index of the time series.
        ts_idx = idx // self.max_samples_per_ts
        ts_target = self.target_series[ts_idx]
        target_vals = ts_target.values(copy=False)

        # determine the actual number of possible samples in this time series
        n_samples_in_ts = len(target_vals) - self.size_of_both_chunks + 1

        raise_if_not(n_samples_in_ts >= 1,
                     'The dataset contains some time series that are too short to contain '
                     '`max(self.input_chunk_length, self.shift + self.output_chunk_length)` '
                     '({}-th series)'.format(ts_idx))

        # Determine the index of the end of the output, starting from the end.
        # It is originally in [0, self.max_samples_per_ts), so we use a modulo to have it in [0, n_samples_in_ts)
        end_of_output_idx = (idx - (ts_idx * self.max_samples_per_ts)) % n_samples_in_ts

        # select forecast point and target period, using the previously computed indexes
        if end_of_output_idx == 0:
            # we need this case because "-0" is not supported as an indexing bound
            future_target = target_vals[-self.output_chunk_length:]
        else:
            future_target = target_vals[-(self.output_chunk_length + end_of_output_idx):-end_of_output_idx]

        # starting from the end
        start_of_input_idx = end_of_output_idx + self.output_chunk_length + self.shift

        # select input period; look at the `input_chunk_length` points before the forecast point
        if -(start_of_input_idx - self.input_chunk_length) == 0:
            # handle "-0" indexing bound
            past_target = target_vals[-start_of_input_idx:]
        else:
            past_target = target_vals[-start_of_input_idx:-(start_of_input_idx - self.input_chunk_length)]

        # optionally also produce the input covariate
        covariate = None
        if self.covariates is not None:
            ts_covariate = self.covariates[ts_idx]
            cov_vals = ts_covariate.values(copy=False)

            if self.shift_covariates:
                # We need to return the future covariates. In this case we use the same indexing as for
                # "future_target" (shifting the index if the time axes of target and covariate are not the same)
                end_of_output_idx_cov = _get_matching_index(ts_target, ts_covariate, end_of_output_idx)
                raise_if_not(end_of_output_idx_cov >= 0,
                             "The dataset contains some covariates that don't extend far enough into the future. "
                             "({}-th sample)".format(idx))
                if end_of_output_idx_cov == 0:
                    # we need this case because "-0" is not supported as an indexing bound
                    covariate = cov_vals[-self.output_chunk_length:]
                else:
                    covariate = cov_vals[-(self.output_chunk_length + end_of_output_idx_cov):-end_of_output_idx_cov]

            else:
                # We need to return the past covariates. In this case we use the same indexing as for
                # "past_target" (shifting the index if the time axes of target and covariate are not the same)
                start_of_input_idx_cov = _get_matching_index(ts_target, ts_covariate, start_of_input_idx)
                if -(start_of_input_idx_cov - self.input_chunk_length) == 0:
                    # handle "-0" indexing bound
                    covariate = cov_vals[-start_of_input_idx_cov:]
                else:
                    covariate = cov_vals[-start_of_input_idx_cov:-(start_of_input_idx_cov - self.input_chunk_length)]

            raise_if_not(len(covariate) == (self.output_chunk_length if self.shift_covariates else
                                            self.input_chunk_length),
                         "The dataset contains some covariate series whose time axis doesn't allow to "
                         "obtain the input (or output) chunk relative to the target series.")

        return past_target, covariate, future_target
