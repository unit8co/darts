"""
Shifted Training Dataset
------------------------
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from darts import TimeSeries
from darts.logging import raise_if_not

from .training_dataset import (
    DualCovariatesTrainingDataset,
    FutureCovariatesTrainingDataset,
    MixedCovariatesTrainingDataset,
    PastCovariatesTrainingDataset,
    SplitCovariatesTrainingDataset,
    TrainingDataset,
)
from .utils import CovariateType


class PastCovariatesShiftedDataset(PastCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        length: int = 12,
        shift: int = 1,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
    ):
        """
        A time series dataset containing tuples of (past_target, past_covariates, static_covariates, future_target)
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
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """
        super().__init__()

        self.ds = GenericShiftedDataset(
            target_series=target_series,
            covariates=covariates,
            input_chunk_length=length,
            output_chunk_length=length,
            shift=shift,
            shift_covariates=False,
            max_samples_per_ts=max_samples_per_ts,
            covariate_type=CovariateType.PAST,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(
        self, idx
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        return self.ds[idx]


class FutureCovariatesShiftedDataset(FutureCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        length: int = 12,
        shift: int = 1,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
    ):
        """
        A time series dataset containing tuples of (past_target, future_covariates, static_covariates, future_target)
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
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """

        super().__init__()

        self.ds = GenericShiftedDataset(
            target_series=target_series,
            covariates=covariates,
            input_chunk_length=length,
            output_chunk_length=length,
            shift=shift,
            shift_covariates=True,
            max_samples_per_ts=max_samples_per_ts,
            covariate_type=CovariateType.FUTURE,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(
        self, idx
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        return self.ds[idx]


class DualCovariatesShiftedDataset(DualCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        length: int = 12,
        shift: int = 1,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
    ):
        """
        A time series dataset containing tuples of
        (past_target, historic_future_covariates, future_covariates, static_covariates, future_target)
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
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """

        super().__init__()

        # This dataset is in charge of serving historical future covariates
        self.ds_past = GenericShiftedDataset(
            target_series=target_series,
            covariates=covariates,
            input_chunk_length=length,
            output_chunk_length=length,
            shift=shift,
            shift_covariates=False,
            max_samples_per_ts=max_samples_per_ts,
            covariate_type=CovariateType.HISTORIC_FUTURE,
            use_static_covariates=use_static_covariates,
        )

        # This dataset is in charge of serving future covariates
        self.ds_future = GenericShiftedDataset(
            target_series=target_series,
            covariates=covariates,
            input_chunk_length=length,
            output_chunk_length=length,
            shift=shift,
            shift_covariates=True,
            max_samples_per_ts=max_samples_per_ts,
            covariate_type=CovariateType.FUTURE,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(
        self, idx
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        past_target, past_covariate, static_covariate, future_target = self.ds_past[idx]
        _, future_covariate, _, _ = self.ds_future[idx]
        return (
            past_target,
            past_covariate,
            future_covariate,
            static_covariate,
            future_target,
        )


class MixedCovariatesShiftedDataset(MixedCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        length: int = 12,
        shift: int = 1,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
    ):
        """
        A time series dataset containing tuples of (past_target, past_covariates, historic_future_covariates,
        future_covariates, static_covariates, future_target) arrays, which all have length `length`.
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
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """
        super().__init__()

        # This dataset is in charge of serving past covariates
        self.ds_past = GenericShiftedDataset(
            target_series=target_series,
            covariates=past_covariates,
            input_chunk_length=length,
            output_chunk_length=length,
            shift=shift,
            shift_covariates=False,
            max_samples_per_ts=max_samples_per_ts,
            covariate_type=CovariateType.PAST,
            use_static_covariates=use_static_covariates,
        )

        # The dual dataset serves both historical and future future covariates
        self.ds_dual = DualCovariatesShiftedDataset(
            target_series=target_series,
            covariates=future_covariates,
            length=length,
            shift=shift,
            max_samples_per_ts=max_samples_per_ts,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(
        self, idx
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:

        past_target, past_covariate, static_covariate, future_target = self.ds_past[idx]
        _, historic_future_covariate, future_covariate, _, _ = self.ds_dual[idx]
        return (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariate,
            future_target,
        )


class SplitCovariatesShiftedDataset(SplitCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        length: int = 12,
        shift: int = 1,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
    ):
        """
        A time series dataset containing tuples of (past_target, past_covariates, future_covariates, static_covariates,
        future_target) arrays, which all have length `length`.
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
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """

        super().__init__()

        # This dataset is in charge of serving past covariates
        self.ds_past = GenericShiftedDataset(
            target_series=target_series,
            covariates=past_covariates,
            input_chunk_length=length,
            output_chunk_length=length,
            shift=shift,
            shift_covariates=False,
            max_samples_per_ts=max_samples_per_ts,
            covariate_type=CovariateType.PAST,
            use_static_covariates=use_static_covariates,
        )

        # This dataset is in charge of serving future covariates
        self.ds_future = GenericShiftedDataset(
            target_series=target_series,
            covariates=future_covariates,
            input_chunk_length=length,
            output_chunk_length=length,
            shift=shift,
            shift_covariates=True,
            max_samples_per_ts=max_samples_per_ts,
            covariate_type=CovariateType.FUTURE,
            use_static_covariates=use_static_covariates,
        )

    def __len__(self):
        return len(self.ds_past)

    def __getitem__(
        self, idx
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        past_target, past_covariate, static_covariate, future_target = self.ds_past[idx]
        _, future_covariate, _, _ = self.ds_future[idx]
        return (
            past_target,
            past_covariate,
            future_covariate,
            static_covariate,
            future_target,
        )


class GenericShiftedDataset(TrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        shift: int = 1,
        shift_covariates: bool = False,
        max_samples_per_ts: Optional[int] = None,
        covariate_type: CovariateType = CovariateType.NONE,
        use_static_covariates: bool = True,
    ):
        """
        Contains (past_target, <X>_covariates, static_covariates, future_target), where "<X>" is past if
        `shift_covariates = False` and future otherwise.
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
        covariate_type
            An instance of `CovariateType` describing the type of `covariates`.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        """
        super().__init__()

        self.target_series = (
            [target_series] if isinstance(target_series, TimeSeries) else target_series
        )
        self.covariates = (
            [covariates] if isinstance(covariates, TimeSeries) else covariates
        )
        self.covariate_type = covariate_type

        raise_if_not(
            covariates is None or len(self.target_series) == len(self.covariates),
            "The provided sequence of target series must have the same length as "
            "the provided sequence of covariate series.",
        )

        self.input_chunk_length, self.output_chunk_length = (
            input_chunk_length,
            output_chunk_length,
        )
        self.shift, self.shift_covariates = shift, shift_covariates
        self.max_samples_per_ts = max_samples_per_ts

        self.size_of_both_chunks = max(
            self.input_chunk_length, self.shift + self.output_chunk_length
        )

        if self.max_samples_per_ts is None:
            # read all time series to get the maximum size
            self.max_samples_per_ts = (
                max(len(ts) for ts in self.target_series) - self.size_of_both_chunks + 1
            )

        self.ideal_nr_samples = len(self.target_series) * self.max_samples_per_ts
        self.use_static_covariates = use_static_covariates

    def __len__(self):
        return self.ideal_nr_samples

    def __getitem__(
        self, idx
    ) -> Tuple[
        np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
    ]:
        # determine the index of the time series.
        target_idx = idx // self.max_samples_per_ts
        target_series = self.target_series[target_idx]
        target_vals = target_series.random_component_values(copy=False)

        # determine the actual number of possible samples in this time series
        n_samples_in_ts = len(target_vals) - self.size_of_both_chunks + 1

        raise_if_not(
            n_samples_in_ts >= 1,
            "The dataset contains some time series that are too short to contain "
            "`max(self.input_chunk_length, self.shift + self.output_chunk_length)` "
            "({}-th series)".format(target_idx),
        )

        # determine the index at the end of the output chunk
        # it is originally in [0, self.max_samples_per_ts), so we use a modulo to have it in [0, n_samples_in_ts)
        end_of_output_idx = (
            len(target_series)
            - (idx - (target_idx * self.max_samples_per_ts)) % n_samples_in_ts
        )

        # optionally, load covariates
        covariate_series = (
            self.covariates[target_idx] if self.covariates is not None else None
        )

        main_covariate_type = CovariateType.NONE
        if self.covariates is not None:
            main_covariate_type = (
                CovariateType.FUTURE if self.shift_covariates else CovariateType.PAST
            )

        # get all indices for the current sample
        (
            past_start,
            past_end,
            future_start,
            future_end,
            covariate_start,
            covariate_end,
        ) = self._memory_indexer(
            target_idx=target_idx,
            target_series=target_series,
            shift=self.shift,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            end_of_output_idx=end_of_output_idx,
            covariate_series=covariate_series,
            covariate_type=main_covariate_type,
        )

        # extract sample target
        future_target = target_vals[future_start:future_end]
        past_target = target_vals[past_start:past_end]

        # optionally, extract sample covariates
        covariate = None
        if self.covariates is not None:
            raise_if_not(
                covariate_end <= len(covariate_series),
                f"The dataset contains {main_covariate_type.value} covariates "
                f"that don't extend far enough into the future. ({idx}-th sample)",
            )

            covariate = covariate_series.random_component_values(copy=False)[
                covariate_start:covariate_end
            ]

            raise_if_not(
                len(covariate)
                == (
                    self.output_chunk_length
                    if self.shift_covariates
                    else self.input_chunk_length
                ),
                f"The dataset contains {main_covariate_type.value} covariates "
                f"whose time axis doesn't allow to obtain the input (or output) chunk relative to the "
                f"target series.",
            )

        if self.use_static_covariates:
            static_covariate = target_series.static_covariates_values(copy=False)
        else:
            static_covariate = None
        return past_target, covariate, static_covariate, future_target
