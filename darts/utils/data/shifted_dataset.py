"""
Shifted Training Dataset
------------------------
"""

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.data.training_dataset import (
    DualCovariatesTrainingDataset,
    FutureCovariatesTrainingDataset,
    MixedCovariatesTrainingDataset,
    PastCovariatesTrainingDataset,
    SplitCovariatesTrainingDataset,
    TrainingDataset,
)
from darts.utils.data.utils import CovariateType, _process_sample_weight
from darts.utils.ts_utils import series2seq

logger = get_logger(__name__)


class PastCovariatesShiftedDataset(PastCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        length: int = 12,
        shift: int = 1,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        A time series dataset containing tuples of (past_target, past_covariates, static_covariates, sample weights,
        future_target)
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
            The number of time steps by which to shift the output chunks relative to the start of the input chunks.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        sample_weight
            Optionally, some sample weights to apply to the target `series` labels. They are applied per observation,
            per label (each step in `output_chunk_length`), and per component.
            If a series or sequence of series, then those weights are used. If the weight series only have a single
            component / column, then the weights are applied globally to all components in `series`. Otherwise, for
            component-specific weights, the number of components must match those of `series`.
            If a string, then the weights are generated using built-in weighting functions. The available options are
            `"linear"` or `"exponential"` decay - the further in the past, the lower the weight. The weights are
            computed globally based on the length of the longest series in `series`. Then for each series, the weights
            are extracted from the end of the global weights. This gives a common time weighting across all series.
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
            sample_weight=sample_weight,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(
        self, idx
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
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
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        A time series dataset containing tuples of (past_target, future_covariates, static_covariates, sample weights,
        future_target)
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
            The number of time steps by which to shift the output chunks relative to the start of the input chunks.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        sample_weight
            Optionally, some sample weights to apply to the target `series` labels. They are applied per observation,
            per label (each step in `output_chunk_length`), and per component.
            If a series or sequence of series, then those weights are used. If the weight series only have a single
            component / column, then the weights are applied globally to all components in `series`. Otherwise, for
            component-specific weights, the number of components must match those of `series`.
            If a string, then the weights are generated using built-in weighting functions. The available options are
            `"linear"` or `"exponential"` decay - the further in the past, the lower the weight. The weights are
            computed globally based on the length of the longest series in `series`. Then for each series, the weights
            are extracted from the end of the global weights. This gives a common time weighting across all series.
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
            sample_weight=sample_weight,
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(
        self, idx
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
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
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        A time series dataset containing tuples of
        (past_target, historic_future_covariates, future_covariates, static_covariates, sample weights,
        future_target)
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
            The number of time steps by which to shift the output chunks relative to the start of the input chunks.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        sample_weight
            Optionally, some sample weights to apply to the target `series` labels. They are applied per observation,
            per label (each step in `output_chunk_length`), and per component.
            If a series or sequence of series, then those weights are used. If the weight series only have a single
            component / column, then the weights are applied globally to all components in `series`. Otherwise, for
            component-specific weights, the number of components must match those of `series`.
            If a string, then the weights are generated using built-in weighting functions. The available options are
            `"linear"` or `"exponential"` decay - the further in the past, the lower the weight. The weights are
            computed globally based on the length of the longest series in `series`. Then for each series, the weights
            are extracted from the end of the global weights. This gives a common time weighting across all series.
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
            sample_weight=sample_weight,
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
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        past_target, past_covariate, static_covariate, sample_weight, future_target = (
            self.ds_past[idx]
        )
        _, future_covariate, _, _, _ = self.ds_future[idx]
        return (
            past_target,
            past_covariate,
            future_covariate,
            static_covariate,
            sample_weight,
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
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        A time series dataset containing tuples of (past_target, past_covariates, historic_future_covariates,
        future_covariates, static_covariates, sample weights, future_target) arrays, which all have length `length`.
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
            The number of time steps by which to shift the output chunks relative to the start of the input chunks.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        sample_weight
            Optionally, some sample weights to apply to the target `series` labels. They are applied per observation,
            per label (each step in `output_chunk_length`), and per component.
            If a series or sequence of series, then those weights are used. If the weight series only have a single
            component / column, then the weights are applied globally to all components in `series`. Otherwise, for
            component-specific weights, the number of components must match those of `series`.
            If a string, then the weights are generated using built-in weighting functions. The available options are
            `"linear"` or `"exponential"` decay - the further in the past, the lower the weight. The weights are
            computed globally based on the length of the longest series in `series`. Then for each series, the weights
            are extracted from the end of the global weights. This gives a common time weighting across all series.
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
            sample_weight=sample_weight,
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
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        past_target, past_covariate, static_covariate, sample_weight, future_target = (
            self.ds_past[idx]
        )
        _, historic_future_covariate, future_covariate, _, _, _ = self.ds_dual[idx]
        return (
            past_target,
            past_covariate,
            historic_future_covariate,
            future_covariate,
            static_covariate,
            sample_weight,
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
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        A time series dataset containing tuples of (past_target, past_covariates, future_covariates, static_covariates,
        sample weights, future_target) arrays, which all have length `length`.
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
            The number of time steps by which to shift the output chunks relative to the start of the input chunks.
        max_samples_per_ts
            This is an upper bound on the number of tuples that can be produced per time series.
            It can be used in order to have an upper bound on the total size of the dataset and
            ensure proper sampling. If `None`, it will read all of the individual time series in advance (at dataset
            creation) to know their sizes, which might be expensive on big datasets.
            If some series turn out to have a length that would allow more than `max_samples_per_ts`, only the
            most recent `max_samples_per_ts` samples will be considered.
        use_static_covariates
            Whether to use/include static covariate data from input series.
        sample_weight
            Optionally, some sample weights to apply to the target `series` labels. They are applied per observation,
            per label (each step in `output_chunk_length`), and per component.
            If a series or sequence of series, then those weights are used. If the weight series only have a single
            component / column, then the weights are applied globally to all components in `series`. Otherwise, for
            component-specific weights, the number of components must match those of `series`.
            If a string, then the weights are generated using built-in weighting functions. The available options are
            `"linear"` or `"exponential"` decay - the further in the past, the lower the weight. The weights are
            computed globally based on the length of the longest series in `series`. Then for each series, the weights
            are extracted from the end of the global weights. This gives a common time weighting across all series.
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
            sample_weight=sample_weight,
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
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        past_target, past_covariate, static_covariate, sample_weight, future_target = (
            self.ds_past[idx]
        )
        _, future_covariate, _, _, _ = self.ds_future[idx]
        return (
            past_target,
            past_covariate,
            future_covariate,
            static_covariate,
            sample_weight,
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
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        Contains (past_target, <X>_covariates, static_covariates, sample weights, future_target), where "<X>" is past
        if `shift_covariates = False` and future otherwise.
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
            The number of time steps by which to shift the output chunks relative to the start of the input chunks.
        shift_covariates
            Whether to shift the covariates forward the same way as the target.
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
        sample_weight
            Optionally, some sample weights to apply to the target `series` labels. They are applied per observation,
            per label (each step in `output_chunk_length`), and per component.
            If a series or sequence of series, then those weights are used. If the weight series only have a single
            component / column, then the weights are applied globally to all components in `series`. Otherwise, for
            component-specific weights, the number of components must match those of `series`.
            If a string, then the weights are generated using built-in weighting functions. The available options are
            `"linear"` or `"exponential"` decay - the further in the past, the lower the weight. The weights are
            computed globally based on the length of the longest series in `series`. Then for each series, the weights
            are extracted from the end of the global weights. This gives a common time weighting across all series.
        """
        super().__init__()

        # setup target and sequence
        self.target_series = series2seq(target_series)
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.shift = shift
        self.max_samples_per_ts = max_samples_per_ts
        self.size_of_both_chunks = max(
            self.input_chunk_length, self.shift + self.output_chunk_length
        )

        # setup covariates; ignore past/historic covariates when `icl==0` and future covariates when `ocl==0`
        main_covariate_type = CovariateType.NONE
        if covariates is not None:
            if shift_covariates and output_chunk_length > 0:
                main_covariate_type = CovariateType.FUTURE
            elif not shift_covariates and input_chunk_length > 0:
                main_covariate_type = CovariateType.PAST
            else:
                main_covariate_type = CovariateType.NONE

        self.main_covariate_type = main_covariate_type
        if main_covariate_type is not CovariateType.NONE:
            self.covariates = series2seq(covariates)
            self.covariate_type = covariate_type
            self.shift_covariates = shift_covariates
        else:
            self.covariates = None
            self.covariate_type = CovariateType.NONE
            self.shift_covariates = 0
        self.use_static_covariates = use_static_covariates

        if self.covariates is not None and len(self.target_series) != len(
            self.covariates
        ):
            raise_log(
                ValueError(
                    "The provided sequence of target series must have the same length as "
                    "the provided sequence of covariate series."
                ),
                logger=logger,
            )

        # setup sample weights; ignore weights when `ocl==0`
        self.sample_weight = None
        if sample_weight is not None:
            if output_chunk_length > 0:
                self.sample_weight = _process_sample_weight(
                    sample_weight, self.target_series
                )
            else:
                self.sample_weight = None

        # setup samples
        if self.max_samples_per_ts is None:
            # read all time series to get the maximum size
            self.max_samples_per_ts = (
                max(len(ts) for ts in self.target_series) - self.size_of_both_chunks + 1
            )
        self.ideal_nr_samples = len(self.target_series) * self.max_samples_per_ts

    def __len__(self):
        return self.ideal_nr_samples

    def __getitem__(
        self, idx
    ) -> tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        np.ndarray,
    ]:
        # determine the index of the time series.
        target_idx = idx // self.max_samples_per_ts
        target_series = self.target_series[target_idx]
        target_vals = target_series.random_component_values(copy=False)

        # determine the actual number of possible samples in this time series
        n_samples_in_ts = len(target_vals) - self.size_of_both_chunks + 1

        if n_samples_in_ts < 1:
            raise_log(
                ValueError(
                    "The dataset contains some time series that are too short to contain "
                    "`max(self.input_chunk_length, self.shift + self.output_chunk_length)` "
                    f"({target_idx}-th series)"
                ),
                logger=logger,
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

        # optionally, load sample weight
        if self.sample_weight is not None:
            sample_weight_series = self.sample_weight[target_idx]
            weight_n_comp = sample_weight_series.n_components
            if weight_n_comp > 1 and weight_n_comp != target_series.n_components:
                raise_log(
                    ValueError(
                        "The number of components in `sample_weight` must either be `1` or match "
                        f"the number of target series components `{target_series.n_components}`. "
                        f"({target_idx}-th series)"
                    ),
                    logger=logger,
                )
        else:
            sample_weight_series = None

        # get all indices for the current sample
        (
            past_start,
            past_end,
            future_start,
            future_end,
            covariate_start,
            covariate_end,
            sample_weight_start,
            sample_weight_end,
        ) = self._memory_indexer(
            target_idx=target_idx,
            target_series=target_series,
            shift=self.shift,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            end_of_output_idx=end_of_output_idx,
            covariate_series=covariate_series,
            covariate_type=self.main_covariate_type,
            sample_weight_series=sample_weight_series,
        )

        # extract sample target
        future_target = target_vals[future_start:future_end]
        past_target = target_vals[past_start:past_end]

        # extract sample covariates
        covariate = None
        if self.covariates is not None:
            if covariate_end > len(covariate_series):
                raise_log(
                    ValueError(
                        f"The dataset contains {self.main_covariate_type.value} covariates "
                        f"that don't extend far enough into the future. ({idx}-th sample)"
                    ),
                    logger=logger,
                )

            covariate = covariate_series.random_component_values(copy=False)[
                covariate_start:covariate_end
            ]

            if len(covariate) != (
                self.output_chunk_length
                if self.shift_covariates
                else self.input_chunk_length
            ):
                raise_log(
                    ValueError(
                        f"The dataset contains {self.main_covariate_type.value} covariates "
                        f"whose time axis doesn't allow to obtain the input (or output) chunk relative to the "
                        f"target series."
                    ),
                    logger=logger,
                )

        # extract sample weights
        sample_weight = None
        if self.sample_weight is not None:
            if sample_weight_end > len(sample_weight_series):
                raise_log(
                    ValueError(
                        f"The dataset contains sample weights "
                        f"that don't extend far enough into the future. ({idx}-th sample)"
                    ),
                    logger=logger,
                )

            sample_weight = sample_weight_series.random_component_values(copy=False)[
                sample_weight_start:sample_weight_end
            ]

            if len(sample_weight) != self.output_chunk_length:
                raise_log(
                    ValueError(
                        "The dataset contains sample weights whose time axis doesn't allow to obtain "
                        "the input (or output) chunk relative to the target series."
                    ),
                    logger=logger,
                )

        # extract sample static covariates
        if self.use_static_covariates:
            static_covariate = target_series.static_covariates_values(copy=False)
        else:
            static_covariate = None
        return past_target, covariate, static_covariate, sample_weight, future_target
