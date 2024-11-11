"""
Sequential Training Dataset
---------------------------
"""

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from darts import TimeSeries
from darts.utils.data.shifted_dataset import GenericShiftedDataset
from darts.utils.data.training_dataset import (
    DualCovariatesTrainingDataset,
    FutureCovariatesTrainingDataset,
    MixedCovariatesTrainingDataset,
    PastCovariatesTrainingDataset,
    SplitCovariatesTrainingDataset,
)
from darts.utils.data.utils import CovariateType


class PastCovariatesSequentialDataset(PastCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        A time series dataset containing tuples of (past_target, past_covariates, static_covariates, sample weights,
        future_target).
        The "past" series have length `input_chunk_length` and the "future" series have
        length `output_chunk_length`. The "future" series are immediately consecutive to the "past" series.
        The slicing of past and future covariates matches that of past and future targets, respectively. The slicing
        itself relies on time indexes to align the series if they have unequal lengths.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `input_chunk_length + output_chunk_length`.
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
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future.
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
        shift = input_chunk_length + output_chunk_shift
        self.ds = GenericShiftedDataset(
            target_series=target_series,
            covariates=covariates,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
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


class FutureCovariatesSequentialDataset(FutureCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        A time series dataset containing tuples of (past_target, future_covariates, static_covariates, sample weights,
        future_target).
        The "past" series have length `input_chunk_length` and the "future" series have
        length `output_chunk_length`. The "future" series are immediately consecutive to the "past" series.
        The slicing of past and future covariates matches that of past and future targets, respectively. The slicing
        itself relies on time indexes to align the series if they have unequal lengths.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `input_chunk_length + output_chunk_length`.
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
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future.
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
        shift = input_chunk_length + output_chunk_shift
        self.ds = GenericShiftedDataset(
            target_series=target_series,
            covariates=covariates,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
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


class DualCovariatesSequentialDataset(DualCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        A time series dataset containing tuples of
        (past_target, historic_future_covariates, future_covariates, static_covariates, sample weights,
        future_target).
        The "past" series (incl `historic_future_covariates`) have length `input_chunk_length`
        and the "future" series have length `output_chunk_length`. The "future" series are immediately consecutive
        to the "past" series. The slicing of past and future covariates matches that of past and future targets,
        respectively. The slicing itself relies on time indexes to align the series if they have unequal lengths.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `input_chunk_length + output_chunk_length`.
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
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future.
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
        shift = input_chunk_length + output_chunk_shift
        # This dataset is in charge of historical future covariates
        self.ds_past = GenericShiftedDataset(
            target_series=target_series,
            covariates=covariates,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
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
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
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


class MixedCovariatesSequentialDataset(MixedCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        A time series dataset containing tuples of
        (past_target, past_covariates, historic_future_covariates, future_covariates, static_covariates,
        sample weights, future_target).
        The "past" series (incl `historic_future_covariates`) have length `input_chunk_length`
        and the "future" series have length `output_chunk_length`. The "future" series are immediately consecutive
        to the "past" series. The slicing of past and future covariates matches that of past and future targets,
        respectively. The slicing itself relies on time indexes to align the series if they have unequal lengths.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `input_chunk_length + output_chunk_length`.
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
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future.
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
        shift = input_chunk_length + output_chunk_shift
        # This dataset is in charge of serving past covariates
        self.ds_past = GenericShiftedDataset(
            target_series=target_series,
            covariates=past_covariates,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            shift=shift,
            shift_covariates=False,
            max_samples_per_ts=max_samples_per_ts,
            covariate_type=CovariateType.PAST,
            use_static_covariates=use_static_covariates,
            sample_weight=sample_weight,
        )

        # This dataset is in charge of serving historical and future future covariates
        self.ds_dual = DualCovariatesSequentialDataset(
            target_series=target_series,
            covariates=future_covariates,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            output_chunk_shift=output_chunk_shift,
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


class SplitCovariatesSequentialDataset(SplitCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        output_chunk_shift: int = 0,
        max_samples_per_ts: Optional[int] = None,
        use_static_covariates: bool = True,
        sample_weight: Optional[Union[TimeSeries, Sequence[TimeSeries], str]] = None,
    ):
        """
        A time series dataset containing tuples of (past_target, past_covariates, future_covariates, static_covariates,
        sample weights, future_target).
        The "past" series have length `input_chunk_length` and the "future" series have
        length `output_chunk_length`. The "future" series are immediately consecutive to the "past" series.
        The slicing of past and future covariates matches that of past and future targets, respectively. The slicing
        itself relies on time indexes to align the series if they have unequal lengths.

        Each series must be long enough to contain at least one (input, output) pair; i.e., each
        series must have length at least `input_chunk_length + output_chunk_length`.
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
        input_chunk_length
            The length of the emitted past series.
        output_chunk_length
            The length of the emitted future series.
        output_chunk_shift
            Optionally, the number of steps to shift the start of the output chunk into the future.
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
        shift = input_chunk_length + output_chunk_shift
        # This dataset is in charge of serving past covariates
        self.ds_past = GenericShiftedDataset(
            target_series=target_series,
            covariates=past_covariates,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
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
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
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
